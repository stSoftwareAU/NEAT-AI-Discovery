//! Issue #2263 (chunk 13c-1 of #2123, the #2096 security-scan overflow audit) —
//! hostile-value probes for the un-cached numeric accessors in the second half
//! of `src/config/user_facing.rs`: from `focus_ranking_budget_ms` and
//! `effective_focus_ranking_budget_ms` through to `drought_reset_after_epochs`,
//! plus `sample_program_override` and the pure resolvers behind the accessors.
//!
//! The hostile set is `u64::MAX`, `-1`, `NaN`, `inf`, `abc`, `0`, the empty
//! string and one whitespace-padded valid value. Every accessor's table covers
//! all of them and pins what HEAD returns today. A widened clamp, a changed
//! fallback, or a guard that starts turning off for garbage input fails here.
//!
//! The six `0` opt-outs (focus budget, analysis reserve, insufficient
//! recording fraction, drought alarm, drought reset, minimum free memory) get
//! two extra pins: garbage must leave the guard active, and each spelling of
//! zero is recorded. The spelling table pins HEAD behaviour, not an
//! endorsement of it.
//!
//! Deliberately **not** tested:
//!
//! - Accessors cached in a `OnceLock` freeze on first read, so they cannot be
//!   re-read in-process. Their resolvers are tested instead.
//! - Cases already covered by the `src/config/mod.rs` unit tests,
//!   `tests/issue_1422_drought_mitigation_config.rs` (defaults when unset),
//!   `tests/issue_3172_focus_ranking_budget_scaling.rs` (mode scaling when
//!   unset) and `tests/issue_1934_sample_fallback.rs`.

use neat_ai_discovery::analysis::creature_drought_alarm::DEFAULT_DROUGHT_ALARM_EPOCHS;
use neat_ai_discovery::analysis::discovery_mode::{
    DEFAULT_CONSERVATIVE_GAIN_MULTIPLIER, DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
    DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
};
use neat_ai_discovery::analysis::novelty_escalation::{
    DEFAULT_GAIN_FLOOR_RELAXATION, DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
};
use neat_ai_discovery::analysis::remove_neuron_drought::{
    DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR, MIN_REMOVE_NEURON_DROUGHT_FACTOR,
    NEUTRAL_DEPRIORITISATION_FACTOR,
};
use neat_ai_discovery::analysis::utils::memory::DEFAULT_MIN_AVAILABLE_MEMORY_GB;
use neat_ai_discovery::config::*;
use serial_test::serial;

macro_rules! var {
    ($s:literal) => {
        concat!("NEAT_AI_DISCOVERY_", $s)
    };
}

/// RAII guard that restores an env var on drop. Every caller is `#[serial]`,
/// so the process-wide mutation is safe.
struct EnvGuard {
    key: String,
    prior: Option<String>,
}

impl EnvGuard {
    fn set(key: &str, value: &str) -> Self {
        let prior = std::env::var(key).ok();
        // SAFETY: tests are marked #[serial] so no concurrent set/unset races.
        unsafe { std::env::set_var(key, value) };
        Self {
            key: key.to_string(),
            prior,
        }
    }

    fn unset(key: &str) -> Self {
        let prior = std::env::var(key).ok();
        // SAFETY: tests are marked #[serial] so no concurrent set/unset races.
        unsafe { std::env::remove_var(key) };
        Self {
            key: key.to_string(),
            prior,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: tests are marked #[serial] so no concurrent set/unset races.
        unsafe {
            match &self.prior {
                Some(v) => std::env::set_var(&self.key, v),
                None => std::env::remove_var(&self.key),
            }
        }
    }
}

/// Set `name` to `value` for the duration of `body`.
fn with_env<T>(name: &str, value: &str, body: impl FnOnce() -> T) -> T {
    let _guard = EnvGuard::set(name, value);
    body()
}

/// The largest `u64`, built from the constant rather than a literal.
fn huge() -> String {
    u64::MAX.to_string()
}

/// `u64::MAX` as parsed into an `f32`: it rounds up to exactly 2^64.
fn huge_as_f32() -> f32 {
    2.0_f32.powi(64)
}

/// Garbage that never parses to a usable number.
const GARBAGE: [&str; 5] = ["-1", "NaN", "inf", "abc", ""];

/// Run `accessor` under each `(value, expected)` pair and assert the result.
fn assert_table<T: PartialEq + std::fmt::Debug>(
    name: &str,
    cases: &[(String, T)],
    accessor: impl Fn() -> T,
) {
    for (value, expected) in cases {
        let actual = with_env(name, value, &accessor);
        assert_eq!(&actual, expected, "{name}={value:?}");
    }
}

/// Build a table: every `garbage` entry maps to `garbage_expected`, `rest`
/// adds literal cases, and the huge value maps to `huge_expected`.
fn table<T: Clone>(
    garbage: &[&str],
    garbage_expected: T,
    rest: &[(&str, T)],
    huge_expected: T,
) -> Vec<(String, T)> {
    let mut out: Vec<(String, T)> = garbage
        .iter()
        .map(|g| ((*g).to_string(), garbage_expected.clone()))
        .collect();
    out.extend(rest.iter().map(|(v, e)| ((*v).to_string(), e.clone())));
    out.push((huge(), huge_expected));
    out
}

/// Define a `#[serial]` test that runs one accessor over a [`table`].
macro_rules! env_table_test {
    ($test:ident, $var:literal, $accessor:expr,
     $garbage:expr => $g:expr, [$($case:expr),* $(,)?], huge => $h:expr) => {
        #[test]
        #[serial]
        fn $test() {
            let cases = table($garbage, $g, &[$($case),*], $h);
            assert_table(var!($var), &cases, $accessor);
        }
    };
}

// ---------------------------------------------------------------------------
// Focus-ranking budget
// ---------------------------------------------------------------------------

env_table_test!(focus_ranking_budget_ms_clamps_disables_and_falls_back,
    "FOCUS_RANKING_BUDGET_MS", focus_ranking_budget_ms,
    &GARBAGE => Some(DEFAULT_FOCUS_RANKING_BUDGET_MS),
    [("0", None), ("5", Some(FOCUS_RANKING_BUDGET_MIN_MS)), (" 5000 ", Some(5000))],
    huge => Some(FOCUS_RANKING_BUDGET_MAX_MS));

// Garbage gets the lazy default: 4 x 120 000 = 480 000.
env_table_test!(effective_focus_ranking_budget_ms_scales_only_the_default,
    "FOCUS_RANKING_BUDGET_MS", || effective_focus_ranking_budget_ms(true, 0),
    &GARBAGE => Some(DEFAULT_FOCUS_RANKING_BUDGET_MS * FOCUS_RANKING_LAZY_BUDGET_MULTIPLIER),
    [("0", None), (" 5000 ", Some(5000))],
    huge => Some(FOCUS_RANKING_BUDGET_MAX_MS));

#[test]
fn scale_default_focus_ranking_budget_ms_saturates_and_clamps() {
    let scale = scale_default_focus_ranking_budget_ms;
    // `u64::MAX` projected megabytes saturate rather than overflow.
    assert_eq!(scale(true, u64::MAX), FOCUS_RANKING_BUDGET_MAX_MS);
    assert_eq!(scale(false, u64::MAX), DEFAULT_FOCUS_RANKING_BUDGET_MS);
    assert_eq!(scale(true, 0), 480_000);
    // In range: base 480 000 plus 1 000 MB at 20 ms each.
    assert_eq!(FOCUS_RANKING_BUDGET_MS_PER_PROJECTED_MB, 20);
    assert_eq!(scale(true, 1000), 500_000);
    // The lazy base is already above the minimum, so only the ceiling bites.
    assert_eq!(scale(true, 200_000), FOCUS_RANKING_BUDGET_MAX_MS);
}

// ---------------------------------------------------------------------------
// Analysis reserve, recording fraction, timing and session limits
// ---------------------------------------------------------------------------

env_table_test!(analysis_reserve_ms_clamps_and_falls_back,
    "ANALYSIS_RESERVE_MS", analysis_reserve_ms,
    &GARBAGE => DEFAULT_ANALYSIS_RESERVE_MS, [("0", 0), (" 7000 ", 7000)],
    huge => ANALYSIS_RESERVE_MAX_MS);

env_table_test!(analysis_reserve_fraction_clamps_and_falls_back,
    "ANALYSIS_RESERVE_FRACTION", analysis_reserve_fraction,
    &GARBAGE => DEFAULT_ANALYSIS_RESERVE_FRACTION,
    [("0", DEFAULT_ANALYSIS_RESERVE_FRACTION), (" 0.3 ", 0.3)],
    huge => ANALYSIS_RESERVE_FRACTION_MAX);

/// The shared table for the accessor and its pure resolver.
fn recording_fraction_cases() -> Vec<(String, Option<f64>)> {
    let default = Some(DEFAULT_INSUFFICIENT_RECORDING_FRACTION);
    table(
        &GARBAGE,
        default,
        &[("0", None), (" 0.5 ", Some(0.5)), ("2", Some(1.0))],
        default,
    )
}

#[test]
#[serial]
fn insufficient_recording_fraction_disables_on_zero_only() {
    // The accessor delegates to the resolver, so the whole table applies.
    let cases = recording_fraction_cases();
    assert_table(
        var!("INSUFFICIENT_RECORDING_FRACTION"),
        &cases,
        insufficient_recording_fraction,
    );
}

#[test]
fn resolve_insufficient_recording_fraction_matches_the_table() {
    for (value, expected) in recording_fraction_cases() {
        let actual = resolve_insufficient_recording_fraction(Some(&value));
        assert_eq!(actual, expected, "raw={value:?}");
    }
    let default = Some(DEFAULT_INSUFFICIENT_RECORDING_FRACTION);
    assert_eq!(resolve_insufficient_recording_fraction(None), default);
}

env_table_test!(focus_ranking_perf_cliff_ms_accepts_huge_unclamped,
    "FOCUS_RANKING_PERF_CLIFF_MS", focus_ranking_perf_cliff_ms,
    &GARBAGE => DEFAULT_FOCUS_RANKING_PERF_CLIFF_MS, [("0", 0), (" 7000 ", 7000)],
    huge => u64::MAX);

env_table_test!(session_ttl_secs_clamps_and_falls_back,
    "SESSION_TTL_SECS", session_ttl_secs,
    &GARBAGE => DEFAULT_SESSION_TTL_SECS, [("0", MIN_SESSION_TTL_SECS), (" 7200 ", 7200)],
    huge => MAX_SESSION_TTL_SECS);

env_table_test!(max_wall_clock_minutes_clamps_and_falls_back,
    "MAX_WALL_CLOCK_MINUTES", max_wall_clock_minutes,
    &GARBAGE => DEFAULT_MAX_WALL_CLOCK_MINUTES, [("0", MIN_WALL_CLOCK_MINUTES), (" 30 ", 30)],
    huge => MAX_WALL_CLOCK_MINUTES);

// Discovery strategy adaptation

env_table_test!(low_success_rate_threshold_ignores_out_of_range,
    "LOW_SUCCESS_RATE_THRESHOLD", low_success_rate_threshold,
    &GARBAGE => DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
    [("0", DEFAULT_LOW_SUCCESS_RATE_THRESHOLD), (" 0.4 ", 0.4)],
    huge => DEFAULT_LOW_SUCCESS_RATE_THRESHOLD);

env_table_test!(conservative_mode_max_epochs_ignores_zero_and_garbage,
    "CONSERVATIVE_MODE_MAX_EPOCHS", conservative_mode_max_epochs,
    &GARBAGE => DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
    [("0", DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS), (" 7 ", 7)],
    huge => DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS);

env_table_test!(conservative_gain_multiplier_floors_at_one,
    "CONSERVATIVE_GAIN_MULTIPLIER", conservative_gain_multiplier,
    &["NaN", "inf", "abc", ""] => DEFAULT_CONSERVATIVE_GAIN_MULTIPLIER,
    [("0", 1.0), ("-1", 1.0), (" 4.5 ", 4.5)],
    huge => huge_as_f32());

// Drought mitigation

env_table_test!(drought_log_threshold_ignores_zero_and_garbage,
    "DROUGHT_LOG_THRESHOLD", drought_log_threshold,
    &GARBAGE => DEFAULT_DROUGHT_LOG_THRESHOLD,
    [("0", DEFAULT_DROUGHT_LOG_THRESHOLD), (" 9 ", 9)],
    huge => DEFAULT_DROUGHT_LOG_THRESHOLD);

env_table_test!(drought_alarm_epochs_disables_on_zero_only,
    "DROUGHT_ALARM_EPOCHS", drought_alarm_epochs,
    &GARBAGE => Some(DEFAULT_DROUGHT_ALARM_EPOCHS), [("0", None), (" 9 ", Some(9))],
    huge => Some(DEFAULT_DROUGHT_ALARM_EPOCHS));

#[test]
fn resolve_drought_alarm_epochs_threads_the_default_through() {
    let cases = table(&GARBAGE, Some(7), &[("0", None), (" 9 ", Some(9))], Some(7));
    for (value, expected) in cases {
        assert_eq!(
            resolve_drought_alarm_epochs(Some(&value), 7),
            expected,
            "raw={value:?}"
        );
    }
    assert_eq!(resolve_drought_alarm_epochs(None, 7), Some(7));
}

/// Shared table for the remove-neuron factor accessor and resolver.
fn remove_neuron_cases(default: f32) -> Vec<(String, f32)> {
    let min = MIN_REMOVE_NEURON_DROUGHT_FACTOR;
    let rest = [("0", min), ("-1", min), (" 0.5 ", 0.5)];
    table(
        &["NaN", "inf", "abc", ""],
        default,
        &rest,
        NEUTRAL_DEPRIORITISATION_FACTOR,
    )
}

#[test]
#[serial]
fn remove_neuron_drought_factor_clamps_into_the_valid_band() {
    let cases = remove_neuron_cases(DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR);
    assert_table(
        var!("REMOVE_NEURON_DROUGHT_FACTOR"),
        &cases,
        remove_neuron_drought_factor,
    );
}

#[test]
fn resolve_remove_neuron_drought_factor_threads_the_default_through() {
    for (value, expected) in remove_neuron_cases(0.25) {
        let actual = resolve_remove_neuron_drought_factor(Some(&value), 0.25);
        assert_eq!(actual, expected, "raw={value:?}");
    }
    assert_eq!(resolve_remove_neuron_drought_factor(None, 0.25), 0.25);
}

env_table_test!(novelty_suppression_ratio_ignores_zero_and_garbage,
    "NOVELTY_SUPPRESSION_RATIO", novelty_suppression_ratio,
    &GARBAGE => DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
    [("0", DEFAULT_SUPPRESSION_RATIO_THRESHOLD), (" 0.6 ", 0.6)],
    huge => DEFAULT_SUPPRESSION_RATIO_THRESHOLD);

env_table_test!(novelty_gain_relaxation_ignores_zero_and_garbage,
    "NOVELTY_GAIN_RELAXATION", novelty_gain_relaxation,
    &GARBAGE => DEFAULT_GAIN_FLOOR_RELAXATION,
    [("0", DEFAULT_GAIN_FLOOR_RELAXATION), (" 0.3 ", 0.3)],
    huge => DEFAULT_GAIN_FLOOR_RELAXATION);

/// Shared table for the minimum-memory accessor and resolver.
fn min_memory_cases(default: f64) -> Vec<(String, f64)> {
    table(
        &GARBAGE,
        default,
        &[("0", 0.0), ("65", default), (" 2.5 ", 2.5)],
        default,
    )
}

#[test]
#[serial]
fn min_available_memory_gb_disables_on_zero_only() {
    const { assert!(MAX_MIN_AVAILABLE_MEMORY_GB < 65.0) };
    let cases = min_memory_cases(DEFAULT_MIN_AVAILABLE_MEMORY_GB);
    assert_table(
        var!("MIN_AVAILABLE_MEMORY_GB"),
        &cases,
        min_available_memory_gb,
    );
}

#[test]
fn resolve_min_available_memory_gb_threads_the_default_through() {
    for (value, expected) in min_memory_cases(3.0) {
        let actual = resolve_min_available_memory_gb(Some(&value), 3.0);
        assert_eq!(actual, expected, "raw={value:?}");
    }
    assert_eq!(resolve_min_available_memory_gb(None, 3.0), 3.0);
}

env_table_test!(drought_reset_after_epochs_disables_on_zero_only,
    "DROUGHT_RESET_AFTER_EPOCHS", drought_reset_after_epochs,
    &GARBAGE => Some(DEFAULT_DROUGHT_RESET_AFTER_EPOCHS), [("0", None), (" 9 ", Some(9))],
    huge => Some(DEFAULT_DROUGHT_RESET_AFTER_EPOCHS));

// ---------------------------------------------------------------------------
// The six `0` opt-outs: garbage keeps the guard active
// ---------------------------------------------------------------------------

/// Assert that every garbage input (and the huge value) leaves a guard active.
fn assert_garbage_keeps_guard(name: &str, guard_active: impl Fn() -> bool) {
    let inputs = GARBAGE.iter().map(|g| (*g).to_string()).chain([huge()]);
    for input in inputs {
        let active = with_env(name, &input, &guard_active);
        assert!(active, "{name}={input:?} must not disable the guard");
    }
}

#[test]
#[serial]
fn garbage_never_disables_any_zero_opt_out() {
    assert_garbage_keeps_guard(var!("FOCUS_RANKING_BUDGET_MS"), || {
        focus_ranking_budget_ms().is_some()
    });
    assert_garbage_keeps_guard(var!("ANALYSIS_RESERVE_MS"), || analysis_reserve_ms() != 0);
    assert_garbage_keeps_guard(var!("INSUFFICIENT_RECORDING_FRACTION"), || {
        insufficient_recording_fraction().is_some()
    });
    assert_garbage_keeps_guard(var!("DROUGHT_ALARM_EPOCHS"), || {
        drought_alarm_epochs().is_some()
    });
    assert_garbage_keeps_guard(var!("DROUGHT_RESET_AFTER_EPOCHS"), || {
        drought_reset_after_epochs().is_some()
    });
    assert_garbage_keeps_guard(var!("MIN_AVAILABLE_MEMORY_GB"), || {
        min_available_memory_gb() > 0.0
    });
}

// ---------------------------------------------------------------------------
// Zero spellings (pins HEAD behaviour, not an endorsement of it)
// ---------------------------------------------------------------------------

/// Spellings of zero, each paired with whether an integer parse disables the
/// guard: `0.0`, `-0` and `0e5` do not parse as an unsigned integer, so they
/// fall back to the armed default.
const INTEGER_ZERO_SPELLINGS: [(&str, bool); 6] = [
    ("0", true),
    ("00", true),
    ("+0", true),
    ("0.0", false),
    ("-0", false),
    ("0e5", false),
];

/// The float opt-outs treat every spelling of zero as disabling the guard.
const FLOAT_ZERO_SPELLINGS: [(&str, bool); 6] = [
    ("0", true),
    ("00", true),
    ("+0", true),
    ("0.0", true),
    ("-0", true),
    ("0e5", true),
];

fn zero_cases(spellings: &[(&str, bool)]) -> Vec<(String, bool)> {
    spellings
        .iter()
        .map(|(v, e)| ((*v).to_string(), *e))
        .collect()
}

#[test]
#[serial]
fn integer_opt_outs_only_accept_integer_zero_spellings() {
    let cases = zero_cases(&INTEGER_ZERO_SPELLINGS);
    assert_table(var!("FOCUS_RANKING_BUDGET_MS"), &cases, || {
        focus_ranking_budget_ms().is_none()
    });
    assert_table(var!("ANALYSIS_RESERVE_MS"), &cases, || {
        analysis_reserve_ms() == 0
    });
    assert_table(var!("DROUGHT_ALARM_EPOCHS"), &cases, || {
        drought_alarm_epochs().is_none()
    });
    assert_table(var!("DROUGHT_RESET_AFTER_EPOCHS"), &cases, || {
        drought_reset_after_epochs().is_none()
    });
}

#[test]
#[serial]
fn float_opt_outs_accept_every_zero_spelling() {
    let cases = zero_cases(&FLOAT_ZERO_SPELLINGS);
    assert_table(var!("INSUFFICIENT_RECORDING_FRACTION"), &cases, || {
        insufficient_recording_fraction().is_none()
    });
    // `-0` yields -0.0, which compares equal to 0.0.
    assert_table(var!("MIN_AVAILABLE_MEMORY_GB"), &cases, || {
        min_available_memory_gb() == 0.0
    });
}

// ---------------------------------------------------------------------------
// sample_program_override
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn sample_program_override_drops_blank_values() {
    for blank in ["", "   "] {
        let actual = with_env(var!("SAMPLE_PROGRAM"), blank, sample_program_override);
        assert_eq!(actual, None, "blank {blank:?} must be treated as unset");
    }
}

#[test]
#[serial]
fn sample_program_override_keeps_a_padded_value_untrimmed() {
    let padded = " /usr/bin/sample ";
    let actual = with_env(var!("SAMPLE_PROGRAM"), padded, sample_program_override);
    assert_eq!(actual.as_deref(), Some(padded));
}

#[test]
#[serial]
fn sample_program_override_is_none_when_unset() {
    let _guard = EnvGuard::unset(var!("SAMPLE_PROGRAM"));
    assert_eq!(sample_program_override(), None);
}
