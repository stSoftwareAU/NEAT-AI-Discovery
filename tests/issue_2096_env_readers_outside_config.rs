//! Issue #2270 (Part B, chunk 13d-1 of parent #2124) — hostile-value probes for
//! the environment readers in `src/analysis/constants/` that sit outside
//! `src/config/`.
//!
//! Each reader parses a `NEAT_AI_DISCOVERY_*` variable and then applies its own
//! policy: `unwrap_or` for unparsable input, `filter` for non-finite or negative
//! floats, and `clamp` for out-of-range values. The probes pin what HEAD returns
//! for each class of hostile input, so a widened clamp, a changed fallback or a
//! filter that stops rejecting `NaN` fails here.
//!
//! Readers covered, all re-exported from `neat_ai_discovery::analysis::constants`:
//!
//! - `candidate_scoring.rs::max_add_neuron_candidates_per_target` (L58)
//! - `candidate_scoring.rs::max_coordinated_per_target_output` (L109)
//! - `candidate_scoring.rs::min_distinct_targets_per_batch` (L155)
//! - `candidate_scoring.rs::risky_squash_prior` (L505)
//! - `candidate_scoring.rs::min_expected_creature_score_gain` (L845)
//! - `candidate_scoring.rs::neuron_ranking_reliability_bands` (L991)
//! - `candidate_scoring.rs::min_bypass_weight_for_collapse` (L1050)
//! - `candidate_scoring.rs::coordinated_noise_floor_multiplier` (L1203)
//! - `candidate_scoring.rs::remove_low_impact_noise_floor` (L1508)
//! - `detection_thresholds.rs::cooldown_conservative_divisor` (L235)
//! - `detection_thresholds.rs::cooldown_extended_drought_divisor` (L249)
//!
//! Integer readers parse with `parse_env` (`src/config/helpers.rs:36`), which
//! trims surrounding whitespace before parsing. `0` parses successfully, so it
//! is clamped to the floor rather than replaced by the default. The
//! `neuron_ranking_reliability_bands` reader parses as `u32`, so a value of
//! `4294967296` fails to parse and falls back to the default rather than
//! wrapping.
//!
//! Rule for this sub-issue: tests only. A failing probe is not fixed here. Its
//! case is marked `#[ignore = "finding: <reason>"]`, keeps its failing
//! assertion, and is handed to 13d-2 as a candidate finding.

use neat_ai_discovery::analysis::constants::{
    COOLDOWN_CONSERVATIVE_DIVISOR, COOLDOWN_DIVISOR_CEILING, COOLDOWN_DIVISOR_FLOOR,
    COOLDOWN_EXTENDED_DROUGHT_DIVISOR, COORDINATED_NOISE_FLOOR_MULTIPLIER_CEILING,
    COORDINATED_NOISE_FLOOR_MULTIPLIER_FLOOR, MAX_ADD_NEURON_CANDIDATES_PER_TARGET,
    MAX_ADD_NEURON_CANDIDATES_PER_TARGET_CEILING, MAX_COORDINATED_PER_TARGET_OUTPUT,
    MAX_COORDINATED_PER_TARGET_OUTPUT_CEILING, MAX_NEURON_RANKING_RELIABILITY_BANDS,
    MAX_RISKY_SQUASH_PRIOR, MIN_ADD_NEURON_CANDIDATES_PER_TARGET, MIN_BYPASS_WEIGHT_FOR_COLLAPSE,
    MIN_BYPASS_WEIGHT_FOR_COLLAPSE_CEILING, MIN_BYPASS_WEIGHT_FOR_COLLAPSE_FLOOR,
    MIN_COORDINATED_PER_TARGET_OUTPUT, MIN_DISTINCT_TARGETS_PER_BATCH,
    MIN_DISTINCT_TARGETS_PER_BATCH_CEILING, MIN_DISTINCT_TARGETS_PER_BATCH_FLOOR,
    MIN_EXPECTED_CREATURE_SCORE_GAIN, MIN_EXPECTED_CREATURE_SCORE_GAIN_CEILING,
    MIN_EXPECTED_CREATURE_SCORE_GAIN_FLOOR, MIN_NEURON_RANKING_RELIABILITY_BANDS,
    MIN_RISKY_SQUASH_PRIOR, NEURON_RANKING_RELIABILITY_BANDS, REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS,
    RISKY_SQUASH_PRIOR_DEFAULT, cooldown_conservative_divisor, cooldown_extended_drought_divisor,
    coordinated_noise_floor_multiplier, max_add_neuron_candidates_per_target,
    max_coordinated_per_target_output, min_bypass_weight_for_collapse,
    min_distinct_targets_per_batch, min_expected_creature_score_gain,
    neuron_ranking_reliability_bands, remove_low_impact_noise_floor, risky_squash_prior,
};
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

/// The non-finite spellings the float readers must reject. Rust's `f32`
/// parser accepts all three.
const NON_FINITE: [&str; 3] = ["NaN", "inf", "-inf"];

/// Run `accessor` under each `(input, expected)` probe. Every result must lie
/// within `[floor, ceiling]` and must equal the expected value.
fn assert_probes<T>(var: &str, floor: T, ceiling: T, accessor: fn() -> T, cases: &[(&str, T)])
where
    T: PartialOrd + Copy + std::fmt::Debug,
{
    for (input, expected) in cases {
        let actual = with_env(var, input, accessor);
        assert!(
            (floor..=ceiling).contains(&actual),
            "{var}={input:?} returned {actual:?}, outside [{floor:?}, {ceiling:?}]"
        );
        assert_eq!(actual, *expected, "{var}={input:?}");
    }
}

/// Each non-finite spelling must come back finite and equal to the default.
fn assert_non_finite_falls_back(var: &str, default: f32, accessor: fn() -> f32) {
    for input in NON_FINITE {
        let actual = with_env(var, input, accessor);
        assert!(
            actual.is_finite(),
            "{var}={input:?} returned non-finite {actual}"
        );
        assert_eq!(
            actual, default,
            "{var}={input:?} must fall back to the default"
        );
    }
}

// ---------------------------------------------------------------------------
// Integer readers (usize and u64): `0` clamps to the floor, garbage and
// negatives fall back to the default, over-ceiling values clamp to the ceiling,
// and padded valid values parse after trimming.
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn max_add_neuron_candidates_per_target_hostile_values() {
    const { assert!(5 >= MIN_ADD_NEURON_CANDIDATES_PER_TARGET) };
    const { assert!(5 <= MAX_ADD_NEURON_CANDIDATES_PER_TARGET_CEILING) };
    assert_probes(
        var!("MAX_ADD_NEURON_PER_TARGET"),
        MIN_ADD_NEURON_CANDIDATES_PER_TARGET,
        MAX_ADD_NEURON_CANDIDATES_PER_TARGET_CEILING,
        max_add_neuron_candidates_per_target,
        &[
            ("0", MIN_ADD_NEURON_CANDIDATES_PER_TARGET),
            ("-1", MAX_ADD_NEURON_CANDIDATES_PER_TARGET),
            ("abc", MAX_ADD_NEURON_CANDIDATES_PER_TARGET),
            ("33", MAX_ADD_NEURON_CANDIDATES_PER_TARGET_CEILING),
            ("99999999", MAX_ADD_NEURON_CANDIDATES_PER_TARGET_CEILING),
            (" 5 ", 5),
        ],
    );
}

#[test]
#[serial]
fn max_coordinated_per_target_output_hostile_values() {
    const { assert!(5 >= MIN_COORDINATED_PER_TARGET_OUTPUT) };
    const { assert!(5 <= MAX_COORDINATED_PER_TARGET_OUTPUT_CEILING) };
    assert_probes(
        var!("MAX_COORDINATED_PER_TARGET"),
        MIN_COORDINATED_PER_TARGET_OUTPUT,
        MAX_COORDINATED_PER_TARGET_OUTPUT_CEILING,
        max_coordinated_per_target_output,
        &[
            ("0", MIN_COORDINATED_PER_TARGET_OUTPUT),
            ("-1", MAX_COORDINATED_PER_TARGET_OUTPUT),
            ("abc", MAX_COORDINATED_PER_TARGET_OUTPUT),
            ("33", MAX_COORDINATED_PER_TARGET_OUTPUT_CEILING),
            ("99999999", MAX_COORDINATED_PER_TARGET_OUTPUT_CEILING),
            (" 5 ", 5),
        ],
    );
}

#[test]
#[serial]
fn min_distinct_targets_per_batch_hostile_values() {
    const { assert!(5 >= MIN_DISTINCT_TARGETS_PER_BATCH_FLOOR) };
    const { assert!(5 <= MIN_DISTINCT_TARGETS_PER_BATCH_CEILING) };
    assert_probes(
        var!("MIN_DISTINCT_TARGETS_PER_BATCH"),
        MIN_DISTINCT_TARGETS_PER_BATCH_FLOOR,
        MIN_DISTINCT_TARGETS_PER_BATCH_CEILING,
        min_distinct_targets_per_batch,
        &[
            ("0", MIN_DISTINCT_TARGETS_PER_BATCH_FLOOR),
            ("-1", MIN_DISTINCT_TARGETS_PER_BATCH),
            ("abc", MIN_DISTINCT_TARGETS_PER_BATCH),
            ("33", MIN_DISTINCT_TARGETS_PER_BATCH_CEILING),
            ("99999999", MIN_DISTINCT_TARGETS_PER_BATCH_CEILING),
            (" 5 ", 5),
        ],
    );
}

#[test]
#[serial]
fn neuron_ranking_reliability_bands_hostile_values() {
    // `u32` parse: `4294967296` overflows and must fall back to the default,
    // not wrap to `0` (which would clamp to the floor).
    const { assert!(5 >= MIN_NEURON_RANKING_RELIABILITY_BANDS) };
    const { assert!(5 <= MAX_NEURON_RANKING_RELIABILITY_BANDS) };
    assert_probes(
        var!("NEURON_RANKING_BANDS"),
        MIN_NEURON_RANKING_RELIABILITY_BANDS,
        MAX_NEURON_RANKING_RELIABILITY_BANDS,
        neuron_ranking_reliability_bands,
        &[
            ("0", MIN_NEURON_RANKING_RELIABILITY_BANDS),
            ("-1", NEURON_RANKING_RELIABILITY_BANDS),
            ("abc", NEURON_RANKING_RELIABILITY_BANDS),
            ("4294967296", NEURON_RANKING_RELIABILITY_BANDS),
            ("1001", MAX_NEURON_RANKING_RELIABILITY_BANDS),
            ("4294967295", MAX_NEURON_RANKING_RELIABILITY_BANDS),
            ("99999999", MAX_NEURON_RANKING_RELIABILITY_BANDS),
            (" 5 ", 5),
        ],
    );
}

#[test]
#[serial]
fn cooldown_conservative_divisor_hostile_values() {
    const { assert!(5 >= COOLDOWN_DIVISOR_FLOOR) };
    const { assert!(5 <= COOLDOWN_DIVISOR_CEILING) };
    assert_probes(
        var!("COOLDOWN_CONSERVATIVE_DIVISOR"),
        COOLDOWN_DIVISOR_FLOOR,
        COOLDOWN_DIVISOR_CEILING,
        cooldown_conservative_divisor,
        &[
            ("0", COOLDOWN_DIVISOR_FLOOR),
            ("-1", COOLDOWN_CONSERVATIVE_DIVISOR),
            ("abc", COOLDOWN_CONSERVATIVE_DIVISOR),
            ("65", COOLDOWN_DIVISOR_CEILING),
            ("99999999", COOLDOWN_DIVISOR_CEILING),
            (" 5 ", 5),
        ],
    );
}

#[test]
#[serial]
fn cooldown_extended_drought_divisor_hostile_values() {
    const { assert!(5 >= COOLDOWN_DIVISOR_FLOOR) };
    const { assert!(5 <= COOLDOWN_DIVISOR_CEILING) };
    assert_probes(
        var!("COOLDOWN_EXTENDED_DROUGHT_DIVISOR"),
        COOLDOWN_DIVISOR_FLOOR,
        COOLDOWN_DIVISOR_CEILING,
        cooldown_extended_drought_divisor,
        &[
            ("0", COOLDOWN_DIVISOR_FLOOR),
            ("-1", COOLDOWN_EXTENDED_DROUGHT_DIVISOR),
            ("abc", COOLDOWN_EXTENDED_DROUGHT_DIVISOR),
            ("65", COOLDOWN_DIVISOR_CEILING),
            ("99999999", COOLDOWN_DIVISOR_CEILING),
            (" 5 ", 5),
        ],
    );
}

// ---------------------------------------------------------------------------
// Float readers: NaN, inf and -inf are filtered out and fall back to the
// default. Unparsable input does the same. Finite out-of-range values clamp.
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn risky_squash_prior_hostile_values() {
    let var = var!("RISKY_SQUASH_PRIOR");
    assert_non_finite_falls_back(var, RISKY_SQUASH_PRIOR_DEFAULT, risky_squash_prior);
    assert_probes(
        var,
        MIN_RISKY_SQUASH_PRIOR,
        MAX_RISKY_SQUASH_PRIOR,
        risky_squash_prior,
        &[
            ("abc", RISKY_SQUASH_PRIOR_DEFAULT),
            ("0", MIN_RISKY_SQUASH_PRIOR),
            ("-1", MIN_RISKY_SQUASH_PRIOR),
            ("99999999", MAX_RISKY_SQUASH_PRIOR),
            (" 0.5 ", 0.5),
        ],
    );
}

#[test]
#[serial]
fn min_expected_creature_score_gain_hostile_values() {
    let var = var!("MIN_EXPECTED_GAIN");
    assert_non_finite_falls_back(
        var,
        MIN_EXPECTED_CREATURE_SCORE_GAIN,
        min_expected_creature_score_gain,
    );
    assert_probes(
        var,
        MIN_EXPECTED_CREATURE_SCORE_GAIN_FLOOR,
        MIN_EXPECTED_CREATURE_SCORE_GAIN_CEILING,
        min_expected_creature_score_gain,
        &[
            ("abc", MIN_EXPECTED_CREATURE_SCORE_GAIN),
            ("-1", MIN_EXPECTED_CREATURE_SCORE_GAIN),
            ("0", MIN_EXPECTED_CREATURE_SCORE_GAIN_FLOOR),
            ("99999999", MIN_EXPECTED_CREATURE_SCORE_GAIN_CEILING),
            (" 0.0005 ", 0.0005),
        ],
    );
}

#[test]
#[serial]
fn min_bypass_weight_for_collapse_hostile_values() {
    let var = var!("MIN_BYPASS_WEIGHT_FOR_COLLAPSE");
    assert_non_finite_falls_back(
        var,
        MIN_BYPASS_WEIGHT_FOR_COLLAPSE,
        min_bypass_weight_for_collapse,
    );
    assert_probes(
        var,
        MIN_BYPASS_WEIGHT_FOR_COLLAPSE_FLOOR,
        MIN_BYPASS_WEIGHT_FOR_COLLAPSE_CEILING,
        min_bypass_weight_for_collapse,
        &[
            ("abc", MIN_BYPASS_WEIGHT_FOR_COLLAPSE),
            ("-1", MIN_BYPASS_WEIGHT_FOR_COLLAPSE),
            ("0", MIN_BYPASS_WEIGHT_FOR_COLLAPSE_FLOOR),
            ("99999999", MIN_BYPASS_WEIGHT_FOR_COLLAPSE_CEILING),
            (" 0.05 ", 0.05),
        ],
    );
}

#[test]
#[serial]
fn coordinated_noise_floor_multiplier_hostile_values() {
    // The default is the literal `1.0` in the body, not a named constant.
    // Zero and negatives fail the `> 0.0` filter and take the default too.
    let var = var!("COORDINATED_NOISE_FLOOR_MULTIPLIER");
    assert_non_finite_falls_back(var, 1.0, coordinated_noise_floor_multiplier);
    assert_probes(
        var,
        COORDINATED_NOISE_FLOOR_MULTIPLIER_FLOOR,
        COORDINATED_NOISE_FLOOR_MULTIPLIER_CEILING,
        coordinated_noise_floor_multiplier,
        &[
            ("abc", 1.0),
            ("-1", 1.0),
            ("0", 1.0),
            ("0.0000001", COORDINATED_NOISE_FLOOR_MULTIPLIER_FLOOR),
            ("99999999", COORDINATED_NOISE_FLOOR_MULTIPLIER_CEILING),
            (" 2.0 ", 2.0),
        ],
    );
}

// ---------------------------------------------------------------------------
// Remove-low-impact noise floor: two overrides. The absolute override wins
// verbatim when valid. Otherwise the units override scales `cost_of_growth`.
// Each probe unsets the other variable so only one is under test.
// ---------------------------------------------------------------------------

/// The `cost_of_growth` argument used by every probe.
const COST: f32 = 1.0;

#[test]
#[serial]
fn remove_low_impact_noise_floor_absolute_hostile_values() {
    const ABSOLUTE: &str = var!("REMOVE_LOW_IMPACT_NOISE_FLOOR");
    const UNITS: &str = var!("REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS");

    // Unset the units override so the absolute probes fall through to it.
    let _units = EnvGuard::unset(UNITS);
    let default = {
        let _absolute = EnvGuard::unset(ABSOLUTE);
        remove_low_impact_noise_floor(COST)
    };
    assert_eq!(
        default,
        REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS * COST,
        "unset overrides must return the units default times cost"
    );

    for input in ["NaN", "inf", "-inf", "abc", "-1"] {
        let actual = with_env(ABSOLUTE, input, || remove_low_impact_noise_floor(COST));
        assert!(
            actual.is_finite(),
            "{ABSOLUTE}={input:?} returned non-finite {actual}"
        );
        assert_eq!(
            actual, default,
            "{ABSOLUTE}={input:?} must be ignored and fall through to the units default"
        );
    }

    // Control: a valid absolute value is returned verbatim, unscaled by cost.
    let control = with_env(ABSOLUTE, " 0.5 ", || remove_low_impact_noise_floor(COST));
    assert_eq!(control, 0.5, "{ABSOLUTE}=\" 0.5 \" must win verbatim");
}

#[test]
#[serial]
fn remove_low_impact_noise_floor_units_hostile_values() {
    const ABSOLUTE: &str = var!("REMOVE_LOW_IMPACT_NOISE_FLOOR");
    const UNITS: &str = var!("REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS");

    // Unset the absolute override so the units probes are the ones that count.
    let _absolute = EnvGuard::unset(ABSOLUTE);
    let default = {
        let _units = EnvGuard::unset(UNITS);
        remove_low_impact_noise_floor(COST)
    };
    assert_eq!(
        default,
        REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS * COST,
        "unset overrides must return the units default times cost"
    );

    for input in ["NaN", "inf", "-inf", "abc", "-1"] {
        let actual = with_env(UNITS, input, || remove_low_impact_noise_floor(COST));
        assert!(
            actual.is_finite(),
            "{UNITS}={input:?} returned non-finite {actual}"
        );
        assert_eq!(
            actual, default,
            "{UNITS}={input:?} must be ignored and fall back to the units default"
        );
    }

    // Control: a valid units value scales the cost.
    let control = with_env(UNITS, " 2.0 ", || remove_low_impact_noise_floor(COST));
    assert_eq!(control, 2.0, "{UNITS}=\" 2.0 \" must scale cost by 2");
}
