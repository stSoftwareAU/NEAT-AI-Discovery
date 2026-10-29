//! Issue #2258 (chunk 13b-1 of the #2096 security-scan overflow audit) —
//! hostile-value probes for every un-cached numeric accessor in the first half
//! of `src/config/user_facing.rs` (top of file through
//! `focus_ranking_memory_margin_mb`).
//!
//! Each test feeds the hostile set — `u64::MAX`, `-1`, `NaN`, `inf`, `abc` and,
//! where it has its own meaning, `0` — to one accessor and pins the value it
//! returns today. A widened clamp, a `NaN` or negative value that stops being
//! rejected, or a changed fallback fails an assertion here.
//!
//! Deliberately **not** tested:
//!
//! - The `OnceLock`-cached accessors (`verbose`, `gpu_timing`,
//!   `gpu_batch_size_override`, `gpu_retry_limit`, `mh_temperature`) freeze on
//!   first read, so they cannot be re-read in-process. Their parsing is covered
//!   by the helper unit tests in `src/config/helpers.rs`.
//! - The boolean accessors are not numeric.
//! - Whitespace trimming is already pinned by
//!   `tests/issue_2006_numeric_env_override_trim.rs`.
//!
//! The `usize` expectations assume a 64-bit target, where `u64::MAX` parses
//! as `usize::MAX`.

use std::time::Duration;

use neat_ai_discovery::analysis::gpu::heartbeat::{
    DEFAULT_GPU_STALL_WINDOW_SECS, GPU_STALL_WINDOW_ENV, MAX_GPU_STALL_WINDOW_SECS,
};
use neat_ai_discovery::config::{
    DEFAULT_BLOCK_SIZE, DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD,
    DEFAULT_FOCUS_IMPACT_GATE_THRESHOLD, DEFAULT_FOCUS_RANKING_MEMORY_MARGIN_MB,
    DEFAULT_FOCUS_RECONSTRUCTION_MISMATCH_WEIGHT, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE, block_size,
    constant_source_effect_threshold, constant_source_threshold_with_dynamic,
    focus_impact_gate_threshold, focus_ranking_memory_budget_mb, focus_ranking_memory_margin_mb,
    focus_reconstruction_mismatch_weight, gpu_stall_window, max_activation_configs_per_target,
    max_cached_blocks, max_parquet_decode_mb, max_sources_per_target, prefetch_depth,
    source_input_index_bias, watchdog_abort_delay, watchdog_stall_timeout,
};
use serial_test::serial;

/// Set `name` to `value`, run `body`, then restore the previous value.
///
/// Every caller is `#[serial]`, so the process-wide mutation is safe.
fn with_env<T>(name: &str, value: &str, body: impl FnOnce() -> T) -> T {
    let previous = std::env::var(name).ok();
    // SAFETY: Serialised via #[serial].
    unsafe { std::env::set_var(name, value) };
    let result = body();
    match previous {
        // SAFETY: Serialised via #[serial].
        Some(v) => unsafe { std::env::set_var(name, v) },
        // SAFETY: Serialised via #[serial].
        None => unsafe { std::env::remove_var(name) },
    }
    result
}

/// The largest `u64`, built from the constant rather than a literal.
fn huge() -> String {
    u64::MAX.to_string()
}

/// `u64::MAX` as parsed into an `f32`: it rounds up to exactly 2^64.
fn huge_as_f32() -> f32 {
    2.0_f32.powi(64)
}

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

/// Build a case table from string literals plus the huge value.
fn cases<T: Clone>(huge_expected: Option<T>, rest: &[(&str, T)]) -> Vec<(String, T)> {
    let mut table: Vec<(String, T)> = rest
        .iter()
        .map(|(v, e)| ((*v).to_string(), e.clone()))
        .collect();
    if let Some(e) = huge_expected {
        table.push((huge(), e));
    }
    table
}

// ---------------------------------------------------------------------------
// GPU and watchdog timing
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn gpu_stall_window_clamps_disables_and_falls_back() {
    let default = Duration::from_secs(DEFAULT_GPU_STALL_WINDOW_SECS);
    let max = Duration::from_secs(MAX_GPU_STALL_WINDOW_SECS);
    assert_eq!(max, Duration::from_secs(600), "the ceiling is 600 s");
    assert_eq!(default, Duration::from_secs(30), "the default is 30 s");
    assert_table(
        GPU_STALL_WINDOW_ENV,
        &cases(
            Some(max),
            &[
                ("1000", max),
                ("0", Duration::ZERO),
                ("-1", default),
                ("NaN", default),
                ("inf", default),
                ("abc", default),
            ],
        ),
        gpu_stall_window,
    );
}

#[test]
#[serial]
fn watchdog_stall_timeout_rejects_negative_and_nan() {
    // The huge value is left out on purpose: the stall-timeout truncation
    // finding (#2259) owns it, and its fix will change the result. `0` and
    // unparsable input are already pinned by issue_2006.
    assert_table(
        "NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS",
        &cases(None, &[("-1", None), ("NaN", None), ("inf", None)]),
        watchdog_stall_timeout,
    );
}

#[test]
#[serial]
fn watchdog_abort_delay_falls_back_to_two_seconds() {
    // The huge value is left out on purpose: the abort-delay finding (#2259)
    // owns it, and its fix will change the result.
    let default = Duration::from_secs(2);
    assert_table(
        "NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS",
        &cases(
            None,
            &[
                ("-1", default),
                ("NaN", default),
                ("inf", default),
                ("abc", default),
            ],
        ),
        watchdog_abort_delay,
    );
}

// ---------------------------------------------------------------------------
// Streaming cache and block sizing
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn max_cached_blocks_accepts_huge_and_rejects_the_rest() {
    assert_table(
        "NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS",
        &cases(
            Some(Some(usize::MAX)),
            &[("-1", None), ("NaN", None), ("inf", None), ("abc", None)],
        ),
        max_cached_blocks,
    );
}

#[test]
#[serial]
fn prefetch_depth_accepts_huge_and_falls_back_to_two() {
    // No production reach: `get_streaming_config_from_env` has no caller, so
    // this pins the accessor alone.
    assert_table(
        "NEAT_AI_DISCOVERY_PREFETCH_DEPTH",
        &cases(
            Some(usize::MAX),
            &[("-1", 2), ("NaN", 2), ("inf", 2), ("abc", 2)],
        ),
        prefetch_depth,
    );
}

#[test]
#[serial]
fn block_size_clamps_and_falls_back() {
    assert_table(
        "NEAT_AI_DISCOVERY_BLOCK_SIZE",
        &cases(
            Some(MAX_BLOCK_SIZE),
            &[
                ("0", MIN_BLOCK_SIZE),
                ("-1", DEFAULT_BLOCK_SIZE),
                ("NaN", DEFAULT_BLOCK_SIZE),
                ("inf", DEFAULT_BLOCK_SIZE),
                ("abc", DEFAULT_BLOCK_SIZE),
            ],
        ),
        block_size,
    );
}

// ---------------------------------------------------------------------------
// Focus weighting and gating
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn focus_reconstruction_mismatch_weight_rejects_non_finite_and_negative() {
    let default = DEFAULT_FOCUS_RECONSTRUCTION_MISMATCH_WEIGHT;
    assert_table(
        "NEAT_AI_DISCOVERY_FOCUS_RECONSTRUCTION_MISMATCH_WEIGHT",
        &cases(
            None,
            &[
                ("0", 0.0),
                ("-1", default),
                ("NaN", default),
                ("inf", default),
                ("abc", default),
            ],
        ),
        focus_reconstruction_mismatch_weight,
    );

    // The huge value is accepted: it parses to a finite `f32` (~1.8e19).
    let value = with_env(
        "NEAT_AI_DISCOVERY_FOCUS_RECONSTRUCTION_MISMATCH_WEIGHT",
        &huge(),
        focus_reconstruction_mismatch_weight,
    );
    assert!(
        value.is_finite(),
        "huge weight must stay finite, got {value}"
    );
    assert_eq!(value, huge_as_f32(), "huge weight is accepted verbatim");
}

#[test]
#[serial]
fn focus_impact_gate_threshold_rejects_non_positive_and_non_finite() {
    let default = DEFAULT_FOCUS_IMPACT_GATE_THRESHOLD;
    assert_table(
        "NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE_THRESHOLD",
        &cases(
            None,
            &[
                ("0", default),
                ("-1", default),
                ("NaN", default),
                ("inf", default),
                ("abc", default),
            ],
        ),
        focus_impact_gate_threshold,
    );

    // The huge value is accepted: it parses to a finite `f32` (~1.8e19).
    let value = with_env(
        "NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE_THRESHOLD",
        &huge(),
        focus_impact_gate_threshold,
    );
    assert!(value.is_finite(), "huge gate must stay finite, got {value}");
    assert_eq!(value, huge_as_f32(), "huge gate is accepted verbatim");
}

// ---------------------------------------------------------------------------
// Source enumeration caps and biases
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn max_activation_configs_per_target_accepts_huge_and_falls_back_to_uncapped() {
    assert_table(
        "NEAT_AI_DISCOVERY_MAX_ACTIVATION_CONFIGS_PER_TARGET",
        &cases(
            Some(usize::MAX),
            &[("0", 0), ("-1", 0), ("NaN", 0), ("inf", 0), ("abc", 0)],
        ),
        max_activation_configs_per_target,
    );
}

#[test]
#[serial]
fn source_input_index_bias_accepts_only_the_finite_range() {
    assert_table(
        "NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS",
        &cases(
            Some(None),
            &[
                ("10", Some(10.0)),
                ("10.5", None),
                ("0", None),
                ("-1", None),
                ("NaN", None),
                ("inf", None),
                ("abc", None),
            ],
        ),
        source_input_index_bias,
    );
}

#[test]
#[serial]
fn max_sources_per_target_accepts_huge_and_rejects_the_rest() {
    assert_table(
        "NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET",
        &cases(
            Some(Some(usize::MAX)),
            &[
                ("0", None),
                ("-1", None),
                ("NaN", None),
                ("inf", None),
                ("abc", None),
            ],
        ),
        max_sources_per_target,
    );
}

// ---------------------------------------------------------------------------
// Constant-source folding threshold
// ---------------------------------------------------------------------------

const CONSTANT_SOURCE_ENV: &str = "NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD";

/// The shared table for both constant-source accessors (huge handled apart).
fn constant_source_cases() -> Vec<(String, Option<f32>)> {
    let default = Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD);
    cases(
        None,
        &[
            ("0", None),
            ("-1", default),
            ("NaN", default),
            ("inf", default),
            ("abc", default),
        ],
    )
}

/// The huge value is accepted as a finite `f32` (~1.8e19).
fn assert_constant_source_huge(accessor: impl Fn() -> Option<f32>) {
    let value = with_env(CONSTANT_SOURCE_ENV, &huge(), accessor)
        .expect("a huge threshold is accepted, not treated as disabled");
    assert!(
        value.is_finite(),
        "huge threshold must stay finite, got {value}"
    );
    assert_eq!(value, huge_as_f32(), "huge threshold is accepted verbatim");
}

#[test]
#[serial]
fn constant_source_effect_threshold_rejects_non_finite_and_negative() {
    assert_table(
        CONSTANT_SOURCE_ENV,
        &constant_source_cases(),
        constant_source_effect_threshold,
    );
    assert_constant_source_huge(constant_source_effect_threshold);
}

#[test]
#[serial]
fn constant_source_threshold_with_dynamic_rejects_non_finite_and_negative() {
    let accessor = || constant_source_threshold_with_dynamic(None);
    assert_table(CONSTANT_SOURCE_ENV, &constant_source_cases(), accessor);
    assert_constant_source_huge(accessor);
}

// ---------------------------------------------------------------------------
// Focus-ranking memory budget, parquet decode ceiling and margin
// ---------------------------------------------------------------------------

/// `0`, negative, non-numeric → `None`; the huge value is accepted verbatim.
fn optional_megabytes_cases() -> Vec<(String, Option<u64>)> {
    cases(
        Some(Some(u64::MAX)),
        &[
            ("0", None),
            ("-1", None),
            ("NaN", None),
            ("inf", None),
            ("abc", None),
        ],
    )
}

#[test]
#[serial]
fn focus_ranking_memory_budget_mb_accepts_huge_and_rejects_the_rest() {
    assert_table(
        "NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB",
        &optional_megabytes_cases(),
        focus_ranking_memory_budget_mb,
    );
}

#[test]
#[serial]
fn max_parquet_decode_mb_accepts_huge_and_rejects_the_rest() {
    assert_table(
        "NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB",
        &optional_megabytes_cases(),
        max_parquet_decode_mb,
    );
}

#[test]
#[serial]
fn focus_ranking_memory_margin_mb_honours_zero_and_falls_back() {
    let default = DEFAULT_FOCUS_RANKING_MEMORY_MARGIN_MB;
    assert_table(
        "NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_MARGIN_MB",
        &cases(
            Some(u64::MAX),
            &[
                ("0", 0),
                ("-1", default),
                ("NaN", default),
                ("inf", default),
                ("abc", default),
            ],
        ),
        focus_ranking_memory_margin_mb,
    );
}
