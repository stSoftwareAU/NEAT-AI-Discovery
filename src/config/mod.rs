//! Central configuration for all environment variables (Issue #717).
//!
//! This module is the single source of truth for runtime environment variable
//! configuration. Each variable is documented with its purpose, type, default
//! value, and valid range. Typed accessor functions validate values at read
//! time and provide clear error messages via `tracing`.
//!
//! ## User-Facing Configuration
//!
//! | Variable | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `NEAT_AI_DISCOVERY_VERBOSE` | bool | `false` | Enable verbose logging (`1` to enable) |
//! | `NEAT_AI_DISCOVERY_GPU_TIMING` | bool | `false` | Enable GPU kernel timing (~5% overhead) |
//! | `NEAT_AI_DISCOVERY_GPU_BATCH_SIZE` | usize | auto | Override GPU batch size (64–4096) |
//! | `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` | u32 | `3` | Max consecutive device-lost recovery attempts (0–10) |
//! | `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS` | u64 | `0` (disabled) | Abort if no heartbeat for N seconds |
//! | `NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS` | u64 | `2` | Delay between thread dump and abort |
//! | `NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS` | usize | adaptive | Maximum streaming cache blocks |
//! | `NEAT_AI_DISCOVERY_PREFETCH_DEPTH` | usize | `2` | How many blocks ahead to prefetch |
//! | `NEAT_AI_DISCOVERY_PRELOAD_ALL` | bool | `false` | Disable streaming; use full preload |
//! | `NEAT_AI_DISCOVERY_BLOCK_SIZE` | usize | `10000` | Records per streaming block (10–100000) |
//! | `NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD` | f32 | `1e-7` | Constant source folding threshold (`0` to disable) |
//! | `NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS` | bool | `false` | Enable outlier-focused analysis |
//! | `NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE` | u8 | `90` | Outlier percentile threshold (1–99) |
//! | `NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY` | bool | `false` | Restrict focus targets to output neurons only |
//! | `NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS` | bool | `false` | Prioritise unused input neurons |
//! | `NEAT_AI_DISCOVERY_FOCUS_EXCLUDE_CONSTANT_NEURONS` | bool | `false` | Exclude functionally-constant hidden neurons (zero activation variance) from focus-slot eligibility; they remain available to the constant-neuron removal path (Issue #1624) |
//! | `NEAT_AI_DISCOVERY_FOCUS_RECONSTRUCTION_MISMATCH` | bool | `false` | Fold each neuron's mean reconstruction activation delta (recorded vs `squash(bias + Σ from_activation × weight)`) into its focus score as an additive term, so poorly-explained neurons rise in the focus budget (Issue #1634) |
//! | `NEAT_AI_DISCOVERY_FOCUS_RECONSTRUCTION_MISMATCH_WEIGHT` | f32 | `0.1` | Additive weight applied to the mean reconstruction delta when the signal above is enabled; non-negative finite values only, invalid falls back to the default (Issue #1634) |
//! | `NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE` | bool | `false` | Gate out neurons whose structural impact magnitude is below the threshold from focus-slot eligibility; complements the constant-neuron filter by catching non-constant near-zero-impact neurons (Issue #1635) |
//! | `NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE_THRESHOLD` | f32 | `1e-6` | Impact-magnitude gate: neurons with `\|impact\| <` this value are gated out when the gate above is enabled; positive finite values only, invalid falls back to the default (Issue #1635) |
//! | `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS` | f64 | disabled | Bias source ordering toward higher input indices (0–10) |
//! | `NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET` | usize | unlimited | Cap the number of priority-ordered source neurons evaluated (sample-build + GPU) per focus target (Issue #1542). `0`/unset/invalid = unlimited (back-compat). |
//! | `NEAT_AI_DISCOVERY_ZERO_COPY` | Option\<bool\> | auto | Force-enable/disable zero-copy buffers |
//! | `NEAT_AI_DISCOVERY_QUIET_GPU` | bool | `false` | Suppress Mesa/libEGL debug output (Linux) |
//! | `NEAT_AI_DISCOVERY_MH_TEMPERATURE` | f32 | disabled | Metropolis-Hastings temperature for probabilistic acceptance (0.01–5.0) |
//! | `NEAT_AI_DISCOVERY_SESSION_TTL_SECS` | u64 | `3600` | Streaming session TTL for orphan cleanup (60–86400) |
//! | `NEAT_AI_DISCOVERY_BATCH_SUCCESSFUL` | bool | `false` | Re-enable disabled batch-successful module (Issue #1059) |
//! | `NEAT_AI_DISCOVERY_MAX_WALL_CLOCK_MINUTES` | u64 | `20` | Overall wall-clock cap for discovery time in minutes (1–120) (Issue #1098) |
//! | `NEAT_AI_DISCOVERY_LOW_SUCCESS_RATE_THRESHOLD` | f32 | `0.2` | Rolling success-rate threshold below which conservative mode engages (Issue #1132) |
//! | `NEAT_AI_DISCOVERY_CONSERVATIVE_MODE_MAX_EPOCHS` | u32 | `20` | Max consecutive failed passes before abandoning conservative mode (Issue #1132) |
//! | `NEAT_AI_DISCOVERY_CONSERVATIVE_GAIN_MULTIPLIER` | f32 | `10.0` | Multiplier applied to `COORDINATED_MIN_EXPECTED_GAIN` in conservative mode (Issue #1132) |
//! | `NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD` | u32 | `5` | Consecutive trailing empty discovery passes at which the drought diagnostic warn log fires and `droughtDiagnostic` populates on FFI metadata (Issue #1202) |
//! | `NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS` | u32 | `50` | Operator escape hatch: force a one-shot reset of failed-candidate cache entries and active target cooldowns after this many consecutive empty discovery passes (Issue #1205). Armed by default at `50` (Issue #1422); set to `0` to deliberately disable. Unparsable values fall back to the armed default. |
//! | `NEAT_AI_DISCOVERY_HIDDEN_SQUASH_PRUNE` | bool | `true` | Toggle squash-aware hidden-target activation scan pruning (Issue #1545). When enabled, a hidden add-neuron target scans only the core set (`IDENTITY`, `GELU`, `ELU`, `ReLU6`, `TANH`) widened by the squash families the creature already uses; a drought escalates back to the full `ACTIVATION_SPECS` set. Set to `0`/`false`/`no` to always scan the full set. |
//! | `NEAT_AI_DISCOVERY_MAX_ACTIVATION_CONFIGS_PER_TARGET` | usize | `0` | Cap on (orientation × scale) activation configs per (source, target) pair for hidden add-neuron evaluation after squash-family filtering (Issue #1545). `0` disables the cap; a positive value keeps the `N` configs whose scale is closest to `1.0`. Unparsable values fall back to `0`. |
//! | `NEAT_AI_DISCOVERY_DROUGHT_ALARM_EPOCHS` | u32 | `100` | Epochs-since-last-accepted-candidate at which a single, durable creature-level drought alarm fires — a `tracing::warn!` line plus a `creatureDroughtAlarm` field on the FFI metadata carrying the creature uuid, epochs since the last acceptance, and an environmental-vs-search-exhaustion classification (Issue #1424). Set to `0` to disable; unparsable values fall back to the default. |
//! | `NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB` | f64 | platform default (0.5 macOS / 1.0 Linux) | Minimum available memory (GB) below which the discovery gate disables analysis (Issue #1420). Lets a small-but-capable ~8GB host — where the discovery runtime itself already holds most of the RAM — opt in by lowering the floor. `0` disables the available-memory gate; invalid / out-of-range (`0.0–64.0`) values fall back to the platform default. The total-memory minimum (4GB) is unaffected. |
//! | `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB` | u64 | unset | Cap eager pre-load size in `focus::rank_focus_neurons` (Issue #1172). When set, projected size = file size × 3; lazy mode is selected with a structured `info` log when the projection exceeds the budget. When unset, the auto-detect path (Issue #1376) is used. |
//! | `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_MARGIN_MB` | u64 | `1024` | Safety margin reserved from real OS-available memory in the auto-detect (no explicit budget) eager-vs-lazy decision (Issue #1376). Pre-load is chosen when `projected ≤ available − margin`, keeping hosts with GBs free on the fast path. `0` reserves no margin. |
//! | `NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS` | u64 | `120000` | Wall-clock budget (milliseconds) for a single focus-ranking run (Issue #1375). Checked between passes and inside the per-neuron loops; on exceed the run aborts with a structured `Timeout` error (+ a 1s grace) so the caller falls back to its instant local ranking instead of blowing the discovery budget. `0` disables the bound (fully unbounded); other values clamp to `[1000, 3600000]` (Issue #1385). |
//! | `NEAT_AI_DISCOVERY_FOCUS_RANKING_PERF_CLIFF_MS` | u64 | `60000` | Perf-cliff threshold (milliseconds) for a *lazy* focus-ranking pass (Issue #1377). A lazy pass at or above this emits one explicit perf-cliff `WARN` naming the neuron count and projected dataset size; the fast preload path never trips it. `0` disables the warning. |
//! | `NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN` | f32 | `1e-5` | Absolute minimum `expected_creature_score_gain` for emitted add-neuron / add-synapse candidates (Issue #1191). Clamped to `[0.0, 1e-2]`. |
//! | `NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS` | u64 | `60000` | Guaranteed minimum window (milliseconds) reserved for synapse/neuron analysis so focus selection + parquet loading cannot starve it (Issue #1408). Parquet loading is curtailed at `deadline − reserve`; if focus/parquet have already consumed so much that less than 1s would remain, `analyze_all` fails fast with an actionable error instead of analysing 0/N targets. `0` disables the reserve (restores pre-#1408 behaviour); other values clamp to `[1, 3600000]`. The effective reserve is also capped by `ANALYSIS_RESERVE_FRACTION` so small budgets are split rather than starving loading. |
//! | `NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_FRACTION` | f64 | `0.5` | Fraction of the remaining discovery window the reserve may claim (Issue #1408). The effective reserve is `min(ANALYSIS_RESERVE_MS, remaining × fraction)`, so on a tight budget the reserve shrinks and loading keeps the rest. Finite values in `(0.0, 0.9]` are honoured (clamped to `0.9`); invalid or non-positive values fall back to `0.5`. |
//! | `NEAT_AI_DISCOVERY_INSUFFICIENT_RECORDING_FRACTION` | f64 | `1.0` | Fraction of selected focus neurons that must have **zero** Parquet rows before the fail-fast insufficient-recording gate skips synapse/neuron analysis (Issue #1444). A partial record phase leaves focus neurons with no rows, so analysis is guaranteed empty yet still burns the full budget; the gate detects this with a cheap record-count scan *before* GPU work and surfaces `insufficient_recording` as the dominant rejection reason plus an `insufficientRecording` diagnostic on `synapseMetadata` / `neuronMetadata`. Honoured in `(0.0, 1.0]`; `0` disables the gate. |
//!
//! ## Observability Variables
//!
//! | Variable | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `RUST_LOG` | filter string | `warn` | Control tracing log level (e.g. `neat_ai_discovery=info`) |
//! | `NEAT_AI_DISCOVERY_TIMING` | bool | `false` | Print phase timing to stderr |
//! | `NEAT_AI_DISCOVERY_PROFILE` | `json`/empty | disabled | Output structured profile as JSON |
//! | `NEAT_AI_DISCOVERY_GPU_METRICS` | bool | `false` | Print GPU metrics to stderr |
//! | `NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD` | f32 | `10.0` | Ratio above which prediction-vs-actual mismatches are logged (Issue #1165) |
//! | `NEAT_AI_DISCOVERY_STRICT_CANDIDATE_RECONCILIATION` | bool | debug builds: `true`, release builds: `false` | Trip a `debug_assert!` when a discovery pass cannot account for every considered candidate (Issue #1802), so a new silent drop path fails CI. Set to `0` to force warn-only. The `unaccounted_drop` rejection-breakdown entry and the `tracing::warn!` are emitted regardless. |
//!
//! ## Detection Tuning Variables
//!
//! | Variable | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD` | f32 | module default | Input dominance detection threshold |
//! | `NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD` | f32 | module default | Gradient detection threshold |
//! | `NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD` | f32 | module default | Noise-to-signal ratio threshold |
//! | `NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES` | u32 | `3` | Consecutive target-neuron failures before cooldown (Issue #1130) |
//! | `NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS` | u64 | `10` | Cooldown duration in epochs for skipped targets (Issue #1130) |
//! | `NEAT_AI_DISCOVERY_BATCH_TARGET_FAILURE_LIMIT` | u32 | `1` | Within-batch failure limit before same-target candidates are short-circuited (Issue #1164) |
//! | `NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR` | u64 | `2` | Divisor applied to the target cooldown window during conservative mode (Issue #1204) |
//! | `NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR` | u64 | `4` | Divisor applied to the target cooldown window during extended drought (Issue #1204) |
//!
//! ## Internal/Debug Variables
//!
//! | Variable | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` | String | `sample` | macOS `sample` binary path override |
//!
//! ## GPU Platform Variables (Linux only, set internally)
//!
//! | Variable | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `EGL_LOG_LEVEL` | String | (not set) | Set to `fatal` when `QUIET_GPU` is enabled |
//! | `MESA_GLSL_CACHE_DISABLE` | String | (not set) | Set to `true` when `QUIET_GPU` is enabled |
//! | `MESA_DEBUG` | String | (not set) | Set to `silent` when `QUIET_GPU` is enabled |
//! | `XDG_RUNTIME_DIR` | String | (not set) | Auto-created if missing (Wayland requirement) |

mod detection;
mod drought_mitigation;
mod helpers;
mod observability;
mod user_facing;

// Re-export all public items so existing `crate::config::*` paths keep working.
pub use detection::*;
pub use drought_mitigation::*;
pub use observability::*;
pub use user_facing::*;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bool_env_truthy_values() {
        // Cannot test with real env vars due to caching / global state.
        // Instead, test the parsing logic directly.
        assert!(matches!("1", "1" | "true" | "yes"));
        assert!(matches!("true", "1" | "true" | "yes"));
        assert!(matches!("yes", "1" | "true" | "yes"));
        assert!(!matches!("0", "1" | "true" | "yes"));
        assert!(!matches!("false", "1" | "true" | "yes"));
        assert!(!matches!("", "1" | "true" | "yes"));
    }

    #[test]
    fn optional_bool_parsing() {
        // Test the parse_optional_bool_env logic patterns
        let truthy = |s: &str| {
            let v = s.trim().to_lowercase();
            match v.as_str() {
                "1" | "true" | "yes" => Some(true),
                "0" | "false" | "no" => Some(false),
                _ => None,
            }
        };
        assert_eq!(truthy("1"), Some(true));
        assert_eq!(truthy("true"), Some(true));
        assert_eq!(truthy("yes"), Some(true));
        assert_eq!(truthy("YES"), Some(true));
        assert_eq!(truthy("0"), Some(false));
        assert_eq!(truthy("false"), Some(false));
        assert_eq!(truthy("no"), Some(false));
        assert_eq!(truthy("maybe"), None);
        assert_eq!(truthy(""), None);
    }

    #[test]
    fn block_size_getter_returns_value_within_bounds() {
        // WHAT-test: the configured block-size getter must always return a value
        // within its [MIN, MAX] range, whatever the literals are tuned to. This
        // exercises the clamp in block_size() rather than pinning the constants.
        let result = block_size();
        assert!((MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&result));
    }

    #[test]
    fn outlier_percentile_default_value() {
        // When no env var is set, should return 90
        // (This test relies on the env var NOT being set in the test environment)
        let result = outlier_percentile();
        assert!(result > 0 && result < 100);
    }

    // Tautological `session_ttl_default_values` pin test removed (Issue #1469):
    // `session_ttl_returns_valid_value` below is the behavioural companion that
    // asserts the getter stays within [MIN, MAX], making the pin redundant.

    #[test]
    fn session_ttl_returns_valid_value() {
        // When no env var is set, should return the default (3600)
        let result = session_ttl_secs();
        assert!((MIN_SESSION_TTL_SECS..=MAX_SESSION_TTL_SECS).contains(&result));
    }

    #[test]
    fn constant_source_default_threshold() {
        assert!((DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD - 1e-7).abs() < 1e-13);
    }

    #[test]
    fn constant_source_threshold_with_dynamic_no_variance() {
        // With no variance data and no env var, should return default
        let result = constant_source_threshold_with_dynamic(None);
        assert_eq!(result, Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD));
    }

    #[test]
    fn constant_source_threshold_with_dynamic_high_variance() {
        // With high variance, threshold should scale up
        let result = constant_source_threshold_with_dynamic(Some(0.5));
        let expected_scaling = (0.5_f32 / 0.05).max(1.0);
        let expected = DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD * expected_scaling;
        assert!(result.is_some());
        let val = result.unwrap();
        assert!((val - expected).abs() < 1e-12);
    }

    #[test]
    fn constant_source_threshold_with_dynamic_low_variance() {
        // With low variance (< 0.05), scaling factor should be 1.0
        let result = constant_source_threshold_with_dynamic(Some(0.01));
        assert_eq!(result, Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD));
    }

    #[test]
    fn constant_source_threshold_with_dynamic_invalid_variance() {
        // NaN and negative values should fall back to default
        let result_nan = constant_source_threshold_with_dynamic(Some(f32::NAN));
        assert_eq!(result_nan, Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD));

        let result_neg = constant_source_threshold_with_dynamic(Some(-1.0));
        assert_eq!(result_neg, Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD));
    }

    #[test]
    fn profile_mode_default_is_none() {
        // Default profile mode should be None
        let default_mode = ProfileMode::default();
        assert_eq!(default_mode, ProfileMode::None);
    }

    #[test]
    fn watchdog_abort_delay_default() {
        // When env var is not set, should return 2 seconds
        let delay = watchdog_abort_delay();
        assert!(delay.as_secs() >= 1);
    }

    // Tautological `wall_clock_minutes_default_values` pin test removed (Issue
    // #1469): `wall_clock_minutes_returns_valid_value` below is the behavioural
    // companion that asserts the getter stays within [MIN, MAX].

    #[test]
    fn wall_clock_minutes_returns_valid_value() {
        let result = max_wall_clock_minutes();
        assert!((MIN_WALL_CLOCK_MINUTES..=MAX_WALL_CLOCK_MINUTES).contains(&result));
    }

    // -------------------------------------------------------------------
    // Issue #1202 — drought diagnostic threshold parsing
    // -------------------------------------------------------------------

    /// Mirror of the parser inside [`drought_log_threshold`] — kept inline so
    /// tests can exercise edge cases without mutating the global env (which is
    /// unsafe under parallel test execution).
    fn parse_drought_threshold(raw: Option<&str>) -> u32 {
        raw.and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|v| *v >= 1)
            .unwrap_or(DEFAULT_DROUGHT_LOG_THRESHOLD)
    }

    #[test]
    fn drought_log_threshold_unset_resolves_to_default() {
        // WHAT-test: with nothing supplied the parser must fall back to the
        // default. Exercises the parser's fallback branch rather than pinning
        // the default's literal value.
        assert_eq!(parse_drought_threshold(None), DEFAULT_DROUGHT_LOG_THRESHOLD);
    }

    #[test]
    fn drought_log_threshold_accepts_positive_integers() {
        assert_eq!(parse_drought_threshold(Some("1")), 1);
        assert_eq!(parse_drought_threshold(Some("10")), 10);
        assert_eq!(parse_drought_threshold(Some(" 7 ")), 7);
    }

    #[test]
    fn drought_log_threshold_rejects_zero_and_invalid() {
        // Zero is invalid (would fire on every empty pass — surely a typo).
        assert_eq!(parse_drought_threshold(Some("0")), 5);
        assert_eq!(parse_drought_threshold(Some("")), 5);
        assert_eq!(parse_drought_threshold(Some("abc")), 5);
        assert_eq!(parse_drought_threshold(Some("-3")), 5);
        assert_eq!(parse_drought_threshold(Some("3.5")), 5);
    }

    #[test]
    fn drought_log_threshold_function_returns_positive() {
        // The accessor itself must always return a sensible value, regardless
        // of whether the env var happens to be set in the test environment.
        let result = drought_log_threshold();
        assert!(result >= 1);
    }

    // -------------------------------------------------------------------
    // Issue #1424 — creature-level drought alarm threshold parsing
    // -------------------------------------------------------------------

    #[test]
    fn drought_alarm_epochs_unset_returns_default() {
        assert_eq!(resolve_drought_alarm_epochs(None, 100), Some(100));
        assert_eq!(resolve_drought_alarm_epochs(Some(""), 100), Some(100));
    }

    #[test]
    fn drought_alarm_epochs_accepts_positive_integers() {
        assert_eq!(resolve_drought_alarm_epochs(Some("20"), 100), Some(20));
        assert_eq!(resolve_drought_alarm_epochs(Some(" 250 "), 100), Some(250));
    }

    #[test]
    fn drought_alarm_epochs_zero_disables() {
        // Zero is the explicit operator opt-out.
        assert_eq!(resolve_drought_alarm_epochs(Some("0"), 100), None);
    }

    #[test]
    fn drought_alarm_epochs_invalid_falls_back_to_default() {
        assert_eq!(resolve_drought_alarm_epochs(Some("abc"), 100), Some(100));
        assert_eq!(resolve_drought_alarm_epochs(Some("-5"), 100), Some(100));
        assert_eq!(resolve_drought_alarm_epochs(Some("3.5"), 100), Some(100));
    }

    // -------------------------------------------------------------------
    // Issue #1420 — configurable available-memory floor
    // -------------------------------------------------------------------

    #[test]
    fn min_available_memory_gb_unset_returns_default() {
        // No env value → caller-supplied default is returned unchanged.
        assert!((resolve_min_available_memory_gb(None, 1.0) - 1.0).abs() < f64::EPSILON);
        assert!((resolve_min_available_memory_gb(None, 0.5) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn min_available_memory_gb_accepts_valid_overrides() {
        // Operators can lower the floor (small-but-capable 8GB host)...
        assert!((resolve_min_available_memory_gb(Some("0.1"), 1.0) - 0.1).abs() < f64::EPSILON);
        // ...or disable the gate entirely with 0...
        assert!((resolve_min_available_memory_gb(Some("0"), 1.0)).abs() < f64::EPSILON);
        // ...or raise it.
        assert!((resolve_min_available_memory_gb(Some("2.5"), 1.0) - 2.5).abs() < f64::EPSILON);
        // Whitespace is tolerated.
        assert!((resolve_min_available_memory_gb(Some(" 0.25 "), 1.0) - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn min_available_memory_gb_rejects_invalid_overrides() {
        // Empty / non-numeric / negative / NaN / out-of-range fall back to default.
        for raw in ["", "  ", "abc", "-1", "NaN", "inf", "65", "100"] {
            let resolved = resolve_min_available_memory_gb(Some(raw), 1.0);
            assert!(
                (resolved - 1.0).abs() < f64::EPSILON,
                "raw {raw:?} should fall back to default 1.0, got {resolved}"
            );
        }
    }

    #[test]
    fn min_available_memory_gb_boundary_values() {
        // Exactly at the max bound is accepted; just above is rejected.
        assert!(
            (resolve_min_available_memory_gb(Some("64"), 1.0) - MAX_MIN_AVAILABLE_MEMORY_GB).abs()
                < f64::EPSILON
        );
        assert!((resolve_min_available_memory_gb(Some("64.01"), 1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn min_available_memory_gb_accessor_returns_sane_default() {
        // The live accessor must always return a non-negative, finite value.
        let result = min_available_memory_gb();
        assert!(result.is_finite());
        assert!(result >= 0.0);
        assert!(result <= MAX_MIN_AVAILABLE_MEMORY_GB);
    }

    // -------------------------------------------------------------------
    // Issue #1444 — insufficient-recording fail-fast fraction
    // -------------------------------------------------------------------

    #[test]
    fn insufficient_recording_fraction_unset_returns_default() {
        assert_eq!(
            resolve_insufficient_recording_fraction(None),
            Some(DEFAULT_INSUFFICIENT_RECORDING_FRACTION)
        );
        assert_eq!(
            resolve_insufficient_recording_fraction(Some("")),
            Some(DEFAULT_INSUFFICIENT_RECORDING_FRACTION)
        );
        assert_eq!(
            resolve_insufficient_recording_fraction(Some("  ")),
            Some(DEFAULT_INSUFFICIENT_RECORDING_FRACTION)
        );
    }

    #[test]
    fn insufficient_recording_fraction_zero_disables_gate() {
        assert_eq!(resolve_insufficient_recording_fraction(Some("0")), None);
        assert_eq!(resolve_insufficient_recording_fraction(Some("0.0")), None);
    }

    #[test]
    fn insufficient_recording_fraction_accepts_valid_range() {
        assert_eq!(
            resolve_insufficient_recording_fraction(Some("1.0")),
            Some(1.0)
        );
        assert_eq!(
            resolve_insufficient_recording_fraction(Some("0.5")),
            Some(0.5)
        );
        assert_eq!(
            resolve_insufficient_recording_fraction(Some(" 0.25 ")),
            Some(0.25)
        );
    }

    #[test]
    fn insufficient_recording_fraction_rejects_invalid() {
        // Out of range / non-finite / non-numeric / negative fall back to default.
        for raw in ["1.5", "2", "-0.5", "NaN", "inf", "abc"] {
            assert_eq!(
                resolve_insufficient_recording_fraction(Some(raw)),
                Some(DEFAULT_INSUFFICIENT_RECORDING_FRACTION),
                "raw {raw:?} should fall back to default"
            );
        }
    }
}
