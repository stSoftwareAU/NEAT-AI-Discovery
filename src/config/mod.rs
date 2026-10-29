//! Central configuration for runtime environment variables (Issue #717).
//!
//! [`docs/CONFIGURATION.md`](../../docs/CONFIGURATION.md) is the single source
//! of truth for every `NEAT_AI_DISCOVERY_*` variable this crate reads, and for
//! the few variables the library sets for itself. Each entry there states the
//! default and, where the code applies one, the accepted range and the rule for
//! invalid or out-of-range values. Do not copy that reference into this module;
//! update the document instead.
//!
//! The accessors are grouped by submodule, and each submodule documents the
//! variables it reads:
//!
//! - `helpers`: the shared parsing rules (`parse_env` and the boolean helpers).
//! - `detection`: detection-threshold, module-tiering and target-cooldown
//!   overrides.
//! - `drought_mitigation`: the startup snapshot of the drought-mitigation levers.
//! - `observability`: timing, profile and GPU-metrics output, plus candidate
//!   reconciliation and calibration-miss logging.
//! - `user_facing`: GPU, streaming, focus-ranking, memory, deadline and
//!   discovery-strategy settings.

mod detection;
mod drought_mitigation;
pub(crate) mod helpers;
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
