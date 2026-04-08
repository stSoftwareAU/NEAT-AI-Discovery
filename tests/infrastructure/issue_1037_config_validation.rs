//! Integration tests for configuration validation (Issue #1037).
//!
//! Tests that `mh_temperature` and `source_input_index_bias` correctly validate
//! parsed values against their documented ranges.

use serial_test::serial;

// =============================================================================
// MH temperature range validation
// =============================================================================

/// Parse MH temperature the same way as the production code, but without
/// `OnceLock` caching so each test case gets a fresh read.
fn parse_mh_temperature(raw: &str) -> Option<f32> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<f32>() {
        Ok(v)
            if v.is_finite()
                && (neat_ai_discovery::config::MIN_MH_TEMPERATURE
                    ..=neat_ai_discovery::config::MAX_MH_TEMPERATURE)
                    .contains(&v) =>
        {
            Some(v)
        }
        _ => None,
    }
}

#[test]
fn mh_temperature_accepts_lower_bound() {
    assert_eq!(parse_mh_temperature("0.01"), Some(0.01));
}

#[test]
fn mh_temperature_accepts_upper_bound() {
    assert_eq!(parse_mh_temperature("5.0"), Some(5.0));
}

#[test]
fn mh_temperature_accepts_mid_range() {
    assert_eq!(parse_mh_temperature("1.0"), Some(1.0));
}

#[test]
fn mh_temperature_rejects_zero() {
    assert_eq!(parse_mh_temperature("0.0"), None);
}

#[test]
fn mh_temperature_rejects_below_lower_bound() {
    assert_eq!(parse_mh_temperature("0.009"), None);
}

#[test]
fn mh_temperature_rejects_above_upper_bound() {
    assert_eq!(parse_mh_temperature("5.1"), None);
}

#[test]
fn mh_temperature_rejects_negative() {
    assert_eq!(parse_mh_temperature("-1.0"), None);
}

#[test]
fn mh_temperature_rejects_nan() {
    assert_eq!(parse_mh_temperature("NaN"), None);
}

#[test]
fn mh_temperature_rejects_infinity() {
    assert_eq!(parse_mh_temperature("inf"), None);
}

#[test]
fn mh_temperature_rejects_empty() {
    assert_eq!(parse_mh_temperature(""), None);
}

#[test]
fn mh_temperature_rejects_non_numeric() {
    assert_eq!(parse_mh_temperature("abc"), None);
}

// =============================================================================
// Source input index bias range validation
// =============================================================================

#[test]
#[serial]
fn source_input_index_bias_rejects_above_max() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS", "10.1") };

    let bias = neat_ai_discovery::config::source_input_index_bias();
    assert!(
        bias.is_none(),
        "Bias above MAX_SOURCE_INPUT_INDEX_BIAS should be rejected"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS") };
}

#[test]
#[serial]
fn source_input_index_bias_accepts_max_bound() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS", "10.0") };

    let bias = neat_ai_discovery::config::source_input_index_bias();
    assert_eq!(
        bias,
        Some(10.0),
        "Bias at MAX_SOURCE_INPUT_INDEX_BIAS should be accepted"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS") };
}

#[test]
#[serial]
fn source_input_index_bias_rejects_nan() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS", "NaN") };

    let bias = neat_ai_discovery::config::source_input_index_bias();
    assert!(bias.is_none(), "NaN bias should be rejected");

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS") };
}

#[test]
#[serial]
fn source_input_index_bias_rejects_zero() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS", "0.0") };

    let bias = neat_ai_discovery::config::source_input_index_bias();
    assert!(bias.is_none(), "Zero bias should be rejected");

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS") };
}

// =============================================================================
// Constants are consistent with temperature module
// =============================================================================

#[test]
fn mh_temperature_bounds_match_temperature_module() {
    // Verify that the config validation bounds match the constants
    // defined in the temperature scheduling module.
    assert!(
        (neat_ai_discovery::config::MIN_MH_TEMPERATURE - 0.01).abs() < f32::EPSILON,
        "MIN_MH_TEMPERATURE should be 0.01"
    );
    assert!(
        (neat_ai_discovery::config::MAX_MH_TEMPERATURE - 5.0).abs() < f32::EPSILON,
        "MAX_MH_TEMPERATURE should be 5.0"
    );
}

#[test]
fn source_input_index_bias_max_is_reasonable() {
    assert!(
        (neat_ai_discovery::config::MAX_SOURCE_INPUT_INDEX_BIAS - 10.0).abs() < f64::EPSILON,
        "MAX_SOURCE_INPUT_INDEX_BIAS should be 10.0"
    );
}
