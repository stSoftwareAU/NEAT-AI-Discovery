//! Integration tests for central configuration module (Issue #717).
//!
//! Tests that typed accessor functions correctly read, validate, and return
//! environment variable values.

use serial_test::serial;

// =============================================================================
// Block size
// =============================================================================

#[test]
#[serial]
fn config_block_size_default() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_BLOCK_SIZE") };

    let size = neat_ai_discovery::config::block_size();
    assert_eq!(size, 10_000, "Default block size should be 10000");
}

#[test]
#[serial]
fn config_block_size_custom_value() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_BLOCK_SIZE", "5000") };

    let size = neat_ai_discovery::config::block_size();
    assert_eq!(size, 5000, "Block size should be 5000 when set via env var");

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_BLOCK_SIZE") };
}

#[test]
#[serial]
fn config_block_size_clamped_low() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_BLOCK_SIZE", "1") };

    let size = neat_ai_discovery::config::block_size();
    assert_eq!(
        size,
        neat_ai_discovery::config::MIN_BLOCK_SIZE,
        "Block size should be clamped to minimum"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_BLOCK_SIZE") };
}

#[test]
#[serial]
fn config_block_size_clamped_high() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_BLOCK_SIZE", "999999") };

    let size = neat_ai_discovery::config::block_size();
    assert_eq!(
        size,
        neat_ai_discovery::config::MAX_BLOCK_SIZE,
        "Block size should be clamped to maximum"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_BLOCK_SIZE") };
}

// =============================================================================
// Outlier configuration
// =============================================================================

#[test]
#[serial]
fn config_outlier_percentile_default() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE") };

    let pct = neat_ai_discovery::config::outlier_percentile();
    assert_eq!(pct, 90, "Default outlier percentile should be 90");
}

#[test]
#[serial]
fn config_outlier_percentile_custom() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE", "75") };

    let pct = neat_ai_discovery::config::outlier_percentile();
    assert_eq!(pct, 75, "Outlier percentile should be 75");

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE") };
}

#[test]
#[serial]
fn config_outlier_percentile_invalid_falls_back_to_default() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE", "0") };

    let pct = neat_ai_discovery::config::outlier_percentile();
    assert_eq!(
        pct, 90,
        "Invalid percentile (0) should fall back to default"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE", "100") };
    let pct = neat_ai_discovery::config::outlier_percentile();
    assert_eq!(
        pct, 90,
        "Invalid percentile (100) should fall back to default"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE") };
}

#[test]
#[serial]
fn config_outlier_analysis_default_disabled() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS") };

    let enabled = neat_ai_discovery::config::outlier_analysis();
    assert!(!enabled, "Outlier analysis should be disabled by default");
}

#[test]
#[serial]
fn config_outlier_analysis_enabled() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS", "1") };

    let enabled = neat_ai_discovery::config::outlier_analysis();
    assert!(enabled, "Outlier analysis should be enabled when set to 1");

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS") };
}

// =============================================================================
// Watchdog
// =============================================================================

#[test]
#[serial]
fn config_watchdog_stall_disabled_by_default() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS") };

    let timeout = neat_ai_discovery::config::watchdog_stall_timeout();
    assert!(timeout.is_none(), "Watchdog should be disabled by default");
}

#[test]
#[serial]
fn config_watchdog_stall_enabled() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS", "300") };

    let timeout = neat_ai_discovery::config::watchdog_stall_timeout();
    assert!(timeout.is_some());
    assert_eq!(
        timeout.unwrap(),
        std::time::Duration::from_secs(300),
        "Watchdog stall timeout should be 300 seconds"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS") };
}

#[test]
#[serial]
fn config_watchdog_abort_delay_default() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS") };

    let delay = neat_ai_discovery::config::watchdog_abort_delay();
    assert_eq!(
        delay,
        std::time::Duration::from_secs(2),
        "Default abort delay should be 2 seconds"
    );
}

// =============================================================================
// Prefetch depth
// =============================================================================

#[test]
#[serial]
fn config_prefetch_depth_default() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_PREFETCH_DEPTH") };

    let depth = neat_ai_discovery::config::prefetch_depth();
    assert_eq!(depth, 2, "Default prefetch depth should be 2");
}

#[test]
#[serial]
fn config_prefetch_depth_custom() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_PREFETCH_DEPTH", "5") };

    let depth = neat_ai_discovery::config::prefetch_depth();
    assert_eq!(depth, 5, "Prefetch depth should be 5");

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_PREFETCH_DEPTH") };
}

// =============================================================================
// Preload all / streaming enabled
// =============================================================================

#[test]
#[serial]
fn config_preload_all_default_disabled() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_PRELOAD_ALL") };

    assert!(
        !neat_ai_discovery::config::preload_all(),
        "Preload all should be disabled by default"
    );
    assert!(
        neat_ai_discovery::config::streaming_enabled(),
        "Streaming should be enabled by default"
    );
}

#[test]
#[serial]
fn config_preload_all_enabled() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_PRELOAD_ALL", "1") };

    assert!(
        neat_ai_discovery::config::preload_all(),
        "Preload all should be enabled when set to 1"
    );
    assert!(
        !neat_ai_discovery::config::streaming_enabled(),
        "Streaming should be disabled when preload_all is enabled"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_PRELOAD_ALL") };
}

// =============================================================================
// Constant source effect threshold
// =============================================================================

#[test]
#[serial]
fn config_constant_source_threshold_default() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD") };

    let threshold = neat_ai_discovery::config::constant_source_effect_threshold();
    assert!(threshold.is_some());
    assert!(
        (threshold.unwrap() - 1e-7).abs() < 1e-13,
        "Default should be 1e-7"
    );
}

#[test]
#[serial]
fn config_constant_source_threshold_disabled() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD", "0") };

    let threshold = neat_ai_discovery::config::constant_source_effect_threshold();
    assert!(
        threshold.is_none(),
        "Setting to 0 should disable constant source folding"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD") };
}

#[test]
#[serial]
fn config_constant_source_threshold_custom() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD", "1e-3") };

    let threshold = neat_ai_discovery::config::constant_source_effect_threshold();
    assert!(threshold.is_some());
    assert!(
        (threshold.unwrap() - 1e-3).abs() < 1e-9,
        "Custom threshold should be 1e-3"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD") };
}

// =============================================================================
// Source input index bias
// =============================================================================

#[test]
#[serial]
fn config_source_input_index_bias_default_disabled() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS") };

    let bias = neat_ai_discovery::config::source_input_index_bias();
    assert!(
        bias.is_none(),
        "Source input index bias should be disabled by default"
    );
}

#[test]
#[serial]
fn config_source_input_index_bias_enabled() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS", "2.5") };

    let bias = neat_ai_discovery::config::source_input_index_bias();
    assert_eq!(bias, Some(2.5), "Bias should be 2.5");

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS") };
}

#[test]
#[serial]
fn config_source_input_index_bias_rejects_negative() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS", "-1.0") };

    let bias = neat_ai_discovery::config::source_input_index_bias();
    assert!(bias.is_none(), "Negative bias should be rejected");

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS") };
}

// =============================================================================
// Zero copy
// =============================================================================

#[test]
#[serial]
fn config_zero_copy_override_auto() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY") };

    let override_val = neat_ai_discovery::config::zero_copy_override();
    assert!(
        override_val.is_none(),
        "Zero-copy should be auto when unset"
    );
}

#[test]
#[serial]
fn config_zero_copy_override_force_enable() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_ZERO_COPY", "1") };

    let override_val = neat_ai_discovery::config::zero_copy_override();
    assert_eq!(
        override_val,
        Some(true),
        "Zero-copy should be force-enabled"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY") };
}

#[test]
#[serial]
fn config_zero_copy_override_force_disable() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_ZERO_COPY", "0") };

    let override_val = neat_ai_discovery::config::zero_copy_override();
    assert_eq!(
        override_val,
        Some(false),
        "Zero-copy should be force-disabled"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY") };
}

// =============================================================================
// Detection thresholds
// =============================================================================

#[test]
#[serial]
fn config_noise_signal_threshold_default() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD") };

    let threshold = neat_ai_discovery::config::noise_signal_threshold(0.5);
    assert!(
        (threshold - 0.5).abs() < 1e-6,
        "Should return default when env var is unset"
    );
}

#[test]
#[serial]
fn config_noise_signal_threshold_custom() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD", "0.8") };

    let threshold = neat_ai_discovery::config::noise_signal_threshold(0.5);
    assert!(
        (threshold - 0.8).abs() < 1e-6,
        "Should return custom value from env var"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD") };
}

// =============================================================================
// Sample program
// =============================================================================

#[test]
#[serial]
fn config_sample_program_default() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_SAMPLE_PROGRAM") };

    let prog = neat_ai_discovery::config::sample_program();
    assert_eq!(prog, "sample", "Default sample program should be 'sample'");
}

#[test]
#[serial]
fn config_sample_program_custom() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_SAMPLE_PROGRAM", "/usr/bin/custom_sample") };

    let prog = neat_ai_discovery::config::sample_program();
    assert_eq!(prog, "/usr/bin/custom_sample");

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_SAMPLE_PROGRAM") };
}

// =============================================================================
// Focus unused observations
// =============================================================================

#[test]
#[serial]
fn config_focus_unused_observations_default() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS") };

    assert!(
        !neat_ai_discovery::config::focus_unused_observations(),
        "Focus unused observations should be disabled by default"
    );
}

#[test]
#[serial]
fn config_focus_unused_observations_enabled() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS", "true") };

    assert!(
        neat_ai_discovery::config::focus_unused_observations(),
        "Focus unused observations should be enabled with 'true'"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS") };
}
