//! Issue #2272 (part of #2212, tracker #2083) — pin today's return values for
//! hostile `NEAT_AI_DISCOVERY_*` input on the un-cached accessors in
//! `src/config/detection.rs` and `src/config/observability.rs`.
//!
//! These tests record what the accessors return now, not what they should
//! return. Hostile values (`NaN`, `inf`, negatives, zero, garbage) are pinned
//! exactly as they behave today, so any change to that behaviour shows up as a
//! deliberate test edit rather than a silent drift.
//!
//! `timing`, `profile_mode` and `gpu_metrics` are memoised in a process-wide
//! `OnceLock`, so their value is frozen by whichever test reads it first. They
//! cannot be exercised in-process here; their trimming and parsing are covered
//! by the helper unit tests in `src/config/helpers.rs` instead.
//!
//! Whitespace-padded values are not repeated here.
//! `tests/issue_2006_numeric_env_override_trim.rs` already covers the trim on
//! these accessors.
//!
//! The detection-threshold pins (`noise_signal_threshold`,
//! `dominance_threshold`, `gradient_threshold`) record a defect filed as
//! Issue #2438: a non-finite or negative threshold is passed straight through
//! to the consumer gates. Those gates are:
//!
//! - `src/analysis/detection/noise_signal.rs:164` (`ratio <= threshold`, with
//!   the division at :171);
//! - `src/analysis/detection/input_sensitivity.rs:277` (`score <=
//!   dominance_threshold`, with the division at :291-293);
//! - `src/analysis/detection/input_sensitivity.rs:449` (`gradient <
//!   gradient_threshold`, with the division at :484-486).
//!
//! When Issue #2438 is fixed, the fix must update the pins below deliberately.

use neat_ai_discovery::analysis::module_tiering::DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD;
use neat_ai_discovery::config::{
    calibration_miss_threshold, dominance_threshold, gradient_threshold,
    module_tiering_hidden_neuron_threshold, noise_signal_threshold,
    strict_candidate_reconciliation, target_cooldown_consecutive_failures_env,
    target_cooldown_epochs_env, within_batch_target_failure_limit_env,
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

// ---------------------------------------------------------------------------
// src/config/detection.rs — float thresholds (defect: Issue #2438)
// ---------------------------------------------------------------------------

// Issue #2438's fix will change this to the default (0.5).
#[test]
#[serial]
fn noise_signal_threshold_passes_nan_through() {
    let value = with_env("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD", "NaN", || {
        noise_signal_threshold(0.5)
    });
    assert!(
        value.is_nan(),
        "NaN override is currently passed through as NaN, got {value}"
    );
}

// Issue #2438's fix will change this to the default (0.5).
#[test]
#[serial]
fn noise_signal_threshold_passes_infinity_through() {
    let value = with_env("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD", "inf", || {
        noise_signal_threshold(0.5)
    });
    assert_eq!(
        value,
        f32::INFINITY,
        "inf override is currently passed through as infinity"
    );
}

// Issue #2438's fix will change this to the default (0.5).
#[test]
#[serial]
fn noise_signal_threshold_passes_negative_one_through() {
    let value = with_env("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD", "-1", || {
        noise_signal_threshold(0.5)
    });
    assert_eq!(
        value, -1.0,
        "negative override is currently passed through unchanged"
    );
}

// Issue #2438's fix will change this to the default (0.5).
#[test]
#[serial]
fn dominance_threshold_passes_nan_through() {
    let value = with_env("NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD", "NaN", || {
        dominance_threshold(0.5)
    });
    assert!(
        value.is_nan(),
        "NaN override is currently passed through as NaN, got {value}"
    );
}

// Issue #2438's fix will change this to the default (0.5).
#[test]
#[serial]
fn dominance_threshold_passes_infinity_through() {
    let value = with_env("NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD", "inf", || {
        dominance_threshold(0.5)
    });
    assert_eq!(
        value,
        f32::INFINITY,
        "inf override is currently passed through as infinity"
    );
}

// Issue #2438's fix will change this to the default (0.5).
#[test]
#[serial]
fn dominance_threshold_passes_negative_one_through() {
    let value = with_env("NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD", "-1", || {
        dominance_threshold(0.5)
    });
    assert_eq!(
        value, -1.0,
        "negative override is currently passed through unchanged"
    );
}

// Issue #2438's fix will change this to the default (0.5).
#[test]
#[serial]
fn gradient_threshold_passes_nan_through() {
    let value = with_env("NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD", "NaN", || {
        gradient_threshold(0.5)
    });
    assert!(
        value.is_nan(),
        "NaN override is currently passed through as NaN, got {value}"
    );
}

// Issue #2438's fix will change this to the default (0.5).
#[test]
#[serial]
fn gradient_threshold_passes_infinity_through() {
    let value = with_env("NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD", "inf", || {
        gradient_threshold(0.5)
    });
    assert_eq!(
        value,
        f32::INFINITY,
        "inf override is currently passed through as infinity"
    );
}

// Issue #2438's fix will change this to the default (0.5).
#[test]
#[serial]
fn gradient_threshold_passes_negative_one_through() {
    let value = with_env("NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD", "-1", || {
        gradient_threshold(0.5)
    });
    assert_eq!(
        value, -1.0,
        "negative override is currently passed through unchanged"
    );
}

// ---------------------------------------------------------------------------
// src/config/detection.rs — module tiering threshold
// ---------------------------------------------------------------------------

/// A value of `0` disables tiering via `tiering_applies`
/// (`src/analysis/module_tiering.rs`, `threshold > 0 && ..`).
#[test]
#[serial]
fn module_tiering_hidden_neuron_threshold_zero_disables_tiering() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_MODULE_TIERING_HIDDEN_THRESHOLD",
        "0",
        module_tiering_hidden_neuron_threshold,
    );
    assert_eq!(
        value, 0,
        "explicit 0 must reach tiering_applies as a disabling threshold"
    );
}

#[test]
#[serial]
fn module_tiering_hidden_neuron_threshold_garbage_falls_back_to_default() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_MODULE_TIERING_HIDDEN_THRESHOLD",
        "garbage",
        module_tiering_hidden_neuron_threshold,
    );
    assert_eq!(
        value, DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD,
        "unparsable override must fall back to the compiled default"
    );
}

// ---------------------------------------------------------------------------
// src/config/detection.rs — target cooldown and batch failure limit (u32/u64)
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn target_cooldown_consecutive_failures_env_zero_is_none() {
    let value = with_env("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES", "0", || {
        target_cooldown_consecutive_failures_env()
    });
    assert_eq!(value, None, "0 is filtered out by the >= 1 guard");
}

#[test]
#[serial]
fn target_cooldown_consecutive_failures_env_negative_is_none() {
    let value = with_env("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES", "-1", || {
        target_cooldown_consecutive_failures_env()
    });
    assert_eq!(value, None, "-1 does not parse as u32, so it is None");
}

#[test]
#[serial]
fn target_cooldown_consecutive_failures_env_garbage_is_none() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES",
        "garbage",
        target_cooldown_consecutive_failures_env,
    );
    assert_eq!(value, None, "unparsable override must be None");
}

#[test]
#[serial]
fn target_cooldown_consecutive_failures_env_positive_control_three_is_some() {
    let value = with_env("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES", "3", || {
        target_cooldown_consecutive_failures_env()
    });
    assert_eq!(
        value,
        Some(3),
        "positive control: a valid override must be read, so the None results above come from the value"
    );
}

#[test]
#[serial]
fn target_cooldown_epochs_env_zero_is_none() {
    let value = with_env("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS", "0", || {
        target_cooldown_epochs_env()
    });
    assert_eq!(value, None, "0 is filtered out by the >= 1 guard");
}

#[test]
#[serial]
fn target_cooldown_epochs_env_negative_is_none() {
    let value = with_env("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS", "-1", || {
        target_cooldown_epochs_env()
    });
    assert_eq!(value, None, "-1 does not parse as u64, so it is None");
}

#[test]
#[serial]
fn target_cooldown_epochs_env_garbage_is_none() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS",
        "garbage",
        target_cooldown_epochs_env,
    );
    assert_eq!(value, None, "unparsable override must be None");
}

#[test]
#[serial]
fn target_cooldown_epochs_env_positive_control_three_is_some() {
    let value = with_env("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS", "3", || {
        target_cooldown_epochs_env()
    });
    assert_eq!(
        value,
        Some(3),
        "positive control: a valid override must be read, so the None results above come from the value"
    );
}

#[test]
#[serial]
fn within_batch_target_failure_limit_env_zero_is_none() {
    let value = with_env("NEAT_AI_DISCOVERY_BATCH_TARGET_FAILURE_LIMIT", "0", || {
        within_batch_target_failure_limit_env()
    });
    assert_eq!(value, None, "0 is filtered out by the >= 1 guard");
}

#[test]
#[serial]
fn within_batch_target_failure_limit_env_negative_is_none() {
    let value = with_env("NEAT_AI_DISCOVERY_BATCH_TARGET_FAILURE_LIMIT", "-1", || {
        within_batch_target_failure_limit_env()
    });
    assert_eq!(value, None, "-1 does not parse as u32, so it is None");
}

#[test]
#[serial]
fn within_batch_target_failure_limit_env_garbage_is_none() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_BATCH_TARGET_FAILURE_LIMIT",
        "garbage",
        within_batch_target_failure_limit_env,
    );
    assert_eq!(value, None, "unparsable override must be None");
}

#[test]
#[serial]
fn within_batch_target_failure_limit_env_positive_control_three_is_some() {
    let value = with_env("NEAT_AI_DISCOVERY_BATCH_TARGET_FAILURE_LIMIT", "3", || {
        within_batch_target_failure_limit_env()
    });
    assert_eq!(
        value,
        Some(3),
        "positive control: a valid override must be read, so the None results above come from the value"
    );
}

// ---------------------------------------------------------------------------
// src/config/observability.rs — calibration miss threshold
// ---------------------------------------------------------------------------

// The default constant lives in pub(crate)
// analysis::diagnostics::mcmc_diagnostics, so an integration test cannot name
// it. The literal 10.0 is therefore pinned here.
#[test]
#[serial]
fn calibration_miss_threshold_nan_falls_back_to_default() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD",
        "NaN",
        calibration_miss_threshold,
    );
    assert_eq!(value, 10.0, "NaN is non-finite, so the default applies");
}

#[test]
#[serial]
fn calibration_miss_threshold_one_falls_back_to_default() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD",
        "1.0",
        calibration_miss_threshold,
    );
    assert_eq!(value, 10.0, "1.0 is not > 1.0, so the default applies");
}

#[test]
#[serial]
fn calibration_miss_threshold_half_falls_back_to_default() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD",
        "0.5",
        calibration_miss_threshold,
    );
    assert_eq!(value, 10.0, "0.5 is not > 1.0, so the default applies");
}

#[test]
#[serial]
fn calibration_miss_threshold_positive_control_four_is_read() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD",
        "4.0",
        calibration_miss_threshold,
    );
    assert_eq!(
        value, 4.0,
        "positive control: a valid override above 1.0 must be read"
    );
}

// ---------------------------------------------------------------------------
// src/config/observability.rs — strict candidate reconciliation
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn strict_candidate_reconciliation_zero_is_false() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_STRICT_CANDIDATE_RECONCILIATION",
        "0",
        strict_candidate_reconciliation,
    );
    assert!(!value, "0 forces warn-only, got {value}");
}

#[test]
#[serial]
fn strict_candidate_reconciliation_false_word_is_false() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_STRICT_CANDIDATE_RECONCILIATION",
        "false",
        strict_candidate_reconciliation,
    );
    assert!(!value, "the word false forces warn-only, got {value}");
}

#[test]
#[serial]
fn strict_candidate_reconciliation_garbage_is_debug_assertions() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_STRICT_CANDIDATE_RECONCILIATION",
        "garbage",
        strict_candidate_reconciliation,
    );
    assert_eq!(
        value,
        cfg!(debug_assertions),
        "unparsable override falls back to the build-profile default"
    );
}

#[test]
#[serial]
fn strict_candidate_reconciliation_positive_control_one_is_true() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_STRICT_CANDIDATE_RECONCILIATION",
        "1",
        strict_candidate_reconciliation,
    );
    assert!(value, "positive control: 1 forces strict mode on");
}
