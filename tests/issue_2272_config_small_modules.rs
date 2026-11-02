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
//! `OnceLock`, so whichever test reads them first freezes the value and they
//! cannot be exercised in-process here. They do not go through `parse_env`:
//! `timing` and `gpu_metrics` are presence-only `env::var(..).is_ok()` reads,
//! and `profile_mode` lowercases without trimming. The helper unit tests in
//! `src/config/helpers.rs` therefore do not cover them either. Their
//! presence-only and opt-out behaviour is out of scope for this file and is
//! tracked under #2213.
//!
//! Whitespace cases for `noise_signal_threshold`, `dominance_threshold` and
//! `gradient_threshold` are not repeated here.
//! `tests/issue_2006_numeric_env_override_trim.rs` covers the trim for those
//! three accessors only.
//!
//! The detection-threshold pins (`noise_signal_threshold`,
//! `dominance_threshold`, `gradient_threshold`) verify the finite-positive
//! override validation from Issue #2438. Their consumer gates are:
//!
//! - `noise_signal.rs::detect_noisy_neurons` (gate `noise_to_signal_ratio <=
//!   threshold`, then the improvement factor divides by the threshold);
//! - `input_sensitivity.rs::detect_dominant_inputs` (gate `sensitivity_score <=
//!   config.dominance_threshold`, factor divides by it);
//! - `input_sensitivity.rs::detect_threshold_effects` (gate `effective_gradient <
//!   config.gradient_threshold`, factor divides by it).
//!
//! Hostile threshold overrides use the caller's fallback before reaching
//! those consumer gates.

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
// src/config/detection.rs — float thresholds (Issue #2438)
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn noise_signal_threshold_nan_uses_fallback() {
    let value = with_env("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD", "NaN", || {
        noise_signal_threshold(0.5)
    });
    assert_eq!(
        value, 0.5,
        "NaN override must use the caller-supplied fallback"
    );
}

#[test]
#[serial]
fn noise_signal_threshold_infinity_uses_fallback() {
    let value = with_env("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD", "inf", || {
        noise_signal_threshold(0.5)
    });
    assert_eq!(
        value, 0.5,
        "inf override must use the caller-supplied fallback"
    );
}

#[test]
#[serial]
fn noise_signal_threshold_negative_one_uses_fallback() {
    let value = with_env("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD", "-1", || {
        noise_signal_threshold(0.5)
    });
    assert_eq!(
        value, 0.5,
        "negative override must use the caller-supplied fallback"
    );
}

#[test]
#[serial]
fn dominance_threshold_nan_uses_fallback() {
    let value = with_env("NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD", "NaN", || {
        dominance_threshold(0.5)
    });
    assert_eq!(
        value, 0.5,
        "NaN override must use the caller-supplied fallback"
    );
}

#[test]
#[serial]
fn dominance_threshold_infinity_uses_fallback() {
    let value = with_env("NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD", "inf", || {
        dominance_threshold(0.5)
    });
    assert_eq!(
        value, 0.5,
        "inf override must use the caller-supplied fallback"
    );
}

#[test]
#[serial]
fn dominance_threshold_negative_one_uses_fallback() {
    let value = with_env("NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD", "-1", || {
        dominance_threshold(0.5)
    });
    assert_eq!(
        value, 0.5,
        "negative override must use the caller-supplied fallback"
    );
}

#[test]
#[serial]
fn gradient_threshold_nan_uses_fallback() {
    let value = with_env("NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD", "NaN", || {
        gradient_threshold(0.5)
    });
    assert_eq!(
        value, 0.5,
        "NaN override must use the caller-supplied fallback"
    );
}

#[test]
#[serial]
fn gradient_threshold_infinity_uses_fallback() {
    let value = with_env("NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD", "inf", || {
        gradient_threshold(0.5)
    });
    assert_eq!(
        value, 0.5,
        "inf override must use the caller-supplied fallback"
    );
}

#[test]
#[serial]
fn gradient_threshold_negative_one_uses_fallback() {
    let value = with_env("NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD", "-1", || {
        gradient_threshold(0.5)
    });
    assert_eq!(
        value, 0.5,
        "negative override must use the caller-supplied fallback"
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
