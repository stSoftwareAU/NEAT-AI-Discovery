//! Detection tuning — threshold overrides for noise, dominance, and gradient.

/// Get the noise-to-signal detection threshold.
///
/// Set `NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD` to override.
/// Requires each detection module's default to be passed as fallback.
pub fn noise_signal_threshold(default: f32) -> f32 {
    std::env::var("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Get the input dominance detection threshold.
///
/// Set `NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD` to override.
/// Requires each detection module's default to be passed as fallback.
pub fn dominance_threshold(default: f32) -> f32 {
    std::env::var("NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Get the gradient detection threshold.
///
/// Set `NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD` to override.
/// Requires each detection module's default to be passed as fallback.
pub fn gradient_threshold(default: f32) -> f32 {
    std::env::var("NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Hidden-neuron count above which expensive discovery modules are tiered out
/// on non-escalation passes (Issue #1547).
///
/// Set `NEAT_AI_DISCOVERY_MODULE_TIERING_HIDDEN_THRESHOLD` to override. A value
/// of `0` disables creature-scale module tiering entirely (every module always
/// runs). Falls back to
/// [`crate::analysis::module_tiering::DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD`]
/// when unset or unparseable.
pub fn module_tiering_hidden_neuron_threshold() -> usize {
    std::env::var("NEAT_AI_DISCOVERY_MODULE_TIERING_HIDDEN_THRESHOLD")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(crate::analysis::module_tiering::DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD)
}

/// Read the env-var override for target-neuron cooldown consecutive failures
/// (Issue #1130).
///
/// Returns `Some(n)` when `NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES` is set
/// to a valid `u32`, otherwise `None` so callers fall back to the compiled
/// default `TARGET_COOLDOWN_CONSECUTIVE_FAILURES`.
pub fn target_cooldown_consecutive_failures_env() -> Option<u32> {
    std::env::var("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .filter(|v| *v >= 1)
}

/// Read the env-var override for target-neuron cooldown duration in epochs
/// (Issue #1130).
///
/// Returns `Some(n)` when `NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS` is set to
/// a valid `u64`, otherwise `None` so callers fall back to the compiled default
/// `TARGET_COOLDOWN_EPOCHS`.
pub fn target_cooldown_epochs_env() -> Option<u64> {
    std::env::var("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .filter(|v| *v >= 1)
}

/// Read the env-var override for the within-batch target-failure short-circuit
/// limit (Issue #1164).
///
/// Returns `Some(n)` when `NEAT_AI_DISCOVERY_BATCH_TARGET_FAILURE_LIMIT` is set
/// to a valid `u32 >= 1`, otherwise `None` so callers fall back to the
/// compiled default `WITHIN_BATCH_TARGET_FAILURE_LIMIT`.
pub fn within_batch_target_failure_limit_env() -> Option<u32> {
    std::env::var("NEAT_AI_DISCOVERY_BATCH_TARGET_FAILURE_LIMIT")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .filter(|v| *v >= 1)
}
