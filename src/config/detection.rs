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
