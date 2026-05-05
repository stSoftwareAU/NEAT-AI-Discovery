//! Structured observability and profiling hooks for NEAT-AI Discovery (Issue #214, #575).
//!
//! This module provides infrastructure for understanding where time is spent during
//! discovery analysis, enabling:
//! - Diagnosis of slow discovery runs in production
//! - Identification of optimisation opportunities
//! - Debugging of GPU-related performance issues
//! - Understanding of resource utilisation patterns
//! - Structured, levelled logging via the `tracing` crate (Issue #575)
//!
//! ## Submodules
//!
//! - [`phase_timer`] — RAII-based phase timing ([`PhaseTimer`], [`ScopedPhaseTimer`])
//! - [`gpu_metrics`] — Thread-safe GPU metrics tracking ([`GpuMetrics`])
//! - [`profile`] — Structured profile data for JSON output ([`ProfileData`])
//!
//! ## Environment Variables
//!
//! | Variable | Values | Description |
//! |----------|--------|-------------|
//! | `RUST_LOG` | filter string | Control log level (e.g. `neat_ai_discovery=info`) |
//! | `NEAT_AI_DISCOVERY_TIMING` | `1` | Print phase timing to stderr |
//! | `NEAT_AI_DISCOVERY_PROFILE` | `json` | Output structured profile as JSON |
//! | `NEAT_AI_DISCOVERY_GPU_METRICS` | `1` | Print GPU metrics to stderr |
//! | `NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD` | f32 (>1) | Threshold (default 10) above which prediction-vs-actual mismatches are logged via `tracing::warn!` (Issue #1165) |

pub mod gain_floor_metrics;
pub mod gpu_metrics;
pub mod phase_timer;
pub mod profile;

// Re-export all public types for backward compatibility.
pub use gain_floor_metrics::*;
pub use gpu_metrics::*;
pub use phase_timer::*;
pub use profile::*;

// =============================================================================
// Tracing Subscriber Initialisation (Issue #575)
// =============================================================================

/// Initialise the `tracing` subscriber with `EnvFilter` for structured logging.
///
/// The subscriber writes human-readable output to stderr, controlled by the
/// `RUST_LOG` environment variable (e.g. `RUST_LOG=neat_ai_discovery=info`).
///
/// If `RUST_LOG` is not set, the default level is `warn` so that existing
/// behaviour (minimal output) is preserved.
///
/// This function is idempotent — calling it more than once is safe (subsequent
/// calls are no-ops).
pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt;
    use tracing_subscriber::prelude::*;

    // Use try_init so that repeated calls (or test environments that already
    // have a subscriber) do not panic.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(true).with_writer(std::io::stderr))
        .try_init();
}

// =============================================================================
// Environment Variable Parsing
// =============================================================================

/// Check if phase timing is enabled via `NEAT_AI_DISCOVERY_TIMING=1`.
///
/// Delegates to [`crate::config::timing()`].
pub fn timing_enabled() -> bool {
    crate::config::timing()
}

/// Check if GPU metrics output is enabled via `NEAT_AI_DISCOVERY_GPU_METRICS=1`.
///
/// Delegates to [`crate::config::gpu_metrics()`].
pub fn gpu_metrics_enabled() -> bool {
    crate::config::gpu_metrics()
}

/// Profile output mode.
///
/// Re-exported from [`crate::config::ProfileMode`] for backward compatibility.
pub type ProfileMode = crate::config::ProfileMode;

/// Get the current profile mode from `NEAT_AI_DISCOVERY_PROFILE` environment variable.
///
/// Delegates to [`crate::config::profile_mode()`].
pub fn profile_mode() -> ProfileMode {
    crate::config::profile_mode()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_timer_records_non_zero_elapsed() {
        let timer = PhaseTimer::new("test_phase");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let elapsed = timer.elapsed_ms();
        drop(timer);
        assert!(
            elapsed >= 1,
            "PhaseTimer should record non-zero elapsed time, got {elapsed}ms"
        );
    }

    #[test]
    fn gpu_metrics_atomic_operations() {
        let metrics = GpuMetrics::new();

        metrics.record_batch(100);
        assert_eq!(metrics.batch_count(), 1);
        assert_eq!(metrics.total_samples_processed(), 100);

        metrics.record_queue_wait_us(1000);
        assert_eq!(metrics.total_queue_wait_us(), 1000);

        metrics.record_gpu_busy_us(4000);
        assert_eq!(metrics.total_gpu_busy_us(), 4000);

        // Utilisation = 4000 / (4000 + 1000) = 80%
        assert!((metrics.utilisation_percent() - 80.0).abs() < 0.1);
    }

    #[test]
    fn profile_data_json_structure() {
        let mut profile = ProfileData::new();
        profile.record_phase("test", 100);
        profile.set_gpu_batch_count(10);
        profile.set_focus_neurons_requested(5);

        let json = profile.to_json();

        assert!(json["timing"]["phases"]["test"].is_number());
        assert_eq!(json["gpu"]["batchCount"], 10);
        assert_eq!(json["analysis"]["focusNeuronsRequested"], 5);
    }

    #[test]
    fn profile_data_to_json_includes_timing_structure() {
        let profile = ProfileData::new();
        let json = profile.to_json();

        // The JSON output must always contain top-level keys
        assert!(
            json["timing"].is_object(),
            "JSON should contain a 'timing' object"
        );
        assert!(
            json["timing"]["totalMs"].is_number(),
            "timing should include totalMs"
        );
        assert!(
            json["timing"]["phases"].is_object(),
            "timing should include phases"
        );
    }
}
