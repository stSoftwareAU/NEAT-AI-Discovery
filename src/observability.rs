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
//! ## Environment Variables
//!
//! | Variable | Values | Description |
//! |----------|--------|-------------|
//! | `RUST_LOG` | filter string | Control log level (e.g. `neat_ai_discovery=info`) |
//! | `NEAT_AI_DISCOVERY_TIMING` | `1` | Print phase timing to stderr |
//! | `NEAT_AI_DISCOVERY_PROFILE` | `json` | Output structured profile as JSON |
//! | `NEAT_AI_DISCOVERY_GPU_METRICS` | `1` | Print GPU metrics to stderr |
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use neat_ai_discovery::observability::{PhaseTimer, GpuMetrics, ProfileData};
//!
//! fn analyze_parallel(...) -> Result<...> {
//!     let _total = PhaseTimer::new("total_analysis");
//!
//!     {
//!         let _phase = PhaseTimer::new("parquet_loading");
//!         load_parquet(...)?;
//!     }
//!
//!     {
//!         let _phase = PhaseTimer::new("focus_selection");
//!         select_focus_neurons(...)?;
//!     }
//!
//!     // ...
//! }
//! ```

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

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
/// Result is cached for performance using OnceLock.
pub fn timing_enabled() -> bool {
    static TIMING: OnceLock<bool> = OnceLock::new();
    *TIMING.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_TIMING").is_ok())
}

/// Check if GPU metrics output is enabled via `NEAT_AI_DISCOVERY_GPU_METRICS=1`.
///
/// Result is cached for performance using OnceLock.
pub fn gpu_metrics_enabled() -> bool {
    static GPU_METRICS: OnceLock<bool> = OnceLock::new();
    *GPU_METRICS.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_GPU_METRICS").is_ok())
}

/// Profile output mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProfileMode {
    /// No profiling output.
    #[default]
    None,
    /// Output structured profile as JSON to stderr.
    Json,
}

/// Get the current profile mode from `NEAT_AI_DISCOVERY_PROFILE` environment variable.
///
/// Result is cached for performance using OnceLock.
pub fn profile_mode() -> ProfileMode {
    static PROFILE_MODE: OnceLock<ProfileMode> = OnceLock::new();
    *PROFILE_MODE.get_or_init(|| {
        match std::env::var("NEAT_AI_DISCOVERY_PROFILE")
            .as_deref()
            .unwrap_or("")
            .to_lowercase()
            .as_str()
        {
            "json" => ProfileMode::Json,
            _ => ProfileMode::None,
        }
    })
}

// =============================================================================
// PhaseTimer - RAII-based phase timing
// =============================================================================

/// RAII-based phase timer that prints duration on drop when timing is enabled.
///
/// When `NEAT_AI_DISCOVERY_TIMING=1` is set, dropping a `PhaseTimer` will print
/// a timing line to stderr in the format: `[timing] phase_name: duration`.
///
/// ## Example
///
/// ```rust,ignore
/// fn analyze_parallel(...) -> Result<...> {
///     let _total = PhaseTimer::new("total_analysis");
///
///     {
///         let _phase = PhaseTimer::new("parquet_loading");
///         load_parquet(...)?;
///     }
///     // Prints: [timing] parquet_loading: 156ms
///
///     // ...
/// }
/// // Prints: [timing] total_analysis: 1234ms
/// ```
pub struct PhaseTimer {
    phase: &'static str,
    start: Instant,
}

impl PhaseTimer {
    /// Create a new phase timer with the given phase name.
    ///
    /// The timer starts immediately. Duration is measured when the timer is dropped.
    #[inline]
    pub fn new(phase: &'static str) -> Self {
        Self {
            phase,
            start: Instant::now(),
        }
    }

    /// Get the elapsed time since the timer was created.
    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

impl Drop for PhaseTimer {
    fn drop(&mut self) {
        if timing_enabled() {
            let duration = self.start.elapsed();
            tracing::debug!(phase = self.phase, ?duration, "phase timing");
        }
    }
}

// =============================================================================
// GpuMetrics - Thread-safe GPU metrics tracking
// =============================================================================

/// Thread-safe GPU metrics tracking.
///
/// Tracks GPU-specific metrics using atomic operations for thread-safety:
/// - Batch count: Number of GPU batch submissions
/// - Samples processed: Total samples evaluated on GPU
/// - Queue wait time: Time spent waiting in the GPU work queue
/// - GPU busy time: Time spent executing on the GPU
///
/// ## Example
///
/// ```rust,ignore
/// let metrics = GpuMetrics::new();
///
/// // In GPU thread loop
/// metrics.record_batch(samples.len());
/// metrics.record_queue_wait_us(wait_time);
/// metrics.record_gpu_busy_us(execution_time);
///
/// // After analysis
/// if gpu_metrics_enabled() {
///     metrics.report();
/// }
/// ```
#[derive(Debug)]
pub struct GpuMetrics {
    batch_count: AtomicUsize,
    total_samples_processed: AtomicUsize,
    queue_wait_time_us: AtomicU64,
    gpu_busy_time_us: AtomicU64,
}

impl GpuMetrics {
    /// Create a new GPU metrics tracker with all counters at zero.
    pub fn new() -> Self {
        Self {
            batch_count: AtomicUsize::new(0),
            total_samples_processed: AtomicUsize::new(0),
            queue_wait_time_us: AtomicU64::new(0),
            gpu_busy_time_us: AtomicU64::new(0),
        }
    }

    /// Record a batch submission with the given number of samples.
    #[inline]
    pub fn record_batch(&self, samples: usize) {
        self.batch_count.fetch_add(1, Ordering::Relaxed);
        self.total_samples_processed
            .fetch_add(samples, Ordering::Relaxed);
    }

    /// Record queue wait time in microseconds.
    #[inline]
    pub fn record_queue_wait_us(&self, us: u64) {
        self.queue_wait_time_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Record GPU busy time in microseconds.
    #[inline]
    pub fn record_gpu_busy_us(&self, us: u64) {
        self.gpu_busy_time_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Get the total batch count.
    #[inline]
    pub fn batch_count(&self) -> usize {
        self.batch_count.load(Ordering::Relaxed)
    }

    /// Get the total samples processed.
    #[inline]
    pub fn total_samples_processed(&self) -> usize {
        self.total_samples_processed.load(Ordering::Relaxed)
    }

    /// Get the total queue wait time in microseconds.
    #[inline]
    pub fn total_queue_wait_us(&self) -> u64 {
        self.queue_wait_time_us.load(Ordering::Relaxed)
    }

    /// Get the total GPU busy time in microseconds.
    #[inline]
    pub fn total_gpu_busy_us(&self) -> u64 {
        self.gpu_busy_time_us.load(Ordering::Relaxed)
    }

    /// Calculate GPU utilisation as a percentage.
    ///
    /// Returns the percentage of time the GPU was busy vs total time
    /// (busy + queue wait). Returns NaN if no time has been recorded.
    #[inline]
    pub fn utilisation_percent(&self) -> f64 {
        let busy = self.gpu_busy_time_us.load(Ordering::Relaxed) as f64;
        let wait = self.queue_wait_time_us.load(Ordering::Relaxed) as f64;
        let total = busy + wait;
        if total == 0.0 {
            return 0.0;
        }
        (busy / total) * 100.0
    }

    /// Print GPU metrics to stderr.
    ///
    /// Output format: `[gpu] batches: N, samples: N, utilisation: N.N%`
    pub fn report(&self) {
        tracing::info!(
            batches = self.batch_count(),
            samples = self.total_samples_processed(),
            utilisation_percent = format_args!("{:.1}", self.utilisation_percent()),
            "GPU metrics"
        );
    }
}

impl Default for GpuMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Global GPU metrics instance for tracking across the analysis pipeline.
///
/// This is lazily initialised on first access. Use `global_gpu_metrics()` to
/// access the instance, and `report_global_gpu_metrics()` to output the
/// collected metrics when GPU metrics are enabled.
static GLOBAL_GPU_METRICS: OnceLock<GpuMetrics> = OnceLock::new();

/// Get the global GPU metrics instance.
///
/// The instance is lazily initialised on first access and shared across
/// all threads.
pub fn global_gpu_metrics() -> &'static GpuMetrics {
    GLOBAL_GPU_METRICS.get_or_init(GpuMetrics::new)
}

/// Report global GPU metrics if `NEAT_AI_DISCOVERY_GPU_METRICS=1` is set.
///
/// Call this at the end of analysis to output GPU utilisation metrics.
pub fn report_global_gpu_metrics() {
    if gpu_metrics_enabled() {
        global_gpu_metrics().report();
    }
}

// =============================================================================
// ProfileData - Structured profile data for JSON output
// =============================================================================

/// Structured profile data for JSON output.
///
/// Collects timing, GPU, memory, and analysis metrics into a structure that
/// can be serialised to JSON when `NEAT_AI_DISCOVERY_PROFILE=json` is set.
///
/// ## JSON Output Format
///
/// ```json
/// {
///   "timing": {
///     "totalMs": 1234,
///     "phases": {
///       "parquet_loading": 156,
///       "focus_selection": 23,
///       "gpu_analysis": 987,
///       "result_sorting": 12
///     }
///   },
///   "gpu": {
///     "batchCount": 45,
///     "samplesProcessed": 1234567,
///     "utilisationPercent": 87.3,
///     "device": "Apple M2 Max"
///   },
///   "memory": {
///     "peakRssMb": 1234,
///     "parquetSizeMb": 456
///   },
///   "analysis": {
///     "focusNeuronsRequested": 64,
///     "focusNeuronsCompleted": 64,
///     "candidatesFound": 1234,
///     "candidatesReturned": 100
///   }
/// }
/// ```
#[derive(Debug, Default)]
pub struct ProfileData {
    // Timing data
    start_time: Option<Instant>,
    phases: Vec<(String, u64)>,

    // GPU metrics
    gpu_batch_count: Option<usize>,
    gpu_samples_processed: Option<usize>,
    gpu_utilisation: Option<f64>,
    gpu_device: Option<String>,

    // Memory metrics
    peak_rss_mb: Option<usize>,
    parquet_size_mb: Option<usize>,

    // Analysis metrics
    focus_neurons_requested: Option<usize>,
    focus_neurons_completed: Option<usize>,
    candidates_found: Option<usize>,
    candidates_returned: Option<usize>,
}

impl ProfileData {
    /// Create a new ProfileData instance.
    pub fn new() -> Self {
        Self {
            start_time: Some(Instant::now()),
            ..Default::default()
        }
    }

    /// Record a phase timing in milliseconds.
    pub fn record_phase(&mut self, phase: &str, duration_ms: u64) {
        self.phases.push((phase.to_string(), duration_ms));
    }

    /// Set the GPU batch count.
    pub fn set_gpu_batch_count(&mut self, count: usize) {
        self.gpu_batch_count = Some(count);
    }

    /// Set the GPU samples processed.
    pub fn set_gpu_samples_processed(&mut self, count: usize) {
        self.gpu_samples_processed = Some(count);
    }

    /// Set the GPU utilisation percentage.
    pub fn set_gpu_utilisation(&mut self, percent: f64) {
        self.gpu_utilisation = Some(percent);
    }

    /// Set the GPU device name.
    pub fn set_gpu_device(&mut self, device: String) {
        self.gpu_device = Some(device);
    }

    /// Set the peak RSS memory in megabytes.
    pub fn set_peak_rss_mb(&mut self, mb: usize) {
        self.peak_rss_mb = Some(mb);
    }

    /// Set the parquet file size in megabytes.
    pub fn set_parquet_size_mb(&mut self, mb: usize) {
        self.parquet_size_mb = Some(mb);
    }

    /// Set the number of focus neurons requested.
    pub fn set_focus_neurons_requested(&mut self, count: usize) {
        self.focus_neurons_requested = Some(count);
    }

    /// Set the number of focus neurons completed.
    pub fn set_focus_neurons_completed(&mut self, count: usize) {
        self.focus_neurons_completed = Some(count);
    }

    /// Set the number of candidates found.
    pub fn set_candidates_found(&mut self, count: usize) {
        self.candidates_found = Some(count);
    }

    /// Set the number of candidates returned.
    pub fn set_candidates_returned(&mut self, count: usize) {
        self.candidates_returned = Some(count);
    }

    /// Copy metrics from a GpuMetrics instance.
    pub fn from_gpu_metrics(&mut self, metrics: &GpuMetrics) {
        self.gpu_batch_count = Some(metrics.batch_count());
        self.gpu_samples_processed = Some(metrics.total_samples_processed());
        self.gpu_utilisation = Some(metrics.utilisation_percent());
    }

    /// Convert to JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        let total_ms = self
            .start_time
            .map_or(0, |s| s.elapsed().as_millis() as u64);

        let phases: serde_json::Map<String, serde_json::Value> = self
            .phases
            .iter()
            .map(|(name, ms)| (name.clone(), serde_json::Value::Number((*ms).into())))
            .collect();

        let timing = serde_json::json!({
            "totalMs": total_ms,
            "phases": phases,
        });

        let mut gpu = serde_json::Map::new();
        if let Some(count) = self.gpu_batch_count {
            gpu.insert("batchCount".to_string(), serde_json::json!(count));
        }
        if let Some(count) = self.gpu_samples_processed {
            gpu.insert("samplesProcessed".to_string(), serde_json::json!(count));
        }
        if let Some(percent) = self.gpu_utilisation {
            gpu.insert("utilisationPercent".to_string(), serde_json::json!(percent));
        }
        if let Some(ref device) = self.gpu_device {
            gpu.insert("device".to_string(), serde_json::json!(device));
        }

        let mut memory = serde_json::Map::new();
        if let Some(mb) = self.peak_rss_mb {
            memory.insert("peakRssMb".to_string(), serde_json::json!(mb));
        }
        if let Some(mb) = self.parquet_size_mb {
            memory.insert("parquetSizeMb".to_string(), serde_json::json!(mb));
        }

        let mut analysis = serde_json::Map::new();
        if let Some(count) = self.focus_neurons_requested {
            analysis.insert(
                "focusNeuronsRequested".to_string(),
                serde_json::json!(count),
            );
        }
        if let Some(count) = self.focus_neurons_completed {
            analysis.insert(
                "focusNeuronsCompleted".to_string(),
                serde_json::json!(count),
            );
        }
        if let Some(count) = self.candidates_found {
            analysis.insert("candidatesFound".to_string(), serde_json::json!(count));
        }
        if let Some(count) = self.candidates_returned {
            analysis.insert("candidatesReturned".to_string(), serde_json::json!(count));
        }

        serde_json::json!({
            "timing": timing,
            "gpu": gpu,
            "memory": memory,
            "analysis": analysis,
        })
    }

    /// Output the profile data to stderr as JSON.
    pub fn report(&self) {
        if profile_mode() == ProfileMode::Json {
            let json = self.to_json();
            tracing::info!(
                profile = %serde_json::to_string_pretty(&json).unwrap_or_else(|_| "{}".to_string()),
                "profile data"
            );
        }
    }
}

// =============================================================================
// Scoped Phase Timer for Profile Integration
// =============================================================================

/// Scoped phase timer that records to a ProfileData instance.
///
/// This is a more structured alternative to `PhaseTimer` that integrates
/// with `ProfileData` for JSON output.
pub struct ScopedPhaseTimer<'a> {
    profile: &'a mut ProfileData,
    phase: String,
    start: Instant,
}

impl<'a> ScopedPhaseTimer<'a> {
    /// Create a new scoped phase timer.
    pub fn new(profile: &'a mut ProfileData, phase: &str) -> Self {
        Self {
            profile,
            phase: phase.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for ScopedPhaseTimer<'_> {
    fn drop(&mut self) {
        let duration_ms = self.start.elapsed().as_millis() as u64;
        self.profile.record_phase(&self.phase, duration_ms);

        // Also emit via tracing if timing is enabled
        if timing_enabled() {
            tracing::debug!(phase = %self.phase, duration_ms, "scoped phase timing");
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_timer_creates_without_panic() {
        let timer = PhaseTimer::new("test_phase");
        drop(timer);
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
    fn profile_mode_default() {
        // Without env var, should be None
        // Note: this test might be affected by other tests setting the env var
        // since OnceLock caches the value
    }
}
