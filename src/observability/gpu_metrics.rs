//! Thread-safe GPU metrics tracking.
//!
//! [`GpuMetrics`] uses atomic operations to safely aggregate GPU batch counts,
//! sample throughput, queue wait time, and GPU busy time across threads.
//! A global instance is available via [`global_gpu_metrics()`].

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::gpu_metrics_enabled;

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
    effective_batch_size: AtomicUsize,
    batch_size_reductions: AtomicUsize,
}

impl GpuMetrics {
    /// Create a new GPU metrics tracker with all counters at zero.
    pub fn new() -> Self {
        Self {
            batch_count: AtomicUsize::new(0),
            total_samples_processed: AtomicUsize::new(0),
            queue_wait_time_us: AtomicU64::new(0),
            gpu_busy_time_us: AtomicU64::new(0),
            effective_batch_size: AtomicUsize::new(0),
            batch_size_reductions: AtomicUsize::new(0),
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

    /// Record a batch size reduction due to memory exhaustion (Issue #1083).
    #[inline]
    pub fn record_batch_size_reduction(&self, new_batch_size: usize) {
        self.effective_batch_size
            .store(new_batch_size, Ordering::Relaxed);
        self.batch_size_reductions.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the current effective batch size (0 if never reduced).
    #[inline]
    pub fn effective_batch_size(&self) -> usize {
        self.effective_batch_size.load(Ordering::Relaxed)
    }

    /// Get the number of batch size reductions due to memory exhaustion.
    #[inline]
    pub fn batch_size_reductions(&self) -> usize {
        self.batch_size_reductions.load(Ordering::Relaxed)
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
        let reductions = self.batch_size_reductions();
        let effective = self.effective_batch_size();
        tracing::info!(
            batches = self.batch_count(),
            samples = self.total_samples_processed(),
            utilisation_percent = format_args!("{:.1}", self.utilisation_percent()),
            batch_size_reductions = reductions,
            effective_batch_size = effective,
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
