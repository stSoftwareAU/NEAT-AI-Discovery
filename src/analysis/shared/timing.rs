//! GPU and CPU timing types for analysis profiling.
//!
//! Contains the [`TimingCollector`] for aggregating timing data from parallel
//! focus neuron processing, plus the RAII [`TimingScope`] guard and the
//! data-carrying types ([`ShaderTiming`], [`GpuTimingBreakdown`],
//! [`CpuTimingBreakdown`], [`AnalysisTiming`]).

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// =============================================================================
// GPU Timing Types (Issue #195)
// =============================================================================

/// Timing statistics for a single shader type.
///
/// Collects call counts and timing data for performance diagnostics.
#[derive(Debug, Clone, Default)]
pub struct ShaderTiming {
    /// Number of times this shader was executed.
    pub calls: u32,
    /// Total execution time in milliseconds.
    pub total_ms: f64,
    /// Average execution time per call in milliseconds.
    pub avg_ms: f64,
}

/// GPU-side timing breakdown.
///
/// Tracks time spent in GPU operations including shader execution
/// and buffer transfers.
#[derive(Debug, Clone, Default)]
pub struct GpuTimingBreakdown {
    /// Total time spent in shader execution (all shaders combined) in milliseconds.
    pub shader_execution_ms: f64,
    /// Total time spent in buffer mapping/transfers in milliseconds.
    pub buffer_transfer_ms: f64,
    /// Per-shader timing statistics.
    /// Keys are shader names: "helpful", "harmful", "relu", "activation", "bias"
    pub shader_timings: HashMap<String, ShaderTiming>,
}

/// CPU-side timing breakdown.
///
/// Tracks time spent in CPU operations during analysis.
#[derive(Debug, Clone, Default)]
pub struct CpuTimingBreakdown {
    /// Time spent building samples for GPU evaluation in milliseconds.
    pub sample_building_ms: f64,
    /// Time spent processing results from GPU in milliseconds.
    pub result_processing_ms: f64,
}

/// Complete timing data for an analysis run.
///
/// This is only populated when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
/// Provides detailed timing breakdown for performance diagnostics.
#[derive(Debug, Clone, Default)]
pub struct AnalysisTiming {
    /// Total wall-clock time for the analysis in milliseconds.
    pub total_analysis_ms: f64,
    /// GPU-side timing breakdown.
    pub gpu: GpuTimingBreakdown,
    /// CPU-side timing breakdown.
    pub cpu: CpuTimingBreakdown,
}

// =============================================================================
// Timing Collector (Issue #195)
// =============================================================================

/// Thread-safe timing collector for GPU kernel profiling.
///
/// This collector aggregates timing data from multiple threads (parallel focus neuron processing)
/// and provides a consolidated view of GPU and CPU timing.
///
/// Only collects timing when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
#[derive(Debug)]
pub struct TimingCollector {
    enabled: bool,
    start_time: Instant,
    /// Per-shader timing data: (name -> (calls, total_ns))
    shader_timings: Mutex<HashMap<String, (u32, u64)>>,
    /// Buffer transfer time in nanoseconds
    buffer_transfer_ns: AtomicU64,
    /// Sample building time in nanoseconds
    sample_building_ns: AtomicU64,
    /// Result processing time in nanoseconds
    result_processing_ns: AtomicU64,
}

impl TimingCollector {
    /// Create a new timing collector.
    ///
    /// If `enabled` is false, all timing operations are no-ops.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            start_time: Instant::now(),
            shader_timings: Mutex::new(HashMap::new()),
            buffer_transfer_ns: AtomicU64::new(0),
            sample_building_ns: AtomicU64::new(0),
            result_processing_ns: AtomicU64::new(0),
        }
    }

    /// Check if timing collection is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record a shader execution.
    pub fn record_shader(&self, shader_name: &str, duration_ns: u64) {
        if !self.enabled {
            return;
        }
        let mut timings = crate::analysis::utils::lock_contention::traced_lock_default(
            &self.shader_timings,
            "shader_timings",
        );
        let entry = timings.entry(shader_name.to_string()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += duration_ns;
    }

    /// Record buffer transfer time.
    pub fn record_buffer_transfer(&self, duration_ns: u64) {
        if !self.enabled {
            return;
        }
        self.buffer_transfer_ns
            .fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Record sample building time.
    pub fn record_sample_building(&self, duration_ns: u64) {
        if !self.enabled {
            return;
        }
        self.sample_building_ns
            .fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Record result processing time.
    pub fn record_result_processing(&self, duration_ns: u64) {
        if !self.enabled {
            return;
        }
        self.result_processing_ns
            .fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Finalize and return the collected timing data.
    ///
    /// Returns `None` if timing is disabled.
    pub fn finalize(&self) -> Option<AnalysisTiming> {
        if !self.enabled {
            return None;
        }

        let total_analysis_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;

        let shader_timings_lock = crate::analysis::utils::lock_contention::traced_lock_default(
            &self.shader_timings,
            "shader_timings_finalize",
        );
        let mut shader_timings = HashMap::new();
        let mut total_shader_ns: u64 = 0;

        for (name, (calls, total_ns)) in shader_timings_lock.iter() {
            let total_ms = *total_ns as f64 / 1_000_000.0;
            let avg_ms = if *calls > 0 {
                total_ms / (*calls as f64)
            } else {
                0.0
            };
            shader_timings.insert(
                name.clone(),
                ShaderTiming {
                    calls: *calls,
                    total_ms,
                    avg_ms,
                },
            );
            total_shader_ns += total_ns;
        }

        let buffer_transfer_ns = self.buffer_transfer_ns.load(Ordering::Relaxed);
        let sample_building_ns = self.sample_building_ns.load(Ordering::Relaxed);
        let result_processing_ns = self.result_processing_ns.load(Ordering::Relaxed);

        Some(AnalysisTiming {
            total_analysis_ms,
            gpu: GpuTimingBreakdown {
                shader_execution_ms: total_shader_ns as f64 / 1_000_000.0,
                buffer_transfer_ms: buffer_transfer_ns as f64 / 1_000_000.0,
                shader_timings,
            },
            cpu: CpuTimingBreakdown {
                sample_building_ms: sample_building_ns as f64 / 1_000_000.0,
                result_processing_ms: result_processing_ns as f64 / 1_000_000.0,
            },
        })
    }
}

impl Default for TimingCollector {
    fn default() -> Self {
        Self::new(false)
    }
}

/// RAII guard for timing a scope.
///
/// Records the duration when dropped.
pub struct TimingScope<'a> {
    collector: &'a TimingCollector,
    shader_name: Option<String>,
    category: TimingCategory,
    start: Instant,
}

/// Category of timing to record.
#[derive(Debug, Clone, Copy)]
pub enum TimingCategory {
    Shader,
    BufferTransfer,
    SampleBuilding,
    ResultProcessing,
}

impl<'a> TimingScope<'a> {
    /// Create a new timing scope for a shader.
    pub fn shader(collector: &'a TimingCollector, shader_name: &str) -> Self {
        Self {
            collector,
            shader_name: Some(shader_name.to_string()),
            category: TimingCategory::Shader,
            start: Instant::now(),
        }
    }

    /// Create a new timing scope for buffer transfers.
    pub fn buffer_transfer(collector: &'a TimingCollector) -> Self {
        Self {
            collector,
            shader_name: None,
            category: TimingCategory::BufferTransfer,
            start: Instant::now(),
        }
    }

    /// Create a new timing scope for sample building.
    pub fn sample_building(collector: &'a TimingCollector) -> Self {
        Self {
            collector,
            shader_name: None,
            category: TimingCategory::SampleBuilding,
            start: Instant::now(),
        }
    }

    /// Create a new timing scope for result processing.
    pub fn result_processing(collector: &'a TimingCollector) -> Self {
        Self {
            collector,
            shader_name: None,
            category: TimingCategory::ResultProcessing,
            start: Instant::now(),
        }
    }
}

impl Drop for TimingScope<'_> {
    fn drop(&mut self) {
        if !self.collector.is_enabled() {
            return;
        }
        let duration_ns = self.start.elapsed().as_nanos() as u64;
        match self.category {
            TimingCategory::Shader => {
                if let Some(name) = &self.shader_name {
                    self.collector.record_shader(name, duration_ns);
                }
            }
            TimingCategory::BufferTransfer => {
                self.collector.record_buffer_transfer(duration_ns);
            }
            TimingCategory::SampleBuilding => {
                self.collector.record_sample_building(duration_ns);
            }
            TimingCategory::ResultProcessing => {
                self.collector.record_result_processing(duration_ns);
            }
        }
    }
}
