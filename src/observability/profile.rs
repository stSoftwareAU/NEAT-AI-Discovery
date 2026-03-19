//! Structured profile data for JSON output.
//!
//! [`ProfileData`] collects timing, GPU, memory, and analysis metrics into a
//! structure that can be serialised to JSON when
//! `NEAT_AI_DISCOVERY_PROFILE=json` is set.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::time::Instant;

use super::gpu_metrics::GpuMetrics;
use super::{ProfileMode, profile_mode};

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
    /// Create a new `ProfileData` instance.
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

    /// Copy metrics from a `GpuMetrics` instance.
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
