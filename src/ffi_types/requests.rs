//! FFI request structs (input from NEAT-AI).
//!
//! All `*Input` types used at the FFI boundary for deserialising JSON requests.

use serde::Deserialize;

use crate::analysis;

use super::{CreatureJson, TrainingRecord};

/// JSON input for `record_discovery` function
#[derive(Debug, Deserialize, Clone)]
pub struct RecordDiscoveryInput {
    pub creature: CreatureJson,
    pub training_data: Vec<TrainingRecord>,
    pub temp_dir: String,
    #[serde(default)]
    pub binary_file_path: Option<String>,
    #[serde(default)]
    pub record_indices: Option<Vec<usize>>,
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeParallelInput {
    pub parquet_file: String,
    pub creature: CreatureJson,
    pub focus_neurons: Vec<String>,
    #[serde(default)]
    pub max_synapse_candidates: Option<usize>,
    #[serde(default)]
    pub max_neuron_candidates: Option<usize>,
    #[serde(default)]
    pub analysis_deadline_ms: Option<u64>,
    /// Optional RNG seed to make analysis ordering reproducible.
    ///
    /// When `None`, the library uses non-deterministic randomness. This is
    /// typically desirable for production runs with timeouts, as repeated runs
    /// will explore different candidates over time.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Previous neuron fingerprints from the last discovery run (Issue #490).
    ///
    /// When provided, neurons whose structural fingerprint is unchanged
    /// are skipped, avoiding redundant GPU computation.
    #[serde(default)]
    pub previous_neuron_fingerprints:
        Option<std::collections::HashMap<String, analysis::neuron_fingerprint::NeuronFingerprint>>,
    /// Historical per-module outcome tracker for adaptive weighting (Issue #792).
    ///
    /// When provided, the tracker's success rates influence candidate scoring
    /// and module statistics in the response metadata. Callers should persist
    /// the returned tracker and pass it back on subsequent runs.
    #[serde(default)]
    pub module_outcome_tracker: Option<analysis::module_weights::ModuleOutcomeTracker>,
    /// Temperature for exploration-exploitation balance (Issue #1020).
    ///
    /// Controls the strictness of candidate acceptance. Higher values (> 1.0)
    /// favour exploration (accept more marginal candidates); lower values (< 1.0)
    /// favour exploitation (only accept strong candidates). Default 1.0 preserves
    /// existing behaviour.
    ///
    /// The caller tracks the evolutionary generation count and may use a cooling
    /// schedule to compute this value (e.g., start at 2.0 and decay to 0.5).
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Memory budget in megabytes for the analysis phase (Issue #1028).
    ///
    /// When set, the analysis phase periodically checks Rust-side heap usage
    /// against this budget. If usage reaches 90% of the budget, remaining
    /// analysis is skipped and partial results are returned. When `None`,
    /// no memory limit is enforced (backwards compatible).
    #[serde(default)]
    pub max_analysis_memory_mb: Option<u64>,
}

/// Internal input structure for synapse analysis (used by `analyze_all`)
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeSynapsesInput {
    pub parquet_file: String,
    pub creature: CreatureJson,
    pub focus_neurons: Vec<String>,
    #[serde(default)]
    pub max_candidates: Option<usize>,
    #[serde(default)]
    pub analysis_deadline_ms: Option<u64>,
    /// Optional RNG seed to make analysis ordering reproducible.
    ///
    /// If not provided, the library uses non-deterministic randomness. This is
    /// typically desirable for production runs with timeouts, as repeated runs
    /// will explore different candidates over time.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Historical per-module outcome tracker for add-synapse gating (Issue #1057).
    ///
    /// When provided with sufficient history, the tracker's success rate for
    /// add-synapse candidates is checked. If the rate falls below the
    /// configurable threshold, add-synapse generation is skipped entirely.
    #[serde(default)]
    pub module_outcome_tracker: Option<analysis::module_weights::ModuleOutcomeTracker>,
    /// Temperature for exploration-exploitation balance (Issue #1020).
    ///
    /// Default 1.0 preserves existing behaviour. See `AnalyzeParallelInput`
    /// for full documentation.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

/// Internal input structure for neuron analysis (used by `analyze_all`)
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeNeuronsInput {
    pub parquet_file: String,
    pub creature: CreatureJson,
    pub focus_neurons: Vec<String>,
    #[serde(default)]
    pub max_candidates: Option<usize>,
    #[serde(default)]
    pub analysis_deadline_ms: Option<u64>,
    /// Optional RNG seed to make analysis ordering reproducible.
    ///
    /// If not provided, the library uses non-deterministic randomness. This is
    /// typically desirable for production runs with timeouts, as repeated runs
    /// will explore different candidates over time.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Temperature for exploration-exploitation balance (Issue #1020).
    ///
    /// Default 1.0 preserves existing behaviour. See `AnalyzeParallelInput`
    /// for full documentation.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

/// Internal input structure for combined analysis (used by `analyze_parallel`)
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeAllInput {
    pub parquet_file: String,
    pub creature: CreatureJson,
    pub focus_neurons: Vec<String>,
    #[serde(default)]
    pub max_synapse_candidates: Option<usize>,
    #[serde(default)]
    pub max_neuron_candidates: Option<usize>,
    #[serde(default)]
    pub analysis_deadline_ms: Option<u64>,
    #[serde(default)]
    pub include_synapse_analysis: Option<bool>,
    #[serde(default)]
    pub include_neuron_analysis: Option<bool>,
    /// Optional RNG seed to make analysis ordering reproducible.
    ///
    /// If not provided, the library uses non-deterministic randomness.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Previous neuron fingerprints from the last discovery run (Issue #490).
    ///
    /// When provided, neurons whose structural fingerprint is unchanged
    /// are skipped, avoiding redundant GPU computation.
    #[serde(default)]
    pub previous_neuron_fingerprints:
        Option<std::collections::HashMap<String, analysis::neuron_fingerprint::NeuronFingerprint>>,
    /// Historical per-module outcome tracker for adaptive weighting (Issue #792).
    ///
    /// When provided, the tracker's success rates influence candidate scoring
    /// and module statistics in the response metadata.
    #[serde(default)]
    pub module_outcome_tracker: Option<analysis::module_weights::ModuleOutcomeTracker>,
    /// Temperature for exploration-exploitation balance (Issue #1020).
    ///
    /// Default 1.0 preserves existing behaviour. See `AnalyzeParallelInput`
    /// for full documentation.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Memory budget in megabytes for the analysis phase (Issue #1028).
    ///
    /// See `AnalyzeParallelInput` for full documentation.
    #[serde(default)]
    pub max_analysis_memory_mb: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RankFocusNeuronsInput {
    pub parquet_file: String,
    pub creature: CreatureJson,
    #[serde(default)]
    pub max_results: Option<usize>,
    /// The cost of growth from NEAT-AI (default: 1e-7).
    /// Neurons with `activation_weighted_impact` below this threshold are
    /// candidates for removal. The default 1e-7 matches NEAT-AI's Score.ts formula.
    /// Lower values (e.g., 1e-9) encourage creature expansion for evolution.
    /// Issue #132: Pass this from NEAT-AI's configured costOfGrowth for consistency.
    #[serde(default)]
    pub cost_of_growth: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MergeParquetInput {
    pub output_file: String,
    pub input_files: Vec<String>,
}

/// JSON input for `export_visualisation_snapshot` function
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportVisualisationSnapshotInput {
    pub parquet_file: String,
    pub creature: CreatureJson,
    pub out_file: String,
    /// Include per-synapse contribution series (default: true)
    #[serde(default = "default_true")]
    pub include_per_synapse_series: bool,
    /// Include reconstruction checks for debugging (default: true)
    #[serde(default = "default_true")]
    pub include_reconstruction_checks: bool,
    /// Maximum number of obsIndex to include (default: all)
    #[serde(default)]
    pub max_obs: Option<u32>,
    /// Top-K worst reconstruction samples to include (default: 20)
    #[serde(default = "default_top_k")]
    pub top_k_worst_samples: Option<usize>,
}

fn default_temperature() -> f32 {
    crate::analysis::constants::DEFAULT_TEMPERATURE
}

fn default_true() -> bool {
    true
}

fn default_top_k() -> Option<usize> {
    Some(20)
}

/// JSON input for `read_discovery_records` function
#[derive(Debug, Deserialize)]
pub struct ReadDiscoveryInput {
    pub parquet_file: String,
    /// Neuron identity string to filter by. Must be a stable UUID or descriptive
    /// identifier — exact string matching is used. Numeric integer IDs are not
    /// permitted (Issue #952).
    pub neuron_uuid: String,
}

/// JSON input for `get_calibration_summary` function (Issue #605).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationSummaryInput {
    /// Serialised `DiscoveryHistory` JSON string containing calibration data.
    pub discovery_history: String,
}
