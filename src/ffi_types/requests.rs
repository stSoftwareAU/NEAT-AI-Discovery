//! FFI request structs (input from NEAT-AI).
//!
//! All `*Input` types used at the FFI boundary for deserialising JSON requests.

use serde::{Deserialize, Deserializer};

use crate::analysis;
use crate::analysis::task_descriptor::TaskDescriptor;

use super::{CreatureJson, TrainingRecord};

/// Permissive deserialiser for the optional `task_descriptor` FFI field
/// (Issue #1402).
///
/// Issue #1314 plumbed the field through with a strict derive: an **absent**
/// field defaults to `None`, but a **present-but-malformed** field (wrong
/// enum casing, unexpected shape, type mismatch) failed the *entire* FFI
/// request to deserialise. That surfaced in production as a
/// "Failed to parse input JSON: unknown variant `unbounded`" error when
/// NEAT-AI #2785 forwarded the internal TypeScript descriptor shape instead
/// of the documented `PascalCase` wire shape, silently disabling Rust
/// synapse/neuron analysis.
///
/// This adapter buffers the field into a [`serde_json::Value`] and then
/// attempts the strict conversion. On any error it logs once and falls back
/// to [`TaskDescriptor::neutral`] — "we know nothing; don't gate on this" —
/// rather than failing the whole call. Valid wire payloads still round-trip
/// verbatim.
fn deserialize_permissive_task_descriptor<'de, D>(
    deserializer: D,
) -> Result<Option<TaskDescriptor>, D::Error>
where
    D: Deserializer<'de>,
{
    // Buffer the field so a malformed subtree cannot abort the parent parse.
    let value = serde_json::Value::deserialize(deserializer)?;
    if value.is_null() {
        return Ok(None);
    }
    match serde_json::from_value::<TaskDescriptor>(value.clone()) {
        Ok(descriptor) => Ok(Some(descriptor)),
        Err(error) => {
            warn_malformed_task_descriptor_once(&value, &error);
            Ok(Some(TaskDescriptor::neutral()))
        }
    }
}

/// Emit a single `warn!` for a malformed `task_descriptor`, regardless of how
/// many requests carry the same bad shape. Repeated producer mistakes would
/// otherwise flood the logs on every discovery call.
fn warn_malformed_task_descriptor_once(value: &serde_json::Value, error: &serde_json::Error) {
    use std::sync::Once;
    static WARN_ONCE: Once = Once::new();
    WARN_ONCE.call_once(|| {
        tracing::warn!(
            malformed_task_descriptor = %value,
            error = %error,
            "task_descriptor failed strict parse; falling back to TaskDescriptor::neutral() \
             (Issue #1402). Producer should send the PascalCase wire shape.",
        );
    });
}

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
    /// Optional task-shape descriptor forwarded by the producer (Issue #1314).
    ///
    /// Pure plumbing for now — no recommendation generator reads this yet.
    /// When absent, consumers should treat it as
    /// [`TaskDescriptor::neutral`]. A present-but-malformed descriptor falls
    /// back to neutral rather than failing the whole request (Issue #1402).
    #[serde(default, deserialize_with = "deserialize_permissive_task_descriptor")]
    pub task_descriptor: Option<TaskDescriptor>,
}

/// FFI request payload for
/// [`crate::ffi_internal::analyze_parallel_internal`] — the parquet file,
/// creature, focus neurons, and tuning knobs that govern combined
/// synapse + neuron analysis.
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
    /// Overall wall-clock cap in minutes for total discovery time (Issue #1098).
    ///
    /// When set, the total elapsed time from discovery start (recording +
    /// analysis) is capped to this value. The analysis deadline is clamped
    /// to `min(analysis_deadline, discovery_start + wall_clock_cap)`.
    /// When `None`, no overall cap is enforced (backwards compatible).
    #[serde(default)]
    pub max_discovery_wall_clock_minutes: Option<u64>,
    /// Per-creature failure cache feeding the prediction calibration
    /// correction (Issue #1131).
    ///
    /// Each entry records what the library predicted vs what actually
    /// happened post-apply, keyed by change-type (`add-neurons`,
    /// `add-synapses`, `coordinated-structural`, ...). When supplied, the
    /// calibration pipeline computes a per-change-type correction factor
    /// that multiplies the compiled `*_PREDICTION_CALIBRATION` constants.
    /// When absent or empty, the compiled constants are used unchanged.
    #[serde(default)]
    pub failure_cache: Option<Vec<analysis::scoring::calibration_correction::FailureCacheEntry>>,
    /// Per-creature rolling log of discovery pass outcomes (Issue #1132).
    ///
    /// Chronological booleans where `true` = at least one candidate was
    /// accepted in that pass and `false` = empty response. The library
    /// derives a rolling success rate over the last
    /// [`analysis::discovery_mode::ROLLING_WINDOW`] entries and, when it
    /// falls below the configured threshold, biases the candidate mix
    /// toward lower-risk change types. When absent or empty, the pipeline
    /// runs in [`analysis::discovery_mode::DiscoveryMode::Normal`].
    #[serde(default)]
    pub discovery_outcome_log: Option<analysis::discovery_mode::DiscoveryOutcomeLog>,
    /// Cost-function name in use by NEAT-AI (Issue #1317).
    ///
    /// Forwarded into the implied-target reconstruction guard so
    /// reconstruction-dependent detectors are enabled for linear-residual
    /// costs (`MSE` / `MAE` / `CROSS_ENTROPY`) and skipped for non-linear
    /// ones (`MAPE` / `MSLE` / `HINGE` / `CATEGORICAL_ERROR`). When absent
    /// or unrecognised the guard conservatively skips those detectors.
    /// See `src/analysis/cost_function_hint.rs` and issue #1250 for the
    /// underlying defect.
    #[serde(default)]
    pub cost_name: Option<String>,
    /// Optional task-shape descriptor forwarded by the producer (Issue #1314).
    ///
    /// Pure plumbing for now — no recommendation generator reads this yet.
    /// When absent, consumers should treat it as
    /// [`TaskDescriptor::neutral`]. A present-but-malformed descriptor falls
    /// back to neutral rather than failing the whole request (Issue #1402).
    #[serde(default, deserialize_with = "deserialize_permissive_task_descriptor")]
    pub task_descriptor: Option<TaskDescriptor>,
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
    /// Temperature for exploration-exploitation balance (Issue #1020).
    ///
    /// Default 1.0 preserves existing behaviour. See `AnalyzeParallelInput`
    /// for full documentation.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Per-creature failure cache (Issue #1131). See `AnalyzeParallelInput`.
    #[serde(default)]
    pub failure_cache: Option<Vec<analysis::scoring::calibration_correction::FailureCacheEntry>>,
    /// Per-creature rolling discovery outcome log (Issue #1132, #1204).
    ///
    /// Threaded down from `AnalyzeAllInput` so the target-cooldown adaptive
    /// relaxation (Issue #1204) can shrink the cooldown window during a
    /// drought instead of locking the pipeline out of its own search space.
    /// When absent or empty, cooldown uses the static configured thresholds.
    #[serde(default)]
    pub discovery_outcome_log: Option<analysis::discovery_mode::DiscoveryOutcomeLog>,
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
    /// Per-creature failure cache (Issue #1131). See `AnalyzeParallelInput`.
    #[serde(default)]
    pub failure_cache: Option<Vec<analysis::scoring::calibration_correction::FailureCacheEntry>>,
    /// Per-creature rolling discovery outcome log (Issue #1132, #1204).
    ///
    /// Threaded down from `AnalyzeAllInput` so the target-cooldown adaptive
    /// relaxation (Issue #1204) can shrink the cooldown window during a
    /// drought instead of locking the pipeline out of its own search space.
    /// When absent or empty, cooldown uses the static configured thresholds.
    #[serde(default)]
    pub discovery_outcome_log: Option<analysis::discovery_mode::DiscoveryOutcomeLog>,
    /// Task-shape descriptor derived from `AnalyzeAllInput::cost_name`
    /// (Issue #1319). Threaded down so the neuron post-processing path can
    /// bias per-class allocation under a `OneHot` descriptor. `None` (or a
    /// non-`OneHot` topology) preserves the existing allocation verbatim —
    /// regression guard.
    #[serde(default)]
    pub task_descriptor: Option<TaskDescriptor>,
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
    /// Overall wall-clock cap in minutes for total discovery time (Issue #1098).
    ///
    /// See `AnalyzeParallelInput` for full documentation.
    #[serde(default)]
    pub max_discovery_wall_clock_minutes: Option<u64>,
    /// Per-creature failure cache feeding calibration correction (Issue #1131).
    ///
    /// See `AnalyzeParallelInput` for full documentation.
    #[serde(default)]
    pub failure_cache: Option<Vec<analysis::scoring::calibration_correction::FailureCacheEntry>>,
    /// Per-creature discovery outcome log for adaptive strategy (Issue #1132).
    ///
    /// See `AnalyzeParallelInput` for full documentation.
    #[serde(default)]
    pub discovery_outcome_log: Option<analysis::discovery_mode::DiscoveryOutcomeLog>,
    /// Cost-function name in use by NEAT-AI (Issue #1317).
    ///
    /// See `AnalyzeParallelInput::cost_name` for the full contract.
    #[serde(default)]
    pub cost_name: Option<String>,
}

/// JSON input for the `rank_focus_neurons` FFI function — the parquet file,
/// creature, and focus-selection tuning knobs.
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
    /// Optional task-shape descriptor forwarded by the producer (Issue #1314).
    ///
    /// Pure plumbing for now — no recommendation generator reads this yet.
    /// When absent, consumers should treat it as
    /// [`TaskDescriptor::neutral`]. A present-but-malformed descriptor falls
    /// back to neutral rather than failing the whole request (Issue #1402).
    #[serde(default, deserialize_with = "deserialize_permissive_task_descriptor")]
    pub task_descriptor: Option<TaskDescriptor>,
    /// Shared absolute discovery deadline in milliseconds (Issue #1407).
    ///
    /// When provided, focus selection bills against the **same** deadline as
    /// the subsequent synapse/neuron analysis phase rather than opening a fresh
    /// independent window. Focus ranking aborts at whichever is sooner: this
    /// deadline or the `NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS` wall-clock
    /// budget. Interpreted with the same heuristic as `analysisDeadlineMs` on
    /// `analyzeParallel`: values at or above year-2000-in-ms are absolute
    /// timestamps; smaller values are relative durations. When absent, the
    /// legacy budget-only behaviour applies (backwards compatible).
    #[serde(default)]
    pub analysis_deadline_ms: Option<u64>,
    /// Number of discovery passes since this creature last had a candidate
    /// accepted (Issue #1445). Drives diversity-aware focus selection:
    ///
    /// - Once it meets or exceeds the drought threshold
    ///   (`NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD`, #1202) focus selection
    ///   switches from weighted ranking to **round-robin rotation** across the
    ///   top `K × N` ranked neurons so a plateaued creature stops revisiting the
    ///   same dominant neuron every pass.
    /// - It also seeds the rotation cursor, so successive passes pick fresh
    ///   targets.
    ///
    /// When absent (or below the threshold) the diversity-floor path applies
    /// instead. Backwards compatible: omitting it preserves the legacy
    /// non-drought selection behaviour.
    #[serde(default)]
    pub epochs_since_last_accepted_candidate: Option<u64>,
    /// Final focus-set size `N` — the number of neurons the caller will
    /// actually analyse this pass (NEAT-AI's `discoveryMaxNeurons`, default 6).
    /// Issue #1445: diversity-aware focus selection picks `N` diverse targets
    /// from the ranked pool (`max_results` neurons). For drought rotation to
    /// draw from unexplored targets, pass `max_results >= K × N`
    /// (K = [`crate::focus::DROUGHT_ROTATION_POOL_FACTOR`]). When absent,
    /// defaults to [`DEFAULT_FOCUS_SET_SIZE`].
    #[serde(default)]
    pub focus_set_size: Option<usize>,
}

/// Default final focus-set size `N` when `focusSetSize` is not supplied
/// (Issue #1445). Matches NEAT-AI's `discoveryMaxNeurons` default.
pub const DEFAULT_FOCUS_SET_SIZE: usize = 6;

/// JSON input for the `merge_discovery_parquet` FFI function — the input
/// Parquet files to merge and the destination output file.
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
