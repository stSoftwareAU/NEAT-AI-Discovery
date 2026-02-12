//! NEAT-AI Discovery Library
//!
//! High-performance Rust library for recording neuron activations and errors
//! during the discovery training phase, then scanning recorded data to identify
//! beneficial new synapses/neurons that would reduce error.

pub mod activations;
pub mod analysis;
pub mod debug;
pub mod discovery_history;
pub mod export;
pub mod focus;
pub mod intern;
pub mod observability;
pub mod parquet_format;
pub mod record;
pub mod streaming;
pub mod types;
mod watchdog;

use anyhow::Result;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};

// Library version from Cargo.toml
const LIB_VERSION: &str = env!("CARGO_PKG_VERSION");

// Static flag to ensure version is logged only once
static VERSION_LOGGED: OnceCell<()> = OnceCell::new();

/// Log library version on first initialization
/// This shows the ACTUAL compiled version embedded in the binary at build time
fn log_version_once() {
    VERSION_LOGGED.get_or_init(|| {
        eprintln!("[NEAT-AI-Discovery] Library version {LIB_VERSION} initialized (compiled version embedded in binary)");
        // Initialise debug handlers (deadlock detection + kill -3 thread dump)
        debug::init_debug_handlers();
    });
}

/// JSON input for record_discovery function
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

/// JSON representation of Creature
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CreatureJson {
    pub neurons: Vec<NeuronJson>,
    pub synapses: Vec<SynapseJson>,
    pub input: usize,
    pub output: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NeuronJson {
    pub uuid: String,
    #[serde(rename = "type")]
    pub neuron_type: String,
    /// Activation function. Defaults to "IDENTITY" for constant neurons.
    #[serde(default = "default_squash")]
    pub squash: String,
    #[serde(default)]
    pub bias: f32,
}

fn default_squash() -> String {
    "IDENTITY".to_string()
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SynapseJson {
    #[serde(default, alias = "fromUUID")]
    pub from_uuid: String,
    #[serde(default, alias = "toUUID")]
    pub to_uuid: String,
    #[serde(default)]
    pub weight: f32,
    /// Synapse type for IF neurons: "condition", "positive", or "negative".
    /// Used to determine which synapses contribute to condition evaluation
    /// versus positive/negative branches. None for non-IF neurons.
    #[serde(default, rename = "type")]
    pub synapse_type: Option<String>,
}

/// Pre-computed neuron data for a single neuron
#[derive(Debug, Deserialize, Clone)]
pub struct NeuronData {
    pub neuron_uuid: String,
    pub activation: f32,
    #[serde(default)]
    pub value: Option<f32>,
    pub errors: Vec<f32>,
}

/// Training data record
#[derive(Debug, Deserialize, Clone)]
pub struct TrainingRecord {
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    #[serde(default)]
    pub neuron_data: Option<Vec<NeuronData>>,
}

/// JSON output from record_discovery function
#[derive(Debug, Serialize)]
pub struct RecordDiscoveryOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ============================================================================
// Streaming Recording API Types
// ============================================================================
// These support incremental recording to avoid JavaScript "Invalid string length"
// errors when serialising large datasets.

/// JSON input for start_discovery_session function
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartSessionInput {
    pub creature: CreatureJson,
    pub temp_dir: String,
}

/// JSON output from start_discovery_session function
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartSessionOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A single observation to append to a streaming session
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamingObservation {
    pub obs_index: u32,
    pub neuron_data: Vec<NeuronData>,
    pub inputs: Vec<f32>,
}

/// JSON input for append_discovery_records function
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppendRecordsInput {
    pub session_id: String,
    pub observations: Vec<StreamingObservation>,
}

/// JSON output from append_discovery_records function
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppendRecordsOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub records_written: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// JSON input for finish_discovery_session function
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinishSessionInput {
    pub session_id: String,
}

/// JSON output from finish_discovery_session function
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FinishSessionOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_records: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// JSON input for cancel_discovery_session function
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelSessionInput {
    pub session_id: String,
}

/// JSON output from cancel_discovery_session function
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelSessionOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CandidateSynapseJson {
    pub from_neuron_uuid: String,
    pub to_neuron_uuid: String,
    /// Index of `from_neuron_uuid` in the creature's forward-only evaluation order.
    ///
    /// This includes input neurons (`input-0..`) followed by `creature.neurons[]` in order.
    /// Provided for debugging and ranking analysis (eg why candidates cluster near the end).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_neuron_index: Option<usize>,
    /// Index of `to_neuron_uuid` in the creature's forward-only evaluation order.
    ///
    /// This includes input neurons (`input-0..`) followed by `creature.neurons[]` in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_neuron_index: Option<usize>,
    pub weight: f32,
    /// Impact of the target neuron on the creature's output (0.0 to 1.0).
    /// Output neurons have impact = 1.0, hidden neurons have discounted impact.
    pub target_neuron_impact: f32,
    /// Expected reduction in creature's error from adding this synapse.
    /// Formula: neuron_error_reduction × target_neuron_impact
    pub expected_creature_error_reduction: f32,
    /// Expected improvement in creature's score from adding this synapse.
    /// Since score = 1 - error, this equals expected_creature_error_reduction.
    pub expected_creature_score_gain: f32,
    pub improved_count: u32,
    pub total_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_neuron_stats: Option<NeuronStatsJson>,
    /// Information about how this candidate affects outlier samples (Issue #192).
    ///
    /// Only populated when outlier analysis is enabled via
    /// `NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS=1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outlier_reduction_info: Option<analysis::OutlierReductionInfo>,
    /// Overall confidence score for this prediction (Issue #194).
    ///
    /// A value between 0.0 and 1.0 indicating how reliable the prediction is.
    /// Higher values mean more reliable predictions. Computed from:
    /// - Sample size (more samples = higher confidence)
    /// - Source variance (higher variance = more reliable correlation)
    /// - Model fit (better fit = higher confidence)
    pub prediction_confidence: f32,
    /// 95% confidence interval for expectedCreatureScoreGain (Issue #194).
    ///
    /// The first element is the lower bound, the second is the upper bound.
    /// The point estimate (expectedCreatureScoreGain) should fall within this interval.
    pub expected_score_gain_confidence_interval: [f32; 2],
    /// Human-readable label identifying weight variants (Issue #513).
    ///
    /// When a synapse candidate is paired with conservative/gentle-nudge/micro-nudge
    /// variants, each variant gets a descriptive comment. The original candidate's
    /// comment lists which variants were included.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

/// Candidate to update the weight of an existing synapse (delta-based).
///
/// This represents a *weight adjustment* (not a new connection). The prediction logic treats
/// `delta_weight` as an additive correction to the existing synapse weight.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SynapseWeightUpdateCandidateJson {
    pub from_neuron_uuid: String,
    pub to_neuron_uuid: String,
    /// Index of `from_neuron_uuid` in the creature's forward-only evaluation order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_neuron_index: Option<usize>,
    /// Index of `to_neuron_uuid` in the creature's forward-only evaluation order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_neuron_index: Option<usize>,
    pub old_weight: f32,
    pub new_weight: f32,
    /// The proposed additive change: `new_weight - old_weight`.
    pub delta_weight: f32,
    /// Impact of the target neuron on the creature's output (0.0 to 1.0).
    pub target_neuron_impact: f32,
    /// Expected reduction in creature error from applying `delta_weight`.
    pub expected_creature_error_reduction: f32,
    /// Expected improvement in creature score from applying `delta_weight`.
    pub expected_creature_score_gain: f32,
    pub improved_count: u32,
    pub total_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_neuron_stats: Option<NeuronStatsJson>,
}

/// A single atomic operation inside a coordinated (grouped) candidate.
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum CoordinatedStructuralOpJson {
    RemoveSynapse {
        #[serde(rename = "fromNeuronUuid")]
        from_neuron_uuid: String,
        #[serde(rename = "toNeuronUuid")]
        to_neuron_uuid: String,
    },
    AddSynapse {
        #[serde(rename = "fromNeuronUuid")]
        from_neuron_uuid: String,
        #[serde(rename = "toNeuronUuid")]
        to_neuron_uuid: String,
        weight: f32,
    },
    /// Add a neuron as part of a coordinated structural candidate.
    ///
    /// Notes (7-Jan-2026):
    /// - `neuronUuid` is emitted by Rust and must be deterministic so coordinated candidates are replayable.
    /// - For forward-only creatures, `insertBeforeNeuronUuid` provides a placement hint so subsequent
    ///   `addSynapse(newNeuron -> target)` can satisfy the forward-only ordering constraints.
    AddNeuron {
        #[serde(rename = "neuronUuid")]
        neuron_uuid: String,
        #[serde(rename = "neuronType")]
        neuron_type: String,
        squash: String,
        bias: f32,
        #[serde(
            rename = "insertBeforeNeuronUuid",
            skip_serializing_if = "Option::is_none"
        )]
        insert_before_neuron_uuid: Option<String>,
    },
    /// Remove a neuron and any attached synapses.
    RemoveNeuron {
        #[serde(rename = "neuronUuid")]
        neuron_uuid: String,
    },
    /// Change a neuron's squash/activation function.
    ChangeSquash {
        #[serde(rename = "neuronUuid")]
        neuron_uuid: String,
        squash: String,
    },
    /// Set a neuron's bias.
    SetBias {
        #[serde(rename = "neuronUuid")]
        neuron_uuid: String,
        bias: f32,
    },
    /// Set an existing synapse's weight (Issue #180).
    ///
    /// This replaces the previous `removeSynapse` + `addSynapse` pattern for weight adjustments,
    /// providing a simpler and more direct representation of the intended change.
    SetWeight {
        #[serde(rename = "fromNeuronUuid")]
        from_neuron_uuid: String,
        #[serde(rename = "toNeuronUuid")]
        to_neuron_uuid: String,
        weight: f32,
    },
}

/// A grouped candidate that must be applied as a single unit.
///
/// This supports "Coordinated Structural Discovery" (Issue #165): beneficial changes that are
/// epistatic (no single edit improves fitness in isolation).
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CoordinatedStructuralCandidateJson {
    pub operations: Vec<CoordinatedStructuralOpJson>,
    pub expected_creature_score_gain: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

#[derive(Debug, Serialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct NeuronStatsJson {
    pub mean_error: f32,
    pub error_variance: f32,
    pub mean_activation: f32,
    pub activation_variance: f32,
    pub error_spike_count: u32,
    pub activation_spike_count: u32,
    pub activation_min: f32,
    pub activation_max: f32,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CandidateNeuronJson {
    pub source_neuron_uuid: String,
    pub target_neuron_uuid: String,
    /// Index of `source_neuron_uuid` in the creature's forward-only evaluation order.
    ///
    /// This includes input neurons (`input-0..`) followed by `creature.neurons[]` in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_neuron_index: Option<usize>,
    /// Index of `target_neuron_uuid` in the creature's forward-only evaluation order.
    ///
    /// This includes input neurons (`input-0..`) followed by `creature.neurons[]` in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_neuron_index: Option<usize>,
    pub incoming_weight: f32,
    pub outgoing_weight: f32,
    pub squash: String,
    pub bias: f32,
    /// Optional human-readable comment for diagnostics and production experiments.
    ///
    /// This is intentionally optional to maintain backwards compatibility with older
    /// consumers that don't expect the field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
    /// Impact of the target neuron on the creature's output (0.0 to 1.0).
    /// Output neurons have impact = 1.0, hidden neurons have discounted impact.
    pub target_neuron_impact: f32,
    /// Expected reduction in creature's error from adding this neuron.
    /// Formula: neuron_error_reduction × target_neuron_impact
    pub expected_creature_error_reduction: f32,
    /// Expected improvement in creature's score from adding this neuron.
    /// Since score = 1 - error, this equals expected_creature_error_reduction.
    pub expected_creature_score_gain: f32,
    pub improved_count: u32,
    pub total_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_neuron_stats: Option<NeuronStatsJson>,
    /// Overall confidence score for this prediction (Issue #194).
    ///
    /// A value between 0.0 and 1.0 indicating how reliable the prediction is.
    /// Higher values mean more reliable predictions. Computed from:
    /// - Sample size (more samples = higher confidence)
    /// - Source variance (higher variance = more reliable correlation)
    /// - Model fit (better fit = higher confidence)
    pub prediction_confidence: f32,
    /// 95% confidence interval for expectedCreatureScoreGain (Issue #194).
    ///
    /// The first element is the lower bound, the second is the upper bound.
    /// The point estimate (expectedCreatureScoreGain) should fall within this interval.
    pub expected_score_gain_confidence_interval: [f32; 2],
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
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeParallelOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub helpful_synapses: Option<Vec<CandidateSynapseJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harmful_synapses: Option<Vec<CandidateSynapseJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synapse_diagnostics: Option<Vec<SynapseDiagnosticJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synapse_gpu_used: Option<bool>,
    /// Synapse analysis metadata for observability (v0.2.17+).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synapse_metadata: Option<SynapseAnalysisMetadataJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub helpful_neurons: Option<Vec<CandidateNeuronJson>>,
    /// Candidates to update weights on existing synapses (v0.2.18+).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synapse_weight_updates: Option<Vec<SynapseWeightUpdateCandidateJson>>,
    /// Coordinated (grouped) structural candidates (v0.2.18+).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinated_structural_candidates: Option<Vec<CoordinatedStructuralCandidateJson>>,
    /// Candidate clusters for redundancy reduction (Issue #224).
    ///
    /// Groups similar candidates by target neuron so the controller can test a
    /// representative first and skip redundant ablation tests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_clusters: Option<Vec<analysis::candidate_clustering::CandidateClusterJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_diagnostics: Option<Vec<NeuronDiagnosticJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_gpu_used: Option<bool>,
    /// Neuron analysis metadata for observability (v0.2.17+).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_metadata: Option<NeuronAnalysisMetadataJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// Coordinated structural candidates are now produced inside synapse analysis and surfaced via
// `AnalyzeSynapsesResult.coordinated_structural_candidates` (Issue #165).

/// JSON representation of synapse analysis metadata.
///
/// Surfaces diagnostic information to help callers understand:
/// - Whether prediction accuracy is degraded (missing `value` data)
/// - Whether candidates were truncated by `maxCandidates`
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SynapseAnalysisMetadataJson {
    /// Whether `targetValue` (pre-activation) was available in the recorded data.
    pub target_value_available: bool,
    /// Whether saturation-aware simulation was used for at least one candidate.
    pub saturation_aware_simulation_used: bool,
    /// Total number of synapse candidates found during analysis (before truncation).
    pub candidates_found: usize,
    /// Number of synapse candidates returned to caller (after `maxCandidates` truncation).
    pub candidates_returned: usize,
    /// True when analysis hit its deadline and returned partial results.
    pub timed_out: bool,
    /// Number of focus neurons completed before returning.
    pub completed_focus_neurons: usize,
    /// Total focus neurons requested for this analysis invocation.
    pub total_focus_neurons: usize,
    /// Minimum input index observed with non-empty records (eg 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_index_min_seen_with_records: Option<usize>,
    /// Maximum input index observed with non-empty records (eg 1555).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_index_max_seen_with_records: Option<usize>,
    /// GPU timing data for performance diagnostics (Issue #195).
    /// Only present when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<AnalysisTimingJson>,
    /// Information about the GPU adapter used (Issue #228).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_info: Option<GpuAdapterInfoJson>,
    /// Per-discovery-module statistics for the current run (Issue #485).
    /// Reports how many candidates each module produced and historical success rates.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub discovery_module_stats: Vec<analysis::module_weights::DiscoveryModuleStatsJson>,
}

/// JSON representation of neuron analysis metadata.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuronAnalysisMetadataJson {
    /// Total number of neuron candidates found during analysis (before truncation).
    pub candidates_found: usize,
    /// Number of neuron candidates returned to caller (after `maxCandidates` truncation).
    pub candidates_returned: usize,
    /// True when analysis hit its deadline and returned partial results.
    pub timed_out: bool,
    /// Number of focus neurons completed before returning.
    pub completed_focus_neurons: usize,
    /// Total focus neurons requested for this analysis invocation.
    pub total_focus_neurons: usize,
    /// GPU timing data for performance diagnostics (Issue #195).
    /// Only present when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<AnalysisTimingJson>,
    /// Information about the GPU adapter used (Issue #228).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_info: Option<GpuAdapterInfoJson>,
}

// =============================================================================
// GPU Info JSON Types (Issue #228)
// =============================================================================

/// JSON representation of GPU adapter information.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GpuAdapterInfoJson {
    /// Human-readable name of the GPU (e.g., "Apple M4 Pro").
    pub name: String,
    /// Whether the GPU has unified memory architecture.
    pub unified_memory: bool,
    /// Whether zero-copy buffer sharing is currently enabled.
    pub zero_copy_enabled: bool,
}

// =============================================================================
// GPU Timing JSON Types (Issue #195)
// =============================================================================

/// JSON representation of per-shader timing statistics.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ShaderTimingJson {
    /// Number of times this shader was executed.
    pub calls: u32,
    /// Total execution time in milliseconds.
    pub total_ms: f64,
    /// Average execution time per call in milliseconds.
    pub avg_ms: f64,
}

/// JSON representation of GPU-side timing breakdown.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GpuTimingBreakdownJson {
    /// Total time spent in shader execution (all shaders combined) in milliseconds.
    pub shader_execution_ms: f64,
    /// Total time spent in buffer mapping/transfers in milliseconds.
    pub buffer_transfer_ms: f64,
    /// Per-shader timing statistics.
    /// Keys are shader names: "helpful", "harmful", "relu", "activation", "bias"
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub shader_timings: std::collections::HashMap<String, ShaderTimingJson>,
}

/// JSON representation of CPU-side timing breakdown.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CpuTimingBreakdownJson {
    /// Time spent building samples for GPU evaluation in milliseconds.
    pub sample_building_ms: f64,
    /// Time spent processing results from GPU in milliseconds.
    pub result_processing_ms: f64,
}

/// JSON representation of complete timing data for an analysis run.
///
/// Only populated when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisTimingJson {
    /// Total wall-clock time for the analysis in milliseconds.
    pub total_analysis_ms: f64,
    /// GPU-side timing breakdown.
    pub gpu: GpuTimingBreakdownJson,
    /// CPU-side timing breakdown.
    pub cpu: CpuTimingBreakdownJson,
}

/// Internal input structure for synapse analysis (used by analyze_all)
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
}

/// Internal input structure for neuron analysis (used by analyze_all)
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
}

/// Internal input structure for combined analysis (used by analyze_parallel)
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
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckGpuOutput {
    pub success: bool,
    pub gpu_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetVersionOutput {
    pub success: bool,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RankFocusNeuronsInput {
    pub parquet_file: String,
    pub creature: CreatureJson,
    #[serde(default)]
    pub max_results: Option<usize>,
    /// The cost of growth from NEAT-AI (default: 1e-7).
    /// Neurons with activation_weighted_impact below this threshold are
    /// candidates for removal. The default 1e-7 matches NEAT-AI's Score.ts formula.
    /// Lower values (e.g., 1e-9) encourage creature expansion for evolution.
    /// Issue #132: Pass this from NEAT-AI's configured costOfGrowth for consistency.
    #[serde(default)]
    pub cost_of_growth: Option<f32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RankedNeuronJson {
    pub neuron_uuid: String,
    pub total_error: f32,
    /// Structural impact based on weight paths to output
    pub impact: f32,
    /// Mean absolute activation value from recorded samples
    pub mean_activation: f32,
    /// Activation-weighted impact = structural_impact × mean_activation
    /// This reflects the actual contribution the neuron makes during inference
    pub activation_weighted_impact: f32,
}

/// A neuron with activation-weighted impact below costOfGrowth threshold - candidate for removal.
/// Removing such neurons improves score because complexity reduction outweighs contribution.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemovalCandidateJson {
    pub neuron_uuid: String,
    pub total_error: f32,
    /// Structural impact based on weight paths to output
    pub impact: f32,
    /// Mean absolute activation value from recorded samples
    pub mean_activation: f32,
    /// Activation-weighted impact = structural_impact × mean_activation
    /// This reflects the actual contribution the neuron makes during inference
    pub activation_weighted_impact: f32,
    /// Number of synapses pointing TO this neuron
    pub incoming_synapses: usize,
    /// Number of synapses pointing FROM this neuron
    pub outgoing_synapses: usize,
    /// The complexity savings from removing this neuron (based on NEAT-AI Score.ts formula)
    pub removal_savings: f32,
    /// Expected creature-level error reduction from removing this neuron.
    /// Issue #117: This is based on activation_weighted_impact, NOT total_error.
    /// For low-impact removal candidates, this will be very small (as it should be).
    pub expected_error_reduction: f32,
    /// Explains why removal improves score
    pub reason: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RankFocusNeuronsOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neurons: Option<Vec<RankedNeuronJson>>,
    /// Neurons with high error but very low impact - candidates for removal
    #[serde(skip_serializing_if = "Option::is_none")]
    pub removal_candidates: Option<Vec<RemovalCandidateJson>>,
    /// Issue #306: Coordinated structural candidates for removing constant-value neurons.
    /// When a hidden neuron has near-zero activation variance (constant output), it can be
    /// removed and its effect folded into bias adjustments for downstream neurons.
    /// Each candidate contains:
    /// - A RemoveNeuron operation for the constant neuron
    /// - SetBias operations for all downstream neurons with adjusted biases
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constant_neuron_removals: Option<Vec<CoordinatedStructuralCandidateJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_error: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processed_neurons: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_neurons: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SynapseDiagnosticJson {
    pub target_neuron_uuid: String,
    pub reason: SynapseDiagnosticReasonJson,
    pub evaluated_candidates: u32,
    pub candidates_with_samples: u32,
    pub target_record_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<SynapseDiagnosticDetailJson>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SynapseDiagnosticReasonJson {
    NoEligibleSources,
    NoDiagnostics,
    NoSamples,
    ZeroImprovement,
    BelowThreshold,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SynapseDiagnosticDetailJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_neuron_uuid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_record_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub improved_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worsened_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_creature_error_reduction: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_weight: Option<f32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuronDiagnosticJson {
    pub target_neuron_uuid: String,
    pub reason: NeuronDiagnosticReasonJson,
    pub evaluated_sources: u32,
    pub sources_with_samples: u32,
    pub target_record_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<NeuronDiagnosticDetailJson>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuronDiagnosticReasonJson {
    NoEligibleSources,
    NoDiagnostics,
    NoSamples,
    NotEnoughActivations,
    WeightDegenerate,
    BelowThreshold,
    /// Hidden neurons are filtered out from add-neuron analysis because their
    /// backpropagated errors don't reliably translate to output error reduction.
    HiddenNeuronFiltered,
    /// Input neurons are filtered out from add-neuron analysis because they're
    /// observation sources, not computation nodes - they have no activation function
    /// or error to reduce.
    InputNeuronFiltered,
    /// Constant neurons are filtered out from add-neuron analysis because they
    /// don't receive inputs - they always output a fixed value regardless of
    /// network state, so adding a connection to them has no effect.
    ConstantNeuronFiltered,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuronDiagnosticDetailJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_neuron_uuid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orientation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub improved_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worsened_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_creature_error_reduction: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outgoing_weight: Option<f32>,
}

/// Convert GPU adapter info to JSON representation.
fn gpu_info_to_json(info: &analysis::GpuAdapterInfo) -> GpuAdapterInfoJson {
    GpuAdapterInfoJson {
        name: info.name.clone(),
        unified_memory: info.has_unified_memory,
        zero_copy_enabled: info.zero_copy_enabled,
    }
}

/// Convert internal timing data to JSON representation.
fn timing_to_json(timing: &analysis::AnalysisTiming) -> AnalysisTimingJson {
    AnalysisTimingJson {
        total_analysis_ms: timing.total_analysis_ms,
        gpu: GpuTimingBreakdownJson {
            shader_execution_ms: timing.gpu.shader_execution_ms,
            buffer_transfer_ms: timing.gpu.buffer_transfer_ms,
            shader_timings: timing
                .gpu
                .shader_timings
                .iter()
                .map(|(name, st)| {
                    (
                        name.clone(),
                        ShaderTimingJson {
                            calls: st.calls,
                            total_ms: st.total_ms,
                            avg_ms: st.avg_ms,
                        },
                    )
                })
                .collect(),
        },
        cpu: CpuTimingBreakdownJson {
            sample_building_ms: timing.cpu.sample_building_ms,
            result_processing_ms: timing.cpu.result_processing_ms,
        },
    }
}

fn synapse_diagnostics_json(
    summaries: &[analysis::SynapseNoCandidateSummary],
) -> Option<Vec<SynapseDiagnosticJson>> {
    if summaries.is_empty() {
        return None;
    }
    Some(
        summaries
            .iter()
            .map(|summary| SynapseDiagnosticJson {
                target_neuron_uuid: summary.target_uuid.clone(),
                reason: match summary.reason {
                    analysis::SynapseNoCandidateReason::NoEligibleSources => {
                        SynapseDiagnosticReasonJson::NoEligibleSources
                    }
                    analysis::SynapseNoCandidateReason::NoDiagnostics => {
                        SynapseDiagnosticReasonJson::NoDiagnostics
                    }
                    analysis::SynapseNoCandidateReason::NoSamples => {
                        SynapseDiagnosticReasonJson::NoSamples
                    }
                    analysis::SynapseNoCandidateReason::ZeroImprovement => {
                        SynapseDiagnosticReasonJson::ZeroImprovement
                    }
                    analysis::SynapseNoCandidateReason::BelowThreshold => {
                        SynapseDiagnosticReasonJson::BelowThreshold
                    }
                },
                evaluated_candidates: summary.evaluated_candidates,
                candidates_with_samples: summary.candidates_with_samples,
                target_record_count: summary.target_record_count,
                detail: summary
                    .detail
                    .as_ref()
                    .map(|detail| SynapseDiagnosticDetailJson {
                        source_neuron_uuid: detail.source_uuid.clone(),
                        sample_count: detail.sample_count,
                        source_record_count: detail.source_record_count,
                        improved_count: detail.improved_count,
                        worsened_count: detail.worsened_count,
                        expected_creature_error_reduction: detail.expected_improvement,
                        threshold: detail.threshold,
                        suggested_weight: detail.suggested_weight,
                    }),
            })
            .collect(),
    )
}

fn neuron_diagnostics_json(
    summaries: &[analysis::NeuronNoCandidateSummary],
) -> Option<Vec<NeuronDiagnosticJson>> {
    if summaries.is_empty() {
        return None;
    }
    Some(
        summaries
            .iter()
            .map(|summary| NeuronDiagnosticJson {
                target_neuron_uuid: summary.target_uuid.clone(),
                reason: match summary.reason {
                    analysis::NeuronNoCandidateReason::NoEligibleSources => {
                        NeuronDiagnosticReasonJson::NoEligibleSources
                    }
                    analysis::NeuronNoCandidateReason::NoDiagnostics => {
                        NeuronDiagnosticReasonJson::NoDiagnostics
                    }
                    analysis::NeuronNoCandidateReason::NoSamples => {
                        NeuronDiagnosticReasonJson::NoSamples
                    }
                    analysis::NeuronNoCandidateReason::NotEnoughActivations => {
                        NeuronDiagnosticReasonJson::NotEnoughActivations
                    }
                    analysis::NeuronNoCandidateReason::WeightDegenerate => {
                        NeuronDiagnosticReasonJson::WeightDegenerate
                    }
                    analysis::NeuronNoCandidateReason::BelowThreshold => {
                        NeuronDiagnosticReasonJson::BelowThreshold
                    }
                    analysis::NeuronNoCandidateReason::HiddenNeuronFiltered => {
                        NeuronDiagnosticReasonJson::HiddenNeuronFiltered
                    }
                    analysis::NeuronNoCandidateReason::InputNeuronFiltered => {
                        NeuronDiagnosticReasonJson::InputNeuronFiltered
                    }
                    analysis::NeuronNoCandidateReason::ConstantNeuronFiltered => {
                        NeuronDiagnosticReasonJson::ConstantNeuronFiltered
                    }
                },
                evaluated_sources: summary.evaluated_sources,
                sources_with_samples: summary.sources_with_samples,
                target_record_count: summary.target_record_count,
                detail: summary
                    .detail
                    .as_ref()
                    .map(|detail| NeuronDiagnosticDetailJson {
                        source_neuron_uuid: detail.source_uuid.clone(),
                        orientation: detail.orientation.clone(),
                        sample_count: detail.sample_count,
                        improved_count: detail.improved_count,
                        worsened_count: detail.worsened_count,
                        expected_creature_error_reduction: detail.expected_improvement,
                        threshold: detail.threshold,
                        outgoing_weight: detail.outgoing_weight,
                    }),
            })
            .collect(),
    )
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MergeParquetInput {
    pub output_file: String,
    pub input_files: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MergeParquetOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ============================================================================
// Visualisation Snapshot Export API Types
// ============================================================================
// These support exporting a debug-friendly JSON snapshot for use with the
// NEAT-AI-Explore visualiser. This is an optional debug tool.

/// JSON input for export_visualisation_snapshot function
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

fn default_true() -> bool {
    true
}

fn default_top_k() -> Option<usize> {
    Some(20)
}

/// JSON output from export_visualisation_snapshot function
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportVisualisationSnapshotOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<ExportVisualisationStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Statistics from the export operation
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportVisualisationStats {
    pub obs_count: usize,
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub output_count: usize,
}

/// Main entry point for recording discovery data
///
/// Takes JSON input and returns JSON output for easy integration with TypeScript/DenoJS
/// Always returns a JSON string, even on error (with success=false)
///
/// This is the internal Rust function. For FFI, use `record_discovery`.
pub fn record_discovery_internal(input_json: &str) -> Result<String> {
    // Parse input JSON - if this fails, return JSON error
    let input: RecordDiscoveryInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Process discovery data - if this fails, return JSON error
    let result = match record::record_discovery_data(&input) {
        Ok(result) => result,
        Err(e) => {
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(e.to_string()),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Success case
    let output = RecordDiscoveryOutput {
        success: true,
        temp_dir: Some(result.temp_dir),
        file: Some(result.file),
        error: None,
    };

    Ok(serde_json::to_string(&output)?)
}

pub fn merge_discovery_parquet_internal(input_json: &str) -> Result<String> {
    let input: MergeParquetInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let output = MergeParquetOutput {
                success: false,
                output_file: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    if input.input_files.is_empty() {
        let output = MergeParquetOutput {
            success: false,
            output_file: None,
            error: Some("No discovery parquet files provided for merge".to_string()),
        };
        return Ok(serde_json::to_string(&output)?);
    }

    match parquet_format::merge_parquet_files(&input.output_file, &input.input_files) {
        Ok(()) => {
            let output = MergeParquetOutput {
                success: true,
                output_file: Some(input.output_file),
                error: None,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let output = MergeParquetOutput {
                success: false,
                output_file: None,
                error: Some(e.to_string()),
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
}

pub fn analyze_parallel_internal(input_json: &str) -> Result<String> {
    let input: AnalyzeParallelInput = match serde_json::from_str::<AnalyzeParallelInput>(input_json)
    {
        Ok(value) => value,
        Err(e) => {
            let output = AnalyzeParallelOutput {
                success: false,
                helpful_synapses: None,
                harmful_synapses: None,
                synapse_diagnostics: None,
                synapse_gpu_used: None,
                synapse_metadata: None,
                helpful_neurons: None,
                synapse_weight_updates: None,
                coordinated_structural_candidates: None,
                candidate_clusters: None,
                neuron_diagnostics: None,
                neuron_gpu_used: None,
                neuron_metadata: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    let combined_input = build_analyze_all_input_from_parallel(input);

    match analysis::analyze_all(&combined_input) {
        Ok(result) => {
            let synapse = result.synapse;
            let neuron = result.neuron;
            let synapse_weight_updates = synapse.as_ref().and_then(|s| {
                if s.synapse_weight_updates.is_empty() {
                    None
                } else {
                    Some(s.synapse_weight_updates.clone())
                }
            });

            let coordinated_structural_candidates = synapse.as_ref().and_then(|s| {
                if s.coordinated_structural_candidates.is_empty() {
                    None
                } else {
                    Some(s.coordinated_structural_candidates.clone())
                }
            });

            let candidate_clusters = synapse.as_ref().and_then(|s| {
                if s.candidate_clusters.is_empty() {
                    None
                } else {
                    Some(s.candidate_clusters.clone())
                }
            });

            let output = AnalyzeParallelOutput {
                success: true,
                helpful_synapses: synapse.as_ref().map(|s| s.helpful_synapses.clone()),
                harmful_synapses: synapse.as_ref().map(|s| s.harmful_synapses.clone()),
                synapse_diagnostics: synapse
                    .as_ref()
                    .and_then(|s| synapse_diagnostics_json(s.no_candidate_reasons.as_slice())),
                synapse_gpu_used: synapse.as_ref().map(|s| s.gpu_used),
                synapse_metadata: synapse.as_ref().map(|s| SynapseAnalysisMetadataJson {
                    target_value_available: s.metadata.target_value_available,
                    saturation_aware_simulation_used: s.metadata.saturation_aware_simulation_used,
                    candidates_found: s.metadata.candidates_found,
                    candidates_returned: s.metadata.candidates_returned,
                    timed_out: s.metadata.timed_out,
                    completed_focus_neurons: s.metadata.completed_focus_neurons,
                    total_focus_neurons: s.metadata.total_focus_neurons,
                    input_index_min_seen_with_records: s.metadata.input_index_min_seen_with_records,
                    input_index_max_seen_with_records: s.metadata.input_index_max_seen_with_records,
                    timing: s.metadata.timing.as_ref().map(timing_to_json),
                    gpu_info: s.metadata.gpu_info.as_ref().map(gpu_info_to_json),
                    discovery_module_stats: s.metadata.discovery_module_stats.clone(),
                }),
                helpful_neurons: neuron.as_ref().map(|n| n.helpful_neurons.clone()),
                synapse_weight_updates,
                coordinated_structural_candidates,
                candidate_clusters,
                neuron_diagnostics: neuron
                    .as_ref()
                    .and_then(|n| neuron_diagnostics_json(n.no_candidate_reasons.as_slice())),
                neuron_gpu_used: neuron.as_ref().map(|n| n.gpu_used),
                neuron_metadata: neuron.as_ref().map(|n| NeuronAnalysisMetadataJson {
                    candidates_found: n.metadata.candidates_found,
                    candidates_returned: n.metadata.candidates_returned,
                    timed_out: n.metadata.timed_out,
                    completed_focus_neurons: n.metadata.completed_focus_neurons,
                    total_focus_neurons: n.metadata.total_focus_neurons,
                    timing: n.metadata.timing.as_ref().map(timing_to_json),
                    gpu_info: n.metadata.gpu_info.as_ref().map(gpu_info_to_json),
                }),
                error: None,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let output = AnalyzeParallelOutput {
                success: false,
                helpful_synapses: None,
                harmful_synapses: None,
                synapse_diagnostics: None,
                synapse_gpu_used: None,
                synapse_metadata: None,
                helpful_neurons: None,
                synapse_weight_updates: None,
                coordinated_structural_candidates: None,
                candidate_clusters: None,
                neuron_diagnostics: None,
                neuron_gpu_used: None,
                neuron_metadata: None,
                error: Some(e.to_string()),
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
}

fn build_analyze_all_input_from_parallel(input: AnalyzeParallelInput) -> AnalyzeAllInput {
    AnalyzeAllInput {
        parquet_file: input.parquet_file,
        creature: input.creature,
        focus_neurons: input.focus_neurons,
        max_synapse_candidates: input.max_synapse_candidates,
        max_neuron_candidates: input.max_neuron_candidates,
        analysis_deadline_ms: input.analysis_deadline_ms,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: input.random_seed,
    }
}

pub fn check_gpu_available_internal() -> Result<String> {
    let result = analysis::GpuAnalyzer::check_gpu_availability();

    // On macOS, missing GPU is an error (Metal should always work).
    // On Linux, missing GPU gracefully disables discovery (common on headless servers).
    let output = if result.is_error {
        CheckGpuOutput {
            success: false,
            gpu_available: false,
            reason: result.reason,
            error: Some("GPU required but not available".to_string()),
        }
    } else {
        CheckGpuOutput {
            success: true,
            gpu_available: result.available,
            reason: result.reason,
            error: None,
        }
    };
    Ok(serde_json::to_string(&output)?)
}

pub fn get_library_version_internal() -> Result<String> {
    let output = GetVersionOutput {
        success: true,
        version: LIB_VERSION.to_string(),
        error: None,
    };
    Ok(serde_json::to_string(&output)?)
}

pub fn rank_focus_neurons_internal(input_json: &str) -> Result<String> {
    let input: RankFocusNeuronsInput = match serde_json::from_str(input_json) {
        Ok(value) => value,
        Err(e) => {
            let output = RankFocusNeuronsOutput {
                success: false,
                neurons: None,
                removal_candidates: None,
                constant_neuron_removals: None,
                max_output_error: None,
                processed_neurons: None,
                total_neurons: None,
                duration_ms: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    match focus::rank_focus_neurons(
        &input.parquet_file,
        &input.creature,
        input.max_results,
        input.cost_of_growth,
    ) {
        Ok(stats) => {
            let neurons: Vec<RankedNeuronJson> = stats
                .neurons
                .into_iter()
                .map(|neuron| RankedNeuronJson {
                    neuron_uuid: neuron.neuron_uuid,
                    total_error: neuron.total_error,
                    impact: neuron.impact,
                    mean_activation: neuron.mean_activation,
                    activation_weighted_impact: neuron.activation_weighted_impact,
                })
                .collect();
            let removal_candidates: Vec<RemovalCandidateJson> = stats
                .removal_candidates
                .into_iter()
                .map(|c| RemovalCandidateJson {
                    neuron_uuid: c.neuron_uuid,
                    total_error: c.total_error,
                    impact: c.impact,
                    mean_activation: c.mean_activation,
                    activation_weighted_impact: c.activation_weighted_impact,
                    incoming_synapses: c.incoming_synapses,
                    outgoing_synapses: c.outgoing_synapses,
                    removal_savings: c.removal_savings,
                    expected_error_reduction: c.expected_error_reduction,
                    reason: c.reason,
                })
                .collect();
            let output = RankFocusNeuronsOutput {
                success: true,
                neurons: Some(neurons),
                removal_candidates: if removal_candidates.is_empty() {
                    None
                } else {
                    Some(removal_candidates)
                },
                // Issue #306: Return constant neuron removal candidates
                constant_neuron_removals: if stats.constant_neuron_removals.is_empty() {
                    None
                } else {
                    Some(stats.constant_neuron_removals)
                },
                max_output_error: Some(stats.max_output_error),
                processed_neurons: Some(stats.processed_neurons),
                total_neurons: Some(stats.total_neurons),
                duration_ms: Some(stats.duration_ms.min(u64::MAX as u128) as u64),
                error: None,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let output = RankFocusNeuronsOutput {
                success: false,
                neurons: None,
                removal_candidates: None,
                constant_neuron_removals: None,
                max_output_error: None,
                processed_neurons: None,
                total_neurons: None,
                duration_ms: None,
                error: Some(e.to_string()),
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
}

/// Export a visualisation snapshot to JSON for debugging with NEAT-AI-Explore.
///
/// This is an optional debug tool that reads a Parquet recording and creature,
/// then writes a comprehensive JSON snapshot with recorded data, impacts, and
/// reconstruction checks.
pub fn export_visualisation_snapshot_internal(input_json: &str) -> Result<String> {
    let input: ExportVisualisationSnapshotInput = match serde_json::from_str(input_json) {
        Ok(value) => value,
        Err(e) => {
            let output = ExportVisualisationSnapshotOutput {
                success: false,
                out_file: None,
                stats: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    let options = export::ExportOptions {
        include_per_synapse_series: input.include_per_synapse_series,
        include_reconstruction_checks: input.include_reconstruction_checks,
        max_obs: input.max_obs,
        top_k_worst_samples: input.top_k_worst_samples.unwrap_or(20),
    };

    match export::export_visualisation_snapshot(
        &input.parquet_file,
        &input.creature,
        &input.out_file,
        &options,
    ) {
        Ok(stats) => {
            let output = ExportVisualisationSnapshotOutput {
                success: true,
                out_file: Some(input.out_file),
                stats: Some(ExportVisualisationStats {
                    obs_count: stats.obs_count,
                    neuron_count: stats.neuron_count,
                    synapse_count: stats.synapse_count,
                    output_count: stats.output_count,
                }),
                error: None,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let output = ExportVisualisationSnapshotOutput {
                success: false,
                out_file: None,
                stats: None,
                error: Some(e.to_string()),
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
}

/// FFI export for recording discovery data
///
/// # Safety
/// This function is unsafe because it deals with raw C strings.
/// The caller must ensure:
/// - input_json is a valid null-terminated C string
/// - The returned pointer is freed using free_discovery_result
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn record_discovery(input_json: *const std::ffi::c_char) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        // Read input C string
        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        // Call the Rust function with original string
        let json_result = match record_discovery_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                // Properly serialize error message to avoid JSON injection issues
                let output = RecordDiscoveryOutput {
                    success: false,
                    temp_dir: None,
                    file: None,
                    error: Some(e.to_string()),
                };
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    // Fallback if serialization fails (shouldn't happen)
                    r#"{"success":false,"error":"Failed to serialize error message"}"#.to_string()
                })
            }
        };

        // Return as C string - this is also protected by catch_unwind
        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                let error = r#"{"success":false,"error":"Failed to create output string"}"#;
                CString::new(error).unwrap().into_raw()
            }
        }
    }))
    .unwrap_or_else(|panic_info| {
        // If panic occurred, create a safe error response
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        // This should never fail, but if it does, we return null pointer
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

// ============================================================================
// Streaming Recording API - FFI Entry Points
// ============================================================================
// These functions support incremental recording to avoid JavaScript
// "Invalid string length" errors when serialising large datasets.
//
// Usage pattern from TypeScript:
//   1. start_discovery_session() → returns session_id
//   2. Loop: append_discovery_records() as data accumulates
//   3. finish_discovery_session() → finalises Parquet file
//
// Benefits:
//   - Each append call is small enough to serialise (e.g., ~50MB)
//   - Unlimited total sample size
//   - Partial data preserved if process crashes

/// Start a new streaming discovery session.
///
/// Creates a Parquet file and returns a session ID for subsequent append/finish calls.
///
/// Input JSON:
/// ```json
/// {
///   "creature": { ... },
///   "tempDir": "/path/to/temp/dir"
/// }
/// ```
///
/// Output JSON:
/// ```json
/// {
///   "success": true,
///   "sessionId": "uuid-string"
/// }
/// ```
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn start_discovery_session(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let input: StartSessionInput = match serde_json::from_str(input_str) {
            Ok(input) => input,
            Err(e) => {
                let output = StartSessionOutput {
                    success: false,
                    session_id: None,
                    error: Some(format!("Failed to parse input JSON: {e}")),
                };
                let json = serde_json::to_string(&output).unwrap();
                return CString::new(json).unwrap().into_raw();
            }
        };

        let output = match streaming::start_session(input.creature, input.temp_dir) {
            Ok(session_id) => StartSessionOutput {
                success: true,
                session_id: Some(session_id),
                error: None,
            },
            Err(e) => StartSessionOutput {
                success: false,
                session_id: None,
                error: Some(e.to_string()),
            },
        };

        let json = serde_json::to_string(&output).unwrap();
        CString::new(json).unwrap().into_raw()
    }))
    .unwrap_or_else(|panic_info| {
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

/// Append records to an existing streaming session.
///
/// Input JSON:
/// ```json
/// {
///   "sessionId": "uuid-string",
///   "observations": [
///     {
///       "obsIndex": 0,
///       "neuronData": [{ "neuronUuid": "...", "activation": 0.5, "value": 0.4, "errors": [0.1] }],
///       "inputs": [0.1, 0.2, 0.3]
///     }
///   ]
/// }
/// ```
///
/// Output JSON:
/// ```json
/// {
///   "success": true,
///   "recordsWritten": 42
/// }
/// ```
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn append_discovery_records(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let input: AppendRecordsInput = match serde_json::from_str(input_str) {
            Ok(input) => input,
            Err(e) => {
                let output = AppendRecordsOutput {
                    success: false,
                    records_written: None,
                    error: Some(format!("Failed to parse input JSON: {e}")),
                };
                let json = serde_json::to_string(&output).unwrap();
                return CString::new(json).unwrap().into_raw();
            }
        };

        // Convert observations to the internal format
        let batches: Vec<(u32, Vec<NeuronData>, Vec<f32>)> = input
            .observations
            .into_iter()
            .map(|obs| (obs.obs_index, obs.neuron_data, obs.inputs))
            .collect();

        let output = match streaming::append_records(&input.session_id, batches) {
            Ok(records_written) => AppendRecordsOutput {
                success: true,
                records_written: Some(records_written),
                error: None,
            },
            Err(e) => AppendRecordsOutput {
                success: false,
                records_written: None,
                error: Some(e.to_string()),
            },
        };

        let json = serde_json::to_string(&output).unwrap();
        CString::new(json).unwrap().into_raw()
    }))
    .unwrap_or_else(|panic_info| {
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

/// Finish a streaming session and finalise the Parquet file.
///
/// Input JSON:
/// ```json
/// {
///   "sessionId": "uuid-string"
/// }
/// ```
///
/// Output JSON:
/// ```json
/// {
///   "success": true,
///   "tempDir": "/path/to/temp/dir",
///   "file": "discovery_data.parquet",
///   "totalRecords": 12345
/// }
/// ```
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn finish_discovery_session(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let input: FinishSessionInput = match serde_json::from_str(input_str) {
            Ok(input) => input,
            Err(e) => {
                let output = FinishSessionOutput {
                    success: false,
                    temp_dir: None,
                    file: None,
                    total_records: None,
                    error: Some(format!("Failed to parse input JSON: {e}")),
                };
                let json = serde_json::to_string(&output).unwrap();
                return CString::new(json).unwrap().into_raw();
            }
        };

        let output = match streaming::finish_session(&input.session_id) {
            Ok((temp_dir, file, total_records)) => FinishSessionOutput {
                success: true,
                temp_dir: Some(temp_dir),
                file: Some(file),
                total_records: Some(total_records),
                error: None,
            },
            Err(e) => FinishSessionOutput {
                success: false,
                temp_dir: None,
                file: None,
                total_records: None,
                error: Some(e.to_string()),
            },
        };

        let json = serde_json::to_string(&output).unwrap();
        CString::new(json).unwrap().into_raw()
    }))
    .unwrap_or_else(|panic_info| {
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

/// Cancel a streaming session without finalising.
///
/// Use this to clean up if recording fails or is cancelled.
///
/// Input JSON:
/// ```json
/// {
///   "sessionId": "uuid-string"
/// }
/// ```
///
/// Output JSON:
/// ```json
/// {
///   "success": true
/// }
/// ```
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn cancel_discovery_session(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let input: CancelSessionInput = match serde_json::from_str(input_str) {
            Ok(input) => input,
            Err(e) => {
                let output = CancelSessionOutput {
                    success: false,
                    error: Some(format!("Failed to parse input JSON: {e}")),
                };
                let json = serde_json::to_string(&output).unwrap();
                return CString::new(json).unwrap().into_raw();
            }
        };

        let output = match streaming::cancel_session(&input.session_id) {
            Ok(()) => CancelSessionOutput {
                success: true,
                error: None,
            },
            Err(e) => CancelSessionOutput {
                success: false,
                error: Some(e.to_string()),
            },
        };

        let json = serde_json::to_string(&output).unwrap();
        CString::new(json).unwrap().into_raw()
    }))
    .unwrap_or_else(|panic_info| {
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn merge_discovery_parquet(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let json_result = match merge_discovery_parquet_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                let output = MergeParquetOutput {
                    success: false,
                    output_file: None,
                    error: Some(e.to_string()),
                };
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    r#"{"success":false,"error":"Failed to serialize error message"}"#.to_string()
                })
            }
        };

        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                let error = r#"{"success":false,"error":"Failed to create output string"}"#;
                CString::new(error).unwrap().into_raw()
            }
        }
    }))
    .unwrap_or_else(|panic_info| {
        // If panic occurred, create a safe error response
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        // This should never fail, but if it does, we return null pointer
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn rank_focus_neurons(input_json: *const std::ffi::c_char) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let json_result = match rank_focus_neurons_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                let output = RankFocusNeuronsOutput {
                    success: false,
                    neurons: None,
                    removal_candidates: None,
                    constant_neuron_removals: None,
                    max_output_error: None,
                    processed_neurons: None,
                    total_neurons: None,
                    duration_ms: None,
                    error: Some(e.to_string()),
                };
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    r#"{"success":false,"error":"Failed to serialize output"}"#.to_string()
                })
            }
        };

        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                let error = r#"{"success":false,"error":"Failed to create output string"}"#;
                CString::new(error).unwrap().into_raw()
            }
        }
    }))
    .unwrap_or_else(|panic_info| {
        // If panic occurred, create a safe error response
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        // This should never fail, but if it does, we return null pointer
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn analyze_parallel(input_json: *const std::ffi::c_char) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let json_result = match analyze_parallel_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                // Properly serialize error message to avoid JSON injection issues
                let output = AnalyzeParallelOutput {
                    success: false,
                    helpful_synapses: None,
                    harmful_synapses: None,
                    synapse_diagnostics: None,
                    synapse_gpu_used: None,
                    synapse_metadata: None,
                    helpful_neurons: None,
                    synapse_weight_updates: None,
                    coordinated_structural_candidates: None,
                    candidate_clusters: None,
                    neuron_diagnostics: None,
                    neuron_gpu_used: None,
                    neuron_metadata: None,
                    error: Some(format!("Failed to serialize output: {e}")),
                };
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    // Fallback if serialization fails (shouldn't happen)
                    r#"{"success":false,"error":"Failed to serialize error message"}"#.to_string()
                })
            }
        };

        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                let error = r#"{"success":false,"error":"Failed to create output string"}"#;
                CString::new(error).unwrap().into_raw()
            }
        }
    }))
    .unwrap_or_else(|panic_info| {
        // If panic occurred, create a safe error response
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        // This should never fail, but if it does, we return null pointer
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn check_gpu_available() -> *mut std::ffi::c_char {
    use std::ffi::CString;
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        let json_result = match check_gpu_available_internal() {
            Ok(json) => json,
            Err(e) => {
                // Properly serialize error message to avoid JSON injection issues
                let output = CheckGpuOutput {
                    success: false,
                    gpu_available: false,
                    reason: None,
                    error: Some(format!("Failed to probe GPU: {e}")),
                };
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    // Fallback if serialization fails (shouldn't happen)
                    r#"{"success":false,"gpuAvailable":false,"error":"Failed to serialize error message"}"#.to_string()
                })
            }
        };

        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                let error = r#"{"success":false,"gpuAvailable":false,"error":"Failed to create output string"}"#;
                CString::new(error).unwrap().into_raw()
            }
        }
    }))
    .unwrap_or_else(|panic_info| {
        // If panic occurred, create a safe error response
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"gpuAvailable\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        // This should never fail, but if it does, we return null pointer
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"gpuAvailable":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

/// FFI export for querying the library version
///
/// Returns the version string that was embedded in the binary at compile time.
/// This allows callers to verify they're using the expected version.
///
/// # Safety
/// The returned pointer must be freed using free_discovery_result
#[unsafe(no_mangle)]
pub extern "C" fn get_library_version() -> *mut std::ffi::c_char {
    use std::ffi::CString;
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        let json_result = match get_library_version_internal() {
            Ok(json) => json,
            Err(e) => {
                // Properly serialize error message to avoid JSON injection issues
                let output = GetVersionOutput {
                    success: false,
                    version: String::new(),
                    error: Some(format!("Failed to get version: {e}")),
                };
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    // Fallback if serialization fails (shouldn't happen)
                    r#"{"success":false,"version":"","error":"Failed to serialize error message"}"#
                        .to_string()
                })
            }
        };

        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                let error =
                    r#"{"success":false,"version":"","error":"Failed to create output string"}"#;
                CString::new(error).unwrap().into_raw()
            }
        }
    }))
    .unwrap_or_else(|panic_info| {
        // If panic occurred, create a safe error response
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"version\":\"\",\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        // This should never fail, but if it does, we return null pointer
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"version":"","error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

/// JSON input for read_discovery_records function
#[derive(Debug, Deserialize)]
pub struct ReadDiscoveryInput {
    pub parquet_file: String,
    pub neuron_uuid: String,
}

/// JSON output from read_discovery_records function
#[derive(Debug, Serialize)]
pub struct ReadDiscoveryOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub records: Option<Vec<DiscoverRecordJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// JSON representation of DiscoverRecord for serialization
#[derive(Debug, Serialize)]
pub struct DiscoverRecordJson {
    pub obs_index: u32,
    pub neuron_uuid: String,
    pub value: Option<f32>,
    pub activation: f32,
    pub errors: Vec<f32>,
}

/// Read discovery records from Parquet file for a specific neuron
///
/// Takes JSON input and returns JSON output for easy integration with TypeScript/DenoJS
pub fn read_discovery_records(input_json: &str) -> Result<String> {
    use crate::parquet_format::read_records_from_parquet;
    use crate::types::DiscoverRecord;

    // Parse input JSON
    let input: ReadDiscoveryInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let output = ReadDiscoveryOutput {
                success: false,
                records: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Read records from Parquet
    let records: Vec<DiscoverRecord> =
        match read_records_from_parquet(&input.parquet_file, &input.neuron_uuid) {
            Ok(records) => records,
            Err(e) => {
                let output = ReadDiscoveryOutput {
                    success: false,
                    records: None,
                    error: Some(e.to_string()),
                };
                return Ok(serde_json::to_string(&output)?);
            }
        };

    // Convert to JSON format
    let json_records: Vec<DiscoverRecordJson> = records
        .into_iter()
        .map(|r| DiscoverRecordJson {
            obs_index: r.obs_index,
            neuron_uuid: r.neuron_uuid,
            value: r.value,
            activation: r.activation,
            errors: r.errors,
        })
        .collect();

    let output = ReadDiscoveryOutput {
        success: true,
        records: Some(json_records),
        error: None,
    };

    let json_string = serde_json::to_string(&output)?;

    Ok(json_string)
}

/// FFI export for reading discovery records
///
/// # Safety
/// This function is unsafe because it deals with raw C strings.
/// The caller must ensure:
/// - input_json is a valid null-terminated C string
/// - The returned pointer is freed using free_discovery_result
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn read_discovery_records_ffi(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        // Read input C string
        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        // Call the Rust function
        let json_result = match read_discovery_records(input_str) {
            Ok(json) => json,
            Err(e) => {
                // Properly serialize error message to avoid JSON injection issues
                let output = ReadDiscoveryOutput {
                    success: false,
                    records: None,
                    error: Some(e.to_string()),
                };
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    // Fallback if serialization fails (shouldn't happen)
                    r#"{"success":false,"error":"Failed to serialize error message"}"#.to_string()
                })
            }
        };

        // Return as C string - this is also protected by catch_unwind
        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                let error = r#"{"success":false,"error":"Failed to create output string"}"#;
                CString::new(error).unwrap().into_raw()
            }
        }
    }))
    .unwrap_or_else(|panic_info| {
        // If panic occurred, create a safe error response
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        // This should never fail, but if it does, we return null pointer
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

/// Export a visualisation snapshot to JSON for debugging with NEAT-AI-Explore.
///
/// Input JSON:
/// ```json
/// {
///   "parquetFile": "/path/to/records.parquet",
///   "creature": { ... },
///   "outFile": "/path/to/snapshot.json",
///   "includePerSynapseSeries": true,
///   "includeReconstructionChecks": true,
///   "maxObs": null,
///   "topKWorstSamples": 20
/// }
/// ```
///
/// Output JSON:
/// ```json
/// {
///   "success": true,
///   "outFile": "/path/to/snapshot.json",
///   "stats": { "obsCount": 1000, "neuronCount": 50, "synapseCount": 200, "outputCount": 1 }
/// }
/// ```
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn export_visualisation_snapshot(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let json_result = match export_visualisation_snapshot_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                let output = ExportVisualisationSnapshotOutput {
                    success: false,
                    out_file: None,
                    stats: None,
                    error: Some(e.to_string()),
                };
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    r#"{"success":false,"error":"Failed to serialize error message"}"#.to_string()
                })
            }
        };

        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                let error = r#"{"success":false,"error":"Failed to create output string"}"#;
                CString::new(error).unwrap().into_raw()
            }
        }
    }))
    .unwrap_or_else(|panic_info| {
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn free_discovery_result(ptr: *mut std::ffi::c_char) {
    use std::ffi::CString;
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This is unlikely to panic, but we protect it anyway for safety
    let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        if !ptr.is_null() {
            unsafe {
                let _ = CString::from_raw(ptr);
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet_format::write_records_to_parquet;
    use crate::types::DiscoverRecord;
    use tempfile::tempdir;

    /// Helper macro to skip tests that require GPU when no GPU is available.
    /// This allows tests to pass gracefully in CI environments without GPUs.
    macro_rules! skip_if_no_gpu {
        () => {
            if !analysis::GpuAnalyzer::gpu_is_available() {
                eprintln!("⚠️  Skipping test: GPU not available");
                return;
            }
        };
    }

    /// Safely truncate a UTF-8 string at character boundaries
    ///
    /// Returns a string truncated to at most `max_bytes` bytes, ensuring the
    /// truncation occurs at a valid UTF-8 character boundary to avoid panics.
    fn truncate_utf8_safe(s: &str, max_bytes: usize) -> &str {
        if s.len() <= max_bytes {
            return s;
        }
        // Find the last valid character boundary at or before max_bytes
        // We iterate through character boundaries and keep the last one <= max_bytes
        let mut last_valid_boundary = 0;
        for (idx, _) in s.char_indices() {
            if idx > max_bytes {
                break;
            }
            last_valid_boundary = idx;
        }
        &s[..last_valid_boundary]
    }

    #[test]
    fn test_record_discovery_json_interface() {
        let input = r#"{
            "creature": {
                "neurons": [],
                "synapses": [],
                "input": 2,
                "output": 1
            },
            "training_data": [],
            "temp_dir": ".discovery/test"
        }"#;

        // Should parse without error
        let parsed: RecordDiscoveryInput = serde_json::from_str(input).unwrap();
        assert_eq!(parsed.creature.input, 2);
        assert_eq!(parsed.creature.output, 1);
    }

    #[test]
    fn test_error_message_json_escaping() {
        // Test that error messages with special characters are properly escaped
        let error_messages = vec![
            r#"Error with "quotes""#,
            r#"Error with \backslash"#,
            "Error with\nnewline",
            r#"Error with "quotes" and \backslash and\nnewline"#,
            r#"Error with "multiple" "quotes" and \multiple\backslashes"#,
        ];

        for error_msg in error_messages {
            // Test RecordDiscoveryOutput
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(error_msg.to_string()),
            };
            let json = serde_json::to_string(&output).unwrap();
            // Verify JSON is valid and can be parsed back
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed["success"], false);
            assert_eq!(parsed["error"].as_str(), Some(error_msg));

            // Test ReadDiscoveryOutput
            let read_output = ReadDiscoveryOutput {
                success: false,
                records: None,
                error: Some(error_msg.to_string()),
            };
            let read_json = serde_json::to_string(&read_output).unwrap();
            // Verify JSON is valid and can be parsed back
            let parsed_read: serde_json::Value = serde_json::from_str(&read_json).unwrap();
            assert_eq!(parsed_read["success"], false);
            assert_eq!(parsed_read["error"].as_str(), Some(error_msg));
        }
    }

    #[test]
    fn test_truncate_utf8_safe_ascii() {
        // Test with ASCII (single-byte characters)
        let s = "Hello, World!";
        assert_eq!(truncate_utf8_safe(s, 5), "Hello");
        assert_eq!(truncate_utf8_safe(s, 13), s);
        assert_eq!(truncate_utf8_safe(s, 100), s);
    }

    #[test]
    fn test_truncate_utf8_safe_multibyte() {
        // Test with multi-byte UTF-8 characters (each emoji is 4 bytes)
        let s = "Hello 🦀 World 🌍";
        // "Hello 🦀" is 11 bytes: "Hello " (6) + "🦀" (4) + " W" (2) = 12 bytes
        // But we want to test truncation at byte 11, which should cut before "🦀"
        let truncated = truncate_utf8_safe(s, 11);
        // Should truncate at character boundary, not in middle of emoji
        assert!(truncated.len() <= 11);
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    fn test_truncate_utf8_safe_boundary_at_500() {
        // Test the specific case mentioned in the issue: truncation at byte 500
        // Create a string with multi-byte characters near position 500
        let mut s = String::new();
        for _ in 0..100 {
            s.push('🦀'); // Each emoji is 4 bytes, so 100 emojis = 400 bytes
        }
        s.push_str("Hello"); // Add 5 more bytes = 405 bytes total

        // Truncate at 500 - should return full string since it's < 500
        assert_eq!(truncate_utf8_safe(&s, 500), s.as_str());

        // Truncate at 400 - should cut at character boundary (after 100 emojis)
        let truncated = truncate_utf8_safe(&s, 400);
        assert_eq!(truncated.len(), 400); // Exactly 100 emojis
        assert!(truncated.is_char_boundary(truncated.len()));

        // Truncate at 401 - should include first character of "Hello" (H = 1 byte)
        let truncated2 = truncate_utf8_safe(&s, 401);
        assert_eq!(truncated2.len(), 401);
        assert!(truncated2.is_char_boundary(truncated2.len()));
    }

    #[test]
    fn test_truncate_utf8_safe_empty() {
        let s = "";
        assert_eq!(truncate_utf8_safe(s, 0), "");
        assert_eq!(truncate_utf8_safe(s, 100), "");
    }

    #[test]
    fn test_truncate_utf8_safe_exact_boundary() {
        // Test truncation exactly at a character boundary
        let s = "Hello🦀World";
        // "Hello" = 5 bytes, "🦀" starts at byte 5
        let truncated = truncate_utf8_safe(s, 5);
        assert_eq!(truncated, "Hello");
        assert_eq!(truncated.len(), 5);
    }

    #[test]
    fn check_gpu_available_internal_returns_well_formed_json() {
        let json =
            check_gpu_available_internal().expect("GPU availability probe should return JSON");
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("GPU availability output should be valid JSON");

        assert!(
            value["success"].is_boolean(),
            "success flag should be a boolean"
        );
        assert!(
            value["gpuAvailable"].is_boolean(),
            "gpuAvailable flag should be a boolean"
        );

        // Platform-specific behaviour:
        // - On macOS: success=false when GPU unavailable (error condition)
        // - On Linux: success=true when GPU unavailable (graceful disable)
        // - reason field provides diagnostic information when GPU is unavailable
        if !value["gpuAvailable"].as_bool().unwrap_or(false) {
            assert!(
                value["reason"].is_string(),
                "reason should be provided when GPU is unavailable"
            );
        }
    }

    #[test]
    fn get_library_version_internal_returns_well_formed_json() {
        let json = get_library_version_internal().expect("Version query should return JSON");
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("Version output should be valid JSON");

        assert_eq!(value["success"], true);
        assert_eq!(
            value["version"]
                .as_str()
                .expect("version should be a string"),
            env!("CARGO_PKG_VERSION")
        );
        assert!(value["error"].is_null(), "error should be null on success");
    }

    #[test]
    fn analyze_parallel_internal_returns_combined_payload() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let parquet_file = temp_dir
            .path()
            .join("records.parquet")
            .to_str()
            .expect("temp path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..12u32 {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to persist discovery records");

        let input_json = serde_json::json!({
            "parquetFile": parquet_file,
            "creature": {
                "neurons": [{
                    "uuid": "output-0",
                    "type": "output",
                    "squash": "IDENTITY",
                    "bias": 0.0
                }],
                "synapses": [{
                    "from_uuid": "input-0",
                    "to_uuid": "output-0",
                    "weight": 0.4
                }],
                "input": 1,
                "output": 1
            },
            "focusNeurons": ["output-0"],
            "maxSynapseCandidates": 5,
            "maxNeuronCandidates": 5,
            "requireGpu": false
        })
        .to_string();

        let output_json =
            analyze_parallel_internal(&input_json).expect("parallel analysis should return JSON");
        let output: serde_json::Value =
            serde_json::from_str(&output_json).expect("output should be valid JSON");

        assert_eq!(output["success"], true);
        assert!(
            output["helpfulSynapses"].is_array(),
            "parallel analysis should include helpful synapse array"
        );
        assert!(
            output["helpfulNeurons"].is_array(),
            "parallel analysis should include helpful neuron array"
        );
        assert!(
            output["synapseGpuUsed"].is_boolean(),
            "parallel analysis should report GPU usage for synapses"
        );
        assert!(
            output["neuronGpuUsed"].is_boolean(),
            "parallel analysis should report GPU usage for neurons"
        );
    }

    #[test]
    fn analyze_parallel_threads_random_seed_into_combined_input() {
        let input_json = serde_json::json!({
            "parquetFile": "example.parquet",
            "creature": {
                "neurons": [],
                "synapses": [],
                "input": 1,
                "output": 1
            },
            "focusNeurons": ["output-0"],
            "maxSynapseCandidates": 5,
            "maxNeuronCandidates": 5,
            "analysisDeadlineMs": 1234,
            "randomSeed": 42
        })
        .to_string();

        let parsed: AnalyzeParallelInput =
            serde_json::from_str(&input_json).expect("input JSON should deserialize");
        let combined = build_analyze_all_input_from_parallel(parsed);

        assert_eq!(combined.random_seed, Some(42));
        assert_eq!(combined.analysis_deadline_ms, Some(1234));
    }

    #[test]
    fn coordinated_structural_candidates_are_exposed_via_analyze_parallel_output_shape() {
        // This test is intentionally light-weight and CPU-only.
        //
        // The end-to-end behaviour is covered by the integration test:
        // `tests/coordinated_structural_mercury_digital.rs`.
        let candidate = CoordinatedStructuralCandidateJson {
            operations: vec![
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: "input-0".to_string(),
                    to_neuron_uuid: "output-0".to_string(),
                },
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: "input-1".to_string(),
                    to_neuron_uuid: "output-0".to_string(),
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: "input-1".to_string(),
                    to_neuron_uuid: "output-0".to_string(),
                    weight: 0.1,
                },
            ],
            expected_creature_score_gain: 0.01,
            comment: Some("Example".to_string()),
        };

        let value = serde_json::to_value(&candidate).expect("candidate should serialise");
        assert!(value["operations"].is_array());
        assert!(value["expectedCreatureScoreGain"].is_number());
    }
}
