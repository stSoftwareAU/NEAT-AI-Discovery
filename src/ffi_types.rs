//! FFI boundary types — JSON request/response structs.
//!
//! All types in this module are used to serialise and deserialise data at the
//! FFI boundary between this Rust library and the TypeScript/Deno controller.

use serde::{Deserialize, Serialize};

use crate::analysis;

// ============================================================================
// Creature / Neuron / Synapse representations
// ============================================================================

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

// ============================================================================
// Recording API types
// ============================================================================

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

// ============================================================================
// Candidate types
// ============================================================================

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

// ============================================================================
// Analysis input/output types
// ============================================================================

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
    /// Current neuron fingerprints for incremental analysis (Issue #490).
    ///
    /// Callers should store these and pass them back as `previousNeuronFingerprints`
    /// on the next discovery run to enable incremental analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_fingerprints:
        Option<std::collections::HashMap<String, analysis::neuron_fingerprint::NeuronFingerprint>>,
    /// Number of focus neurons skipped due to unchanged fingerprints (Issue #490).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint_cache_hits: Option<usize>,
    /// Number of focus neurons analysed (changed or new fingerprints) (Issue #490).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint_cache_misses: Option<usize>,
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
    /// Previous neuron fingerprints from the last discovery run (Issue #490).
    ///
    /// When provided, neurons whose structural fingerprint is unchanged
    /// are skipped, avoiding redundant GPU computation.
    #[serde(default)]
    pub previous_neuron_fingerprints:
        Option<std::collections::HashMap<String, analysis::neuron_fingerprint::NeuronFingerprint>>,
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

// ============================================================================
// Read discovery records types
// ============================================================================

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

// ============================================================================
// Conversion helpers
// ============================================================================

/// Convert GPU adapter info to JSON representation.
pub(crate) fn gpu_info_to_json(info: &analysis::GpuAdapterInfo) -> GpuAdapterInfoJson {
    GpuAdapterInfoJson {
        name: info.name.clone(),
        unified_memory: info.has_unified_memory,
        zero_copy_enabled: info.zero_copy_enabled,
    }
}

/// Convert internal timing data to JSON representation.
pub(crate) fn timing_to_json(timing: &analysis::AnalysisTiming) -> AnalysisTimingJson {
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

pub(crate) fn synapse_diagnostics_json(
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

pub(crate) fn neuron_diagnostics_json(
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
