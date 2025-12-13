//! NEAT-AI Discovery Library
//!
//! High-performance Rust library for recording neuron activations and errors
//! during the discovery training phase, then scanning recorded data to identify
//! beneficial new synapses/neurons that would reduce error.

pub mod analysis;
pub mod debug;
pub mod focus;
pub mod parquet_format;
pub mod record;
pub mod types;

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
    pub squash: String,
    pub bias: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SynapseJson {
    #[serde(default)]
    pub from_uuid: String,
    #[serde(default)]
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

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CandidateSynapseJson {
    pub from_neuron_uuid: String,
    pub to_neuron_uuid: String,
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
}

#[derive(Debug, Serialize, Clone)]
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
    pub incoming_weight: f32,
    pub outgoing_weight: f32,
    pub squash: String,
    pub bias: f32,
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
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeParallelInput {
    pub parquet_file: String,
    pub creature: CreatureJson,
    pub focus_neurons: Vec<String>,
    #[serde(default)]
    pub improvement_threshold: Option<f32>,
    #[serde(default)]
    pub harmful_threshold: Option<f32>,
    #[serde(default)]
    pub max_synapse_candidates: Option<usize>,
    #[serde(default)]
    pub max_neuron_candidates: Option<usize>,
    #[serde(default)]
    pub analysis_deadline_ms: Option<u64>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub helpful_neurons: Option<Vec<CandidateNeuronJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_diagnostics: Option<Vec<NeuronDiagnosticJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_gpu_used: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Internal input structure for synapse analysis (used by analyze_all)
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeSynapsesInput {
    pub parquet_file: String,
    pub creature: CreatureJson,
    pub focus_neurons: Vec<String>,
    #[serde(default)]
    pub improvement_threshold: Option<f32>,
    #[serde(default)]
    pub max_candidates: Option<usize>,
    #[serde(default)]
    pub analysis_deadline_ms: Option<u64>,
}

/// Internal input structure for neuron analysis (used by analyze_all)
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeNeuronsInput {
    pub parquet_file: String,
    pub creature: CreatureJson,
    pub focus_neurons: Vec<String>,
    #[serde(default)]
    pub improvement_threshold: Option<f32>,
    #[serde(default)]
    pub max_candidates: Option<usize>,
    #[serde(default)]
    pub analysis_deadline_ms: Option<u64>,
}

/// Internal input structure for combined analysis (used by analyze_parallel)
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeAllInput {
    pub parquet_file: String,
    pub creature: CreatureJson,
    pub focus_neurons: Vec<String>,
    #[serde(default)]
    pub improvement_threshold: Option<f32>,
    #[serde(default)]
    pub harmful_threshold: Option<f32>,
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
                helpful_neurons: None,
                neuron_diagnostics: None,
                neuron_gpu_used: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    let combined_input = AnalyzeAllInput {
        parquet_file: input.parquet_file,
        creature: input.creature,
        focus_neurons: input.focus_neurons,
        improvement_threshold: input.improvement_threshold,
        harmful_threshold: input.harmful_threshold,
        max_synapse_candidates: input.max_synapse_candidates,
        max_neuron_candidates: input.max_neuron_candidates,
        analysis_deadline_ms: input.analysis_deadline_ms,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
    };

    match analysis::analyze_all(&combined_input) {
        Ok(result) => {
            let synapse = result.synapse;
            let neuron = result.neuron;
            let output = AnalyzeParallelOutput {
                success: true,
                helpful_synapses: synapse.as_ref().map(|s| s.helpful_synapses.clone()),
                harmful_synapses: synapse.as_ref().map(|s| s.harmful_synapses.clone()),
                synapse_diagnostics: synapse
                    .as_ref()
                    .and_then(|s| synapse_diagnostics_json(s.no_candidate_reasons.as_slice())),
                synapse_gpu_used: synapse.as_ref().map(|s| s.gpu_used),
                helpful_neurons: neuron.as_ref().map(|n| n.helpful_neurons.clone()),
                neuron_diagnostics: neuron
                    .as_ref()
                    .and_then(|n| neuron_diagnostics_json(n.no_candidate_reasons.as_slice())),
                neuron_gpu_used: neuron.as_ref().map(|n| n.gpu_used),
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
                helpful_neurons: None,
                neuron_diagnostics: None,
                neuron_gpu_used: None,
                error: Some(e.to_string()),
            };
            Ok(serde_json::to_string(&output)?)
        }
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
                max_output_error: None,
                processed_neurons: None,
                total_neurons: None,
                duration_ms: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    match focus::rank_focus_neurons(&input.parquet_file, &input.creature, input.max_results) {
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

/// FFI export for recording discovery data
///
/// # Safety
/// This function is unsafe because it deals with raw C strings.
/// The caller must ensure:
/// - input_json is a valid null-terminated C string
/// - The returned pointer is freed using free_discovery_result
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
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

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
                    helpful_neurons: None,
                    neuron_diagnostics: None,
                    neuron_gpu_used: None,
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

#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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

/// FFI export for freeing memory allocated by read_discovery_records_ffi
///
/// # Safety
/// This function is unsafe because it frees memory allocated by Rust.
/// The caller must ensure ptr was returned by read_discovery_records_ffi.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
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
            "improvementThreshold": 0.01,
            "harmfulThreshold": -0.05,
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
}
