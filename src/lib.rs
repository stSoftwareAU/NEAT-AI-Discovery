//! NEAT-AI Discovery Library
//!
//! High-performance Rust library for recording neuron activations and errors
//! during the discovery training phase, then scanning recorded data to identify
//! beneficial new synapses/neurons that would reduce error.

pub mod analysis;
pub mod focus;
pub mod parquet_format;
pub mod record;
pub mod types;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// JSON input for record_discovery function
#[derive(Debug, Deserialize)]
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
#[derive(Debug, Deserialize)]
pub struct CreatureJson {
    pub neurons: Vec<NeuronJson>,
    pub synapses: Vec<SynapseJson>,
    pub input: usize,
    pub output: usize,
}

#[derive(Debug, Deserialize)]
pub struct NeuronJson {
    pub uuid: String,
    #[serde(rename = "type")]
    pub neuron_type: String,
    pub squash: String,
    pub bias: f32,
}

#[derive(Debug, Deserialize)]
pub struct SynapseJson {
    pub from_uuid: String,
    pub to_uuid: String,
    pub weight: f32,
}

/// Pre-computed neuron data for a single neuron
#[derive(Debug, Deserialize)]
pub struct NeuronData {
    pub neuron_uuid: String,
    pub activation: f32,
    #[serde(default)]
    pub value: Option<f32>,
    pub errors: Vec<f32>,
}

/// Training data record
#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
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
    pub require_gpu: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateSynapseJson {
    pub from_neuron_uuid: String,
    pub to_neuron_uuid: String,
    pub weight: f32,
    pub expected_improvement_percentage: f32,
    pub improved_count: u32,
    pub total_count: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeSynapsesOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_used: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub helpful_synapses: Option<Vec<CandidateSynapseJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harmful_synapses: Option<Vec<CandidateSynapseJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<SynapseDiagnosticJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
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
    pub require_gpu: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateNeuronJson {
    pub source_neuron_uuid: String,
    pub target_neuron_uuid: String,
    pub incoming_weight: f32,
    pub outgoing_weight: f32,
    pub squash: String,
    pub bias: f32,
    pub expected_improvement_percentage: f32,
    pub improved_count: u32,
    pub total_count: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeNeuronsOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_used: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub helpful_neurons: Option<Vec<CandidateNeuronJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<NeuronDiagnosticJson>>,
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
    pub impact: f32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RankFocusNeuronsOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neurons: Option<Vec<RankedNeuronJson>>,
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
    pub expected_improvement_percentage: Option<f32>,
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

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuronDiagnosticReasonJson {
    NoEligibleSources,
    NoDiagnostics,
    NoSamples,
    NotEnoughActivations,
    WeightDegenerate,
    BelowThreshold,
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
    pub expected_improvement_percentage: Option<f32>,
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
                        expected_improvement_percentage: detail.expected_improvement,
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
                        expected_improvement_percentage: detail.expected_improvement,
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

pub fn rank_focus_neurons_internal(input_json: &str) -> Result<String> {
    let input: RankFocusNeuronsInput = match serde_json::from_str(input_json) {
        Ok(value) => value,
        Err(e) => {
            let output = RankFocusNeuronsOutput {
                success: false,
                neurons: None,
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
                })
                .collect();
            let output = RankFocusNeuronsOutput {
                success: true,
                neurons: Some(neurons),
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
    let result = match record_discovery_internal(input_str) {
        Ok(json) => json,
        Err(e) => {
            // Properly serialize error message to avoid JSON injection issues
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(e.to_string()),
            };
            let error_json = serde_json::to_string(&output).unwrap_or_else(|_| {
                // Fallback if serialization fails (shouldn't happen)
                r#"{"success":false,"error":"Failed to serialize error message"}"#.to_string()
            });
            return CString::new(error_json).unwrap().into_raw();
        }
    };

    // Return as C string
    match CString::new(result) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => {
            let error = r#"{"success":false,"error":"Failed to create output string"}"#;
            CString::new(error).unwrap().into_raw()
        }
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn merge_discovery_parquet(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};

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

    let result = match merge_discovery_parquet_internal(input_str) {
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

    match CString::new(result) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => {
            let error = r#"{"success":false,"error":"Failed to create output string"}"#;
            CString::new(error).unwrap().into_raw()
        }
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn rank_focus_neurons(input_json: *const std::ffi::c_char) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};

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

    let result = match rank_focus_neurons_internal(input_str) {
        Ok(json) => json,
        Err(e) => {
            let output = RankFocusNeuronsOutput {
                success: false,
                neurons: None,
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

    match CString::new(result) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => {
            let error = r#"{"success":false,"error":"Failed to create output string"}"#;
            CString::new(error).unwrap().into_raw()
        }
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn analyze_synapses(input_json: *const std::ffi::c_char) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};

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

    let output = match serde_json::from_str::<AnalyzeSynapsesInput>(input_str) {
        Ok(input) => match analysis::analyze_synapses(&input) {
            Ok(result) => AnalyzeSynapsesOutput {
                success: true,
                gpu_used: Some(result.gpu_used),
                helpful_synapses: Some(result.helpful_synapses),
                harmful_synapses: Some(result.harmful_synapses),
                diagnostics: synapse_diagnostics_json(&result.no_candidate_reasons),
                error: None,
            },
            Err(e) => AnalyzeSynapsesOutput {
                success: false,
                gpu_used: None,
                helpful_synapses: None,
                harmful_synapses: None,
                diagnostics: None,
                error: Some(e.to_string()),
            },
        },
        Err(e) => AnalyzeSynapsesOutput {
            success: false,
            gpu_used: None,
            helpful_synapses: None,
            harmful_synapses: None,
            diagnostics: None,
            error: Some(format!("Failed to parse input JSON: {e}")),
        },
    };

    let json = match serde_json::to_string(&output) {
        Ok(json) => json,
        Err(e) => {
            let fallback =
                format!("{{\"success\":false,\"error\":\"Failed to serialize output: {e}\"}}");
            return CString::new(fallback).unwrap().into_raw();
        }
    };

    match CString::new(json) {
        Ok(result) => result.into_raw(),
        Err(_) => {
            let error = r#"{"success":false,"error":"Failed to create output string"}"#;
            CString::new(error).unwrap().into_raw()
        }
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn analyze_neurons(input_json: *const std::ffi::c_char) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};

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

    let output = match serde_json::from_str::<AnalyzeNeuronsInput>(input_str) {
        Ok(input) => match analysis::analyze_neurons(&input) {
            Ok(result) => AnalyzeNeuronsOutput {
                success: true,
                gpu_used: Some(result.gpu_used),
                helpful_neurons: Some(result.helpful_neurons),
                diagnostics: neuron_diagnostics_json(&result.no_candidate_reasons),
                error: None,
            },
            Err(e) => AnalyzeNeuronsOutput {
                success: false,
                gpu_used: None,
                helpful_neurons: None,
                diagnostics: None,
                error: Some(e.to_string()),
            },
        },
        Err(e) => AnalyzeNeuronsOutput {
            success: false,
            gpu_used: None,
            helpful_neurons: None,
            diagnostics: None,
            error: Some(format!("Failed to parse input JSON: {e}")),
        },
    };

    let json = match serde_json::to_string(&output) {
        Ok(json) => json,
        Err(e) => {
            let fallback =
                format!("{{\"success\":false,\"error\":\"Failed to serialize output: {e}\"}}");
            return CString::new(fallback).unwrap().into_raw();
        }
    };

    match CString::new(json) {
        Ok(result) => result.into_raw(),
        Err(_) => {
            let error = r#"{"success":false,"error":"Failed to create output string"}"#;
            CString::new(error).unwrap().into_raw()
        }
    }
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
    let result = match read_discovery_records(input_str) {
        Ok(json) => json,
        Err(e) => {
            // Properly serialize error message to avoid JSON injection issues
            let output = ReadDiscoveryOutput {
                success: false,
                records: None,
                error: Some(e.to_string()),
            };
            let error_json = serde_json::to_string(&output).unwrap_or_else(|_| {
                // Fallback if serialization fails (shouldn't happen)
                r#"{"success":false,"error":"Failed to serialize error message"}"#.to_string()
            });
            return CString::new(error_json).unwrap().into_raw();
        }
    };

    // Return as C string
    match CString::new(result) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => {
            let error = r#"{"success":false,"error":"Failed to create output string"}"#;
            CString::new(error).unwrap().into_raw()
        }
    }
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
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
