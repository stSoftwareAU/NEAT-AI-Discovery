//! Streaming session types for the FFI boundary.
//!
//! These support incremental recording to avoid JavaScript "Invalid string length"
//! errors when serialising large datasets.

use serde::{Deserialize, Serialize};

use super::creature::NeuronData;

/// JSON input for start_discovery_session function
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartSessionInput {
    pub creature: super::creature::CreatureJson,
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
