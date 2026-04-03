//! Data structures for visualisation snapshot export.
//!
//! Contains all serialisable types used by the export pipeline: snapshot
//! metadata, neuron/synapse recordings, reconstruction checks, and export
//! options.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::CreatureJson;

/// Snapshot metadata
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotMeta {
    pub exported_at: String,
    pub discovery_version: String,
    pub parquet_file: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub notes: Option<String>,
}

/// Per-neuron stats (aggregates)
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct NeuronRecordingStats {
    pub mean_activation: f32,
    pub activation_variance: f32,
    pub activation_min: f32,
    pub activation_max: f32,
    /// Arithmetic mean of errors (can be misleading as +/- cancel out)
    pub mean_error: f32,
    /// Mean Absolute Error - used in focus neuron ranking
    pub mean_absolute_error: f32,
    /// Mean Squared Error - matches production creature scoring
    pub mean_squared_error: f32,
    pub error_variance: f32,
    pub error_min: f32,
    pub error_max: f32,
    pub record_count: usize,
}

/// Per-neuron recorded data (columnar format)
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuronRecording {
    /// Activation values per obsIndex (aligned with obsIndices array)
    pub activation: Vec<f32>,
    /// Value (pre-activation) per obsIndex; null encoded as JSON null
    pub value: Vec<Option<f32>>,
    /// Errors per obsIndex (each element is a vector of error values)
    pub errors: Vec<Vec<f32>>,
    /// Aggregated stats
    pub stats: NeuronRecordingStats,
}

/// Per-synapse stats (aggregates)
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SynapseStats {
    pub mean_contribution: f32,
    pub contribution_variance: f32,
    pub contribution_min: f32,
    pub contribution_max: f32,
}

/// Per-synapse derived data (columnar format)
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SynapseDerived {
    pub from_uuid: String,
    pub to_uuid: String,
    pub weight: f32,
    /// Contribution = fromActivation * weight per obsIndex
    pub contribution: Vec<f32>,
    /// Aggregated stats
    pub stats: SynapseStats,
}

/// Reconstruction check results for a single neuron
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReconstructionCheck {
    pub neuron_uuid: String,
    pub squash: String,
    pub bias: f32,
    /// max|recordedValue - reconstructedValue|
    pub max_value_delta: f32,
    /// max|recordedActivation - reconstructedActivation|
    pub max_activation_delta: f32,
    /// mean|valueDelta|
    pub mean_value_delta: f32,
    /// mean|activationDelta|
    pub mean_activation_delta: f32,
    /// Top-K worst obsIndex + deltas
    pub worst_samples: Vec<ReconstructionSample>,
}

/// A single sample's reconstruction delta
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReconstructionSample {
    pub obs_index: u32,
    pub recorded_value: Option<f32>,
    pub reconstructed_value: f32,
    pub value_delta: f32,
    pub recorded_activation: f32,
    pub reconstructed_activation: f32,
    pub activation_delta: f32,
}

/// Derived data section
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DerivedData {
    /// Impact scores keyed by neuron UUID
    pub impacts_by_neuron_uuid: HashMap<String, f32>,
    /// Synapse-level derived data keyed by "fromUuid->toUuid"
    pub synapses: HashMap<String, SynapseDerived>,
    /// Reconstruction checks per neuron (non-input neurons only)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reconstruction_checks: Option<Vec<ReconstructionCheck>>,
}

/// Recording data section
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingData {
    /// Ordered obsIndex values
    pub obs_indices: Vec<u32>,
    /// Per-neuron recorded data keyed by neuron UUID
    pub neurons: HashMap<String, NeuronRecording>,
}

/// The full snapshot structure written to JSON
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualisationSnapshot {
    pub meta: SnapshotMeta,
    pub creature: CreatureJson,
    pub recording: RecordingData,
    pub derived: DerivedData,
}

/// Export options
pub struct ExportOptions {
    pub include_per_synapse_series: bool,
    pub include_reconstruction_checks: bool,
    pub max_obs: Option<u32>,
    pub top_k_worst_samples: usize,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_per_synapse_series: true,
            include_reconstruction_checks: true,
            max_obs: None,
            top_k_worst_samples: 20,
        }
    }
}

/// Export result statistics
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportStats {
    pub obs_count: usize,
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub output_count: usize,
}
