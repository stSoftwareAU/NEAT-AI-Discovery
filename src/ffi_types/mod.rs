//! FFI boundary types — JSON request/response structs.
//!
//! All types in this module are used to serialise and deserialise data at the
//! FFI boundary between this Rust library and the TypeScript/Deno controller.

mod candidates;
mod error_classification;
mod requests;
mod responses;
mod session;

// Re-export all sub-module types to preserve the existing public API.
pub use candidates::*;
pub use error_classification::*;
pub use requests::*;
pub use responses::*;
pub use session::*;

use serde::{Deserialize, Deserializer, Serialize};

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
    ///
    /// Normalised to ASCII uppercase at deserialisation time (Issue #753) so
    /// downstream detection modules can match without per-neuron allocations.
    #[serde(default = "default_squash", deserialize_with = "deserialise_squash")]
    pub squash: String,
    #[serde(default)]
    pub bias: f32,
}

fn default_squash() -> String {
    "IDENTITY".to_string()
}

/// Deserialise a squash name and normalise to ASCII uppercase (Issue #753).
fn deserialise_squash<'de, D>(deserialiser: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = String::deserialize(deserialiser)?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(String::new());
    }
    // Fast path: already uppercase ASCII — avoid allocation.
    if trimmed.bytes().all(|b| !b.is_ascii_lowercase()) {
        return Ok(trimmed.to_string());
    }
    Ok(trimmed.to_ascii_uppercase())
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
