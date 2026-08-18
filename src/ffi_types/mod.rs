//! FFI boundary types — JSON request/response structs.
//!
//! All types in this module are used to serialise and deserialise data at the
//! FFI boundary between this Rust library and the TypeScript/Deno controller.

mod candidates;
mod cleanup;
mod creature_bounds;
mod error_classification;
mod forward_only_validation;
mod requests;
mod responses;
mod session;

// Re-export all sub-module types to preserve the existing public API.
pub use candidates::*;
pub use cleanup::*;
pub use creature_bounds::{
    MAX_CREATURE_INPUT_NEURONS, observation_width_error, validate_creature_input_bounds,
};
pub use error_classification::*;
pub use forward_only_validation::validate_forward_only_synapses;
pub use requests::*;
pub use responses::*;
pub use session::*;

use serde::{Deserialize, Deserializer, Serialize};

// ============================================================================
// FFI contract version (Issue #952)
// ============================================================================

/// Current discovery FFI schema version.
///
/// Callers can use this to reject stale cached payloads instead of guessing
/// compatibility. Bump this when the wire format changes.
pub const SCHEMA_VERSION: &str = "2";

// ============================================================================
// UUID validation (Issue #952)
// ============================================================================

/// Returns `true` if the string is a purely numeric integer (e.g. `"0"`, `"42"`,
/// `"999999"`). These are runtime integer IDs that must never cross the FFI
/// boundary.
fn is_numeric_id(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Deserialise a neuron UUID string, rejecting purely numeric identifiers.
///
/// Accepts RFC 4122 UUIDs, `input-N` format, and any other descriptive string
/// identifier. Rejects stringified integers (Issue #952).
fn deserialise_neuron_uuid<'de, D>(deserialiser: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = String::deserialize(deserialiser)?;
    if is_numeric_id(&raw) {
        return Err(serde::de::Error::custom(format!(
            "numeric integer neuron ID \"{raw}\" is not permitted in FFI payloads \
             (Issue #952). Use a stable UUID string instead."
        )));
    }
    Ok(raw)
}

/// Deserialise a synapse UUID string, rejecting purely numeric identifiers.
///
/// Empty strings are allowed (serde default for missing fields).
fn deserialise_synapse_uuid<'de, D>(deserialiser: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = String::deserialize(deserialiser)?;
    if is_numeric_id(&raw) {
        return Err(serde::de::Error::custom(format!(
            "numeric integer synapse UUID \"{raw}\" is not permitted in FFI payloads \
             (Issue #952). Use a stable UUID string instead."
        )));
    }
    Ok(raw)
}

// ============================================================================
// Observation width validation (Issue #2020)
// ============================================================================

/// Deserialise the creature's `input` count, rejecting `input < 1`
/// (Issue #2020).
///
/// The top-level `input` integer is the observation width and cannot be
/// re-derived — `neurons` lists only non-input neurons — so a zero here is a
/// corrupt creature, not a creature with no inputs. Fail at the boundary.
fn deserialise_input_width<'de, D>(deserialiser: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = usize::deserialize(deserialiser)?;
    if let Some(detail) = observation_width_error("input", raw) {
        return Err(serde::de::Error::custom(detail));
    }
    Ok(raw)
}

/// Deserialise the creature's `output` count, rejecting `output < 1`
/// (Issue #2020).
fn deserialise_output_width<'de, D>(deserialiser: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = usize::deserialize(deserialiser)?;
    if let Some(detail) = observation_width_error("output", raw) {
        return Err(serde::de::Error::custom(detail));
    }
    Ok(raw)
}

/// Serialise the creature's `input` count, refusing to emit `input < 1`
/// (Issue #2020).
///
/// A creature that leaves this library without its observation width cannot
/// be repaired downstream, so a width_less creature is an error at the point
/// of emission rather than a payload the host has to reject later.
fn serialise_input_width<S>(value: &usize, serialiser: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if let Some(detail) = observation_width_error("input", *value) {
        return Err(serde::ser::Error::custom(detail));
    }
    value.serialize(serialiser)
}

/// Serialise the creature's `output` count, refusing to emit `output < 1`
/// (Issue #2020).
fn serialise_output_width<S>(value: &usize, serialiser: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if let Some(detail) = observation_width_error("output", *value) {
        return Err(serde::ser::Error::custom(detail));
    }
    value.serialize(serialiser)
}

// ============================================================================
// Creature / Neuron / Synapse representations
// ============================================================================

/// JSON representation of Creature
///
/// `input` and `output` are the creature's observation width — the number of
/// input and output neurons. Input neurons are **not** listed in `neurons`
/// (only non-input neurons are), so these two integers are authoritative and
/// cannot be re-derived from the neuron list. Both are required (no serde
/// default) and both must be `>= 1`: a value below one is rejected at
/// deserialisation and refused at serialisation (Issue #2020).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CreatureJson {
    pub neurons: Vec<NeuronJson>,
    pub synapses: Vec<SynapseJson>,
    /// Number of input neurons (observation width). Must be `>= 1`.
    #[serde(
        deserialize_with = "deserialise_input_width",
        serialize_with = "serialise_input_width"
    )]
    pub input: usize,
    /// Number of output neurons. Must be `>= 1`.
    #[serde(
        deserialize_with = "deserialise_output_width",
        serialize_with = "serialise_output_width"
    )]
    pub output: usize,
}

/// JSON representation of a single neuron on the FFI boundary — identity,
/// type, activation function (squash), and bias.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NeuronJson {
    /// Stable neuron identity string. Must be a UUID or descriptive identifier
    /// (e.g. `"550e8400-e29b-41d4-…"`, `"input-0"`).
    ///
    /// Purely numeric integer IDs are rejected at the FFI boundary (Issue #952).
    #[serde(deserialize_with = "deserialise_neuron_uuid")]
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

/// JSON representation of a single synapse on the FFI boundary — source
/// and target neuron identities, weight, and optional IF-neuron branch
/// classification.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SynapseJson {
    /// Source neuron identity string. Must be a UUID or descriptive identifier.
    /// Purely numeric integer IDs are rejected (Issue #952).
    #[serde(
        default,
        alias = "fromUUID",
        deserialize_with = "deserialise_synapse_uuid"
    )]
    pub from_uuid: String,
    /// Target neuron identity string. Must be a UUID or descriptive identifier.
    /// Purely numeric integer IDs are rejected (Issue #952).
    #[serde(
        default,
        alias = "toUUID",
        deserialize_with = "deserialise_synapse_uuid"
    )]
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
    /// Neuron identity string. Must match the corresponding [`NeuronJson::uuid`]
    /// exactly. Purely numeric integer IDs are rejected (Issue #952).
    #[serde(deserialize_with = "deserialise_neuron_uuid")]
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

/// JSON-serialised summary statistics for a single neuron — error / activation
/// means and variances plus spike and activation-range counters. Attached to
/// candidate payloads so the host can reason about target-neuron behaviour
/// without re-reading parquet.
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
