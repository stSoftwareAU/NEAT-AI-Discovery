//! FFI boundary types — JSON request/response structs.
//!
//! All types in this module are used to serialise and deserialise data at the
//! FFI boundary between this Rust library and the TypeScript/Deno controller.

mod candidates;
mod cleanup;
mod creature_bounds;
mod creature_validation;
mod error_classification;
mod forward_only_validation;
mod neuron_bias;
mod requests;
mod responses;
mod session;

// Re-export all sub-module types to preserve the existing public API.
pub use candidates::*;
pub use cleanup::*;
pub use creature_bounds::{
    MAX_CREATURE_INPUT_NEURONS, MAX_CREATURE_OUTPUT_NEURONS, observation_width_error,
    validate_creature_input_bounds,
};
pub use creature_validation::validate_creature;
pub use error_classification::*;
pub use forward_only_validation::validate_forward_only_synapses;
// Crate-internal (Issue #2133): the bias gate is reached through
// `validate_creature`, so it adds no name to the public FFI surface.
pub(crate) use neuron_bias::{non_finite_bias_detail, validate_neuron_biases};
pub use requests::*;
pub use responses::*;
pub use session::*;

use serde::ser::Error as SerialiseError; // codespell:ignore ser
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
/// be repaired downstream, so a width-less creature is an error at the point
/// of emission rather than a payload the host has to reject later.
fn serialise_input_width<S>(value: &usize, serialiser: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if let Some(detail) = observation_width_error("input", *value) {
        return Err(SerialiseError::custom(detail));
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
        return Err(SerialiseError::custom(detail));
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
    /// Neuron bias. Defaults to zero when omitted, and must be finite: a
    /// non-finite bias is rejected at deserialisation and refused at
    /// serialisation (Issue #2133).
    #[serde(
        default,
        deserialize_with = "deserialise_neuron_bias",
        serialize_with = "serialise_neuron_bias"
    )]
    pub bias: f32,
}

/// Deserialise a neuron's `bias`, rejecting Infinity and `NaN` (Issue #2133).
///
/// `f32::deserialize` performs the `f64` → `f32` narrowing itself, so a JSON
/// number that is finite as an `f64` but too large for an `f32` (`1e39`)
/// arrives here already collapsed to an infinity — which is exactly the value
/// this check refuses. Doing the narrowing through serde rather than a cast of
/// our own keeps the helper free of an unchecked `as`.
fn deserialise_neuron_bias<'de, D>(deserialiser: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let bias = f32::deserialize(deserialiser)?;
    if !bias.is_finite() {
        return Err(serde::de::Error::custom(non_finite_bias_detail(
            "The neuron",
            bias,
        )));
    }
    Ok(bias)
}

/// Serialise a neuron's `bias`, refusing to emit Infinity or `NaN`
/// (Issue #2133).
///
/// `serde_json` renders a non-finite `f32` as JSON `null`, which a host reading
/// the snapshot back cannot distinguish from an omitted bias — the corruption
/// would arrive silently as a zero. A neuron built in Rust therefore fails at
/// the point of emission rather than downstream.
fn serialise_neuron_bias<S>(value: &f32, serialiser: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if !value.is_finite() {
        return Err(SerialiseError::custom(non_finite_bias_detail(
            "The neuron",
            *value,
        )));
    }
    value.serialize(serialiser)
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

// ==== Synapse weight finitude validation (Issue #2132) ====

/// Deserialise a synapse weight, rejecting any non-finite value (Issue #2132).
///
/// JSON has no `Infinity` literal, but any magnitude above `f32::MAX` — such as
/// `1e39` — saturates to `f32::INFINITY` when serde casts the parsed f64 down,
/// and serde raises nothing. An infinite weight then poisons every downstream
/// analysis: `weight * activation` stays infinite, so dormancy, polarity-flip
/// and noise thresholds are all silently bypassed and the caller receives
/// incorrect results. Reject at the boundary, once, rather than re-guarding
/// each of the consumption sites.
fn deserialise_synapse_weight<'de, D>(deserialiser: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = f32::deserialize(deserialiser)?;
    if !raw.is_finite() {
        return Err(serde::de::Error::custom(format!(
            "synapse weight must be finite, got {raw} (Issue #2132). \
             Infinity and NaN are not permitted in FFI payloads."
        )));
    }
    Ok(raw)
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
    /// Connection weight. Must be finite — Infinity and NaN are rejected at the
    /// FFI boundary (Issue #2132).
    #[serde(default, deserialize_with = "deserialise_synapse_weight")]
    pub weight: f32,
    /// Synapse type for IF neurons: "condition", "positive", or "negative".
    /// Used to determine which synapses contribute to condition evaluation
    /// versus positive/negative branches. None for non-IF neurons.
    #[serde(default, rename = "type")]
    pub synapse_type: Option<String>,
}

// ==== NeuronData float finitude validation (Issue #2134) ====

/// The Issue #2134 rejection message for a non-finite `NeuronData` float.
fn non_finite_neuron_data_detail(field: &str, raw: f32) -> String {
    format!(
        "neuron data {field} must be finite, got {raw} (Issue #2134). \
         Infinity and NaN are not permitted in FFI payloads."
    )
}

/// Deserialise a recorded float, rejecting any non-finite value (Issue #2134).
///
/// A JSON magnitude above `f32::MAX` — such as `1e39` — is a perfectly ordinary
/// `f64`, so `serde_json` accepts it and the narrowing cast to `f32` saturates
/// to infinity without complaint. That infinity then flows into every analysis
/// site: means, variances and covariances become Infinity or NaN, and target
/// values (`activation + error`) become Infinity, so the caller is handed
/// confidently wrong metrics. Reject at the boundary, once, rather than
/// re-guarding each consumption site.
fn deserialise_finite_f32<'de, D>(deserialiser: D, field: &str) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = f32::deserialize(deserialiser)?;
    if !raw.is_finite() {
        return Err(serde::de::Error::custom(non_finite_neuron_data_detail(
            field, raw,
        )));
    }
    Ok(raw)
}

fn deserialise_activation<'de, D>(deserialiser: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    deserialise_finite_f32(deserialiser, "activation")
}

/// An absent or explicitly null `value` stays `None`; a present one must be
/// finite (Issue #2134).
fn deserialise_optional_value<'de, D>(deserialiser: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(raw) = Option::<f32>::deserialize(deserialiser)? else {
        return Ok(None);
    };
    if !raw.is_finite() {
        return Err(serde::de::Error::custom(non_finite_neuron_data_detail(
            "value", raw,
        )));
    }
    Ok(Some(raw))
}

/// Every element is checked, not merely the first — one poisoned entry anywhere
/// in the vector is enough to corrupt the error statistics (Issue #2134).
fn deserialise_finite_errors<'de, D>(deserialiser: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = Vec::<f32>::deserialize(deserialiser)?;
    if let Some(bad) = raw.iter().copied().find(|e| !e.is_finite()) {
        return Err(serde::de::Error::custom(non_finite_neuron_data_detail(
            "errors", bad,
        )));
    }
    Ok(raw)
}

/// Pre-computed neuron data for a single neuron
#[derive(Debug, Deserialize, Clone)]
pub struct NeuronData {
    /// Neuron identity string. Must match the corresponding [`NeuronJson::uuid`]
    /// exactly. Purely numeric integer IDs are rejected (Issue #952).
    #[serde(deserialize_with = "deserialise_neuron_uuid")]
    pub neuron_uuid: String,
    /// Recorded activation. Must be finite — Infinity and NaN are rejected at
    /// the FFI boundary (Issue #2134).
    #[serde(deserialize_with = "deserialise_activation")]
    pub activation: f32,
    /// Recorded pre-squash value, when the host supplies one. Must be finite
    /// when present (Issue #2134).
    #[serde(default, deserialize_with = "deserialise_optional_value")]
    pub value: Option<f32>,
    /// Per-sample errors. Every element must be finite (Issue #2134).
    #[serde(deserialize_with = "deserialise_finite_errors")]
    pub errors: Vec<f32>,
}

// ==== TrainingRecord vector finitude validation (Issue #2135) ====

/// The Issue #2135 rejection message for a non-finite training vector element.
fn non_finite_training_vector_detail(field: &str, index: usize, raw: f32) -> String {
    format!(
        "training record {field}[{index}] must be finite, got {raw} (Issue #2135). \
         Infinity and NaN are not permitted in FFI payloads."
    )
}

/// Deserialise a training feature or target vector, rejecting any non-finite
/// element (Issue #2135).
///
/// Every element is checked, not merely the first — one poisoned entry anywhere
/// in the vector is enough to make the covariance and correlation matrices built
/// over the records Infinity or NaN, so the caller is handed confidently wrong
/// training metrics. Rejecting here, once, protects every consumption site by
/// construction rather than re-guarding each of them.
fn deserialise_finite_training_vector<'de, D>(
    deserialiser: D,
    field: &str,
) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = Vec::<f32>::deserialize(deserialiser)?;
    if let Some((index, bad)) = raw.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(serde::de::Error::custom(non_finite_training_vector_detail(
            field, index, bad,
        )));
    }
    Ok(raw)
}

fn deserialise_training_input<'de, D>(deserialiser: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialise_finite_training_vector(deserialiser, "input")
}

fn deserialise_training_output<'de, D>(deserialiser: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialise_finite_training_vector(deserialiser, "output")
}

/// Training data record
#[derive(Debug, Deserialize, Clone)]
pub struct TrainingRecord {
    /// Feature vector. Every element must be finite — Infinity and NaN are
    /// rejected at the FFI boundary (Issue #2135).
    #[serde(deserialize_with = "deserialise_training_input")]
    pub input: Vec<f32>,
    /// Target vector. Every element must be finite (Issue #2135).
    #[serde(deserialize_with = "deserialise_training_output")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde::de::IntoDeserializer;
    use serde::de::value::{Error as ValueError, F32Deserializer, SeqDeserializer};

    /// NaN cannot be written in JSON, so the JSON-reachable cases live in
    /// `tests/ffi/issue_2132_synapse_weight_finitude.rs`. Reaching the NaN
    /// branch needs a non-JSON deserialiser feeding the module-private helper
    /// directly, which only an in-crate test can do (CONTRIBUTING: place a test
    /// under `src/` only when the public API cannot exercise the behaviour).
    #[test]
    fn deserialise_synapse_weight_rejects_nan() {
        let deserialiser: F32Deserializer<ValueError> = f32::NAN.into_deserializer();
        let err = deserialise_synapse_weight(deserialiser)
            .expect_err("a NaN weight must be rejected (Issue #2132)");
        let msg = err.to_string();
        assert!(msg.contains("finite"), "error must name finitude: {msg}");
        assert!(
            msg.contains("Issue #2132"),
            "error must cite the issue: {msg}"
        );
    }

    /// NaN is unreachable from JSON, so the JSON-reachable cases live in
    /// `tests/ffi/issue_2134_neuron_data_finitude.rs`; only an in-crate test can
    /// feed NaN to the module-private helpers directly.
    #[test]
    fn deserialise_neuron_data_floats_reject_nan() {
        // `value` is omitted here: serde's primitive deserialisers cannot answer
        // `deserialize_option`, so there is no way to hand NaN to the optional
        // helper. Its finitude check shares `non_finite_neuron_data_detail` with
        // the two below and is exercised over JSON by the integration test.
        let nan: F32Deserializer<ValueError> = f32::NAN.into_deserializer();
        let msg = deserialise_activation(nan)
            .expect_err("a NaN activation must be rejected")
            .to_string();
        assert!(
            msg.contains("finite") && msg.contains("Issue #2134"),
            "activation must name finitude and cite the issue: {msg}"
        );

        let errors: SeqDeserializer<std::vec::IntoIter<f32>, ValueError> =
            vec![-0.05_f32, f32::NAN].into_deserializer();
        let err = deserialise_finite_errors(errors)
            .expect_err("a NaN error element must be rejected")
            .to_string();
        assert!(
            err.contains("Issue #2134") && err.contains("errors"),
            "errors must name the field and cite the issue: {err}"
        );
    }

    /// NaN is unreachable from JSON, so the JSON-reachable cases live in
    /// `tests/ffi/issue_2135_training_record_finitude.rs`; only an in-crate test
    /// can feed NaN to the module-private helpers directly.
    #[test]
    fn deserialise_training_vectors_reject_nan() {
        for (field, deserialise) in [
            (
                "input",
                deserialise_training_input
                    as fn(
                        SeqDeserializer<std::vec::IntoIter<f32>, ValueError>,
                    ) -> Result<Vec<f32>, ValueError>,
            ),
            ("output", deserialise_training_output),
        ] {
            let vector: SeqDeserializer<std::vec::IntoIter<f32>, ValueError> =
                vec![0.5_f32, f32::NAN].into_deserializer();
            let err = deserialise(vector)
                .expect_err("a NaN vector element must be rejected")
                .to_string();
            assert!(
                err.contains("Issue #2135") && err.contains(&format!("{field}[1]")),
                "{field} must name the offending index and cite the issue: {err}"
            );
        }
    }

    #[test]
    fn deserialise_synapse_weight_rejects_infinities() {
        for weight in [f32::INFINITY, f32::NEG_INFINITY] {
            let deserialiser: F32Deserializer<ValueError> = weight.into_deserializer();
            let err = deserialise_synapse_weight(deserialiser)
                .expect_err("an infinite weight must be rejected")
                .to_string();
            assert!(
                err.contains("Issue #2132"),
                "error for {weight} must cite the issue: {err}"
            );
        }
    }

    #[test]
    fn deserialise_synapse_weight_accepts_finite_values() {
        for weight in [0.0_f32, -0.75, f32::MAX, f32::MIN] {
            let deserialiser: F32Deserializer<ValueError> = weight.into_deserializer();
            let parsed = deserialise_synapse_weight(deserialiser)
                .unwrap_or_else(|e| panic!("finite weight {weight} must be accepted: {e}"));
            assert_eq!(
                parsed, weight,
                "a finite weight must pass through unchanged"
            );
        }
    }
}
