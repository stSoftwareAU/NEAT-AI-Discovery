//! Neuron-bias finitude at the FFI boundary (Issue #2133).
//!
//! `NeuronJson::bias` carried only `#[serde(default)]`, so nothing checked that
//! the value was finite. Two doors were open:
//!
//! * **The narrowing cast.** serde hands `serde_json`'s `f64` to the `f32`
//!   visitor, which casts. `1e39` is an ordinary finite JSON number and an
//!   ordinary finite `f64`, but it overflows `f32` — the cast silently yields
//!   `f32::INFINITY`.
//! * **Creatures built in Rust.** A `CreatureJson` constructed in Rust and
//!   handed straight to an entry point never passes through serde at all, so it
//!   can carry `f32::NAN` — for which JSON has no literal.
//!
//! A non-finite bias then reaches `neuron.bias += …` in
//! `dominated_branch_collapse`, the `f64::from` fold in
//! `remove_neuron_bias_fold`, and — worst — `bias.to_bits()` hashing in
//! `neuron_fingerprint`, where a payload-bearing `NaN` makes the fingerprint
//! itself unstable.
//!
//! The defence mirrors the Issue #2020 observation-width pattern.
//! [`NeuronJson`](super::NeuronJson)'s serde impl rejects a non-finite bias at
//! deserialisation; [`validate_neuron_biases`] is the belt-and-braces gate for
//! creatures constructed in Rust and handed to an entry point directly. Both
//! layers phrase the fault through [`non_finite_bias_detail`], so the two can
//! never drift apart.

use super::{CreatureJson, DiscoveryError};

/// The Issue #2133 rejection message for a non-finite neuron bias.
///
/// `subject` names what carries the bias: the quoted neuron UUID at the
/// validation gate, or simply `The neuron` at deserialisation, where the
/// neuron's identity has not been read yet.
#[must_use]
pub fn non_finite_bias_detail(subject: &str, bias: f32) -> String {
    format!(
        "{subject} bias is not finite: {bias} (Issue #2133). A neuron bias must \
         be finite — Infinity and NaN corrupt the bias arithmetic in \
         `dominated_branch_collapse`, the `f64::from` fold in \
         `remove_neuron_bias_fold`, and the `to_bits()` neuron fingerprint hash. \
         Note that a magnitude above ~3.4e38 is finite as JSON but overflows the \
         `f32` the field is stored in."
    )
}

/// Reject a creature carrying a neuron whose bias is Infinity or `NaN`.
///
/// Returns `DiscoveryError::InvalidInput` — classified as `data_validation` —
/// naming the first offending neuron, so the caller can correct it.
///
/// # Errors
///
/// Returns an error when any neuron's `bias` is not finite.
pub fn validate_neuron_biases(creature: &CreatureJson) -> Result<(), DiscoveryError> {
    for neuron in &creature.neurons {
        if !neuron.bias.is_finite() {
            return Err(DiscoveryError::InvalidInput {
                detail: non_finite_bias_detail(
                    &format!("Neuron \"{}\"", neuron.uuid),
                    neuron.bias,
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi_types::NeuronJson;

    fn creature_with_biases(biases: &[f32]) -> CreatureJson {
        CreatureJson {
            neurons: biases
                .iter()
                .enumerate()
                .map(|(index, &bias)| NeuronJson {
                    uuid: format!("neuron-{index}"),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias,
                })
                .collect(),
            synapses: Vec::new(),
            input: 1,
            output: 1,
        }
    }

    #[test]
    fn finite_biases_pass() {
        validate_neuron_biases(&creature_with_biases(&[0.0, -2.5, f32::MAX, f32::MIN]))
            .expect("finite biases must pass the gate");
    }

    #[test]
    fn a_creature_without_neurons_passes() {
        validate_neuron_biases(&creature_with_biases(&[])).expect("no neurons, no bias fault");
    }

    #[test]
    fn an_infinite_bias_is_rejected() {
        for bias in [f32::INFINITY, f32::NEG_INFINITY] {
            let err = validate_neuron_biases(&creature_with_biases(&[0.0, bias]))
                .expect_err("an infinite bias must be rejected");
            assert!(
                err.to_string().contains("Issue #2133"),
                "error must cite the issue: {err}"
            );
        }
    }

    #[test]
    fn a_nan_bias_is_rejected_and_names_its_neuron() {
        let err = validate_neuron_biases(&creature_with_biases(&[0.0, f32::NAN]))
            .expect_err("a NaN bias must be rejected");
        assert!(
            err.to_string().contains("neuron-1"),
            "error must name the offending neuron: {err}"
        );
    }

    #[test]
    fn the_detail_names_its_subject_and_value() {
        let detail = non_finite_bias_detail("Neuron \"hidden-0\"", f32::INFINITY);
        assert!(detail.contains("hidden-0"), "subject missing: {detail}");
        assert!(detail.contains("inf"), "value missing: {detail}");
        assert!(detail.contains("Issue #2133"), "issue missing: {detail}");
    }
}
