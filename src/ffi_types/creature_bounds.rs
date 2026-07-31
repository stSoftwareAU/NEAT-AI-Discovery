//! Creature input-count bounds at the FFI boundary (Issue #1867).
//!
//! `CreatureJson.input` arrives as a bare `usize` deserialised from
//! caller-supplied JSON and nothing downstream bounded it. It drives three
//! size computations in the recording path plus per-input map allocations in
//! the analysis path, so `{"input": 10000000000}` asked the library to
//! materialise ten billion `String`s. That is not a fallible
//! `Vec::try_reserve` — allocation failure routes through
//! `handle_alloc_error`, which **aborts**, and the `panic::catch_unwind`
//! wrapping every FFI entry point cannot intercept an abort. The host
//! process died instead of receiving an error response.
//!
//! Creature models are loaded from disk and synchronised between fleet
//! hosts, so a corrupt, truncated or hostile model file reaches this path
//! without any host compromise — the same threat model that motivated
//! [`super::validate_forward_only_synapses`] (Issue #1184).
//!
//! The bound is an absolute cap, **not** `creature.neurons.len()`: input
//! neurons are implied by the count and are not listed in
//! `creature.neurons` (see the documented `record_discovery` payload in
//! `docs/FFI_API.md`, which pairs `"input": 20` with a single listed
//! neuron), so a neuron-count-relative bound would reject ordinary
//! creatures.

use super::{CreatureJson, DiscoveryError};

/// Maximum accepted `creature.input` (inclusive).
///
/// One million input neurons already far exceeds any real NEAT creature —
/// the recording path alone would hold ~50 MB of `input-N` UUID strings at
/// this ceiling. The cap exists to keep a corrupt count from turning into an
/// allocation abort, so it is deliberately generous rather than tuned.
pub const MAX_CREATURE_INPUT_NEURONS: usize = 1_000_000;

/// Verify that `creature.input` is within [`MAX_CREATURE_INPUT_NEURONS`]
/// (Issue #1867).
///
/// Runs at every FFI entry point that accepts a `CreatureJson`, immediately
/// after JSON deserialisation and before any business logic, alongside
/// [`super::validate_forward_only_synapses`].
///
/// Returns `DiscoveryError::InvalidInput` — classified as
/// `data_validation` — when the count exceeds the cap.
pub fn validate_creature_input_bounds(creature: &CreatureJson) -> Result<(), DiscoveryError> {
    if creature.input > MAX_CREATURE_INPUT_NEURONS {
        return Err(DiscoveryError::InvalidInput {
            detail: format!(
                "creature declares {} input neurons, exceeding the maximum of {} \
                 (Issue #1867). An input count this large is a corrupt creature: \
                 it would drive an unbounded allocation that aborts the process \
                 rather than returning an error.",
                creature.input, MAX_CREATURE_INPUT_NEURONS
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi_types::{DiscoveryErrorKind, NeuronJson};

    fn creature(input: usize, neuron_count: usize) -> CreatureJson {
        CreatureJson {
            neurons: (0..neuron_count)
                .map(|i| NeuronJson {
                    uuid: format!("hidden-{i}"),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                })
                .collect(),
            synapses: Vec::new(),
            input,
            output: 1,
        }
    }

    #[test]
    fn ordinary_creature_is_accepted() {
        validate_creature_input_bounds(&creature(2, 2)).expect("ordinary creature must validate");
    }

    #[test]
    fn input_count_above_neuron_count_is_accepted() {
        // Input neurons are not listed in `neurons` — the documented payload
        // pairs 20 inputs with one listed neuron.
        validate_creature_input_bounds(&creature(20, 1))
            .expect("wide-input creature must validate");
    }

    #[test]
    fn zero_inputs_is_accepted() {
        validate_creature_input_bounds(&creature(0, 1)).expect("zero inputs must validate");
    }

    #[test]
    fn limit_is_inclusive() {
        validate_creature_input_bounds(&creature(MAX_CREATURE_INPUT_NEURONS, 1))
            .expect("the limit itself must validate");
    }

    #[test]
    fn above_limit_is_rejected_as_data_validation() {
        let err = validate_creature_input_bounds(&creature(MAX_CREATURE_INPUT_NEURONS + 1, 1))
            .expect_err("above the limit must be rejected");
        assert_eq!(err.error_kind(), DiscoveryErrorKind::DataValidation);
        let msg = err.to_string();
        assert!(
            msg.contains("Issue #1867"),
            "msg should cite the tracking issue: {msg}"
        );
    }

    #[test]
    fn usize_max_is_rejected() {
        validate_creature_input_bounds(&creature(usize::MAX, 1))
            .expect_err("usize::MAX must be rejected");
    }
}
