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
//!
//! The same gate enforces the **lower** bound (Issue #2020): `input < 1` or
//! `output < 1` is never accepted anywhere. The top-level counts are the
//! observation width and cannot be re-derived — `neurons` carries no input
//! neurons — so a zero is a corrupt creature, not an empty one. The
//! `CreatureJson` serde impl already rejects it at deserialisation and refuses
//! to emit it at serialisation; this gate is the belt-and-braces check for
//! creatures constructed in Rust and handed to an entry point directly.

use super::{CreatureJson, DiscoveryError};

/// Return the Issue #2020 rejection message when an observation-width count
/// (`input` or `output`) is below one, or `None` when the count is valid.
///
/// The wording mirrors the TypeScript reference (`CreatureValidate.ts`):
/// `Must have at least one input neurons was: N`.
#[must_use]
pub fn observation_width_error(field: &str, count: usize) -> Option<String> {
    (count < 1).then(|| {
        format!(
            "Must have at least one {field} neurons was: {count} (Issue #2020). \
             The creature's top-level `{field}` count is the observation width \
             and cannot be derived from `neurons`."
        )
    })
}

/// Maximum accepted `creature.input` (inclusive).
///
/// One million input neurons already far exceeds any real NEAT creature —
/// the recording path alone would hold ~50 MB of `input-N` UUID strings at
/// this ceiling. The cap exists to keep a corrupt count from turning into an
/// allocation abort, so it is deliberately generous rather than tuned.
pub const MAX_CREATURE_INPUT_NEURONS: usize = 1_000_000;

/// Verify that `creature.input` is within `1..=`[`MAX_CREATURE_INPUT_NEURONS`]
/// and `creature.output >= 1` (Issues #1867, #2020).
///
/// Runs at every FFI entry point that accepts a `CreatureJson`, immediately
/// after JSON deserialisation and before any business logic, alongside
/// [`super::validate_forward_only_synapses`].
///
/// Returns `DiscoveryError::InvalidInput` — classified as
/// `data_validation` — when either count is below one or the input count
/// exceeds the cap.
pub fn validate_creature_input_bounds(creature: &CreatureJson) -> Result<(), DiscoveryError> {
    // Issue #2020: the lower bound comes first — a zero width is the more
    // fundamental corruption and its message names the field.
    if let Some(detail) = observation_width_error("input", creature.input)
        .or_else(|| observation_width_error("output", creature.output))
    {
        return Err(DiscoveryError::InvalidInput { detail });
    }

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
    fn zero_inputs_is_rejected_as_data_validation() {
        // Issue #2020: `input < 1` is never accepted — the count is the
        // observation width and cannot be derived from `neurons`.
        let err = validate_creature_input_bounds(&creature(0, 1))
            .expect_err("zero inputs must be rejected");
        assert_eq!(err.error_kind(), DiscoveryErrorKind::DataValidation);
        let msg = err.to_string();
        assert!(
            msg.contains("Must have at least one input neurons was: 0"),
            "msg should mirror the TS reference wording: {msg}"
        );
    }

    #[test]
    fn zero_outputs_is_rejected_as_data_validation() {
        let mut zero_output = creature(2, 1);
        zero_output.output = 0;
        let err = validate_creature_input_bounds(&zero_output)
            .expect_err("zero outputs must be rejected");
        assert_eq!(err.error_kind(), DiscoveryErrorKind::DataValidation);
        let msg = err.to_string();
        assert!(
            msg.contains("Must have at least one output neurons was: 0"),
            "msg should mirror the TS reference wording: {msg}"
        );
    }

    #[test]
    fn observation_width_error_only_fires_below_one() {
        assert!(observation_width_error("input", 0).is_some());
        assert!(observation_width_error("output", 0).is_some());
        assert!(observation_width_error("input", 1).is_none());
        assert!(observation_width_error("output", usize::MAX).is_none());
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
