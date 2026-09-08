//! The composed creature-validation gate for the FFI boundary (Issue #2046).
//!
//! Every FFI entry point that accepts a `CreatureJson` must run **both**
//! defence-in-depth gates — [`validate_forward_only_synapses`] (Issue #1184)
//! and [`validate_creature_input_bounds`] (Issues #1867, #2020) — immediately
//! after deserialisation and before any business logic. That composition was
//! re-typed as a two-call `and_then` chain at all five call sites, so the
//! invariant depended on every future entry point copying the pair, and the
//! order, correctly.
//!
//! [`validate_creature`] is that composition, expressed once. A new entry
//! point calls one function; the pair and its order cannot drift.

use super::{
    CreatureJson, DiscoveryError, validate_creature_input_bounds, validate_forward_only_synapses,
};

/// Run the full creature-validation gate on `creature`.
///
/// Checks the forward-only synapse invariant first, then the observation-width
/// and input-count bounds. The order is deliberate: a back-edge is a topology
/// corruption that describes the creature as a whole, whereas the bounds fault
/// names a single field, so reporting the topology fault first gives the
/// caller the more diagnostic message when a creature violates both.
///
/// Returns `DiscoveryError::InvalidInput` — classified as `data_validation` —
/// for either violation, so every entry point's response shape is unchanged.
pub fn validate_creature(creature: &CreatureJson) -> Result<(), DiscoveryError> {
    validate_forward_only_synapses(creature).and_then(|()| validate_creature_input_bounds(creature))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi_types::{DiscoveryErrorKind, NeuronJson, SynapseJson};

    fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
        serde_json::from_value(serde_json::json!({
            "uuid": uuid,
            "type": neuron_type,
            "squash": "IDENTITY",
            "bias": 0.0
        }))
        .expect("valid neuron JSON")
    }

    fn synapse(from: &str, to: &str) -> SynapseJson {
        serde_json::from_value(serde_json::json!({
            "fromUUID": from,
            "toUUID": to,
            "weight": 0.5
        }))
        .expect("valid synapse JSON")
    }

    fn creature() -> CreatureJson {
        CreatureJson {
            neurons: vec![neuron("hidden-0", "hidden"), neuron("output-0", "output")],
            synapses: vec![synapse("hidden-0", "output-0")],
            input: 2,
            output: 1,
        }
    }

    #[test]
    fn accepts_a_forward_only_in_bounds_creature() {
        validate_creature(&creature()).expect("an ordinary creature must validate");
    }

    #[test]
    fn rejects_a_back_edge() {
        let mut c = creature();
        c.synapses = vec![synapse("output-0", "hidden-0")];
        let err = validate_creature(&c).expect_err("a back-edge must be rejected");
        assert_eq!(err.error_kind(), DiscoveryErrorKind::DataValidation);
        assert!(
            err.to_string().contains("forward-only"),
            "error must name the forward-only invariant: {err}"
        );
    }

    #[test]
    fn rejects_an_oversized_input_count() {
        let mut c = creature();
        c.input = usize::MAX;
        let err = validate_creature(&c).expect_err("an unbounded input count must be rejected");
        assert_eq!(err.error_kind(), DiscoveryErrorKind::DataValidation);
        assert!(
            err.to_string().contains("Issue #1867"),
            "error must cite the input-bound issue: {err}"
        );
    }

    #[test]
    fn rejects_a_zero_observation_width() {
        let mut c = creature();
        c.output = 0;
        let err = validate_creature(&c).expect_err("a zero output width must be rejected");
        assert!(
            err.to_string().contains("Issue #2020"),
            "error must cite the observation-width issue: {err}"
        );
    }

    /// The forward-only check runs first, so a creature violating both
    /// invariants reports the topology fault rather than the bound.
    #[test]
    fn reports_the_forward_only_fault_before_the_bound() {
        let mut c = creature();
        c.synapses = vec![synapse("output-0", "hidden-0")];
        c.input = usize::MAX;
        let err = validate_creature(&c).expect_err("both invariants are violated");
        assert!(
            err.to_string().contains("forward-only"),
            "the forward-only check must run first: {err}"
        );
    }
}
