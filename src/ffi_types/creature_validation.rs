//! The composed creature-validation gate for the FFI boundary (Issue #2046).
//!
//! Every FFI entry point that accepts a `CreatureJson` must run **all** the
//! defence-in-depth gates — [`validate_forward_only_synapses`] (Issue #1184),
//! [`validate_creature_input_bounds`] (Issues #1867, #2020) and
//! [`validate_neuron_biases`] (Issue #2133) — immediately after
//! deserialisation and before any business logic. That composition was once
//! re-typed as an `and_then` chain at all five call sites, so the invariant
//! depended on every future entry point copying the set, and the order,
//! correctly.
//!
//! [`validate_creature`] is that composition, expressed once. A new entry
//! point calls one function; the set and its order cannot drift.

use super::{
    CreatureJson, DiscoveryError, validate_creature_input_bounds, validate_forward_only_synapses,
    validate_neuron_biases,
};

/// Run the full creature-validation gate on `creature`.
///
/// Checks the forward-only synapse invariant first, then the observation-width
/// and input-count bounds, then neuron-bias finitude. The order is deliberate
/// and runs from widest scope to narrowest: a back-edge is a topology
/// corruption that describes the creature as a whole, the bounds fault names a
/// single top-level field, and a non-finite bias names one neuron. Reporting
/// the widest fault first gives the caller the most diagnostic message when a
/// creature violates more than one.
///
/// Returns `DiscoveryError::InvalidInput` — classified as `data_validation` —
/// for any violation, so every entry point's response shape is unchanged.
pub fn validate_creature(creature: &CreatureJson) -> Result<(), DiscoveryError> {
    validate_forward_only_synapses(creature)
        .and_then(|()| validate_creature_input_bounds(creature))
        .and_then(|()| validate_neuron_biases(creature))
}
