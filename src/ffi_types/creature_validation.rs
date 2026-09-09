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
