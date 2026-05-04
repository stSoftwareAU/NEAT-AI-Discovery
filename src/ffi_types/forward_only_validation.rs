//! Forward-only synapse validation at the FFI boundary (Issue #1184).
//!
//! Discovery requires forward-only creatures: every synapse must point from
//! an earlier neuron to a later neuron in the creature's evaluation order
//! (see `AGENTS.md` — Forward-only Activation Order). Recurrent or
//! self-looping synapses produce undefined behaviour in the analysis
//! pipeline because [`crate::analysis::synapse::target_analysis`] silently
//! filters sources by `neuron.index < target_index`, dropping any back-edges
//! without diagnostics.
//!
//! Issue #1184 mirrors a corruption sweep observed in `node-sloth.log` where
//! 28 `output-0 -> output-0` self-loops survived the upstream `loadFrom`
//! strip in NEAT-AI. The strip is only a warn-and-continue, so corrupt
//! creatures could silently enter discovery. This module is the
//! defence-in-depth gate: it rejects offending creatures with a structured
//! `DiscoveryError::InvalidInput` so the controller can fail fast and the
//! upstream producer can be diagnosed.
//!
//! The check runs on `CreatureJson` after JSON deserialisation but before
//! any business logic, so every FFI entry point that accepts a creature is
//! protected by a single call.

use std::collections::HashMap;

use super::{CreatureJson, DiscoveryError};

/// Maximum number of offending synapses included in the error message, to
/// keep the JSON response bounded when a corrupt creature carries hundreds
/// of bad edges.
const MAX_REPORTED_VIOLATIONS: usize = 5;

/// Verify that every synapse in `creature` respects the forward-only
/// activation ordering invariant (Issue #1184).
///
/// A synapse `from -> to` is forward-only when the source neuron's index in
/// the creature's `neurons` vector is strictly less than the target
/// neuron's index. Self-loops (`from == to`) and back-edges are recurrent
/// and must not enter the discovery pipeline.
///
/// Returns `Ok(())` when every synapse is forward-only. Returns
/// `DiscoveryError::InvalidInput` with a descriptive message listing up
/// to a bounded number of offending synapses (see the module-private
/// `MAX_REPORTED_VIOLATIONS` constant) when violations are found.
///
/// Synapses referencing neuron UUIDs that are not present in the creature
/// are **not** flagged here — the existing pipeline tolerates such
/// dangling references (the downstream `target_analysis` filter drops
/// them). The scope of this gate is narrowly the forward-only invariant
/// from Issue #1184, where both endpoints are present but the ordering
/// is reversed (or identical for a self-loop).
pub fn validate_forward_only_synapses(creature: &CreatureJson) -> Result<(), DiscoveryError> {
    let mut index_by_uuid: HashMap<&str, usize> = HashMap::with_capacity(creature.neurons.len());
    for (idx, neuron) in creature.neurons.iter().enumerate() {
        index_by_uuid.insert(neuron.uuid.as_str(), idx);
    }

    let mut violations: Vec<String> = Vec::new();
    let mut total_violations: usize = 0;

    for synapse in &creature.synapses {
        // Only validate when both endpoints are resolvable. Dangling
        // references are out of scope for Issue #1184 — they are handled
        // (silently filtered) by the existing analysis pipeline.
        let (Some(&from_idx), Some(&to_idx)) = (
            index_by_uuid.get(synapse.from_uuid.as_str()),
            index_by_uuid.get(synapse.to_uuid.as_str()),
        ) else {
            continue;
        };

        if from_idx < to_idx {
            continue;
        }

        total_violations += 1;
        if violations.len() < MAX_REPORTED_VIOLATIONS {
            let detail = if from_idx == to_idx {
                format!(
                    "self-loop {} -> {} at index {}",
                    synapse.from_uuid, synapse.to_uuid, from_idx
                )
            } else {
                format!(
                    "back-edge {} (index {}) -> {} (index {})",
                    synapse.from_uuid, from_idx, synapse.to_uuid, to_idx
                )
            };
            violations.push(detail);
        }
    }

    if total_violations == 0 {
        return Ok(());
    }

    let extra = total_violations.saturating_sub(violations.len());
    let suffix = if extra > 0 {
        format!(" (+{extra} more)")
    } else {
        String::new()
    };

    Err(DiscoveryError::InvalidInput {
        detail: format!(
            "creature has {total_violations} recurrent synapse(s) violating \
             the forward-only invariant: {}{suffix}. Discovery requires \
             forward-only networks (Issue #1184).",
            violations.join("; ")
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi_types::{NeuronJson, SynapseJson};

    fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: neuron_type.to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }
    }

    fn synapse(from: &str, to: &str) -> SynapseJson {
        SynapseJson {
            from_uuid: from.to_string(),
            to_uuid: to.to_string(),
            weight: 0.5,
            synapse_type: None,
        }
    }

    fn creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
        CreatureJson {
            neurons,
            synapses,
            input: 1,
            output: 1,
        }
    }

    #[test]
    fn forward_only_creature_is_accepted() {
        let c = creature(
            vec![
                neuron("input-0", "input"),
                neuron("hidden-0", "hidden"),
                neuron("output-0", "output"),
            ],
            vec![
                synapse("input-0", "hidden-0"),
                synapse("hidden-0", "output-0"),
                synapse("input-0", "output-0"),
            ],
        );
        validate_forward_only_synapses(&c).expect("forward-only creature must validate");
    }

    #[test]
    fn output_self_loop_is_rejected() {
        // Mirrors the GRQ-10 sweep pattern: output-0 -> output-0 self-loops
        // on a forward-only creature.
        let c = creature(
            vec![neuron("input-0", "input"), neuron("output-0", "output")],
            vec![
                synapse("input-0", "output-0"),
                synapse("output-0", "output-0"),
            ],
        );
        let err =
            validate_forward_only_synapses(&c).expect_err("output self-loop must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("self-loop"),
            "msg should mention self-loop: {msg}"
        );
        assert!(
            msg.contains("output-0"),
            "msg should name the offending neuron: {msg}"
        );
        assert!(
            msg.contains("Issue #1184"),
            "msg should cite the tracking issue: {msg}"
        );
    }

    #[test]
    fn back_edge_between_distinct_neurons_is_rejected() {
        let c = creature(
            vec![
                neuron("input-0", "input"),
                neuron("hidden-0", "hidden"),
                neuron("hidden-1", "hidden"),
                neuron("output-0", "output"),
            ],
            // hidden-1 -> hidden-0 is a back-edge (index 2 -> 1).
            vec![synapse("hidden-1", "hidden-0")],
        );
        let err = validate_forward_only_synapses(&c).expect_err("back-edge must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("back-edge"),
            "msg should mention back-edge: {msg}"
        );
    }

    #[test]
    fn unknown_source_neuron_is_tolerated() {
        // Dangling references are out of scope for Issue #1184 — the
        // existing pipeline silently filters them. The validation only
        // catches forward-only violations between resolvable endpoints.
        let c = creature(
            vec![neuron("input-0", "input"), neuron("output-0", "output")],
            vec![synapse("ghost", "output-0")],
        );
        validate_forward_only_synapses(&c).expect("dangling reference must not be rejected");
    }

    #[test]
    fn many_violations_are_truncated_with_extra_count() {
        let mut neurons = vec![neuron("input-0", "input")];
        let mut synapses = Vec::new();
        // Build 8 output neurons each with a self-loop (8 violations).
        for i in 0..8 {
            let uuid = format!("output-{i}");
            neurons.push(neuron(&uuid, "output"));
            synapses.push(synapse(&uuid, &uuid));
        }
        let c = creature(neurons, synapses);
        let err =
            validate_forward_only_synapses(&c).expect_err("multiple self-loops must be rejected");
        let msg = err.to_string();
        // 5 reported, 3 hidden behind a `(+3 more)` suffix.
        assert!(msg.contains("8 recurrent"), "should report total: {msg}");
        assert!(msg.contains("(+3 more)"), "should truncate excess: {msg}");
    }

    #[test]
    fn empty_synapse_list_is_accepted() {
        let c = creature(
            vec![neuron("input-0", "input"), neuron("output-0", "output")],
            Vec::new(),
        );
        validate_forward_only_synapses(&c).expect("creature with no synapses must validate");
    }

    #[test]
    fn rejection_classifies_as_data_validation() {
        let c = creature(
            vec![neuron("output-0", "output")],
            vec![synapse("output-0", "output-0")],
        );
        let err = validate_forward_only_synapses(&c).expect_err("self-loop must be rejected");
        // The downstream FFI layer relies on this kind to set
        // `error_kind: "data_validation"` on the response.
        assert_eq!(
            err.error_kind(),
            crate::ffi_types::DiscoveryErrorKind::DataValidation
        );
    }
}
