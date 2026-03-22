//! Tests for Issue #907: Interleave hidden sources with input sources
//! during source ordering to ensure hidden-to-hidden synapse candidates
//! are evaluated even under deadline constraints.
//!
//! Previously, `order_eligible_sources()` always placed all input neurons
//! before all hidden neurons. Under tight deadlines, hidden sources were
//! rarely or never evaluated. This change interleaves hidden sources at
//! regular intervals so a meaningful proportion are always evaluated.

#![allow(clippy::cast_possible_truncation)]
use neat_ai_discovery::analysis::constants::HIDDEN_SOURCE_INTERLEAVE_INTERVAL;
use neat_ai_discovery::analysis::utils::{
    OrderedNeuron, order_eligible_sources, parse_input_index,
};
use std::collections::HashSet;

// =============================================================================
// Hidden source interleaving
// =============================================================================

/// Helper to create a mixed set of input and hidden neurons.
fn make_mixed_neurons(input_count: usize, hidden_count: usize) -> Vec<OrderedNeuron> {
    let mut neurons = Vec::with_capacity(input_count + hidden_count);
    for i in 0..input_count {
        neurons.push(OrderedNeuron {
            uuid: format!("input-{i}"),
            index: i,
        });
    }
    for i in 0..hidden_count {
        neurons.push(OrderedNeuron {
            uuid: format!("hidden-uuid-{i}"),
            index: input_count + i,
        });
    }
    neurons
}

#[test]
fn hidden_sources_are_interleaved_among_inputs() {
    // With 9 inputs and 6 hidden neurons, hidden neurons should appear
    // interleaved among inputs, not all at the end.
    let neurons = make_mixed_neurons(9, 6);
    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();

    order_eligible_sources::<String>(&mut sources, Some(42), "test", 9, None);

    // Check that at least one hidden neuron appears in the first half
    let half = sources.len() / 2;
    let hidden_in_first_half = sources[..half]
        .iter()
        .filter(|n| parse_input_index(&n.uuid).is_none())
        .count();

    assert!(
        hidden_in_first_half > 0,
        "At least one hidden neuron should appear in the first half of the ordering. \
         Order: {:?}",
        sources.iter().map(|n| &n.uuid).collect::<Vec<_>>()
    );
}

#[test]
fn all_elements_preserved_after_interleaving() {
    let neurons = make_mixed_neurons(6, 4);
    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();

    order_eligible_sources::<String>(&mut sources, Some(42), "test", 6, None);

    assert_eq!(sources.len(), 10, "All 10 neurons should be preserved");

    let input_count = sources
        .iter()
        .filter(|n| parse_input_index(&n.uuid).is_some())
        .count();
    let hidden_count = sources
        .iter()
        .filter(|n| parse_input_index(&n.uuid).is_none())
        .count();

    assert_eq!(input_count, 6, "All 6 input neurons should be preserved");
    assert_eq!(hidden_count, 4, "All 4 hidden neurons should be preserved");
}

#[test]
fn interleaving_is_deterministic_with_seed() {
    let neurons = make_mixed_neurons(6, 4);

    let mut sources1: Vec<&OrderedNeuron> = neurons.iter().collect();
    let mut sources2: Vec<&OrderedNeuron> = neurons.iter().collect();

    order_eligible_sources::<String>(&mut sources1, Some(99), "test", 6, None);
    order_eligible_sources::<String>(&mut sources2, Some(99), "test", 6, None);

    let uuids1: Vec<&str> = sources1.iter().map(|n| n.uuid.as_str()).collect();
    let uuids2: Vec<&str> = sources2.iter().map(|n| n.uuid.as_str()).collect();

    assert_eq!(uuids1, uuids2, "Same seed should produce same ordering");
}

#[test]
fn hidden_sources_evaluated_under_simulated_deadline_pressure() {
    // Simulate deadline pressure by only looking at the first N sources
    // (representing what gets evaluated before deadline expires).
    // With interleaving, hidden sources should be present in the first N.
    let neurons = make_mixed_neurons(12, 6);
    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();

    order_eligible_sources::<String>(&mut sources, Some(42), "test", 12, None);

    // Only the first 6 sources would be evaluated under tight deadlines
    let evaluated_count = 6;
    let hidden_in_evaluated = sources[..evaluated_count]
        .iter()
        .filter(|n| parse_input_index(&n.uuid).is_none())
        .count();

    assert!(
        hidden_in_evaluated > 0,
        "Hidden sources should be present among the first {evaluated_count} sources \
         evaluated under deadline pressure. Order: {:?}",
        sources
            .iter()
            .take(evaluated_count)
            .map(|n| &n.uuid)
            .collect::<Vec<_>>()
    );
}

// Compile-time validation that the interleave interval is within a reasonable range
const _: () = assert!(HIDDEN_SOURCE_INTERLEAVE_INTERVAL >= 2);
const _: () = assert!(HIDDEN_SOURCE_INTERLEAVE_INTERVAL <= 5);

#[test]
fn only_inputs_no_hidden_neurons_still_works() {
    let neurons = make_mixed_neurons(5, 0);
    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();

    order_eligible_sources::<String>(&mut sources, Some(42), "test", 5, None);

    assert_eq!(sources.len(), 5);
    for n in &sources {
        assert!(
            parse_input_index(&n.uuid).is_some(),
            "All should be input neurons"
        );
    }
}

#[test]
fn only_hidden_no_input_neurons_still_works() {
    let neurons = make_mixed_neurons(0, 5);
    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();

    order_eligible_sources::<String>(&mut sources, Some(42), "test", 0, None);

    assert_eq!(sources.len(), 5);
    for n in &sources {
        assert!(
            parse_input_index(&n.uuid).is_none(),
            "All should be hidden neurons"
        );
    }
}

#[test]
fn interleaving_consistent_across_seeds() {
    // Verify that hidden sources are interleaved regardless of seed
    let neurons = make_mixed_neurons(9, 6);

    for seed in [0u64, 1, 42, 100, 999, 12345] {
        let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();
        order_eligible_sources::<String>(&mut sources, Some(seed), "test", 9, None);

        // With 9 inputs and 6 hidden, the first 6 sources should contain
        // at least one hidden neuron due to interleaving
        let hidden_in_first_6 = sources[..6]
            .iter()
            .filter(|n| parse_input_index(&n.uuid).is_none())
            .count();

        assert!(
            hidden_in_first_6 > 0,
            "Seed {seed}: hidden sources should be interleaved in the first 6 positions. \
             Order: {:?}",
            sources.iter().map(|n| &n.uuid).collect::<Vec<_>>()
        );
    }
}

#[test]
fn focus_unused_observations_also_interleaves_hidden() {
    // When FOCUS_UNUSED_OBSERVATIONS is active and used_inputs is provided,
    // hidden neurons should still be interleaved (not all at the end)
    let neurons = make_mixed_neurons(6, 4);
    let used_inputs: HashSet<String> = ["input-0".to_string(), "input-1".to_string()].into();

    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();
    order_eligible_sources(&mut sources, Some(42), "test", 6, Some(&used_inputs));

    // All elements should be preserved
    assert_eq!(sources.len(), 10);

    // Hidden neurons should appear among the sources, not just at the very end
    let last_three = &sources[sources.len() - 3..];
    let all_hidden_at_end = last_three
        .iter()
        .all(|n| parse_input_index(&n.uuid).is_none());

    // It's acceptable for some hidden neurons to be at the end, but not ALL
    // hidden neurons should be at the very end if interleaving is working.
    // With 4 hidden neurons, at most 3 can be at the last 3 positions.
    let hidden_count = sources
        .iter()
        .filter(|n| parse_input_index(&n.uuid).is_none())
        .count();

    if hidden_count > 3 && all_hidden_at_end {
        // Check if ALL hidden neurons are at the end
        let hidden_at_end = sources[sources.len() - hidden_count..]
            .iter()
            .filter(|n| parse_input_index(&n.uuid).is_none())
            .count();
        assert!(
            hidden_at_end < hidden_count,
            "Not all hidden neurons should be pushed to the end. \
             Order: {:?}",
            sources.iter().map(|n| &n.uuid).collect::<Vec<_>>()
        );
    }
}
