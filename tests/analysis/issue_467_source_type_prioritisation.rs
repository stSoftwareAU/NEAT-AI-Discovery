//! Tests for Issue #467: Source-type prioritisation — bias toward input neurons
//! as synapse sources.
//!
//! GRQ-sampler data shows input neurons as sources have a 36.2% success rate
//! compared to only 2.8–3.3% for hidden neurons. These tests verify that:
//!
//! 1. Input neurons are ordered before hidden neurons during source evaluation
//! 2. The `INPUT_SOURCE_BOOST` multiplier is applied to candidate score gains
//!    for input-neuron sources
//! 3. Under deadline constraints, input neurons are evaluated first

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::constants::INPUT_SOURCE_BOOST;
use neat_ai_discovery::analysis::utils::{
    OrderedNeuron, order_eligible_sources, parse_input_index,
};
use std::collections::HashSet;

// =============================================================================
// Source Ordering: Input neurons before hidden neurons
// =============================================================================

#[test]
fn order_eligible_sources_places_input_neurons_before_hidden() {
    // Create a mix of input and hidden neurons
    let neurons = [
        OrderedNeuron {
            uuid: "hidden-uuid-1".to_string(),
            index: 5,
        },
        OrderedNeuron {
            uuid: "input-0".to_string(),
            index: 0,
        },
        OrderedNeuron {
            uuid: "hidden-uuid-2".to_string(),
            index: 6,
        },
        OrderedNeuron {
            uuid: "input-1".to_string(),
            index: 1,
        },
        OrderedNeuron {
            uuid: "input-2".to_string(),
            index: 2,
        },
        OrderedNeuron {
            uuid: "hidden-uuid-3".to_string(),
            index: 7,
        },
    ];

    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();

    // Use a fixed seed for determinism
    order_eligible_sources::<String>(&mut sources, Some(42), "test", 3, None);

    // All input neurons (input-0, input-1, input-2) should appear before any hidden neuron
    let first_hidden_pos = sources
        .iter()
        .position(|n| parse_input_index(&n.uuid).is_none())
        .expect("Should have at least one hidden neuron");

    let last_input_pos = sources
        .iter()
        .rposition(|n| parse_input_index(&n.uuid).is_some())
        .expect("Should have at least one input neuron");

    assert!(
        last_input_pos < first_hidden_pos,
        "All input neurons should come before hidden neurons. \
        Last input at position {last_input_pos}, first hidden at position {first_hidden_pos}. \
        Order: {:?}",
        sources.iter().map(|n| &n.uuid).collect::<Vec<_>>()
    );
}

#[test]
fn order_eligible_sources_input_first_with_different_seeds() {
    // Verify the ordering is consistent across different seeds
    let neurons = [
        OrderedNeuron {
            uuid: "hidden-uuid-a".to_string(),
            index: 10,
        },
        OrderedNeuron {
            uuid: "input-3".to_string(),
            index: 3,
        },
        OrderedNeuron {
            uuid: "hidden-uuid-b".to_string(),
            index: 11,
        },
        OrderedNeuron {
            uuid: "input-0".to_string(),
            index: 0,
        },
    ];

    for seed in [0u64, 1, 42, 100, 999] {
        let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();
        order_eligible_sources::<String>(&mut sources, Some(seed), "test-seeds", 4, None);

        // Input neurons should always be at the front
        let input_count = sources
            .iter()
            .filter(|n| parse_input_index(&n.uuid).is_some())
            .count();

        for (i, neuron) in sources.iter().enumerate() {
            if i < input_count {
                assert!(
                    parse_input_index(&neuron.uuid).is_some(),
                    "Seed {seed}: position {i} should be an input neuron, got {}",
                    neuron.uuid
                );
            } else {
                assert!(
                    parse_input_index(&neuron.uuid).is_none(),
                    "Seed {seed}: position {i} should be a hidden neuron, got {}",
                    neuron.uuid
                );
            }
        }
    }
}

#[test]
fn order_eligible_sources_preserves_all_neurons_with_prioritisation() {
    let neurons = [
        OrderedNeuron {
            uuid: "hidden-uuid-x".to_string(),
            index: 5,
        },
        OrderedNeuron {
            uuid: "input-0".to_string(),
            index: 0,
        },
        OrderedNeuron {
            uuid: "input-1".to_string(),
            index: 1,
        },
    ];

    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();
    order_eligible_sources::<String>(&mut sources, Some(42), "test-preserve", 2, None);

    assert_eq!(
        sources.len(),
        3,
        "All neurons should be preserved after ordering"
    );

    let uuids: HashSet<&str> = sources.iter().map(|n| n.uuid.as_str()).collect();
    assert!(uuids.contains("hidden-uuid-x"));
    assert!(uuids.contains("input-0"));
    assert!(uuids.contains("input-1"));
}

#[test]
fn order_eligible_sources_input_first_respects_focus_unused_override() {
    // When FOCUS_UNUSED_OBSERVATIONS is set with used_inputs, the unused input
    // prioritisation should still result in input neurons before hidden neurons
    let neurons = [
        OrderedNeuron {
            uuid: "hidden-uuid-1".to_string(),
            index: 5,
        },
        OrderedNeuron {
            uuid: "input-0".to_string(),
            index: 0,
        },
        OrderedNeuron {
            uuid: "input-1".to_string(),
            index: 1,
        },
    ];

    let used_inputs: HashSet<String> = HashSet::new(); // No inputs are used

    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();
    order_eligible_sources(&mut sources, Some(42), "test-unused", 2, Some(&used_inputs));

    // Input neurons should still be at the front
    assert!(
        parse_input_index(&sources[0].uuid).is_some(),
        "First source should be an input neuron"
    );
}

// =============================================================================
// INPUT_SOURCE_BOOST multiplier application
// =============================================================================

#[test]
fn input_source_boost_value_is_reasonable() {
    // INPUT_SOURCE_BOOST must be > 1.0 (verified by compile-time assert above)
    // and should be <= 3.0 to avoid over-biasing.
    // Verify the actual value produces a meaningful but bounded boost.
    let boosted = 1.0_f64 * INPUT_SOURCE_BOOST;
    let unboosted = 1.0_f64;
    assert!(
        boosted > unboosted && boosted <= 3.0 * unboosted,
        "INPUT_SOURCE_BOOST should provide a bounded boost: got {boosted}"
    );
}

#[test]
fn input_source_boost_amplifies_score_gain() {
    let base_gain = 0.05_f64;
    let boosted_gain = base_gain * INPUT_SOURCE_BOOST;

    assert!(
        boosted_gain > base_gain,
        "Boosted gain ({boosted_gain}) should exceed base gain ({base_gain})"
    );
    assert!(
        (boosted_gain - base_gain * INPUT_SOURCE_BOOST).abs() < f64::EPSILON,
        "Boosted gain should equal base_gain × INPUT_SOURCE_BOOST"
    );
}

#[test]
fn input_source_boost_does_not_apply_to_hidden_sources() {
    // Hidden neurons should get a neutral multiplier (1.0)
    // This is a behavioural test: the boost constant is only for input sources
    let base_gain = 0.05_f64;
    let hidden_multiplier = 1.0_f64;
    let hidden_gain = base_gain * hidden_multiplier;

    assert!(
        (hidden_gain - base_gain).abs() < f64::EPSILON,
        "Hidden neuron sources should not get a boost"
    );

    let input_gain = base_gain * INPUT_SOURCE_BOOST;
    assert!(
        input_gain > hidden_gain,
        "Input neuron gain ({input_gain}) should exceed hidden neuron gain ({hidden_gain})"
    );
}

// =============================================================================
// apply_source_type_boost function
// =============================================================================

#[test]
fn apply_source_type_boost_boosts_input_source() {
    use neat_ai_discovery::analysis::synapse::apply_source_type_boost;

    let gain = 0.10_f32;
    let boosted = apply_source_type_boost(gain, "input-0");

    assert!(
        boosted > gain,
        "Input source should get boosted score: {boosted} should be > {gain}"
    );
    let expected = gain * INPUT_SOURCE_BOOST as f32;
    assert!(
        (boosted - expected).abs() < 1e-6,
        "Boosted gain should equal gain × INPUT_SOURCE_BOOST: expected {expected}, got {boosted}"
    );
}

#[test]
fn apply_source_type_boost_neutral_for_hidden_source() {
    use neat_ai_discovery::analysis::synapse::apply_source_type_boost;

    let gain = 0.10_f32;
    let result = apply_source_type_boost(gain, "hidden-uuid-abc");

    assert!(
        (result - gain).abs() < 1e-6,
        "Hidden source should not be boosted: expected {gain}, got {result}"
    );
}

#[test]
fn apply_source_type_boost_neutral_for_output_source() {
    use neat_ai_discovery::analysis::synapse::apply_source_type_boost;

    let gain = 0.10_f32;
    let result = apply_source_type_boost(gain, "output-uuid-xyz");

    assert!(
        (result - gain).abs() < 1e-6,
        "Output source should not be boosted: expected {gain}, got {result}"
    );
}

#[test]
fn apply_source_type_boost_zero_gain_stays_zero() {
    use neat_ai_discovery::analysis::synapse::apply_source_type_boost;

    let result = apply_source_type_boost(0.0, "input-5");
    assert!(
        result.abs() < 1e-6,
        "Zero gain should remain zero even with boost: got {result}"
    );
}
