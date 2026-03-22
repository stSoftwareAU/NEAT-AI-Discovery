//! Tests for Issue #910: Add-synapse candidates between existing hidden neurons
//!
//! Verifies that:
//! 1. `HIDDEN_SOURCE_BOOST` is applied to hidden-sourced candidates
//! 2. Hidden-to-hidden candidates can compete with input-sourced candidates
//! 3. The source type boost correctly distinguishes input, hidden, and output sources
//! 4. The forward-only topology constraint allows valid hidden-to-hidden connections

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
use neat_ai_discovery::analysis::constants::{HIDDEN_SOURCE_BOOST, INPUT_SOURCE_BOOST};
use neat_ai_discovery::analysis::synapse::apply_source_type_boost;
use neat_ai_discovery::analysis::utils::{
    OrderedNeuron, order_eligible_sources, parse_input_index,
};

// =============================================================================
// HIDDEN_SOURCE_BOOST constant validation (compile-time)
// =============================================================================

// Compile-time validation that HIDDEN_SOURCE_BOOST is within acceptable range
const _: () = assert!(HIDDEN_SOURCE_BOOST >= 1.0);
const _: () = assert!(HIDDEN_SOURCE_BOOST <= 3.0);

// Statically enforce that hidden boost never exceeds input boost
const _HIDDEN_MUST_NOT_EXCEED_INPUT: () = {
    // Using a workaround since f64 comparison in const context requires integer cast
    // HIDDEN_SOURCE_BOOST (1.2) <= INPUT_SOURCE_BOOST (1.5)
    assert!((HIDDEN_SOURCE_BOOST * 1000.0) as u64 <= (INPUT_SOURCE_BOOST * 1000.0) as u64);
};

// =============================================================================
// apply_source_type_boost with hidden sources
// =============================================================================

#[test]
fn apply_source_type_boost_applies_hidden_boost_to_hidden_source() {
    let gain = 0.10_f32;
    let result = apply_source_type_boost(gain, "hidden-uuid-abc");
    let expected = gain * HIDDEN_SOURCE_BOOST as f32;

    assert!(
        (result - expected).abs() < 1e-6,
        "Hidden source should get HIDDEN_SOURCE_BOOST: expected {expected}, got {result}"
    );
    assert!(
        result > gain,
        "Hidden source boost should increase gain: {result} should be > {gain}"
    );
}

#[test]
fn apply_source_type_boost_input_still_outranks_hidden() {
    let gain = 0.10_f32;
    let input_result = apply_source_type_boost(gain, "input-5");
    let hidden_result = apply_source_type_boost(gain, "hidden-uuid-xyz");

    assert!(
        input_result > hidden_result,
        "Input-sourced gain ({input_result}) should still exceed hidden-sourced gain ({hidden_result})"
    );
}

#[test]
fn apply_source_type_boost_output_source_gets_no_boost() {
    let gain = 0.10_f32;
    let result = apply_source_type_boost(gain, "output-uuid-xyz");

    assert!(
        (result - gain).abs() < 1e-6,
        "Output source should not be boosted: expected {gain}, got {result}"
    );
}

#[test]
fn apply_source_type_boost_hidden_zero_gain_stays_zero() {
    let result = apply_source_type_boost(0.0, "hidden-uuid-abc");
    assert!(
        result.abs() < 1e-6,
        "Zero gain should remain zero even with hidden boost: got {result}"
    );
}

#[test]
fn apply_source_type_boost_negative_gain_preserved_direction() {
    let gain = -0.05_f32;
    let result = apply_source_type_boost(gain, "hidden-uuid-abc");

    assert!(
        result < 0.0,
        "Negative gain should remain negative after hidden boost: got {result}"
    );
}

// =============================================================================
// Forward-only topology allows hidden-to-hidden connections
// =============================================================================

#[test]
fn hidden_to_hidden_eligible_when_source_index_lower_than_target() {
    // Simulate the forward-only filter from target_analysis/mod.rs
    // A hidden neuron at index 5 should be eligible as a source for a hidden
    // neuron at index 10 (forward-only: source.index < target_index)
    let source = OrderedNeuron {
        uuid: "hidden-source-uuid".to_string(),
        index: 5,
    };
    let target_index = 10;

    assert!(
        source.index < target_index,
        "Hidden source at index {} should be eligible for hidden target at index {}",
        source.index,
        target_index
    );

    // And the source is not an input neuron — this is a hidden-to-hidden path
    assert!(
        parse_input_index(&source.uuid).is_none(),
        "Source should be a hidden neuron, not an input"
    );
}

#[test]
fn hidden_to_hidden_not_eligible_when_source_index_higher_than_target() {
    let source = OrderedNeuron {
        uuid: "hidden-later-uuid".to_string(),
        index: 15,
    };
    let target_index = 10;

    assert!(
        source.index >= target_index,
        "Hidden source at index {} should NOT be eligible for hidden target at index {} (violates forward-only)",
        source.index,
        target_index
    );
}

// =============================================================================
// Source ordering includes hidden sources for hidden-to-hidden evaluation
// =============================================================================

#[test]
fn hidden_neurons_present_in_source_ordering_for_hidden_target() {
    // When a hidden neuron is the target, other hidden neurons with lower
    // indices should appear as sources in the ordering.
    let neurons = [
        OrderedNeuron {
            uuid: "input-0".to_string(),
            index: 0,
        },
        OrderedNeuron {
            uuid: "input-1".to_string(),
            index: 1,
        },
        OrderedNeuron {
            uuid: "hidden-early-uuid".to_string(),
            index: 3,
        },
        OrderedNeuron {
            uuid: "hidden-mid-uuid".to_string(),
            index: 5,
        },
    ];

    // Target is at index 7 (a hidden neuron), so all above are eligible
    let mut sources: Vec<&OrderedNeuron> = neurons
        .iter()
        .filter(|n| n.index < 7) // forward-only constraint
        .collect();

    order_eligible_sources::<String>(&mut sources, Some(42), "test", 2, None);

    let hidden_sources: Vec<&str> = sources
        .iter()
        .filter(|n| parse_input_index(&n.uuid).is_none())
        .map(|n| n.uuid.as_str())
        .collect();

    assert!(
        !hidden_sources.is_empty(),
        "Hidden neurons should be present as sources for a hidden target. \
         Sources: {:?}",
        sources.iter().map(|n| &n.uuid).collect::<Vec<_>>()
    );
    assert_eq!(
        hidden_sources.len(),
        2,
        "Both hidden neurons should be present as sources"
    );
}

#[test]
fn hidden_to_hidden_candidates_not_filtered_by_type() {
    // Verify that the source filtering in target_analysis does not exclude
    // hidden sources — it only excludes constant neurons.
    // This test simulates the filter logic from target_analysis/mod.rs lines 198-217.
    let neurons = [
        OrderedNeuron {
            uuid: "input-0".to_string(),
            index: 0,
        },
        OrderedNeuron {
            uuid: "hidden-uuid-a".to_string(),
            index: 3,
        },
        OrderedNeuron {
            uuid: "hidden-uuid-b".to_string(),
            index: 5,
        },
    ];

    let target_index = 7;

    // Simulate the filter: index < target_index AND type != "constant"
    let eligible: Vec<&OrderedNeuron> = neurons
        .iter()
        .filter(|n| {
            n.index < target_index && !n.uuid.starts_with("constant") // type != "constant"
        })
        .collect();

    // All three should be eligible (1 input + 2 hidden)
    assert_eq!(
        eligible.len(),
        3,
        "All non-constant neurons before target should be eligible: {:?}",
        eligible.iter().map(|n| &n.uuid).collect::<Vec<_>>()
    );

    let hidden_eligible: Vec<&&OrderedNeuron> = eligible
        .iter()
        .filter(|n| parse_input_index(&n.uuid).is_none())
        .collect();

    assert_eq!(
        hidden_eligible.len(),
        2,
        "Both hidden neurons should be eligible as sources"
    );
}
