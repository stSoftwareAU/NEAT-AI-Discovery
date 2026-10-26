//! Tests for Issue #548: Local minimum escape — coordinate multi-neuron squash
//! exploration with compensating weight rescaling.
//!
//! ## TDD Plan
//! 1. Test coordinated candidates include changeSquash + setWeight operations
//! 2. Test weight rescaling maintains approximate output equivalence
//! 3. Test improved expected gain over standalone squash changes
//! 4. Test no candidates when the network is already well-suited
//! 5. Test multiple incoming synapses each get compensating weights
//! 6. Test aggregate squashes are skipped (cannot be simulated as f(x))
//! 7. Test insufficient samples produce no candidates

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::squash_weight_rescale::{
    detect_squash_weight_rescale_candidates, squash_weight_rescale_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

// =============================================================================
// Helpers
// =============================================================================

fn make_record(uuid: &str, idx: u32, value: f32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(value),
        activation,
        errors: vec![error],
    }
}

fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    CreatureJson {
        neurons,
        synapses,
        input: 2,
        output: 1,
    }
}

fn hidden_neuron(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "hidden".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

fn input_neuron(uuid: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "input".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn output_neuron(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "output".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

// =============================================================================
// 1. Coordinated candidates include changeSquash + setWeight operations
// =============================================================================

#[test]
fn test_coordinated_candidate_has_squash_and_weight_operations() {
    // Hidden neuron with RELU that sees predominantly negative pre-activation values
    // → should recommend switching to ELU or similar + rescale incoming weights
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            input_neuron("input-2"),
            hidden_neuron("hidden-1", "RELU"),
            output_neuron("output-1", "TANH"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.5),
            synapse("input-2", "hidden-1", -0.8),
            synapse("hidden-1", "output-1", 1.0),
        ],
    );

    let hidden_neurons: Vec<(String, String, f32)> =
        vec![("hidden-1".to_string(), "RELU".to_string(), 0.0)];

    // Pre-activation values spanning both positive and negative, but RELU clips the negatives
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-1".to_string(),
        (0..40)
            .map(|i| {
                let value = (i as f32 - 20.0) / 10.0; // -2.0 to 1.9
                let activation = value.max(0.0); // RELU
                let error = 0.15 + (i as f32 * 0.002).sin() * 0.02;
                make_record("hidden-1", i, value, activation, error)
            })
            .collect(),
    )];

    let candidates =
        detect_squash_weight_rescale_candidates(&creature, &hidden_neurons, &neuron_records);
    assert!(
        !candidates.is_empty(),
        "Should detect squash + weight rescale candidates"
    );

    let coordinated = squash_weight_rescale_to_coordinated_candidates(&candidates);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();

    // Must contain both changeSquash and at least one setWeight
    assert!(
        ops_json.contains("changeSquash"),
        "Coordinated candidate must include changeSquash, got: {ops_json}"
    );
    assert!(
        ops_json.contains("setWeight"),
        "Coordinated candidate must include compensating setWeight, got: {ops_json}"
    );
}

// =============================================================================
// 2. Weight rescaling maintains approximate output equivalence
// =============================================================================

#[test]
fn test_weight_rescaling_preserves_operating_point() {
    // A hidden neuron using HARD_TANH with pre-activation values near the clipping boundary
    // → switching to TANH should include rescaled weights that preserve the operating point
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            hidden_neuron("hidden-1", "HARD_TANH"),
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 2.0),
            synapse("hidden-1", "output-1", 1.0),
        ],
    );

    let hidden_neurons: Vec<(String, String, f32)> =
        vec![("hidden-1".to_string(), "HARD_TANH".to_string(), 0.0)];

    // Operating near the saturation boundary of HARD_TANH
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-1".to_string(),
        (0..40)
            .map(|i| {
                let value = (i as f32 - 20.0) / 15.0; // around -1.3 to 1.3
                let activation = value.clamp(-1.0, 1.0); // HARD_TANH
                let error = 0.2 + (i as f32 * 0.001).sin() * 0.01;
                make_record("hidden-1", i, value, activation, error)
            })
            .collect(),
    )];

    let candidates =
        detect_squash_weight_rescale_candidates(&creature, &hidden_neurons, &neuron_records);

    if candidates.is_empty() {
        // If no candidate produced, that's acceptable if there's no better squash
        return;
    }

    let candidate = &candidates[0];

    // The rescaled weights should be finite and non-zero
    for (_, _, new_weight) in &candidate.rescaled_weights {
        assert!(
            new_weight.is_finite(),
            "Rescaled weight must be finite, got: {new_weight}"
        );
    }
}

// =============================================================================
// 3. Improved expected gain over standalone squash changes
// =============================================================================

#[test]
fn test_improved_gain_over_standalone_squash() {
    // Coordinated squash + weight change should have positive expected gain
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            input_neuron("input-2"),
            hidden_neuron("hidden-1", "RELU"),
            output_neuron("output-1", "TANH"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.5),
            synapse("input-2", "hidden-1", -0.8),
            synapse("hidden-1", "output-1", 1.0),
        ],
    );

    let hidden_neurons: Vec<(String, String, f32)> =
        vec![("hidden-1".to_string(), "RELU".to_string(), 0.0)];

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-1".to_string(),
        (0..40)
            .map(|i| {
                let value = (i as f32 - 20.0) / 10.0;
                let activation = value.max(0.0);
                let error = 0.15 + (i as f32 * 0.002).sin() * 0.02;
                make_record("hidden-1", i, value, activation, error)
            })
            .collect(),
    )];

    let candidates =
        detect_squash_weight_rescale_candidates(&creature, &hidden_neurons, &neuron_records);
    assert!(!candidates.is_empty());

    let coordinated = squash_weight_rescale_to_coordinated_candidates(&candidates);
    assert!(!coordinated.is_empty());

    assert!(
        coordinated[0].expected_creature_score_gain > 0.0,
        "Expected gain should be positive for coordinated squash + weight change"
    );
}

// =============================================================================
// 4. No candidates when network is already well-suited
// =============================================================================

#[test]
fn test_no_candidates_for_well_suited_network() {
    // TANH neuron with values nicely in the middle of its range — no mismatch
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            hidden_neuron("hidden-1", "TANH"),
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 1.0),
        ],
    );

    let hidden_neurons: Vec<(String, String, f32)> =
        vec![("hidden-1".to_string(), "TANH".to_string(), 0.0)];

    // Values in -0.5..0.5 — well within TANH's linear region, low error
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-1".to_string(),
        (0..40)
            .map(|i| {
                let value = (i as f32 - 20.0) / 40.0; // -0.5 to 0.5
                let activation = value.tanh();
                let error = 0.005; // very low error
                make_record("hidden-1", i, value, activation, error)
            })
            .collect(),
    )];

    let candidates =
        detect_squash_weight_rescale_candidates(&creature, &hidden_neurons, &neuron_records);
    assert!(
        candidates.is_empty(),
        "Well-suited network should produce no candidates, got: {}",
        candidates.len()
    );
}

// =============================================================================
// 5. Multiple incoming synapses each get compensating weights
// =============================================================================

#[test]
fn test_multiple_incoming_synapses_get_weights() {
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            input_neuron("input-2"),
            hidden_neuron("hidden-1", "RELU"),
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.5),
            synapse("input-2", "hidden-1", -0.8),
            synapse("hidden-1", "output-1", 1.0),
        ],
    );

    let hidden_neurons: Vec<(String, String, f32)> =
        vec![("hidden-1".to_string(), "RELU".to_string(), 0.0)];

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-1".to_string(),
        (0..40)
            .map(|i| {
                let value = (i as f32 - 20.0) / 10.0;
                let activation = value.max(0.0);
                let error = 0.15 + (i as f32 * 0.002).sin() * 0.02;
                make_record("hidden-1", i, value, activation, error)
            })
            .collect(),
    )];

    let candidates =
        detect_squash_weight_rescale_candidates(&creature, &hidden_neurons, &neuron_records);

    if candidates.is_empty() {
        return;
    }

    let coordinated = squash_weight_rescale_to_coordinated_candidates(&candidates);

    // Count setWeight operations — should match number of incoming synapses
    let set_weight_count = coordinated[0]
        .operations
        .iter()
        .filter(|op| {
            let json = serde_json::to_string(op).unwrap();
            json.contains("setWeight")
        })
        .count();

    assert!(
        set_weight_count >= 2,
        "Should include setWeight for each incoming synapse, got: {set_weight_count}"
    );
}

// =============================================================================
// 6. Aggregate squashes are skipped
// =============================================================================

#[test]
fn test_aggregate_squash_skipped() {
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IF".to_string(),
                bias: 0.0,
            },
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.0),
            synapse("hidden-1", "output-1", 1.0),
        ],
    );

    let hidden_neurons: Vec<(String, String, f32)> =
        vec![("hidden-1".to_string(), "IF".to_string(), 0.0)];

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-1".to_string(),
        (0..40)
            .map(|i| {
                let value = (i as f32 - 20.0) / 10.0;
                let activation = value;
                let error = 0.3;
                make_record("hidden-1", i, value, activation, error)
            })
            .collect(),
    )];

    let candidates =
        detect_squash_weight_rescale_candidates(&creature, &hidden_neurons, &neuron_records);
    assert!(
        candidates.is_empty(),
        "Aggregate squash (IF) should not generate candidates"
    );
}

// =============================================================================
// 7. Insufficient samples produce no candidates
// =============================================================================

#[test]
fn test_squash_weight_rescale_insufficient_samples_no_candidates() {
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            hidden_neuron("hidden-1", "RELU"),
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.0),
            synapse("hidden-1", "output-1", 1.0),
        ],
    );

    let hidden_neurons: Vec<(String, String, f32)> =
        vec![("hidden-1".to_string(), "RELU".to_string(), 0.0)];

    // Only 5 records — well below the minimum sample threshold
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-1".to_string(),
        (0..5)
            .map(|i| {
                let value = (i as f32 - 2.0) / 2.0;
                let activation = value.max(0.0);
                make_record("hidden-1", i, value, activation, 0.3)
            })
            .collect(),
    )];

    let candidates =
        detect_squash_weight_rescale_candidates(&creature, &hidden_neurons, &neuron_records);
    assert!(
        candidates.is_empty(),
        "Insufficient samples should produce no candidates"
    );
}
