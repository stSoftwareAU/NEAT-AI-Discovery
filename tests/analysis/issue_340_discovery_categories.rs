//! Issue #340: Validate discovery category contract
//!
//! This test documents and validates all discovery categories currently supported
//! by the library. It serves as a contract test ensuring that the expected
//! candidate types and coordinated structural operations remain available.
//!
//! Discovery categories:
//! 1. Helpful synapses (add synapse candidates)
//! 2. Harmful synapses (remove synapse candidates)
//! 3. Synapse weight updates (delta-based weight adjustments)
//! 4. Helpful neurons (add neuron candidates with activation function)
//! 5. Coordinated structural candidates (grouped atomic operations)
//!    - `RemoveSynapse`, `AddSynapse`, `AddNeuron`, `RemoveNeuron`,
//!      `ChangeSquash`, `SetBias`, `SetWeight`
//!
//! Future discovery categories (see open issues):
//! - Dead neuron detection (#341)
//! - Saturated neuron detection (#342)
//! - Bottleneck neuron detection (#343)
//! - Correlated error pattern detection (#344)

use neat_ai_discovery::{
    CandidateNeuronJson, CandidateSynapseJson, CoordinatedStructuralCandidateJson,
    CoordinatedStructuralOpJson, SynapseWeightUpdateCandidateJson,
};

/// Verify that `CandidateSynapseJson` serialises with the expected camelCase field names.
/// This is the contract between NEAT-AI-Discovery and the TypeScript controller.
#[test]
fn candidate_synapse_json_serialisation_contract() {
    let candidate = CandidateSynapseJson {
        from_neuron_uuid: "input-0".to_string(),
        to_neuron_uuid: "output-0".to_string(),
        from_neuron_index: Some(0),
        to_neuron_index: Some(1),
        weight: 0.5,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.01,
        expected_creature_score_gain: 0.01,
        improved_count: 100,
        total_count: 200,
        target_neuron_stats: None,
        outlier_reduction_info: None,
        prediction_confidence: 0.9,
        expected_score_gain_confidence_interval: [0.005, 0.015],
        comment: None,
    };

    let json = serde_json::to_value(&candidate).expect("serialisation should succeed");

    // Verify camelCase field names
    assert!(
        json.get("fromNeuronUuid").is_some(),
        "expected fromNeuronUuid"
    );
    assert!(json.get("toNeuronUuid").is_some(), "expected toNeuronUuid");
    assert!(json.get("weight").is_some(), "expected weight");
    assert!(
        json.get("expectedCreatureScoreGain").is_some(),
        "expected expectedCreatureScoreGain"
    );
    assert!(
        json.get("improvedCount").is_some(),
        "expected improvedCount"
    );
    assert!(json.get("totalCount").is_some(), "expected totalCount");
    assert!(
        json.get("predictionConfidence").is_some(),
        "expected predictionConfidence"
    );
}

/// Verify that `CandidateNeuronJson` serialises with the expected camelCase field names.
#[test]
fn candidate_neuron_json_serialisation_contract() {
    let candidate = CandidateNeuronJson {
        source_neuron_uuid: "input-0".to_string(),
        target_neuron_uuid: "output-0".to_string(),
        source_neuron_index: Some(0),
        target_neuron_index: Some(1),
        incoming_weight: 0.5,
        outgoing_weight: 0.3,
        squash: "TANH".to_string(),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.02,
        expected_creature_score_gain: 0.02,
        improved_count: 150,
        total_count: 200,
        target_neuron_stats: None,
        prediction_confidence: 0.85,
        expected_score_gain_confidence_interval: [0.01, 0.03],
        target_saturation_factor: None,
    };

    let json = serde_json::to_value(&candidate).expect("serialisation should succeed");

    assert!(
        json.get("sourceNeuronUuid").is_some(),
        "expected sourceNeuronUuid"
    );
    assert!(
        json.get("targetNeuronUuid").is_some(),
        "expected targetNeuronUuid"
    );
    assert!(
        json.get("incomingWeight").is_some(),
        "expected incomingWeight"
    );
    assert!(
        json.get("outgoingWeight").is_some(),
        "expected outgoingWeight"
    );
    assert!(json.get("squash").is_some(), "expected squash");
    assert!(json.get("bias").is_some(), "expected bias");
}

/// Verify that `SynapseWeightUpdateCandidateJson` serialises correctly.
#[test]
fn synapse_weight_update_serialisation_contract() {
    let candidate = SynapseWeightUpdateCandidateJson {
        from_neuron_uuid: "hidden-1".to_string(),
        to_neuron_uuid: "output-0".to_string(),
        from_neuron_index: None,
        to_neuron_index: None,
        old_weight: 0.5,
        new_weight: 0.7,
        delta_weight: 0.2,
        expected_creature_error_reduction: 0.005,
        expected_creature_score_gain: 0.005,
        target_neuron_impact: 0.8,
        improved_count: 80,
        total_count: 200,
        target_neuron_stats: None,
    };

    let json = serde_json::to_value(&candidate).expect("serialisation should succeed");

    assert!(json.get("oldWeight").is_some(), "expected oldWeight");
    assert!(json.get("newWeight").is_some(), "expected newWeight");
    assert!(json.get("deltaWeight").is_some(), "expected deltaWeight");
}

/// Verify all seven coordinated structural operation types serialise correctly.
/// These operations form the vocabulary for grouped/epistatic candidates.
#[test]
fn coordinated_structural_operations_contract() {
    let ops = vec![
        CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        },
        CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: "c".to_string(),
            to_neuron_uuid: "d".to_string(),
            weight: 0.5,
        },
        CoordinatedStructuralOpJson::AddNeuron {
            neuron_uuid: "new-1".to_string(),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
            insert_before_neuron_uuid: Some("d".to_string()),
        },
        CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: "old-1".to_string(),
        },
        CoordinatedStructuralOpJson::ChangeSquash {
            neuron_uuid: "h-1".to_string(),
            squash: "RELU".to_string(),
        },
        CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: "h-2".to_string(),
            bias: -0.5,
        },
        CoordinatedStructuralOpJson::SetWeight {
            from_neuron_uuid: "e".to_string(),
            to_neuron_uuid: "f".to_string(),
            weight: 1.2,
        },
    ];

    let candidate = CoordinatedStructuralCandidateJson {
        operations: ops,
        expected_creature_score_gain: 0.05,
        comment: Some("Test all operation types".to_string()),
    };

    let json = serde_json::to_value(&candidate).expect("serialisation should succeed");
    let ops_json = json.get("operations").expect("expected operations array");
    let ops_array = ops_json.as_array().expect("operations should be an array");

    assert_eq!(ops_array.len(), 7, "expected all 7 operation types");

    // Verify each operation has the correct type tag
    let expected_types = [
        "removeSynapse",
        "addSynapse",
        "addNeuron",
        "removeNeuron",
        "changeSquash",
        "setBias",
        "setWeight",
    ];

    for (i, expected_type) in expected_types.iter().enumerate() {
        let op_type = ops_array[i]
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("missing");
        assert_eq!(
            op_type, *expected_type,
            "operation {i} should have type '{expected_type}', got '{op_type}'"
        );
    }

    // Verify AddNeuron includes insertBeforeNeuronUuid when present
    let add_neuron = &ops_array[2];
    assert!(
        add_neuron.get("insertBeforeNeuronUuid").is_some(),
        "AddNeuron should include insertBeforeNeuronUuid"
    );
}

/// Verify that the coordinated structural candidate omits optional fields when None.
#[test]
fn coordinated_structural_candidate_optional_fields() {
    let candidate = CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: "x".to_string(),
        }],
        expected_creature_score_gain: 0.01,
        comment: None,
    };

    let json = serde_json::to_value(&candidate).expect("serialisation should succeed");

    // comment should be omitted when None
    assert!(
        json.get("comment").is_none(),
        "comment should be omitted when None"
    );
}
