//! Issue #337: Verify that all candidate types returned by NEAT-AI-Discovery
//! match the contract expected by the NEAT-AI TypeScript consumer.
//!
//! NEAT-AI (TypeScript) must implement all candidate types that this library can
//! return. This test suite verifies:
//!
//! 1. All 7 `CoordinatedStructuralOpJson` variants serialise to the JSON format
//!    that NEAT-AI's `ApplyCoordinatedStructuralCandidate.ts` expects.
//! 2. The top-level `AnalyzeParallelOutput` contains all candidate fields that
//!    NEAT-AI's `DiscoverResult.ts` consumes.
//! 3. The `RankFocusNeuronsOutput` contains all fields that NEAT-AI expects,
//!    including `constantNeuronRemovals` (Issue #306).
//! 4. `SynapseWeightUpdateCandidateJson` serialises correctly for NEAT-AI.
//!
//! If a new candidate type or operation variant is added to the Rust library,
//! a corresponding test MUST be added here and NEAT-AI must be updated.

use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson,
    SynapseWeightUpdateCandidateJson,
};

// ---------------------------------------------------------------------------
// 1. CoordinatedStructuralOpJson — all 7 variants
// ---------------------------------------------------------------------------

#[test]
fn coordinated_op_remove_synapse_serialises_for_neat_ai() {
    let op = CoordinatedStructuralOpJson::RemoveSynapse {
        from_neuron_uuid: "input-0".to_string(),
        to_neuron_uuid: "output-0".to_string(),
    };

    let json: serde_json::Value = serde_json::to_value(&op).expect("serialisation should succeed");

    assert_eq!(json["type"], "removeSynapse");
    assert_eq!(json["fromNeuronUuid"], "input-0");
    assert_eq!(json["toNeuronUuid"], "output-0");
    // NEAT-AI's ApplyCoordinatedStructuralCandidate.ts checks op.type === "removeSynapse"
    // and reads fromNeuronUuid + toNeuronUuid.
}

#[test]
fn coordinated_op_add_synapse_serialises_for_neat_ai() {
    let op = CoordinatedStructuralOpJson::AddSynapse {
        from_neuron_uuid: "input-1".to_string(),
        to_neuron_uuid: "output-0".to_string(),
        weight: 0.42,
    };

    let json: serde_json::Value = serde_json::to_value(&op).expect("serialisation should succeed");

    assert_eq!(json["type"], "addSynapse");
    assert_eq!(json["fromNeuronUuid"], "input-1");
    assert_eq!(json["toNeuronUuid"], "output-0");
    assert!(
        (json["weight"].as_f64().unwrap() - 0.42).abs() < 1e-6,
        "weight should be approximately 0.42"
    );
}

#[test]
fn coordinated_op_add_neuron_serialises_for_neat_ai() {
    let op = CoordinatedStructuralOpJson::AddNeuron {
        neuron_uuid: "coordinated-hidden-abc123".to_string(),
        neuron_type: "hidden".to_string(),
        squash: "ReLU".to_string(),
        bias: 0.5,
        insert_before_neuron_uuid: Some("output-0".to_string()),
    };

    let json: serde_json::Value = serde_json::to_value(&op).expect("serialisation should succeed");

    assert_eq!(json["type"], "addNeuron");
    assert_eq!(json["neuronUuid"], "coordinated-hidden-abc123");
    assert_eq!(json["neuronType"], "hidden");
    assert_eq!(json["squash"], "ReLU");
    assert!((json["bias"].as_f64().unwrap() - 0.5).abs() < 1e-6);
    assert_eq!(json["insertBeforeNeuronUuid"], "output-0");
}

#[test]
fn coordinated_op_add_neuron_omits_insert_before_when_none() {
    let op = CoordinatedStructuralOpJson::AddNeuron {
        neuron_uuid: "hidden-1".to_string(),
        neuron_type: "hidden".to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
        insert_before_neuron_uuid: None,
    };

    let json: serde_json::Value = serde_json::to_value(&op).expect("serialisation should succeed");

    assert_eq!(json["type"], "addNeuron");
    // insertBeforeNeuronUuid should be absent (skip_serializing_if = "Option::is_none")
    assert!(
        json.get("insertBeforeNeuronUuid").is_none() || json["insertBeforeNeuronUuid"].is_null(),
        "insertBeforeNeuronUuid should be absent when None"
    );
}

#[test]
fn coordinated_op_remove_neuron_serialises_for_neat_ai() {
    let op = CoordinatedStructuralOpJson::RemoveNeuron {
        neuron_uuid: "hidden-0".to_string(),
    };

    let json: serde_json::Value = serde_json::to_value(&op).expect("serialisation should succeed");

    assert_eq!(json["type"], "removeNeuron");
    assert_eq!(json["neuronUuid"], "hidden-0");
}

#[test]
fn coordinated_op_change_squash_serialises_for_neat_ai() {
    let op = CoordinatedStructuralOpJson::ChangeSquash {
        neuron_uuid: "hidden-0".to_string(),
        squash: "GELU".to_string(),
    };

    let json: serde_json::Value = serde_json::to_value(&op).expect("serialisation should succeed");

    assert_eq!(json["type"], "changeSquash");
    assert_eq!(json["neuronUuid"], "hidden-0");
    assert_eq!(json["squash"], "GELU");
}

#[test]
fn coordinated_op_set_bias_serialises_for_neat_ai() {
    let op = CoordinatedStructuralOpJson::SetBias {
        neuron_uuid: "hidden-0".to_string(),
        bias: -1.5,
    };

    let json: serde_json::Value = serde_json::to_value(&op).expect("serialisation should succeed");

    assert_eq!(json["type"], "setBias");
    assert_eq!(json["neuronUuid"], "hidden-0");
    assert!((json["bias"].as_f64().unwrap() - (-1.5)).abs() < 1e-6);
}

#[test]
fn coordinated_op_set_weight_serialises_for_neat_ai() {
    // Issue #180: SetWeight replaces the remove+add pattern for weight adjustments.
    // NEAT-AI's ApplyCoordinatedStructuralCandidate.ts handles op.type === "setWeight".
    let op = CoordinatedStructuralOpJson::SetWeight {
        from_neuron_uuid: "input-0".to_string(),
        to_neuron_uuid: "output-0".to_string(),
        weight: 0.006,
    };

    let json: serde_json::Value = serde_json::to_value(&op).expect("serialisation should succeed");

    assert_eq!(json["type"], "setWeight");
    assert_eq!(json["fromNeuronUuid"], "input-0");
    assert_eq!(json["toNeuronUuid"], "output-0");
    assert!((json["weight"].as_f64().unwrap() - 0.006).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// 2. CoordinatedStructuralCandidateJson — wrapper shape
// ---------------------------------------------------------------------------

#[test]
fn coordinated_candidate_serialises_with_all_required_fields() {
    let candidate = CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: "output-0".to_string(),
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: "output-0".to_string(),
                weight: 0.5,
            },
        ],
        expected_creature_score_gain: 0.001,
        comment: Some("Epistatic pair: remove noisy, add trusted".to_string()),
    };

    let json: serde_json::Value =
        serde_json::to_value(&candidate).expect("serialisation should succeed");

    // NEAT-AI's CoordinatedStructuralCandidate.ts expects:
    //   operations: CoordinatedStructuralOperation[]
    //   expectedCreatureScoreGain: number
    //   comment?: string
    assert!(json["operations"].is_array());
    assert_eq!(json["operations"].as_array().unwrap().len(), 2);
    assert!(json["expectedCreatureScoreGain"].as_f64().is_some());
    assert_eq!(json["comment"], "Epistatic pair: remove noisy, add trusted");
}

#[test]
fn coordinated_candidate_omits_comment_when_none() {
    let candidate = CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: "h-1".to_string(),
            bias: 0.0,
        }],
        expected_creature_score_gain: 0.0,
        comment: None,
    };

    let json: serde_json::Value =
        serde_json::to_value(&candidate).expect("serialisation should succeed");

    assert!(
        json.get("comment").is_none() || json["comment"].is_null(),
        "comment should be absent when None"
    );
}

// ---------------------------------------------------------------------------
// 3. Synergistic discovery (Issue #189) produces valid coordinated candidates
// ---------------------------------------------------------------------------

#[test]
fn synergistic_candidate_uses_add_synapse_operations() {
    // Issue #189: Synergistic discovery detects cross-neuron interactions
    // (e.g., XOR-like patterns) and returns them as CoordinatedStructuralCandidateJson
    // with AddSynapse operations. NEAT-AI handles these via the existing
    // coordinated-structural path — no new operation type is needed.
    let candidate = CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: "output-0".to_string(),
                weight: 0.3,
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "input-1".to_string(),
                to_neuron_uuid: "output-0".to_string(),
                weight: -0.3,
            },
        ],
        expected_creature_score_gain: 0.005,
        comment: Some("Synergistic: XOR-like pattern".to_string()),
    };

    let json: serde_json::Value =
        serde_json::to_value(&candidate).expect("serialisation should succeed");

    let ops = json["operations"].as_array().unwrap();
    assert_eq!(ops.len(), 2);
    assert_eq!(ops[0]["type"], "addSynapse");
    assert_eq!(ops[1]["type"], "addSynapse");
    assert!(json["expectedCreatureScoreGain"].as_f64().unwrap() > 0.0);
}

// ---------------------------------------------------------------------------
// 4. SynapseWeightUpdateCandidateJson — standalone weight update
// ---------------------------------------------------------------------------

#[test]
fn synapse_weight_update_candidate_serialises_for_neat_ai() {
    let candidate = SynapseWeightUpdateCandidateJson {
        from_neuron_uuid: "input-0".to_string(),
        to_neuron_uuid: "output-0".to_string(),
        from_neuron_index: Some(0),
        to_neuron_index: Some(1),
        old_weight: 0.1,
        new_weight: 0.3,
        delta_weight: 0.2,
        target_neuron_impact: 0.5,
        expected_creature_error_reduction: 0.01,
        expected_creature_score_gain: 0.001,
        improved_count: 40,
        total_count: 100,
        target_neuron_stats: None,
    };

    let json: serde_json::Value =
        serde_json::to_value(&candidate).expect("serialisation should succeed");

    // NEAT-AI expects camelCase field names
    assert_eq!(json["fromNeuronUuid"], "input-0");
    assert_eq!(json["toNeuronUuid"], "output-0");
    assert_eq!(json["fromNeuronIndex"], 0);
    assert_eq!(json["toNeuronIndex"], 1);
    assert!((json["oldWeight"].as_f64().unwrap() - 0.1).abs() < 1e-6);
    assert!((json["newWeight"].as_f64().unwrap() - 0.3).abs() < 1e-6);
    assert!((json["deltaWeight"].as_f64().unwrap() - 0.2).abs() < 1e-6);
    assert!(json["targetNeuronImpact"].as_f64().is_some());
    assert!(json["expectedCreatureErrorReduction"].as_f64().is_some());
    assert!(json["expectedCreatureScoreGain"].as_f64().is_some());
    assert_eq!(json["improvedCount"], 40);
    assert_eq!(json["totalCount"], 100);
}

// ---------------------------------------------------------------------------
// 5. AnalyzeParallelOutput — all candidate fields present
// ---------------------------------------------------------------------------

#[test]
fn analyze_parallel_output_contains_all_candidate_fields() {
    // Verify that AnalyzeParallelOutput serialises all candidate type fields
    // that NEAT-AI's DiscoverResult.ts and DiscoveryCandidates.ts consume.
    let output = neat_ai_discovery::AnalyzeParallelOutput {
        success: true,
        schema_version: neat_ai_discovery::SCHEMA_VERSION.to_string(),
        helpful_synapses: Some(vec![]),
        harmful_synapses: Some(vec![]),
        synapse_diagnostics: None,
        synapse_gpu_used: Some(true),
        synapse_metadata: None,
        helpful_neurons: Some(vec![]),
        synapse_weight_updates: Some(vec![]),
        coordinated_structural_candidates: Some(vec![]),
        candidate_clusters: None,
        neuron_diagnostics: None,
        neuron_gpu_used: Some(true),
        neuron_metadata: None,
        neuron_fingerprints: None,
        fingerprint_cache_hits: None,
        fingerprint_cache_misses: None,
        module_outcome_tracker: None,
        memory_budget_exceeded: None,
        cancelled: None,
        memory_pressure_cancelled: None,
        error: None,
        error_kind: None,
        retryable: None,
    };

    let json: serde_json::Value =
        serde_json::to_value(&output).expect("serialisation should succeed");

    // Fields consumed by NEAT-AI DiscoverResult.ts / DiscoveryCandidates.ts:
    assert_eq!(json["success"], true);
    assert!(
        json["helpfulSynapses"].is_array(),
        "helpfulSynapses must be present for add-synapses candidates"
    );
    assert!(
        json["harmfulSynapses"].is_array(),
        "harmfulSynapses must be present for remove-harmful-synapse candidates"
    );
    assert!(
        json["helpfulNeurons"].is_array(),
        "helpfulNeurons must be present for add-neurons candidates"
    );
    assert!(
        json["synapseWeightUpdates"].is_array(),
        "synapseWeightUpdates must be present (v0.2.18+)"
    );
    assert!(
        json["coordinatedStructuralCandidates"].is_array(),
        "coordinatedStructuralCandidates must be present for coordinated-structural candidates"
    );
}

// ---------------------------------------------------------------------------
// 6. RankFocusNeuronsOutput — removal and constant neuron fields
// ---------------------------------------------------------------------------

#[test]
fn rank_focus_neurons_output_contains_removal_candidate_fields() {
    // Verify that RankFocusNeuronsOutput serialises all fields NEAT-AI expects,
    // including constantNeuronRemovals (Issue #306).
    let output = neat_ai_discovery::RankFocusNeuronsOutput {
        success: true,
        schema_version: neat_ai_discovery::SCHEMA_VERSION.to_string(),
        neurons: Some(vec![]),
        removal_candidates: Some(vec![]),
        constant_neuron_removals: Some(vec![]),
        max_output_error: Some(0.01),
        processed_neurons: Some(5),
        total_neurons: Some(10),
        duration_ms: Some(100),
        error: None,
        error_kind: None,
        retryable: None,
    };

    let json: serde_json::Value =
        serde_json::to_value(&output).expect("serialisation should succeed");

    assert_eq!(json["success"], true);
    assert!(json["neurons"].is_array());
    assert!(
        json["removalCandidates"].is_array(),
        "removalCandidates must be present for remove-low-impact candidates"
    );
    assert!(
        json["constantNeuronRemovals"].is_array(),
        "constantNeuronRemovals must be present (Issue #306)"
    );
}

// ---------------------------------------------------------------------------
// 7. All 7 operation type strings match NEAT-AI's switch/if checks
// ---------------------------------------------------------------------------

#[test]
fn all_operation_types_match_neat_ai_type_discriminator() {
    // NEAT-AI's ApplyCoordinatedStructuralCandidate.ts uses op.type to dispatch.
    // This test ensures every variant produces the exact string NEAT-AI expects.
    let ops_and_expected_types = vec![
        (
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "a".to_string(),
                to_neuron_uuid: "b".to_string(),
            },
            "removeSynapse",
        ),
        (
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "a".to_string(),
                to_neuron_uuid: "b".to_string(),
                weight: 1.0,
            },
            "addSynapse",
        ),
        (
            CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: "n".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "ReLU".to_string(),
                bias: 0.0,
                insert_before_neuron_uuid: None,
            },
            "addNeuron",
        ),
        (
            CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: "n".to_string(),
            },
            "removeNeuron",
        ),
        (
            CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: "n".to_string(),
                squash: "TANH".to_string(),
            },
            "changeSquash",
        ),
        (
            CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: "n".to_string(),
                bias: 0.0,
            },
            "setBias",
        ),
        (
            CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid: "a".to_string(),
                to_neuron_uuid: "b".to_string(),
                weight: 1.0,
            },
            "setWeight",
        ),
    ];

    for (op, expected_type) in &ops_and_expected_types {
        let json: serde_json::Value =
            serde_json::to_value(op).expect("serialisation should succeed");
        assert_eq!(
            json["type"].as_str().unwrap(),
            *expected_type,
            "Operation variant should serialise type as \"{expected_type}\""
        );
    }

    // Ensure we tested all 7 variants — if a new variant is added, this count
    // must be updated and a corresponding NEAT-AI implementation must be verified.
    assert_eq!(
        ops_and_expected_types.len(),
        7,
        "Expected exactly 7 CoordinatedStructuralOpJson variants. \
         If a new variant was added, update this test AND verify NEAT-AI handles it."
    );
}
