//! Integration tests for Issue #922: Extend candidate compression to non-linear
//! squash functions (TANH, GELU).
//!
//! ## TDD Plan
//! 1. Verify TANH compression produces valid coordinated candidates in linear regime
//! 2. Verify TANH compression with saturated inputs produces diminished gain
//! 3. Verify GELU compression produces valid coordinated candidates
//! 4. Verify benefit ratio filtering rejects marginal combined gains
//! 5. Verify non-linear compressed candidates coexist with IDENTITY compressed
//!    candidates in the pipeline
//! 6. Verify non-linear compression uses target neuron's squash when applicable

use neat_ai_discovery::analysis::candidate_compression::{
    compress_identity_candidates, compress_nonlinear_candidates,
};
use neat_ai_discovery::analysis::constants::coordinated_empirical_discount;
use neat_ai_discovery::{
    CandidateSynapseJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson, SynapseJson,
};

/// Helper: create a `CandidateSynapseJson`.
fn candidate(from: &str, to: &str, weight: f32, gain: f32) -> CandidateSynapseJson {
    CandidateSynapseJson {
        from_neuron_uuid: from.to_string(),
        to_neuron_uuid: to.to_string(),
        from_neuron_index: None,
        to_neuron_index: None,
        weight,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: gain,
        expected_creature_score_gain: gain,
        improved_count: 80,
        total_count: 100,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        outlier_reduction_info: None,
        prediction_confidence: 0.8,
        expected_score_gain_confidence_interval: [gain * 0.5, gain * 1.5],
        comment: None,
        variant_key: None,
    }
}

/// Helper: build a `NeuronJson`.
fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

/// Helper: build a `NeuronJson` with a specific squash function.
fn neuron_with_squash(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Helper: build a `SynapseJson`.
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Helper: build a minimal creature.
fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    let input_count = neurons.iter().filter(|n| n.neuron_type == "input").count();
    let output_count = neurons.iter().filter(|n| n.neuron_type == "output").count();
    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: output_count,
    }
}

/// Test 1: TANH compression in linear regime produces valid coordinated candidate.
#[test]
fn test_tanh_compression_linear_regime() {
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 0.05),
        candidate("input-b", "output-1", 0.4, 0.06),
    ];
    let creature = make_creature(
        vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-a", "output-1", 0.3),
            synapse("input-b", "output-1", 0.4),
        ],
    );

    let compressed = compress_nonlinear_candidates(&candidates, &creature);
    assert!(
        !compressed.is_empty(),
        "TANH compression should succeed for small weights in linear regime"
    );

    let c = &compressed[0];
    // 2 inputs → 4 operations: AddNeuron + 2 AddSynapse (inputs) + 1 AddSynapse (output).
    assert_eq!(c.operations.len(), 4, "Should have 4 operations");

    // Verify hidden neuron uses TANH.
    match &c.operations[0] {
        CoordinatedStructuralOpJson::AddNeuron { squash, bias, .. } => {
            assert_eq!(squash, "TANH", "Hidden neuron should use TANH");
            assert!((*bias).abs() < f32::EPSILON, "Bias should be 0.0");
        }
        _ => panic!("First operation should be AddNeuron"),
    }

    // Gain should be positive and discounted.
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Gain should be positive"
    );
}

/// Test 2: TANH compression with saturated inputs produces diminished or filtered gain.
#[test]
fn test_tanh_saturated_inputs_diminished() {
    // Large weights (3.0) push TANH deep into saturation.
    let candidates = vec![
        candidate("input-a", "output-1", 3.0, 0.05),
        candidate("input-b", "output-1", 3.0, 0.06),
    ];
    let creature = make_creature(
        vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-a", "output-1", 3.0),
            synapse("input-b", "output-1", 3.0),
        ],
    );

    let compressed = compress_nonlinear_candidates(&candidates, &creature);

    if !compressed.is_empty() {
        // If it survives filtering, gain must be less than linear sum.
        let linear_sum: f32 = candidates
            .iter()
            .map(|c| c.expected_creature_score_gain)
            .sum();
        // After discounting, it should be significantly less than the linear case.
        assert!(
            compressed[0].expected_creature_score_gain < linear_sum,
            "Saturated TANH gain should be less than linear sum"
        );
    }
    // It's also acceptable for the candidate to be completely filtered out.
}

/// Test 3: GELU compression produces valid coordinated candidate.
#[test]
fn test_gelu_compression() {
    let candidates = vec![
        candidate("input-a", "output-1", 0.5, 0.05),
        candidate("input-b", "output-1", 0.6, 0.06),
    ];
    let creature = make_creature(
        vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron_with_squash("output-1", "output", "GELU"),
        ],
        vec![
            synapse("input-a", "output-1", 0.5),
            synapse("input-b", "output-1", 0.6),
        ],
    );

    let compressed = compress_nonlinear_candidates(&candidates, &creature);

    if !compressed.is_empty() {
        // Verify GELU activation is used (target neuron's squash).
        match &compressed[0].operations[0] {
            CoordinatedStructuralOpJson::AddNeuron { squash, .. } => {
                assert_eq!(squash, "GELU", "Should use target neuron's GELU squash");
            }
            _ => panic!("First operation should be AddNeuron"),
        }
    }
}

/// Test 4: Benefit ratio filtering rejects marginal combined gains.
#[test]
fn test_benefit_ratio_filters_marginal_gains() {
    // With heavily saturated inputs (w=5.0), TANH(5) ≈ TANH(10) ≈ 1.0.
    // Combining adds no benefit over individual.
    let candidates = vec![
        candidate("input-a", "output-1", 5.0, 0.05),
        candidate("input-b", "output-1", 5.0, 0.06),
    ];
    let creature = make_creature(
        vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-a", "output-1", 5.0),
            synapse("input-b", "output-1", 5.0),
        ],
    );

    let compressed = compress_nonlinear_candidates(&candidates, &creature);
    assert!(
        compressed.is_empty(),
        "Heavily saturated inputs should fail benefit ratio check"
    );
}

/// Test 5: Non-linear and IDENTITY compressed candidates coexist in the pipeline.
#[test]
fn test_nonlinear_coexists_with_identity() {
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 0.05),
        candidate("input-b", "output-1", 0.4, 0.06),
    ];
    let creature = make_creature(
        vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-a", "output-1", 0.3),
            synapse("input-b", "output-1", 0.4),
        ],
    );

    let identity_compressed = compress_identity_candidates(&candidates, &creature);
    let nonlinear_compressed = compress_nonlinear_candidates(&candidates, &creature);

    // Both should be able to produce candidates from the same input.
    assert!(
        !identity_compressed.is_empty(),
        "IDENTITY compression should produce candidates"
    );

    // Combine them as the orchestration pipeline does.
    let mut all_compressed = identity_compressed;
    all_compressed.extend(nonlinear_compressed);

    // Should have at least the IDENTITY candidate.
    assert!(
        !all_compressed.is_empty(),
        "Combined pipeline should have at least one candidate"
    );

    // Verify IDENTITY and non-linear are distinct candidates.
    if all_compressed.len() >= 2 {
        let first_squash = match &all_compressed[0].operations[0] {
            CoordinatedStructuralOpJson::AddNeuron { squash, .. } => squash.clone(),
            _ => String::new(),
        };
        let second_squash = match &all_compressed[1].operations[0] {
            CoordinatedStructuralOpJson::AddNeuron { squash, .. } => squash.clone(),
            _ => String::new(),
        };
        assert_ne!(
            first_squash, second_squash,
            "IDENTITY and non-linear candidates should use different squash functions"
        );
    }
}

/// Test 6: Non-linear compression uses target neuron's squash when it is TANH.
#[test]
fn test_uses_target_tanh_squash() {
    let candidates = vec![
        candidate("input-a", "hidden-1", 0.3, 0.05),
        candidate("input-b", "hidden-1", 0.4, 0.06),
    ];
    let creature = make_creature(
        vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron_with_squash("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-a", "hidden-1", 0.3),
            synapse("input-b", "hidden-1", 0.4),
            synapse("hidden-1", "output-1", 0.5),
        ],
    );

    let compressed = compress_nonlinear_candidates(&candidates, &creature);

    if !compressed.is_empty() {
        match &compressed[0].operations[0] {
            CoordinatedStructuralOpJson::AddNeuron { squash, .. } => {
                assert_eq!(squash, "TANH", "Should use target's TANH squash");
            }
            _ => panic!("First operation should be AddNeuron"),
        }
    }
}

/// Test 7: Non-linear compression applies operation-count discount.
#[test]
fn test_nonlinear_operation_count_discount() {
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 0.05),
        candidate("input-b", "output-1", 0.4, 0.06),
    ];
    let creature = make_creature(
        vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-a", "output-1", 0.3),
            synapse("input-b", "output-1", 0.4),
        ],
    );

    let compressed = compress_nonlinear_candidates(&candidates, &creature);

    if !compressed.is_empty() {
        let c = &compressed[0];
        // 4 operations → empirical discount for 4+ ops.
        // The gain should be significantly less than the raw combined gain.
        let max_possible = (0.05_f32 + 0.06) * coordinated_empirical_discount(4);
        assert!(
            c.expected_creature_score_gain <= max_possible + 1e-6,
            "Discounted gain should not exceed max possible: {} > {}",
            c.expected_creature_score_gain,
            max_possible
        );
    }
}
