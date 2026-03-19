//! Tests for Issue #570: Skip-connection discovery — identify beneficial residual
//! connections across layers.
//!
//! Analyses neuron topological depth and gradient attenuation to recommend
//! skip connections (addSynapse) that bridge large depth gaps in deep networks.
//!
//! ## TDD Plan
//! 1. Test layer depth computation from inputs
//! 2. Test gradient attenuation detection in deep networks
//! 3. Test candidate generation prioritises largest depth gaps
//! 4. Test shallow networks produce no candidates
//! 5. Test conservative initial weights (near zero)
//! 6. Test existing skip connections are not duplicated
//! 7. Test edge cases (empty network, no hidden neurons, insufficient samples)

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::skip_connection::{
    detect_skip_connection_candidates, skip_connections_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a `DiscoverRecord` with custom errors.
fn record(neuron_uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: None,
        activation,
        errors,
    }
}

/// Build records for a neuron: `count` samples with varying activations and errors.
fn make_records(uuid: &str, count: u32, activation_scale: f32, error: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| {
            record(
                uuid,
                i,
                (i as f32 * 0.1).sin() * activation_scale,
                vec![error * (1.0 + (i as f32 * 0.05).sin())],
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test network builders
// ---------------------------------------------------------------------------

/// Deep network: input-0 → h0 → h1 → h2 → h3 → h4 → output-0
/// h0 is at depth 1 from input, h4 at depth 5. The deep neurons (h2–h4)
/// have significantly attenuated error gradients compared to shallow ones.
fn deep_network_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h0".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h1".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h2".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h3".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h4".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "h0".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h0".into(),
                to_uuid: "h1".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".into(),
                to_uuid: "h2".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h2".into(),
                to_uuid: "h3".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h3".into(),
                to_uuid: "h4".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h4".into(),
                to_uuid: "output-0".into(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    }
}

/// Shallow network: input-0 → h0 → output-0, input-1 → h1 → output-0
/// All neurons at depth <= 1 from input. No gradient attenuation expected.
fn shallow_network_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h0".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h1".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "h0".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".into(),
                to_uuid: "h1".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h0".into(),
                to_uuid: "output-0".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".into(),
                to_uuid: "output-0".into(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 1,
    }
}

/// Deep network with two inputs feeding different depths, creating
/// gradient attenuation at deeper levels.
fn multi_input_deep_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h0".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h1".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h2".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h3".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "h0".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".into(),
                to_uuid: "h1".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h0".into(),
                to_uuid: "h2".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".into(),
                to_uuid: "h2".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h2".into(),
                to_uuid: "h3".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h3".into(),
                to_uuid: "output-0".into(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 1,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test 1: Deep network with gradient attenuation produces skip connection candidates.
#[test]
fn test_deep_network_detects_skip_connection_candidates() {
    let creature = deep_network_creature();

    // Simulate gradient attenuation: shallow neurons have high error,
    // deep neurons have significantly lower error (attenuated gradients).
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.8, 0.50)),
        ("h1".into(), make_records("h1", 30, 0.6, 0.35)),
        ("h2".into(), make_records("h2", 30, 0.4, 0.08)),
        ("h3".into(), make_records("h3", 30, 0.3, 0.05)),
        ("h4".into(), make_records("h4", 30, 0.2, 0.03)),
    ];

    let candidates = detect_skip_connection_candidates(&creature, &neuron_records);

    assert!(
        !candidates.is_empty(),
        "Deep network with gradient attenuation should produce skip-connection candidates"
    );
}

/// Test 2: Shallow network produces no skip-connection candidates.
#[test]
fn test_shallow_network_no_candidates() {
    let creature = shallow_network_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.5, 0.2)),
        ("h1".into(), make_records("h1", 30, 0.5, 0.2)),
    ];

    let candidates = detect_skip_connection_candidates(&creature, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Shallow network (depth <= 2) should not produce skip-connection candidates, found {}",
        candidates.len()
    );
}

/// Test 3: Candidates prioritise largest depth gaps.
#[test]
fn test_candidates_prioritise_largest_depth_gaps() {
    let creature = deep_network_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.8, 0.50)),
        ("h1".into(), make_records("h1", 30, 0.6, 0.35)),
        ("h2".into(), make_records("h2", 30, 0.4, 0.08)),
        ("h3".into(), make_records("h3", 30, 0.3, 0.05)),
        ("h4".into(), make_records("h4", 30, 0.2, 0.03)),
    ];

    let candidates = detect_skip_connection_candidates(&creature, &neuron_records);

    if candidates.len() >= 2 {
        // First candidate should have a larger depth gap than later ones
        assert!(
            candidates[0].depth_gap >= candidates[1].depth_gap,
            "Candidates should be sorted by depth gap (largest first): {} vs {}",
            candidates[0].depth_gap,
            candidates[1].depth_gap
        );
    }
}

/// Test 4: Candidates have positive estimated improvement.
#[test]
fn test_skip_connection_candidates_have_positive_improvement() {
    let creature = deep_network_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.8, 0.50)),
        ("h1".into(), make_records("h1", 30, 0.6, 0.35)),
        ("h2".into(), make_records("h2", 30, 0.4, 0.08)),
        ("h3".into(), make_records("h3", 30, 0.3, 0.05)),
        ("h4".into(), make_records("h4", 30, 0.2, 0.03)),
    ];

    let candidates = detect_skip_connection_candidates(&creature, &neuron_records);

    for c in &candidates {
        assert!(
            c.estimated_improvement > 0.0,
            "All skip-connection candidates should have positive estimated improvement, got {}",
            c.estimated_improvement
        );
    }
}

/// Test 5: Conversion to coordinated candidates uses addSynapse with conservative weights.
#[test]
fn test_coordinated_candidates_use_conservative_weights() {
    let creature = deep_network_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.8, 0.50)),
        ("h1".into(), make_records("h1", 30, 0.6, 0.35)),
        ("h2".into(), make_records("h2", 30, 0.4, 0.08)),
        ("h3".into(), make_records("h3", 30, 0.3, 0.05)),
        ("h4".into(), make_records("h4", 30, 0.2, 0.03)),
    ];

    let detected = detect_skip_connection_candidates(&creature, &neuron_records);
    assert!(!detected.is_empty(), "Should detect candidates first");

    let coordinated = skip_connections_to_coordinated_candidates(&detected, &creature);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated structural candidates"
    );

    for c in &coordinated {
        assert!(
            c.expected_creature_score_gain > 0.0,
            "Coordinated candidate should have positive expected gain"
        );

        // Verify all operations are addSynapse with conservative weights
        for op in &c.operations {
            let json = serde_json::to_string(op).unwrap_or_default();
            assert!(
                json.contains("addSynapse"),
                "Skip-connection candidates should use addSynapse, got: {json}"
            );

            // Extract weight from the JSON and check it's conservative (near zero)
            if let Some(w_start) = json.find("\"weight\":") {
                let weight_str = &json[w_start + 9..];
                if let Some(end) = weight_str.find([',', '}']) {
                    let weight: f32 = weight_str[..end].parse().unwrap_or(999.0);
                    assert!(
                        weight.abs() <= 0.15,
                        "Skip-connection weight should be conservative (near zero), got {weight}"
                    );
                }
            }
        }
    }
}

/// Test 6: Existing skip connections are not duplicated.
#[test]
fn test_skip_connection_existing_skip_connection_not_duplicated() {
    // Deep chain with an existing skip: input-0 → h0 directly connected
    let mut creature = deep_network_creature();
    // Add an existing skip: input-0 → h4
    creature.synapses.push(SynapseJson {
        from_uuid: "input-0".into(),
        to_uuid: "h4".into(),
        weight: 0.1,
        synapse_type: None,
    });

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.8, 0.50)),
        ("h1".into(), make_records("h1", 30, 0.6, 0.35)),
        ("h2".into(), make_records("h2", 30, 0.4, 0.08)),
        ("h3".into(), make_records("h3", 30, 0.3, 0.05)),
        ("h4".into(), make_records("h4", 30, 0.2, 0.03)),
    ];

    let detected = detect_skip_connection_candidates(&creature, &neuron_records);
    let coordinated = skip_connections_to_coordinated_candidates(&detected, &creature);

    // No candidate should suggest input-0 → h4 since it already exists
    for c in &coordinated {
        let json = serde_json::to_string(&c.operations).unwrap_or_default();
        let duplicates = json.contains("\"fromNeuronUuid\":\"input-0\"")
            && json.contains("\"toNeuronUuid\":\"h4\"");
        assert!(
            !duplicates,
            "Should not suggest skip connection that already exists (input-0 → h4)"
        );
    }
}

/// Test 7: Empty network (no hidden neurons) produces no candidates.
#[test]
fn test_skip_connection_no_hidden_neurons_no_candidates() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".into(),
            to_uuid: "output-0".into(),
            weight: 0.5,
            synapse_type: None,
        }],
        input: 1,
        output: 1,
    };

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![];
    let candidates = detect_skip_connection_candidates(&creature, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Network with no hidden neurons should produce no candidates"
    );
}

/// Test 8: Insufficient samples do not trigger detection.
#[test]
fn test_skip_connection_insufficient_samples_no_candidates() {
    let creature = deep_network_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 5, 0.8, 0.50)),
        ("h1".into(), make_records("h1", 5, 0.6, 0.35)),
        ("h2".into(), make_records("h2", 5, 0.4, 0.08)),
        ("h3".into(), make_records("h3", 5, 0.3, 0.05)),
        ("h4".into(), make_records("h4", 5, 0.2, 0.03)),
    ];

    let candidates = detect_skip_connection_candidates(&creature, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Insufficient samples should produce no candidates"
    );
}

/// Test 9: No records at all produces no candidates.
#[test]
fn test_no_records_no_candidates() {
    let creature = deep_network_creature();
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let candidates = detect_skip_connection_candidates(&creature, &neuron_records);

    assert!(
        candidates.is_empty(),
        "No records should produce no candidates"
    );
}

/// Test 10: Multi-input deep network detects candidates for deep neurons.
#[test]
fn test_multi_input_deep_network() {
    let creature = multi_input_deep_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.7, 0.40)),
        ("h1".into(), make_records("h1", 30, 0.7, 0.40)),
        ("h2".into(), make_records("h2", 30, 0.4, 0.08)),
        ("h3".into(), make_records("h3", 30, 0.2, 0.04)),
    ];

    let candidates = detect_skip_connection_candidates(&creature, &neuron_records);

    // Deep neurons (h2, h3) with attenuated gradients should be candidates
    if !candidates.is_empty() {
        // All candidates should be for deep neurons (depth > 2)
        for c in &candidates {
            assert!(
                c.target_depth > 2,
                "Skip-connection candidates should only target deep neurons, got depth {} for {}",
                c.target_depth,
                c.target_uuid
            );
        }
    }
}

/// Test 11: Source neuron is shallow (input or shallow hidden).
#[test]
fn test_source_is_shallow_neuron() {
    let creature = deep_network_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.8, 0.50)),
        ("h1".into(), make_records("h1", 30, 0.6, 0.35)),
        ("h2".into(), make_records("h2", 30, 0.4, 0.08)),
        ("h3".into(), make_records("h3", 30, 0.3, 0.05)),
        ("h4".into(), make_records("h4", 30, 0.2, 0.03)),
    ];

    let candidates = detect_skip_connection_candidates(&creature, &neuron_records);

    for c in &candidates {
        assert!(
            c.source_depth < c.target_depth,
            "Source should be shallower than target: source depth {} vs target depth {}",
            c.source_depth,
            c.target_depth
        );
    }
}

/// Test 12: Candidates are sorted by estimated improvement (best first).
#[test]
fn test_skip_connection_candidates_sorted_by_improvement() {
    let creature = deep_network_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.8, 0.50)),
        ("h1".into(), make_records("h1", 30, 0.6, 0.35)),
        ("h2".into(), make_records("h2", 30, 0.4, 0.08)),
        ("h3".into(), make_records("h3", 30, 0.3, 0.05)),
        ("h4".into(), make_records("h4", 30, 0.2, 0.03)),
    ];

    let candidates = detect_skip_connection_candidates(&creature, &neuron_records);

    for window in candidates.windows(2) {
        assert!(
            window[0].estimated_improvement >= window[1].estimated_improvement,
            "Candidates should be sorted by estimated improvement (best first)"
        );
    }
}

/// Test 13: Uniform error across depths (no attenuation) produces no candidates.
#[test]
fn test_uniform_error_no_attenuation_no_candidates() {
    let creature = deep_network_creature();

    // All neurons have the same error — no gradient attenuation
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.5, 0.20)),
        ("h1".into(), make_records("h1", 30, 0.5, 0.20)),
        ("h2".into(), make_records("h2", 30, 0.5, 0.20)),
        ("h3".into(), make_records("h3", 30, 0.5, 0.20)),
        ("h4".into(), make_records("h4", 30, 0.5, 0.20)),
    ];

    let candidates = detect_skip_connection_candidates(&creature, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Uniform error (no gradient attenuation) should produce no candidates, found {}",
        candidates.len()
    );
}
