//! Tests for Issue #835: Reduce lock contention on shared cache in impact computation.
//!
//! Verifies that impact computation produces identical results after migrating
//! from Mutex<HashMap> to DashMap for the shared impact cache. These tests
//! exercise the concurrent cache with various network topologies to ensure
//! deterministic results under parallel execution.

use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Verify that a wide network (many neurons → single output) produces
/// correct, deterministic impact values with the concurrent cache.
#[test]
fn issue_835_wide_network_impact_values_are_deterministic() {
    let width = 50;
    let mut neurons = Vec::with_capacity(width + 1);
    let mut synapses = Vec::with_capacity(width);

    for i in 0..width {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
        synapses.push(SynapseJson {
            from_uuid: format!("hidden-{i}"),
            to_uuid: "output-0".to_string(),
            weight: 1.0,
            synapse_type: None,
        });
    }
    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    let creature = CreatureJson {
        input: 0,
        output: 1,
        neurons,
        synapses,
    };

    // Run multiple times to verify determinism under parallel execution
    let baseline = compute_impacts_public(&creature);
    for run in 0..5 {
        let impacts = compute_impacts_public(&creature);
        for (uuid, &expected) in &baseline {
            let actual = impacts.get(uuid).copied().unwrap_or(f32::NAN);
            assert!(
                (actual - expected).abs() < 1e-6,
                "Run {run}: impact for {uuid} differs: expected {expected}, got {actual}"
            );
        }
    }

    // Output should be 1.0
    assert_eq!(baseline.get("output-0").copied().unwrap_or(0.0), 1.0);

    // Each hidden neuron has weight 1.0 out of total 50.0 → impact = 1/50 = 0.02
    let expected_impact = 1.0 / width as f32;
    for i in 0..width {
        let uuid = format!("hidden-{i}");
        let impact = baseline.get(&uuid).copied().unwrap_or(0.0);
        assert!(
            (impact - expected_impact).abs() < 1e-6,
            "{uuid}: expected {expected_impact}, got {impact}"
        );
    }
}

/// Verify that a deep chain network produces correct impacts with the concurrent cache.
#[test]
fn issue_835_deep_chain_impact_values_are_deterministic() {
    let depth = 30;
    let mut neurons = Vec::with_capacity(depth + 1);
    let mut synapses = Vec::with_capacity(depth);

    for i in 0..depth {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
        if i > 0 {
            synapses.push(SynapseJson {
                from_uuid: format!("hidden-{}", i - 1),
                to_uuid: format!("hidden-{i}"),
                weight: 0.9,
                synapse_type: None,
            });
        }
    }
    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });
    synapses.push(SynapseJson {
        from_uuid: format!("hidden-{}", depth - 1),
        to_uuid: "output-0".to_string(),
        weight: 1.0,
        synapse_type: None,
    });

    let creature = CreatureJson {
        input: 0,
        output: 1,
        neurons,
        synapses,
    };

    // Run multiple times to verify determinism
    let baseline = compute_impacts_public(&creature);
    for run in 0..5 {
        let impacts = compute_impacts_public(&creature);
        for (uuid, &expected) in &baseline {
            let actual = impacts.get(uuid).copied().unwrap_or(f32::NAN);
            assert!(
                (actual - expected).abs() < 1e-6,
                "Run {run}: impact for {uuid} differs: expected {expected}, got {actual}"
            );
        }
    }

    // In a sole-connection chain, every neuron has impact 1.0
    // (each is the only input to its successor)
    for i in 0..depth {
        let uuid = format!("hidden-{i}");
        let impact = baseline.get(&uuid).copied().unwrap_or(0.0);
        assert!(
            (impact - 1.0).abs() < 1e-3,
            "{uuid}: expected ~1.0, got {impact}"
        );
    }
}

/// Verify that a diamond mesh network produces correct impacts with the concurrent cache.
/// This topology creates maximum contention as many neurons share paths to the output.
#[test]
fn issue_835_diamond_mesh_impact_values_are_deterministic() {
    let layers = 5;
    let width = 10;
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();

    for layer in 0..layers {
        for i in 0..width {
            neurons.push(NeuronJson {
                uuid: format!("h-{layer}-{i}"),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            });
            if layer > 0 {
                for j in 0..width {
                    synapses.push(SynapseJson {
                        from_uuid: format!("h-{}-{j}", layer - 1),
                        to_uuid: format!("h-{layer}-{i}"),
                        weight: 1.0 / width as f32,
                        synapse_type: None,
                    });
                }
            }
        }
    }
    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });
    for i in 0..width {
        synapses.push(SynapseJson {
            from_uuid: format!("h-{}-{i}", layers - 1),
            to_uuid: "output-0".to_string(),
            weight: 1.0 / width as f32,
            synapse_type: None,
        });
    }

    let creature = CreatureJson {
        input: 0,
        output: 1,
        neurons,
        synapses,
    };

    // Run multiple times to verify determinism
    let baseline = compute_impacts_public(&creature);
    for run in 0..5 {
        let impacts = compute_impacts_public(&creature);
        for (uuid, &expected) in &baseline {
            let actual = impacts.get(uuid).copied().unwrap_or(f32::NAN);
            assert!(
                (actual - expected).abs() < 1e-6,
                "Run {run}: impact for {uuid} differs: expected {expected}, got {actual}"
            );
        }
    }

    // All neurons should have finite positive impacts
    for (uuid, &impact) in &baseline {
        assert!(
            impact.is_finite() && impact > 0.0,
            "{uuid} should have finite positive impact, got {impact}"
        );
    }

    // Output should be 1.0
    assert_eq!(baseline.get("output-0").copied().unwrap_or(0.0), 1.0);
}

/// Verify that cyclic networks produce finite positive impacts with the concurrent cache.
/// Note: Cycles with parallel execution are inherently non-deterministic because
/// thread scheduling determines which path hits cycle detection first. We verify
/// that results are always finite and positive rather than exact values.
#[test]
fn issue_835_cyclic_network_produces_finite_positive_impacts() {
    let creature = CreatureJson {
        input: 0,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "h-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "h-1".to_string(),
                to_uuid: "h-2".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h-2".to_string(),
                to_uuid: "h-1".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h-2".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
    };

    // Run multiple times — all runs must produce finite positive results
    for run in 0..10 {
        let impacts = compute_impacts_public(&creature);

        assert_eq!(
            impacts.get("output-0").copied().unwrap_or(0.0),
            1.0,
            "Run {run}: output should be 1.0"
        );

        let h1 = impacts.get("h-1").copied().unwrap_or(0.0);
        let h2 = impacts.get("h-2").copied().unwrap_or(0.0);
        assert!(
            h1.is_finite() && h1 > 0.0,
            "Run {run}: h-1 impact should be finite positive, got {h1}"
        );
        assert!(
            h2.is_finite() && h2 > 0.0,
            "Run {run}: h-2 impact should be finite positive, got {h2}"
        );
    }
}
