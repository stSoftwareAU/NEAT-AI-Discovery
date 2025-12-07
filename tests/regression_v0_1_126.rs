//! REGRESSION TESTS for impact calculation
//!
//! These tests verify that impact is correctly NORMALISED by total inbound weight.
//!
//! ## Key insight:
//!
//! If an output neuron has 100 incoming synapses with total |weight| = 343,
//! and one neuron contributes weight 3.0, that neuron provides:
//!   3.0 / 343 ≈ 0.9% of the output's input
//!
//! Removing that neuron reduces output by ~0.9%, not 300%!
//!
//! This normalisation is recursive through the network.

mod common;

use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Test: Impact is normalised by total inbound weight.
///
/// When a target has multiple inputs, each input's impact is proportional
/// to its share of the total input, not its absolute weight.
#[test]
fn test_impact_normalised_by_total_inbound() {
    // Network:
    //   candidate → output (weight 3.0)
    //   other inputs → output (total weight 97.0)
    //
    // candidate provides 3/100 = 3% of output's input
    // So candidate's impact should be ~0.03, not 3.0
    let creature = CreatureJson {
        input: 10,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "candidate".to_string(),
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
        synapses: {
            let mut synapses = vec![
                // Candidate → output with weight 3.0
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "candidate".to_string(),
                    weight: 1.0,
                },
                SynapseJson {
                    from_uuid: "candidate".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 3.0,
                },
            ];
            // Add 97 units of weight from other inputs
            for i in 1..10 {
                synapses.push(SynapseJson {
                    from_uuid: format!("input-{i}"),
                    to_uuid: "output-0".to_string(),
                    weight: 97.0 / 9.0, // ~10.8 each, total ~97
                });
            }
            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let candidate_impact = *impacts.get("candidate").unwrap_or(&0.0);

    // Total inbound to output = 3.0 + 97.0 = 100.0
    // candidate's share = 3.0 / 100.0 = 0.03 (3%)
    assert!(
        (candidate_impact - 0.03).abs() < 0.005,
        "candidate impact should be ~0.03 (3% of output's input), got {candidate_impact}"
    );
}

/// Test: Deep network impact accumulates normalisation at each step.
#[test]
fn test_deep_network_normalised_impact() {
    // Network:
    //   candidate → hidden (weight 1.0, hidden has total inbound 10)
    //   hidden → output (weight 2.0, output has total inbound 4)
    //
    // candidate's share of hidden = 1/10 = 10%
    // hidden's share of output = 2/4 = 50%
    // candidate's impact = 0.1 × 0.5 = 0.05 (5%)
    let creature = CreatureJson {
        input: 10,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "candidate".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden".to_string(),
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
        synapses: {
            let mut synapses = vec![
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "candidate".to_string(),
                    weight: 1.0,
                },
                // candidate → hidden (1.0 out of total 10)
                SynapseJson {
                    from_uuid: "candidate".to_string(),
                    to_uuid: "hidden".to_string(),
                    weight: 1.0,
                },
                // hidden → output (2.0 out of total 4)
                SynapseJson {
                    from_uuid: "hidden".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 2.0,
                },
            ];
            // Add 9 more units to hidden (total inbound = 10)
            for i in 1..10 {
                synapses.push(SynapseJson {
                    from_uuid: format!("input-{i}"),
                    to_uuid: "hidden".to_string(),
                    weight: 1.0,
                });
            }
            // Add 2 more units to output (total inbound = 4)
            synapses.push(SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 2.0,
            });
            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let candidate_impact = *impacts.get("candidate").unwrap_or(&0.0);

    // candidate → hidden: 1/10 = 0.1
    // hidden → output: 2/4 = 0.5
    // combined: 0.1 × 0.5 = 0.05
    assert!(
        (candidate_impact - 0.05).abs() < 0.01,
        "candidate impact should be ~0.05 (10% × 50%), got {candidate_impact}"
    );
}

/// Test: Negligible weight neuron has negligible normalised impact.
#[test]
fn test_negligible_weight_has_negligible_impact() {
    // A neuron with weight 1e-8 to output (which has total inbound ~1.0)
    // should have impact ~1e-8
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "negligible".to_string(),
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
                from_uuid: "input-0".to_string(),
                to_uuid: "negligible".to_string(),
                weight: 1.0,
            },
            SynapseJson {
                from_uuid: "negligible".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1e-8,
            },
            // Add another input so total isn't just the negligible one
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
            },
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let negligible_impact = *impacts.get("negligible").unwrap_or(&0.0);

    // Total inbound to output = 1e-8 + 1.0 ≈ 1.0
    // negligible's share ≈ 1e-8 / 1.0 = 1e-8
    assert!(
        negligible_impact < 1e-6 && negligible_impact > 0.0,
        "negligible neuron should have tiny impact, got {negligible_impact}"
    );
}
