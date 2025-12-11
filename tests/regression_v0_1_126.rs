//! REGRESSION TESTS for impact calculation (v0.1.126 → v0.1.145)
//!
//! ## History:
//!
//! v0.1.126 introduced NORMALISED impact calculation:
//!   impact = |weight| / total_inbound × downstream_impact
//!
//! This was WRONG. Production data (Dec 2024) showed impacts underestimated
//! by up to 145 BILLION times:
//!   - Calculated: 1.39e-12
//!   - Actual: -20% error increase (0.2)
//!
//! v0.1.145 fixes this with ABSOLUTE impact:
//!   impact = |weight| × downstream_impact
//!
//! ## Why normalisation was wrong:
//!
//! When you remove a neuron, the network doesn't "redistribute" its inputs.
//! If a neuron contributes 3 units to an output with total input 100:
//!   - Normalised said: "removing loses 3% of output" (impact 0.03)
//!   - Reality: "removing loses 3 units of contribution" (impact 3.0)
//!
//! The normalised approach answered "what share of blame?" not "what happens
//! when removed?" - a fundamentally different question.

mod common;

use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Test: Impact is ABSOLUTE (weight × downstream_impact), not normalised.
///
/// v0.1.145: This test was updated to verify the fix. Previously it tested
/// normalised impact which caused massive underestimation in production.
#[test]
fn test_impact_is_absolute_not_normalised() {
    // Network:
    //   candidate → output (weight 3.0)
    //   other inputs → output (total weight 97.0)
    //
    // With ABSOLUTE impact: candidate impact = 3.0 × 1.0 = 3.0
    // (The old normalised approach gave 0.03, which was wrong)
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
                    synapse_type: None,
                },
                SynapseJson {
                    from_uuid: "candidate".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 3.0,
                    synapse_type: None,
                },
            ];
            // Add 97 units of weight from other inputs (shouldn't affect candidate's impact)
            for i in 1..10 {
                synapses.push(SynapseJson {
                    from_uuid: format!("input-{i}"),
                    to_uuid: "output-0".to_string(),
                    weight: 97.0 / 9.0, // ~10.8 each, total ~97
                    synapse_type: None,
                });
            }
            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let candidate_impact = *impacts.get("candidate").unwrap_or(&0.0);

    // ABSOLUTE impact: weight × downstream = 3.0 × 1.0 = 3.0
    // (Other inputs don't dilute this - they're independent contributions)
    assert!(
        (candidate_impact - 3.0).abs() < 0.01,
        "candidate impact should be 3.0 (absolute weight × downstream), got {candidate_impact}"
    );
}

/// Test: Deep network impact multiplies weights (not fractions).
#[test]
fn test_deep_network_absolute_impact() {
    // Network:
    //   candidate → hidden (weight 1.0)
    //   hidden → output (weight 2.0)
    //
    // With ABSOLUTE impact: candidate = 1.0 × 2.0 = 2.0
    // (The old normalised approach gave 0.05, which was wrong)
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
                    synapse_type: None,
                },
                // candidate → hidden (1.0)
                SynapseJson {
                    from_uuid: "candidate".to_string(),
                    to_uuid: "hidden".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                },
                // hidden → output (2.0)
                SynapseJson {
                    from_uuid: "hidden".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 2.0,
                    synapse_type: None,
                },
            ];
            // Add other synapses (shouldn't affect candidate's impact)
            for i in 1..10 {
                synapses.push(SynapseJson {
                    from_uuid: format!("input-{i}"),
                    to_uuid: "hidden".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                });
            }
            synapses.push(SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 2.0,
                synapse_type: None,
            });
            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let candidate_impact = *impacts.get("candidate").unwrap_or(&0.0);

    // ABSOLUTE: candidate → hidden (1.0) × hidden → output (2.0) = 2.0
    assert!(
        (candidate_impact - 2.0).abs() < 0.01,
        "candidate impact should be 2.0 (1.0 × 2.0), got {candidate_impact}"
    );
}

/// Test: Negligible weight neuron has negligible absolute impact.
#[test]
fn test_negligible_weight_has_negligible_impact() {
    // A neuron with weight 1e-8 to output should have impact ~1e-8
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
                synapse_type: None,
            },
            // Very small weight to output
            SynapseJson {
                from_uuid: "negligible".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1e-8,
                synapse_type: None,
            },
            // Normal weight direct connection
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let impact = *impacts.get("negligible").unwrap_or(&0.0);

    // With absolute impact: 1e-8 × 1.0 = 1e-8
    // The direct connection doesn't affect this neuron's impact
    assert!(
        impact < 1e-6 && impact > 1e-10,
        "negligible neuron impact should be ~1e-8, got {impact}"
    );
}
