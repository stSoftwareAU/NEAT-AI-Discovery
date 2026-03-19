//! REGRESSION TESTS for impact calculation (Issue #130 fix)
//!
//! ## History:
//!
//! v0.1.126 introduced NORMALISED impact calculation (which was correct):
//!   impact = |weight| / `total_inbound` × `downstream_impact`
//!
//! v0.1.145 changed to ABSOLUTE impact (which was WRONG for prediction discounting):
//!   impact = |weight| × `downstream_impact`
//!
//! The absolute approach was introduced to fix removal candidate assessment, but it
//! broke prediction discounting by allowing hidden neurons to have impact >= 1.0.
//!
//! ## Issue #130: Why absolute was wrong for prediction discounting
//!
//! Production data showed `targetNeuronImpact = 1.0` for hidden neurons with large
//! weights to outputs. This caused predictions to NOT be discounted, leading to
//! massive overestimation (35% predicted vs 0% actual improvement).
//!
//! ## v0.2.1 fix (Issue #130): Restore normalised calculation
//!
//! The impact calculation now uses the documented formula:
//!   impact = |weight| / `total_inbound` × `downstream_impact`
//!
//! This is correct because it measures ATTRIBUTION (fraction of influence),
//! not SENSITIVITY (absolute change). For prediction discounting, we need attribution.
//!
//! Key properties of normalised impact:
//! - Output neurons: impact = 1.0 (always)
//! - Hidden neurons: impact <= 1.0 (depending on fraction of total input weight)
//! - Sole input: impact = 1.0 (100% of target's input comes from this neuron)
//! - Competing inputs: impact < 1.0 (shared influence)

use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Test: Impact is NORMALISED as documented.
///
/// v0.2.1 (Issue #130): This test verifies the fix. The old absolute formula
/// gave impact = 3.0, which was then clamped to 1.0, incorrectly treating the
/// hidden neuron as having full output-level impact.
#[test]
fn test_impact_is_normalised_as_documented() {
    // Network:
    //   candidate → output (weight 3.0)
    //   other inputs → output (total weight 97.0)
    //
    // With NORMALISED impact: candidate impact = 3.0 / 100.0 × 1.0 = 0.03
    // (The incorrect absolute approach gave 3.0)
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
            // Add 97 units of weight from other inputs (dilutes candidate's impact)
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

    // NORMALISED impact: weight / total × downstream = 3.0 / 100.0 × 1.0 = 0.03
    // Other inputs DILUTE this - they're competing for influence
    assert!(
        (candidate_impact - 0.03).abs() < 0.01,
        "candidate impact should be 0.03 (3/100 normalised), got {candidate_impact}"
    );

    // Crucially: impact < 1.0 (Issue #130 fix)
    assert!(
        candidate_impact < 1.0,
        "hidden neuron should have impact < 1.0, got {candidate_impact}"
    );
}

/// Test: Deep network impact uses normalised weights at each layer.
///
/// v0.2.1 (Issue #130): With normalisation, impact dilutes through deep networks.
#[test]
fn test_deep_network_normalised_impact() {
    // Network:
    //   candidate → hidden (weight 1.0)
    //   hidden → output (weight 2.0)
    //   + 9 other inputs to hidden (weight 1.0 each)
    //   + 1 other input to output (weight 2.0)
    //
    // hidden → output: 2.0 / (2.0 + 2.0) = 0.5 (half the output's input weight)
    // candidate → hidden: 1.0 / (1.0 + 9.0) = 0.1 (10% of hidden's input weight)
    // candidate total: 0.1 × 0.5 = 0.05
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
            // Add 9 other inputs to hidden (dilutes candidate's contribution)
            for i in 1..10 {
                synapses.push(SynapseJson {
                    from_uuid: format!("input-{i}"),
                    to_uuid: "hidden".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                });
            }
            // Add direct input to output (dilutes hidden's contribution)
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

    // NORMALISED:
    // hidden → output: 2.0 / (2.0 + 2.0) × 1.0 = 0.5
    // candidate → hidden: 1.0 / 10.0 × 0.5 = 0.05
    assert!(
        (candidate_impact - 0.05).abs() < 0.01,
        "candidate impact should be 0.05 (normalised through layers), got {candidate_impact}"
    );
}

/// Test: Negligible weight neuron has negligible normalised impact.
///
/// Even with normalisation, a tiny weight gives tiny impact.
#[test]
fn test_negligible_weight_has_negligible_impact() {
    // A neuron with weight 1e-8 to output (competing with weight 1.0)
    // Normalised impact: 1e-8 / (1e-8 + 1.0) × 1.0 ≈ 1e-8
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
            // Normal weight direct connection (competes with negligible)
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

    // Normalised impact: 1e-8 / (1e-8 + 1.0) ≈ 1e-8
    assert!(
        impact < 1e-6 && impact > 1e-10,
        "negligible neuron impact should be ~1e-8, got {impact}"
    );
}
