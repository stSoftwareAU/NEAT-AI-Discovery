//! Regression tests for Issue #130: targetNeuronImpact is obviously wrong
//!
//! ## The Fundamental Flaw
//!
//! The impact calculation was using ABSOLUTE weights instead of NORMALISED weights:
//!
//! - **Incorrect (absolute)**: `contribution = |weight| × child_impact`
//! - **Correct (normalised)**: `contribution = |weight| / total_inbound × child_impact`
//!
//! The absolute formula gives "sensitivity" (how much does output change per unit change).
//! The normalised formula gives "attribution" (what fraction of output is due to this neuron).
//!
//! For prediction discounting, we need ATTRIBUTION, which is always ≤ 1.0.
//! The absolute formula could give values > 1.0 for large weights, which was then
//! clamped to 1.0, making hidden neurons appear to have the same impact as outputs.
//!
//! ## The Fix
//!
//! Use the normalised formula as documented in IMPACT_CALCULATION.md:
//!   `contribution = |weight| / total_inbound_weight × child_impact`
//!
//! This mathematically guarantees that hidden neurons always have impact < 1.0
//! (unless they are the ONLY input to an output, in which case impact = 1.0 is correct).

use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Issue #130: Hidden neuron with large weight should NOT have impact > 1.0
///
/// Scenario: Hidden -> Output with weight 3.0 (only input)
///
/// With absolute formula: impact = 3.0 × 1.0 = 3.0 (WRONG!)
/// With normalised formula: impact = 3.0 / 3.0 × 1.0 = 1.0 (correct for sole input)
#[test]
fn test_sole_input_hidden_neuron_has_impact_one() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
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
            // Hidden -> Output with large weight (sole input)
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 3.0,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    };

    let impacts = compute_impacts_public(&creature);

    // Output neuron should have impact = 1.0
    let output_impact = *impacts.get("output-0").expect("output should have impact");
    assert!(
        (output_impact - 1.0).abs() < 0.001,
        "Output neuron should have impact = 1.0, got {output_impact}"
    );

    // Hidden neuron is the SOLE input to output, so its normalised impact = 1.0
    // (3.0 / 3.0 × 1.0 = 1.0)
    // This is correct: if it's the only input, it has 100% influence
    let hidden_impact = *impacts.get("hidden-1").expect("hidden should have impact");
    assert!(
        (hidden_impact - 1.0).abs() < 0.001,
        "Sole input hidden neuron should have impact = 1.0 (100% influence), got {hidden_impact}"
    );
}

/// Issue #130: Multiple inputs should dilute impact
///
/// Scenario:
/// - Hidden (weight 3.0) -> Output
/// - 9 other inputs (weight 1.0 each) -> Output
///
/// Total inbound weight = 3.0 + 9.0 = 12.0
/// Hidden's normalised impact = 3.0 / 12.0 × 1.0 = 0.25
#[test]
fn test_multiple_inputs_dilute_impact() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
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
                // Hidden -> Output with weight 3.0
                SynapseJson {
                    from_uuid: "hidden-1".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 3.0,
                    synapse_type: None,
                },
            ];
            // Add 9 other inputs (direct from input neurons)
            for i in 0..9 {
                synapses.push(SynapseJson {
                    from_uuid: format!("input-{i}"),
                    to_uuid: "output-0".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                });
            }
            synapses
        },
        input: 9,
        output: 1,
    };

    let impacts = compute_impacts_public(&creature);

    // Hidden neuron's normalised impact = 3.0 / (3.0 + 9.0) × 1.0 = 0.25
    let hidden_impact = *impacts.get("hidden-1").expect("hidden should have impact");
    let expected = 3.0 / 12.0;
    assert!(
        (hidden_impact - expected).abs() < 0.001,
        "Hidden neuron should have impact = {expected:.3} (3/12), got {hidden_impact}"
    );

    // Crucially: impact < 1.0 (not clamped to 1.0!)
    assert!(
        hidden_impact < 1.0,
        "Hidden neuron with competing inputs should have impact < 1.0"
    );
}

/// Issue #130: Multiple outputs should sum impacts correctly
///
/// Scenario:
/// - Hidden -> Output1 (weight 0.5, sole input)
/// - Hidden -> Output2 (weight 0.5, sole input)
/// - Hidden -> Output3 (weight 0.5, sole input)
///
/// Each contribution = 0.5 / 0.5 × 1.0 = 1.0 (sole input to each output)
/// Total impact = 1.0 + 1.0 + 1.0 = 3.0
///
/// Note: This is correct! The hidden neuron affects THREE outputs, so its
/// total impact on the creature is 3x an output neuron.
#[test]
fn test_multiple_outputs_sum_impacts() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-1".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-2".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5, // Sole input to output-0
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 0.5, // Sole input to output-1
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-2".to_string(),
                weight: 0.5, // Sole input to output-2
                synapse_type: None,
            },
        ],
        input: 1,
        output: 3,
    };

    let impacts = compute_impacts_public(&creature);

    // Hidden neuron's impact = sum of contributions to each output
    // Each contribution = (0.5 / 0.5) × 1.0 = 1.0 (sole input)
    // Total = 3.0
    let hidden_impact = *impacts.get("hidden-1").expect("hidden should have impact");
    assert!(
        (hidden_impact - 3.0).abs() < 0.001,
        "Hidden neuron feeding 3 outputs should have impact = 3.0, got {hidden_impact}"
    );
}

/// Issue #130: Competing inputs with large total weight
///
/// This is the exact scenario from the issue:
/// - Hidden (weight 3.0) -> Output
/// - Other inputs (total weight 97.0) -> Output
///
/// Old (absolute): impact = 3.0 (clamped to 1.0)
/// New (normalised): impact = 3.0 / 100.0 = 0.03
#[test]
fn test_large_competing_weight_gives_small_impact() {
    let creature = CreatureJson {
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
                // Candidate -> output with weight 3.0
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
            // Add 97 units of weight from other inputs
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
        input: 10,
        output: 1,
    };

    let impacts = compute_impacts_public(&creature);
    let candidate_impact = *impacts
        .get("candidate")
        .expect("candidate should have impact");

    // Normalised impact = 3.0 / (3.0 + 97.0) × 1.0 = 0.03
    let expected = 3.0 / 100.0;
    assert!(
        (candidate_impact - expected).abs() < 0.01,
        "Candidate with 3% of total weight should have impact ≈ 0.03, got {candidate_impact}"
    );

    // Crucially: impact << 1.0 (not clamped to 1.0!)
    assert!(
        candidate_impact < 0.1,
        "Candidate should have small impact, got {candidate_impact}"
    );
}

/// Issue #130: Deep network chain should have impact < 1.0 per layer
///
/// Scenario: hidden-3 -> hidden-2 -> hidden-1 -> output
/// All weights = 1.0 (sole connections)
///
/// With normalisation:
/// - hidden-1: 1.0 / 1.0 × 1.0 = 1.0 (sole input to output)
/// - hidden-2: 1.0 / 1.0 × 1.0 = 1.0 (sole input to hidden-1)
/// - hidden-3: 1.0 / 1.0 × 1.0 = 1.0 (sole input to hidden-2)
///
/// In a chain with sole connections, all neurons have full influence.
#[test]
fn test_deep_chain_sole_connections() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-3".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
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
                from_uuid: "hidden-3".to_string(),
                to_uuid: "hidden-2".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    };

    let impacts = compute_impacts_public(&creature);

    // All neurons in sole-connection chain have impact = 1.0
    // (each is the ONLY input to its downstream neuron)
    for uuid in ["hidden-1", "hidden-2", "hidden-3"] {
        let impact = *impacts.get(uuid).expect("neuron should have impact");
        assert!(
            (impact - 1.0).abs() < 0.001,
            "{uuid} in sole-connection chain should have impact = 1.0, got {impact}"
        );
    }
}

/// Issue #130: Deep network with competing inputs at each layer
///
/// Scenario: hidden-3 -> hidden-2 -> hidden-1 -> output
/// Each layer has 10 inputs (1 from previous layer, 9 from inputs)
/// All weights = 1.0
///
/// With normalisation:
/// - hidden-1: 1.0 / 10.0 × 1.0 = 0.1
/// - hidden-2: 1.0 / 10.0 × 0.1 = 0.01
/// - hidden-3: 1.0 / 10.0 × 0.01 = 0.001
///
/// Impact dilutes exponentially with depth!
#[test]
fn test_deep_chain_competing_inputs_dilutes_impact() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-3".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
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
            let mut synapses = Vec::new();

            // Chain connections
            synapses.push(SynapseJson {
                from_uuid: "hidden-3".to_string(),
                to_uuid: "hidden-2".to_string(),
                weight: 1.0,
                synapse_type: None,
            });
            synapses.push(SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: 1.0,
                synapse_type: None,
            });
            synapses.push(SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            });

            // 9 direct inputs to each layer
            for i in 0..9 {
                synapses.push(SynapseJson {
                    from_uuid: format!("input-{i}"),
                    to_uuid: "output-0".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                });
                synapses.push(SynapseJson {
                    from_uuid: format!("input-{i}"),
                    to_uuid: "hidden-1".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                });
                synapses.push(SynapseJson {
                    from_uuid: format!("input-{i}"),
                    to_uuid: "hidden-2".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                });
            }

            synapses
        },
        input: 9,
        output: 1,
    };

    let impacts = compute_impacts_public(&creature);

    // hidden-1: 1.0 / 10.0 × 1.0 = 0.1
    let h1 = *impacts
        .get("hidden-1")
        .expect("hidden-1 should have impact");
    assert!(
        (h1 - 0.1).abs() < 0.01,
        "hidden-1 should have impact ≈ 0.1, got {h1}"
    );

    // hidden-2: 1.0 / 10.0 × 0.1 = 0.01
    let h2 = *impacts
        .get("hidden-2")
        .expect("hidden-2 should have impact");
    assert!(
        (h2 - 0.01).abs() < 0.01,
        "hidden-2 should have impact ≈ 0.01, got {h2}"
    );

    // hidden-3: 1.0 / 10.0 × 0.01 = 0.001
    let h3 = *impacts
        .get("hidden-3")
        .expect("hidden-3 should have impact");
    assert!(
        (h3 - 0.001).abs() < 0.01,
        "hidden-3 should have impact ≈ 0.001, got {h3}"
    );

    // Impact should decrease with distance from output
    assert!(h1 > h2, "h1 should have more impact than h2: {h1} vs {h2}");
    assert!(h2 > h3, "h2 should have more impact than h3: {h2} vs {h3}");
}

/// Issue #130: SINE squash (from the issue) should have normalised impact
#[test]
fn test_sine_neuron_normalised_impact() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-sine".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "SINE".to_string(),
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
                // SINE hidden -> output (weight 2.0)
                SynapseJson {
                    from_uuid: "hidden-sine".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 2.0,
                    synapse_type: None,
                },
            ];
            // Add other inputs (total weight 8.0)
            for i in 0..8 {
                synapses.push(SynapseJson {
                    from_uuid: format!("input-{i}"),
                    to_uuid: "output-0".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                });
            }
            synapses
        },
        input: 8,
        output: 1,
    };

    let impacts = compute_impacts_public(&creature);

    // Normalised impact = 2.0 / (2.0 + 8.0) × 1.0 = 0.2
    let impact = *impacts
        .get("hidden-sine")
        .expect("hidden should have impact");
    assert!(
        (impact - 0.2).abs() < 0.01,
        "SINE hidden neuron should have normalised impact = 0.2, got {impact}"
    );

    // Impact < 1.0 (the key fix for Issue #130)
    assert!(
        impact < 1.0,
        "SINE hidden neuron must have impact < 1.0, got {impact}"
    );
}
