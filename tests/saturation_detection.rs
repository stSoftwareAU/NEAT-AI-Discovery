//! Tests for saturation detection in impact calculation
//!
//! ## Production Issue (GRQ fittest creature)
//!
//! The fittest creature has:
//! - Weights up to ±100,000
//! - Intermediate neurons with 335 inputs, total weight -10,447
//! - These neurons are COMPLETELY SATURATED (e.g., LOGISTIC at input 613 → output ≈1.0)
//!
//! ## The Problem
//!
//! When a neuron is saturated:
//! - Its output is at the min/max of its squash function
//! - The derivative is effectively 0
//! - Changes to upstream neurons have ZERO effect on downstream
//!
//! But our impact calculation doesn't account for this! It gives structural
//! impact based on path weights, ignoring that the path is "blocked" by saturation.
//!
//! ## Example from Production
//!
//! Target neuron → intermediate (LOGISTIC, input sum 613, saturated at 1.0) → output
//!
//! - Structural impact: Some positive value based on path weights
//! - Actual impact: ~0 (changes to target don't affect output because intermediate is saturated)
//!
//! ## Squash Saturation Thresholds
//!
//! | Squash | Range | Saturation Input |
//! |--------|-------|------------------|
//! | TANH | [-1, 1] | |input| > 3 → ~saturated |
//! | LOGISTIC | [0, 1] | input > 5 or input < -5 → saturated |
//! | HARD_TANH | [-1, 1] | |input| > 1 → clamped |
//! | ArcTan | [-π/2, π/2] | |input| > 10 → ~saturated |

use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn hidden(uuid: &str, squash: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "hidden".to_string(),
        squash: squash.to_string(),
        bias,
    }
}

fn output(uuid: &str, squash: &str) -> NeuronJson {
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

/// Test: Production scenario - target through saturated intermediate
///
/// This recreates the GRQ production issue:
/// - Target neuron connects to intermediate with LOGISTIC squash
/// - Intermediate has HUGE total input (saturated at output ≈ 1.0)
/// - Intermediate connects to output with small weight
///
/// Expected: Changes to target have ~0 effect (but structural impact ≠ 0)
#[test]
fn test_production_scenario_saturated_intermediate() {
    // Simplified version of GRQ topology:
    // target → intermediate (LOGISTIC, saturated) → output
    let creature = CreatureJson {
        input: 100,
        output: 1,
        neurons: vec![
            hidden("target", "Swish", 1.15),         // Like d28eb8a4
            hidden("intermediate", "LOGISTIC", 0.0), // Like insider-volume-check
            output("output-0", "HARD_TANH"),
        ],
        synapses: {
            let mut synapses = vec![
                // Target connects to intermediate (relatively small weight)
                synapse("target", "intermediate", -3.45),
                // Intermediate connects to output
                synapse("intermediate", "output-0", -0.024),
            ];

            // Add HUGE input sum to intermediate (like production: 613)
            // This will saturate the LOGISTIC to output ≈ 1.0
            for i in 0..60 {
                synapses.push(synapse(&format!("input-{i}"), "intermediate", 10.0));
            }
            // Total: 60 × 10 = 600 + target's -3.45 ≈ 596 → LOGISTIC saturated!

            // Add some inputs to output for normalisation
            for i in 60..100 {
                synapses.push(synapse(&format!("input-{i}"), "output-0", 1.0));
            }

            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let target_impact = *impacts.get("target").unwrap_or(&0.0);
    let intermediate_impact = *impacts.get("intermediate").unwrap_or(&0.0);

    println!("=== Production Saturation Scenario ===");
    println!("Target impact (structural): {target_impact}");
    println!("Intermediate impact (structural): {intermediate_impact}");

    // Current behaviour: structural impact is calculated
    // But ACTUAL effective impact should be ~0 due to saturation!

    // Document current behaviour
    assert!(
        target_impact > 0.0,
        "Structural impact should be positive, got {target_impact}"
    );

    // TODO: With saturation detection, this should be ~0
    // The intermediate LOGISTIC is saturated (input ~600, output ~1.0)
    // Derivative at saturation is ~0, so upstream changes don't propagate

    println!("NOTE: Structural impact doesn't account for saturation!");
    println!("ACTUAL effective impact should be ~0 because intermediate is saturated.");
}

/// Test: LOGISTIC saturation at various input levels
#[test]
fn test_logistic_saturation_levels() {
    // LOGISTIC(x) = 1 / (1 + e^-x)
    // At x = 5: output ≈ 0.9933, derivative ≈ 0.0066
    // At x = 10: output ≈ 0.99995, derivative ≈ 0.00005
    // At x = 600: output = 1.0, derivative ≈ 0

    let input_levels = [0.0, 5.0, 10.0, 100.0, 600.0];

    for input_sum in input_levels {
        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![
                hidden("upstream", "IDENTITY", 0.0),
                hidden("logistic", "LOGISTIC", 0.0),
                output("output-0", "IDENTITY"),
            ],
            synapses: vec![
                synapse("input-0", "upstream", 1.0),
                synapse("upstream", "logistic", 1.0),
                // Add bias-like weight to simulate large input
                synapse("input-0", "logistic", input_sum - 1.0),
                synapse("logistic", "output-0", 1.0),
            ],
        };

        let impacts = compute_impacts_public(&creature);
        let upstream_impact = *impacts.get("upstream").unwrap_or(&0.0);

        // Calculate actual LOGISTIC derivative at this input
        let logistic_output = 1.0 / (1.0 + (-input_sum as f64).exp());
        let logistic_derivative = logistic_output * (1.0 - logistic_output);

        println!(
            "Input sum: {input_sum:6.1}, LOGISTIC output: {logistic_output:.6}, derivative: {logistic_derivative:.6}, structural impact: {upstream_impact:.4}"
        );

        // At high input, derivative → 0, so effective impact should → 0
        // But structural impact doesn't change!
    }

    println!("\nNOTE: Structural impact is constant, but derivative → 0 at saturation!");
}

/// Test: TANH saturation
#[test]
fn test_tanh_saturation_levels() {
    // TANH(x): at |x| > 3, output ≈ ±1, derivative ≈ 0

    let input_levels: [f64; 6] = [0.0, 1.0, 3.0, 5.0, 10.0, 100.0];

    for input_sum in input_levels {
        // Calculate actual TANH derivative at this input
        let tanh_output = input_sum.tanh();
        let tanh_derivative = 1.0 - tanh_output * tanh_output;

        println!(
            "Input: {input_sum:6.1}, TANH output: {tanh_output:+.6}, derivative: {tanh_derivative:.6}"
        );
    }

    println!("\nAt |input| > 3, derivative → 0 (saturated)");
}

/// Test: Chain of saturated neurons blocks signal propagation
#[test]
fn test_chain_saturation_blocks_propagation() {
    // Network: upstream → tanh1 (saturated) → tanh2 (saturated) → output
    // Even if each path weight is 1.0, saturation blocks propagation

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY", 0.0),
            hidden("tanh1", "TANH", 0.0),
            hidden("tanh2", "TANH", 0.0),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "upstream", 1.0),
            synapse("upstream", "tanh1", 100.0), // HUGE → saturates tanh1
            synapse("tanh1", "tanh2", 100.0),    // HUGE → saturates tanh2
            synapse("tanh2", "output-0", 1.0),
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let upstream_impact = *impacts.get("upstream").unwrap_or(&0.0);

    println!("=== Chain of Saturated TANHs ===");
    println!("Structural impact of upstream: {upstream_impact}");

    // Actual signal propagation:
    // upstream = x
    // tanh1 = TANH(100 × x) → always ±1 for any non-zero x (saturated)
    // tanh2 = TANH(100 × ±1) = TANH(±100) → always ±1 (saturated)
    // output = ±1 (constant!)
    //
    // Changes to upstream DON'T CHANGE OUTPUT (it's always ±1)
    // But structural impact says upstream has impact!

    println!("ACTUAL impact should be ~0 (output is constant due to saturation)");
}

/// Test: Multiple large weights to single neuron cause saturation
#[test]
fn test_many_inputs_cause_saturation() {
    // Production pattern: intermediate neuron has 335 inputs with total weight -10,447
    // Even with ArcTan (which has wider range than TANH), this is saturated

    let creature = CreatureJson {
        input: 100,
        output: 1,
        neurons: vec![
            hidden("focus", "IDENTITY", 0.0),
            hidden("hub", "TANH", 0.0), // Hub with many inputs
            output("output-0", "IDENTITY"),
        ],
        synapses: {
            let mut synapses = vec![
                synapse("focus", "hub", 0.5),
                synapse("hub", "output-0", 1.0),
            ];

            // Add many other inputs to hub (simulating production pattern)
            for i in 0..99 {
                synapses.push(synapse(&format!("input-{i}"), "hub", 10.0));
            }
            // Total input to hub: 99 × 10 + focus × 0.5 ≈ 990 → MASSIVE saturation!

            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let focus_impact = *impacts.get("focus").unwrap_or(&0.0);
    let hub_impact = *impacts.get("hub").unwrap_or(&0.0);

    println!("=== Many Inputs Cause Saturation ===");
    println!("Focus impact: {focus_impact}");
    println!("Hub impact: {hub_impact}");

    // With normalised formula: focus impact = 0.5 / (990 + 0.5) × hub_impact
    // ≈ 0.0005 × hub_impact

    // But even this doesn't account for saturation!
    // The hub TANH receives ~990 total input → completely saturated
    // Changes to focus have ~0 actual effect on output
}
