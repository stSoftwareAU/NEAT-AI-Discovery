//! Tests for impact calculation with bounded squash functions
//!
//! ## The Problem
//!
//! Squash functions like TANH, LOGISTIC, HARD_TANH have bounded outputs:
//! - TANH: [-1, 1]
//! - LOGISTIC: [0, 1]
//! - HARD_TANH: [-1, 1]
//!
//! This affects impact calculation in two ways:
//!
//! 1. **Output range limitation**: A TANH neuron with weight 100 to output can
//!    only contribute between -100 and +100 to the output sum, regardless of
//!    how many inputs it receives or how large they are.
//!
//! 2. **Saturation (derivative = 0)**: When a TANH neuron is in saturation
//!    (activation near ±1), small changes to its input cause almost NO change
//!    to its output. The impact of upstream neurons is effectively zero.
//!
//! ## Current Implementation
//!
//! The current impact calculation treats Linear squashes differently from
//! Threshold and Selection squashes, but doesn't account for saturation
//! in bounded Linear squashes like TANH.
//!
//! ## Test Cases
//!
//! These tests verify the behaviour and help identify the fundamental issue.

use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: Create a hidden neuron
fn hidden(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "hidden".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Helper: Create an output neuron
fn output(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "output".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Helper: Create a synapse
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Test: TANH neuron with huge weight to output
///
/// Scenario:
/// - TANH hidden neuron connects to IDENTITY output with weight 100
/// - Output has other inputs totalling 1000
///
/// Question: What should the TANH neuron's impact be?
///
/// The TANH neuron's output is bounded to [-1, 1], so its contribution
/// to the output is bounded to [-100, 100]. This is at most 100/1100 ≈ 9%
/// of the total input weight range.
#[test]
fn test_tanh_neuron_huge_weight_to_output() {
    let creature = CreatureJson {
        input: 10,
        output: 1,
        neurons: vec![
            hidden("tanh-hidden", "TANH"),
            output("output-0", "IDENTITY"),
        ],
        synapses: {
            let mut synapses = vec![
                synapse("input-0", "tanh-hidden", 1.0),
                synapse("tanh-hidden", "output-0", 100.0), // Huge weight
            ];
            // Add other inputs to output (total weight 1000)
            for i in 1..10 {
                synapses.push(synapse(&format!("input-{i}"), "output-0", 1000.0 / 9.0));
            }
            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let tanh_impact = *impacts.get("tanh-hidden").unwrap_or(&0.0);

    // With normalised formula: 100 / (100 + 1000) × 1.0 ≈ 0.091
    // This accounts for the weight proportion but NOT the bounded output
    println!("TANH hidden neuron impact with weight 100 (total inbound 1100): {tanh_impact}");

    // Document current behaviour (this test is exploratory)
    assert!(
        tanh_impact > 0.0,
        "TANH neuron should have positive impact, got {tanh_impact}"
    );

    // The normalised formula gives ~0.091
    // Is this correct? The TANH output is bounded to [-1, 1], so the
    // actual contribution range is [-100, 100] regardless of upstream.
}

/// Test: Compare TANH vs IDENTITY with same weight
///
/// If both neurons have the same weight to output, does the impact differ?
/// Structurally they're the same, but TANH has bounded output.
#[test]
fn test_tanh_vs_identity_same_weight() {
    // Create two identical networks, one with TANH hidden, one with IDENTITY
    let tanh_creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![hidden("hidden", "TANH"), output("output-0", "IDENTITY")],
        synapses: vec![
            synapse("input-0", "hidden", 1.0),
            synapse("hidden", "output-0", 1.0), // Same weight
        ],
    };

    let identity_creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![hidden("hidden", "IDENTITY"), output("output-0", "IDENTITY")],
        synapses: vec![
            synapse("input-0", "hidden", 1.0),
            synapse("hidden", "output-0", 1.0), // Same weight
        ],
    };

    let tanh_impacts = compute_impacts_public(&tanh_creature);
    let identity_impacts = compute_impacts_public(&identity_creature);

    let tanh_impact = *tanh_impacts.get("hidden").unwrap_or(&0.0);
    let identity_impact = *identity_impacts.get("hidden").unwrap_or(&0.0);

    println!("TANH hidden impact: {tanh_impact}");
    println!("IDENTITY hidden impact: {identity_impact}");

    // Currently, both have the same structural impact
    // But should they? TANH output is bounded, IDENTITY is not.
    assert!(
        (tanh_impact - identity_impact).abs() < 0.001,
        "Currently TANH and IDENTITY have same impact: {tanh_impact} vs {identity_impact}"
    );
}

/// Test: TANH neuron with HUGE incoming weight (saturation scenario)
///
/// If a TANH neuron receives huge input (weight 1000 from input), it will
/// be in saturation (output ≈ ±1). In this state:
/// - Changes to input have almost no effect on output (derivative ≈ 0)
/// - Upstream neuron's "impact" on the output is effectively zero
///
/// But our structural impact calculation doesn't account for this!
#[test]
fn test_tanh_saturation_upstream_impact() {
    // Network: upstream -> TANH (huge weight) -> output
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY"),
            hidden("saturated-tanh", "TANH"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "upstream", 1.0),
            synapse("upstream", "saturated-tanh", 1000.0), // HUGE weight -> saturation
            synapse("saturated-tanh", "output-0", 1.0),
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let upstream_impact = *impacts.get("upstream").unwrap_or(&0.0);
    let tanh_impact = *impacts.get("saturated-tanh").unwrap_or(&0.0);

    println!("Upstream (before saturated TANH) impact: {upstream_impact}");
    println!("Saturated TANH impact: {tanh_impact}");

    // Current formula: upstream -> tanh (1000/1000) × tanh -> output (1/1) = 1.0
    // But in reality, the TANH is saturated, so upstream has ~0 effective impact!

    // This is a documentation test - what does the current implementation give?
    assert!(
        upstream_impact > 0.0,
        "Upstream impact should be calculated, got {upstream_impact}"
    );

    // NOTE: The structural impact doesn't reflect that the TANH is saturated.
    // This is a known limitation - we'd need activation data to detect saturation.
}

/// Test: LOGISTIC neuron (bounded to [0, 1])
#[test]
fn test_logistic_bounded_output() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("logistic-hidden", "LOGISTIC"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "logistic-hidden", 1.0),
            synapse("logistic-hidden", "output-0", 50.0), // Large weight
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let logistic_impact = *impacts.get("logistic-hidden").unwrap_or(&0.0);

    println!("LOGISTIC hidden impact with weight 50: {logistic_impact}");

    // LOGISTIC output is bounded to [0, 1]
    // With weight 50, max contribution to output is 50
    // But the normalised formula doesn't account for this ceiling
}

/// Test: HARD_TANH neuron (hard clamping to [-1, 1])
#[test]
fn test_hard_tanh_bounded_output() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("hard-tanh", "HARD_TANH"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "hard-tanh", 1.0),
            synapse("hard-tanh", "output-0", 100.0), // Large weight
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let hard_tanh_impact = *impacts.get("hard-tanh").unwrap_or(&0.0);

    println!("HARD_TANH hidden impact with weight 100: {hard_tanh_impact}");

    // HARD_TANH clamps output to [-1, 1]
    // Max contribution to output is 100, not infinity
}

/// Test: ReLU neuron (bounded below at 0, unbounded above)
///
/// ReLU is different - it's bounded below (output >= 0) but unbounded above.
/// This means positive activations can grow without limit.
#[test]
fn test_relu_half_bounded() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("relu-hidden", "ReLU"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "relu-hidden", 1.0),
            synapse("relu-hidden", "output-0", 10.0),
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let relu_impact = *impacts.get("relu-hidden").unwrap_or(&0.0);

    println!("ReLU hidden impact with weight 10: {relu_impact}");

    // ReLU is unbounded above, so the weight matters more than for TANH
}

/// Test: Chain of TANH neurons - does impact compound incorrectly?
///
/// If we have: input -> TANH1 -> TANH2 -> TANH3 -> output
/// Each TANH bounds its output to [-1, 1]
/// But the weight products might suggest larger impact.
#[test]
fn test_tanh_chain_impact() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("tanh-1", "TANH"),
            hidden("tanh-2", "TANH"),
            hidden("tanh-3", "TANH"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "tanh-1", 10.0), // Large weights throughout
            synapse("tanh-1", "tanh-2", 10.0),
            synapse("tanh-2", "tanh-3", 10.0),
            synapse("tanh-3", "output-0", 10.0),
        ],
    };

    let impacts = compute_impacts_public(&creature);

    let t1 = *impacts.get("tanh-1").unwrap_or(&0.0);
    let t2 = *impacts.get("tanh-2").unwrap_or(&0.0);
    let t3 = *impacts.get("tanh-3").unwrap_or(&0.0);

    println!("TANH chain impacts:");
    println!("  tanh-1: {t1}");
    println!("  tanh-2: {t2}");
    println!("  tanh-3: {t3}");

    // In a chain of sole connections, all should have impact 1.0 (normalised)
    // But the actual effect is bounded by each TANH's output range
    //
    // With weight 10 at each layer:
    // - tanh-1 output: [-1, 1]
    // - tanh-2 input: [-10, 10], output: [-1, 1] (saturated!)
    // - tanh-3 input: [-10, 10], output: [-1, 1] (saturated!)
    // - output input: [-10, 10]
    //
    // The actual range is bounded, but structural impact = 1.0 for all
}

/// Test: TANH with tiny weight vs huge weight - same normalised impact?
///
/// If TANH output is bounded to [-1, 1]:
/// - Weight 0.01: contribution to output is [-0.01, 0.01]
/// - Weight 100: contribution to output is [-100, 100]
///
/// The weight DOES matter for the contribution range, even though
/// the TANH output itself is bounded.
#[test]
fn test_tanh_tiny_vs_huge_weight() {
    // Tiny weight
    let tiny_creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![hidden("tanh", "TANH"), output("output-0", "IDENTITY")],
        synapses: vec![
            synapse("input-0", "tanh", 1.0),
            synapse("tanh", "output-0", 0.01),   // Tiny weight
            synapse("input-0", "output-0", 1.0), // Direct connection
        ],
    };

    // Huge weight
    let huge_creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![hidden("tanh", "TANH"), output("output-0", "IDENTITY")],
        synapses: vec![
            synapse("input-0", "tanh", 1.0),
            synapse("tanh", "output-0", 100.0),  // Huge weight
            synapse("input-0", "output-0", 1.0), // Same direct connection
        ],
    };

    let tiny_impacts = compute_impacts_public(&tiny_creature);
    let huge_impacts = compute_impacts_public(&huge_creature);

    let tiny_impact = *tiny_impacts.get("tanh").unwrap_or(&0.0);
    let huge_impact = *huge_impacts.get("tanh").unwrap_or(&0.0);

    println!("TANH with weight 0.01 (total inbound 1.01): impact = {tiny_impact}");
    println!("TANH with weight 100 (total inbound 101): impact = {huge_impact}");

    // Normalised formula:
    // tiny: 0.01 / 1.01 ≈ 0.0099
    // huge: 100 / 101 ≈ 0.99
    //
    // The huge weight DOES give higher impact, which makes sense:
    // the TANH's bounded output is multiplied by a larger weight,
    // contributing more to the output's input sum.

    assert!(
        huge_impact > tiny_impact,
        "Huge weight should give higher impact than tiny weight"
    );
}

/// Test: What happens when TANH feeds into another TANH?
///
/// TANH -> TANH chain: both have bounded outputs
/// The weight between them matters for whether the second TANH saturates.
#[test]
fn test_tanh_to_tanh_chain() {
    // Small weight: second TANH won't saturate
    let small_weight = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("tanh-1", "TANH"),
            hidden("tanh-2", "TANH"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "tanh-1", 1.0),
            synapse("tanh-1", "tanh-2", 0.5), // Small weight - won't saturate
            synapse("tanh-2", "output-0", 1.0),
        ],
    };

    // Large weight: second TANH will saturate
    let large_weight = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("tanh-1", "TANH"),
            hidden("tanh-2", "TANH"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "tanh-1", 1.0),
            synapse("tanh-1", "tanh-2", 100.0), // Large weight - will saturate
            synapse("tanh-2", "output-0", 1.0),
        ],
    };

    let small_impacts = compute_impacts_public(&small_weight);
    let large_impacts = compute_impacts_public(&large_weight);

    let small_t1 = *small_impacts.get("tanh-1").unwrap_or(&0.0);
    let large_t1 = *large_impacts.get("tanh-1").unwrap_or(&0.0);

    println!("TANH-1 impact (small weight to TANH-2): {small_t1}");
    println!("TANH-1 impact (large weight to TANH-2): {large_t1}");

    // Currently, structural impact is the same (sole connections = 1.0)
    // But in reality:
    // - Large weight: TANH-2 is saturated, so TANH-1 changes have ~0 effect
    // - Small weight: TANH-2 is in linear region, TANH-1 changes propagate
}
