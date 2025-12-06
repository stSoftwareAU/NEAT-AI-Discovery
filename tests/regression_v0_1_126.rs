//! REGRESSION TESTS for v0.1.126 fixes
//!
//! These tests catch regressions if the v0.1.126 fixes are accidentally reverted.
//! DO NOT DELETE OR COMMENT OUT THESE TESTS - they protect against known bugs.
//!
//! ## Fixes covered:
//!
//! 1. **Impact calculation uses absolute weights**: The impact calculation was
//!    incorrectly normalising weights by `total_inbound`, causing massive
//!    underestimation. Production data showed ~75% of "low-impact" neuron
//!    removals actually INCREASED error.
//!
//!    Example: A neuron with 0.001 weight to a target with total_inbound=100
//!    was calculated as 0.001/100 = 1e-5 impact instead of 0.001 actual impact.
//!
//! If any of these tests fail after a code change, the fix has regressed.

mod common;

use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// REGRESSION TEST: Impact must use ABSOLUTE weight paths, not normalised.
///
/// BUG (fixed in v0.1.126): Impact was computed as `weight / total_inbound × child_impact`
/// instead of `weight × child_impact`. This caused impact to be underestimated by
/// orders of magnitude when the downstream neuron had many/large other inputs.
///
/// Production evidence: Neurons with calculated impact 1e-10 to 1e-17 caused
/// score deltas of 1e-5 to 1e-2 when removed - off by 5-15 orders of magnitude!
/// Result: ~75% of "low-impact" removal candidates actually hurt the score.
///
/// The fix: Use absolute weight products for impact, not normalised fractions.
#[test]
fn regression_impact_must_use_absolute_weights_not_normalised() {
    // Network structure:
    //   input-0 → candidate (weight 1.0)
    //   candidate → target (weight 0.001)  ← small weight
    //   input-1 → target (weight 100.0)    ← large weight dominates total_inbound
    //   target → output-0 (weight 1.0)
    //
    // The candidate neuron's contribution to output is:
    //   activation × 0.001 × 1.0 = 0.001 × activation
    //
    // With BUG (normalised): impact = 0.001 / (100.0 + 0.001) × 1.0 ≈ 1e-5
    // With FIX (absolute):   impact = 0.001 × 1.0 = 0.001
    //
    // The normalised formula underestimates by 100x!
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "candidate".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "target".to_string(),
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
            // candidate → target with small weight
            SynapseJson {
                from_uuid: "candidate".to_string(),
                to_uuid: "target".to_string(),
                weight: 0.001,
            },
            // Large competing weight - this caused the normalisation bug
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "target".to_string(),
                weight: 100.0,
            },
            // target → output with weight 1.0
            SynapseJson {
                from_uuid: "target".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
            },
        ],
    };

    let impacts = compute_impacts_public(&creature);

    let candidate_impact = impacts.get("candidate").copied().unwrap_or(-1.0);
    let target_impact = impacts.get("target").copied().unwrap_or(-1.0);
    let output_impact = impacts.get("output-0").copied().unwrap_or(-1.0);

    // Output should have impact = 1.0
    assert!(
        (output_impact - 1.0).abs() < 0.01,
        "Output neuron should have impact ≈ 1.0, got {output_impact:.6}"
    );

    // Target has direct connection to output with weight 1.0, so impact = 1.0
    assert!(
        (target_impact - 1.0).abs() < 0.01,
        "Target neuron should have impact ≈ 1.0, got {target_impact:.6}"
    );

    // CRITICAL: Candidate's impact should be based on ABSOLUTE weight path
    // candidate → target (0.001) × target → output (1.0) = 0.001
    //
    // With the bug (normalised): 0.001 / 100.001 × 1.0 ≈ 1e-5
    // With the fix (absolute):   0.001 × 1.0 = 0.001
    //
    // We check that impact is at least 0.0001 (allowing some margin)
    // If impact is ~1e-5, the normalisation bug is present
    assert!(
        candidate_impact >= 0.0001,
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  REGRESSION DETECTED: Impact using normalised weights instead of absolute!    ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║  Candidate neuron has impact = {candidate_impact:.2e}                         \n\
        ║                                                                              ║\n\
        ║  Expected: ~0.001 (absolute weight path: 0.001 × 1.0)                         ║\n\
        ║  Got:      ~1e-5 (normalised: 0.001 / 100.001 × 1.0)                          ║\n\
        ║                                                                              ║\n\
        ║  The normalised formula computes 'fraction of downstream input' instead of   ║\n\
        ║  'absolute contribution to output'. This caused ~75% of low-impact removal   ║\n\
        ║  candidates to actually INCREASE error when removed.                          ║\n\
        ║                                                                              ║\n\
        ║  FIX: In compute_impact_recursive(), use:                                     ║\n\
        ║       contribution = weight.abs() * child_impact                              ║\n\
        ║  NOT:                                                                         ║\n\
        ║       contribution = (weight.abs() / total_inbound) * child_impact            ║\n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n\n"
    );

    // More precise check: impact should be approximately 0.001
    assert!(
        (candidate_impact - 0.001).abs() < 0.0005,
        "Candidate impact should be ≈ 0.001 (absolute path weight), got {candidate_impact:.6}"
    );
}

/// REGRESSION TEST: Multiple large competing inputs must not suppress impact.
///
/// This test creates a scenario closer to production: a neuron feeding into a
/// target that has MANY large competing inputs. The normalisation bug becomes
/// more severe with more/larger competing weights.
#[test]
fn regression_many_competing_inputs_must_not_suppress_impact() {
    // Network: candidate → target ← (10 competing inputs with weight 10.0 each)
    //          target → output-0
    //
    // Total inbound to target = 0.5 + (10 × 10.0) = 100.5
    //
    // With BUG: impact = 0.5 / 100.5 × 1.0 ≈ 0.005
    // With FIX: impact = 0.5 × 1.0 = 0.5
    //
    // The bug causes 100x underestimation!
    let mut synapses = vec![
        SynapseJson {
            from_uuid: "candidate".to_string(),
            to_uuid: "target".to_string(),
            weight: 0.5,
        },
        SynapseJson {
            from_uuid: "target".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 1.0,
        },
    ];

    // Add 10 competing inputs to target
    for i in 0..10 {
        synapses.push(SynapseJson {
            from_uuid: format!("input-{i}"),
            to_uuid: "target".to_string(),
            weight: 10.0,
        });
    }

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
                uuid: "target".to_string(),
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
        synapses,
    };

    let impacts = compute_impacts_public(&creature);
    let candidate_impact = impacts.get("candidate").copied().unwrap_or(-1.0);

    // With absolute weights: impact = 0.5 × 1.0 = 0.5
    // With normalised (bug): impact = 0.5 / 100.5 × 1.0 ≈ 0.005
    assert!(
        candidate_impact >= 0.1,
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  REGRESSION DETECTED: Many competing inputs suppressing impact!               ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║  Candidate neuron has impact = {candidate_impact:.4}                          \n\
        ║                                                                              ║\n\
        ║  Expected: ~0.5 (absolute weight path)                                        ║\n\
        ║  Got:      ~0.005 if normalised (100x underestimate)                          ║\n\
        ║                                                                              ║\n\
        ║  With 10 competing inputs of weight 10.0 each, total_inbound = 100.5          ║\n\
        ║  Normalising by total_inbound severely underestimates the actual impact.      ║\n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n\n"
    );

    // More precise: should be approximately 0.5
    assert!(
        (candidate_impact - 0.5).abs() < 0.1,
        "Candidate impact should be ≈ 0.5 (absolute path weight), got {candidate_impact:.4}"
    );
}

/// REGRESSION TEST: Deep networks must accumulate impact correctly.
///
/// For a chain: A → B → C → output
/// Impact of A should be: weight_AB × weight_BC × weight_C_output
/// NOT: (weight_AB / inbound_B) × (weight_BC / inbound_C) × 1.0
#[test]
fn regression_deep_network_impact_accumulation() {
    // Chain: candidate → hidden-1 → hidden-2 → output-0
    // All weights = 0.5
    //
    // With absolute: impact = 0.5 × 0.5 × 0.5 = 0.125
    // With normalised (if each has extra inputs): much smaller
    let creature = CreatureJson {
        input: 3,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "candidate".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-2".to_string(),
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
            // Main chain
            SynapseJson {
                from_uuid: "candidate".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: 0.5,
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "hidden-2".to_string(),
                weight: 0.5,
            },
            SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
            },
            // Competing inputs at each layer (to trigger normalisation bug)
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: 10.0,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-2".to_string(),
                weight: 10.0,
            },
            SynapseJson {
                from_uuid: "input-2".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 10.0,
            },
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let candidate_impact = impacts.get("candidate").copied().unwrap_or(-1.0);

    // With absolute weights: 0.5 × 0.5 × 0.5 = 0.125
    // With normalised at each step: would be much smaller
    // e.g., (0.5/10.5) × (0.5/10.5) × (0.5/10.5) ≈ 0.0001
    assert!(
        candidate_impact >= 0.01,
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  REGRESSION DETECTED: Deep network impact severely underestimated!            ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║  Candidate neuron (3 hops from output) has impact = {candidate_impact:.6}     \n\
        ║                                                                              ║\n\
        ║  Expected: ~0.125 (0.5 × 0.5 × 0.5)                                           ║\n\
        ║  With normalisation bug: ~0.0001 (compounding underestimate)                  ║\n\
        ║                                                                              ║\n\
        ║  Deep networks compound the normalisation error at each layer.                ║\n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n\n"
    );

    // Should be approximately 0.125
    assert!(
        (candidate_impact - 0.125).abs() < 0.05,
        "Candidate impact should be ≈ 0.125 (product of absolute weights), got {candidate_impact:.4}"
    );
}
