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

/// REGRESSION TEST: Removal candidates must be found for neurons with negligible contribution.
///
/// A neuron is a removal candidate when its contribution to output is less than the
/// complexity savings from removing it:
///
///   activation_weighted_impact < savings
///
/// Where savings = growthCost × (1 + (incoming + outgoing) / 10)
///
/// For a neuron with 1 outgoing synapse:
///   savings = 1e-7 × (1 + 1/10) = 1.1e-7
///
/// Removal candidate when: activation_weighted_impact < 1.1e-7
#[test]
fn regression_removal_candidates_found_for_negligible_neurons() {
    use neat_ai_discovery::focus::rank_focus_neurons;
    use neat_ai_discovery::parquet_format::write_records_to_parquet;
    use neat_ai_discovery::types::DiscoverRecord;
    use tempfile::NamedTempFile;

    // Network with a neuron that has TRULY negligible weights:
    //   negligible → output-0 (weight 1e-8)
    //
    // Structural impact = 1e-8 × 1.0 = 1e-8
    // With mean_activation = 0.5:
    //   activation_weighted_impact = 1e-8 × 0.5 = 5e-9
    //
    // Synapse count: 0 incoming, 1 outgoing
    // Removal savings = 1e-7 × (1 + 1/10) = 1.1e-7
    //
    // 5e-9 < 1.1e-7 ✓ (contribution < savings, so removal improves score)
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
        synapses: vec![SynapseJson {
            from_uuid: "negligible".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 1e-8, // Truly negligible weight
        }],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records with typical activation (0.5)
    let records = vec![
        DiscoverRecord::new(0, "negligible".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "negligible".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // Verify the impact calculation is correct (absolute weights)
    let negligible_neuron = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "negligible")
        .expect("negligible neuron should be in results");

    assert!(
        (negligible_neuron.impact - 1e-8).abs() < 1e-9,
        "Impact should be ~1e-8 (absolute weight), got {}",
        negligible_neuron.impact
    );

    // Verify activation_weighted_impact = impact × activation = 1e-8 × 0.5 = 5e-9
    assert!(
        (negligible_neuron.activation_weighted_impact - 5e-9).abs() < 1e-9,
        "activation_weighted_impact should be ~5e-9, got {}",
        negligible_neuron.activation_weighted_impact
    );

    // CRITICAL: This neuron should be a removal candidate.
    //
    // Clean criterion (no scale factor):
    //   - 0 incoming synapses, 1 outgoing synapse
    //   - savings = 1e-7 × (1 + 1/10) = 1.1e-7
    //   - activation_weighted_impact = 5e-9 < 1.1e-7 ✓
    //   - Contribution is smaller than complexity cost, so removal improves score
    let removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "negligible");

    assert!(
        removal.is_some(),
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════════════════════╗\n\
        ║  REGRESSION: No removal candidates found for neuron with negligible contribution! ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════════╣\n\
        ║  Neuron 'negligible' has:                                                         \n\
        ║    - structural_impact = {:.2e}                                                   \n\
        ║    - mean_activation = {:.2}                                                      \n\
        ║    - activation_weighted_impact = {:.2e}                                          \n\
        ║                                                                                   ║\n\
        ║  Clean criterion (0 in + 1 out synapse):                                          ║\n\
        ║    savings = 1e-7 × 1.1 = 1.1e-7                                                  ║\n\
        ║    activation_weighted_impact ({:.2e}) < savings (1.1e-7) should pass             ║\n\
        ║                                                                                   ║\n\
        ║  Found {} removal candidates: {:?}                                                \n\
        ╚══════════════════════════════════════════════════════════════════════════════════╝\n\n",
        negligible_neuron.impact,
        negligible_neuron.mean_activation,
        negligible_neuron.activation_weighted_impact,
        negligible_neuron.activation_weighted_impact,
        result.removal_candidates.len(),
        result
            .removal_candidates
            .iter()
            .map(|c| &c.neuron_uuid)
            .collect::<Vec<_>>()
    );

    // Verify the removal candidate has correct synapse counts
    let candidate = removal.unwrap();
    assert_eq!(
        candidate.incoming_synapses, 0,
        "Should have 0 incoming synapses"
    );
    assert_eq!(
        candidate.outgoing_synapses, 1,
        "Should have 1 outgoing synapse"
    );
    assert!(
        (candidate.removal_savings - 1.1e-7).abs() < 1e-9,
        "Removal savings should be ~1.1e-7, got {}",
        candidate.removal_savings
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
