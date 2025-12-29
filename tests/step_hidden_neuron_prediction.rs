//! Test case for STEP squash hidden neuron prediction accuracy.
//!
//! This test demonstrates the bug where add-neuron predictions for STEP hidden neurons
//! report "flip rate" as expected improvement, but actual creature error reduction
//! is nearly zero.
//!
//! The issue: For discrete activation functions (STEP/BIPOLAR), we calculate
//! "what % of samples would flip" but NOT "what % of creature error would reduce".
//! These are completely different metrics!
//!
//! When the target is a HIDDEN STEP neuron:
//! - Flipping the hidden neuron changes its contribution to downstream neurons
//! - The actual error reduction depends on how this propagates to output neurons
//! - The prediction incorrectly equates "flip rate" with "error reduction"

mod common;

use neat_ai_discovery::analysis::{analyze_neurons, GpuAnalyzer};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeNeuronsInput, CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Helper to create a creature matching the problematic structure from GRQ-18-1:
/// - Source neuron (input or hidden)
/// - Hidden STEP neuron (the focus neuron with threshold activation)
/// - Output neuron that receives from the STEP hidden neuron
fn create_step_hidden_creature() -> CreatureJson {
    CreatureJson {
        input: 5, // input-0 through input-4
        output: 1,
        neurons: vec![
            // Hidden STEP neuron - this is the focus neuron
            NeuronJson {
                uuid: "hidden-step".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "STEP".to_string(),
                bias: 0.005, // Small bias like GRQ-18-1
            },
            // Output neuron that receives from the STEP hidden
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            // Input connections to the STEP hidden neuron
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-step".to_string(),
                weight: 4.5, // Similar to GRQ-18-1
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-step".to_string(),
                weight: -11.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-2".to_string(),
                to_uuid: "hidden-step".to_string(),
                weight: 344.0, // Large weight like input-211 in GRQ-18-1
                synapse_type: None,
            },
            // STEP hidden to output - small weight, so flipping STEP has small effect
            SynapseJson {
                from_uuid: "hidden-step".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.001, // Tiny weight - flipping STEP barely affects output!
                synapse_type: None,
            },
            // Direct input to output (dominant connection)
            SynapseJson {
                from_uuid: "input-3".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
    }
}

/// Generate synthetic records that would trigger a high "flip rate" prediction
/// for the STEP hidden neuron, but result in near-zero actual creature error reduction.
fn generate_step_flip_records() -> Vec<DiscoverRecord> {
    let mut records = Vec::new();

    // Generate 100 observations
    for obs in 0..100u32 {
        // Input neurons
        for i in 0..5 {
            records.push(DiscoverRecord::new(
                obs,
                format!("input-{i}"),
                Some(if i == 4 { 0.5 } else { 0.1 }), // Source neuron (input-4) has some activation
                if i == 4 { 0.5 } else { 0.1 },
                vec![0.0], // Inputs have no error
            ));
        }

        // Hidden STEP neuron
        // For ~50% of samples: value just below 0 (outputs 0), should output 1 (error > 0)
        // For ~50% of samples: value just above 0 (outputs 1), correct (error = 0)
        let value = if obs < 50 { -0.05 } else { 0.05 };
        let activation = if value > 0.0 { 1.0 } else { 0.0 }; // STEP function
        let error = if obs < 50 {
            0.5 // Wants to flip 0->1
        } else {
            0.0 // Correct
        };

        records.push(DiscoverRecord::new(
            obs,
            "hidden-step".to_string(),
            Some(value),
            activation,
            vec![error],
        ));

        // Output neuron - receives from hidden-step (weight 0.001) and input-3 (weight 1.0)
        // Output ≈ input-3 × 1.0 + hidden-step × 0.001
        // Since hidden-step contribution is tiny (0 or 0.001), output error is mostly from input-3
        let output_value = 0.1 * 1.0 + activation * 0.001;
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(output_value),
            output_value, // IDENTITY squash
            vec![0.05],   // Some baseline error
        ));
    }

    records
}

/// Test that demonstrates the bug: high "expected improvement" but near-zero actual reduction.
///
/// The prediction says "~50% of samples would flip" (the hidden STEP neuron),
/// but actual creature error reduction is near-zero because:
/// 1. The STEP neuron has tiny weight (0.001) to output
/// 2. Flipping 0->1 or 1->0 only changes output by ±0.001
/// 3. This tiny change barely affects MSE
#[test]
fn test_step_hidden_neuron_prediction_vs_reality() {
    skip_without_gpu!();

    let creature = create_step_hidden_creature();
    let records = generate_step_flip_records();

    let temp_file = NamedTempFile::new().unwrap();
    let parquet_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(parquet_path, &records).unwrap();

    // Run add-neuron analysis targeting the STEP hidden neuron
    let input = AnalyzeNeuronsInput {
        parquet_file: parquet_path.to_string(),
        creature: creature.clone(),
        focus_neurons: vec!["hidden-step".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: None,
    };

    let result = analyze_neurons(&input).expect("Analysis should succeed");

    println!(
        "Analysis result: {} candidates",
        result.helpful_neurons.len()
    );

    // If we found any candidates for this STEP hidden neuron, check the prediction accuracy
    for candidate in &result.helpful_neurons {
        if candidate.target_neuron_uuid == "hidden-step" {
            let expected_pct = candidate.expected_creature_score_gain;
            println!(
                "Candidate {} -> {}: expected improvement = {:.4}%",
                candidate.source_neuron_uuid,
                candidate.target_neuron_uuid,
                expected_pct * 100.0
            );

            // The bug: For STEP hidden neurons, `expected_creature_score_gain` is "flip rate"
            // not actual error reduction. If this is > 5%, the prediction is likely wrong.
            //
            // With our test data, ~50% of samples would flip, so expected_pct might be ~0.5 (50%)
            // But actual creature error reduction would be near-zero due to tiny weight.
            //
            // A correct implementation should either:
            // 1. Not recommend add-neuron for STEP hidden neurons
            // 2. Calculate actual predicted MSE change, not flip rate
            // 3. Apply impact discount BEFORE reporting (flip_rate × impact = actual reduction)

            // This assertion documents the bug - expected improvement should be tiny because
            // the STEP neuron has minimal impact on output
            let impact = 0.001; // hidden-step's weight to output
            let max_reasonable_improvement = expected_pct * impact;

            // The prediction should be proportional to actual impact
            // If expected_pct is high but impact is low, prediction is misleading
            if expected_pct > 0.05 {
                // More than 5% "flip rate"
                println!(
                    "BUG DETECTED: High flip rate ({:.2}%) but actual creature impact would be tiny",
                    expected_pct * 100.0
                );
                println!("  - Hidden STEP neuron impact on output: {impact}");
                println!(
                    "  - Maximum reasonable improvement: {:.6}%",
                    max_reasonable_improvement * 100.0
                );

                // The prediction should NOT exceed what's physically possible
                // given the neuron's impact on outputs
                assert!(
                    expected_pct <= 0.01 || max_reasonable_improvement >= expected_pct * 0.1,
                    "STEP hidden neuron prediction is misleading: reports {:.2}% improvement but \
                    neuron has only {:.4} impact on output. Max reasonable: {:.6}%",
                    expected_pct * 100.0,
                    impact,
                    max_reasonable_improvement * 100.0
                );
            }
        }
    }
}

/// Test that IDENTITY neuron with bias=0 is NOT recommended for STEP targets.
///
/// IDENTITY(x + 0) × w = x × w, which is equivalent to a direct synapse.
/// For STEP/BIPOLAR targets, add-neuron with IDENTITY+bias=0 wastes a neuron
/// for no benefit - the same effect could be achieved with add-synapse.
///
/// The discrete evaluation now correctly filters these candidates.
#[test]
fn test_identity_zero_bias_should_not_be_recommended() {
    skip_without_gpu!();

    let creature = create_step_hidden_creature();
    let records = generate_step_flip_records();

    let temp_file = NamedTempFile::new().unwrap();
    let parquet_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(parquet_path, &records).unwrap();

    let input = AnalyzeNeuronsInput {
        parquet_file: parquet_path.to_string(),
        creature,
        focus_neurons: vec!["hidden-step".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: None,
    };

    let result = analyze_neurons(&input).expect("Analysis should succeed");

    // STEP targets should NOT have IDENTITY+bias=0 candidates
    // These are now correctly filtered because they're equivalent to synapses
    for candidate in &result.helpful_neurons {
        if candidate.squash == "IDENTITY" && candidate.bias.abs() < 1e-10 {
            panic!(
                "BUG: IDENTITY+bias=0 candidate should be filtered! \
                Found: {} -> {} with {:.4}% improvement. \
                This is equivalent to a synapse: IDENTITY(x×{} + 0) × {} = x × {}",
                candidate.source_neuron_uuid,
                candidate.target_neuron_uuid,
                candidate.expected_creature_score_gain * 100.0,
                candidate.incoming_weight,
                candidate.outgoing_weight,
                candidate.incoming_weight * candidate.outgoing_weight
            );
        }
    }

    // For STEP targets, we expect no add-neuron candidates because:
    // 1. IDENTITY+bias=0 is filtered (equivalent to synapse)
    // 2. The discrete evaluation only tries IDENTITY squash
    // This is correct - STEP targets should use add-synapse analysis instead
    println!(
        "STEP hidden target correctly has {} add-neuron candidates (IDENTITY filtered)",
        result.helpful_neurons.len()
    );
}

/// Test that STEP output neurons also don't get IDENTITY+bias=0 candidates.
///
/// Even for OUTPUT STEP neurons, IDENTITY+bias=0 is equivalent to a synapse.
/// The discrete evaluation correctly filters these for all STEP/BIPOLAR targets.
#[test]
fn test_step_output_neuron_no_identity_candidates() {
    skip_without_gpu!();

    // Create a simple creature with STEP output
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "STEP".to_string(),
            bias: 0.0,
        }],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 1.0,
            synapse_type: None,
        }],
    };

    // Generate records where:
    // - 80% of samples: output should be 1 but is 0 (error = 1.0, value just below 0)
    // - 20% of samples: output correct (error = 0.0)
    let mut records = Vec::new();
    for obs in 0..100u32 {
        // Input neurons
        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(0.3),
            0.3,
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-1".to_string(),
            Some(0.5),
            0.5, // Source neuron we'll add from
            vec![0.0],
        ));

        // Output STEP neuron
        let value = if obs < 80 { -0.05 } else { 0.05 };
        let activation = if value > 0.0 { 1.0 } else { 0.0 };
        let error = if obs < 80 { 1.0 } else { 0.0 }; // Big error when should flip

        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(value),
            activation,
            vec![error],
        ));
    }

    let temp_file = NamedTempFile::new().unwrap();
    let parquet_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(parquet_path, &records).unwrap();

    let input = AnalyzeNeuronsInput {
        parquet_file: parquet_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: None,
    };

    let result = analyze_neurons(&input).expect("Analysis should succeed");

    println!(
        "STEP output analysis: {} candidates",
        result.helpful_neurons.len()
    );

    // STEP output targets should also NOT have IDENTITY+bias=0 candidates
    // (same reason as hidden: equivalent to synapse, wasteful)
    for candidate in &result.helpful_neurons {
        if candidate.squash == "IDENTITY" && candidate.bias.abs() < 1e-10 {
            panic!(
                "BUG: IDENTITY+bias=0 candidate for STEP output should be filtered! \
                Found: {} -> {}",
                candidate.source_neuron_uuid, candidate.target_neuron_uuid
            );
        }
    }

    // For STEP targets, we expect no add-neuron candidates
    // Add-synapse analysis should handle STEP targets instead
    assert_eq!(
        result.helpful_neurons.len(),
        0,
        "STEP targets should have no IDENTITY add-neuron candidates (use add-synapse instead)"
    );
}
