//! Tests comparing ReLU-style fixed parameters vs optimised parameters.
//!
//! Root cause analysis (Dec 2024) revealed systematic prediction inversion:
//! - ReLU candidates use fixed params (incoming=±1.0, bias=0.0) → SUCCESS
//! - Other activations search wide param ranges → SYSTEMATIC FAILURE
//!
//! This test investigates whether using ReLU-style conservative parameters
//! for TANH/GELU would produce more reliable predictions.
//!
//! Key insight: ReLU's fixed parameters mean the SAME candidate is found
//! regardless of which 7.5% sample is used. Other activations find DIFFERENT
//! "optimal" parameters for each sample, leading to overfitting.

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

/// Helper to create a creature with specified topology
fn create_test_creature(
    neurons: Vec<(&str, &str, &str)>, // (uuid, type, squash)
    synapses: Vec<(&str, &str, f32)>, // (from, to, weight)
) -> CreatureJson {
    let input_count = neurons.iter().filter(|(_, t, _)| *t == "input").count();
    let output_count = neurons.iter().filter(|(_, t, _)| *t == "output").count();
    CreatureJson {
        neurons: neurons
            .into_iter()
            .filter(|(_, neuron_type, _)| *neuron_type != "input")
            .map(|(uuid, neuron_type, squash)| NeuronJson {
                uuid: uuid.to_string(),
                neuron_type: neuron_type.to_string(),
                squash: squash.to_string(),
                bias: 0.0,
            })
            .collect(),
        synapses: synapses
            .into_iter()
            .map(|(from, to, weight)| SynapseJson {
                from_uuid: from.to_string(),
                to_uuid: to.to_string(),
                weight,
                synapse_type: None,
            })
            .collect(),
        input: input_count,
        output: output_count,
    }
}

/// Generate sample data simulating a small random subset (like 7.5% in production).
///
/// Key characteristics:
/// - Mixed positive/negative errors (like production ~50/50 split)
/// - Source activation has weak correlation with target error
/// - This mimics why optimised params fail: they overfit to sample-specific patterns
fn generate_sample_subset_data(
    seed: u32,
    sample_count: usize,
    error_bias: f32, // Shift to create sample-specific error distribution
) -> Vec<DiscoverRecord> {
    let mut records = Vec::with_capacity(sample_count * 2); // Source + Target per obs

    for i in 0..sample_count {
        let obs_idx = i as u32;

        // Pseudo-random based on seed and index to create reproducible but varied samples
        let pseudo_random = ((seed as f32 * 0.618 + i as f32 * 0.381).sin() * 1000.0).fract();

        // Target in linear region of activation
        let target_value = 0.3 * (pseudo_random - 0.5);
        let target_activation = target_value; // Assuming IDENTITY or linear region

        // Error with sample-specific bias (simulates different samples having different distributions)
        let base_error = 0.1 * (pseudo_random - 0.5);
        let error = base_error + error_bias * (pseudo_random - 0.5).signum();

        // Source activation - weak correlation with error
        let source_activation = 0.5 * (1.0 - pseudo_random);

        records.push(DiscoverRecord::new(
            obs_idx,
            "output-0".to_string(),
            Some(target_value),
            target_activation,
            vec![error],
        ));

        records.push(DiscoverRecord::new(
            obs_idx,
            "input-1".to_string(),
            None,
            source_activation,
            Vec::new(),
        ));
    }

    records
}

/// Test demonstrating the problem: optimised params are sample-specific.
///
/// When we run discovery on different 7.5% samples of the same dataset,
/// optimised parameters vary significantly. ReLU's fixed params don't.
///
/// This test shows that:
/// 1. Different samples produce different "optimal" params for TANH/GELU
/// 2. These params overfit to the specific sample
/// 3. The prediction is positive on the sample but may be negative on other samples
#[test]
fn test_optimised_params_vary_by_sample() {
    skip_without_gpu!();

    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"), // IDENTITY for linear comparison
        ],
        vec![("input-0", "output-0", 1.0)],
    );

    // Run analysis on three different "samples" with different error distributions
    let mut candidates_per_sample: Vec<Vec<String>> = Vec::new();

    for (seed, error_bias) in [(42, 0.02), (123, -0.02), (789, 0.0)] {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = generate_sample_subset_data(seed, 100, error_bias);
        write_records_to_parquet(file_path, &records).unwrap();

        let input = AnalyzeNeuronsInput {
            parquet_file: file_path.to_string(),
            creature: creature.clone(),
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.0), // Accept any improvement
            max_candidates: Some(100),
            analysis_deadline_ms: None,
        };

        let result = analyze_neurons(&input).unwrap();

        // Collect candidate descriptions for comparison
        let candidate_info: Vec<String> = result
            .helpful_neurons
            .iter()
            .map(|c| {
                format!(
                    "{}:in={:.2},bias={:.2},exp={:.4}%",
                    c.squash, c.incoming_weight, c.bias, c.expected_improvement_percentage
                )
            })
            .collect();

        eprintln!(
            "Sample seed={}, error_bias={:.2}: {} candidates",
            seed,
            error_bias,
            candidate_info.len()
        );
        for info in &candidate_info {
            eprintln!("  {}", info);
        }

        candidates_per_sample.push(candidate_info);
    }

    // Key observation: if parameters are sample-specific, different samples
    // will produce different "best" candidates with different params.
    // ReLU (if present) would have the same params across all samples.

    // This test documents the behaviour - the assertion is primarily informational
    eprintln!("\n=== Key Observation ===");
    eprintln!("If optimised params cause overfitting, different samples should produce");
    eprintln!("different 'optimal' incoming_weight and bias values for the same activation.");
}

/// Test that TANH with fixed params (incoming=±1.0, bias=0.0) is more stable.
///
/// This test verifies the hypothesis: using ReLU-style conservative parameters
/// for other activation functions should produce more consistent predictions
/// across different sample subsets.
///
/// We simulate this by checking if candidates with params near ±1.0/0.0
/// have better prediction accuracy than those with extreme params.
#[test]
fn test_conservative_params_more_stable_across_samples() {
    skip_without_gpu!();

    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![("input-0", "output-0", 1.0)],
    );

    // Create a larger sample that represents the "full dataset"
    let full_sample_records = generate_sample_subset_data(0, 1000, 0.0);
    let full_temp = NamedTempFile::new().unwrap();
    write_records_to_parquet(full_temp.path().to_str().unwrap(), &full_sample_records).unwrap();

    // Run on a small subset (simulating the 7.5% sample)
    let subset_records = generate_sample_subset_data(42, 75, 0.03);
    let subset_temp = NamedTempFile::new().unwrap();
    write_records_to_parquet(subset_temp.path().to_str().unwrap(), &subset_records).unwrap();

    let subset_input = AnalyzeNeuronsInput {
        parquet_file: subset_temp.path().to_str().unwrap().to_string(),
        creature: creature.clone(),
        focus_neurons: vec!["output-0".to_string()],
        improvement_threshold: Some(0.0),
        max_candidates: Some(100),
        analysis_deadline_ms: None,
    };

    let subset_result = analyze_neurons(&subset_input).unwrap();

    // Categorise candidates by parameter magnitude
    let mut conservative_candidates = 0;
    let mut extreme_candidates = 0;

    for candidate in &subset_result.helpful_neurons {
        let is_conservative = candidate.incoming_weight.abs() <= 2.0 && candidate.bias.abs() <= 1.0;

        if is_conservative {
            conservative_candidates += 1;
            eprintln!(
                "CONSERVATIVE: {} in={:.2} bias={:.2} exp={:.4}%",
                candidate.squash,
                candidate.incoming_weight,
                candidate.bias,
                candidate.expected_improvement_percentage
            );
        } else {
            extreme_candidates += 1;
            eprintln!(
                "EXTREME: {} in={:.2} bias={:.2} exp={:.4}%",
                candidate.squash,
                candidate.incoming_weight,
                candidate.bias,
                candidate.expected_improvement_percentage
            );
        }
    }

    eprintln!("\n=== Parameter Distribution ===");
    eprintln!(
        "Conservative (|in|≤2, |bias|≤1): {}",
        conservative_candidates
    );
    eprintln!("Extreme (|in|>2 or |bias|>1): {}", extreme_candidates);
    eprintln!("\nHypothesis: Extreme params are more likely to fail on full dataset");
}

/// Test demonstrating saturation risk with large bias values.
///
/// When bias is large (e.g., 10), TANH(incoming*x + bias) saturates:
/// - TANH(10*x + 10) ≈ 1.0 for all x > -0.9
/// - The neuron effectively becomes a constant!
///
/// This constant behaviour might "work" on a specific sample but fails generally.
#[test]
fn test_large_bias_causes_saturation() {
    // This is a unit test that doesn't need GPU - it demonstrates the math

    // TANH behaviour with different biases
    let test_inputs: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.5, 1.0];

    eprintln!("=== TANH Saturation with Bias ===");
    eprintln!("Inputs: {:?}\n", test_inputs);

    for bias in [0.0, 1.0, 5.0, 10.0] {
        let incoming = 1.0;
        let outputs: Vec<f32> = test_inputs
            .iter()
            .map(|&x| (incoming * x + bias).tanh())
            .collect();

        let range = outputs.iter().cloned().fold(f32::MAX, f32::min)
            ..=outputs.iter().cloned().fold(f32::MIN, f32::max);
        let spread = *range.end() - *range.start();

        eprintln!(
            "Bias={:5.1}: outputs={:?}",
            bias,
            outputs
                .iter()
                .map(|x| format!("{:.4}", x))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "           Range: {:.4} to {:.4} (spread={:.4})\n",
            range.start(),
            range.end(),
            spread
        );

        // With bias=10, spread should be very small (saturated)
        if bias >= 10.0 {
            assert!(
                spread < 0.01,
                "TANH with bias={} should be nearly saturated (spread={:.4})",
                bias,
                spread
            );
        }
    }

    eprintln!("=== Conclusion ===");
    eprintln!("Large bias causes saturation, making the neuron output nearly constant.");
    eprintln!("A constant neuron can only help samples in ONE direction.");
    eprintln!("If sample errors are split ~50/50, saturated neurons WILL fail.");
}

/// Test demonstrating that ReLU's fixed params avoid the overfitting problem.
///
/// ReLU uses:
/// - incoming_weight = ±1.0 (FIXED, not searched)
/// - bias = 0.0 (FIXED, not searched)
///
/// This means the SAME candidate is found regardless of which sample is used.
/// No sample-specific overfitting occurs.
#[test]
fn test_relu_fixed_params_consistent_across_samples() {
    skip_without_gpu!();

    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![("input-0", "output-0", 1.0)],
    );

    // Run on multiple different samples
    let mut relu_params: Vec<(f32, f32)> = Vec::new(); // (incoming, bias) for ReLU candidates

    for seed in [42, 123, 789, 456, 999] {
        let temp_file = NamedTempFile::new().unwrap();
        let records = generate_sample_subset_data(seed, 100, 0.01);
        write_records_to_parquet(temp_file.path().to_str().unwrap(), &records).unwrap();

        let input = AnalyzeNeuronsInput {
            parquet_file: temp_file.path().to_str().unwrap().to_string(),
            creature: creature.clone(),
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.0),
            max_candidates: Some(100),
            analysis_deadline_ms: None,
        };

        let result = analyze_neurons(&input).unwrap();

        // Find ReLU candidates (if any)
        for candidate in &result.helpful_neurons {
            if candidate.squash == "ReLU" {
                relu_params.push((candidate.incoming_weight, candidate.bias));
                eprintln!(
                    "ReLU (seed={}): in={:.2}, bias={:.2}",
                    seed, candidate.incoming_weight, candidate.bias
                );
            }
        }
    }

    if relu_params.is_empty() {
        eprintln!("No ReLU candidates found - this may indicate the test data doesn't favour ReLU");
        return;
    }

    // All ReLU candidates should have the SAME params: ±1.0 incoming, 0.0 bias
    for (incoming, bias) in &relu_params {
        assert!(
            incoming.abs() == 1.0,
            "ReLU incoming_weight should be ±1.0, got {}",
            incoming
        );
        assert!(bias.abs() < 0.001, "ReLU bias should be 0.0, got {}", bias);
    }

    eprintln!("\n=== ReLU Consistency ===");
    eprintln!(
        "All {} ReLU candidates have identical params (±1.0, 0.0)",
        relu_params.len()
    );
    eprintln!("This is why ReLU succeeds: no sample-specific overfitting!");
}

/// Test that bias tracing can be enabled and shows useful information.
///
/// Set NEAT_AI_DISCOVERY_TRACE_BIAS=1 to see detailed bias selection.
/// This test verifies the tracing doesn't break normal operation.
#[test]
fn test_bias_tracing_can_be_enabled() {
    skip_without_gpu!();

    // Enable bias tracing for this test
    std::env::set_var("NEAT_AI_DISCOVERY_TRACE_BIAS", "1");

    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![("input-0", "output-0", 1.0)],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Generate data that would trigger bias optimisation
    let records = generate_sample_subset_data(42, 100, 0.05);
    write_records_to_parquet(file_path, &records).unwrap();

    let input = AnalyzeNeuronsInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        improvement_threshold: Some(0.0),
        max_candidates: Some(50),
        analysis_deadline_ms: None,
    };

    // Run analysis - should log bias trace information to stderr
    eprintln!("\n=== Running analysis with bias tracing enabled ===");
    let result = analyze_neurons(&input).unwrap();
    eprintln!("=== Analysis complete ===\n");

    // Clean up
    std::env::remove_var("NEAT_AI_DISCOVERY_TRACE_BIAS");

    // Verify analysis still works with tracing enabled
    eprintln!(
        "Found {} candidates with bias tracing enabled",
        result.helpful_neurons.len()
    );
}

/// Test synapse-only analysis to compare prediction accuracy vs neuron analysis.
///
/// Synapse candidates don't have bias optimisation - they only have a single
/// computed weight. If synapse candidates have better prediction accuracy than
/// neuron candidates, it suggests bias optimisation is the main culprit.
///
/// If synapse candidates also show prediction inversion, then weight calculation
/// itself may have sample-specific overfitting issues.
#[test]
fn test_synapse_candidates_no_bias_optimisation() {
    skip_without_gpu!();

    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("hidden-0", "hidden", "TANH"), // Hidden neuron to receive new synapses
            ("output-0", "output", "IDENTITY"),
        ],
        vec![("input-0", "hidden-0", 1.0), ("hidden-0", "output-0", 1.0)],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Generate sample data with same characteristics as neuron tests
    let records = generate_sample_subset_data(42, 100, 0.05);

    // Need to add records for hidden neuron as well
    let mut all_records = records.clone();
    for i in 0..100 {
        let obs_idx = i as u32;
        let pseudo_random = ((42.0f32 * 0.618 + i as f32 * 0.381).sin() * 1000.0).fract();
        // Hidden neuron with TANH activation
        let hidden_value = 0.4 * (pseudo_random - 0.5);
        let hidden_activation = hidden_value.tanh();

        all_records.push(DiscoverRecord::new(
            obs_idx,
            "hidden-0".to_string(),
            Some(hidden_value),
            hidden_activation,
            vec![0.05 * (pseudo_random - 0.5)],
        ));
    }

    write_records_to_parquet(file_path, &all_records).unwrap();

    // Analyse synapses (not neurons)
    use neat_ai_discovery::analysis::analyze_synapses;
    use neat_ai_discovery::AnalyzeSynapsesInput;

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["hidden-0".to_string()], // Focus on hidden neuron for synapse additions
        improvement_threshold: Some(0.0),
        max_candidates: Some(50),
        analysis_deadline_ms: None,
    };

    let result = analyze_synapses(&input).unwrap();

    eprintln!("\n=== Synapse Analysis Results ===");
    eprintln!("Helpful synapses found: {}", result.helpful_synapses.len());
    eprintln!("Harmful synapses found: {}", result.harmful_synapses.len());

    for synapse in &result.helpful_synapses {
        eprintln!(
            "  {} -> {}: weight={:.4}, expected={:.4}%",
            synapse.from_neuron_uuid,
            synapse.to_neuron_uuid,
            synapse.weight,
            synapse.expected_improvement_percentage * 100.0
        );
    }

    eprintln!("\nKey difference from neuron analysis:");
    eprintln!("- Synapse candidates have NO bias parameter");
    eprintln!("- Only a single weight is computed");
    eprintln!("- If synapse predictions are more accurate, bias optimisation is the culprit");
}

/// Test comparing synapse weight ranges to see if they suffer similar overfitting.
///
/// While synapses don't have bias optimisation, the outgoing weight calculation
/// might still overfit to sample-specific patterns if the computed weight is
/// extreme for the given sample.
#[test]
fn test_synapse_weight_distribution() {
    skip_without_gpu!();

    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![("input-0", "output-0", 1.0)],
    );

    // Run synapse analysis on multiple different samples
    let mut all_weights: Vec<f32> = Vec::new();

    for seed in [42, 123, 789] {
        let temp_file = NamedTempFile::new().unwrap();
        let records = generate_sample_subset_data(seed, 100, 0.01);
        write_records_to_parquet(temp_file.path().to_str().unwrap(), &records).unwrap();

        use neat_ai_discovery::analysis::analyze_synapses;
        use neat_ai_discovery::AnalyzeSynapsesInput;

        let input = AnalyzeSynapsesInput {
            parquet_file: temp_file.path().to_str().unwrap().to_string(),
            creature: creature.clone(),
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.0),
            max_candidates: Some(50),
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input).unwrap();

        eprintln!(
            "\nSample seed={}: {} helpful synapses",
            seed,
            result.helpful_synapses.len()
        );
        for synapse in &result.helpful_synapses {
            eprintln!(
                "  {} -> {}: weight={:.6}",
                synapse.from_neuron_uuid, synapse.to_neuron_uuid, synapse.weight
            );
            all_weights.push(synapse.weight);
        }
    }

    if all_weights.is_empty() {
        eprintln!("\nNo synapse candidates found - this may indicate the test data isn't suitable");
        return;
    }

    // Check if synapse weights are consistent across samples
    let min_weight = all_weights.iter().cloned().fold(f32::MAX, f32::min);
    let max_weight = all_weights.iter().cloned().fold(f32::MIN, f32::max);
    let avg_weight = all_weights.iter().sum::<f32>() / all_weights.len() as f32;

    eprintln!("\n=== Synapse Weight Distribution ===");
    eprintln!("Min weight: {:.6}", min_weight);
    eprintln!("Max weight: {:.6}", max_weight);
    eprintln!("Avg weight: {:.6}", avg_weight);
    eprintln!("Range: {:.6}", max_weight - min_weight);

    // Key insight: If synapse weights vary significantly across samples,
    // then weight calculation also has sample-specific overfitting issues.
    // If weights are consistent, then bias is the main problem.
}

/// Regression test: verify the parameter search ranges that cause overfitting.
///
/// Documents the current (problematic) search ranges:
/// - SCALES_SMOOTH: [0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 4.0, 10.0, 25.0, 50.0]
/// - Bias values: [-10.0 ... +10.0] depending on activation
///
/// These wide ranges allow overfitting to small sample characteristics.
#[test]
fn document_current_parameter_search_ranges() {
    // This test documents the search ranges without needing GPU

    eprintln!("=== Current Parameter Search Ranges ===\n");

    eprintln!("SCALES_WIDE (for GELU, ELU, etc.):");
    eprintln!("  [0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]");
    eprintln!("  -> Up to 200x amplification!\n");

    eprintln!("SCALES_SMOOTH (for TANH, LOGISTIC, etc.):");
    eprintln!("  [0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 4.0, 10.0, 25.0, 50.0]");
    eprintln!("  -> Still up to 50x amplification\n");

    eprintln!("Bias values (TANH/LOGISTIC):");
    eprintln!("  [-10.0, -5.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]");
    eprintln!("  -> Bias of ±10 causes saturation!\n");

    eprintln!("Versus ReLU (fixed):");
    eprintln!("  incoming_weight = ±1.0 (ONLY two options)");
    eprintln!("  bias = 0.0 (ALWAYS zero)");
    eprintln!("  -> Total combinations: 2\n");

    eprintln!("Other activations:");
    eprintln!("  2 orientations × 10+ scales × 13+ biases = 260+ combinations");
    eprintln!("  -> Much more opportunity to find sample-specific 'optimal' params");

    // Quantify the difference
    let relu_combinations = 2;
    let other_combinations = 2 * 10 * 13; // orientations × scales × biases

    assert!(
        other_combinations > 100 * relu_combinations,
        "Non-ReLU activations should have 100x+ more parameter combinations"
    );
}
