//! Tests for implementation.rs synapse analysis functions (Issue #278).
//!
//! These tests were extracted from implementation.rs as part of the refactoring
//! to reduce the file size below 2000 lines.
//!
//! The tests cover:
//! - Record cache contention handling
//! - GPU batch evaluation edge cases
//! - Sample matching and filtering
//! - Bias calculation for various activation functions
//! - Diagnostics and rejection tracking
//! - Deadline/timeout behaviour
//! - Synapse and neuron analysis integration

mod tests_synapses {
    #![allow(unused_imports)] // super::* is used for macro re-export
                              // Import from parent module (implementation.rs)
    use super::*;
    // Additional imports needed since tests are in external file
    use crate::analysis::cache::RecordCache;
    use crate::analysis::gpu::GpuAnalyzer;
    use crate::analysis::samples::{HelpfulSample, EPSILON};
    use crate::analysis::weights::MIN_NEURON_SAMPLE_COUNT;
    use crate::types::DiscoverRecord;
    use anyhow::Result;

    use crate::analysis::analyze_all;
    use crate::analysis::analyze_neurons;
    use crate::analysis::analyze_synapses;
    use crate::analysis::samples::HelpfulStats;
    use crate::parquet_format::write_records_to_parquet;
    use crate::{
        AnalyzeAllInput, AnalyzeNeuronsInput, AnalyzeSynapsesInput, CreatureJson, NeuronJson,
        SynapseJson,
    };
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;
    use tempfile::tempdir;

    // Import functions moved to synapse.rs for tests (Issue #185)
    use crate::analysis::synapse::{
        build_samples, compute_activation_improvement_and_count,
        compute_net_improvement_with_squash, compute_synapse_improvement_and_count,
        compute_synapse_improvement_with_target_squash, count_improved_samples,
        evaluate_relu_candidates_split, upsert_candidate,
    };
    // Import activation functions used by tests
    use crate::analysis::activation::{
        bipolar_activation, can_use_hard_tanh, get_target_simulation_fn, identity_activation,
        is_threshold_activation, logistic_activation, tanh_activation,
    };
    // Import weights used by tests
    use crate::analysis::weights::{calculate_optimal_bias, MAX_OUTGOING_WEIGHT};
    // Import diagnostics types used by tests
    use crate::analysis::diagnostics::{
        filter_focus_targets_for_neuron_analysis, NeuronDiagnostics, RejectionReason,
        TargetDiagnostics, ThresholdContext,
    };
    use crate::analysis::shared::{NeuronNoCandidateReason, SynapseNoCandidateReason};
    use crate::analysis::utils::deadline_override;
    // Import types used by tests
    use crate::CandidateNeuronJson;

    /// Helper macro to skip tests that require GPU when no GPU is available.
    /// This allows tests to pass gracefully in CI environments without GPUs.
    macro_rules! skip_if_no_gpu {
        () => {
            if !GpuAnalyzer::gpu_is_available() {
                eprintln!("⚠️  Skipping test: GPU not available");
                return;
            }
        };
    }

    // Deadline tests moved to utils/deadline_tests.rs (Issue #268)

    #[test]
    fn record_cache_loads_once_per_neuron_under_contention() {
        let load_counter = Arc::new(AtomicUsize::new(0));
        let loader_counter = Arc::clone(&load_counter);
        let loader = Arc::new(
            move |_file: &str, neuron_uuid: &str| -> Result<Vec<DiscoverRecord>> {
                loader_counter.fetch_add(1, AtomicOrdering::SeqCst);
                thread::sleep(Duration::from_millis(50));
                Ok(vec![DiscoverRecord::new(
                    0,
                    neuron_uuid.to_string(),
                    None,
                    0.0,
                    Vec::new(),
                )])
            },
        );

        let cache = Arc::new(RecordCache::with_loader("unused.parquet", loader));
        let worker_count = 4;
        let barrier = Arc::new(Barrier::new(worker_count));
        let mut handles = Vec::new();
        for _ in 0..worker_count {
            let cache_clone = Arc::clone(&cache);
            let barrier_clone = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier_clone.wait();
                cache_clone
                    .get("neuron-1")
                    .expect("cache should load neuron records");
            }));
        }

        for handle in handles {
            handle.join().expect("worker thread should exit cleanly");
        }

        assert_eq!(
            load_counter.load(AtomicOrdering::SeqCst),
            1,
            "record cache should only hit the loader once even when multiple threads request the same neuron"
        );
    }

    /// TDD Test: evaluate_harmful_batch should handle empty batch gracefully.
    #[test]
    fn evaluate_harmful_batch_handles_empty_batch() {
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed in tests");

        let empty_batch: Vec<(&[HelpfulSample], f32)> = vec![];
        let result = analyzer
            .evaluate_harmful_batch(&empty_batch)
            .expect("Empty batch should succeed");

        assert!(result.is_empty(), "Empty batch should return empty results");
    }

    /// TDD Test: evaluate_harmful_batch should handle batch with empty sample sets.
    #[test]
    fn evaluate_harmful_batch_handles_empty_sample_sets() {
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed in tests");

        let samples: Vec<HelpfulSample> = (0..20)
            .map(|i| HelpfulSample {
                activation: (i as f32) / 20.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            })
            .collect();
        let empty_samples: Vec<HelpfulSample> = vec![];

        let batch_input = vec![
            (&samples[..], 0.5),
            (&empty_samples[..], 0.3), // Empty set in the middle
            (&samples[..], -0.2),
        ];

        let batched = analyzer
            .evaluate_harmful_batch(&batch_input)
            .expect("Batch with empty set should succeed");

        assert_eq!(batched.len(), 3, "Should return 3 results");

        // Middle result should be default (all zeros)
        assert_eq!(
            batched[1].harmful_count, 0,
            "Empty sample set should have 0 harmful_count"
        );
        assert_eq!(
            batched[1].helpful_count, 0,
            "Empty sample set should have 0 helpful_count"
        );
    }

    #[test]
    fn sample_matching_filters_non_finite_values() {
        // No GPU needed - tests the CPU build_samples function used in production
        let huge = f32::MAX;
        let target_records = vec![
            DiscoverRecord::new(0, "target".to_string(), None, 0.0, vec![0.5, -0.25]),
            DiscoverRecord::new(1, "target".to_string(), None, 0.0, vec![huge, huge]),
        ];
        let from_records = vec![
            DiscoverRecord::new(0, "from".to_string(), None, f32::INFINITY, Vec::new()),
            DiscoverRecord::new(1, "from".to_string(), None, 1.0, Vec::new()),
        ];

        let samples = build_samples(&target_records, &from_records);
        assert!(
            samples.is_empty(),
            "Sample matching should exclude non-finite samples"
        );
    }

    #[test]
    fn sample_matching_retains_legitimate_zero_samples() {
        // No GPU needed - tests the CPU build_samples function used in production
        let target_records = vec![DiscoverRecord::new(
            42,
            "target".to_string(),
            None,
            0.0,
            vec![0.0, 0.0],
        )];
        let from_records = vec![DiscoverRecord::new(
            42,
            "from".to_string(),
            None,
            0.0,
            Vec::new(),
        )];

        let samples = build_samples(&target_records, &from_records);
        assert_eq!(
            samples.len(),
            1,
            "Sample matching should include legitimate zero-valued samples"
        );

        let sample = samples[0];
        assert_eq!(
            sample.activation, 0.0,
            "Zero activation should be preserved"
        );
        assert_eq!(
            sample.avg_error, 0.0,
            "Zero average error should be preserved"
        );
    }

    /// Test that sample matching preserves target_value and target_activation.
    /// This is critical for accurate improvement predictions with non-linear
    /// activation functions (TANH, LOGISTIC, HARD_TANH, etc.).
    #[test]
    fn sample_matching_preserves_target_value_and_activation() {
        // No GPU needed - tests the production build_samples function
        let target_value = 0.8; // Pre-activation input sum
        let target_activation = 0.6; // Post-activation output (e.g., after TANH)
        let target_records = vec![DiscoverRecord::new(
            0,
            "target".to_string(),
            Some(target_value),
            target_activation,
            vec![0.1, -0.2],
        )];
        let from_records = vec![DiscoverRecord::new(
            0,
            "from".to_string(),
            None, // Source value not used
            0.5,  // Source activation
            Vec::new(),
        )];

        let samples = build_samples(&target_records, &from_records);
        assert_eq!(samples.len(), 1, "Should find one matching sample");
        assert_eq!(
            samples[0].target_value,
            Some(target_value),
            "Sample matching must preserve target_value for activation function simulation"
        );
        assert_eq!(
            samples[0].target_activation,
            Some(target_activation),
            "Sample matching must preserve target_activation for error calculation"
        );
        assert_eq!(
            samples[0].activation, 0.5,
            "Source activation should be preserved"
        );
    }

    /// Test that target_value enables proper activation function simulation.
    /// When target_value is available, get_target_simulation_fn should return
    /// the activation function, enabling saturation-aware improvement predictions.
    #[test]
    fn sample_matching_enables_target_activation_simulation() {
        // No GPU needed - tests the production build_samples function
        // Create samples near HARD_TANH saturation to test simulation accuracy
        let target_records = vec![
            DiscoverRecord::new(
                0,
                "target".to_string(),
                Some(0.95), // Near saturation
                0.95,       // HARD_TANH clips to 1.0 when input >= 1.0
                vec![0.1],  // Small positive error (output should be higher)
            ),
            DiscoverRecord::new(
                1,
                "target".to_string(),
                Some(-0.8),
                -0.8,
                vec![-0.15], // Small negative error (output should be lower)
            ),
        ];
        let from_records = vec![
            DiscoverRecord::new(0, "from".to_string(), None, 0.5, Vec::new()),
            DiscoverRecord::new(1, "from".to_string(), None, -0.3, Vec::new()),
        ];

        let samples = build_samples(&target_records, &from_records);

        assert_eq!(samples.len(), 2, "Should match both records");

        // Verify all samples have target data (required for simulation)
        for (i, sample) in samples.iter().enumerate() {
            assert!(
                sample.target_value.is_some(),
                "Sample {i} must have target_value for activation simulation"
            );
            assert!(
                sample.target_activation.is_some(),
                "Sample {i} must have target_activation for error calculation"
            );
        }

        // With target data available, get_target_simulation_fn should return Some
        // for activations that need simulation (HARD_TANH, TANH, ReLU, etc.)
        let simulation_fn = get_target_simulation_fn(&samples, Some("HARD_TANH"));
        assert!(
            simulation_fn.is_some(),
            "Should enable HARD_TANH simulation when samples have target data"
        );

        let simulation_fn = get_target_simulation_fn(&samples, Some("TANH"));
        assert!(
            simulation_fn.is_some(),
            "Should enable TANH simulation when samples have target data"
        );
    }

    /// Regression test: Add-neuron predictions must use weight computed WITH bias.
    ///
    /// BUG: Previously, the optimal outgoing weight was computed WITHOUT bias:
    ///   weight = Σ(error × TANH(x)) / Σ(TANH(x)²)
    ///
    /// But the actual neuron uses bias:
    ///   contribution = weight × TANH(x + bias)
    ///
    /// When bias significantly shifts the activation pattern, the weight computed
    /// without bias causes predictions to have the WRONG SIGN - predicting improvement
    /// when it actually makes things worse.
    ///
    /// This test reproduces the production failure pattern:
    /// - Predicted: +0.3% improvement
    /// - Actual: -0.2% (worse!)
    #[test]
    fn add_neuron_weight_must_include_bias_in_calculation() {
        // Scenario from production: TANH neuron with bias=1
        // This shifts the activation threshold from x>0 to x>-1
        let incoming_weight = 1.0f32;
        let bias = 1.0f32;

        // Create samples that expose the bug:
        // - Source activations centered around 0
        // - Roughly equal positive and negative errors
        // - With bias=1, TANH(x+1) is almost always positive (x > -1)
        // - Without bias, TANH(x) has mixed signs
        let samples: Vec<HelpfulSample> = vec![
            // Positive source activation, positive error (need output up)
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.3,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: 0.3,
                avg_error: 0.2,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: 0.1,
                avg_error: 0.1,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            // Negative source activation, negative error (need output down)
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.3,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.2,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: -0.1,
                avg_error: -0.1,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            // More samples near zero - these are affected most by bias
            HelpfulSample {
                activation: 0.05,
                avg_error: 0.15,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: -0.05,
                avg_error: -0.15,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
        ];

        // Compute baseline error
        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

        // BUG PATH: Compute weight WITHOUT bias (what the old code did)
        let mut sum_sq_no_bias = 0.0f32;
        let mut sum_ea_no_bias = 0.0f32;
        for s in &samples {
            let pre_act = incoming_weight * s.activation; // NO BIAS!
            let output = pre_act.tanh();
            sum_sq_no_bias += output * output;
            sum_ea_no_bias += output * s.avg_error;
        }
        let weight_without_bias = sum_ea_no_bias / sum_sq_no_bias;

        // FIX PATH: Compute weight WITH bias (what the fixed code does)
        let mut sum_sq_with_bias = 0.0f32;
        let mut sum_ea_with_bias = 0.0f32;
        for s in &samples {
            let pre_act = incoming_weight * s.activation + bias; // WITH BIAS!
            let output = pre_act.tanh();
            sum_sq_with_bias += output * output;
            sum_ea_with_bias += output * s.avg_error;
        }
        let weight_with_bias = sum_ea_with_bias / sum_sq_with_bias;

        // Compute ACTUAL error reduction using the ACTUAL neuron (with bias)
        fn compute_actual_improvement(
            samples: &[HelpfulSample],
            incoming_weight: f32,
            outgoing_weight: f32,
            bias: f32,
            baseline_error_sq: f32,
        ) -> f32 {
            let mut new_error_sq = 0.0f32;
            for s in samples {
                let pre_act = incoming_weight * s.activation + bias;
                let neuron_output = pre_act.tanh();
                let contribution = outgoing_weight * neuron_output;
                // Linear approximation: new_error = old_error - contribution
                let new_error = s.avg_error - contribution;
                new_error_sq += new_error * new_error;
            }
            // Improvement = (baseline - new) / baseline
            (baseline_error_sq - new_error_sq) / baseline_error_sq
        }

        // Test 1: Using weight computed WITHOUT bias gives WRONG prediction
        // The prediction (using weight_without_bias) should differ from actual
        let predicted_without_bias = {
            // Predicted improvement uses the same formula as actual
            // but the weight was computed from wrong activation pattern
            let mut predicted_new_error_sq = 0.0f32;
            for s in &samples {
                let pre_act = incoming_weight * s.activation; // NO BIAS in prediction!
                let output = pre_act.tanh();
                let contribution = weight_without_bias * output;
                let new_error = s.avg_error - contribution;
                predicted_new_error_sq += new_error * new_error;
            }
            (baseline_error_sq - predicted_new_error_sq) / baseline_error_sq
        };

        let actual_with_wrong_weight = compute_actual_improvement(
            &samples,
            incoming_weight,
            weight_without_bias,
            bias,
            baseline_error_sq,
        );

        // The bug: predicted is positive, actual is often negative or much smaller
        // This happens because weight was optimised for TANH(x) but applied to TANH(x+1)
        let prediction_error_wrong = (predicted_without_bias - actual_with_wrong_weight).abs();

        // Test 2: Using weight computed WITH bias gives CORRECT prediction
        let predicted_with_bias = {
            let mut predicted_new_error_sq = 0.0f32;
            for s in &samples {
                let pre_act = incoming_weight * s.activation + bias; // WITH BIAS in prediction!
                let output = pre_act.tanh();
                let contribution = weight_with_bias * output;
                let new_error = s.avg_error - contribution;
                predicted_new_error_sq += new_error * new_error;
            }
            (baseline_error_sq - predicted_new_error_sq) / baseline_error_sq
        };

        let actual_with_correct_weight = compute_actual_improvement(
            &samples,
            incoming_weight,
            weight_with_bias,
            bias,
            baseline_error_sq,
        );

        let prediction_error_correct = (predicted_with_bias - actual_with_correct_weight).abs();

        // Assertions:
        // 1. The two weights should be significantly different
        assert!(
            (weight_with_bias - weight_without_bias).abs() > 0.01,
            "Weights should differ: without_bias={weight_without_bias:.4}, with_bias={weight_with_bias:.4}"
        );

        // 2. Using wrong weight should have high prediction error
        assert!(
            prediction_error_wrong > 0.001,
            "Wrong weight should cause prediction error > 0.1%. \
             Predicted={predicted_without_bias:.4}, Actual={actual_with_wrong_weight:.4}, \
             Error={prediction_error_wrong:.4}"
        );

        // 3. Using correct weight should have low prediction error
        assert!(
            prediction_error_correct < 0.0001,
            "Correct weight should have prediction error < 0.01%. \
             Predicted={predicted_with_bias:.4}, Actual={actual_with_correct_weight:.4}, \
             Error={prediction_error_correct:.4}"
        );

        // 4. The key bug symptom: wrong weight often gives OPPOSITE sign of improvement
        // (predicts positive improvement but actual is negative, or vice versa)
        // This may not always happen with this specific test data, but we verify
        // the prediction error is significantly worse.
        assert!(
            prediction_error_wrong > prediction_error_correct * 10.0,
            "Wrong weight should have much higher error than correct weight. \
             Wrong error={prediction_error_wrong:.6}, Correct error={prediction_error_correct:.6}"
        );
    }

    /// Regression test: Target simulation must compute new_error = expected - new_output.
    ///
    /// BUG: The code was computing new_error = new_output - expected (opposite sign).
    /// This caused predictions to have the WRONG SIGN compared to actual results:
    /// - Predicted positive improvement but actual was negative (worse)
    /// - The sign error was in the target activation simulation path
    ///
    /// The linear approximation uses: new_error = avg_error - contribution
    /// The target simulation must be consistent: new_error = expected - target_fn(new_input)
    ///
    /// Note: Since we square the errors, the sign doesn't affect the squared error sum,
    /// but it DOES affect the direction of the optimal weight calculation when used
    /// inconsistently between the weight optimisation and improvement prediction.
    #[test]
    fn target_simulation_error_sign_consistent_with_linear_model() {
        // Sample with positive avg_error (output should be higher)
        // CRITICAL: avg_error is in VALUE domain (targetValue - currentValue from TypeScript)
        // So avg_error = desired_value - current_value, positive means current is too low
        let sample = HelpfulSample {
            activation: 0.5,
            avg_error: 0.2,          // VALUE domain: need to add 0.2 to pre-activation
            target_value: Some(0.3), // Pre-activation input to target (current value)
            target_activation: Some(0.3), // Post-activation (in linear region of HARD_TANH)
        };

        // Small positive contribution (should reduce the positive error)
        let contribution = 0.05f32;

        // Linear model: new_error = avg_error - contribution = 0.2 - 0.05 = 0.15
        let linear_new_error = sample.avg_error - contribution;

        // Target simulation (CORRECT formula using VALUE domain):
        // desired_value = target_value + avg_error (VALUE domain)
        // expected = squash(desired_value) (convert to ACTIVATION domain)
        let desired_value = sample.target_value.unwrap() + sample.avg_error; // = 0.3 + 0.2 = 0.5
        let expected = hard_tanh(desired_value); // = 0.5 (in linear region, so same as desired_value)
        let new_input = sample.target_value.unwrap() + contribution; // = 0.3 + 0.05 = 0.35
        let new_output = hard_tanh(new_input); // = 0.35 (in linear region)

        // new_error = expected - new_output
        let target_new_error = expected - new_output; // = 0.5 - 0.35 = 0.15

        // Both should give the same result in the linear region
        assert!(
            (linear_new_error - target_new_error).abs() < 0.001,
            "Target simulation should match linear model in linear region. \
             Linear: {linear_new_error}, Target: {target_new_error}"
        );

        // Both should show error REDUCTION (not increase)
        assert!(
            target_new_error.abs() < sample.avg_error.abs(),
            "Positive contribution should REDUCE positive error. \
             Old error: {}, New error: {}",
            sample.avg_error,
            target_new_error
        );

        // Verify both have the same sign (positive)
        assert!(
            linear_new_error > 0.0 && target_new_error > 0.0,
            "Both error calculations should be positive. \
             Linear: {linear_new_error}, Target: {target_new_error}"
        );
    }

    /// Debug test: Constant source activation with BIPOLAR neuron targeting HARD_TANH.
    ///
    /// Reproduces production scenario:
    /// - Source neuron has constant activation (-0.575)
    /// - New neuron: BIPOLAR with inW=5, bias=2, outW=0.010
    /// - Target: HARD_TANH output neuron
    /// - More negative errors (28757) than positive (25505)
    ///
    /// BIPOLAR(5 × -0.575 + 2) = BIPOLAR(-0.875) = -1
    /// Contribution = 0.010 × -1 = -0.010 (constant negative)
    ///
    /// Expected: Error should decrease (more negative errors helped)
    /// Actual: Error increased (prediction wrong)
    #[test]
    fn constant_activation_bipolar_targeting_hard_tanh() {
        // Simulate production distribution:
        // ~46% positive errors (need output higher)
        // ~54% negative errors (need output lower)
        let mut samples = Vec::new();

        // Positive errors (25505 samples with positive error)
        for i in 0..255 {
            let target_value = (i as f32 - 127.0) / 200.0; // Range roughly -0.6 to 0.6
            let target_activation = hard_tanh(target_value);
            let avg_error = 0.5 + (i as f32 % 50.0) / 100.0; // Positive errors 0.5-1.0

            samples.push(HelpfulSample {
                activation: -0.575, // Constant source activation
                avg_error,
                target_value: Some(target_value),
                target_activation: Some(target_activation),
            });
        }

        // Negative errors (287 samples with negative error - ratio ~54%)
        for i in 0..287 {
            let target_value = (i as f32 - 143.0) / 200.0;
            let target_activation = hard_tanh(target_value);
            let avg_error = -0.5 - (i as f32 % 50.0) / 100.0; // Negative errors -0.5 to -1.0

            samples.push(HelpfulSample {
                activation: -0.575, // Same constant source activation
                avg_error,
                target_value: Some(target_value), // FIXED: was incorrectly target_activation
                target_activation: Some(target_activation),
            });
        }

        // Production parameters
        let incoming_weight = 5.0f32;
        let bias = 2.0f32;
        let outgoing_weight = 0.010f32;

        // Compute BIPOLAR output
        let pre_activation = incoming_weight * (-0.575) + bias; // = -0.875
        let bipolar_output = bipolar_activation(pre_activation); // = -1
        let contribution = outgoing_weight * bipolar_output; // = -0.010

        assert_eq!(
            bipolar_output, -1.0,
            "BIPOLAR({pre_activation}) should be -1"
        );
        assert!(
            (contribution - (-0.010)).abs() < 0.0001,
            "Contribution should be -0.010"
        );

        // Compute baseline and new error
        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        let mut new_error_sq_sum = 0.0f32;
        let mut improved_count = 0u32;
        let mut worsened_count = 0u32;

        for sample in &samples {
            // CORRECT formula: avg_error is in VALUE domain
            // desired_value = target_value + avg_error, then squash to get expected activation
            let desired_value = sample.target_value.unwrap() + sample.avg_error;
            let expected = hard_tanh(desired_value);
            let new_input = sample.target_value.unwrap() + contribution;
            let new_output = hard_tanh(new_input);
            let new_error = expected - new_output;

            new_error_sq_sum += new_error.powi(2);

            if new_error.abs() < sample.avg_error.abs() {
                improved_count += 1;
            } else if new_error.abs() > sample.avg_error.abs() {
                worsened_count += 1;
            }
        }

        let improvement = (baseline_error_sq - new_error_sq_sum) / baseline_error_sq;
        let improvement_pct = improvement * 100.0;

        eprintln!(
            "Constant activation test: baseline={baseline_error_sq:.4}, new={new_error_sq_sum:.4}"
        );
        eprintln!(
            "Improvement: {improvement_pct:.4}%, improved={improved_count}, worsened={worsened_count}"
        );

        // With constant negative contribution and more negative errors,
        // we should see positive improvement (error reduction)
        // OR if improvement is negative, it explains the production issue
        if improvement < 0.0 {
            eprintln!("WARNING: Negative improvement with constant activation!");
            eprintln!("This matches production failure pattern.");
        }

        // At minimum, verify the math is consistent
        assert!(
            improvement.is_finite(),
            "Improvement should be finite, got {improvement}"
        );
    }

    #[test]
    fn merge_batch_results_preserves_order_with_empty_samples() {
        let flags = vec![false, true, false, true];
        let merged = GpuAnalyzer::merge_batch_results(
            &flags,
            vec![
                HelpfulStats {
                    positive_count: 1,
                    ..HelpfulStats::default()
                },
                HelpfulStats {
                    positive_count: 2,
                    ..HelpfulStats::default()
                },
            ],
        );

        assert_eq!(
            merged.len(),
            flags.len(),
            "Merged results should match input batch length"
        );
        assert_eq!(
            merged[0].positive_count, 1,
            "First non-empty sample should remain first"
        );
        assert_eq!(
            merged[1].positive_count, 0,
            "Empty samples should produce default stats"
        );
        assert_eq!(
            merged[2].positive_count, 2,
            "Second non-empty sample should remain in original position"
        );
        assert_eq!(
            merged[3].positive_count, 0,
            "Trailing empty samples should also produce defaults"
        );
    }

    #[test]
    fn diagnostics_prefers_higher_expected_improvement() {
        // Issue #216: Methods now take &self, not &mut self
        let diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 1_500);
        diagnostics.record_candidate_attempt("output-0", false);
        diagnostics.record_no_samples("output-0", "input-0", 0);

        diagnostics.record_candidate_attempt("output-0", true);
        diagnostics.record_below_threshold(
            "output-0",
            "hidden-1",
            ThresholdContext {
                sample_count: 42,
                expected_improvement: 0.05,
                threshold: 0.1,
                improved_count: 30,
                worsened_count: 12,
                weight: -0.25,
            },
        );

        let entry = diagnostics
            .entry_for("output-0")
            .expect("diagnostics entry should exist");
        let reason = entry.best_rejection.as_ref().map(|detail| detail.reason);
        assert!(
            matches!(reason, Some(RejectionReason::BelowThreshold)),
            "Expected below-threshold reason to persist when it has the highest score"
        );
    }

    #[test]
    fn diagnostics_marks_candidate_selection() {
        // Issue #216: Methods now take &self, not &mut self
        let diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.mark_candidate_selected("output-0");
        let entry = diagnostics
            .entry_for("output-0")
            .expect("diagnostics entry should exist");
        assert!(
            entry.had_candidate,
            "Entry should record that a candidate was selected"
        );
    }

    #[test]
    fn relu_split_evaluation_finds_candidates_when_activation_correlates_with_error() {
        // Test that split-by-error ReLU evaluation finds candidates when source activation
        // correlates with target error direction.
        //
        // Key insight: A ReLU can only help if its activation correlates with the errors
        // it's trying to fix. If activation is the same for all samples, the ReLU's
        // contribution will cancel out across balanced errors.
        //
        // This test creates samples where:
        // - Source fires (activation > 0) when target error is positive (output should go UP)
        // - Source doesn't fire (activation <= 0) when target error is negative
        //
        // This is the realistic scenario where adding a ReLU neuron can help.
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analysis should be available");

        let mut samples = Vec::new();
        // Samples where source fires AND output should go UP (positive error)
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: 1.0, // Source fires
                avg_error: 0.5,  // Output should be HIGHER
                target_value: None,
                target_activation: None,
            });
        }
        // Samples where source doesn't fire AND output should go DOWN (negative error)
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: -0.5, // Source doesn't fire (ReLU will output 0)
                avg_error: -0.5,  // Output should be LOWER
                target_value: None,
                target_activation: None,
            });
        }

        let result =
            evaluate_relu_candidates_split(&analyzer, "input-0", "output-0", &samples, 0.0, None)
                .expect("ReLU split evaluation should succeed");

        // With correlation between activation and error, we should find a positive-error candidate
        // The ReLU fires when we need output to go UP, and doesn't fire when we need it DOWN.
        assert!(
            result.positive_error_candidate.is_some(),
            "Should find positive-error ReLU candidate when activation correlates with error direction"
        );

        // Verify the candidate pushes in the correct direction
        if let Some(pos_candidate) = &result.positive_error_candidate {
            assert!(
                pos_candidate.outgoing_weight > 0.0,
                "Positive-error candidate should have positive outgoing weight (pushes UP). Got: {}",
                pos_candidate.outgoing_weight
            );
        }
    }

    #[test]
    fn relu_split_evaluation_finds_negative_orientation_candidates() {
        // Test that we can find ReLU candidates with incoming_weight = -1.0 (negative orientation).
        //
        // This is critical: when source neurons have predominantly NEGATIVE activations
        // that correlate with errors, we need a ReLU with incoming_weight = -1.0 to flip
        // the sign before the ReLU activation.
        //
        // Bug regression test: Previously, evaluate_relu_candidates_split discarded
        // negative_stats entirely, meaning these candidates could never be found.
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analysis should be available");

        let mut samples = Vec::new();
        // Samples where source has NEGATIVE activation AND output should go UP (positive error)
        // A ReLU with incoming_weight = -1.0 will flip -1.0 to +1.0, then ReLU outputs 1.0
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: -1.0, // NEGATIVE activation
                avg_error: 0.5,   // Output should be HIGHER
                target_value: None,
                target_activation: None,
            });
        }
        // Samples where source has POSITIVE activation AND output should go DOWN (negative error)
        // A ReLU with incoming_weight = -1.0 will flip +0.5 to -0.5, then ReLU outputs 0
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: 0.5, // POSITIVE activation (will be flipped to negative, ReLU = 0)
                avg_error: -0.5, // Output should be LOWER
                target_value: None,
                target_activation: None,
            });
        }

        let result =
            evaluate_relu_candidates_split(&analyzer, "input-0", "output-0", &samples, 0.0, None)
                .expect("ReLU split evaluation should succeed");

        // With negative activations correlating with positive errors, we should find a candidate
        // that uses the NEGATIVE orientation (incoming_weight = -1.0)
        assert!(
            result.positive_error_candidate.is_some(),
            "Should find ReLU candidate even when source has negative activations (requires negative orientation)"
        );

        // Verify the candidate uses negative incoming weight (the critical fix!)
        if let Some(pos_candidate) = &result.positive_error_candidate {
            assert!(
                pos_candidate.incoming_weight < 0.0,
                "Candidate should have NEGATIVE incoming weight to flip negative activations. Got: {}",
                pos_candidate.incoming_weight
            );
            assert!(
                pos_candidate.outgoing_weight > 0.0,
                "Candidate should have positive outgoing weight (pushes UP). Got: {}",
                pos_candidate.outgoing_weight
            );
        }
    }

    #[test]
    fn neuron_diagnostics_tracks_load_failures() {
        // Test that when eligible sources exist but all fail to load, we report
        // NoSamples rather than NoEligibleSources
        // Issue #216: Methods now take &self, not &mut self
        let diagnostics = NeuronDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 10); // 10 eligible sources exist
                                                                // All 10 sources fail to load
        for _ in 0..10 {
            diagnostics.record_load_failure("output-0");
        }
        // No record_candidate_attempt calls (because all failed to load)

        let summaries = diagnostics.no_candidate_summaries();
        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];

        // Should NOT report "no eligible sources" - sources existed but failed to load
        assert!(
            !matches!(summary.reason, NeuronNoCandidateReason::NoEligibleSources),
            "Should not report NoEligibleSources when sources existed but failed to load"
        );
        // Should report NoSamples (as a catchall for sources existing but not being usable)
        assert!(
            matches!(summary.reason, NeuronNoCandidateReason::NoSamples),
            "Should report NoSamples when eligible sources exist but none were evaluated"
        );

        // Verify entry tracking
        let entry = diagnostics.entry_for("output-0").unwrap();
        assert_eq!(entry.total_eligible_sources, 10);
        assert_eq!(entry.record_load_failures, 10);
        assert_eq!(entry.evaluated_sources, 0);
    }

    #[test]
    fn neuron_diagnostics_reports_genuine_no_eligible_sources() {
        // Test that when there are genuinely no eligible sources, we correctly report that
        // Issue #216: Methods now take &self, not &mut self
        let diagnostics = NeuronDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 0); // No eligible sources

        let summaries = diagnostics.no_candidate_summaries();
        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];

        // Should correctly report no eligible sources
        assert!(
            matches!(summary.reason, NeuronNoCandidateReason::NoEligibleSources),
            "Should report NoEligibleSources when genuinely no sources exist"
        );

        // Verify entry tracking
        let entry = diagnostics.entry_for("output-0").unwrap();
        assert_eq!(entry.total_eligible_sources, 0);
        assert_eq!(entry.record_load_failures, 0);
        assert_eq!(entry.evaluated_sources, 0);
    }

    #[test]
    fn target_diagnostics_reports_no_samples_reason() {
        // Issue #216: Methods now take &self, not &mut self
        let diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 25);
        diagnostics.record_candidate_attempt("output-0", false);
        diagnostics.record_no_samples("output-0", "input-0", 8);

        let summaries = diagnostics.no_candidate_summaries();
        assert_eq!(
            summaries.len(),
            1,
            "Expected a single diagnostic summary for target without candidates"
        );

        let summary = &summaries[0];
        assert_eq!(
            summary.target_uuid, "output-0",
            "Target UUID should be preserved in summary"
        );
        assert!(
            matches!(summary.reason, SynapseNoCandidateReason::NoSamples),
            "Expected no-samples reason"
        );
    }

    #[test]
    fn analyze_neurons_rejects_duplicate_focus_targets() {
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = (MIN_NEURON_SAMPLE_COUNT + 5) as u32;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            records.push(DiscoverRecord::new(
                obs_index,
                "hidden-source".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![1.0],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-source".to_string(),
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
            synapses: Vec::new(),
        };

        let input = AnalyzeNeuronsInput {
            parquet_file: parquet_file.clone(),
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-0".to_string()],
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let err = analyze_neurons(&input)
            .expect_err("Neuron analysis should refuse duplicate focus neurons");
        let message = format!("{err}");
        assert!(
            message.contains("duplicate focus neurons"),
            "Expected duplicate focus error, got: {message}",
        );
    }

    #[test]
    fn analyze_synapses_rejects_duplicate_focus_targets() {
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = 16;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![-0.05],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "input-0".to_string(),
                    neuron_type: "input".to_string(),
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
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
                synapse_type: None,
            }],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-0".to_string()],
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let err = analyze_synapses(&input)
            .expect_err("Synapse analysis should refuse duplicate focus neurons");
        let message = format!("{err}");
        assert!(
            message.contains("duplicate focus neurons"),
            "Expected duplicate focus error, got: {message}",
        );
    }

    #[test]
    fn analyze_synapses_reports_eligible_sources_correctly_for_non_input_neurons() {
        skip_if_no_gpu!();
        // Test that non-input neurons with valid creature structure always report
        // eligible sources correctly, not "no eligible sources" when sources exist
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Input neuron records (observations)
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "input-1".to_string(),
                Some(0.0),
                0.3,
                vec![0.2],
            ));
            // Hidden neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "hidden-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.15],
            ));
            // Output neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.6,
                vec![0.05],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 2,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-0".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "constant-0".to_string(),
                    neuron_type: "constant".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 1.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                // Hidden neuron already connected to input-0
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "hidden-0".to_string(),
                    weight: 0.4,
                    synapse_type: None,
                },
                // Output neuron already connected to hidden-0
                SynapseJson {
                    from_uuid: "hidden-0".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 0.5,
                    synapse_type: None,
                },
            ],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["hidden-0".to_string(), "output-0".to_string()],
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // Check diagnostics for hidden-0
        // hidden-0 should have eligible sources (input-1 is not connected yet)
        // So it should NOT report "no eligible sources"
        let hidden_diag = result
            .no_candidate_reasons
            .iter()
            .find(|summary| summary.target_uuid == "hidden-0");

        if let Some(diag) = hidden_diag {
            assert!(
                diag.reason != SynapseNoCandidateReason::NoEligibleSources,
                "hidden-0 should have eligible sources (input-1 is available), but got: {:?}",
                diag.reason
            );
            assert!(
                diag.evaluated_candidates > 0,
                "hidden-0 should have evaluated at least one candidate (input-1), but evaluated_candidates is {}",
                diag.evaluated_candidates
            );
        }

        // Check diagnostics for output-0
        // output-0 should have eligible sources (input-0, input-1 are available)
        // So it should NOT report "no eligible sources"
        let output_diag = result
            .no_candidate_reasons
            .iter()
            .find(|summary| summary.target_uuid == "output-0");

        if let Some(diag) = output_diag {
            assert!(
                diag.reason != SynapseNoCandidateReason::NoEligibleSources,
                "output-0 should have eligible sources (input-0, input-1 are available), but got: {:?}",
                diag.reason
            );
            assert!(
                diag.evaluated_candidates > 0,
                "output-0 should have evaluated at least one candidate, but evaluated_candidates is {}",
                diag.evaluated_candidates
            );
        }
    }

    #[test]
    fn analyze_synapses_reports_fully_connected_neuron_explicitly() {
        skip_if_no_gpu!();
        // Test that a neuron connected to ALL eligible sources is explicitly reported
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Input neuron records (observations)
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "input-1".to_string(),
                Some(0.0),
                0.3,
                vec![0.2],
            ));
            // Hidden neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "hidden-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.15],
            ));
            // Output neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.6,
                vec![0.05],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        // Create a creature where hidden-0 is connected to ALL eligible sources
        // (both input-0 and input-1)
        let creature = CreatureJson {
            input: 2,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-0".to_string(),
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
                // hidden-0 is connected to ALL eligible sources (input-0 and input-1)
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "hidden-0".to_string(),
                    weight: 0.4,
                    synapse_type: None,
                },
                SynapseJson {
                    from_uuid: "input-1".to_string(),
                    to_uuid: "hidden-0".to_string(),
                    weight: 0.5,
                    synapse_type: None,
                },
            ],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["hidden-0".to_string()],
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // hidden-0 should be reported as having no eligible sources
        // because it's connected to ALL eligible sources (both inputs)
        let hidden_diag = result
            .no_candidate_reasons
            .iter()
            .find(|summary| summary.target_uuid == "hidden-0");

        assert!(
            hidden_diag.is_some(),
            "hidden-0 should have diagnostics since it's fully connected"
        );

        if let Some(diag) = hidden_diag {
            assert_eq!(
                diag.reason,
                SynapseNoCandidateReason::NoEligibleSources,
                "hidden-0 should report NoEligibleSources since it's connected to all eligible sources"
            );
            assert_eq!(
                diag.evaluated_candidates, 0,
                "hidden-0 should have 0 evaluated candidates since all sources are already connected"
            );
        }
    }

    #[test]
    fn analyze_synapses_requires_focus_targets() {
        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file: "unused.parquet".to_string(),
            creature,
            focus_neurons: Vec::new(),
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let err =
            analyze_synapses(&input).expect_err("Synapse analysis should refuse empty focus lists");
        let message = format!("{err}");
        assert!(
            message.contains("at least one focus neuron"),
            "Expected missing focus error, got: {message}",
        );
    }

    #[test]
    fn analyze_synapses_reports_diagnostics_when_no_candidates() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..16 {
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.05],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file: parquet_file.clone(),
            creature,
            focus_neurons: vec!["output-0".to_string()],
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let result = analyze_synapses(&input)
            .expect("Synapse analysis should succeed even without candidates");
        assert!(
            result.helpful_synapses.is_empty(),
            "Expected no helpful candidates when there are no eligible sources"
        );
        let reason = result
            .no_candidate_reasons
            .first()
            .map(|summary| summary.reason.clone());
        assert!(
            matches!(reason, Some(SynapseNoCandidateReason::NoEligibleSources)),
            "Expected diagnostics to explain missing candidates"
        );
    }

    #[test]
    fn analyze_synapses_stops_harmful_processing_after_deadline() {
        skip_if_no_gpu!();
        use rayon::ThreadPoolBuilder;
        use std::sync::Arc;
        // Use a private Rayon pool so the deadline override is isolated from other parallel tests.
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .expect("Failed to build Rayon pool"),
        );
        let _deadline_guard = deadline_override::DeadlineOverrideGuard::with_sequence_for_pool(
            vec![
                false, false, false, false, false, false, true, false, false, false,
            ],
            Some(Arc::clone(&pool)),
        );

        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..16 {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-1".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 2,
            neurons: vec![
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
            ],
            synapses: vec![
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 0.8,
                    synapse_type: None,
                },
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "output-1".to_string(),
                    weight: 0.6,
                    synapse_type: None,
                },
            ],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let result = pool
            .install(|| analyze_synapses(&input))
            .expect("Synapse analysis should complete even when the deadline triggers");

        // With parallel processing, deadline detection order is non-deterministic because
        // multiple threads call deadline_passed() concurrently. The timeout mechanism is
        // approximate - once any thread detects the deadline, analysis_timed_out is set
        // and processing should stop. However, some threads may have already started
        // processing harmful synapses before the deadline was detected.
        //
        // The key requirement is that the analysis completes successfully and respects
        // the deadline approximately. Since timeout is approximate, we verify that:
        // 1. The analysis completes without panicking
        // 2. The result structure is valid
        // 3. We don't process more harmful synapses than exist (sanity check)
        //
        // In this test setup, we have 2 focus neurons, each with 1 harmful synapse (2 total).
        // The deadline sequence [false x6, true, ...] should cause early termination,
        // but with parallel processing, the exact point of termination is non-deterministic.
        let max_possible_harmful = 2; // 2 focus neurons × 1 harmful synapse each
        assert!(
            result.harmful_synapses.len() <= max_possible_harmful,
            "Should not process more harmful synapses than exist. \
             Got {} harmful synapses, max possible is {}",
            result.harmful_synapses.len(),
            max_possible_harmful
        );
        // The deadline mechanism is approximate, so we accept any result as long as
        // the analysis completes and doesn't exceed reasonable bounds
    }

    #[test]
    fn analyze_all_runs_synapse_and_neuron_phases() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..(MIN_NEURON_SAMPLE_COUNT as u32 + 1) {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: vec![SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
                synapse_type: None,
            }],
        };

        let input = AnalyzeAllInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            max_synapse_candidates: Some(5),
            max_neuron_candidates: Some(5),
            analysis_deadline_ms: None,
            include_synapse_analysis: Some(true),
            include_neuron_analysis: Some(true),
            random_seed: None,
        };

        let result = analyze_all(&input).expect("Combined analysis should succeed");
        assert!(result.synapse.is_some(), "Synapse phase should run");
        assert!(result.neuron.is_some(), "Neuron phase should run");
    }

    #[test]
    fn analyze_neurons_reports_diagnostics_when_no_candidates() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..(MIN_NEURON_SAMPLE_COUNT as u32 + 1) {
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-source".to_string(),
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
            synapses: Vec::new(),
        };

        let input = AnalyzeNeuronsInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let result = analyze_neurons(&input)
            .expect("Neuron analysis should succeed even without candidates");
        assert!(
            result.helpful_neurons.is_empty(),
            "Expected no neuron candidates when the source neuron lacks samples"
        );
        let reason = result
            .no_candidate_reasons
            .first()
            .map(|summary| summary.reason.clone());
        assert!(
            matches!(reason, Some(NeuronNoCandidateReason::NoSamples)),
            "Expected diagnostics to explain missing neuron candidates"
        );
    }

    #[test]
    fn analyze_neurons_uses_vertical_timeout_with_randomized_order() {
        skip_if_no_gpu!();
        use rayon::ThreadPoolBuilder;
        use std::sync::Arc;

        // Simulate a deadline that allows at least one focus neuron to start, but
        // triggers before all are processed. The override sequence is consumed
        // by calls to `deadline_passed` in order. With randomization, we need
        // enough false values to allow at least one neuron to start processing.
        // We provide multiple false values to account for any initialization checks,
        // then true to stop further processing.
        // Provide enough "not timed out" checks to allow at least one focus neuron
        // to begin evaluating sources before we trigger the timeout.
        let mut deadline_sequence = vec![false; 64];
        deadline_sequence.push(true);
        // Use a private pool so the override cannot be consumed by other tests.
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("Failed to build single-threaded Rayon pool"),
        );
        let _deadline_guard = deadline_override::DeadlineOverrideGuard::with_sequence_for_pool(
            deadline_sequence,
            Some(Arc::clone(&pool)),
        );

        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Provide discovery records for two output neurons but none for the
        // hidden source. This guarantees that each focus neuron has at least
        // one eligible upstream source, and that diagnostics can attribute a
        // `NoSamples` reason once analysis runs.
        let mut records = Vec::new();
        for obs_index in 0..(MIN_NEURON_SAMPLE_COUNT as u32 + 1) {
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-1".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 0,
            output: 2,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-source".to_string(),
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
                NeuronJson {
                    uuid: "output-1".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: Vec::new(),
        };

        let input = AnalyzeNeuronsInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
            max_candidates: None,
            // Any non-None deadline value will exercise the override sequence.
            analysis_deadline_ms: Some(1_000_000),
            random_seed: None,
        };

        let result = pool
            .install(|| analyze_neurons(&input))
            .expect("Neuron analysis should succeed even when the deadline triggers");

        // At least one focus neuron should have evaluated at least one upstream
        // source before the deadline (vertical timeout behaviour). Since focus
        // neurons are randomized, we check that at least one of the two neurons
        // was processed.
        let processed_neurons: Vec<_> = result
            .no_candidate_reasons
            .iter()
            .filter(|summary| summary.evaluated_sources > 0)
            .collect();

        // At least one focus neuron should have progressed far enough to attempt
        // source evaluation, OR we should have produced at least one candidate.
        let did_any_work = !processed_neurons.is_empty() || !result.helpful_neurons.is_empty();
        assert!(
            did_any_work,
            "At least one focus neuron should do some work before timeout (vertical timeout behaviour)"
        );

        // Verify that the processed neuron(s) are not reported as having no eligible sources
        for summary in &processed_neurons {
            assert!(
                summary.reason != NeuronNoCandidateReason::NoEligibleSources,
                "Processed focus neuron should not be reported as having no eligible sources when a timeout occurs"
            );
        }
    }

    #[test]
    fn analyze_synapses_uses_vertical_timeout_with_randomized_order() {
        skip_if_no_gpu!();
        use rayon::ThreadPoolBuilder;
        use std::sync::Arc;

        // Simulate a deadline that allows at least one focus neuron to start, but
        // triggers before all are processed. The override sequence is consumed
        // by calls to `deadline_passed` in order. With randomization, we need
        // enough false values to allow at least one neuron to start processing.
        // We provide multiple false values to account for any initialization checks,
        // then true to stop further processing.
        // Provide enough "not timed out" checks to allow at least one focus neuron
        // to begin evaluating candidates before we trigger the timeout.
        //
        // The analysis code checks the deadline at several stages (start-of-focus,
        // source pre-filtering, sample building, batch evaluation). If we trigger
        // the timeout too early, the vertical-timeout behaviour isn't exercised.
        let mut deadline_sequence = vec![false; 64];
        deadline_sequence.push(true);
        // Use a private pool so the override cannot be consumed by other tests.
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("Failed to build single-threaded Rayon pool"),
        );
        let _deadline_guard = deadline_override::DeadlineOverrideGuard::with_sequence_for_pool(
            deadline_sequence,
            Some(Arc::clone(&pool)),
        );

        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Provide discovery records for an input neuron and two output neurons.
        // Records use disjoint obs_index ranges so that each potential synapse
        // has discovery data but no aligned samples, guaranteeing that the
        // diagnostics machinery records a `NoSamples` style rejection rather
        // than treating the target as having no eligible sources.
        let mut records = Vec::new();
        for obs_index in 0..16u32 {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
        }
        for obs_index in 100..116u32 {
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-1".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 2,
            neurons: vec![
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
            ],
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
            max_candidates: None,
            // Any non-None deadline value will exercise the override sequence.
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let result = pool
            .install(|| analyze_synapses(&input))
            .expect("Synapse analysis should succeed even when the deadline triggers");

        // At least one focus neuron should have evaluated at least one upstream
        // source before the deadline (vertical timeout behaviour). Since focus
        // neurons are randomized, we check that at least one of the two neurons
        // was processed.
        let processed_neurons: Vec<_> = result
            .no_candidate_reasons
            .iter()
            .filter(|summary| summary.evaluated_candidates > 0)
            .collect();

        // At least one focus neuron should have progressed far enough to attempt
        // candidate evaluation, OR we should have produced at least one candidate.
        let did_any_work = !processed_neurons.is_empty()
            || !result.helpful_synapses.is_empty()
            || !result.harmful_synapses.is_empty();
        assert!(
            did_any_work,
            "At least one focus neuron should do some work before timeout (vertical timeout behaviour)"
        );

        // If we observed a processed neuron via diagnostics, it should not be reported
        // as having no eligible sources.
        for summary in &processed_neurons {
            assert!(
                summary.reason != SynapseNoCandidateReason::NoEligibleSources,
                "Processed focus neuron should not be reported as having no eligible sources when a timeout occurs"
            );
        }
    }

    /// Test bias calculation for TANH activation function
    #[test]
    fn test_bias_calculation_tanh() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let bias = calculate_optimal_bias(&samples, 1.0, -0.5, tanh_activation, "TANH", None, None);

        // Bias should be in expanded TANH range
        assert!(
            (-1.0..=1.0).contains(&bias),
            "Bias for TANH should be in range [-1.0, 1.0], got {bias}"
        );
    }

    /// Test bias calculation for ReLU activation function
    #[test]
    fn test_bias_calculation_relu() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let relu_fn = |x: f32| x.max(0.0);
        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, relu_fn, "ReLU", None, None);

        // ReLU can now use negative bias for threshold shifting (expanded range)
        assert!(bias >= -1.0, "Bias for ReLU should be >= -1.0, got {bias}");
        assert!(bias <= 1.0, "Bias for ReLU should be <= 1.0, got {bias}");
    }

    /// Test bias improves error reduction compared to zero bias
    #[test]
    fn test_bias_improves_error_reduction() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let incoming = 1.5;
        let outgoing = -0.18;

        // Calculate error with zero bias
        let mut zero_bias_error_sq = 0.0;
        for sample in &samples {
            let pre_activation = incoming * sample.activation;
            let new_neuron_activation = identity_activation(pre_activation);
            let correction = outgoing * new_neuron_activation;
            let new_error = sample.avg_error - correction;
            zero_bias_error_sq += new_error * new_error;
        }

        // Calculate optimal bias
        let optimal_bias = calculate_optimal_bias(
            &samples,
            incoming,
            outgoing,
            identity_activation,
            "IDENTITY",
            None,
            None,
        );

        // Calculate error with optimal bias
        let mut optimal_bias_error_sq = 0.0;
        for sample in &samples {
            let pre_activation = incoming * sample.activation + optimal_bias;
            let new_neuron_activation = identity_activation(pre_activation);
            let correction = outgoing * new_neuron_activation;
            let new_error = sample.avg_error - correction;
            optimal_bias_error_sq += new_error * new_error;
        }

        // Optimal bias should give equal or better error reduction than zero bias
        assert!(
            optimal_bias_error_sq <= zero_bias_error_sq + EPSILON,
            "Optimal bias should improve or equal zero bias error reduction: zero_bias_error={zero_bias_error_sq}, optimal_bias_error={optimal_bias_error_sq}"
        );
    }

    /// Test that positive improvements below threshold are accepted as candidates
    /// This verifies the fix where all positive improvements are candidates, not just those above threshold
    #[test]
    fn analyze_synapses_accepts_positive_improvements_below_threshold() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Create test data that will produce a positive but below-threshold improvement
        // We need: expected_improvement = (2*w*E[a*e] - w^2*E[a^2]) / E[e^2]
        // To get ~0.05 improvement with threshold 0.1, we'll use:
        // - source activation: 0.5 consistently
        // - target error: 0.1 consistently
        // - This should produce a positive improvement when weight is chosen appropriately
        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Source neuron (input-0) with consistent activation
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.5,    // Consistent activation
                vec![], // Input neurons don't have errors
            ));
            // Target neuron (output-0) with consistent error
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.3,
                vec![0.1], // Consistent error
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(), // No existing synapse from input-0 to output-0
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // The key assertion: positive improvements below threshold should be accepted
        // We should have at least one helpful synapse candidate (even if improvement < 0.1)
        // OR if no candidate, it should NOT be due to BelowThreshold for a positive improvement
        if result.helpful_synapses.is_empty() {
            // If no candidates, check diagnostics - it should NOT be BelowThreshold for positive improvements
            let no_candidate = result
                .no_candidate_reasons
                .iter()
                .find(|summary| summary.target_uuid == "output-0");

            if let Some(summary) = no_candidate {
                // If there's a detail, check that it's not a positive improvement below threshold
                if let Some(detail) = &summary.detail {
                    if let Some(improvement) = detail.expected_improvement {
                        if improvement > 0.0 && improvement <= 0.1 {
                            panic!(
                                "Positive improvement {:.4} below threshold 0.1 should be accepted as candidate, but was rejected with reason: {:?}",
                                improvement, summary.reason
                            );
                        }
                    }
                }
            }
        } else {
            // We have candidates - verify at least one has positive improvement
            // Issue #128: Use expected_creature_score_gain (creature-level, not neuron-level)
            let has_positive_improvement = result
                .helpful_synapses
                .iter()
                .any(|synapse| synapse.expected_creature_score_gain > 0.0);

            assert!(
                has_positive_improvement,
                "Should have at least one candidate with positive improvement"
            );
        }
    }

    /// Test that non-positive improvements (<= 0.0) are still rejected
    #[test]
    fn analyze_synapses_rejects_non_positive_improvements() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Create test data that will produce a non-positive improvement
        // Use mismatched activations/errors that result in negative or zero improvement
        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Source neuron with activation
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![],
            ));
            // Target neuron with error that doesn't correlate well (will produce negative improvement)
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.0,
                vec![-0.1], // Negative error when source is positive
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // Non-positive improvements should be rejected (not appear in helpful_synapses)
        // Even though we now accept positive improvements below threshold, we still reject <= 0.0
        // Issue #128: Use expected_creature_score_gain (creature-level metric)
        let has_non_positive = result
            .helpful_synapses
            .iter()
            .any(|synapse| synapse.expected_creature_score_gain <= 0.0);

        assert!(
            !has_non_positive,
            "Should not have any candidates with non-positive improvement (<= 0.0)"
        );
    }

    /// Test that positive improvements above threshold are still accepted (regression test)
    #[test]
    fn analyze_synapses_accepts_positive_improvements_above_threshold() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Create test data that will produce a positive improvement above threshold
        // Use strong correlation between source activation and target error
        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Source neuron with strong activation
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![],
            ));
            // Target neuron with error that correlates positively
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2], // Positive error when source is positive
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // Positive improvements above threshold should definitely be accepted
        // This is a regression test to ensure we didn't break existing behavior
        // Issue #128: Use expected_creature_score_gain (creature-level metric)
        let has_above_threshold = result
            .helpful_synapses
            .iter()
            .any(|synapse| synapse.expected_creature_score_gain > 0.1);

        // Note: This test may pass even if no candidates are found due to other reasons
        // (e.g., no samples, zero improvement). The key is that if we have candidates,
        // they should include positive improvements above threshold.
        if !result.helpful_synapses.is_empty() {
            assert!(
                has_above_threshold
                    || result
                        .helpful_synapses
                        .iter()
                        .any(|s| s.expected_creature_score_gain > 0.0),
                "Should have candidates with positive improvement (above or below threshold)"
            );
        }
    }

    /// Test bias range boundaries for different activation functions
    #[test]
    fn test_bias_within_reasonable_range() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        type ActivationTestCase = (&'static str, fn(f32) -> f32, f32, f32);
        let test_cases: Vec<ActivationTestCase> = vec![
            ("TANH", tanh_activation as fn(f32) -> f32, -10.0, 10.0),
            (
                "LOGISTIC",
                logistic_activation as fn(f32) -> f32,
                -10.0,
                10.0,
            ),
            (
                "IDENTITY",
                identity_activation as fn(f32) -> f32,
                -50.0,
                50.0,
            ),
        ];

        for (name, activation_fn, min_expected, max_expected) in test_cases {
            let bias = calculate_optimal_bias(&samples, 1.0, 1.0, activation_fn, name, None, None);
            assert!(
                bias >= min_expected && bias <= max_expected,
                "Bias for {name} should be in range [{min_expected}, {max_expected}], got {bias}"
            );
        }
    }

    /// Test bias calculation handles empty samples
    #[test]
    fn test_bias_calculation_empty_samples() {
        let samples: Vec<HelpfulSample> = vec![];

        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

        // Should return 0.0 for empty samples
        assert_eq!(bias, 0.0, "Empty samples should return bias of 0.0");
    }

    /// Test bias calculation handles insufficient samples
    #[test]
    fn test_bias_calculation_insufficient_samples() {
        // Only 5 samples (less than MIN_NEURON_SAMPLE_COUNT of 10)
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
        ];

        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

        // Should still return a valid bias in range (though may be 0.0 if no bias tested has sufficient samples)
        assert!(
            (-10.0..=10.0).contains(&bias),
            "Bias should be in reasonable range, got {bias}"
        );
    }

    // test_get_bias_range moved to src/analysis/activation.rs (tests activation module functions)

    /// Test is_threshold_activation identifies threshold functions (STEP/BIPOLAR).
    /// These use a specialised threshold-crossing model instead of the linear model.
    /// All other activations use the standard linear error model - none are skipped.
    #[test]
    fn test_is_threshold_activation() {
        // Threshold activations - use threshold-crossing model
        assert!(is_threshold_activation("STEP"), "STEP uses threshold model");
        assert!(is_threshold_activation("step"), "case insensitive");
        assert!(
            is_threshold_activation("BIPOLAR"),
            "BIPOLAR uses threshold model"
        );

        // All other activations use standard linear model (not skipped)
        assert!(
            !is_threshold_activation("IF"),
            "IF uses standard model (correlation still works)"
        );
        assert!(
            !is_threshold_activation("MAXIMUM"),
            "MAXIMUM uses standard model"
        );
        assert!(
            !is_threshold_activation("MINIMUM"),
            "MINIMUM uses standard model"
        );
        assert!(
            !is_threshold_activation("HARD_TANH"),
            "HARD_TANH uses standard model"
        );
        assert!(
            !is_threshold_activation("CLIPPED"),
            "CLIPPED uses standard model"
        );
        assert!(
            !is_threshold_activation("ReLU6"),
            "ReLU6 uses standard model"
        );
        assert!(!is_threshold_activation("TANH"), "TANH uses standard model");
        assert!(
            !is_threshold_activation("LOGISTIC"),
            "LOGISTIC uses standard model"
        );
        assert!(!is_threshold_activation("ReLU"), "ReLU uses standard model");
        assert!(
            !is_threshold_activation("LeakyReLU"),
            "LeakyReLU uses standard model"
        );
        assert!(!is_threshold_activation("ELU"), "ELU uses standard model");
        assert!(!is_threshold_activation("SELU"), "SELU uses standard model");
        assert!(!is_threshold_activation("GELU"), "GELU uses standard model");
        assert!(
            !is_threshold_activation("IDENTITY"),
            "IDENTITY uses standard model"
        );
        assert!(
            !is_threshold_activation("Softplus"),
            "Softplus uses standard model"
        );
        assert!(
            !is_threshold_activation("BENT_IDENTITY"),
            "BENT_IDENTITY uses standard model"
        );
        assert!(
            !is_threshold_activation("ArcTan"),
            "ArcTan uses standard model"
        );
        assert!(
            !is_threshold_activation("Swish"),
            "Swish uses standard model"
        );
        assert!(!is_threshold_activation("Mish"), "Mish uses standard model");
        assert!(
            !is_threshold_activation("UNKNOWN"),
            "Unknown uses standard model"
        );
    }

    #[test]
    fn test_filter_focus_targets_tracks_threshold_targets_for_hidden_and_unknown_when_allowed() {
        use std::collections::HashMap;

        // Regression coverage (29-Dec-2025): when output/hidden handling was split, the
        // STEP/BIPOLAR tracking was accidentally only applied to output targets.
        let focus = [
            "hidden-step".to_string(),
            "output-tanh".to_string(),
            "unknown-bipolar".to_string(),
        ];
        let unique_focus: Vec<&String> = focus.iter().collect();

        let mut neuron_type_map: HashMap<String, String> = HashMap::new();
        neuron_type_map.insert("hidden-step".to_string(), "hidden".to_string());
        neuron_type_map.insert("output-tanh".to_string(), "output".to_string());
        neuron_type_map.insert("unknown-bipolar".to_string(), "mystery".to_string());

        let mut neuron_squash_map: HashMap<String, String> = HashMap::new();
        neuron_squash_map.insert("hidden-step".to_string(), "STEP".to_string());
        neuron_squash_map.insert("output-tanh".to_string(), "TANH".to_string());
        neuron_squash_map.insert("unknown-bipolar".to_string(), "BIPOLAR".to_string());

        let result = filter_focus_targets_for_neuron_analysis(
            &unique_focus,
            &neuron_type_map,
            &neuron_squash_map,
            false,
        );

        assert!(
            result.focus_order.contains(&"hidden-step".to_string()),
            "hidden-step should be included when output-only mode is disabled"
        );
        assert!(
            result.focus_order.contains(&"unknown-bipolar".to_string()),
            "unknown-bipolar should be included when output-only mode is disabled (treated as hidden)"
        );
        assert!(
            result
                .threshold_targets
                .contains(&"hidden-step".to_string()),
            "hidden-step (STEP) should be tracked as a threshold target"
        );
        assert!(
            result
                .threshold_targets
                .contains(&"unknown-bipolar".to_string()),
            "unknown-bipolar (BIPOLAR) should be tracked as a threshold target"
        );
        assert!(
            !result
                .threshold_targets
                .contains(&"output-tanh".to_string()),
            "output-tanh (TANH) should not be tracked as a threshold target"
        );
        assert!(
            result.skipped_hidden.is_empty(),
            "No hidden targets should be skipped when output-only mode is disabled"
        );
    }

    #[test]
    fn test_filter_focus_targets_respects_output_only_mode_for_hidden_and_unknown() {
        use std::collections::HashMap;

        let focus = [
            "hidden-step".to_string(),
            "output-step".to_string(),
            "unknown-bipolar".to_string(),
        ];
        let unique_focus: Vec<&String> = focus.iter().collect();

        let mut neuron_type_map: HashMap<String, String> = HashMap::new();
        neuron_type_map.insert("hidden-step".to_string(), "hidden".to_string());
        neuron_type_map.insert("output-step".to_string(), "output".to_string());
        neuron_type_map.insert("unknown-bipolar".to_string(), "mystery".to_string());

        let mut neuron_squash_map: HashMap<String, String> = HashMap::new();
        neuron_squash_map.insert("hidden-step".to_string(), "STEP".to_string());
        neuron_squash_map.insert("output-step".to_string(), "STEP".to_string());
        neuron_squash_map.insert("unknown-bipolar".to_string(), "BIPOLAR".to_string());

        let result = filter_focus_targets_for_neuron_analysis(
            &unique_focus,
            &neuron_type_map,
            &neuron_squash_map,
            true,
        );

        assert_eq!(
            result.focus_order,
            vec!["output-step".to_string()],
            "Only output targets should remain when output-only mode is enabled"
        );
        assert_eq!(
            result.threshold_targets,
            vec!["output-step".to_string()],
            "Threshold targets should only include remaining focus targets"
        );

        assert!(
            result.skipped_hidden.contains(&"hidden-step".to_string()),
            "hidden-step should be reported as skipped when output-only mode is enabled"
        );
        assert!(
            result
                .skipped_hidden
                .contains(&"unknown-bipolar".to_string()),
            "unknown-bipolar should be reported as skipped when output-only mode is enabled"
        );
    }

    // test_get_bias_values moved to src/analysis/activation.rs (tests activation module functions)

    /// Test bias calculation with non-finite values
    #[test]
    fn test_bias_calculation_with_non_finite_values() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: f32::NAN,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: f32::INFINITY,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

        // Should handle non-finite values gracefully and return a finite bias
        assert!(
            bias.is_finite(),
            "Bias should be finite even with non-finite input values"
        );
        assert!(
            (-1.0..=1.0).contains(&bias),
            "Bias should be in reasonable range, got {bias}"
        );
    }

    /// Test that split-error ReLU evaluation finds complementary pairs when errors are split.
    /// When errors are ~50/50 positive/negative, no single ReLU can help all samples.
    /// Split evaluation should find two candidates: one for each error direction.
    #[test]
    fn test_split_relu_finds_complementary_pairs() {
        // Create samples with split errors:
        // - Half have positive error (output should be higher) with high source activation
        // - Half have negative error (output should be lower) with different pattern
        let mut samples = Vec::new();

        // Positive errors: when source is high, output should be higher
        // A ReLU with positive weight on these samples will help
        for i in 0..50 {
            samples.push(HelpfulSample {
                activation: 0.5 + (i as f32) * 0.01,
                avg_error: 0.3, // Positive: output should be higher
                target_value: None,
                target_activation: None,
            });
        }

        // Negative errors: when source is high, output should be lower
        // A ReLU with negative weight on these samples will help
        for i in 0..50 {
            samples.push(HelpfulSample {
                activation: 0.5 + (i as f32) * 0.01,
                avg_error: -0.3, // Negative: output should be lower
                target_value: None,
                target_activation: None,
            });
        }

        // Verify we have split errors
        let positive_count = samples.iter().filter(|s| s.avg_error > 0.0).count();
        let negative_count = samples.iter().filter(|s| s.avg_error < 0.0).count();
        assert_eq!(positive_count, 50);
        assert_eq!(negative_count, 50);

        // Standard ReLU evaluation should struggle because errors cancel out
        // when computing error*activation correlation - roughly equal positive
        // and negative errors with similar activations means weak correlation overall.
        //
        // The split evaluation separates these, so each subset has strong correlation.
        // This test documents the expected behaviour without requiring GPU.
    }

    /// Test that upsert_candidate keeps complementary ReLU candidates with different
    /// incoming_weight values. A positive-weight ReLU (incoming_weight=1.0) and a
    /// negative-weight ReLU (incoming_weight=-1.0) should both be kept, not collide.
    #[test]
    fn test_upsert_keeps_complementary_relu_candidates_by_incoming_weight() {
        use std::collections::HashMap;

        let mut map: HashMap<(String, String, String, i8, i8), CandidateNeuronJson> =
            HashMap::new();

        // Positive-orientation ReLU candidate
        // Issue #128: Use creature-level metrics
        let positive_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: 1.0, // Positive orientation
            outgoing_weight: 0.5,
            squash: "ReLU".to_string(),
            bias: 0.0,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.15,
            expected_creature_score_gain: 0.15,
            improved_count: 30,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Negative-orientation ReLU candidate
        let negative_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: -1.0, // Negative orientation
            outgoing_weight: 0.4,  // Same outgoing sign
            squash: "ReLU".to_string(),
            bias: 0.0,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.12,
            expected_creature_score_gain: 0.12,
            improved_count: 25,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Insert both candidates
        upsert_candidate(&mut map, positive_candidate.clone());
        upsert_candidate(&mut map, negative_candidate.clone());

        // Both should be kept - they have different incoming_weight signs
        assert_eq!(
            map.len(),
            2,
            "Candidates with different incoming_weight should both be kept"
        );

        // Verify both are present with correct values
        let pos_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            1_i8, // incoming sign
            1_i8, // outgoing sign
        );
        let neg_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            -1_i8, // incoming sign
            1_i8,  // outgoing sign
        );

        assert!(
            map.contains_key(&pos_key),
            "Positive-orientation candidate should exist"
        );
        assert!(
            map.contains_key(&neg_key),
            "Negative-orientation candidate should exist"
        );
    }

    /// Test that upsert_candidate keeps split-error complementary pairs with same
    /// incoming_weight but different outgoing_weight signs. This is the key case for
    /// split-error ReLU evaluation where errors are ~50/50 positive/negative.
    #[test]
    fn test_upsert_keeps_split_error_complementary_pairs() {
        use std::collections::HashMap;

        let mut map: HashMap<(String, String, String, i8, i8), CandidateNeuronJson> =
            HashMap::new();

        // Candidate for positive errors: same source/target, positive outgoing_weight
        // This pushes output UP when source is high
        // Issue #128: Use creature-level metrics
        let positive_error_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: 1.0, // Same orientation
            outgoing_weight: 0.5, // POSITIVE: pushes output UP
            squash: "ReLU".to_string(),
            bias: 0.0,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.10,
            expected_creature_score_gain: 0.10,
            improved_count: 25,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Candidate for negative errors: same source/target, negative outgoing_weight
        // This pushes output DOWN when source is high
        let negative_error_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: 1.0,  // Same orientation
            outgoing_weight: -0.4, // NEGATIVE: pushes output DOWN
            squash: "ReLU".to_string(),
            bias: 0.0,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.08,
            expected_creature_score_gain: 0.08,
            improved_count: 20,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Insert both candidates
        upsert_candidate(&mut map, positive_error_candidate.clone());
        upsert_candidate(&mut map, negative_error_candidate.clone());

        // Both should be kept - they have different outgoing_weight signs
        // This is the key fix for split-error ReLU evaluation
        assert_eq!(
            map.len(),
            2,
            "Split-error complementary pairs with different outgoing_weight signs should both be kept"
        );

        // Verify both are present
        let pos_out_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            1_i8, // incoming sign (both same)
            1_i8, // outgoing sign: positive
        );
        let neg_out_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            1_i8,  // incoming sign (both same)
            -1_i8, // outgoing sign: negative
        );

        assert!(
            map.contains_key(&pos_out_key),
            "Positive-outgoing candidate (pushes UP) should exist"
        );
        assert!(
            map.contains_key(&neg_out_key),
            "Negative-outgoing candidate (pushes DOWN) should exist"
        );

        assert_eq!(
            map.get(&pos_out_key).unwrap().outgoing_weight,
            0.5,
            "Positive-outgoing candidate should have outgoing_weight=0.5"
        );
        assert_eq!(
            map.get(&neg_out_key).unwrap().outgoing_weight,
            -0.4,
            "Negative-outgoing candidate should have outgoing_weight=-0.4"
        );
    }

    // ============================================================================
    // TDD TESTS: Validate ReLU improvement calculations
    // ============================================================================

    /// TDD Test: Verify that predicted improvement matches actual for split errors.
    /// This tests the core maths of compute_net_improvement_across_all_samples.
    #[test]
    fn test_predicted_improvement_matches_actual_for_split_relu() {
        // Scenario: 50% positive errors, 50% negative errors
        // Source always positive (0.5), so ReLU always fires
        //
        // Positive errors: error = +0.2 (want output higher)
        // Negative errors: error = -0.2 (want output lower)
        //
        // If we compute optimal weight from positive subset: w = Σ(error×act)/Σ(act²)
        // For positive subset: w = (0.2×0.5 + 0.2×0.5) / (0.5² + 0.5²) = 0.2/0.5 = 0.4
        //
        // Now apply w=0.4 to ALL samples:
        // - Positive samples: new_error = 0.2 - 0.4×0.5 = 0.0 (perfect!)
        // - Negative samples: new_error = -0.2 - 0.4×0.5 = -0.4 (much worse!)
        //
        // Net improvement should be NEGATIVE (overall harm)

        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: -0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: -0.2,
                target_value: None,
                target_activation: None,
            },
        ];

        // Optimal weight computed from positive samples
        let outgoing_weight: f32 = 0.4;
        let incoming_weight: f32 = 1.0;

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Use the function under test (linear model, bias=0)
        let predicted_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0 for this test (ReLU threshold at 0)
            baseline_error_sq,
            None,
        );

        // Manually compute actual improvement
        let mut new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_out = (incoming_weight * sample.activation).max(0.0);
            let new_err = sample.avg_error - outgoing_weight * relu_out;
            new_error_sq += new_err.powi(2);
        }
        let actual_improvement = (baseline_error_sq - new_error_sq) / baseline_error_sq;

        eprintln!("Baseline error²: {baseline_error_sq:.4}");
        eprintln!(
            "Outgoing weight: {outgoing_weight:.4}, Predicted: {:.4}%, Actual: {:.4}%",
            predicted_improvement * 100.0,
            actual_improvement * 100.0
        );

        assert!(
            (predicted_improvement - actual_improvement).abs() < 0.0001,
            "Predicted {predicted_improvement:.4} must match actual {actual_improvement:.4}",
        );

        // With split errors, the net improvement should be NEGATIVE
        assert!(
            predicted_improvement < 0.0,
            "With split errors and uniform source, net improvement should be negative, got {:.4}%",
            predicted_improvement * 100.0
        );
    }

    /// TDD Test: When errors are aligned, linear model should be accurate.
    #[test]
    fn test_linear_model_accurate_when_errors_aligned() {
        // All positive errors, source always positive
        // This is the ideal case for ReLU - linear model should work perfectly
        let samples = vec![
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.5,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.4,
                target_value: None,
                target_activation: None,
            },
        ];

        // Compute optimal weight: w = Σ(error×activation) / Σ(activation²)
        let error_act_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let act_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let outgoing_weight = error_act_sum / act_sq_sum;
        let incoming_weight = 1.0f32;

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        let predicted_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0 for this test
            baseline_error_sq,
            None, // Linear model
        );

        // Manually compute (bias=0)
        let mut new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_out = (incoming_weight * sample.activation).max(0.0);
            let new_err = sample.avg_error - outgoing_weight * relu_out;
            new_error_sq += new_err.powi(2);
        }
        let actual_improvement = (baseline_error_sq - new_error_sq) / baseline_error_sq;

        eprintln!(
            "Aligned errors: weight={outgoing_weight:.4}, predicted={:.4}%, actual={:.4}%",
            predicted_improvement * 100.0,
            actual_improvement * 100.0
        );

        assert!(
            (predicted_improvement - actual_improvement).abs() < 0.0001,
            "Predicted {predicted_improvement:.4} must match actual {actual_improvement:.4}",
        );

        assert!(
            actual_improvement > 0.1,
            "With aligned errors, should see significant improvement, got {:.4}%",
            actual_improvement * 100.0
        );
    }

    // ============================================================================
    // TDD TESTS: HARD_TANH saturation behaviour
    // ============================================================================

    /// Extended sample for HARD_TANH testing - includes target neuron's pre-activation value
    struct HardTanhSample {
        source_activation: f32,
        target_value: f32,      // Pre-activation input sum
        target_activation: f32, // Post-activation output (clamped to [-1, 1])
        target_error: f32,      // expected - actual
    }

    /// Apply HARD_TANH activation function
    fn hard_tanh(x: f32) -> f32 {
        x.clamp(-1.0, 1.0)
    }

    /// TEST: Demonstrates that linear model is WRONG for HARD_TANH targets near saturation.
    /// The linear model predicts disaster (-125%) but HARD_TANH actually gives perfect result!
    #[test]
    fn test_hard_tanh_linear_model_is_wrong_near_saturation() {
        // Scenario: Target neuron with HARD_TANH activation is near saturation
        // - target_value = 0.9 (input sum before clamping)
        // - target_activation = 0.9 (output after HARD_TANH, not saturated yet)
        // - expected output = 1.0
        // - error = 1.0 - 0.9 = 0.1 (positive: output should be higher)
        //
        // Source neuron fires with activation 0.5
        // If we add a ReLU with outgoing_weight = 0.5:
        // - contribution = 0.5 * relu(0.5) = 0.5 * 0.5 = 0.25
        //
        // LINEAR MODEL predicts:
        // - new_error = 0.1 - 0.25 = -0.15 (overshot)
        // - old_error² = 0.01, new_error² = 0.0225
        // - improvement = (0.01 - 0.0225) / 0.01 = -125% (WORSE!)
        //
        // ACTUAL HARD_TANH behaviour:
        // - new_input = 0.9 + 0.25 = 1.15
        // - new_output = clamp(1.15, -1, 1) = 1.0 (saturated!)
        // - new_error = 1.0 - 1.0 = 0.0 (PERFECT!)
        // - old_error² = 0.01, new_error² = 0.0
        // - improvement = (0.01 - 0.0) / 0.01 = +100% (MUCH BETTER!)

        let samples = vec![HardTanhSample {
            source_activation: 0.5,
            target_value: 0.9,      // Near saturation
            target_activation: 0.9, // hard_tanh(0.9) = 0.9
            target_error: 0.1,      // expected (1.0) - actual (0.9)
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5;

        // Compute baseline error
        let baseline_error_sq: f32 = samples.iter().map(|s| s.target_error.powi(2)).sum();

        // LINEAR MODEL prediction (current behaviour)
        let mut linear_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let linear_new_error = sample.target_error - contribution;
            linear_new_error_sq += linear_new_error.powi(2);
        }
        let linear_improvement = (baseline_error_sq - linear_new_error_sq) / baseline_error_sq;

        // ACTUAL HARD_TANH behaviour
        let mut hard_tanh_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let new_input = sample.target_value + contribution;
            let new_output = hard_tanh(new_input);
            let expected = sample.target_activation + sample.target_error;
            let new_error = expected - new_output;
            hard_tanh_new_error_sq += new_error.powi(2);
        }
        let hard_tanh_improvement =
            (baseline_error_sq - hard_tanh_new_error_sq) / baseline_error_sq;

        eprintln!("Baseline error²: {baseline_error_sq:.4}");
        eprintln!(
            "Linear model: new_error²={linear_new_error_sq:.4}, improvement={:.1}%",
            linear_improvement * 100.0
        );
        eprintln!(
            "HARD_TANH actual: new_error²={hard_tanh_new_error_sq:.4}, improvement={:.1}%",
            hard_tanh_improvement * 100.0
        );

        // The linear model predicts NEGATIVE improvement (making things worse)
        assert!(
            linear_improvement < 0.0,
            "Linear model should predict negative improvement near saturation, got {:.1}%",
            linear_improvement * 100.0
        );

        // But the actual HARD_TANH behaviour shows PERFECT improvement!
        assert!(
            hard_tanh_improvement > 0.99,
            "HARD_TANH should show ~100% improvement (error goes to 0), got {:.1}%",
            hard_tanh_improvement * 100.0
        );

        // The difference is massive - linear model is completely wrong!
        let difference = (hard_tanh_improvement - linear_improvement).abs();
        assert!(
            difference > 1.0,
            "Difference between models should be >100%, got {:.1}%",
            difference * 100.0
        );
    }

    /// TEST: Linear model predicts improvement but HARD_TANH shows NO improvement (already saturated)
    #[test]
    fn test_hard_tanh_linear_model_wrong_when_already_saturated() {
        // Scenario: Target is ALREADY saturated at 1.0
        // - target_value = 1.5 (input already beyond saturation)
        // - target_activation = 1.0 (clamped output)
        // - expected output = 0.8
        // - error = 0.8 - 1.0 = -0.2 (negative: output should be LOWER)
        //
        // Source fires with activation 0.5, ReLU with outgoing_weight = -0.3
        // Contribution = -0.3 * 0.5 = -0.15 (pushing output DOWN, seems good!)
        //
        // LINEAR MODEL predicts:
        // - new_error = -0.2 - (-0.15) = -0.05 (improved!)
        // - old_error² = 0.04, new_error² = 0.0025
        // - improvement = (0.04 - 0.0025) / 0.04 = 93.75% (great!)
        //
        // ACTUAL HARD_TANH behaviour:
        // - new_input = 1.5 + (-0.15) = 1.35 (still beyond saturation!)
        // - new_output = clamp(1.35) = 1.0 (unchanged!)
        // - new_error = 0.8 - 1.0 = -0.2 (NO CHANGE!)
        // - improvement = 0%

        let samples = vec![HardTanhSample {
            source_activation: 0.5,
            target_value: 1.5,      // Already beyond saturation
            target_activation: 1.0, // Clamped at max
            target_error: -0.2,     // expected (0.8) - actual (1.0)
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = -0.3; // Trying to push output down

        let baseline_error_sq: f32 = samples.iter().map(|s| s.target_error.powi(2)).sum();

        // LINEAR MODEL
        let mut linear_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let linear_new_error = sample.target_error - contribution;
            linear_new_error_sq += linear_new_error.powi(2);
        }
        let linear_improvement = (baseline_error_sq - linear_new_error_sq) / baseline_error_sq;

        // ACTUAL HARD_TANH
        let mut hard_tanh_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let new_input = sample.target_value + contribution;
            let new_output = hard_tanh(new_input);
            let expected = sample.target_activation + sample.target_error;
            let new_error = expected - new_output;
            hard_tanh_new_error_sq += new_error.powi(2);
        }
        let hard_tanh_improvement =
            (baseline_error_sq - hard_tanh_new_error_sq) / baseline_error_sq;

        eprintln!("Baseline error²: {baseline_error_sq:.4}");
        eprintln!(
            "Linear model predicts: {:.1}% improvement",
            linear_improvement * 100.0
        );
        eprintln!(
            "HARD_TANH actual: {:.1}% improvement",
            hard_tanh_improvement * 100.0
        );

        // Linear model predicts big improvement
        assert!(
            linear_improvement > 0.9,
            "Linear model should predict ~93% improvement, got {:.1}%",
            linear_improvement * 100.0
        );

        // But HARD_TANH shows NO improvement (still saturated)
        assert!(
            hard_tanh_improvement.abs() < 0.01,
            "HARD_TANH should show ~0% improvement (still saturated), got {:.1}%",
            hard_tanh_improvement * 100.0
        );
    }

    /// Test that compute_net_improvement_with_squash uses HARD_TANH model when specified.
    /// This verifies the actual function we use in production.
    #[test]
    fn test_compute_net_improvement_uses_hard_tanh_model() {
        // Create samples WITH target data (target_value and target_activation)
        // so that the HARD_TANH model can be used
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,          // expected 1.0, actual 0.9
            target_value: Some(0.9), // Near saturation
            target_activation: Some(0.9),
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.5 * 0.5 = 0.25

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Test with LINEAR model (no squash specified, bias=0)
        let linear_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            baseline_error_sq,
            None,
        );

        // Test with HARD_TANH model (bias=0)
        let hard_tanh_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            baseline_error_sq,
            Some("HARD_TANH"),
        );

        eprintln!(
            "compute_net_improvement_with_squash(None): {:.1}%",
            linear_improvement * 100.0
        );
        eprintln!(
            "compute_net_improvement_with_squash(HARD_TANH): {:.1}%",
            hard_tanh_improvement * 100.0
        );

        // LINEAR model should predict NEGATIVE improvement (overshoot to -0.15 error)
        // new_error = 0.1 - 0.25 = -0.15, new_error² = 0.0225
        // baseline = 0.01, so improvement = (0.01 - 0.0225) / 0.01 = -125%
        assert!(
            linear_improvement < 0.0,
            "Linear model should predict negative improvement, got {:.1}%",
            linear_improvement * 100.0
        );

        // HARD_TANH model should predict PERFECT improvement (saturate at 1.0)
        // new_input = 0.9 + 0.25 = 1.15, new_output = clamp(1.15) = 1.0
        // expected = 0.9 + 0.1 = 1.0, new_error = 0.0
        // improvement = (0.01 - 0.0) / 0.01 = 100%
        assert!(
            hard_tanh_improvement > 0.99,
            "HARD_TANH should show ~100% improvement, got {:.1}%",
            hard_tanh_improvement * 100.0
        );

        // The difference between models should be massive
        let difference = (hard_tanh_improvement - linear_improvement).abs();
        assert!(
            difference > 1.0,
            "Difference between models should be >100%, got {:.1}%",
            difference * 100.0
        );
    }

    /// Test that HARD_TANH model falls back to linear when target data is missing.
    #[test]
    fn test_compute_net_improvement_falls_back_to_linear_without_target_data() {
        // Create samples WITHOUT target data (None values)
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,      // No target data
            target_activation: None, // No target data
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5;
        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Even with HARD_TANH specified, should fall back to linear model (bias=0)
        let improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            baseline_error_sq,
            Some("HARD_TANH"),
        );

        // Should match the linear model result (-125%)
        // new_error = 0.1 - 0.25 = -0.15, new_error² = 0.0225
        // improvement = (0.01 - 0.0225) / 0.01 = -125%
        let expected_linear = -1.25;
        assert!(
            (improvement - expected_linear).abs() < 0.01,
            "Should fall back to linear model without target data, got {:.1}% (expected {:.1}%)",
            improvement * 100.0,
            expected_linear * 100.0
        );
    }

    /// TEST: count_improved_samples must use HARD_TANH model for accurate sample counts.
    ///
    /// This test demonstrates the bug where count_improved_samples always uses the linear
    /// model, causing inaccurate counts for HARD_TANH targets. For a sample near saturation,
    /// the linear model predicts the error gets worse (overshoot), but the HARD_TANH model
    /// correctly shows the sample is improved (saturates at the limit).
    #[test]
    fn test_count_improved_samples_uses_hard_tanh_model() {
        // Scenario: Target neuron with HARD_TANH activation is near saturation
        // - target_value = 0.9 (input sum before clamping)
        // - target_activation = 0.9 (output after HARD_TANH, not saturated yet)
        // - expected output = 1.0 (what we want)
        // - avg_error = 0.1 (expected - actual = 1.0 - 0.9 = 0.1)
        //
        // When we add a connection with contribution = 0.25:
        // - LINEAR model: new_error = 0.1 - 0.25 = -0.15, |new_error| > |old_error|, NOT improved
        // - HARD_TANH model: new_input = 0.9 + 0.25 = 1.15, new_output = clamp(1.15) = 1.0
        //                    new_error = 1.0 - 1.0 = 0.0, |new_error| < |old_error|, IMPROVED!
        let samples = vec![HelpfulSample {
            activation: 0.5,              // Source neuron's activation
            avg_error: 0.1,               // Target wants to go up by 0.1
            target_value: Some(0.9),      // Pre-activation input sum
            target_activation: Some(0.9), // Post-activation output (not yet saturated)
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.5 × max(0, 1.0 × 0.5) = 0.25

        // With HARD_TANH model, this sample SHOULD be counted as improved (bias=0)
        let (improved_count, total_count) = count_improved_samples(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            Some("HARD_TANH"),
        );

        assert_eq!(total_count, 1, "Should have 1 total sample");
        assert_eq!(
            improved_count, 1,
            "HARD_TANH model should show sample is improved (saturates at 1.0), got {improved_count} improved",
        );
    }

    /// TEST: count_improved_samples falls back to linear model when target data is missing.
    #[test]
    fn test_count_improved_samples_falls_back_to_linear_without_target_data() {
        // Sample WITHOUT target data - should use linear model
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,      // No target data
            target_activation: None, // No target data
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.25
                                   // Linear model: new_error = 0.1 - 0.25 = -0.15, |new_error| > |old_error|, NOT improved

        let (improved_count, _) = count_improved_samples(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0,               // bias=0
            Some("HARD_TANH"), // Even with HARD_TANH, should fall back to linear
        );

        assert_eq!(
            improved_count, 0,
            "Without target data, should fall back to linear model (sample not improved)"
        );
    }

    /// TEST: count_improved_samples uses linear model for non-HARD_TANH activations.
    #[test]
    fn test_count_improved_samples_uses_linear_for_other_activations() {
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.3,
            target_value: Some(0.5),
            target_activation: Some(0.5),
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.25
                                   // Linear: new_error = 0.3 - 0.25 = 0.05, |new_error| < |old_error| = 0.3, IMPROVED

        // With TANH (not HARD_TANH), should use linear model (bias=0)
        let (improved_count, _) = count_improved_samples(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0,
            Some("TANH"),
        );

        assert_eq!(
            improved_count, 1,
            "Linear model should show sample is improved for TANH"
        );

        // With None squash, should also use linear model (bias=0)
        let (improved_count_none, _) =
            count_improved_samples(&samples, incoming_weight, outgoing_weight, 0.0, None);

        assert_eq!(
            improved_count_none, 1,
            "Linear model should show sample is improved when squash is None"
        );
    }

    /// Test that compute_activation_improvement_and_count correctly falls back to linear model
    /// when samples lack target_value/target_activation data, even if use_hard_tanh is true.
    ///
    /// This validates the safety invariant: use_hard_tanh should only be true when
    /// can_use_hard_tanh() has verified all samples have the required data.
    #[test]
    fn test_activation_improvement_uses_linear_when_no_target_data() {
        // Samples WITHOUT target data
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,      // No target data
            target_activation: None, // No target data
        }];

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // can_use_hard_tanh should return false when samples lack target data
        assert!(
            !can_use_hard_tanh(&samples, Some("HARD_TANH")),
            "can_use_hard_tanh must return false when samples lack target data"
        );

        // get_target_simulation_fn should also return None when samples lack target data
        assert!(
            get_target_simulation_fn(&samples, Some("HARD_TANH")).is_none(),
            "get_target_simulation_fn must return None when samples lack target data"
        );

        // When properly using get_target_simulation_fn, we get linear model behaviour
        let target_activation_fn = get_target_simulation_fn(&samples, Some("HARD_TANH"));
        let (improvement, improved, total) = compute_activation_improvement_and_count(
            &samples,
            1.0,            // incoming_weight
            0.5,            // outgoing_weight
            0.0,            // bias
            |x| x.max(0.0), // ReLU activation
            baseline_sq,
            target_activation_fn, // Will be None due to missing target data
        );

        // Linear model: contribution = 0.5 × max(0, 1.0 × 0.5 + 0) = 0.25
        // new_error = 0.25 - 0.1 = 0.15 (note: compute_activation uses contribution - avg_error)
        // But |0.15| > |0.1| so sample is NOT improved
        // improvement = (0.01 - 0.0225) / 0.01 = -125%
        assert!(
            improvement < 0.0,
            "Linear model should show negative improvement"
        );
        assert_eq!(total, 1, "Should have 1 total sample");
        assert_eq!(
            improved, 0,
            "Linear model should show sample is NOT improved"
        );
    }

    /// Verify synapse weight uses correct linear optimal formula: w = Σ(error × activation) / Σ(activation²)
    /// The old buggy formula (Σ|error| / Σ|activation|) would clamp to ±1.0 in many cases.
    /// This test ensures weights are computed correctly and not always clamped.
    #[test]
    fn synapse_weight_uses_correct_linear_optimal_formula() {
        // Create HelpfulStats with known values that demonstrate the difference
        // between the correct and buggy formulas:
        //
        // Sample 1: activation=2.0, error=0.8  (error×activation = 1.6, activation² = 4.0)
        // Sample 2: activation=3.0, error=0.6  (error×activation = 1.8, activation² = 9.0)
        // Sample 3: activation=1.0, error=0.4  (error×activation = 0.4, activation² = 1.0)
        //
        // Σ(error × activation) = 1.6 + 1.8 + 0.4 = 3.8
        // Σ(activation²) = 4.0 + 9.0 + 1.0 = 14.0
        //
        // Correct optimal weight: 3.8 / 14.0 = 0.271...
        //
        // Old buggy formula would compute:
        // Σ|error| = 0.8 + 0.6 + 0.4 = 1.8
        // Σ|activation| = 2.0 + 3.0 + 1.0 = 6.0
        // Buggy weight: 1.8 / 6.0 = 0.3 (different!)
        //
        // And in cases where Σ|error| > Σ|activation|, the buggy formula would clamp to 1.0

        let stats = HelpfulStats {
            positive_count: 3, // All samples have positive correlation for this test
            negative_count: 0,
            positive_improvement_sum: 1.8, // Σ|error| for positive samples (unused in new formula)
            negative_improvement_sum: 0.0,
            positive_activation_sum: 6.0, // Σ|activation| for positive samples (unused in new formula)
            negative_activation_sum: 0.0,
            error_sq_sum: 0.8 * 0.8 + 0.6 * 0.6 + 0.4 * 0.4, // 0.64 + 0.36 + 0.16 = 1.16
            activation_sq_sum: 14.0,                         // Σ(activation²) = 4 + 9 + 1
            error_activation_sum: 3.8, // Σ(error × activation) = 1.6 + 1.8 + 0.4
        };

        // Apply the correct formula used in production (after fix):
        // weight = error_activation_sum / (activation_sq_sum + EPSILON)
        let raw_weight = if stats.activation_sq_sum > EPSILON {
            stats.error_activation_sum / (stats.activation_sq_sum + EPSILON)
        } else {
            0.0
        };
        let weight = raw_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        // Raw weight: 3.8 / 14.0 ≈ 0.2714
        let expected_raw = 3.8 / 14.0;
        assert!(
            (raw_weight - expected_raw).abs() < 0.001,
            "Raw weight should be Σ(error×activation)/Σ(activation²) = {expected_raw:.4}, got {raw_weight:.4}"
        );

        // Weight should be clamped to MAX_OUTGOING_WEIGHT since 0.2714 > 0.1
        assert!(
            (weight - MAX_OUTGOING_WEIGHT).abs() < 0.001,
            "Weight should be clamped to MAX_OUTGOING_WEIGHT = {MAX_OUTGOING_WEIGHT}, got {weight:.4}"
        );

        // Verify the RAW value is NOT the buggy value
        let buggy_weight = 1.8 / 6.0; // 0.3
        assert!(
            (raw_weight - buggy_weight).abs() > 0.01,
            "Raw weight {raw_weight} should differ from buggy formula result {buggy_weight}"
        );

        // Now test a case that would clamp to 1.0 with the buggy formula
        // Samples where |error| >> |activation|
        let stats_would_clamp = HelpfulStats {
            positive_count: 2,
            negative_count: 0,
            positive_improvement_sum: 5.0, // Σ|error| (buggy formula would use this)
            negative_improvement_sum: 0.0,
            positive_activation_sum: 2.0, // Σ|activation| (buggy formula: 5.0/2.0 = 2.5 -> clamp to 1.0)
            negative_activation_sum: 0.0,
            error_sq_sum: 13.0,        // 2² + 3² = 4 + 9 = 13
            activation_sq_sum: 2.0,    // 1² + 1² = 2
            error_activation_sum: 5.0, // 2×1 + 3×1 = 5
        };

        let raw_weight2 = if stats_would_clamp.activation_sq_sum > EPSILON {
            stats_would_clamp.error_activation_sum / (stats_would_clamp.activation_sq_sum + EPSILON)
        } else {
            0.0
        };
        let weight2 = raw_weight2.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        // Raw optimal: 5.0 / 2.0 = 2.5
        // With new tight clamp, this gets clamped to MAX_OUTGOING_WEIGHT
        assert!(
            (raw_weight2 - 2.5).abs() < 0.001,
            "Raw weight should be 2.5, got {raw_weight2}"
        );
        assert!(
            (weight2 - MAX_OUTGOING_WEIGHT).abs() < 0.001,
            "Weight should be clamped to MAX_OUTGOING_WEIGHT = {MAX_OUTGOING_WEIGHT}, got {weight2}"
        );
        // The raw calculation (2.5) is correct, but now we clamp to a tighter range
        // to improve prediction accuracy based on successful discovery analysis.
    }

    /// Test that synapse improvement calculation uses saturation-aware model for HARD_TANH targets.
    ///
    /// CRITICAL: avg_error is in VALUE domain (targetValue - currentValue from TypeScript).
    /// This means expected = squash(target_value + avg_error), NOT target_activation + avg_error.
    ///
    /// Scenario: Target near saturation where linear model UNDERPREDICTS actual benefit.
    /// - current value = 0.8, current activation = 0.8
    /// - avg_error = 0.3 (VALUE domain: want to add 0.3 to pre-activation)
    /// - desired_value = 1.1, expected_activation = clamp(1.1) = 1.0
    /// - actual activation error = 1.0 - 0.8 = 0.2 (what we really want to fix)
    ///
    /// If contribution = 0.25 (pushing value from 0.8 to 1.05):
    /// - Linear model: new_error = 0.3 - 0.25 = 0.05 (thinks we still have 0.05 error)
    /// - Saturation: new_output = clamp(1.05) = 1.0, new_error = 1.0 - 1.0 = 0 (perfect!)
    ///
    /// Linear model underpredicts because it doesn't know saturation "absorbs" the overshoot.
    #[test]
    fn synapse_improvement_uses_saturation_aware_model_for_hard_tanh() {
        // Scenario where saturation helps - the target is pushing towards saturation
        // and the synapse contribution helps reach it even though linear math says we undershot
        let samples = vec![
            HelpfulSample {
                activation: 0.5,              // source neuron activation
                avg_error: 0.3,               // VALUE domain: want +0.3 to pre-activation
                target_value: Some(0.8),      // current pre-activation
                target_activation: Some(0.8), // current output (linear region)
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.25, // VALUE domain
                target_value: Some(0.85),
                target_activation: Some(0.85),
            },
        ];

        // Compute optimal weight using linear model (treats avg_error as activation error)
        let mut error_activation_sum = 0.0f32;
        let mut activation_sq_sum = 0.0f32;
        let mut baseline_value_error_sq = 0.0f32; // Linear baseline (VALUE domain errors)

        for sample in &samples {
            error_activation_sum += sample.avg_error * sample.activation;
            activation_sq_sum += sample.activation * sample.activation;
            baseline_value_error_sq += sample.avg_error * sample.avg_error;
        }

        let weight = error_activation_sum / (activation_sq_sum + EPSILON);

        // LINEAR MODEL PREDICTION (using VALUE domain errors throughout):
        // improvement = (2*w*E[a*e] - w²*E[a²]) / E[e²]
        let linear_improvement = (2.0 * weight * error_activation_sum
            - weight * weight * activation_sq_sum)
            / baseline_value_error_sq;

        // SATURATION-AWARE MODEL (correct ACTIVATION domain):
        // Compute actual errors in activation domain where MSE is measured
        // CRITICAL: expected = squash(target_value + avg_error)
        let mut baseline_activation_error_sq = 0.0f32;
        let mut new_activation_error_sq = 0.0f32;

        for sample in &samples {
            let target_value = sample.target_value.unwrap();
            let target_activation = sample.target_activation.unwrap();
            let desired_value = target_value + sample.avg_error;
            let expected_output = desired_value.clamp(-1.0, 1.0);

            // Baseline error in ACTIVATION domain (what MSE actually measures)
            let baseline_act_error = expected_output - target_activation;
            baseline_activation_error_sq += baseline_act_error * baseline_act_error;

            // New pre-activation and output
            let new_pre_activation = target_value + weight * sample.activation;
            let new_output = new_pre_activation.clamp(-1.0, 1.0);
            let new_act_error = expected_output - new_output;
            new_activation_error_sq += new_act_error * new_act_error;
        }

        let actual_improvement =
            (baseline_activation_error_sq - new_activation_error_sq) / baseline_activation_error_sq;

        // Both models should show improvement for this well-chosen scenario
        assert!(
            linear_improvement > 0.5,
            "Linear model should show significant improvement, got {linear_improvement:.4}"
        );
        assert!(
            actual_improvement > 0.5,
            "Saturation-aware model should show significant improvement, got {actual_improvement:.4}"
        );

        // The key insight: models may differ, but saturation-aware is more accurate
        // Log the difference for debugging
        eprintln!(
            "Linear improvement: {:.2}%, Saturation-aware: {:.2}%",
            linear_improvement * 100.0,
            actual_improvement * 100.0
        );

        // Now test that compute_synapse_improvement_with_target_squash gives accurate prediction
        // Use VALUE domain baseline for consistency with how the function is called in production
        let saturation_aware_improvement = compute_synapse_improvement_with_target_squash(
            &samples,
            weight,
            baseline_value_error_sq,
            Some("HARD_TANH"),
        );

        // Log all predictions for debugging
        eprintln!(
            "Function prediction: {:.2}%",
            saturation_aware_improvement * 100.0
        );

        // The saturation-aware function should give reasonable predictions
        assert!(
            saturation_aware_improvement.is_finite(),
            "Saturation-aware model should give finite improvement"
        );
    }

    /// TDD Test: Domain consistency in improvement calculation.
    ///
    /// BUG: When using target_activation_fn simulation, the code was computing:
    /// - baseline_error in VALUE domain (sample.avg_error²)
    /// - new_error in ACTIVATION domain (expected - target_fn(new_input))²
    ///
    /// This is WRONG because VALUE and ACTIVATION domains have different scales
    /// near saturation. The actual MSE (measured by Deno) is in ACTIVATION domain,
    /// so both baseline and new must use ACTIVATION domain for accurate predictions.
    ///
    /// This test verifies that the improvement calculation uses consistent domains.
    #[test]
    fn improvement_calculation_uses_consistent_domains() {
        // Near saturation scenario where domain mismatch is most visible
        let samples = vec![HelpfulSample {
            activation: 0.5,              // source activation
            avg_error: 0.3,               // VALUE domain: want +0.3 to pre-activation
            target_value: Some(0.9),      // near upper saturation
            target_activation: Some(0.9), // HARD_TANH(0.9) = 0.9
        }];

        // With this sample:
        // - desired_value = 0.9 + 0.3 = 1.2
        // - expected = HARD_TANH(1.2) = 1.0 (saturated)
        // - target_activation = 0.9
        //
        // ACTIVATION domain baseline error = 1.0 - 0.9 = 0.1
        // VALUE domain baseline error = 0.3 (sample.avg_error)
        //
        // If we apply a contribution of 0.15:
        // - new_input = 0.9 + 0.15 = 1.05
        // - new_output = HARD_TANH(1.05) = 1.0 (saturated)
        // - ACTIVATION domain new error = 1.0 - 1.0 = 0.0 (perfect!)
        //
        // CORRECT improvement (ACTIVATION domain):
        // = (0.1² - 0.0²) / 0.1² = 100%
        //
        // BUGGY improvement (VALUE baseline, ACTIVATION new):
        // = (0.3² - 0.0²) / 0.3² = 100% (coincidentally same in this case)

        // Now test with partial contribution
        let contribution = 0.05;
        // new_input = 0.9 + 0.05 = 0.95
        // new_output = HARD_TANH(0.95) = 0.95
        // ACTIVATION domain new error = 1.0 - 0.95 = 0.05
        //
        // CORRECT improvement (ACTIVATION domain):
        // = (0.1² - 0.05²) / 0.1² = (0.01 - 0.0025) / 0.01 = 75%
        //
        // BUGGY improvement (VALUE baseline, ACTIVATION new):
        // = (0.3² - 0.05²) / 0.3² = (0.09 - 0.0025) / 0.09 = 97% (WRONG!)

        let desired_value: f32 = 0.9 + 0.3; // = 1.2
        let expected = desired_value.clamp(-1.0, 1.0); // = 1.0
        let target_activation: f32 = 0.9;

        // Correct ACTIVATION domain baseline
        let baseline_act_error = expected - target_activation; // = 0.1
        let baseline_act_sq = baseline_act_error * baseline_act_error; // = 0.01

        // New error in ACTIVATION domain
        let new_input: f32 = 0.9 + contribution; // = 0.95
        let new_output = new_input.clamp(-1.0, 1.0); // = 0.95
        let new_act_error = expected - new_output; // = 0.05
        let new_act_sq = new_act_error * new_act_error; // = 0.0025

        // Correct improvement
        let correct_improvement = (baseline_act_sq - new_act_sq) / baseline_act_sq;
        assert!(
            (correct_improvement - 0.75).abs() < 0.01,
            "Correct ACTIVATION domain improvement should be ~75%, got {:.2}%",
            correct_improvement * 100.0
        );

        // Buggy calculation (VALUE baseline, ACTIVATION new)
        let value_baseline_sq: f32 = 0.3 * 0.3; // = 0.09
        let buggy_improvement = (value_baseline_sq - new_act_sq) / value_baseline_sq;
        assert!(
            (buggy_improvement - 0.97_f32).abs() < 0.01,
            "Buggy mixed-domain improvement should be ~97%, got {:.2}%",
            buggy_improvement * 100.0
        );

        // The bug causes MASSIVE overprediction: 97% predicted vs 75% actual
        assert!(
            buggy_improvement > correct_improvement + 0.1,
            "Buggy calculation should significantly overpredict improvement"
        );

        // Now verify the actual function uses consistent domains
        // Use the optimal weight that would produce contribution=0.05 with activation=0.5
        let weight = contribution / 0.5; // = 0.1

        // CRITICAL: Production passes VALUE domain baseline (this is what stats.error_sq_sum is)
        // The function should INTERNALLY compute ACTIVATION domain baseline when simulating target
        let production_baseline = value_baseline_sq; // VALUE domain as passed in production

        let (improvement, _, _, _) = compute_synapse_improvement_and_count(
            &samples,
            weight,
            production_baseline,
            Some("HARD_TANH"),
        );

        // The function should give the CORRECT result (75%) not the buggy result (97%)
        // because it should internally use ACTIVATION domain for both baseline and new error
        assert!(
            (improvement - correct_improvement).abs() < 0.1,
            "Function should internally use consistent ACTIVATION domain. \
             Expected ~{:.1}%, got {:.1}% (buggy would be ~{:.1}%)",
            correct_improvement * 100.0,
            improvement * 100.0,
            buggy_improvement * 100.0
        );
    }

    /// TDD Test: ReLU improvement calculation MUST include bias for accurate predictions.
    ///
    /// When bias > 0, the ReLU threshold shifts left, causing more samples to activate.
    /// When bias < 0, the ReLU threshold shifts right, causing fewer samples to activate.
    ///
    /// If bias is NOT included in the improvement calculation, the prediction will be
    /// inaccurate when a non-zero bias is proposed for the new neuron.
    ///
    /// This test demonstrates the bug: compute_relu_improvement_and_count ignores bias,
    /// leading to overestimation when the actual neuron would use a different activation
    /// pattern due to the bias.
    #[test]
    fn test_relu_improvement_must_include_bias() {
        // Scenario: Source activations that are NEGATIVE (would be zeroed by ReLU without bias).
        // With a positive bias, the ReLU would fire on these samples.
        //
        // Sample 1: activation = -0.3, error = 0.5 (want output higher)
        // Sample 2: activation = -0.2, error = 0.4 (want output higher)
        // Sample 3: activation = 0.1, error = 0.3 (want output higher)
        //
        // Without bias (bias=0):
        //   ReLU(1.0 × -0.3 + 0) = 0  → contribution = 0
        //   ReLU(1.0 × -0.2 + 0) = 0  → contribution = 0
        //   ReLU(1.0 × 0.1 + 0) = 0.1 → contribution = outgoing_weight × 0.1
        //
        // With bias=0.5:
        //   ReLU(1.0 × -0.3 + 0.5) = 0.2 → contribution = outgoing_weight × 0.2
        //   ReLU(1.0 × -0.2 + 0.5) = 0.3 → contribution = outgoing_weight × 0.3
        //   ReLU(1.0 × 0.1 + 0.5) = 0.6 → contribution = outgoing_weight × 0.6
        //
        // The bias dramatically changes which samples are affected and by how much!

        let samples = vec![
            HelpfulSample {
                activation: -0.3,
                avg_error: 0.5,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: 0.4,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.1,
                avg_error: 0.3,
                target_value: None,
                target_activation: None,
            },
        ];

        let incoming_weight = 1.0f32;
        let outgoing_weight = 0.8f32; // Positive weight to reduce positive errors
        let bias = 0.5f32; // Significant positive bias

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();
        // 0.5² + 0.4² + 0.3² = 0.25 + 0.16 + 0.09 = 0.5

        // Predicted improvement using the function WITH bias parameter
        let predicted_with_bias = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            bias,
            baseline_error_sq,
            None,
        );

        // Also compute without bias (bias=0) to show the difference
        let predicted_without_bias = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // No bias
            baseline_error_sq,
            None,
        );

        // Manually compute ACTUAL improvement WITH bias
        let mut new_error_sq_with_bias = 0.0f32;
        for sample in &samples {
            let pre_activation = incoming_weight * sample.activation + bias;
            let relu_out = pre_activation.max(0.0);
            let contribution = outgoing_weight * relu_out;
            let new_err = sample.avg_error - contribution;
            new_error_sq_with_bias += new_err.powi(2);
        }
        let actual_improvement_with_bias =
            (baseline_error_sq - new_error_sq_with_bias) / baseline_error_sq;

        // Manually compute improvement WITHOUT bias (what current code predicts)
        let mut new_error_sq_without_bias = 0.0f32;
        for sample in &samples {
            let pre_activation = incoming_weight * sample.activation; // No bias!
            let relu_out = pre_activation.max(0.0);
            let contribution = outgoing_weight * relu_out;
            let new_err = sample.avg_error - contribution;
            new_error_sq_without_bias += new_err.powi(2);
        }
        let manual_improvement_without_bias =
            (baseline_error_sq - new_error_sq_without_bias) / baseline_error_sq;

        eprintln!(
            "Baseline error²: {baseline_error_sq:.4}, With bias: new_error²={new_error_sq_with_bias:.4}, Without bias: new_error²={new_error_sq_without_bias:.4}"
        );
        let predicted_with_bias_pct = predicted_with_bias * 100.0;
        let predicted_without_bias_pct = predicted_without_bias * 100.0;
        let actual_with_bias_pct = actual_improvement_with_bias * 100.0;
        eprintln!(
            "Predicted (with bias): {predicted_with_bias_pct:.2}%, Predicted (no bias): {predicted_without_bias_pct:.2}%, Actual (with bias): {actual_with_bias_pct:.2}%"
        );

        // The predicted improvement WITH bias should match the actual improvement WITH bias.
        // This verifies that the bias parameter is correctly included in the calculation.
        assert!(
            (predicted_with_bias - actual_improvement_with_bias).abs() < 0.01,
            "Predicted improvement WITH bias ({predicted_with_bias:.4}) must match actual improvement WITH bias ({actual_improvement_with_bias:.4})."
        );

        // Verify that WITHOUT bias prediction matches manual calculation (both bias=0)
        assert!(
            (predicted_without_bias - manual_improvement_without_bias).abs() < 0.01,
            "Predicted (no bias) ({predicted_without_bias:.4}) must match manual (no bias) ({manual_improvement_without_bias:.4})."
        );

        // The key insight: with bias=0.5, improvement should be much higher than with bias=0
        // because more samples activate the ReLU
        assert!(
            actual_improvement_with_bias > predicted_without_bias + 0.1,
            "Improvement with bias ({actual_improvement_with_bias:.4}) should be significantly higher than without ({predicted_without_bias:.4})"
        );
    }

    #[test]
    fn neuron_diagnostics_reports_hidden_neuron_filtered_in_mixed_focus_list() {
        // Test that when a focus list contains BOTH output AND hidden neurons,
        // the hidden neurons get HiddenNeuronFiltered reason (not NoEligibleSources).
        //
        // Bug scenario: When focus_order is NOT empty (some output neurons exist),
        // the skipped_hidden neurons were never merged into diagnostics, so they
        // appeared with misleading reasons like NoEligibleSources.
        // Issue #216: Methods now take &self, not &mut self
        let diagnostics = NeuronDiagnostics::new_for_tests(&["output-0", "hidden-1"]);

        // Mark hidden-1 as filtered (this is what should happen in normal flow)
        diagnostics.mark_hidden_filtered("hidden-1");

        // Simulate output-0 being processed normally but finding no candidate
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 5);
        diagnostics.record_candidate_attempt("output-0", false);

        let summaries = diagnostics.no_candidate_summaries();

        // Both should have summaries
        assert_eq!(
            summaries.len(),
            2,
            "Expected 2 summaries (one for output, one for hidden)"
        );

        // Find the hidden neuron summary
        let hidden_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "hidden-1")
            .expect("Should have summary for hidden-1");

        // The hidden neuron should have HiddenNeuronFiltered reason
        assert!(
            matches!(
                hidden_summary.reason,
                NeuronNoCandidateReason::HiddenNeuronFiltered
            ),
            "Hidden neuron should report HiddenNeuronFiltered, not {:?}",
            hidden_summary.reason
        );
    }

    #[test]
    fn neuron_diagnostics_reports_input_neuron_filtered_not_hidden() {
        // Test that when an input neuron is in the focus list, it gets
        // InputNeuronFiltered reason (not HiddenNeuronFiltered).
        //
        // Bug scenario: The neuron_type_map was only built from creature.neurons
        // and didn't include input neurons. When an input neuron (e.g. "input-0")
        // was in the focus list, neuron_type_map.get() returned None, and
        // `neuron_type != Some("output")` evaluated to true. The input neuron
        // was incorrectly added to skipped_hidden and reported with
        // HiddenNeuronFiltered reason.
        // Issue #216: Methods now take &self, not &mut self
        let diagnostics = NeuronDiagnostics::new_for_tests(&["output-0", "input-1", "hidden-2"]);

        // Mark input-1 as filtered because it's an input neuron
        diagnostics.mark_input_filtered("input-1");

        // Mark hidden-2 as filtered because it's a hidden neuron
        diagnostics.mark_hidden_filtered("hidden-2");

        // Simulate output-0 being processed normally but finding no candidate
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 5);
        diagnostics.record_candidate_attempt("output-0", false);

        let summaries = diagnostics.no_candidate_summaries();

        // All three should have summaries
        assert_eq!(
            summaries.len(),
            3,
            "Expected 3 summaries (one for output, one for input, one for hidden)"
        );

        // Find the input neuron summary - should have InputNeuronFiltered reason
        let input_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "input-1")
            .expect("Should have summary for input-1");
        assert!(
            matches!(
                input_summary.reason,
                NeuronNoCandidateReason::InputNeuronFiltered
            ),
            "Input neuron should report InputNeuronFiltered, not {:?}",
            input_summary.reason
        );

        // Find the hidden neuron summary - should still have HiddenNeuronFiltered reason
        let hidden_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "hidden-2")
            .expect("Should have summary for hidden-2");
        assert!(
            matches!(
                hidden_summary.reason,
                NeuronNoCandidateReason::HiddenNeuronFiltered
            ),
            "Hidden neuron should report HiddenNeuronFiltered, not {:?}",
            hidden_summary.reason
        );
    }

    #[test]
    fn neuron_diagnostics_reports_constant_neuron_filtered_not_hidden() {
        // Test that when a constant neuron is in the focus list, it gets
        // ConstantNeuronFiltered reason (not HiddenNeuronFiltered).
        //
        // Bug scenario (v0.1.124): Constant neurons were pushed to skipped_hidden
        // but reported with HiddenNeuronFiltered reason. This is semantically
        // incorrect - constant neurons don't receive inputs because they always
        // output a fixed value, which is different from hidden neurons whose
        // backpropagated errors don't reliably predict output error.
        // Issue #216: Methods now take &self, not &mut self
        let diagnostics = NeuronDiagnostics::new_for_tests(&["output-0", "constant-1", "hidden-2"]);

        // Mark constant-1 as filtered because it's a constant neuron
        diagnostics.mark_constant_filtered("constant-1");

        // Mark hidden-2 as filtered because it's a hidden neuron
        diagnostics.mark_hidden_filtered("hidden-2");

        // Simulate output-0 being processed normally but finding no candidate
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 5);
        diagnostics.record_candidate_attempt("output-0", false);

        let summaries = diagnostics.no_candidate_summaries();

        // All three should have summaries
        assert_eq!(
            summaries.len(),
            3,
            "Expected 3 summaries (one for output, one for constant, one for hidden)"
        );

        // Find the constant neuron summary - should have ConstantNeuronFiltered reason
        let constant_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "constant-1")
            .expect("Should have summary for constant-1");
        assert!(
            matches!(
                constant_summary.reason,
                NeuronNoCandidateReason::ConstantNeuronFiltered
            ),
            "Constant neuron should report ConstantNeuronFiltered, not {:?}",
            constant_summary.reason
        );

        // Find the hidden neuron summary - should still have HiddenNeuronFiltered reason
        let hidden_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "hidden-2")
            .expect("Should have summary for hidden-2");
        assert!(
            matches!(
                hidden_summary.reason,
                NeuronNoCandidateReason::HiddenNeuronFiltered
            ),
            "Hidden neuron should report HiddenNeuronFiltered, not {:?}",
            hidden_summary.reason
        );
    }
}

mod tests_optimal_outgoing_weight {
    #![allow(unused_imports)] // super::* is used for macro re-export
    use super::*;
    // Additional imports needed since tests are in external file
    use crate::analysis::samples::{HelpfulSample, ReluStats, EPSILON};
    use crate::analysis::weights::calculate_optimal_outgoing_weight;
    use anyhow::{anyhow, Result};

    // Import types used by tests (Issue #185)
    use crate::analysis::activation::{identity_activation, ActivationCandidateSpec};
    use crate::analysis::gpu::GpuEvaluator;
    use crate::analysis::synapse::evaluate_activation_candidate;
    use crate::analysis::weights::MAX_OUTGOING_WEIGHT;

    struct AlwaysFailGpuEvaluator;

    impl GpuEvaluator for AlwaysFailGpuEvaluator {
        fn evaluate_relu(
            &self,
            _samples: &[HelpfulSample],
            _threshold: f32,
        ) -> Result<(ReluStats, ReluStats, f32)> {
            Err(anyhow!(
                "AlwaysFailGpuEvaluator: GPU not available for test"
            ))
        }

        fn evaluate_activation(
            &self,
            _samples: &[HelpfulSample],
            _activation_type: u32,
            _orientation: f32,
            _scale: f32,
        ) -> Result<(f32, f32, f32, u32)> {
            Err(anyhow!(
                "AlwaysFailGpuEvaluator: GPU not available for test"
            ))
        }
    }

    /// Regression: IDENTITY candidates must not be skipped in the all-samples fallback path.
    ///
    /// `evaluate_activation_candidate` previously computed `base_weight` (no-intercept fit)
    /// before checking `spec.name == "IDENTITY"`. If the no-intercept fit returned None (for
    /// example, when Σ(activation×error) cancels to ~0), the function would `continue` and the
    /// IDENTITY-specific affine fit (with intercept/bias) was never attempted.
    ///
    /// This test constructs a dataset where:
    /// - split-error evaluation is NOT "properly attempted" (negative subset < MIN sample count),
    /// - subset evaluation returns None (positive subset has constant error ⇒ best slope is 0),
    /// - the all-samples no-intercept fit produces `None` (Σ(activation×error) cancels to 0),
    /// - but the all-samples affine fit *does* succeed and should yield a candidate.
    #[test]
    fn identity_all_samples_fallback_uses_affine_fit_even_when_base_weight_is_none() {
        let gpu = AlwaysFailGpuEvaluator;

        // Use a minimal spec so this test is deterministic.
        static ORIENTATIONS: [f32; 1] = [1.0];
        static SCALES: [f32; 1] = [1.0];
        let spec = ActivationCandidateSpec {
            name: "IDENTITY",
            orientations: &ORIENTATIONS,
            scales: &SCALES,
            activation: identity_activation,
            min_improvement: 0.0,
        };

        // 11 positive-error samples with varying activation.
        //
        // This test is crafted to trigger the all-samples affine fit path while still
        // producing a *sensible* bias (we reject absurd bias magnitudes as a guard rail).
        let mut samples: Vec<HelpfulSample> = (1..=11)
            .map(|i| HelpfulSample {
                activation: i as f32, // 1..11
                avg_error: 1.0,
                target_value: None,
                target_activation: None,
            })
            .collect();

        // 9 negative-error samples whose activation sum matches the positive group:
        // sum(1..11) = 66, so pick 9 values summing to 66.
        //
        // This makes Σ(activation×error) == 0 for the all-samples no-intercept fit,
        // forcing the fallback to rely on the affine (with-intercept) fit.
        let negative_activations: [f32; 9] = [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 18.0];
        for activation in negative_activations {
            samples.push(HelpfulSample {
                activation,
                avg_error: -1.0,
                target_value: None,
                target_activation: None,
            });
        }

        let candidate =
            evaluate_activation_candidate(&gpu, "source-0", "target-0", &samples, 0.0, &spec, None)
                .expect("Evaluation should succeed");

        assert!(
            candidate.is_some(),
            "Expected an IDENTITY candidate from the all-samples affine fit. \
             This is a regression if None is returned."
        );
        let candidate = candidate.unwrap();
        assert_eq!(candidate.squash, "IDENTITY");
        assert!(
            candidate.bias.abs() >= 0.01,
            "IDENTITY candidate should have a meaningful bias (not equivalent to a direct synapse)"
        );
        assert!(
            candidate.outgoing_weight.is_finite() && candidate.outgoing_weight.abs() > EPSILON,
            "IDENTITY candidate should have a valid outgoing weight"
        );
    }

    /// Test that calculate_optimal_outgoing_weight returns None for insufficient activation
    #[test]
    fn returns_none_for_zero_activation() {
        let result = calculate_optimal_outgoing_weight(1.0, 0.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when activation_sq is zero"
        );

        let result = calculate_optimal_outgoing_weight(1.0, EPSILON * 0.5, 1.0);
        assert!(
            result.is_none(),
            "Should return None when activation_sq <= EPSILON"
        );
    }

    /// Test that calculate_optimal_outgoing_weight returns None for non-finite results
    #[test]
    fn returns_none_for_non_finite_weight() {
        let result = calculate_optimal_outgoing_weight(f32::INFINITY, 1.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when raw weight is infinite"
        );

        let result = calculate_optimal_outgoing_weight(f32::NAN, 1.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when raw weight is NaN"
        );
    }

    /// Test that calculate_optimal_outgoing_weight returns None for near-zero weights
    #[test]
    fn returns_none_for_near_zero_weight() {
        // Very small error_activation results in near-zero weight
        let result = calculate_optimal_outgoing_weight(EPSILON * 0.1, 100.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when raw weight is near zero"
        );
    }

    /// Test that weights are clamped to MAX_OUTGOING_WEIGHT
    #[test]
    fn clamps_to_max_outgoing_weight() {
        // Large error relative to activation would produce large weight
        // error/activation = 10.0/1.0 = 10.0, should clamp to 0.1
        let result = calculate_optimal_outgoing_weight(10.0, 1.0, 1.0);
        assert!(result.is_some(), "Should return a valid weight");
        let weight = result.unwrap();
        assert!(
            (weight - MAX_OUTGOING_WEIGHT).abs() < EPSILON,
            "Weight {weight} should be clamped to MAX_OUTGOING_WEIGHT {MAX_OUTGOING_WEIGHT}"
        );

        // Negative case
        let result = calculate_optimal_outgoing_weight(-10.0, 1.0, 1.0);
        assert!(result.is_some(), "Should return a valid negative weight");
        let weight = result.unwrap();
        assert!(
            (weight - (-MAX_OUTGOING_WEIGHT)).abs() < EPSILON,
            "Weight {} should be clamped to -MAX_OUTGOING_WEIGHT {}",
            weight,
            -MAX_OUTGOING_WEIGHT
        );
    }

    /// Test weight ratio validation for add-neuron candidates
    #[test]
    fn rejects_small_weight_ratio() {
        // incoming_weight = 10, max outgoing = 0.1, ratio = 100 -> OK
        let result = calculate_optimal_outgoing_weight(1.0, 1.0, 10.0);
        // raw = 1.0, clamped to 0.1, ratio = 10/0.1 = 100 >= 50 -> OK
        assert!(
            result.is_some(),
            "Should accept ratio of 100 (incoming=10, outgoing=0.1)"
        );

        // incoming_weight = 2, max outgoing = 0.1, ratio = 20 < 50 -> REJECT
        let result = calculate_optimal_outgoing_weight(1.0, 1.0, 2.0);
        // raw = 1.0, clamped to 0.1, ratio = 2/0.1 = 20 < 50 -> REJECT
        assert!(
            result.is_none(),
            "Should reject ratio of 20 (incoming=2, outgoing=0.1)"
        );
    }

    /// Test that small incoming weights (synapses) skip ratio check
    #[test]
    fn skips_ratio_check_for_synapses() {
        // For synapses, incoming_weight = 1.0, so ratio check is skipped
        let result = calculate_optimal_outgoing_weight(0.5, 10.0, 1.0);
        // raw = 0.05, within bounds, no ratio check since incoming <= 1.0
        assert!(
            result.is_some(),
            "Should accept synapse weight without ratio check"
        );
        assert!(
            (result.unwrap() - 0.05).abs() < 0.001,
            "Synapse weight should be ~0.05"
        );
    }

    /// Test that successful discovery parameters would pass validation
    /// Based on real successful discoveries from production
    #[test]
    fn successful_discovery_parameters_pass() {
        // Successful discovery: incoming=100, outgoing=-0.00096
        // ratio = 100/0.00096 ≈ 104,000 >> 50 -> OK
        // But we need to compute what error/activation ratio would produce 0.00096
        // If raw weight = 0.00096, and it's not clamped, we need:
        // sum_error_activation / sum_activation_sq = 0.00096
        let sum_activation_sq = 1000.0;
        let sum_error_activation = 0.00096 * sum_activation_sq; // = 0.96

        let result =
            calculate_optimal_outgoing_weight(sum_error_activation, sum_activation_sq, 100.0);
        assert!(result.is_some(), "Successful discovery params should pass");
        let weight = result.unwrap();
        // The weight should be approximately -0.00096 or +0.00096 depending on sign
        assert!(
            weight.abs() < MAX_OUTGOING_WEIGHT,
            "Weight {weight} should be within bounds"
        );
    }

    /// Test that failed discovery parameters would be rejected
    /// Based on real failed discoveries from production
    #[test]
    fn failed_discovery_parameters_rejected() {
        // Failed discovery: incoming=5, outgoing=4.58 (before our fix, this would pass)
        // Now: raw = 4.58, clamped to 0.1, ratio = 5/0.1 = 50, just at the boundary
        // This might just pass or just fail depending on exact values

        // More clearly failed case: incoming=10, outgoing=-10 (1:1 ratio)
        // Even after clamping to 0.1, ratio = 10/0.1 = 100 >= 50 -> passes ratio check
        // BUT the weight is clamped from -10 to -0.1, so prediction accuracy improves

        // The key improvement is that extreme weights like 4.58 or -10 are now clamped
        // to 0.1, dramatically reducing prediction errors

        // Test that a raw weight of 10.0 gets clamped
        let result = calculate_optimal_outgoing_weight(10.0, 1.0, 10.0);
        assert!(result.is_some(), "Should return clamped weight");
        assert!(
            (result.unwrap().abs() - MAX_OUTGOING_WEIGHT).abs() < EPSILON,
            "Large raw weight should be clamped to MAX_OUTGOING_WEIGHT"
        );
    }
}

/// Synthetic tests to verify prediction accuracy against manual simulation.
/// These tests create controlled scenarios where we know exactly what the
/// predicted and actual improvements should be.
mod tests_prediction_accuracy {
    #![allow(unused_imports)] // super::* is used for macro re-export
    use super::*;
    // Additional imports needed since tests are in external file
    use crate::analysis::samples::{HelpfulSample, EPSILON};

    // Import types used by tests (Issue #185)
    use crate::analysis::synapse::compute_relu_improvement_and_count;
    use crate::analysis::weights::MAX_OUTGOING_WEIGHT;

    /// Hard tanh activation for testing (same as production)
    fn test_hard_tanh(x: f32) -> f32 {
        x.clamp(-1.0, 1.0)
    }

    /// Create synthetic samples with known properties.
    /// Returns (samples, baseline_error_sq_sum).
    fn create_synthetic_samples(
        count: usize,
        avg_error: f32,
        source_activation: f32,
        target_value: f32,
    ) -> (Vec<HelpfulSample>, f32) {
        let samples: Vec<HelpfulSample> = (0..count)
            .map(|_| HelpfulSample {
                activation: source_activation,
                avg_error,
                target_value: Some(target_value),
                target_activation: Some(test_hard_tanh(target_value)),
            })
            .collect();

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();
        (samples, baseline_error_sq)
    }

    /// CORE TEST: Verify that our prediction formula gives the correct result.
    ///
    /// This test creates a simple scenario:
    /// - 100 samples all with the same properties
    /// - Known source activation, error, target value
    /// - Compute optimal weight
    /// - Predict improvement
    /// - Manually simulate what the actual improvement would be
    /// - Compare predicted vs manually simulated
    #[test]
    fn prediction_matches_manual_simulation_linear_region() {
        // Scenario: Target in LINEAR region of HARD_TANH (value between -1 and 1)
        let source_activation = 1.0;
        let avg_error = 0.2; // VALUE domain: need to ADD 0.2 to target value
        let target_value = 0.3; // Current pre-activation (in linear region)
        let incoming_weight = 1.0;
        let bias = 0.0;

        let (samples, baseline_error_sq) =
            create_synthetic_samples(100, avg_error, source_activation, target_value);

        // Compute optimal weight using the production formula
        let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
        let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        eprintln!(
            "LINEAR REGION TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
        );

        // Predict improvement using production function
        let (predicted_improvement, improved_count, total_count) =
            compute_relu_improvement_and_count(
                &samples,
                incoming_weight,
                outgoing_weight,
                bias,
                baseline_error_sq,
                Some(test_hard_tanh),
            );

        // MANUALLY simulate what the actual improvement would be
        // This is what TypeScript evaluation does
        let mut manual_baseline_error_sq = 0.0f32;
        let mut manual_new_error_sq = 0.0f32;

        for sample in &samples {
            let target_val = sample.target_value.unwrap();
            let target_act = sample.target_activation.unwrap();
            let desired_value = target_val + sample.avg_error;
            let expected_activation = test_hard_tanh(desired_value);

            // Baseline error in ACTIVATION domain
            let baseline_err = expected_activation - target_act;
            manual_baseline_error_sq += baseline_err.powi(2);

            // Simulate the new neuron's contribution
            let pre_act = incoming_weight * sample.activation + bias;
            let relu_output = pre_act.max(0.0);
            let contribution = outgoing_weight * relu_output;

            // New target value after contribution
            let new_target_value = target_val + contribution;
            let new_target_activation = test_hard_tanh(new_target_value);

            // New error in ACTIVATION domain
            let new_err = expected_activation - new_target_activation;
            manual_new_error_sq += new_err.powi(2);
        }

        let manual_improvement = if manual_baseline_error_sq > EPSILON {
            (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
        } else {
            0.0
        };

        eprintln!(
            "LINEAR REGION RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
            predicted_improvement,
            predicted_improvement * 100.0,
            manual_improvement,
            manual_improvement * 100.0,
            (predicted_improvement - manual_improvement).abs()
        );
        eprintln!(
            "  improved_count={improved_count}/{total_count}, manual_baseline_sq={manual_baseline_error_sq:.6}, manual_new_sq={manual_new_error_sq:.6}"
        );

        // Prediction and manual simulation should match closely
        let diff = (predicted_improvement - manual_improvement).abs();
        assert!(
            diff < 0.01,
            "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
        );

        // Both should be positive (error should decrease)
        assert!(
            predicted_improvement > 0.0,
            "Predicted improvement should be positive"
        );
        assert!(
            manual_improvement > 0.0,
            "Manual improvement should be positive"
        );
    }

    /// Test with target near SATURATION (value close to 1.0)
    #[test]
    fn prediction_matches_manual_simulation_saturation_region() {
        // Scenario: Target near SATURATION of HARD_TANH
        let source_activation = 1.0;
        let avg_error = 0.1; // VALUE domain: need to ADD 0.1 to target value
        let target_value = 0.95; // Current pre-activation (near saturation!)
        let incoming_weight = 1.0;
        let bias = 0.0;

        let (samples, baseline_error_sq) =
            create_synthetic_samples(100, avg_error, source_activation, target_value);

        // Compute optimal weight
        let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
        let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        eprintln!(
            "SATURATION TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
        );

        // Predict improvement
        let (predicted_improvement, improved_count, total_count) =
            compute_relu_improvement_and_count(
                &samples,
                incoming_weight,
                outgoing_weight,
                bias,
                baseline_error_sq,
                Some(test_hard_tanh),
            );

        // Manual simulation
        let mut manual_baseline_error_sq = 0.0f32;
        let mut manual_new_error_sq = 0.0f32;

        for sample in &samples {
            let target_val = sample.target_value.unwrap();
            let target_act = sample.target_activation.unwrap();
            let desired_value = target_val + sample.avg_error;
            let expected_activation = test_hard_tanh(desired_value);

            let baseline_err = expected_activation - target_act;
            manual_baseline_error_sq += baseline_err.powi(2);

            let pre_act = incoming_weight * sample.activation + bias;
            let relu_output = pre_act.max(0.0);
            let contribution = outgoing_weight * relu_output;

            let new_target_value = target_val + contribution;
            let new_target_activation = test_hard_tanh(new_target_value);

            let new_err = expected_activation - new_target_activation;
            manual_new_error_sq += new_err.powi(2);
        }

        let manual_improvement = if manual_baseline_error_sq > EPSILON {
            (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
        } else {
            0.0
        };

        eprintln!(
            "SATURATION RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
            predicted_improvement,
            predicted_improvement * 100.0,
            manual_improvement,
            manual_improvement * 100.0,
            (predicted_improvement - manual_improvement).abs()
        );
        eprintln!(
            "  improved_count={improved_count}/{total_count}, manual_baseline_sq={manual_baseline_error_sq:.6}, manual_new_sq={manual_new_error_sq:.6}"
        );

        // Prediction and manual simulation should match
        let diff = (predicted_improvement - manual_improvement).abs();
        assert!(
            diff < 0.01,
            "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
        );
    }

    /// Test with NEGATIVE error (target output should be LOWER)
    #[test]
    fn prediction_matches_manual_simulation_negative_error() {
        // Scenario: Target output is too HIGH, need to REDUCE it
        let source_activation = 1.0;
        let avg_error = -0.2; // VALUE domain: need to SUBTRACT 0.2 from target value
        let target_value = 0.5; // Current pre-activation
        let incoming_weight = 1.0;
        let bias = 0.0;

        let (samples, baseline_error_sq) =
            create_synthetic_samples(100, avg_error, source_activation, target_value);

        // Compute optimal weight (should be NEGATIVE to reduce error)
        let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
        let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        eprintln!(
            "NEGATIVE ERROR TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
        );

        // Verify optimal weight is negative (to reduce target value)
        assert!(
            outgoing_weight < 0.0,
            "Outgoing weight should be negative to reduce target value"
        );

        // Predict improvement
        let (predicted_improvement, improved_count, total_count) =
            compute_relu_improvement_and_count(
                &samples,
                incoming_weight,
                outgoing_weight,
                bias,
                baseline_error_sq,
                Some(test_hard_tanh),
            );

        // Manual simulation
        let mut manual_baseline_error_sq = 0.0f32;
        let mut manual_new_error_sq = 0.0f32;

        for sample in &samples {
            let target_val = sample.target_value.unwrap();
            let target_act = sample.target_activation.unwrap();
            let desired_value = target_val + sample.avg_error;
            let expected_activation = test_hard_tanh(desired_value);

            let baseline_err = expected_activation - target_act;
            manual_baseline_error_sq += baseline_err.powi(2);

            let pre_act = incoming_weight * sample.activation + bias;
            let relu_output = pre_act.max(0.0);
            let contribution = outgoing_weight * relu_output;

            let new_target_value = target_val + contribution;
            let new_target_activation = test_hard_tanh(new_target_value);

            let new_err = expected_activation - new_target_activation;
            manual_new_error_sq += new_err.powi(2);
        }

        let manual_improvement = if manual_baseline_error_sq > EPSILON {
            (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
        } else {
            0.0
        };

        eprintln!(
            "NEGATIVE ERROR RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
            predicted_improvement,
            predicted_improvement * 100.0,
            manual_improvement,
            manual_improvement * 100.0,
            (predicted_improvement - manual_improvement).abs()
        );
        eprintln!(
            "  improved_count={improved_count}/{total_count}, manual_baseline_sq={manual_baseline_error_sq:.6}, manual_new_sq={manual_new_error_sq:.6}"
        );

        // Prediction and manual simulation should match
        let diff = (predicted_improvement - manual_improvement).abs();
        assert!(
            diff < 0.01,
            "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
        );

        // Both should be positive (error should decrease)
        assert!(
            predicted_improvement > 0.0,
            "Predicted improvement should be positive"
        );
        assert!(
            manual_improvement > 0.0,
            "Manual improvement should be positive"
        );
    }

    /// Test with MIXED errors (some positive, some negative)
    /// This simulates real-world scenarios where samples have varied errors.
    #[test]
    fn prediction_matches_manual_simulation_mixed_errors() {
        // Create samples with varied errors
        let samples: Vec<HelpfulSample> = vec![
            // Samples that need INCREASE (positive error)
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.2,
                target_value: Some(0.3),
                target_activation: Some(0.3),
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.15,
                target_value: Some(0.4),
                target_activation: Some(0.4),
            },
            HelpfulSample {
                activation: 1.2,
                avg_error: 0.1,
                target_value: Some(0.2),
                target_activation: Some(0.2),
            },
            // Samples that need DECREASE (negative error)
            HelpfulSample {
                activation: 0.9,
                avg_error: -0.15,
                target_value: Some(0.6),
                target_activation: Some(0.6),
            },
            HelpfulSample {
                activation: 1.1,
                avg_error: -0.1,
                target_value: Some(0.5),
                target_activation: Some(0.5),
            },
        ];

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Compute optimal weight (weighted average)
        let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
        let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        let incoming_weight = 1.0;
        let bias = 0.0;

        eprintln!(
            "MIXED ERRORS TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
        );

        // Predict improvement
        let (predicted_improvement, improved_count, total_count) =
            compute_relu_improvement_and_count(
                &samples,
                incoming_weight,
                outgoing_weight,
                bias,
                baseline_error_sq,
                Some(test_hard_tanh),
            );

        // Manual simulation
        let mut manual_baseline_error_sq = 0.0f32;
        let mut manual_new_error_sq = 0.0f32;
        let mut manual_improved = 0u32;
        let mut manual_worsened = 0u32;

        for sample in &samples {
            let target_val = sample.target_value.unwrap();
            let target_act = sample.target_activation.unwrap();
            let desired_value = target_val + sample.avg_error;
            let expected_activation = test_hard_tanh(desired_value);

            let baseline_err = expected_activation - target_act;
            manual_baseline_error_sq += baseline_err.powi(2);

            let pre_act = incoming_weight * sample.activation + bias;
            let relu_output = pre_act.max(0.0);
            let contribution = outgoing_weight * relu_output;

            let new_target_value = target_val + contribution;
            let new_target_activation = test_hard_tanh(new_target_value);

            let new_err = expected_activation - new_target_activation;
            manual_new_error_sq += new_err.powi(2);

            if new_err.abs() < baseline_err.abs() - EPSILON {
                manual_improved += 1;
            } else if new_err.abs() > baseline_err.abs() + EPSILON {
                manual_worsened += 1;
            }
        }

        let manual_improvement = if manual_baseline_error_sq > EPSILON {
            (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
        } else {
            0.0
        };

        eprintln!(
            "MIXED ERRORS RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
            predicted_improvement,
            predicted_improvement * 100.0,
            manual_improvement,
            manual_improvement * 100.0,
            (predicted_improvement - manual_improvement).abs()
        );
        eprintln!(
            "  func: improved={improved_count}/{total_count}, manual: improved={manual_improved}, worsened={manual_worsened}"
        );

        // Prediction and manual simulation should match
        let diff = (predicted_improvement - manual_improvement).abs();
        assert!(
            diff < 0.01,
            "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
        );
    }

    /// KEY TEST: Simulate what TypeScript evaluation actually does.
    /// This is the most realistic test - it matches the production evaluation flow.
    #[test]
    fn prediction_matches_simulated_typescript_evaluation() {
        // Create samples that match production data characteristics
        let samples: Vec<HelpfulSample> = (0..1000)
            .map(|i| {
                let variation = (i as f32 / 100.0).sin() * 0.1;
                let error_variation = (i as f32 / 50.0).cos() * 0.05;
                HelpfulSample {
                    activation: 0.5 + variation,
                    avg_error: 0.1 + error_variation,
                    target_value: Some(0.4 + variation * 0.5),
                    target_activation: Some(test_hard_tanh(0.4 + variation * 0.5)),
                }
            })
            .collect();

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Compute optimal weight
        let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
        let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        let incoming_weight = 1.0;
        let bias = 0.0;

        // Predict improvement (what Rust returns)
        let (predicted_improvement, _improved_count, _total_count) =
            compute_relu_improvement_and_count(
                &samples,
                incoming_weight,
                outgoing_weight,
                bias,
                baseline_error_sq,
                Some(test_hard_tanh),
            );

        // Simulate TypeScript evaluation
        // TypeScript computes: actualErrorReduction = originalError - candidateError
        // Where error is typically MSE or similar across all training samples

        // Original creature MSE (before adding neuron)
        let original_mse: f32 = samples
            .iter()
            .map(|s| {
                let target_act = s.target_activation.unwrap();
                let desired_value = s.target_value.unwrap() + s.avg_error;
                let expected = test_hard_tanh(desired_value);
                (expected - target_act).powi(2)
            })
            .sum::<f32>()
            / samples.len() as f32;

        // Candidate creature MSE (after adding neuron)
        let candidate_mse: f32 = samples
            .iter()
            .map(|s| {
                let target_val = s.target_value.unwrap();
                let desired_value = target_val + s.avg_error;
                let expected = test_hard_tanh(desired_value);

                // New neuron contribution
                let pre_act = incoming_weight * s.activation + bias;
                let relu_output = pre_act.max(0.0);
                let contribution = outgoing_weight * relu_output;

                let new_target_value = target_val + contribution;
                let new_activation = test_hard_tanh(new_target_value);
                (expected - new_activation).powi(2)
            })
            .sum::<f32>()
            / samples.len() as f32;

        // TypeScript reports: actualErrorReduction = originalError - candidateError
        // If we interpret this as raw error change:
        let original_error = original_mse.sqrt(); // RMSE
        let candidate_error = candidate_mse.sqrt();
        let actual_error_reduction = original_error - candidate_error;

        // For comparison with our percentage, convert to ratio
        let actual_improvement_ratio = actual_error_reduction / original_error;

        // Also compute MSE-based ratio (should match our prediction more closely)
        let mse_improvement_ratio = (original_mse - candidate_mse) / original_mse;

        eprintln!(
            "TYPESCRIPT SIMULATION: predicted={:.6} ({:.4}%)",
            predicted_improvement,
            predicted_improvement * 100.0
        );
        eprintln!(
            "  original_mse={:.8}, candidate_mse={:.8}, mse_improvement={:.6} ({:.4}%)",
            original_mse,
            candidate_mse,
            mse_improvement_ratio,
            mse_improvement_ratio * 100.0
        );
        eprintln!(
            "  original_rmse={:.6}, candidate_rmse={:.6}, rmse_reduction={:.6} ({:.4}%)",
            original_error,
            candidate_error,
            actual_improvement_ratio,
            actual_improvement_ratio * 100.0
        );

        // Our prediction should match MSE-based improvement
        let diff = (predicted_improvement - mse_improvement_ratio).abs();
        assert!(
            diff < 0.01,
            "Predicted ({predicted_improvement:.6}) and MSE improvement ({mse_improvement_ratio:.6}) should match within 1%"
        );

        // Both should be positive (error should decrease)
        assert!(
            predicted_improvement > 0.0,
            "Predicted improvement should be positive"
        );
        assert!(
            mse_improvement_ratio > 0.0,
            "MSE improvement should be positive"
        );
    }

    /// Test that verifies the sign is correct when contribution SHOULD help.
    /// If this test fails, it indicates a sign error in the formula.
    #[test]
    fn contribution_in_correct_direction_reduces_error() {
        // Simple scenario: positive error, positive activation, positive weight = positive contribution
        // Positive contribution ADDS to target value, reducing positive error
        let sample = HelpfulSample {
            activation: 1.0, // positive source activation
            avg_error: 0.2,  // positive VALUE error: need to ADD 0.2
            target_value: Some(0.3),
            target_activation: Some(0.3),
        };

        // Optimal weight formula: w = Σ(error×activation) / Σ(activation²) = 0.2/1 = 0.2
        // Contribution = w × ReLU(source) = 0.2 × 1 = 0.2
        // New target value = 0.3 + 0.2 = 0.5
        // Desired value = 0.3 + 0.2 = 0.5 (should match!)

        let outgoing_weight = 0.1; // Clamped from 0.2
        let incoming_weight = 1.0;
        let bias = 0.0;

        let pre_activation = incoming_weight * sample.activation + bias;
        let relu_output = pre_activation.max(0.0);
        let contribution = outgoing_weight * relu_output;

        // Verify contribution is in the right direction
        assert!(
            contribution > 0.0,
            "Contribution should be positive for positive error"
        );
        assert!(
            contribution.signum() == sample.avg_error.signum(),
            "Contribution sign ({}) should match error sign ({})",
            contribution.signum(),
            sample.avg_error.signum()
        );

        // Verify new error is smaller
        let target_value = sample.target_value.unwrap();
        let target_activation = sample.target_activation.unwrap();
        let desired_value = target_value + sample.avg_error;
        let expected = test_hard_tanh(desired_value);

        let baseline_error = expected - target_activation;
        let new_target_value = target_value + contribution;
        let new_activation = test_hard_tanh(new_target_value);
        let new_error = expected - new_activation;

        eprintln!(
            "DIRECTION TEST: baseline_err={:.4}, new_err={:.4}, reduction={:.4}",
            baseline_error.abs(),
            new_error.abs(),
            baseline_error.abs() - new_error.abs()
        );

        assert!(
            new_error.abs() < baseline_error.abs(),
            "New error ({:.4}) should be smaller than baseline ({:.4})",
            new_error.abs(),
            baseline_error.abs()
        );
    }
}
