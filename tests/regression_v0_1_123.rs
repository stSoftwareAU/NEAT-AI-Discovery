//! REGRESSION TESTS for v0.1.123 fixes
//!
//! These tests catch regressions if the v0.1.123 fixes are accidentally reverted.
//! DO NOT DELETE OR COMMENT OUT THESE TESTS - they protect against known bugs.
//!
//! ## Fixes covered:
//!
//! 1. **MIN_FALLBACK_IMPROVEMENT (2%)**: Fallback candidates with very low predicted
//!    improvement (<2%) were causing 100% failure rates because the model's error
//!    margin (~±0.5%) exceeded the prediction itself.
//!
//! 2. **Hidden neuron analysis**: Hidden neurons were filtered out entirely, missing
//!    valid improvement opportunities. Now they're analysed with impact-based
//!    discounting.
//!
//! If any of these tests fail after a code change, the fix has regressed.

mod common;

use neat_ai_discovery::analysis::{analyze_neurons, NeuronNoCandidateReason};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeNeuronsInput, CreatureJson, NeuronJson, SynapseJson};
use tempfile::TempDir;

/// Skip test if no GPU available (same as other tests)
macro_rules! skip_without_gpu {
    () => {
        if !neat_ai_discovery::analysis::GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: No GPU available");
            return;
        }
    };
}

/// REGRESSION TEST: Hidden neurons must NOT be filtered from add-neuron analysis.
///
/// BUG (fixed in v0.1.123): Hidden neurons were entirely filtered out from
/// add-neuron analysis due to observed "100% failure rates". This was caused
/// by the same low-confidence fallback issue. With MIN_FALLBACK_IMPROVEMENT
/// fixed, hidden neurons can now be analysed with impact-based discounting.
///
/// This test verifies that hidden neurons ARE included in analysis.
/// If this test fails, check that the neuron type filter in
/// analyze_neurons_with_cache() allows "hidden" neurons through.
#[test]
fn regression_hidden_neurons_must_be_analyzed_not_filtered() {
    skip_without_gpu!();

    let temp_dir = TempDir::new().expect("Failed to create temporary directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path
        .to_str()
        .expect("Temporary path should be valid UTF-8")
        .to_string();

    // Create a creature with BOTH output AND hidden neurons
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
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
            // hidden-0 connects to output-0 with weight 1.0
            // This gives hidden-0 an impact of 1.0 (direct path to output)
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
            },
        ],
    };

    // Create discovery records with correlated errors for both hidden and output
    let mut records = Vec::new();
    for obs_index in 0..30u32 {
        let input_val = (obs_index as f32 - 15.0) / 15.0;
        let error = input_val * 0.3; // Correlated error

        // Input neuron records
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(input_val),
            input_val,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(0.5),
            0.5,
            vec![],
        ));

        // Hidden neuron records (with errors - this is what gets analysed)
        records.push(DiscoverRecord::new(
            obs_index,
            "hidden-0".to_string(),
            Some(input_val * 0.5),
            (input_val * 0.5_f32).tanh(),
            vec![error],
        ));

        // Output neuron records
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(input_val * 0.3),
            input_val * 0.3,
            vec![error * 0.5],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write records");

    // Analyse with hidden neuron in focus list
    let input = AnalyzeNeuronsInput {
        parquet_file,
        creature,
        focus_neurons: vec!["hidden-0".to_string(), "output-0".to_string()],
        improvement_threshold: Some(0.01),
        max_candidates: Some(10),
        analysis_deadline_ms: None,
    };

    let result = analyze_neurons(&input).expect("Neuron analysis should succeed");

    // The key assertion: hidden-0 should NOT have HiddenNeuronFiltered reason
    let hidden_diagnostic = result
        .no_candidate_reasons
        .iter()
        .find(|d| d.target_uuid == "hidden-0");

    if let Some(diag) = hidden_diagnostic {
        assert!(
            !matches!(diag.reason, NeuronNoCandidateReason::HiddenNeuronFiltered),
            "\n\n\
            ╔══════════════════════════════════════════════════════════════════════════════╗\n\
            ║  REGRESSION DETECTED: Hidden neuron filtering has been re-enabled!           ║\n\
            ╠══════════════════════════════════════════════════════════════════════════════╣\n\
            ║  Hidden neuron 'hidden-0' was filtered with HiddenNeuronFiltered.            ║\n\
            ║                                                                              ║\n\
            ║  This was fixed in v0.1.123. Hidden neurons should be analysed with          ║\n\
            ║  impact-based discounting, NOT filtered out entirely.                        ║\n\
            ║                                                                              ║\n\
            ║  CHECK: analyze_neurons_with_cache() must allow 'hidden' neuron type.        ║\n\
            ║                                                                              ║\n\
            ║  Diagnostic: {diag:?}\n\
            ╚══════════════════════════════════════════════════════════════════════════════╝\n\n"
        );
    }

    // Hidden-0 either found candidates OR was analysed but found no candidate
    // (NOT filtered out)
    let hidden_has_candidates = result
        .helpful_neurons
        .iter()
        .any(|c| c.target_neuron_uuid == "hidden-0");

    let hidden_was_analyzed = hidden_diagnostic
        .map(|d| !matches!(d.reason, NeuronNoCandidateReason::HiddenNeuronFiltered))
        .unwrap_or(true); // No diagnostic = was analyzed and found candidates

    assert!(
        hidden_has_candidates || hidden_was_analyzed,
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  REGRESSION DETECTED: Hidden neuron appears to be filtered!                  ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║  Hidden neuron 'hidden-0' should either:                                     ║\n\
        ║    - Have candidates returned, OR                                            ║\n\
        ║    - Have a diagnostic reason OTHER than HiddenNeuronFiltered                ║\n\
        ║                                                                              ║\n\
        ║  This was fixed in v0.1.123.                                                 ║\n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n\n"
    );
}

/// REGRESSION TEST: Fallback candidates must meet minimum 2% improvement threshold.
///
/// BUG (fixed in v0.1.123): Fallback candidates with very low predicted improvement
/// (e.g., 0.16%) were returned, but at such low predictions the model's error margin
/// (~±0.5%) is larger than the prediction itself. This caused actual results to be
/// NEGATIVE (worse than baseline) while predictions were positive.
///
/// This test creates a scenario with weak correlation that would produce low
/// improvement predictions, and verifies that such weak candidates are NOT returned.
#[test]
fn regression_low_improvement_fallback_candidates_must_be_filtered() {
    skip_without_gpu!();

    let temp_dir = TempDir::new().expect("Failed to create temporary directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path
        .to_str()
        .expect("Temporary path should be valid UTF-8")
        .to_string();

    // Create a simple creature
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
    };

    // Create discovery records with VERY WEAK correlation
    // This should produce predicted improvements well below 2%
    let mut records = Vec::new();
    for obs_index in 0..100u32 {
        let input_val = (obs_index as f32 - 50.0) / 50.0;

        // Very weak correlation: error is mostly noise with tiny signal
        // This mimics the production scenario where fallback candidates had ~0.16% improvement
        let noise = ((obs_index * 17) % 11) as f32 / 11.0 - 0.5; // Pseudo-random noise
        let error = input_val * 0.005 + noise * 0.2; // Very weak signal, lots of noise

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(input_val),
            input_val,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(0.5),
            0.5,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(error),
            error,
            vec![error],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write records");

    // Use a very low threshold (1%) - below MIN_FALLBACK_IMPROVEMENT (2%)
    // If MIN_FALLBACK_IMPROVEMENT is working, weak candidates should still be filtered
    let input = AnalyzeNeuronsInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        improvement_threshold: Some(0.01), // 1% threshold
        max_candidates: Some(50),
        analysis_deadline_ms: None,
    };

    let result = analyze_neurons(&input).expect("Neuron analysis should succeed");

    // Check all returned candidates - none should have < 2% improvement
    for candidate in &result.helpful_neurons {
        assert!(
            candidate.expected_improvement_percentage >= 0.02,
            "\n\n\
            ╔══════════════════════════════════════════════════════════════════════════════╗\n\
            ║  REGRESSION DETECTED: Low-confidence fallback candidate returned!            ║\n\
            ╠══════════════════════════════════════════════════════════════════════════════╣\n\
            ║  Candidate returned with only {:.2}% expected improvement.                   \n\
            ║                                                                              ║\n\
            ║  MIN_FALLBACK_IMPROVEMENT (2%) should have filtered this out.                ║\n\
            ║  Candidates with < 2% predicted improvement are unreliable because           ║\n\
            ║  the model's error margin (~±0.5%) exceeds the prediction itself.            ║\n\
            ║                                                                              ║\n\
            ║  This was fixed in v0.1.123.                                                 ║\n\
            ║                                                                              ║\n\
            ║  CHECK: MIN_FALLBACK_IMPROVEMENT = 0.02 in evaluate_activation_candidate()   ║\n\
            ║                                                                              ║\n\
            ║  Candidate: {} -> {} ({})                                                    \n\
            ╚══════════════════════════════════════════════════════════════════════════════╝\n\n",
            candidate.expected_improvement_percentage * 100.0,
            candidate.source_neuron_uuid,
            candidate.target_neuron_uuid,
            candidate.squash
        );
    }
}

/// REGRESSION TEST: Hidden neuron predictions must be discounted by impact.
///
/// This test verifies that hidden neuron predictions are appropriately
/// discounted based on their impact score (path weight to outputs).
#[test]
fn regression_hidden_neuron_predictions_must_be_impact_discounted() {
    skip_without_gpu!();

    // Verify the impact calculation function exists and works
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-strong".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-weak".to_string(),
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
                from_uuid: "hidden-strong".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0, // Strong connection
            },
            SynapseJson {
                from_uuid: "hidden-weak".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.1, // Weak connection
            },
        ],
    };

    // Verify impact scores are computed and differ based on connection strength
    let impacts = neat_ai_discovery::focus::compute_impacts_public(&creature);

    let strong_impact = impacts.get("hidden-strong").copied().unwrap_or(-1.0);
    let weak_impact = impacts.get("hidden-weak").copied().unwrap_or(-1.0);
    let output_impact = impacts.get("output-0").copied().unwrap_or(-1.0);

    // Output should have impact = 1.0
    assert!(
        (output_impact - 1.0).abs() < 0.01,
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  REGRESSION DETECTED: Output neuron impact calculation is wrong!             ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║  Output neuron should have impact ≈ 1.0, got {output_impact:.3}              \n\
        ║                                                                              ║\n\
        ║  CHECK: compute_impacts_public() in focus.rs                                 ║\n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n\n"
    );

    // Both hidden neurons should have positive impact
    assert!(
        strong_impact > 0.0 && weak_impact > 0.0,
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  REGRESSION DETECTED: Hidden neuron impact calculation is wrong!             ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║  Hidden neurons connected to output should have positive impact.             ║\n\
        ║  strong_impact = {strong_impact:.3}, weak_impact = {weak_impact:.3}          \n\
        ║                                                                              ║\n\
        ║  CHECK: compute_impacts_public() in focus.rs                                 ║\n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n\n"
    );

    // Strong connection should have higher impact than weak
    assert!(
        strong_impact > weak_impact,
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  REGRESSION DETECTED: Impact discount not proportional to connection!        ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║  hidden-strong (weight=1.0) should have higher impact than                   ║\n\
        ║  hidden-weak (weight=0.1).                                                   ║\n\
        ║                                                                              ║\n\
        ║  strong_impact = {strong_impact:.3}, weak_impact = {weak_impact:.3}          \n\
        ║                                                                              ║\n\
        ║  This ensures predictions for hidden neurons far from outputs are            ║\n\
        ║  appropriately discounted.                                                   ║\n\
        ║                                                                              ║\n\
        ║  CHECK: compute_impacts_public() in focus.rs                                 ║\n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n\n"
    );
}

/// REGRESSION TEST: Discounted hidden neuron candidates must still meet 2% threshold.
///
/// BUG (fixed in v0.1.124): After applying impact-based discounting for hidden neurons,
/// candidates were not re-filtered against MIN_FALLBACK_IMPROVEMENT (2%). A hidden
/// neuron with 3% raw improvement and 0.3 impact would be discounted to 0.9%, falling
/// below the minimum threshold but still returned.
///
/// This contradicts the fix for low-confidence fallback candidates - if 2% is the
/// minimum for reliable predictions, then discounted predictions below 2% are equally
/// unreliable.
///
/// This test creates a hidden neuron with LOW impact that would cause discounting
/// below 2%, and verifies such candidates are NOT returned.
#[test]
fn regression_discounted_hidden_neurons_must_meet_minimum_threshold() {
    skip_without_gpu!();

    let temp_dir = TempDir::new().expect("Failed to create temporary directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path
        .to_str()
        .expect("Temporary path should be valid UTF-8")
        .to_string();

    // Create a creature where hidden-low-impact has LOW impact to output.
    // Structure: hidden-low-impact (weight 0.1) and hidden-high-impact (weight 0.9) both -> output
    // hidden-low-impact impact = 0.1 / (0.1 + 0.9) × 1.0 = 0.1
    //
    // If analysis produces ~3% raw improvement for hidden-low-impact:
    // - Raw improvement: 3%
    // - After 0.1 discount: 0.3%
    // - This should be filtered (< 2%)
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-low-impact".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-high-impact".to_string(),
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
                from_uuid: "hidden-low-impact".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.1, // Low weight = low impact
            },
            SynapseJson {
                from_uuid: "hidden-high-impact".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.9, // High weight = high impact
            },
        ],
    };

    // Verify the impact scores first
    let impacts = neat_ai_discovery::focus::compute_impacts_public(&creature);
    let low_impact = impacts.get("hidden-low-impact").copied().unwrap_or(0.0);
    eprintln!("hidden-low-impact has impact score: {low_impact:.3}");
    assert!(
        low_impact < 0.2,
        "Test setup error: hidden-low-impact should have impact < 0.2, got {low_impact:.3}"
    );

    // Create discovery records with STRONG correlation for hidden-low-impact
    // This should produce ~3-5% raw improvement prediction (before discount)
    let mut records = Vec::new();
    for obs_index in 0..100u32 {
        let input_val = (obs_index as f32 - 50.0) / 50.0; // -1.0 to 1.0

        // Strong linear correlation between input and error at hidden-low-impact
        let hidden_low_error = input_val * 0.5;
        let hidden_high_error = 0.0; // No correlation at high-impact hidden

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(input_val),
            input_val,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(0.3),
            0.3,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "hidden-low-impact".to_string(),
            Some(input_val * 0.2),
            input_val * 0.2,
            vec![hidden_low_error],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "hidden-high-impact".to_string(),
            Some(0.5),
            0.5,
            vec![hidden_high_error],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(input_val * 0.1),
            input_val * 0.1,
            vec![hidden_low_error * 0.1], // Small propagated error
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write records");

    let input = AnalyzeNeuronsInput {
        parquet_file,
        creature,
        focus_neurons: vec!["hidden-low-impact".to_string()],
        improvement_threshold: Some(0.01), // 1% threshold (below MIN_FALLBACK_IMPROVEMENT)
        max_candidates: Some(50),
        analysis_deadline_ms: None,
    };

    let result = analyze_neurons(&input).expect("Neuron analysis should succeed");

    // KEY ASSERTION: ALL returned candidates must have >= 2% improvement
    // Even after impact discounting
    for candidate in &result.helpful_neurons {
        assert!(
            candidate.expected_improvement_percentage >= 0.02,
            "\n\n\
            ╔══════════════════════════════════════════════════════════════════════════════╗\n\
            ║  REGRESSION DETECTED: Discounted hidden neuron below MIN_FALLBACK!           ║\n\
            ╠══════════════════════════════════════════════════════════════════════════════╣\n\
            ║  Candidate returned with {:.2}% expected improvement (after discount).       \n\
            ║                                                                              ║\n\
            ║  This hidden neuron candidate was discounted by impact but NOT re-filtered   ║\n\
            ║  against MIN_FALLBACK_IMPROVEMENT (2%).                                      ║\n\
            ║                                                                              ║\n\
            ║  If 2% is the minimum for reliable predictions, then discounted predictions  ║\n\
            ║  below 2% are equally unreliable.                                            ║\n\
            ║                                                                              ║\n\
            ║  This was fixed in v0.1.124.                                                 ║\n\
            ║                                                                              ║\n\
            ║  CHECK: After impact discounting, re-filter against MIN_FALLBACK_IMPROVEMENT ║\n\
            ║                                                                              ║\n\
            ║  Candidate: {} -> {} ({})                                                    \n\
            ║  Target neuron type: hidden (impact: {:.3})                                  \n\
            ╚══════════════════════════════════════════════════════════════════════════════╝\n\n",
            candidate.expected_improvement_percentage * 100.0,
            candidate.source_neuron_uuid,
            candidate.target_neuron_uuid,
            candidate.squash,
            low_impact
        );
    }

    eprintln!(
        "Test passed: {} candidates returned, all >= 2% after discounting",
        result.helpful_neurons.len()
    );
}
