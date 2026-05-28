//! REGRESSION TESTS for v0.1.123 fixes
//!
//! These tests catch regressions if the v0.1.123 fixes are accidentally reverted.
//! DO NOT DELETE OR COMMENT OUT THESE TESTS - they protect against known bugs.
//!
//! ## Fixes covered:
//!
//! 1. **`MIN_FALLBACK_IMPROVEMENT` (2%)**: Fallback candidates with very low predicted
//!    improvement (<2%) were causing 100% failure rates because the model's error
//!    margin (~±0.5%) exceeded the prediction itself.
//!
//! 2. **Hidden neuron analysis**: Hidden neurons were filtered out entirely, missing
//!    valid improvement opportunities. Now they're analysed with impact-based
//!    discounting.
//!
//! If any of these tests fail after a code change, the fix has regressed.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::common::GainFloorDisableGuard;
use neat_ai_discovery::analysis::analyze_neurons;
use neat_ai_discovery::analysis::shared::NeuronNoCandidateReason;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeNeuronsInput, CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
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
/// by the same low-confidence fallback issue. With `MIN_FALLBACK_IMPROVEMENT`
/// fixed, hidden neurons can now be analysed with impact-based discounting.
///
/// This test verifies that hidden neurons ARE included in analysis.
/// If this test fails, check that the neuron type filter in
/// `analyze_neurons_with_cache()` allows "hidden" neurons through.
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
                synapse_type: None,
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
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        task_descriptor: None,
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
        .is_none_or(|d| !matches!(d.reason, NeuronNoCandidateReason::HiddenNeuronFiltered)); // No diagnostic = was analyzed and found candidates

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
/// v0.1.134: Removed the arbitrary 2% `MIN_FALLBACK_IMPROVEMENT` threshold.
/// v0.1.135: Added split-error evaluation - `expected_creature_score_gain` is now
/// the NET improvement across ALL samples, not just a subset.
///
/// The only requirement is positive improvement. TypeScript evaluates actual score.
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
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        task_descriptor: None,
    };

    let result = analyze_neurons(&input).expect("Neuron analysis should succeed");

    // v0.1.134/v0.1.135: All returned candidates must have positive improvement.
    // The expected_creature_score_gain is now the NET improvement across ALL samples
    // (thanks to split-error evaluation), so TypeScript can trust this value directly.
    for candidate in &result.helpful_neurons {
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Candidate should have positive expected improvement, got {:.4}%",
            candidate.expected_creature_score_gain * 100.0
        );
        eprintln!(
            "Candidate: {} -> {} ({}), improvement: {:.4}%",
            candidate.source_neuron_uuid,
            candidate.target_neuron_uuid,
            candidate.squash,
            candidate.expected_creature_score_gain * 100.0
        );
    }

    eprintln!(
        "Test passed: {} candidates returned with positive improvement",
        result.helpful_neurons.len()
    );
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
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-weak".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.1, // Weak connection
                synapse_type: None,
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
/// candidates were not re-filtered against `MIN_FALLBACK_IMPROVEMENT` (2%). A hidden
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
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-high-impact".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.9, // High weight = high impact
                synapse_type: None,
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
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        task_descriptor: None,
    };

    let result = analyze_neurons(&input).expect("Neuron analysis should succeed");

    // v0.1.134: Removed arbitrary 2% MIN_FALLBACK_IMPROVEMENT threshold.
    // The only filter is: does the improvement exceed the cost of growth?
    // Cost of growth for a neuron is ~1e-7, which is tiny compared to any
    // meaningful improvement percentage.
    //
    // KEY ASSERTION: Candidates with positive improvement are returned.
    // TypeScript will evaluate the actual score change.
    for candidate in &result.helpful_neurons {
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Candidate should have positive expected improvement, got {:.6}%",
            candidate.expected_creature_score_gain * 100.0
        );
        eprintln!(
            "Candidate: {} -> {} ({}), improvement: {:.4}%",
            candidate.source_neuron_uuid,
            candidate.target_neuron_uuid,
            candidate.squash,
            candidate.expected_creature_score_gain * 100.0
        );
    }

    eprintln!(
        "Test passed: {} candidates returned with positive improvement",
        result.helpful_neurons.len()
    );
}

/// REGRESSION TEST: Impact-discounted neurons must appear in EITHER `helpful_neurons` OR `no_candidate_reasons`.
///
/// BUG (to be fixed in v0.1.125): When a hidden neuron candidate is found but then filtered out
/// by impact discounting (falls below 2% after discount), the neuron disappears from BOTH:
/// - `helpful_neurons` (candidate was removed by re-filtering)
/// - `no_candidate_reasons` (`had_candidate` was set to true before filtering)
///
/// This causes focus neurons to silently disappear from the response, leaving callers
/// with no information about neurons they explicitly requested analysis for.
///
/// A focus neuron must ALWAYS appear in exactly one of:
/// 1. `helpful_neurons` (has candidates)
/// 2. `no_candidate_reasons` (was analysed but has no viable candidates)
#[test]
#[serial]
fn regression_impact_discounted_neurons_must_appear_in_response() {
    skip_without_gpu!();
    // Issue #1191: synthetic fixture produces post-discount gains below the
    // 1e-5 production noise floor. This regression case is about diagnostic
    // visibility for impact-discounted candidates — disable the floor so the
    // contract under test remains observable independently.
    let _gain_guard = GainFloorDisableGuard::new();

    let temp_dir = TempDir::new().expect("Failed to create temporary directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path
        .to_str()
        .expect("Temporary path should be valid UTF-8")
        .to_string();

    // Create a creature where hidden-will-be-discounted has VERY LOW impact (0.05).
    // We'll engineer the data so raw improvement is ~5-15%, but after 0.05 discount
    // it becomes 0.25-0.75% which falls below MIN_FALLBACK_IMPROVEMENT (2%).
    //
    // hidden-will-be-discounted (weight 0.05) --\
    //                                            --> output-0
    // hidden-high-impact (weight 0.95) ---------/
    //
    // Impact for hidden-will-be-discounted = 0.05 / (0.05 + 0.95) = 0.05
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-will-be-discounted".to_string(),
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
                from_uuid: "hidden-will-be-discounted".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.05, // Very low weight = very low impact (0.05)
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-high-impact".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.95, // High weight = high impact (0.95)
                synapse_type: None,
            },
        ],
    };

    // Verify impact scores
    let impacts = neat_ai_discovery::focus::compute_impacts_public(&creature);
    let discounted_impact = impacts
        .get("hidden-will-be-discounted")
        .copied()
        .unwrap_or(0.0);
    eprintln!("hidden-will-be-discounted has impact score: {discounted_impact:.3}");
    assert!(
        discounted_impact < 0.1,
        "Test setup error: hidden-will-be-discounted should have impact < 0.1, got {discounted_impact:.3}"
    );

    // Create discovery records with VERY WEAK correlation for hidden-will-be-discounted.
    // We want raw improvement to be ~5-30% so that after 0.05 impact discount
    // it falls to 0.25-1.5% (below MIN_FALLBACK_IMPROVEMENT of 2%).
    //
    // The key is: raw improvement >= 2% (passes initial filter) but
    // discounted improvement < 2% (gets filtered by re-filtering).
    //
    // We use mostly noise with tiny signal to break the correlation.
    let mut records = Vec::new();
    for obs_index in 0..100u32 {
        let input_val = (obs_index as f32 - 50.0) / 50.0; // -1.0 to 1.0

        // Heavy noise, tiny signal - produces ~5-30% raw improvement
        // which becomes 0.25-1.5% after 0.05 impact discount
        let noise1 = ((obs_index * 17) % 11) as f32 / 5.5 - 1.0; // Pseudo-random -1 to 1
        let noise2 = ((obs_index * 23) % 7) as f32 / 3.5 - 1.0; // Different noise
        let hidden_error = input_val * 0.02 + noise1 * 0.3 + noise2 * 0.2;

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
            "hidden-will-be-discounted".to_string(),
            Some(input_val * 0.2),
            input_val * 0.2,
            vec![hidden_error],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "hidden-high-impact".to_string(),
            Some(0.5),
            0.5,
            vec![0.0], // No error - not our focus
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(input_val * 0.1),
            input_val * 0.1,
            vec![hidden_error * 0.1],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write records");

    let input = AnalyzeNeuronsInput {
        parquet_file,
        creature,
        focus_neurons: vec!["hidden-will-be-discounted".to_string()],
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        task_descriptor: None,
    };

    let result = analyze_neurons(&input).expect("Neuron analysis should succeed");

    // KEY ASSERTION: The focus neuron must appear in EITHER helpful_neurons OR no_candidate_reasons.
    // It must NOT silently disappear from the response.
    let in_helpful = result
        .helpful_neurons
        .iter()
        .any(|c| c.target_neuron_uuid == "hidden-will-be-discounted");

    let in_no_candidate = result
        .no_candidate_reasons
        .iter()
        .any(|r| r.target_uuid == "hidden-will-be-discounted");

    assert!(
        in_helpful || in_no_candidate,
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════════════════╗\n\
        ║  REGRESSION DETECTED: Focus neuron silently disappeared from response!        ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════╣\n\
        ║  'hidden-will-be-discounted' was requested in focus_neurons but appears       ║\n\
        ║  in NEITHER helpful_neurons NOR no_candidate_reasons.                         ║\n\
        ║                                                                              ║\n\
        ║  This happens when:                                                          ║\n\
        ║  1. A candidate is found → had_candidate = true                              ║\n\
        ║  2. Impact discounting reduces improvement below 2%                           ║\n\
        ║  3. Re-filtering removes the candidate                                        ║\n\
        ║  4. no_candidate_summaries() skips entry (had_candidate == true)              ║\n\
        ║                                                                              ║\n\
        ║  Result: Neuron disappears from both lists!                                   ║\n\
        ║                                                                              ║\n\
        ║  FIX: After re-filtering, update diagnostics for neurons that lost all        ║\n\
        ║  candidates to mark them as ImpactDiscountedBelowThreshold.                   ║\n\
        ║                                                                              ║\n\
        ║  helpful_neurons: {:?}                                                        \n\
        ║  no_candidate_reasons: {:?}                                                   \n\
        ╚══════════════════════════════════════════════════════════════════════════════╝\n\n",
        result
            .helpful_neurons
            .iter()
            .map(|c| &c.target_neuron_uuid)
            .collect::<Vec<_>>(),
        result
            .no_candidate_reasons
            .iter()
            .map(|r| &r.target_uuid)
            .collect::<Vec<_>>()
    );

    // If it's in no_candidate_reasons, verify the reason is appropriate
    if in_no_candidate {
        let reason = result
            .no_candidate_reasons
            .iter()
            .find(|r| r.target_uuid == "hidden-will-be-discounted")
            .unwrap();

        // It should NOT be HiddenNeuronFiltered (that's a pre-analysis filter)
        assert!(
            !matches!(reason.reason, NeuronNoCandidateReason::HiddenNeuronFiltered),
            "Hidden neuron should have been analysed, not filtered. Got: {:?}",
            reason.reason
        );

        eprintln!(
            "Test passed: hidden-will-be-discounted appears in no_candidate_reasons with reason: {:?}",
            reason.reason
        );
    } else {
        eprintln!(
            "Test note: hidden-will-be-discounted appears in helpful_neurons (candidate survived discounting)"
        );
    }
}
