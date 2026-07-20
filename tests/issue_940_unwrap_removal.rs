//! Issue #940: Verify production code handles edge cases gracefully (no panics)
//! after replacing `unwrap()` calls with proper error handling.
//!
//! These tests exercise the public API to confirm that functions previously
//! using `unwrap()` on Option values now handle None gracefully — either by
//! skipping samples, returning defaults, or using safe fallbacks.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts in test data

use neat_ai_discovery::analysis::detection::correlated_error::{
    correlated_errors_to_coordinated_candidates, detect_correlated_error_patterns,
};
use neat_ai_discovery::analysis::detection::output_squash_mismatch::detect_output_squash_mismatches;
use neat_ai_discovery::analysis::ensemble_scoring::apply_ensemble_scoring;
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::scoring::confidence::compute_confidence_metrics;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson,
    SynapseJson,
};

// =============================================================================
// confidence.rs — t_critical_95 table lookup
// =============================================================================

/// Issue #940: Verify confidence metrics computation handles edge cases without panic.
#[test]
fn test_confidence_metrics_no_panic_empty_samples() {
    let metrics = compute_confidence_metrics(&[], 0.1, None);
    assert_eq!(metrics.prediction_confidence, 0.0);
}

/// Issue #940: Verify confidence interval with single sample does not panic.
#[test]
fn test_confidence_metrics_single_sample_no_panic() {
    let samples = vec![HelpfulSample {
        activation: 0.5,
        avg_error: 0.1,
        target_value: None,
        target_activation: None,
    }];
    let metrics = compute_confidence_metrics(&samples, 0.05, Some(0.5));
    assert!(metrics.prediction_confidence.is_finite());
    let [lower, upper] = metrics.expected_score_gain_confidence_interval;
    assert!(lower.is_finite() && upper.is_finite());
}

// =============================================================================
// ensemble_scoring.rs — combine_agreeing_candidates
// =============================================================================

/// Issue #940: Verify ensemble scoring handles single candidate without panic.
#[test]
fn test_ensemble_scoring_single_candidate_no_panic() {
    let candidates = vec![CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: "output-0".to_string(),
            bias: 0.1,
        }],
        expected_creature_score_gain: 0.05,
        comment: Some("test module: bias_drift".to_string()),
    }];

    let tracker = ModuleOutcomeTracker::new();
    let result = apply_ensemble_scoring(candidates, &tracker);
    assert_eq!(result.candidates.len(), 1);
    assert_eq!(result.single_module_candidates, 1);
}

/// Issue #940: Verify ensemble scoring handles multiple agreeing candidates without panic.
#[test]
fn test_ensemble_scoring_agreeing_candidates_no_panic() {
    let candidates = vec![
        CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            operations: vec![CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: "output-0".to_string(),
                bias: 0.1,
            }],
            expected_creature_score_gain: 0.05,
            comment: Some("test module: bias_drift".to_string()),
        },
        CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            operations: vec![CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: "output-0".to_string(),
                bias: 0.15,
            }],
            expected_creature_score_gain: 0.03,
            comment: Some("test module: output_bias".to_string()),
        },
    ];

    let tracker = ModuleOutcomeTracker::new();
    let result = apply_ensemble_scoring(candidates, &tracker);
    // Should combine into one ensemble candidate
    assert_eq!(result.candidates.len(), 1);
    assert_eq!(result.ensemble_candidates, 1);
    assert!(result.candidates[0].expected_creature_score_gain > 0.0);
}

// =============================================================================
// correlated_error.rs — detect_correlated_error_patterns
// =============================================================================

/// Issue #940: Verify correlated error detection with minimal creature (no panic).
#[test]
fn test_correlated_error_no_panic_with_minimal_data() {
    let creature = CreatureJson {
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
                squash: "TANH".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 1.0,
            synapse_type: None,
        }],
        input: 1,
        output: 1,
    };

    // Only one output — should return empty (nothing to correlate)
    let groups =
        detect_correlated_error_patterns(&creature, &Vec::<(String, Vec<DiscoverRecord>)>::new());
    assert!(groups.is_empty());
}

/// Issue #940: Verify correlated error candidate conversion with empty groups (no panic).
#[test]
fn test_correlated_error_to_candidates_empty_groups_no_panic() {
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
        input: 0,
        output: 1,
    };

    let candidates = correlated_errors_to_coordinated_candidates(&[], &creature);
    assert!(candidates.is_empty());
}

// =============================================================================
// output_squash_mismatch.rs — detect_output_squash_mismatches
// =============================================================================

/// Issue #940: Verify output squash mismatch detection with records missing
/// pre-activation values (value = None) — should not panic.
#[test]
fn test_output_squash_mismatch_no_panic_missing_values() {
    let output_neurons = vec![("output-0".to_string(), "HARD_TANH".to_string(), 0.0_f32)];

    // Records with value = None (missing pre-activation)
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "output-0".to_string(),
            value: None,
            activation: (i as f32 / 50.0) * 2.0 - 1.0,
            errors: vec![0.1],
        })
        .collect();

    let neuron_records = vec![("output-0".to_string(), records)];

    // Should not panic — records without values should be filtered out
    let candidates = detect_output_squash_mismatches(&output_neurons, &neuron_records);
    // May or may not find candidates depending on detection strategies,
    // but must not panic
    assert!(
        candidates.iter().all(|c| c.confidence.is_finite()),
        "All confidence values should be finite"
    );
}
