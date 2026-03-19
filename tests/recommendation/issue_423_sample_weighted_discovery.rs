//! Tests for Issue #423: Sample-weighted discovery — prioritise high-error samples.
//!
//! Current discovery treats all samples equally. Samples with high error are more
//! important for improvement but don't receive proportional analysis attention.
//! This module weights samples by error magnitude and focuses analysis on neurons
//! active during high-error samples.
//!
//! ## TDD Plan
//! 1. Test sample importance weighting by absolute error magnitude
//! 2. Test detection of high-error-dominant neurons (neurons that disproportionately
//!    contribute to high-error samples)
//! 3. Test stratified analysis — easy vs hard sample separation
//! 4. Test that balanced-error neurons are NOT flagged
//! 5. Test candidate generation produces valid coordinated candidates
//! 6. Test edge cases: empty records, single sample, all-equal errors
//! 7. Test outlier sample detection (consistently failing samples)
//! 8. Test weighted improvement estimation

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::recommendation::sample_weighted::{
    SampleWeightedConfig, compute_sample_weights, detect_high_error_neurons,
    high_error_neurons_to_coordinated_candidates, stratify_samples,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helper: create records with specified error magnitudes
// =============================================================================

fn make_records(neuron_uuid: &str, errors: &[f32]) -> Vec<DiscoverRecord> {
    errors
        .iter()
        .enumerate()
        .map(|(i, &err)| DiscoverRecord {
            obs_index: i as u32,
            neuron_uuid: neuron_uuid.to_string(),
            value: Some(0.5),
            activation: 0.5,
            errors: vec![err],
        })
        .collect()
}

// =============================================================================
// Test 1: Sample importance weighting by absolute error magnitude
// =============================================================================

/// Samples with higher absolute error should receive higher weights.
#[test]
fn test_sample_weights_proportional_to_error() {
    let errors = vec![0.1, 0.2, 0.5, 1.0, 2.0];
    let records = make_records("neuron-a", &errors);

    let weights = compute_sample_weights(&records);

    assert_eq!(weights.len(), 5);

    // Higher error → higher weight
    for i in 1..weights.len() {
        assert!(
            weights[i] >= weights[i - 1],
            "Weight at index {i} ({}) should be >= weight at index {} ({})",
            weights[i],
            i - 1,
            weights[i - 1]
        );
    }

    // Weights should sum to approximately 1.0 (normalised)
    let sum: f32 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Weights should sum to ~1.0, got {sum}"
    );

    // The highest-error sample should get a noticeably larger weight
    assert!(
        weights[4] > weights[0] * 1.5,
        "Highest error weight ({}) should be significantly larger than lowest ({})",
        weights[4],
        weights[0]
    );
}

/// All-equal errors should produce uniform weights.
#[test]
fn test_uniform_errors_produce_equal_weights() {
    let errors = vec![0.5; 50];
    let records = make_records("neuron-a", &errors);

    let weights = compute_sample_weights(&records);

    assert_eq!(weights.len(), 50);

    let expected_weight = 1.0 / 50.0;
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            (w - expected_weight).abs() < 0.001,
            "Weight at index {i} should be ~{expected_weight}, got {w}"
        );
    }
}

/// Empty records should produce empty weights.
#[test]
fn test_empty_records_produce_empty_weights() {
    let weights = compute_sample_weights(&[]);
    assert!(weights.is_empty());
}

// =============================================================================
// Test 2: Detection of high-error-dominant neurons
// =============================================================================

/// A neuron that has disproportionately high errors on high-error samples
/// should be detected as a candidate.
#[test]
fn test_detects_high_error_neuron() {
    // Neuron with mostly high errors — a problem neuron
    let high_error_records = make_records(
        "problem-neuron",
        &[
            0.8, 0.9, 1.0, 0.7, 0.85, 0.95, 0.6, 0.75, 0.88, 0.92, 0.82, 0.91, 0.87, 0.93, 0.78,
            0.86, 0.94, 0.89, 0.81, 0.96,
        ],
    );

    // Neuron with low errors — a healthy neuron
    let low_error_records = make_records(
        "healthy-neuron",
        &[
            0.01, 0.02, 0.03, 0.01, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.01, 0.02, 0.03, 0.01,
            0.02, 0.01, 0.03, 0.02, 0.01, 0.02,
        ],
    );

    let records = vec![
        ("problem-neuron".to_string(), high_error_records),
        ("healthy-neuron".to_string(), low_error_records),
    ];

    let config = SampleWeightedConfig::default();
    let candidates = detect_high_error_neurons(&records, &config);

    assert!(
        !candidates.is_empty(),
        "Should detect at least one high-error neuron"
    );

    // The problem neuron should be detected
    assert!(
        candidates.iter().any(|c| c.neuron_uuid == "problem-neuron"),
        "Should detect 'problem-neuron' as high-error"
    );

    // The healthy neuron should NOT be detected
    assert!(
        !candidates.iter().any(|c| c.neuron_uuid == "healthy-neuron"),
        "Should NOT detect 'healthy-neuron' as high-error"
    );
}

/// Neurons with uniformly low errors should NOT be flagged.
#[test]
fn test_low_error_neurons_not_flagged() {
    let records = vec![
        (
            "neuron-a".to_string(),
            make_records("neuron-a", &[0.01; 30]),
        ),
        (
            "neuron-b".to_string(),
            make_records("neuron-b", &[0.02; 30]),
        ),
    ];

    let config = SampleWeightedConfig::default();
    let candidates = detect_high_error_neurons(&records, &config);

    assert!(
        candidates.is_empty(),
        "No high-error neurons expected when all errors are low, got {} candidate(s)",
        candidates.len()
    );
}

/// Detection needs a minimum number of samples.
#[test]
fn test_sample_weighted_insufficient_samples_returns_empty() {
    let records = vec![(
        "neuron-a".to_string(),
        make_records("neuron-a", &[0.9, 0.8]),
    )];

    let config = SampleWeightedConfig::default();
    let candidates = detect_high_error_neurons(&records, &config);

    assert!(
        candidates.is_empty(),
        "Should return empty for insufficient samples"
    );
}

// =============================================================================
// Test 3: Stratified analysis — easy vs hard samples
// =============================================================================

/// Stratified analysis should separate samples into easy (low error) and hard
/// (high error) groups based on the median error threshold.
#[test]
fn test_stratify_samples_separates_easy_and_hard() {
    // Mix of easy and hard samples
    let records = make_records(
        "neuron-a",
        &[
            0.01, 0.02, 0.03, 0.04, 0.05, // Easy samples
            0.5, 0.6, 0.7, 0.8, 0.9,
        ], // Hard samples
    );

    let stratified = stratify_samples(&records);

    assert!(
        !stratified.easy_samples.is_empty(),
        "Should have easy samples"
    );
    assert!(
        !stratified.hard_samples.is_empty(),
        "Should have hard samples"
    );

    // Mean error of hard samples should be higher than easy samples
    assert!(
        stratified.hard_mean_error > stratified.easy_mean_error,
        "Hard mean error ({}) should exceed easy mean error ({})",
        stratified.hard_mean_error,
        stratified.easy_mean_error
    );

    // Hard-to-easy ratio should be meaningful
    assert!(
        stratified.hard_to_easy_ratio > 1.0,
        "Hard-to-easy ratio ({}) should be > 1.0",
        stratified.hard_to_easy_ratio
    );
}

/// Stratifying uniform errors should produce similar easy and hard groups.
#[test]
fn test_stratify_uniform_errors_balanced() {
    let records = make_records("neuron-a", &[0.5; 20]);

    let stratified = stratify_samples(&records);

    // With uniform errors, hard-to-easy ratio should be close to 1.0
    assert!(
        stratified.hard_to_easy_ratio < 1.5,
        "Uniform errors should have ratio near 1.0, got {}",
        stratified.hard_to_easy_ratio
    );
}

/// Stratifying empty records should return an empty/default stratification.
#[test]
fn test_stratify_empty_records() {
    let stratified = stratify_samples(&[]);

    assert!(stratified.easy_samples.is_empty());
    assert!(stratified.hard_samples.is_empty());
}

// =============================================================================
// Test 4: Candidate conversion produces valid coordinated candidates
// =============================================================================

/// Detected high-error neurons should convert to valid coordinated candidates
/// with positive expected improvement and appropriate comments.
#[test]
fn test_sample_weighted_candidates_have_positive_improvement() {
    let high_error_records = make_records(
        "problem-neuron",
        &[
            0.8, 0.9, 1.0, 0.7, 0.85, 0.95, 0.6, 0.75, 0.88, 0.92, 0.82, 0.91, 0.87, 0.93, 0.78,
            0.86, 0.94, 0.89, 0.81, 0.96,
        ],
    );

    let records = vec![("problem-neuron".to_string(), high_error_records)];

    let config = SampleWeightedConfig::default();
    let detected = detect_high_error_neurons(&records, &config);

    assert!(!detected.is_empty(), "Should detect candidates");

    let coordinated = high_error_neurons_to_coordinated_candidates(&detected);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    for candidate in &coordinated {
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Expected improvement should be positive, got {}",
            candidate.expected_creature_score_gain
        );
        assert!(
            candidate.comment.is_some(),
            "Candidate should have a comment"
        );
        assert!(
            !candidate.operations.is_empty(),
            "Candidate should have at least one operation"
        );
    }
}

/// Candidates should be sorted by expected improvement (best first).
#[test]
fn test_sample_weighted_candidates_sorted_by_improvement() {
    // Create multiple neurons with varying error levels
    let records = vec![
        (
            "slight-problem".to_string(),
            make_records("slight-problem", &[0.3; 30]),
        ),
        (
            "big-problem".to_string(),
            make_records("big-problem", &[0.9; 30]),
        ),
        (
            "medium-problem".to_string(),
            make_records("medium-problem", &[0.6; 30]),
        ),
    ];

    let config = SampleWeightedConfig::default();
    let detected = detect_high_error_neurons(&records, &config);

    if detected.len() >= 2 {
        let coordinated = high_error_neurons_to_coordinated_candidates(&detected);

        for i in 1..coordinated.len() {
            assert!(
                coordinated[i - 1].expected_creature_score_gain
                    >= coordinated[i].expected_creature_score_gain,
                "Candidates should be sorted by improvement (descending)"
            );
        }
    }
}

// =============================================================================
// Test 5: Weighted improvement estimation
// =============================================================================

/// A neuron with high weighted error should have a higher estimated improvement
/// than one with moderate weighted error.
#[test]
fn test_weighted_improvement_scales_with_error() {
    let records = vec![
        (
            "high-error-neuron".to_string(),
            make_records("high-error-neuron", &[0.9; 30]),
        ),
        (
            "moderate-error-neuron".to_string(),
            make_records("moderate-error-neuron", &[0.4; 30]),
        ),
    ];

    let config = SampleWeightedConfig::default();
    let detected = detect_high_error_neurons(&records, &config);

    let high = detected
        .iter()
        .find(|c| c.neuron_uuid == "high-error-neuron");
    let moderate = detected
        .iter()
        .find(|c| c.neuron_uuid == "moderate-error-neuron");

    if let (Some(h), Some(m)) = (high, moderate) {
        assert!(
            h.estimated_improvement > m.estimated_improvement,
            "High-error neuron improvement ({}) should exceed moderate ({})",
            h.estimated_improvement,
            m.estimated_improvement
        );
    }
}

// =============================================================================
// Test 6: Edge cases
// =============================================================================

/// A single sample should not produce candidates (insufficient for statistics).
#[test]
fn test_single_sample_no_candidates() {
    let records = vec![("neuron-a".to_string(), make_records("neuron-a", &[1.0]))];

    let config = SampleWeightedConfig::default();
    let candidates = detect_high_error_neurons(&records, &config);

    assert!(
        candidates.is_empty(),
        "Single sample should not produce candidates"
    );
}

/// NaN and infinite errors should be safely handled.
#[test]
fn test_non_finite_errors_handled() {
    let mut records_data: Vec<f32> = vec![0.1; 20];
    records_data.push(f32::NAN);
    records_data.push(f32::INFINITY);

    let records = make_records("neuron-a", &records_data);
    let weights = compute_sample_weights(&records);

    // Should not panic and should filter out non-finite values
    assert!(!weights.is_empty());

    // All weights should be finite
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            w.is_finite(),
            "Weight at index {i} should be finite, got {w}"
        );
    }
}

/// Zero errors should produce minimal weights.
#[test]
fn test_zero_errors_produce_minimal_weights() {
    let errors = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let records = make_records("neuron-a", &errors);

    let weights = compute_sample_weights(&records);

    // All weights should be equal (uniform) since all errors are zero
    let expected = 1.0 / weights.len() as f32;
    for &w in &weights {
        assert!(
            (w - expected).abs() < 0.01,
            "All-zero errors should produce uniform weights"
        );
    }
}

// =============================================================================
// Test 7: Detection with mixed-error patterns
// =============================================================================

/// A neuron with bimodal errors (some very low, some very high) should be detected
/// as a high-error candidate based on the high-error subpopulation.
#[test]
fn test_bimodal_error_neuron_detected() {
    let mut errors: Vec<f32> = vec![0.01; 15]; // Easy samples
    errors.extend(vec![0.9; 15]); // Hard samples

    let records = vec![(
        "bimodal-neuron".to_string(),
        make_records("bimodal-neuron", &errors),
    )];

    let config = SampleWeightedConfig::default();
    let candidates = detect_high_error_neurons(&records, &config);

    // The bimodal neuron should still be detected because the high-error
    // subpopulation has significant weighted error
    assert!(
        !candidates.is_empty(),
        "Bimodal error neuron should be detected"
    );
}

// =============================================================================
// Test 8: Config customisation
// =============================================================================

/// Custom config thresholds should affect detection.
#[test]
fn test_custom_config_threshold() {
    let records = vec![("neuron-a".to_string(), make_records("neuron-a", &[0.3; 30]))];

    // Very low threshold should detect more
    let lenient_config = SampleWeightedConfig {
        min_weighted_error: 0.1,
        ..SampleWeightedConfig::default()
    };
    let lenient_candidates = detect_high_error_neurons(&records, &lenient_config);

    // Very high threshold should detect fewer
    let strict_config = SampleWeightedConfig {
        min_weighted_error: 0.95,
        ..SampleWeightedConfig::default()
    };
    let strict_candidates = detect_high_error_neurons(&records, &strict_config);

    assert!(
        lenient_candidates.len() >= strict_candidates.len(),
        "Lenient config should detect >= strict config candidates ({} vs {})",
        lenient_candidates.len(),
        strict_candidates.len()
    );
}
