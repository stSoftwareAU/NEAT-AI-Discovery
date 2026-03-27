//! Tests for Issue #941: Extract common patterns from detection modules into shared helpers.
//!
//! Verifies that the new shared helper utilities produce correct results for
//! the common patterns used across detection modules.
//!
//! ## TDD Plan
//! 1. `compute_activation_stats` — mean, variance, std dev from records
//! 2. `compute_activation_range` — min/max activation from records
//! 3. `compute_mean_abs_activation` — mean absolute activation from records
//! 4. `sort_candidates_by_score_gain` — sort coordinated candidates descending
//! 5. `weighted_confidence` — multi-factor confidence scoring with floor/ceiling

use neat_ai_discovery::CoordinatedStructuralCandidateJson;
use neat_ai_discovery::analysis::detection::helpers::{
    ConfidenceFactor, compute_activation_range, compute_activation_stats,
    compute_mean_abs_activation, sort_candidates_by_score_gain, weighted_confidence,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: create a `DiscoverRecord` with the given activation.
fn make_record(activation: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: 0,
        neuron_uuid: "test".to_string(),
        value: Some(activation),
        activation,
        errors: vec![0.01],
    }
}

// ============================================================================
// compute_activation_stats tests
// ============================================================================

/// Typical input: positive activations with known mean and variance.
#[test]
fn test_activation_stats_typical() {
    let records = vec![
        make_record(1.0),
        make_record(2.0),
        make_record(3.0),
        make_record(4.0),
        make_record(5.0),
    ];
    let stats = compute_activation_stats(&records);
    // Mean = 3.0
    assert!((stats.mean - 3.0).abs() < 1e-6);
    // Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5 = 10/5 = 2.0
    assert!((stats.variance - 2.0).abs() < 1e-6);
    // Std dev = sqrt(2.0) ≈ 1.4142
    assert!((stats.std_dev - 2.0_f32.sqrt()).abs() < 1e-5);
}

/// Identical activations produce zero variance.
#[test]
fn test_activation_stats_constant() {
    let records = vec![make_record(0.5); 10];
    let stats = compute_activation_stats(&records);
    assert!((stats.mean - 0.5).abs() < 1e-6);
    assert!(stats.variance < 1e-10);
    assert!(stats.std_dev < 1e-5);
}

/// Empty records produce zeroed stats.
#[test]
fn test_activation_stats_empty() {
    let records: Vec<DiscoverRecord> = vec![];
    let stats = compute_activation_stats(&records);
    assert!(stats.mean.abs() < 1e-10);
    assert!(stats.variance.abs() < 1e-10);
    assert!(stats.std_dev.abs() < 1e-10);
}

/// Single record produces zero variance.
#[test]
fn test_activation_stats_single() {
    let records = vec![make_record(42.0)];
    let stats = compute_activation_stats(&records);
    assert!((stats.mean - 42.0).abs() < 1e-6);
    assert!(stats.variance < 1e-10);
    assert!(stats.std_dev < 1e-5);
}

/// Negative activations are handled correctly.
#[test]
fn test_activation_stats_negative_values() {
    let records = vec![make_record(-2.0), make_record(-4.0)];
    let stats = compute_activation_stats(&records);
    assert!((stats.mean - (-3.0)).abs() < 1e-6);
    // Variance = ((−2−(−3))^2 + (−4−(−3))^2) / 2 = (1+1)/2 = 1.0
    assert!((stats.variance - 1.0).abs() < 1e-6);
}

// ============================================================================
// compute_activation_range tests
// ============================================================================

/// Typical input: known min and max.
#[test]
fn test_activation_range_typical() {
    let records = vec![
        make_record(0.1),
        make_record(-0.5),
        make_record(0.8),
        make_record(0.3),
    ];
    let range = compute_activation_range(&records);
    assert!((range.min - (-0.5)).abs() < 1e-6);
    assert!((range.max - 0.8).abs() < 1e-6);
    assert!((range.span - 1.3).abs() < 1e-5);
}

/// Constant activations produce zero span.
#[test]
fn test_activation_range_constant() {
    let records = vec![make_record(0.5); 5];
    let range = compute_activation_range(&records);
    assert!((range.min - 0.5).abs() < 1e-6);
    assert!((range.max - 0.5).abs() < 1e-6);
    assert!(range.span.abs() < 1e-10);
}

/// Empty records return default range (infinity / neg-infinity pattern).
#[test]
fn test_activation_range_empty() {
    let records: Vec<DiscoverRecord> = vec![];
    let range = compute_activation_range(&records);
    assert!(
        range.span <= 0.0,
        "empty records should have non-positive span"
    );
}

// ============================================================================
// compute_mean_abs_activation tests
// ============================================================================

/// Typical input with mixed positive/negative values.
#[test]
fn test_mean_abs_activation_mixed() {
    let records = vec![
        make_record(1.0),
        make_record(-1.0),
        make_record(3.0),
        make_record(-3.0),
    ];
    let result = compute_mean_abs_activation(&records);
    // (1 + 1 + 3 + 3) / 4 = 2.0
    assert!((result - 2.0).abs() < 1e-6);
}

/// All zero activations produce zero mean abs.
#[test]
fn test_mean_abs_activation_zeros() {
    let records = vec![make_record(0.0); 5];
    let result = compute_mean_abs_activation(&records);
    assert!(result.abs() < 1e-10);
}

/// Empty records produce zero.
#[test]
fn test_mean_abs_activation_empty() {
    let records: Vec<DiscoverRecord> = vec![];
    let result = compute_mean_abs_activation(&records);
    assert!(result.abs() < 1e-10);
}

// ============================================================================
// sort_candidates_by_score_gain tests
// ============================================================================

/// Candidates are sorted by `expected_creature_score_gain` descending.
#[test]
fn test_sort_candidates_descending() {
    let mut candidates = vec![
        CoordinatedStructuralCandidateJson {
            operations: vec![],
            expected_creature_score_gain: 0.001,
            comment: Some("low".to_string()),
        },
        CoordinatedStructuralCandidateJson {
            operations: vec![],
            expected_creature_score_gain: 0.010,
            comment: Some("high".to_string()),
        },
        CoordinatedStructuralCandidateJson {
            operations: vec![],
            expected_creature_score_gain: 0.005,
            comment: Some("mid".to_string()),
        },
    ];

    sort_candidates_by_score_gain(&mut candidates);

    assert_eq!(candidates[0].comment.as_deref(), Some("high"));
    assert_eq!(candidates[1].comment.as_deref(), Some("mid"));
    assert_eq!(candidates[2].comment.as_deref(), Some("low"));
}

/// Empty slice is a no-op.
#[test]
fn test_sort_candidates_empty() {
    let mut candidates: Vec<CoordinatedStructuralCandidateJson> = vec![];
    sort_candidates_by_score_gain(&mut candidates);
    assert!(candidates.is_empty());
}

/// Single element is already sorted.
#[test]
fn test_sort_candidates_single() {
    let mut candidates = vec![CoordinatedStructuralCandidateJson {
        operations: vec![],
        expected_creature_score_gain: 0.005,
        comment: Some("only".to_string()),
    }];

    sort_candidates_by_score_gain(&mut candidates);
    assert_eq!(candidates.len(), 1);
}

// ============================================================================
// weighted_confidence tests
// ============================================================================

/// Single factor produces correctly scaled confidence.
#[test]
fn test_weighted_confidence_single_factor() {
    let factors = vec![ConfidenceFactor {
        value: 0.5,
        weight: 1.0,
    }];
    // raw = 0.5, floor=0.5, ceiling=1.0 → 0.5 + 0.5 * 0.5 = 0.75
    let result = weighted_confidence(&factors, 0.5, 1.0);
    assert!((result - 0.75).abs() < 1e-6);
}

/// Multiple weighted factors produce correct result.
#[test]
fn test_weighted_confidence_multiple_factors() {
    let factors = vec![
        ConfidenceFactor {
            value: 1.0,
            weight: 0.4,
        },
        ConfidenceFactor {
            value: 0.5,
            weight: 0.4,
        },
        ConfidenceFactor {
            value: 0.0,
            weight: 0.2,
        },
    ];
    // raw = 1.0*0.4 + 0.5*0.4 + 0.0*0.2 = 0.4 + 0.2 + 0.0 = 0.6
    // floor=0.5, ceiling=1.0 → 0.5 + 0.6 * 0.5 = 0.8
    let result = weighted_confidence(&factors, 0.5, 1.0);
    assert!((result - 0.80).abs() < 1e-6);
}

/// All factors at maximum produce ceiling.
#[test]
fn test_weighted_confidence_all_max() {
    let factors = vec![
        ConfidenceFactor {
            value: 1.0,
            weight: 0.5,
        },
        ConfidenceFactor {
            value: 1.0,
            weight: 0.5,
        },
    ];
    // raw = 1.0, floor=0.5, ceiling=1.0 → 0.5 + 1.0 * 0.5 = 1.0
    let result = weighted_confidence(&factors, 0.5, 1.0);
    assert!((result - 1.0).abs() < 1e-6);
}

/// All factors at zero produce floor.
#[test]
fn test_weighted_confidence_all_zero() {
    let factors = vec![
        ConfidenceFactor {
            value: 0.0,
            weight: 0.5,
        },
        ConfidenceFactor {
            value: 0.0,
            weight: 0.5,
        },
    ];
    // raw = 0.0, floor=0.3, ceiling=0.9 → 0.3 + 0.0 * 0.6 = 0.3
    let result = weighted_confidence(&factors, 0.3, 0.9);
    assert!((result - 0.3).abs() < 1e-6);
}

/// Empty factors produce floor.
#[test]
fn test_weighted_confidence_empty_factors() {
    let factors: Vec<ConfidenceFactor> = vec![];
    let result = weighted_confidence(&factors, 0.5, 1.0);
    assert!((result - 0.5).abs() < 1e-6);
}

/// Values are clamped to [0, 1] before computation.
#[test]
fn test_weighted_confidence_clamped_values() {
    let factors = vec![ConfidenceFactor {
        value: 2.0, // exceeds 1.0, should be clamped
        weight: 1.0,
    }];
    // raw = min(2.0, 1.0) * 1.0 = 1.0
    // floor=0.5, ceiling=1.0 → 0.5 + 1.0 * 0.5 = 1.0
    let result = weighted_confidence(&factors, 0.5, 1.0);
    assert!((result - 1.0).abs() < 1e-6);
}

/// Custom floor and ceiling work correctly.
#[test]
fn test_weighted_confidence_custom_range() {
    let factors = vec![ConfidenceFactor {
        value: 0.5,
        weight: 1.0,
    }];
    // raw = 0.5, floor=0.3, ceiling=0.9 → 0.3 + 0.5 * 0.6 = 0.6
    let result = weighted_confidence(&factors, 0.3, 0.9);
    assert!((result - 0.6).abs() < 1e-6);
}
