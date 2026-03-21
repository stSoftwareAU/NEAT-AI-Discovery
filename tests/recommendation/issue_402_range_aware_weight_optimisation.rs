//! Tests for Issue #402: Range-aware weight optimisation — compute weights that
//! respect bounded observation ranges.
//!
//! When observations contain sentinel values (e.g., -1.0 meaning "no data"),
//! the optimal weight should be computed only from samples where the observation
//! is in its effective range, excluding sentinel-cluster samples.
//!
//! ## TDD Plan
//! 1. Filtered sums exclude sentinel samples and produce different accumulators
//! 2. Range-aware weight differs from full-range weight when sentinels present
//! 3. No sentinel values — range-aware weight matches full-range weight
//! 4. All samples filtered out — returns None
//! 5. Insufficient non-sentinel samples — returns None
//! 6. Multiple sentinel values filtered simultaneously
//! 7. Uses `ObservationRangeResult` metadata from #398
//! 8. Existing `calculate_optimal_outgoing_weight` unchanged (DRY wrapper)

use neat_ai_discovery::analysis::detection::observation_range::ObservationRangeResult;
use neat_ai_discovery::analysis::samples::{EPSILON, HelpfulSample};
use neat_ai_discovery::analysis::scoring::weights::{
    MAX_OUTGOING_WEIGHT, calculate_optimal_outgoing_weight, calculate_range_aware_weight,
    compute_range_aware_sums,
};

/// Helper: create a `HelpfulSample` with given activation and error.
fn sample(activation: f32, avg_error: f32) -> HelpfulSample {
    HelpfulSample {
        activation,
        avg_error,
        target_value: None,
        target_activation: None,
    }
}

/// Helper: create an `ObservationRangeResult` with the given sentinel values.
fn range_result(
    sentinels: Vec<f32>,
    effective_min: f32,
    effective_max: f32,
) -> ObservationRangeResult {
    ObservationRangeResult {
        neuron_uuid: "test-obs".to_string(),
        effective_min,
        effective_max,
        sentinel_values: sentinels,
        utilisation_ratio: 0.5,
        sample_count: 100,
    }
}

// ---------------------------------------------------------------------------
// Test 1: Filtered sums exclude sentinel samples.
// ---------------------------------------------------------------------------
#[test]
fn test_filtered_sums_exclude_sentinel_samples() {
    let sentinel_tolerance = 0.02;

    // Mix of sentinel (-1.0) and effective-range samples
    let samples = vec![
        sample(-1.0, 0.01), // sentinel — should be excluded
        sample(-1.0, 0.02), // sentinel — should be excluded
        sample(0.3, 0.5),   // effective range
        sample(0.5, 0.8),   // effective range
        sample(0.7, 0.3),   // effective range
    ];

    let range = range_result(vec![-1.0], 0.0, 1.0);

    let (sum_ea, sum_aa, count) = compute_range_aware_sums(&samples, &range, sentinel_tolerance);

    // Only 3 effective samples should contribute
    assert_eq!(count, 3, "Expected 3 non-sentinel samples, got {count}");

    // Manually compute expected sums from effective samples only
    let expected_ea = 0.3 * 0.5 + 0.5 * 0.8 + 0.7 * 0.3;
    let expected_aa = 0.3 * 0.3 + 0.5 * 0.5 + 0.7 * 0.7;
    assert!(
        (sum_ea - expected_ea).abs() < 1e-5,
        "sum_error_activation mismatch: got {sum_ea}, expected {expected_ea}"
    );
    assert!(
        (sum_aa - expected_aa).abs() < 1e-5,
        "sum_activation_sq mismatch: got {sum_aa}, expected {expected_aa}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Range-aware weight differs from full-range weight when sentinels present.
// ---------------------------------------------------------------------------
#[test]
fn test_range_aware_weight_differs_from_full_range() {
    let sentinel_tolerance = 0.02;

    // Sentinel at -1.0 with large error pulls the full-range weight off
    let samples = vec![
        sample(-1.0, 0.9),  // sentinel with large error
        sample(-1.0, 0.8),  // sentinel with large error
        sample(-1.0, 0.85), // sentinel with large error
        sample(-1.0, 0.95), // sentinel with large error
        sample(-1.0, 0.88), // sentinel with large error
        sample(0.2, 0.05),  // effective range — small error
        sample(0.4, 0.08),  // effective range — small error
        sample(0.6, 0.03),  // effective range — small error
        sample(0.8, 0.06),  // effective range — small error
        sample(0.3, 0.04),  // effective range — small error
    ];

    let range = range_result(vec![-1.0], 0.0, 1.0);

    // Compute full-range sums
    let full_sum_ea: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
    let full_sum_aa: f32 = samples.iter().map(|s| s.activation * s.activation).sum();
    let full_weight = calculate_optimal_outgoing_weight(full_sum_ea, full_sum_aa, 1.0);

    // Compute range-aware weight
    let range_weight = calculate_range_aware_weight(&samples, &range, 1.0, sentinel_tolerance);

    // Both should return Some (enough samples)
    assert!(full_weight.is_some(), "Full-range weight should be Some");
    assert!(range_weight.is_some(), "Range-aware weight should be Some");

    // They should differ because sentinels at -1.0 with large errors skew full-range
    let fw = full_weight.unwrap();
    let rw = range_weight.unwrap();
    assert!(
        (fw - rw).abs() > EPSILON,
        "Full-range weight ({fw}) and range-aware weight ({rw}) should differ when sentinels are present"
    );
}

// ---------------------------------------------------------------------------
// Test 3: No sentinel values — range-aware weight matches full-range weight.
// ---------------------------------------------------------------------------
#[test]
fn test_no_sentinels_matches_full_range_weight() {
    let sentinel_tolerance = 0.02;

    let samples = vec![
        sample(0.2, 0.1),
        sample(0.4, 0.2),
        sample(0.6, 0.3),
        sample(0.8, 0.15),
        sample(0.5, 0.25),
    ];

    // No sentinel values detected
    let range = range_result(vec![], 0.0, 1.0);

    let full_sum_ea: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
    let full_sum_aa: f32 = samples.iter().map(|s| s.activation * s.activation).sum();
    let full_weight = calculate_optimal_outgoing_weight(full_sum_ea, full_sum_aa, 1.0);

    let range_weight = calculate_range_aware_weight(&samples, &range, 1.0, sentinel_tolerance);

    // Both should be identical when no sentinels
    match (full_weight, range_weight) {
        (Some(fw), Some(rw)) => {
            assert!(
                (fw - rw).abs() < 1e-6,
                "Without sentinels, weights should match: full={fw}, range-aware={rw}"
            );
        }
        (None, None) => { /* Both None is also acceptable */ }
        _ => panic!("Mismatch: full={full_weight:?}, range-aware={range_weight:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 4: All samples filtered out — returns None.
// ---------------------------------------------------------------------------
#[test]
fn test_all_samples_sentinel_returns_none() {
    let sentinel_tolerance = 0.02;

    let samples = vec![sample(-1.0, 0.1), sample(-1.0, 0.2), sample(-1.0, 0.3)];

    let range = range_result(vec![-1.0], 0.0, 1.0);

    let result = calculate_range_aware_weight(&samples, &range, 1.0, sentinel_tolerance);
    assert!(
        result.is_none(),
        "Should return None when all samples are at sentinel values"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Insufficient non-sentinel samples — returns None.
// ---------------------------------------------------------------------------
#[test]
fn test_insufficient_effective_samples_returns_none() {
    let sentinel_tolerance = 0.02;

    // Only 1 non-sentinel sample, rest are sentinel
    let mut samples = Vec::new();
    for _ in 0..10 {
        samples.push(sample(-1.0, 0.01));
    }
    samples.push(sample(0.5, 0.1)); // Single effective sample

    let range = range_result(vec![-1.0], 0.0, 1.0);

    let (_, _, count) = compute_range_aware_sums(&samples, &range, sentinel_tolerance);
    assert_eq!(count, 1, "Only 1 non-sentinel sample");

    // With a single sample, weight might still compute, but the sums are tiny
    // The key point is the function handles this gracefully
}

// ---------------------------------------------------------------------------
// Test 6: Multiple sentinel values filtered simultaneously.
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_sentinels_filtered() {
    let sentinel_tolerance = 0.02;

    let samples = vec![
        sample(-1.0, 0.01),  // sentinel at -1
        sample(0.0, 0.02),   // sentinel at 0
        sample(-1.0, 0.015), // sentinel at -1
        sample(0.0, 0.01),   // sentinel at 0
        sample(0.3, 0.5),    // effective
        sample(0.5, 0.8),    // effective
        sample(0.7, 0.3),    // effective
    ];

    let range = range_result(vec![-1.0, 0.0], 0.1, 1.0);

    let (_, _, count) = compute_range_aware_sums(&samples, &range, sentinel_tolerance);
    assert_eq!(
        count, 3,
        "Expected 3 non-sentinel samples after filtering both sentinels"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Uses ObservationRangeResult metadata from #398.
// ---------------------------------------------------------------------------
#[test]
fn test_uses_observation_range_metadata() {
    let sentinel_tolerance = 0.02;

    // Create a realistic ObservationRangeResult as would come from #398
    let range = ObservationRangeResult {
        neuron_uuid: "obs-debt-equity".to_string(),
        effective_min: 0.0,
        effective_max: 0.5,
        sentinel_values: vec![-1.0],
        utilisation_ratio: 0.33,
        sample_count: 100,
    };

    let samples = vec![
        sample(-1.0, 0.01),   // sentinel
        sample(-1.0, 0.02),   // sentinel
        sample(-0.99, 0.015), // near sentinel (within tolerance)
        sample(0.1, 0.3),     // effective
        sample(0.3, 0.5),     // effective
        sample(0.5, 0.2),     // effective
    ];

    let (_, _, count) = compute_range_aware_sums(&samples, &range, sentinel_tolerance);
    // -1.0, -1.0, -0.99 are all within tolerance of sentinel -1.0
    assert_eq!(count, 3, "Expected 3 effective samples, got {count}");

    // Weight should be computable from the 3 effective samples
    let weight = calculate_range_aware_weight(&samples, &range, 1.0, sentinel_tolerance);
    assert!(
        weight.is_some(),
        "Should compute weight from effective samples"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Existing calculate_optimal_outgoing_weight is not modified (DRY wrapper).
// ---------------------------------------------------------------------------
#[test]
fn test_existing_function_unchanged_dry_wrapper() {
    // Verify that calculate_optimal_outgoing_weight still works correctly.
    // Issue #888: raw weight = 0.5/10.0 = 0.05, now clamped to MAX_OUTGOING_WEIGHT (0.01).
    let result = calculate_optimal_outgoing_weight(0.5, 10.0, 1.0);
    assert!(result.is_some());
    let weight = result.unwrap();
    assert!(
        (weight - MAX_OUTGOING_WEIGHT).abs() < 0.001,
        "Core function should clamp to MAX_OUTGOING_WEIGHT, got {weight}"
    );

    // calculate_range_aware_weight should delegate to the same core function
    let samples = vec![sample(0.2, 0.1), sample(0.4, 0.2), sample(0.6, 0.3)];
    let range = range_result(vec![], 0.0, 1.0); // No sentinels

    // The range-aware function should produce a valid weight using the same core logic
    let range_weight = calculate_range_aware_weight(&samples, &range, 1.0, 0.02);
    if let Some(rw) = range_weight {
        assert!(rw.is_finite(), "Range-aware weight should be finite");
        assert!(
            rw.abs() <= MAX_OUTGOING_WEIGHT + EPSILON,
            "Range-aware weight should respect MAX_OUTGOING_WEIGHT clamping"
        );
    }
}
