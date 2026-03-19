//! Tests for Issue #395: Discover bounded ranges of observations and hidden neurons.
//!
//! Observations are normalised to -1…1, but -1 often represents "null" or "invalid"
//! rather than a meaningful low value. This module detects neurons where a significant
//! cluster of values sits at a boundary (e.g., -1) separate from the "useful" range,
//! and recommends adding a gating neuron so that the null region does not negatively
//! impact the creature's score.
//!
//! ## TDD Plan
//! 1. Detect observation with bimodal distribution: cluster at -1 (null) and useful range
//! 2. Verify useful range bounds are computed correctly
//! 3. Verify neurons without boundary clustering are not flagged
//! 4. Test hidden neurons with boundary clustering
//! 5. Test edge cases: insufficient samples, uniform distribution, all-same values
//! 6. Test coordinated structural candidate conversion

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::common::{hidden, make_creature, neuron, output, record, synapse};
use neat_ai_discovery::analysis::detection::bounded_range::{
    BoundedRangeCandidate, bounded_range_to_coordinated_candidates, detect_bounded_range_neurons,
};
use neat_ai_discovery::types::DiscoverRecord;

// ---------------------------------------------------------------------------
// Test 1: Observation with cluster at -1 (null sentinel) is detected.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_observation_with_null_cluster_at_minus_one() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    // 40% of samples at -1 (null sentinel), 60% in useful range [0.1, 0.8]
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records.push(record("input-obs", i, -1.0, None));
    }
    for i in 40..100 {
        let useful_val = 0.1 + 0.7 * ((i - 40) as f32 / 60.0);
        records.push(record("input-obs", i, useful_val, None));
    }

    let candidates = detect_bounded_range_neurons(&creature, &[("input-obs".to_string(), records)]);

    assert_eq!(
        candidates.len(),
        1,
        "Should detect one bounded-range neuron"
    );
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "input-obs");
    assert!(
        c.boundary_fraction >= 0.3,
        "Boundary fraction should be >= 0.3, got {}",
        c.boundary_fraction
    );
    assert!(
        c.useful_range_min > -0.95,
        "Useful range min should exclude the -1 cluster, got {}",
        c.useful_range_min
    );
}

// ---------------------------------------------------------------------------
// Test 2: Observation with cluster at +1 (null sentinel) is detected.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_observation_with_null_cluster_at_plus_one() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    // 35% at +1 (null), 65% in range [-0.5, 0.3]
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..35 {
        records.push(record("input-obs", i, 1.0, None));
    }
    for i in 35..100 {
        let useful_val = -0.5 + 0.8 * ((i - 35) as f32 / 65.0);
        records.push(record("input-obs", i, useful_val, None));
    }

    let candidates = detect_bounded_range_neurons(&creature, &[("input-obs".to_string(), records)]);

    assert_eq!(candidates.len(), 1, "Should detect boundary cluster at +1");
    let c = &candidates[0];
    assert!(
        c.useful_range_max < 0.95,
        "Useful range max should exclude the +1 cluster, got {}",
        c.useful_range_max
    );
}

// ---------------------------------------------------------------------------
// Test 3: Neuron with uniform distribution is NOT flagged.
// ---------------------------------------------------------------------------
#[test]
fn test_does_not_flag_uniform_distribution() {
    let creature = make_creature(
        vec![
            neuron("input-uniform", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-uniform", "output-1", 0.5)],
    );

    // Evenly spread across [-1, 1]
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let val = -1.0 + 2.0 * (i as f32 / 99.0);
            record("input-uniform", i, val, None)
        })
        .collect();

    let candidates =
        detect_bounded_range_neurons(&creature, &[("input-uniform".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Uniform distribution should not be flagged"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Hidden neuron with boundary clustering is detected.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_hidden_neuron_with_boundary_cluster() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("hidden-bounded", "TANH"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-bounded", 0.5),
            synapse("hidden-bounded", "output-1", 0.3),
        ],
    );

    // Hidden neuron: 45% at -1.0 (TANH saturation as sentinel), 55% in [-0.3, 0.6]
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..45 {
        records.push(record("hidden-bounded", i, -1.0, Some(-5.0)));
    }
    for i in 45..100 {
        let useful_val = -0.3 + 0.9 * ((i - 45) as f32 / 55.0);
        records.push(record("hidden-bounded", i, useful_val, Some(useful_val)));
    }

    let candidates =
        detect_bounded_range_neurons(&creature, &[("hidden-bounded".to_string(), records)]);

    assert_eq!(
        candidates.len(),
        1,
        "Should detect hidden neuron with boundary cluster"
    );
    assert_eq!(candidates[0].neuron_uuid, "hidden-bounded");
}

// ---------------------------------------------------------------------------
// Test 5: Insufficient samples → no detection.
// ---------------------------------------------------------------------------
#[test]
fn test_bounded_range_insufficient_samples_not_detected() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    // Only 5 samples (below minimum threshold)
    let records: Vec<DiscoverRecord> = (0..5).map(|i| record("input-obs", i, -1.0, None)).collect();

    let candidates = detect_bounded_range_neurons(&creature, &[("input-obs".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Should not detect with insufficient samples"
    );
}

// ---------------------------------------------------------------------------
// Test 6: All-same values → no detection (no useful range to separate).
// ---------------------------------------------------------------------------
#[test]
fn test_all_same_values_not_detected() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("input-obs", i, 0.5, None))
        .collect();

    let candidates = detect_bounded_range_neurons(&creature, &[("input-obs".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "All-same values should not be flagged (no boundary cluster)"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Cluster at 0 (common null sentinel) is detected.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_cluster_at_zero_sentinel() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    // 50% at 0.0 (null sentinel), 50% in [0.3, 0.9]
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..50 {
        records.push(record("input-obs", i, 0.0, None));
    }
    for i in 50..100 {
        let useful_val = 0.3 + 0.6 * ((i - 50) as f32 / 50.0);
        records.push(record("input-obs", i, useful_val, None));
    }

    let candidates = detect_bounded_range_neurons(&creature, &[("input-obs".to_string(), records)]);

    assert_eq!(candidates.len(), 1, "Should detect cluster at 0 sentinel");
    let c = &candidates[0];
    assert!(
        c.boundary_fraction >= 0.4,
        "Boundary fraction should be >= 0.4, got {}",
        c.boundary_fraction
    );
}

// ---------------------------------------------------------------------------
// Test 8: Coordinated candidate conversion produces valid operations.
// ---------------------------------------------------------------------------
#[test]
fn test_bounded_range_coordinated_candidate_conversion() {
    let candidates = vec![BoundedRangeCandidate {
        neuron_uuid: "input-obs".to_string(),
        boundary_value: -1.0,
        boundary_fraction: 0.4,
        useful_range_min: 0.1,
        useful_range_max: 0.8,
        sample_count: 100,
        detection_confidence: 0.85,
        estimated_improvement: 0.01,
    }];

    let coordinated = bounded_range_to_coordinated_candidates(&candidates);

    assert!(
        !coordinated.is_empty(),
        "Should produce at least one coordinated candidate"
    );
    let c = &coordinated[0];
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
    assert!(
        !c.operations.is_empty(),
        "Should have at least one operation"
    );
    assert!(c.comment.is_some(), "Should include a descriptive comment");
}

// ---------------------------------------------------------------------------
// Test 9: Multiple neurons — only boundary-clustered ones are detected.
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_neurons_only_boundary_clustered_detected() {
    let creature = make_creature(
        vec![
            neuron("input-good", "input", "IDENTITY"),
            neuron("input-bad", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-good", "output-1", 0.5),
            synapse("input-bad", "output-1", 0.3),
        ],
    );

    // input-good: uniform spread — no boundary cluster
    let good_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let val = -0.8 + 1.6 * (i as f32 / 99.0);
            record("input-good", i, val, None)
        })
        .collect();

    // input-bad: 40% at -1 (sentinel), 60% in [0.0, 0.5]
    let mut bad_records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        bad_records.push(record("input-bad", i, -1.0, None));
    }
    for i in 40..100 {
        let useful_val = 0.5 * ((i - 40) as f32 / 60.0);
        bad_records.push(record("input-bad", i, useful_val, None));
    }

    let candidates = detect_bounded_range_neurons(
        &creature,
        &[
            ("input-good".to_string(), good_records),
            ("input-bad".to_string(), bad_records),
        ],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Only the boundary-clustered neuron should be detected"
    );
    assert_eq!(candidates[0].neuron_uuid, "input-bad");
}

// ---------------------------------------------------------------------------
// Test 10: Output neurons are excluded from bounded range detection.
// ---------------------------------------------------------------------------
#[test]
fn test_bounded_range_output_neurons_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            output("output-1", "TANH"),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Output neuron with boundary cluster — should NOT be flagged
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records.push(record("output-1", i, -1.0, Some(-5.0)));
    }
    for i in 40..100 {
        let val = 0.3 * ((i - 40) as f32 / 60.0);
        records.push(record("output-1", i, val, Some(val)));
    }

    let candidates = detect_bounded_range_neurons(&creature, &[("output-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Output neurons should be excluded from detection"
    );
}
