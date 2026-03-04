//! Tests for Issue #751: Improve bimodal detection with cluster coherence validation.
//!
//! Adds a secondary validation step to bimodal detection: after detecting a gap,
//! verify that both resulting clusters have activation variance meaningfully lower
//! than the overall variance (i.e., each cluster is internally coherent). This
//! follows the Hartigan Dip Test principle without requiring the full statistical
//! machinery.
//!
//! ## TDD Plan
//! 1. Uniform distribution → no bimodality detected
//! 2. Bimodal with equal-sized modes → detected
//! 3. Bimodal with unequal-sized modes → detected
//! 4. Unimodal with one outlier → should NOT be flagged as bimodal
//! 5. Heavy-tailed unimodal → should NOT be flagged as bimodal

use neat_ai_discovery::analysis::detection::bimodal_neuron::detect_bimodal_neurons;
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: create a DiscoverRecord with a specific pre-activation value.
fn make_record(neuron_uuid: &str, obs_index: u32, value: Option<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value,
        activation: value.unwrap_or(0.0).tanh(),
        errors: vec![0.01],
    }
}

/// Test 1: Uniform distribution → no bimodality detected.
#[test]
fn test_uniform_distribution_not_bimodal() {
    let neurons = vec![("uniform".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = -5.0 + 10.0 * (i as f32 / 99.0);
            make_record("uniform", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("uniform".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Uniform distribution should not be flagged as bimodal"
    );
}

/// Test 2: Bimodal with equal-sized modes → detected.
///
/// Two well-separated clusters of equal size should be detected as bimodal.
/// Each cluster is internally tight (low variance) while the overall
/// distribution has high variance.
#[test]
fn test_bimodal_equal_modes_detected() {
    let neurons = vec![("equal-modes".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = if i < 50 {
                -3.0 + 0.1 * (i as f32 * 0.1).sin()
            } else {
                3.0 + 0.1 * ((i - 50) as f32 * 0.1).sin()
            };
            make_record("equal-modes", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("equal-modes".to_string(), records)]);

    assert!(
        !candidates.is_empty(),
        "Bimodal distribution with equal-sized modes should be detected"
    );
    let c = &candidates[0];
    assert_eq!(c.lower_mode_count, 50);
    assert_eq!(c.upper_mode_count, 50);
}

/// Test 3: Bimodal with unequal-sized modes → detected.
///
/// Two well-separated clusters with 70/30 split should still be detected.
/// Both clusters are internally coherent despite unequal sizes.
#[test]
fn test_bimodal_unequal_modes_detected() {
    let neurons = vec![("unequal-modes".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = if i < 70 {
                -2.0 + 0.05 * (i as f32 * 0.1).sin()
            } else {
                4.0 + 0.05 * ((i - 70) as f32 * 0.1).sin()
            };
            make_record("unequal-modes", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("unequal-modes".to_string(), records)]);

    assert!(
        !candidates.is_empty(),
        "Bimodal distribution with unequal-sized modes should be detected"
    );
}

/// Test 4: Unimodal with scattered outlier group → should NOT be flagged as bimodal.
///
/// 80 points tightly clustered in [0, 1], then 20 points spread widely across
/// [10, 50]. The gap creates a large ratio, but the outlier group is NOT a
/// coherent cluster — its internal variance is nearly as high as the overall
/// variance. Cluster coherence validation should reject this.
#[test]
fn test_unimodal_with_scattered_outlier_group_not_bimodal() {
    let neurons = vec![("scattered-outlier".to_string(), "TANH".to_string(), 0.0)];
    let mut records: Vec<DiscoverRecord> = Vec::new();

    // Main cluster: 80 points uniformly in [0, 1]
    for i in 0..80 {
        let value = i as f32 / 79.0;
        records.push(make_record("scattered-outlier", i, Some(value)));
    }

    // Scattered outlier group: 20 points uniformly in [10, 50]
    // This creates a large gap (10 - 1 = 9) but the "cluster" has high
    // internal variance (range of 40) — not a coherent mode.
    for i in 80..100 {
        let value = 10.0 + 40.0 * ((i - 80) as f32 / 19.0);
        records.push(make_record("scattered-outlier", i, Some(value)));
    }

    let candidates =
        detect_bimodal_neurons(&neurons, &[("scattered-outlier".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Scattered outlier group should NOT be flagged as bimodal — \
         the outlier 'cluster' has variance comparable to overall distribution"
    );
}

/// Test 5: Heavy-tailed skewed distribution → should NOT be flagged as bimodal.
///
/// A distribution where 80 points are in [0, 2] and 20 points follow a
/// geometric progression from 15 to 300. The large gap between 2 and 15
/// triggers gap-based detection, but the tail is not a coherent cluster.
#[test]
fn test_heavy_tail_skewed_not_bimodal() {
    let neurons = vec![("heavy-skew".to_string(), "TANH".to_string(), 0.0)];
    let mut records: Vec<DiscoverRecord> = Vec::new();

    // Main body: 80 points uniformly in [0, 2]
    for i in 0..80 {
        let value = 2.0 * (i as f32 / 79.0);
        records.push(make_record("heavy-skew", i, Some(value)));
    }

    // Heavy tail: 20 points spread in [15, 300] (geometric-like spacing)
    for i in 80..100 {
        let t = (i - 80) as f32 / 19.0; // 0.0 to 1.0
        let value = 15.0 + 285.0 * t * t; // quadratic spacing for heavy tail
        records.push(make_record("heavy-skew", i, Some(value)));
    }

    let candidates = detect_bimodal_neurons(&neurons, &[("heavy-skew".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Heavy-tailed skewed distribution should NOT be flagged as bimodal — \
         the tail 'cluster' has very high internal variance"
    );
}
