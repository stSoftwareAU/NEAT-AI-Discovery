//! Tests for Issue #2042: one authoritative sentinel-cluster accept/reject rule.
//!
//! `observation_range` (Issue #398) and `sentinel_gating` (Issue #400) both decide
//! "is this value cluster a sentinel?" from the same evidence: a sufficient gap
//! between the cluster and the useful range **and** lower error variance inside the
//! cluster. The two copies had diverged — `observation_range` had a dead
//! `|| gap >= MIN_GAP` disjunct that made the variance half of the rule a no-op.
//!
//! These tests pin the shared rule by driving both detectors with the same samples
//! and asserting they agree.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::common::{make_creature, neuron, output, synapse};
use neat_ai_discovery::analysis::detection::observation_range::detect_observation_ranges;
use neat_ai_discovery::analysis::detection::sentinel_cluster::{
    assess_sentinel_cluster, compute_error_variance,
};
use neat_ai_discovery::analysis::detection::sentinel_gating::detect_sentinel_gating_candidates;
use neat_ai_discovery::types::DiscoverRecord;

fn record_with_error(
    neuron_uuid: &str,
    obs_index: u32,
    activation: f32,
    error: f32,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: None,
        activation,
        errors: vec![error],
    }
}

fn single_input_creature() -> neat_ai_discovery::CreatureJson {
    make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    )
}

/// Cluster at -1.0 separated by a wide gap, but its error variance is *higher*
/// than the useful range's — the cluster still carries information, so neither
/// detector may treat it as a sentinel.
fn records_high_variance_cluster() -> Vec<DiscoverRecord> {
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        // Sentinel candidate: errors swing wildly → high variance.
        let error = if i % 2 == 0 { -1.0 } else { 1.0 };
        records.push(record_with_error("input-obs", i, -1.0, error));
    }
    for i in 40..100 {
        // Useful range: near-constant error → low variance.
        let useful_val = 0.1 + 0.7 * ((i - 40) as f32 / 60.0);
        records.push(record_with_error("input-obs", i, useful_val, 0.01));
    }
    records
}

/// The mirror case: the cluster's error variance is lower than the useful
/// range's, so both detectors must accept it.
fn records_low_variance_cluster() -> Vec<DiscoverRecord> {
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records.push(record_with_error("input-obs", i, -1.0, 0.01));
    }
    for i in 40..100 {
        let useful_val = 0.1 + 0.7 * ((i - 40) as f32 / 60.0);
        let error = if i % 2 == 0 { -1.0 } else { 1.0 };
        records.push(record_with_error("input-obs", i, useful_val, error));
    }
    records
}

// ---------------------------------------------------------------------------
// Test 1: a high-error-variance cluster is rejected by `observation_range`
// even when the gap is wide (the regression this issue reports).
// ---------------------------------------------------------------------------
#[test]
fn test_observation_range_rejects_high_error_variance_cluster() {
    let creature = single_input_creature();

    let results = detect_observation_ranges(
        &creature,
        &[("input-obs".to_string(), records_high_variance_cluster())],
    );

    assert!(
        results.is_empty(),
        "A cluster with higher error variance than the useful range is not a \
         sentinel, even with a wide gap; got {results:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: both detectors reject the same high-variance cluster.
// ---------------------------------------------------------------------------
#[test]
fn test_both_detectors_reject_high_error_variance_cluster() {
    let creature = single_input_creature();

    let ranges = detect_observation_ranges(
        &creature,
        &[("input-obs".to_string(), records_high_variance_cluster())],
    );
    let gating = detect_sentinel_gating_candidates(
        &creature,
        &[("input-obs".to_string(), records_high_variance_cluster())],
    );

    assert_eq!(
        ranges.is_empty(),
        gating.is_empty(),
        "Both detectors share one accept/reject rule, so they must agree: \
         ranges={ranges:?}, gating={gating:?}"
    );
    assert!(gating.is_empty(), "Gating must reject the cluster");
}

// ---------------------------------------------------------------------------
// Test 3: both detectors accept the same low-variance cluster.
// ---------------------------------------------------------------------------
#[test]
fn test_both_detectors_accept_low_error_variance_cluster() {
    let creature = single_input_creature();

    let ranges = detect_observation_ranges(
        &creature,
        &[("input-obs".to_string(), records_low_variance_cluster())],
    );
    let gating = detect_sentinel_gating_candidates(
        &creature,
        &[("input-obs".to_string(), records_low_variance_cluster())],
    );

    assert_eq!(
        ranges.len(),
        1,
        "Range detection should accept the sentinel"
    );
    assert!(
        ranges[0]
            .sentinel_values
            .iter()
            .any(|&s| (s + 1.0).abs() < 0.1),
        "Sentinel -1.0 should be reported, got {:?}",
        ranges[0].sentinel_values
    );
    assert_eq!(gating.len(), 1, "Gating should accept the same sentinel");
    assert!(
        (gating[0].sentinel_value + 1.0).abs() < 0.1,
        "Gating should report sentinel -1.0, got {}",
        gating[0].sentinel_value
    );
}

// ---------------------------------------------------------------------------
// Test 4: an insufficient gap is rejected by both detectors, whatever the
// error variance.
// ---------------------------------------------------------------------------
#[test]
fn test_both_detectors_reject_insufficient_gap() {
    let creature = single_input_creature();

    // Useful range starts at -0.95, so the gap from -1.0 (+ tolerance) is 0.03,
    // below MIN_SENTINEL_GAP (0.05).
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records.push(record_with_error("input-obs", i, -1.0, 0.01));
    }
    for i in 40..100 {
        let useful_val = -0.95 + 0.9 * ((i - 40) as f32 / 60.0);
        let error = if i % 2 == 0 { -1.0 } else { 1.0 };
        records.push(record_with_error("input-obs", i, useful_val, error));
    }

    let ranges =
        detect_observation_ranges(&creature, &[("input-obs".to_string(), records.clone())]);
    let gating =
        detect_sentinel_gating_candidates(&creature, &[("input-obs".to_string(), records)]);

    assert!(
        ranges.is_empty(),
        "An insufficient gap is not a sentinel, got {ranges:?}"
    );
    assert!(
        gating.is_empty(),
        "An insufficient gap is not a sentinel, got {gating:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: the shared rule directly — accepts a dense, separated, decorrelated
// cluster and reports the evidence behind the decision.
// ---------------------------------------------------------------------------
#[test]
fn test_assess_sentinel_cluster_accepts_and_reports_evidence() {
    let mut activations: Vec<f32> = vec![-1.0; 40];
    let mut errors: Vec<f32> = vec![0.01; 40];
    for i in 0..60 {
        activations.push(0.1 + 0.7 * (i as f32 / 60.0));
        errors.push(if i % 2 == 0 { -1.0 } else { 1.0 });
    }

    let cluster = assess_sentinel_cluster(&activations, &errors, -1.0)
        .expect("dense, separated, decorrelated cluster is a sentinel");

    assert!((cluster.sentinel_value + 1.0).abs() < f32::EPSILON);
    assert_eq!(cluster.sentinel_indices.len(), 40);
    assert_eq!(cluster.non_sentinel_indices.len(), 60);
    assert!((cluster.sentinel_fraction - 0.4).abs() < 1e-6);
    assert!(
        cluster.sentinel_error_var < cluster.non_sentinel_error_var,
        "the accepted cluster must be the lower-variance one"
    );
    assert!(cluster.gap > 1.0, "gap should span the empty region");
    assert!((cluster.useful_min - 0.1).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Test 6: the shared rule rejects sparse clusters, clusters inside the useful
// range, and empty input.
// ---------------------------------------------------------------------------
#[test]
fn test_assess_sentinel_cluster_rejects_non_sentinels() {
    // Sparse: only 5% of samples sit at the candidate value.
    let mut activations: Vec<f32> = vec![-1.0; 5];
    let mut errors: Vec<f32> = vec![0.01; 5];
    for i in 0..95 {
        activations.push(0.1 + 0.7 * (i as f32 / 95.0));
        errors.push(if i % 2 == 0 { -1.0 } else { 1.0 });
    }
    assert!(
        assess_sentinel_cluster(&activations, &errors, -1.0).is_none(),
        "a sparse cluster is not a sentinel"
    );

    // Inside the useful range: no clear boundary, so the gap is zero.
    let mut activations: Vec<f32> = vec![0.0; 40];
    let mut errors: Vec<f32> = vec![0.01; 40];
    for i in 0..60 {
        activations.push(-1.0 + 2.0 * (i as f32 / 59.0));
        errors.push(if i % 2 == 0 { -1.0 } else { 1.0 });
    }
    assert!(
        assess_sentinel_cluster(&activations, &errors, 0.0).is_none(),
        "a cluster inside the useful range is not a sentinel"
    );

    assert!(
        assess_sentinel_cluster(&[], &[], -1.0).is_none(),
        "no samples means no sentinel"
    );
}

// ---------------------------------------------------------------------------
// Test 7: error variance is the population variance over the given indices.
// ---------------------------------------------------------------------------
#[test]
fn test_compute_error_variance() {
    let errors = [1.0, 3.0, 5.0, 100.0];

    // Mean of [1, 3, 5] is 3; population variance is (4 + 0 + 4) / 3.
    let variance = compute_error_variance(&errors, &[0, 1, 2]);
    assert!(
        (variance - 8.0 / 3.0).abs() < 1e-5,
        "expected 2.666…, got {variance}"
    );

    assert!(
        (compute_error_variance(&errors, &[3]) - 0.0).abs() < f32::EPSILON,
        "a single sample has zero variance"
    );
    assert!(
        (compute_error_variance(&errors, &[]) - 0.0).abs() < f32::EPSILON,
        "no indices means zero variance"
    );
}
