//! Tests for Issue #398: Detect observation effective ranges from recorded samples.
//!
//! Observations are finite numbers, often normalised to -1…1. Since observations
//! lack a concept of null, sentinel values like -1 or 0 are used instead. This module
//! detects the "effective range" of each observation where the values actually
//! correlate with meaningful error changes, excluding sentinel clusters.
//!
//! ## TDD Plan
//! 1. Detect observation with sentinel cluster at -1 and compute effective range
//! 2. Detect sentinel at 0 with useful range above
//! 3. Uniform distribution — full range is effective
//! 4. Multiple sentinels detected on same neuron
//! 5. Insufficient samples — no detection
//! 6. Utilisation ratio computed correctly
//! 7. Only input neurons (observations) are analysed
//! 8. Error correlation distinguishes sentinel from effective range
//! 9. Multiple observations — independent ranges per neuron

mod common;

use common::{make_creature, neuron, output, synapse};
use neat_ai_discovery::analysis::detection::observation_range::detect_observation_ranges;
use neat_ai_discovery::types::DiscoverRecord;

/// Helper to create a record with a specific error value.
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

// ---------------------------------------------------------------------------
// Test 1: Observation with sentinel cluster at -1.0 — effective range excludes it.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_effective_range_excluding_sentinel_at_minus_one() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    // 40% at -1.0 (sentinel with near-zero error variance), 60% in [0.1, 0.8]
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        // Sentinel samples: constant small error (no correlation)
        records.push(record_with_error("input-obs", i, -1.0, 0.01));
    }
    for i in 40..100 {
        // Useful samples: varied error correlated with activation
        let useful_val = 0.1 + 0.7 * ((i - 40) as f32 / 60.0);
        let error = 0.5 - useful_val; // Error varies with activation
        records.push(record_with_error("input-obs", i, useful_val, error));
    }

    let results = detect_observation_ranges(&creature, &[("input-obs".to_string(), records)]);

    assert_eq!(results.len(), 1, "Should detect one observation range");
    let r = &results[0];
    assert_eq!(r.neuron_uuid, "input-obs");
    assert!(
        r.effective_min > -0.5,
        "Effective min should exclude sentinel at -1, got {}",
        r.effective_min
    );
    assert!(
        r.effective_max > 0.5,
        "Effective max should reflect the useful range, got {}",
        r.effective_max
    );
    assert!(
        !r.sentinel_values.is_empty(),
        "Should detect at least one sentinel value"
    );
    assert!(
        r.sentinel_values.iter().any(|&s| (s - (-1.0)).abs() < 0.1),
        "Sentinel values should include -1.0"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Sentinel cluster at 0.0 — effective range is above zero.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_effective_range_with_sentinel_at_zero() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    let mut records: Vec<DiscoverRecord> = Vec::new();
    // 50% at 0.0 (sentinel)
    for i in 0..50 {
        records.push(record_with_error("input-obs", i, 0.0, 0.01));
    }
    // 50% in [0.3, 0.9]
    for i in 50..100 {
        let useful_val = 0.3 + 0.6 * ((i - 50) as f32 / 50.0);
        let error = 0.5 - useful_val;
        records.push(record_with_error("input-obs", i, useful_val, error));
    }

    let results = detect_observation_ranges(&creature, &[("input-obs".to_string(), records)]);

    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(
        r.effective_min > 0.1,
        "Effective min should be above sentinel at 0, got {}",
        r.effective_min
    );
    assert!(
        r.sentinel_values.iter().any(|&s| s.abs() < 0.1),
        "Should detect 0.0 as sentinel"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Uniform distribution — full range is effective, no sentinels.
// ---------------------------------------------------------------------------
#[test]
fn test_uniform_distribution_full_range_effective() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    // Evenly spread across [-1, 1] with varied errors
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let val = -1.0 + 2.0 * (i as f32 / 99.0);
            let error = 0.5 - val * 0.3;
            record_with_error("input-obs", i, val, error)
        })
        .collect();

    let results = detect_observation_ranges(&creature, &[("input-obs".to_string(), records)]);

    // With uniform distribution, no sentinel cluster exists
    if results.len() == 1 {
        let r = &results[0];
        assert!(
            r.sentinel_values.is_empty(),
            "Uniform distribution should have no sentinels, got {:?}",
            r.sentinel_values
        );
        assert!(
            r.utilisation_ratio > 0.8,
            "Utilisation ratio should be high for uniform distribution, got {}",
            r.utilisation_ratio
        );
    }
    // It's also acceptable to return no results if no sentinels are found
}

// ---------------------------------------------------------------------------
// Test 4: Insufficient samples — no detection.
// ---------------------------------------------------------------------------
#[test]
fn test_observation_range_insufficient_samples_no_detection() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| record_with_error("input-obs", i, -1.0, 0.01))
        .collect();

    let results = detect_observation_ranges(&creature, &[("input-obs".to_string(), records)]);

    assert!(
        results.is_empty(),
        "Should not detect with insufficient samples"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Utilisation ratio is fraction of full range that is effective.
// ---------------------------------------------------------------------------
#[test]
fn test_utilisation_ratio_computed_correctly() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    // Range is [-1, 1] (full=2.0), sentinel at -1 with useful range [0.0, 0.8]
    // Effective range = 0.8, so utilisation ≈ 0.8/2.0 = 0.4
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records.push(record_with_error("input-obs", i, -1.0, 0.01));
    }
    for i in 40..100 {
        let useful_val = 0.8 * ((i - 40) as f32 / 60.0);
        let error = 0.3 - useful_val * 0.2;
        records.push(record_with_error("input-obs", i, useful_val, error));
    }

    let results = detect_observation_ranges(&creature, &[("input-obs".to_string(), records)]);

    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(
        r.utilisation_ratio > 0.0 && r.utilisation_ratio <= 1.0,
        "Utilisation ratio should be in (0, 1], got {}",
        r.utilisation_ratio
    );
    assert!(
        r.utilisation_ratio < 0.8,
        "Utilisation ratio should be less than 0.8 when sentinel consumes range, got {}",
        r.utilisation_ratio
    );
}

// ---------------------------------------------------------------------------
// Test 6: Output neurons are excluded (only input observations analysed).
// ---------------------------------------------------------------------------
#[test]
fn test_observation_range_output_neurons_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            output("output-1", "TANH"),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Output neuron with sentinel-like pattern — should NOT be analysed
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records.push(record_with_error("output-1", i, -1.0, 0.01));
    }
    for i in 40..100 {
        let val = 0.5 * ((i - 40) as f32 / 60.0);
        records.push(record_with_error("output-1", i, val, 0.1));
    }

    let results = detect_observation_ranges(&creature, &[("output-1".to_string(), records)]);

    assert!(
        results.is_empty(),
        "Output neurons should be excluded from observation range detection"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Error correlation — sentinel has low error variance vs effective range.
// ---------------------------------------------------------------------------
#[test]
fn test_error_correlation_distinguishes_sentinel() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    // Sentinel at -1: constant error (no variance)
    // Useful range [0.2, 0.8]: error varies proportionally
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..30 {
        records.push(record_with_error("input-obs", i, -1.0, 0.05));
    }
    for i in 30..100 {
        let useful_val = 0.2 + 0.6 * ((i - 30) as f32 / 70.0);
        let error = useful_val * 0.5; // Linearly correlated error
        records.push(record_with_error("input-obs", i, useful_val, error));
    }

    let results = detect_observation_ranges(&creature, &[("input-obs".to_string(), records)]);

    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(
        r.effective_min > -0.5,
        "Effective range should start above sentinel"
    );
    assert!(
        !r.sentinel_values.is_empty(),
        "Should detect sentinel values"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Multiple observations — independent ranges per neuron.
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_observations_independent_ranges() {
    let creature = make_creature(
        vec![
            neuron("obs-a", "input", "IDENTITY"),
            neuron("obs-b", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("obs-a", "output-1", 0.5),
            synapse("obs-b", "output-1", 0.3),
        ],
    );

    // obs-a: sentinel at -1
    let mut records_a: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records_a.push(record_with_error("obs-a", i, -1.0, 0.01));
    }
    for i in 40..100 {
        let val = 0.1 + 0.7 * ((i - 40) as f32 / 60.0);
        records_a.push(record_with_error("obs-a", i, val, 0.5 - val));
    }

    // obs-b: sentinel at 0
    let mut records_b: Vec<DiscoverRecord> = Vec::new();
    for i in 0..35 {
        records_b.push(record_with_error("obs-b", i, 0.0, 0.02));
    }
    for i in 35..100 {
        let val = 0.3 + 0.6 * ((i - 35) as f32 / 65.0);
        records_b.push(record_with_error("obs-b", i, val, 0.4 - val));
    }

    let results = detect_observation_ranges(
        &creature,
        &[
            ("obs-a".to_string(), records_a),
            ("obs-b".to_string(), records_b),
        ],
    );

    assert_eq!(
        results.len(),
        2,
        "Should detect ranges for both observations"
    );

    let result_a = results.iter().find(|r| r.neuron_uuid == "obs-a");
    let result_b = results.iter().find(|r| r.neuron_uuid == "obs-b");

    assert!(result_a.is_some(), "Should have result for obs-a");
    assert!(result_b.is_some(), "Should have result for obs-b");

    let a = result_a.unwrap();
    let b = result_b.unwrap();

    // obs-a sentinel at -1, obs-b sentinel at 0 — different sentinels
    assert!(
        a.sentinel_values.iter().any(|&s| (s - (-1.0)).abs() < 0.1),
        "obs-a should have sentinel near -1"
    );
    assert!(
        b.sentinel_values.iter().any(|&s| s.abs() < 0.1),
        "obs-b should have sentinel near 0"
    );
}

// ---------------------------------------------------------------------------
// Test 9: All values identical — no effective range to report.
// ---------------------------------------------------------------------------
#[test]
fn test_all_identical_values_no_detection() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record_with_error("input-obs", i, 0.5, 0.01))
        .collect();

    let results = detect_observation_ranges(&creature, &[("input-obs".to_string(), records)]);

    assert!(
        results.is_empty(),
        "All-identical values should not produce a detection"
    );
}
