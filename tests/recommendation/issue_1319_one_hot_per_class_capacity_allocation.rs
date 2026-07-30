//! Tests for Issue #1319: Per-class capacity allocation under `one_hot` using
//! the existing per-target failure counters.
//!
//! Under a `OneHot` task descriptor the growth/candidate budget must skew
//! toward output neurons (classes) with the highest per-target failure
//! counts. For every other descriptor (`Independent`, `Margin`, `Unknown`,
//! `OTHER`, absent) the existing allocation must hold verbatim — regression
//! guard.

#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use neat_ai_discovery::CandidateNeuronJson;
use neat_ai_discovery::analysis::one_hot_class_allocation::{
    apply_class_priority_spread, compute_class_failure_counts,
};
use neat_ai_discovery::analysis::scoring::calibration_correction::FailureCacheEntry;
use neat_ai_discovery::analysis::task_descriptor::TaskDescriptor;

fn fc_entry(target_uuid: &str) -> FailureCacheEntry {
    FailureCacheEntry {
        change_type: "add-neurons".to_string(),
        expected_error_reduction: 0.1,
        actual_error_reduction: -0.01,
        target_squash: None,
        variant_key: None,
        target_uuid: Some(target_uuid.to_string()),
        improved_count: None,
        total_count: None,
        age_epochs: None,
    }
}

fn candidate(target_uuid: &str, gain: f32) -> CandidateNeuronJson {
    CandidateNeuronJson {
        source_neuron_uuid: format!("source-{target_uuid}-{gain}"),
        target_neuron_uuid: target_uuid.to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 1.0,
        outgoing_weight: 1.0,
        squash: format!("SQUASH-{gain}"),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: gain,
        expected_creature_score_gain: gain,
        improved_count: 10,
        total_count: 10,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [gain, gain],
        target_saturation_factor: None,
        variant_key: None,
    }
}

/// Acceptance #1: under `OneHot`, per-class failure counts derived from
/// `failure_cache` map output neurons to their cumulative failure count.
#[test]
fn one_hot_descriptor_computes_per_class_failure_counts() {
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 3);
    let cache = vec![
        fc_entry("output-A"),
        fc_entry("output-A"),
        fc_entry("output-A"),
        fc_entry("output-B"),
        fc_entry("hidden-X"), // hidden, must not be counted as a class
    ];

    let outputs: std::collections::HashSet<String> = ["output-A", "output-B", "output-C"]
        .into_iter()
        .map(String::from)
        .collect();

    let counts = compute_class_failure_counts(&descriptor, &cache, |uuid| outputs.contains(uuid))
        .expect("OneHot descriptor must produce a failure-count map");

    assert_eq!(counts.get("output-A").copied(), Some(3));
    assert_eq!(counts.get("output-B").copied(), Some(1));
    // Output-C never failed — must be absent (not present with 0).
    assert!(!counts.contains_key("output-C"));
    // Hidden neurons must never appear in the class-failure map.
    assert!(!counts.contains_key("hidden-X"));
}

/// Acceptance #2: non-OneHot descriptors yield `None`, locking the caller
/// into the existing allocation path (regression guard).
#[test]
fn neutral_descriptor_yields_no_class_failure_counts() {
    let cache = vec![fc_entry("output-A")];
    let outputs: std::collections::HashSet<String> =
        ["output-A"].into_iter().map(String::from).collect();

    assert!(
        compute_class_failure_counts(&TaskDescriptor::neutral(), &cache, |u| outputs.contains(u))
            .is_none(),
        "neutral descriptor must opt out of per-class allocation",
    );

    let mse = TaskDescriptor::from_name("MSE", 3);
    assert!(
        compute_class_failure_counts(&mse, &cache, |u| outputs.contains(u)).is_none(),
        "Independent descriptor must opt out of per-class allocation",
    );

    let other = TaskDescriptor::from_name("OTHER", 3);
    assert!(
        compute_class_failure_counts(&other, &cache, |u| outputs.contains(u)).is_none(),
        "OTHER descriptor must opt out of per-class allocation",
    );

    let hinge = TaskDescriptor::from_name("HINGE", 1);
    assert!(
        compute_class_failure_counts(&hinge, &cache, |u| outputs.contains(u)).is_none(),
        "Margin descriptor must opt out of per-class allocation",
    );
}

/// Acceptance #3: under `OneHot` priority, the highest-failure-count classes
/// occupy the front of the reordered candidate list — pulling growth budget
/// toward the worst-performing classes ahead of the per-target cap.
#[test]
fn class_priority_spread_pulls_worst_classes_to_front() {
    // target-A: 6 candidates, gain dominant.
    // target-B: 1 candidate, mid gain.
    // target-C: 1 candidate, low gain.
    // target-D: 1 candidate, even lower gain.
    let mut candidates = vec![
        candidate("target-A", 0.99),
        candidate("target-A", 0.95),
        candidate("target-A", 0.92),
        candidate("target-A", 0.90),
        candidate("target-A", 0.87),
        candidate("target-A", 0.85),
        candidate("target-B", 0.50),
        candidate("target-C", 0.40),
        candidate("target-D", 0.30),
    ];
    // Pre-sort by gain to mirror the documented precondition.
    candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    // Priority: D worst, then C, then B; A unseen (priority 0).
    let mut priority: HashMap<String, u32> = HashMap::new();
    priority.insert("target-D".to_string(), 9);
    priority.insert("target-C".to_string(), 6);
    priority.insert("target-B".to_string(), 3);

    apply_class_priority_spread(&mut candidates, &priority, 3);

    let front: Vec<&str> = candidates[0..3]
        .iter()
        .map(|c| c.target_neuron_uuid.as_str())
        .collect();
    assert_eq!(
        front,
        vec!["target-D", "target-C", "target-B"],
        "front of the list must lead with the worst-performing classes",
    );
    // No candidate dropped.
    assert_eq!(candidates.len(), 9);
    // Tail retains the remaining target-A candidates in gain order.
    let tail_a_gains: Vec<f32> = candidates[3..]
        .iter()
        .filter(|c| c.target_neuron_uuid == "target-A")
        .map(|c| c.expected_creature_score_gain)
        .collect();
    let mut sorted_a = tail_a_gains.clone();
    sorted_a.sort_by(|a, b| b.total_cmp(a));
    assert_eq!(
        tail_a_gains, sorted_a,
        "target-A tail must remain in gain-descending order",
    );
}

/// Acceptance #4 (regression guard): an empty priority map is a no-op — the
/// gain-sorted order is preserved verbatim. Equivalent to running on a
/// neutral / OTHER descriptor.
#[test]
fn empty_priority_map_is_a_noop() {
    let mut candidates = vec![
        candidate("target-A", 0.9),
        candidate("target-B", 0.5),
        candidate("target-C", 0.4),
        candidate("target-D", 0.3),
    ];
    candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });
    let before: Vec<(String, f32)> = candidates
        .iter()
        .map(|c| (c.target_neuron_uuid.clone(), c.expected_creature_score_gain))
        .collect();

    apply_class_priority_spread(&mut candidates, &HashMap::new(), 3);

    let after: Vec<(String, f32)> = candidates
        .iter()
        .map(|c| (c.target_neuron_uuid.clone(), c.expected_creature_score_gain))
        .collect();
    assert_eq!(
        before, after,
        "empty priority map must not reorder candidates",
    );
}

/// Acceptance #5: priority spread keeps the highest-gain representative for
/// each priority target — within a class, gain order still decides which
/// candidate occupies the spread slot.
#[test]
fn class_priority_spread_keeps_highest_gain_representative() {
    // target-B has two candidates with very different gains.
    let mut candidates = vec![
        candidate("target-A", 0.9),
        candidate("target-B", 0.05),
        candidate("target-B", 0.6),
        candidate("target-C", 0.4),
    ];
    candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });
    let mut priority = HashMap::new();
    priority.insert("target-B".to_string(), 7);
    priority.insert("target-C".to_string(), 5);

    apply_class_priority_spread(&mut candidates, &priority, 3);

    // Front: target-B (priority 7, gain 0.6), target-C (priority 5, gain 0.4),
    // target-A (priority 0, gain 0.9).
    assert_eq!(candidates[0].target_neuron_uuid, "target-B");
    assert!((candidates[0].expected_creature_score_gain - 0.6).abs() < f32::EPSILON);
    assert_eq!(candidates[1].target_neuron_uuid, "target-C");
    assert_eq!(candidates[2].target_neuron_uuid, "target-A");
}

/// Acceptance #6: if the pool has fewer distinct targets than the spread
/// requires, the function falls through and preserves gain-order — same
/// contract as the legacy `apply_distinct_target_spread`.
#[test]
fn class_priority_spread_falls_through_when_pool_too_narrow() {
    let mut candidates = vec![
        candidate("target-A", 0.9),
        candidate("target-A", 0.8),
        candidate("target-B", 0.6),
    ];
    candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });
    let before: Vec<(String, f32)> = candidates
        .iter()
        .map(|c| (c.target_neuron_uuid.clone(), c.expected_creature_score_gain))
        .collect();

    let mut priority = HashMap::new();
    priority.insert("target-A".to_string(), 5);

    // min_distinct=3 but only 2 distinct targets — must no-op.
    apply_class_priority_spread(&mut candidates, &priority, 3);

    let after: Vec<(String, f32)> = candidates
        .iter()
        .map(|c| (c.target_neuron_uuid.clone(), c.expected_creature_score_gain))
        .collect();
    assert_eq!(before, after);
}
