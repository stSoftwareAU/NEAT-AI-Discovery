//! Cross-target diversity selection tests (Issue #1193).
//!
//! Acceptance criteria from the issue:
//! - Constructs a candidate pool where the top six candidates all hit one
//!   target.
//! - Asserts the emitted batch contains at least
//!   `MIN_DISTINCT_TARGETS_PER_BATCH` distinct target neurons (provided that
//!   many exist in the pool).

use crate::CandidateNeuronJson;
use crate::analysis::constants::min_distinct_targets_per_batch;
use crate::analysis::neuron::post_processing::apply_per_target_cap;
use std::collections::HashSet;

fn candidate(target_uuid: &str, gain: f32) -> CandidateNeuronJson {
    let squash = format!("SQUASH-{target_uuid}-{gain}");
    CandidateNeuronJson {
        source_neuron_uuid: format!("source-{target_uuid}-{gain}"),
        target_neuron_uuid: target_uuid.to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 1.0,
        outgoing_weight: 1.0,
        squash,
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

/// Acceptance test for Issue #1193.
///
/// The top six gain-ranked candidates all target `neuron-1978541840` — the
/// failure-cache pattern from Issue #1189. Three additional targets each have
/// one lower-gain candidate. After post-processing the emitted batch must
/// contain at least `MIN_DISTINCT_TARGETS_PER_BATCH` distinct targets.
#[test]
fn emitted_batch_covers_min_distinct_targets_when_pool_supports_it() {
    let mut candidates = vec![
        // Six candidates against the dominant problematic target.
        candidate("neuron-1978541840", 0.99),
        candidate("neuron-1978541840", 0.95),
        candidate("neuron-1978541840", 0.92),
        candidate("neuron-1978541840", 0.90),
        candidate("neuron-1978541840", 0.87),
        candidate("neuron-1978541840", 0.85),
        // One candidate each against three other targets.
        candidate("neuron-other-B", 0.50),
        candidate("neuron-other-C", 0.40),
        candidate("neuron-other-D", 0.30),
    ];

    let _dropped = apply_per_target_cap(&mut candidates);

    let distinct: HashSet<&str> = candidates
        .iter()
        .map(|c| c.target_neuron_uuid.as_str())
        .collect();
    let min_distinct = min_distinct_targets_per_batch();
    assert!(
        distinct.len() >= min_distinct,
        "emitted batch must include at least {} distinct targets, got {} ({:?})",
        min_distinct,
        distinct.len(),
        distinct
    );
    // The dominant target keeps its full per-target quota of three (three of
    // the original six are dropped by the cap); three other targets are
    // represented with one candidate each, for a total of six retained.
    assert_eq!(candidates.len(), 6);
}

/// Fall-through case: when the pool only contains a single distinct target,
/// the spread does not over-rotate — the per-target cap of three still
/// caps the dominant target without forcing distinctness that does not exist.
#[test]
fn fall_through_preserves_per_target_cap_when_only_one_target() {
    let mut candidates = vec![
        candidate("only-target", 0.9),
        candidate("only-target", 0.8),
        candidate("only-target", 0.7),
        candidate("only-target", 0.6),
        candidate("only-target", 0.5),
    ];

    let dropped = apply_per_target_cap(&mut candidates);

    assert_eq!(candidates.len(), 3, "cap retains the full quota of 3");
    assert_eq!(dropped, 2);
    let distinct: HashSet<&str> = candidates
        .iter()
        .map(|c| c.target_neuron_uuid.as_str())
        .collect();
    assert_eq!(
        distinct.len(),
        1,
        "no spread possible — pool has one target"
    );
}
