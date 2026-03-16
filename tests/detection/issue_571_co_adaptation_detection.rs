//! Tests for Issue #571: Activation co-adaptation detection — identify redundant
//! neuron pairs.
//!
//! Analyses pairwise activation correlation between hidden neurons to detect
//! co-adapted pairs whose outputs are highly correlated (or anti-correlated),
//! wasting network capacity.
//!
//! ## TDD Plan
//! 1. Test correlated neuron pairs are detected
//! 2. Test anti-correlated neuron pairs are detected
//! 3. Test independent neurons are ignored
//! 4. Test candidates are generated for detected pairs (RemoveNeuron + SetWeight)
//! 5. Test insufficient samples are skipped
//! 6. Test single hidden neuron produces no candidates
//! 7. Test output/input neurons are excluded from pairing

use crate::common::{hidden, make_creature, neuron, output, record, synapse};
use neat_ai_discovery::analysis::detection::co_adaptation::{
    co_adapted_pairs_to_coordinated_candidates, detect_co_adapted_neurons,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: build N records for a neuron with activations following a pattern.
fn make_correlated_records(uuid: &str, count: u32, scale: f32, offset: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| {
            let base = (i as f32 * 0.1).sin();
            record(uuid, i, base * scale + offset, None)
        })
        .collect()
}

/// Helper: build N records with independent (pseudo-random) activations.
fn make_independent_records(uuid: &str, count: u32, seed: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| {
            // Use different frequency/phase so correlation is low
            let activation = (i as f32 * seed + 1.7).sin() * 0.5 + (i as f32 * seed * 2.3).cos();
            record(uuid, i, activation, None)
        })
        .collect()
}

/// Two hidden neurons that produce highly similar activations (same signal
/// shape, same scale) should be detected as co-adapted.
#[test]
fn test_correlated_pair_detected() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            hidden("h0", "TANH"),
            hidden("h1", "LOGISTIC"),
            output("out-0", "TANH"),
        ],
        vec![
            synapse("in-0", "h0", 1.0),
            synapse("in-0", "h1", 0.8),
            synapse("h0", "out-0", 1.0),
            synapse("h1", "out-0", 0.5),
        ],
    );

    let count = 50_u32;
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_correlated_records("h0", count, 1.0, 0.0)),
        ("h1".into(), make_correlated_records("h1", count, 1.0, 0.0)),
    ];

    let detected = detect_co_adapted_neurons(&creature, &records);

    assert!(
        !detected.is_empty(),
        "Should detect co-adapted pair for highly correlated neurons"
    );
    // The pair should involve h0 and h1
    let pair = &detected[0];
    let uuids = [pair.neuron_a_uuid.as_str(), pair.neuron_b_uuid.as_str()];
    assert!(uuids.contains(&"h0"), "Pair should contain h0");
    assert!(uuids.contains(&"h1"), "Pair should contain h1");
    assert!(
        pair.correlation.abs() > 0.9,
        "Correlation should be > 0.9, got {}",
        pair.correlation
    );
}

/// Two neurons with anti-correlated activations (one goes up when the other
/// goes down) should also be detected — they encode the same information
/// with opposite signs.
#[test]
fn test_anti_correlated_pair_detected() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            hidden("h0", "TANH"),
            hidden("h1", "TANH"),
            output("out-0", "TANH"),
        ],
        vec![
            synapse("in-0", "h0", 1.0),
            synapse("in-0", "h1", -1.0),
            synapse("h0", "out-0", 1.0),
            synapse("h1", "out-0", -0.5),
        ],
    );

    let count = 50_u32;
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_correlated_records("h0", count, 1.0, 0.0)),
        ("h1".into(), make_correlated_records("h1", count, -1.0, 0.0)),
    ];

    let detected = detect_co_adapted_neurons(&creature, &records);

    assert!(!detected.is_empty(), "Should detect anti-correlated pair");
    let pair = &detected[0];
    assert!(
        pair.correlation < -0.9,
        "Anti-correlated pair should have correlation < -0.9, got {}",
        pair.correlation
    );
}

/// Two neurons with independent (uncorrelated) activation patterns should
/// NOT be flagged as co-adapted.
#[test]
fn test_independent_neurons_ignored() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            hidden("h0", "TANH"),
            hidden("h1", "TANH"),
            output("out-0", "TANH"),
        ],
        vec![
            synapse("in-0", "h0", 1.0),
            synapse("in-0", "h1", 1.0),
            synapse("h0", "out-0", 1.0),
            synapse("h1", "out-0", 0.5),
        ],
    );

    let count = 50_u32;
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_independent_records("h0", count, 0.7)),
        ("h1".into(), make_independent_records("h1", count, 3.1)),
    ];

    let detected = detect_co_adapted_neurons(&creature, &records);

    assert!(
        detected.is_empty(),
        "Independent neurons should not be detected as co-adapted, got {} pairs",
        detected.len()
    );
}

/// Detected co-adapted pairs should produce coordinated structural candidates:
/// - RemoveNeuron for the lower-impact neuron
/// - SetWeight perturbation as an alternative
#[test]
fn test_candidates_generated_for_co_adapted_pair() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            hidden("h0", "TANH"),
            hidden("h1", "TANH"),
            output("out-0", "TANH"),
        ],
        vec![
            synapse("in-0", "h0", 1.0),
            synapse("in-0", "h1", 0.9),
            synapse("h0", "out-0", 1.0),
            synapse("h1", "out-0", 0.5),
        ],
    );

    let count = 50_u32;
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_correlated_records("h0", count, 1.0, 0.0)),
        ("h1".into(), make_correlated_records("h1", count, 1.0, 0.0)),
    ];

    let detected = detect_co_adapted_neurons(&creature, &records);
    assert!(!detected.is_empty());

    let candidates = co_adapted_pairs_to_coordinated_candidates(&detected, &creature);

    assert!(
        !candidates.is_empty(),
        "Should produce coordinated candidates for co-adapted pairs"
    );

    // Each candidate should have operations and a positive expected gain
    for c in &candidates {
        assert!(
            !c.operations.is_empty(),
            "Candidate should have at least one operation"
        );
        assert!(
            c.expected_creature_score_gain > 0.0,
            "Expected gain should be positive"
        );
        assert!(c.comment.is_some(), "Candidate should have a comment");
        let comment = c.comment.as_ref().unwrap();
        assert!(
            comment.contains("#571"),
            "Comment should reference issue #571"
        );
    }
}

/// Neurons with fewer samples than the minimum threshold should be skipped.
#[test]
fn test_co_adaptation_insufficient_samples_skipped() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            hidden("h0", "TANH"),
            hidden("h1", "TANH"),
            output("out-0", "TANH"),
        ],
        vec![
            synapse("in-0", "h0", 1.0),
            synapse("in-0", "h1", 1.0),
            synapse("h0", "out-0", 1.0),
            synapse("h1", "out-0", 0.5),
        ],
    );

    // Only 3 samples — below minimum
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_correlated_records("h0", 3, 1.0, 0.0)),
        ("h1".into(), make_correlated_records("h1", 3, 1.0, 0.0)),
    ];

    let detected = detect_co_adapted_neurons(&creature, &records);

    assert!(
        detected.is_empty(),
        "Should not detect pairs when samples are insufficient"
    );
}

/// A network with only one hidden neuron cannot have co-adapted pairs.
#[test]
fn test_co_adaptation_single_hidden_neuron_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            hidden("h0", "TANH"),
            output("out-0", "TANH"),
        ],
        vec![synapse("in-0", "h0", 1.0), synapse("h0", "out-0", 1.0)],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> =
        vec![("h0".into(), make_correlated_records("h0", 50, 1.0, 0.0))];

    let detected = detect_co_adapted_neurons(&creature, &records);

    assert!(
        detected.is_empty(),
        "Single hidden neuron should produce no co-adapted pairs"
    );
}

/// Input and output neurons should not be included in co-adaptation pairing.
#[test]
fn test_only_hidden_neurons_paired() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            neuron("in-1", "input", "IDENTITY"),
            hidden("h0", "TANH"),
            output("out-0", "TANH"),
        ],
        vec![
            synapse("in-0", "h0", 1.0),
            synapse("in-1", "h0", 1.0),
            synapse("h0", "out-0", 1.0),
        ],
    );

    // Even though in-0 and in-1 have correlated records, they shouldn't be paired
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("in-0".into(), make_correlated_records("in-0", 50, 1.0, 0.0)),
        ("in-1".into(), make_correlated_records("in-1", 50, 1.0, 0.0)),
        ("h0".into(), make_correlated_records("h0", 50, 1.0, 0.0)),
    ];

    let detected = detect_co_adapted_neurons(&creature, &records);

    assert!(
        detected.is_empty(),
        "Input/output neurons should not be in co-adapted pairs, only hidden neurons"
    );
}
