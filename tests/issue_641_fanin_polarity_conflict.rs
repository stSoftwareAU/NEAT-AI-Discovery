//! Tests for Issue #641: Fan-in weight polarity conflict detector.
//!
//! When a hidden neuron receives incoming synapses with strongly positive weights
//! and strongly negative weights of similar magnitude, the inputs fight each other.
//! Most of the input signal cancels out, wasting representational capacity.
//!
//! This module detects the general case of "this neuron's incoming weights are in
//! fundamental tension" — distinct from opposing_synapse (same-source pairs) and
//! weight_coherence (ratio consistency).
//!
//! ## TDD Plan
//! 1. Test detection of clear polarity conflict (balanced positive/negative fan-in)
//! 2. Test healthy fan-in (same sign) produces no candidates
//! 3. Test single-sign fan-in with one tiny opposing weight is not flagged
//! 4. Test candidate proposes addNeuron to split positive/negative pathways
//! 5. Test insufficient samples produce no candidates
//! 6. Test empty records produce no candidates
//! 7. Test input/output neurons are excluded (hidden only)
//! 8. Test candidates sorted by conflict score

mod common;

use common::{hidden, make_creature, neuron, output, synapse};
use neat_ai_discovery::analysis::detection::fanin_polarity_conflict::{
    detect_fanin_polarity_conflicts, fanin_polarity_conflicts_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helper: create records with specified activations and errors
// =============================================================================

fn make_record(neuron_uuid: &str, obs_index: u32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![error],
    }
}

fn make_records(neuron_uuid: &str, count: u32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| make_record(neuron_uuid, i, (i as f32 + 1.0) / 10.0, 0.01))
        .collect()
}

// =============================================================================
// Test 1: Detects clear polarity conflict (balanced positive/negative fan-in)
// =============================================================================

/// A hidden neuron with five +2.0 synapses and three -2.0 synapses has strong
/// internal cancellation. The conflict score should exceed the threshold.
#[test]
fn test_detects_clear_polarity_conflict() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            neuron("in-1", "input", "IDENTITY"),
            neuron("in-2", "input", "IDENTITY"),
            neuron("in-3", "input", "IDENTITY"),
            neuron("in-4", "input", "IDENTITY"),
            neuron("in-5", "input", "IDENTITY"),
            neuron("in-6", "input", "IDENTITY"),
            neuron("in-7", "input", "IDENTITY"),
            hidden("h-0", "TANH"),
            output("out-0", "IDENTITY"),
        ],
        vec![
            // Five positive incoming synapses
            synapse("in-0", "h-0", 2.0),
            synapse("in-1", "h-0", 2.0),
            synapse("in-2", "h-0", 2.0),
            synapse("in-3", "h-0", 2.0),
            synapse("in-4", "h-0", 2.0),
            // Three negative incoming synapses of similar magnitude
            synapse("in-5", "h-0", -2.0),
            synapse("in-6", "h-0", -2.0),
            synapse("in-7", "h-0", -2.0),
            // Outgoing synapse
            synapse("h-0", "out-0", 1.0),
        ],
    );

    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for i in 0..8 {
        records.push((format!("in-{i}"), make_records(&format!("in-{i}"), 30)));
    }
    records.push(("h-0".to_string(), make_records("h-0", 30)));
    records.push(("out-0".to_string(), make_records("out-0", 30)));

    let candidates = detect_fanin_polarity_conflicts(&creature, &records);

    assert!(
        !candidates.is_empty(),
        "Should detect polarity conflict for hidden neuron with balanced opposing weights"
    );

    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "h-0");
    assert!(
        c.conflict_score > 0.0,
        "Conflict score should be positive: got {}",
        c.conflict_score
    );
    assert!(
        c.positive_weight_sum > 0.0,
        "Positive weight sum should be > 0"
    );
    assert!(
        c.negative_weight_sum > 0.0,
        "Negative weight sum (absolute) should be > 0"
    );
}

// =============================================================================
// Test 2: Healthy fan-in (same sign) produces no candidates
// =============================================================================

/// A hidden neuron with all positive incoming weights has no polarity conflict.
#[test]
fn test_healthy_fanin_no_conflict() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            neuron("in-1", "input", "IDENTITY"),
            neuron("in-2", "input", "IDENTITY"),
            hidden("h-0", "TANH"),
            output("out-0", "IDENTITY"),
        ],
        vec![
            synapse("in-0", "h-0", 2.0),
            synapse("in-1", "h-0", 1.5),
            synapse("in-2", "h-0", 3.0),
            synapse("h-0", "out-0", 1.0),
        ],
    );

    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for i in 0..3 {
        records.push((format!("in-{i}"), make_records(&format!("in-{i}"), 30)));
    }
    records.push(("h-0".to_string(), make_records("h-0", 30)));
    records.push(("out-0".to_string(), make_records("out-0", 30)));

    let candidates = detect_fanin_polarity_conflicts(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Healthy same-sign fan-in should produce no conflict candidates, got {}",
        candidates.len()
    );
}

// =============================================================================
// Test 3: One tiny opposing weight is not flagged
// =============================================================================

/// A fan-in with many positive weights and one small negative weight should
/// not be flagged, because the conflict score is too low.
#[test]
fn test_tiny_opposing_weight_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            neuron("in-1", "input", "IDENTITY"),
            neuron("in-2", "input", "IDENTITY"),
            neuron("in-3", "input", "IDENTITY"),
            hidden("h-0", "TANH"),
            output("out-0", "IDENTITY"),
        ],
        vec![
            synapse("in-0", "h-0", 2.0),
            synapse("in-1", "h-0", 2.0),
            synapse("in-2", "h-0", 2.0),
            synapse("in-3", "h-0", -0.1), // Tiny opposing weight
            synapse("h-0", "out-0", 1.0),
        ],
    );

    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for i in 0..4 {
        records.push((format!("in-{i}"), make_records(&format!("in-{i}"), 30)));
    }
    records.push(("h-0".to_string(), make_records("h-0", 30)));
    records.push(("out-0".to_string(), make_records("out-0", 30)));

    let candidates = detect_fanin_polarity_conflicts(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Tiny opposing weight should not trigger conflict, got {} candidate(s)",
        candidates.len()
    );
}

// =============================================================================
// Test 4: Candidate proposes addNeuron to split positive/negative pathways
// =============================================================================

/// The coordinated candidate should propose an addNeuron operation to split
/// the conflicting pathways into separate neurons.
#[test]
fn test_candidate_proposes_add_neuron_split() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            neuron("in-1", "input", "IDENTITY"),
            neuron("in-2", "input", "IDENTITY"),
            neuron("in-3", "input", "IDENTITY"),
            hidden("h-0", "TANH"),
            output("out-0", "IDENTITY"),
        ],
        vec![
            synapse("in-0", "h-0", 2.0),
            synapse("in-1", "h-0", 2.0),
            synapse("in-2", "h-0", -2.0),
            synapse("in-3", "h-0", -2.0),
            synapse("h-0", "out-0", 1.0),
        ],
    );

    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for i in 0..4 {
        records.push((format!("in-{i}"), make_records(&format!("in-{i}"), 30)));
    }
    records.push(("h-0".to_string(), make_records("h-0", 30)));
    records.push(("out-0".to_string(), make_records("out-0", 30)));

    let candidates = detect_fanin_polarity_conflicts(&creature, &records);
    assert!(!candidates.is_empty(), "Should detect conflict");

    let coordinated = fanin_polarity_conflicts_to_coordinated_candidates(&candidates, &creature);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    let coord = &coordinated[0];
    assert!(
        coord.expected_creature_score_gain > 0.0,
        "Should have positive expected improvement"
    );

    // Should contain an AddNeuron operation
    use neat_ai_discovery::CoordinatedStructuralOpJson;
    let has_add_neuron = coord
        .operations
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::AddNeuron { .. }));
    assert!(
        has_add_neuron,
        "Coordinated candidate should include an addNeuron operation to split pathways"
    );
}

// =============================================================================
// Test 5: Insufficient samples produce no candidates
// =============================================================================

#[test]
fn test_insufficient_samples_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            neuron("in-1", "input", "IDENTITY"),
            hidden("h-0", "TANH"),
            output("out-0", "IDENTITY"),
        ],
        vec![
            synapse("in-0", "h-0", 2.0),
            synapse("in-1", "h-0", -2.0),
            synapse("h-0", "out-0", 1.0),
        ],
    );

    // Only 3 samples — below minimum
    let records = vec![
        ("in-0".to_string(), make_records("in-0", 3)),
        ("in-1".to_string(), make_records("in-1", 3)),
        ("h-0".to_string(), make_records("h-0", 3)),
        ("out-0".to_string(), make_records("out-0", 3)),
    ];

    let candidates = detect_fanin_polarity_conflicts(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Insufficient samples should produce no candidates"
    );
}

// =============================================================================
// Test 6: Empty records produce no candidates
// =============================================================================

#[test]
fn test_empty_records_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            hidden("h-0", "TANH"),
            output("out-0", "IDENTITY"),
        ],
        vec![synapse("in-0", "h-0", 2.0), synapse("h-0", "out-0", 1.0)],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];
    let candidates = detect_fanin_polarity_conflicts(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

// =============================================================================
// Test 7: Input/output neurons are excluded (hidden only)
// =============================================================================

/// Polarity conflict detection only applies to hidden neurons.
/// Output neurons with conflicting fan-in are not flagged.
#[test]
fn test_only_hidden_neurons_flagged() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            neuron("in-1", "input", "IDENTITY"),
            // Output neuron with conflicting fan-in
            output("out-0", "IDENTITY"),
        ],
        vec![
            synapse("in-0", "out-0", 2.0),
            synapse("in-1", "out-0", -2.0),
        ],
    );

    let records = vec![
        ("in-0".to_string(), make_records("in-0", 30)),
        ("in-1".to_string(), make_records("in-1", 30)),
        ("out-0".to_string(), make_records("out-0", 30)),
    ];

    let candidates = detect_fanin_polarity_conflicts(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Output neurons should not be flagged for polarity conflict, got {} candidate(s)",
        candidates.len()
    );
}

// =============================================================================
// Test 8: Candidates sorted by conflict score
// =============================================================================

#[test]
fn test_candidates_sorted_by_conflict_score() {
    let creature = make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            neuron("in-1", "input", "IDENTITY"),
            neuron("in-2", "input", "IDENTITY"),
            neuron("in-3", "input", "IDENTITY"),
            // h-0: severe conflict (equal magnitude both sides)
            hidden("h-0", "TANH"),
            // h-1: moderate conflict (less balanced)
            hidden("h-1", "TANH"),
            output("out-0", "IDENTITY"),
        ],
        vec![
            // h-0: 2 × +3.0 and 2 × -3.0 = perfect balance = max conflict
            synapse("in-0", "h-0", 3.0),
            synapse("in-1", "h-0", 3.0),
            synapse("in-2", "h-0", -3.0),
            synapse("in-3", "h-0", -3.0),
            // h-1: 2 × +3.0 and 2 × -1.5 = less balanced
            synapse("in-0", "h-1", 3.0),
            synapse("in-1", "h-1", 3.0),
            synapse("in-2", "h-1", -1.5),
            synapse("in-3", "h-1", -1.5),
            // Outgoing
            synapse("h-0", "out-0", 1.0),
            synapse("h-1", "out-0", 1.0),
        ],
    );

    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for i in 0..4 {
        records.push((format!("in-{i}"), make_records(&format!("in-{i}"), 30)));
    }
    records.push(("h-0".to_string(), make_records("h-0", 30)));
    records.push(("h-1".to_string(), make_records("h-1", 30)));
    records.push(("out-0".to_string(), make_records("out-0", 30)));

    let candidates = detect_fanin_polarity_conflicts(&creature, &records);

    if candidates.len() >= 2 {
        for i in 1..candidates.len() {
            assert!(
                candidates[i - 1].conflict_score >= candidates[i].conflict_score,
                "Candidates should be sorted by conflict score (descending): {} < {}",
                candidates[i - 1].conflict_score,
                candidates[i].conflict_score
            );
        }
        // h-0 (perfect balance) should have higher conflict than h-1
        assert_eq!(
            candidates[0].neuron_uuid, "h-0",
            "Perfectly balanced conflict should rank first"
        );
    }
}
