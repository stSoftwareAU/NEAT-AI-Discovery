//! Tests for bounded range detection (Issue #395).
//!
//! ## TDD Plan
//!
//! 1. `test_empty_records_returns_empty` — no records → no candidates
//! 2. `test_single_neuron_narrow_range_detected` — TANH neuron using [0.1, 0.3] of [-1, 1] → detected
//! 3. `test_neuron_using_full_range_not_detected` — TANH neuron using [-0.9, 0.9] → not detected
//! 4. `test_unbounded_squash_not_flagged` — IDENTITY neuron with narrow range → not flagged
//! 5. `test_dead_neuron_excluded` — near-zero activation neuron → excluded (dead neuron territory)
//! 6. `test_input_neurons_excluded` — input neurons are not analysed for bounded range
//! 7. `test_output_neurons_excluded` — output neurons are not analysed for bounded range
//! 8. `test_constant_neurons_excluded` — constant neurons are not analysed
//! 9. `test_insufficient_samples_skipped` — too few samples → skip
//! 10. `test_sentinel_cluster_detected` — many samples at exact -1.0 → sentinel detected
//! 11. `test_no_sentinel_when_values_spread` — uniformly spread values → no sentinel
//! 12. `test_utilisation_ratio_computed_correctly` — verify range/theoretical calculation
//! 13. `test_candidates_sorted_by_improvement` — candidates ordered best-first
//! 14. `test_coordinated_candidates_contain_change_squash` — output has correct operations
//! 15. `test_multiple_neurons_analysed` — two narrow-range neurons → two candidates
//! 16. `test_logistic_narrow_range_detected` — LOGISTIC neuron using [0.45, 0.55] → detected
//! 17. `test_hard_tanh_narrow_range_detected` — HARD_TANH neuron using [0.0, 0.1] → detected

mod common;

use neat_ai_discovery::analysis::bounded_range::{
    bounded_range_to_coordinated_candidates, detect_bounded_range_neurons,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CoordinatedStructuralOpJson, CreatureJson};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn make_records(neuron_uuid: &str, activations: &[f32]) -> Vec<DiscoverRecord> {
    activations
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            DiscoverRecord::new(i as u32, neuron_uuid.to_string(), Some(a), a, vec![0.01])
        })
        .collect()
}

fn make_neuron_records(entries: &[(&str, &[f32])]) -> Vec<(String, Vec<DiscoverRecord>)> {
    entries
        .iter()
        .map(|(uuid, activations)| (uuid.to_string(), make_records(uuid, activations)))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_empty_records_returns_empty() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];
    let result = detect_bounded_range_neurons(&creature, &records);
    assert!(result.is_empty(), "No records → no candidates");
}

#[test]
fn test_single_neuron_narrow_range_detected() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );

    // TANH has range [-1, 1] = 2.0 total
    // Neuron uses [0.1, 0.3] = 0.2 range → 10% utilisation
    let activations: Vec<f32> = (0..50).map(|i| 0.1 + 0.2 * (i as f32 / 49.0)).collect();
    let records = make_neuron_records(&[("h1", &activations)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert_eq!(
        result.len(),
        1,
        "One narrow-range neuron should be detected"
    );
    assert_eq!(result[0].neuron_uuid, "h1");
    assert!(
        result[0].utilisation_ratio < 0.2,
        "Utilisation should be ~10%, got {:.2}",
        result[0].utilisation_ratio
    );
}

#[test]
fn test_neuron_using_full_range_not_detected() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );

    // Neuron uses [-0.9, 0.9] = 1.8 range → 90% utilisation
    let activations: Vec<f32> = (0..50).map(|i| -0.9 + 1.8 * (i as f32 / 49.0)).collect();
    let records = make_neuron_records(&[("h1", &activations)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert!(
        result.is_empty(),
        "Full-range neuron should not be detected"
    );
}

#[test]
fn test_unbounded_squash_not_flagged() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "IDENTITY"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );

    // IDENTITY is unbounded — no theoretical range to compare against
    let activations: Vec<f32> = (0..50).map(|i| 0.1 + 0.01 * (i as f32 / 49.0)).collect();
    let records = make_neuron_records(&[("h1", &activations)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert!(
        result.is_empty(),
        "Unbounded activation (IDENTITY) should not be flagged"
    );
}

#[test]
fn test_dead_neuron_excluded() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );

    // Near-zero activation — this is dead neuron territory
    let activations: Vec<f32> = (0..50).map(|_| 0.000001).collect();
    let records = make_neuron_records(&[("h1", &activations)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert!(
        result.is_empty(),
        "Dead neuron (near-zero activation) should be excluded"
    );
}

#[test]
fn test_input_neurons_excluded() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );

    // Even if input has narrow range, it should not be flagged
    let activations: Vec<f32> = (0..50).map(|i| 0.1 + 0.01 * (i as f32 / 49.0)).collect();
    let records = make_neuron_records(&[("input-0", &activations)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert!(result.is_empty(), "Input neurons should be excluded");
}

#[test]
fn test_output_neurons_excluded() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );

    // Output neurons should not be flagged for bounded range
    let activations: Vec<f32> = (0..50).map(|i| 0.1 + 0.01 * (i as f32 / 49.0)).collect();
    let records = make_neuron_records(&[("output-0", &activations)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert!(result.is_empty(), "Output neurons should be excluded");
}

#[test]
fn test_constant_neurons_excluded() {
    let creature = CreatureJson {
        neurons: vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::neuron("const-0", "constant", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::output("output-0", "TANH"),
        ],
        synapses: vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("const-0", "h1", 0.5),
            common::synapse("h1", "output-0", 1.0),
        ],
        input: 1,
        output: 1,
    };

    let activations: Vec<f32> = (0..50).map(|_| 1.0).collect();
    let records = make_neuron_records(&[("const-0", &activations)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert!(result.is_empty(), "Constant neurons should be excluded");
}

#[test]
fn test_insufficient_samples_skipped() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );

    // Only 5 samples — not enough
    let activations: Vec<f32> = vec![0.1, 0.15, 0.2, 0.25, 0.3];
    let records = make_neuron_records(&[("h1", &activations)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert!(result.is_empty(), "Too few samples should be skipped");
}

#[test]
fn test_utilisation_ratio_computed_correctly() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );

    // TANH theoretical range = 2.0 (from -1 to 1)
    // Observed range = 0.5 - 0.0 = 0.5
    // Utilisation = 0.5 / 2.0 = 0.25
    let activations: Vec<f32> = (0..50).map(|i| 0.5 * (i as f32 / 49.0)).collect();
    let records = make_neuron_records(&[("h1", &activations)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert!(!result.is_empty(), "25% utilisation should be detected");
    let ratio = result[0].utilisation_ratio;
    assert!(
        (ratio - 0.25).abs() < 0.02,
        "Utilisation ratio should be ~0.25, got {ratio:.3}"
    );
}

#[test]
fn test_candidates_sorted_by_improvement() {
    let creature = CreatureJson {
        neurons: vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::hidden("h2", "TANH"),
            common::output("output-0", "TANH"),
        ],
        synapses: vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("input-0", "h2", 1.0),
            common::synapse("h1", "output-0", 1.0),
            common::synapse("h2", "output-0", 1.0),
        ],
        input: 1,
        output: 1,
    };

    // h1: uses 10% of range (very restricted)
    let h1_acts: Vec<f32> = (0..50).map(|i| 0.1 + 0.2 * (i as f32 / 49.0)).collect();
    // h2: uses 25% of range (less restricted)
    let h2_acts: Vec<f32> = (0..50).map(|i| 0.5 * (i as f32 / 49.0)).collect();
    let records = make_neuron_records(&[("h1", &h1_acts), ("h2", &h2_acts)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert!(result.len() >= 2, "Both neurons should be detected");
    assert!(
        result[0].estimated_improvement >= result[1].estimated_improvement,
        "Candidates should be sorted best-first"
    );
}

#[test]
fn test_coordinated_candidates_contain_change_squash() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );

    let activations: Vec<f32> = (0..50).map(|i| 0.1 + 0.2 * (i as f32 / 49.0)).collect();
    let records = make_neuron_records(&[("h1", &activations)]);

    let candidates = detect_bounded_range_neurons(&creature, &records);
    assert!(!candidates.is_empty());

    let coordinated = bounded_range_to_coordinated_candidates(&candidates);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    // Verify at least one candidate has a ChangeSquash or SetBias operation
    let has_relevant_op = coordinated.iter().any(|c| {
        c.operations.iter().any(|op| {
            matches!(
                op,
                CoordinatedStructuralOpJson::ChangeSquash { .. }
                    | CoordinatedStructuralOpJson::SetBias { .. }
            )
        })
    });
    assert!(
        has_relevant_op,
        "Coordinated candidates should include ChangeSquash or SetBias operations"
    );
}

#[test]
fn test_multiple_neurons_analysed() {
    let creature = CreatureJson {
        neurons: vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "TANH"),
            common::hidden("h2", "LOGISTIC"),
            common::output("output-0", "TANH"),
        ],
        synapses: vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("input-0", "h2", 1.0),
            common::synapse("h1", "output-0", 1.0),
            common::synapse("h2", "output-0", 1.0),
        ],
        input: 1,
        output: 1,
    };

    // h1 (TANH): narrow range [0.1, 0.3]
    let h1_acts: Vec<f32> = (0..50).map(|i| 0.1 + 0.2 * (i as f32 / 49.0)).collect();
    // h2 (LOGISTIC): narrow range [0.45, 0.55] (out of [0, 1])
    let h2_acts: Vec<f32> = (0..50).map(|i| 0.45 + 0.1 * (i as f32 / 49.0)).collect();
    let records = make_neuron_records(&[("h1", &h1_acts), ("h2", &h2_acts)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert!(
        result.len() >= 2,
        "Both narrow-range neurons should be detected, got {}",
        result.len()
    );
}

#[test]
fn test_logistic_narrow_range_detected() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "LOGISTIC"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );

    // LOGISTIC has range [0, 1] = 1.0 total
    // Neuron uses [0.45, 0.55] = 0.1 range → 10% utilisation
    let activations: Vec<f32> = (0..50).map(|i| 0.45 + 0.1 * (i as f32 / 49.0)).collect();
    let records = make_neuron_records(&[("h1", &activations)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].neuron_uuid, "h1");
    assert!(
        result[0].utilisation_ratio < 0.2,
        "LOGISTIC utilisation should be ~10%"
    );
}

#[test]
fn test_hard_tanh_narrow_range_detected() {
    let creature = common::make_creature(
        vec![
            common::neuron("input-0", "input", "IDENTITY"),
            common::hidden("h1", "HARD_TANH"),
            common::output("output-0", "TANH"),
        ],
        vec![
            common::synapse("input-0", "h1", 1.0),
            common::synapse("h1", "output-0", 1.0),
        ],
    );

    // HARD_TANH has range [-1, 1] = 2.0 total
    // Neuron uses [0.0, 0.1] = 0.1 range → 5% utilisation
    let activations: Vec<f32> = (0..50).map(|i| 0.1 * (i as f32 / 49.0)).collect();
    let records = make_neuron_records(&[("h1", &activations)]);

    let result = detect_bounded_range_neurons(&creature, &records);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].neuron_uuid, "h1");
    assert!(
        result[0].utilisation_ratio < 0.1,
        "HARD_TANH utilisation should be ~5%"
    );
}
