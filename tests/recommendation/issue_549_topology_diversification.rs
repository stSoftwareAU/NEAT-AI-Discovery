//! Tests for Issue #549: Topology diversification for structural jumps.
//!
//! Detects when the network topology is too simple for the problem — all paths
//! from input to output are too short (no intermediate processing) and error
//! remains high despite healthy individual neuron metrics. Recommends `addNeuron`
//! candidates at strategic insertion points.
//!
//! ## TDD Plan
//! 1. Test detection of under-connected direct-path network with high error
//! 2. Test no detection for well-connected network with hidden layers
//! 3. Test no detection for low-error direct-path network (already converged)
//! 4. Test insufficient samples returns empty
//! 5. Test coordinated candidate conversion produces addNeuron operations
//! 6. Test multiple output neurons with mixed structural needs
//! 7. Test no false positive when error is parametric (high variance per neuron)

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::common::{hidden, make_creature, neuron, output, synapse};
use neat_ai_discovery::analysis::detection::topology_diversification::{
    detect_topology_diversification_candidates, topology_diversification_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helpers
// =============================================================================

fn make_record(uuid: &str, idx: u32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![error],
    }
}

fn make_records_for(uuid: &str, count: u32, activation: f32, error: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| make_record(uuid, i, activation + (i as f32 * 0.01), error))
        .collect()
}

// =============================================================================
// 1. Under-connected network: direct input→output with high error
// =============================================================================

#[test]
fn test_detects_insufficient_topology_direct_paths() {
    // Network: input-1 → output-1, input-2 → output-1
    // No hidden neurons, high error → topology is too simple
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            output("output-1", "TANH"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-2", "output-1", -0.3),
        ],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-1".to_string(),
            make_records_for("input-1", 40, 0.5, 0.0),
        ),
        (
            "input-2".to_string(),
            make_records_for("input-2", 40, -0.2, 0.0),
        ),
        (
            "output-1".to_string(),
            (0..40)
                .map(|i| {
                    // High consistent error — structure is insufficient
                    make_record("output-1", i, 0.3, 0.25)
                })
                .collect(),
        ),
    ];

    let candidates = detect_topology_diversification_candidates(&creature, &records);

    assert!(
        !candidates.is_empty(),
        "Should detect insufficient topology for direct input→output with high error"
    );
    assert!(
        candidates[0].estimated_improvement > 0.0,
        "Estimated improvement should be positive"
    );
}

// =============================================================================
// 2. Well-connected network: no detection needed
// =============================================================================

#[test]
fn test_no_detection_for_well_connected_network() {
    // Network with multiple hidden layers — topology is sufficiently complex
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            hidden("hidden-1", "TANH"),
            hidden("hidden-2", "TANH"),
            hidden("hidden-3", "TANH"),
            output("output-1", "TANH"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("input-2", "hidden-2", 0.3),
            synapse("hidden-1", "hidden-3", 0.4),
            synapse("hidden-2", "hidden-3", 0.2),
            synapse("hidden-3", "output-1", 0.6),
        ],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-1".to_string(),
            make_records_for("input-1", 40, 0.5, 0.0),
        ),
        (
            "input-2".to_string(),
            make_records_for("input-2", 40, -0.2, 0.0),
        ),
        (
            "hidden-1".to_string(),
            make_records_for("hidden-1", 40, 0.4, 0.1),
        ),
        (
            "hidden-2".to_string(),
            make_records_for("hidden-2", 40, 0.2, 0.05),
        ),
        (
            "hidden-3".to_string(),
            make_records_for("hidden-3", 40, 0.3, 0.08),
        ),
        (
            "output-1".to_string(),
            make_records_for("output-1", 40, 0.3, 0.15),
        ),
    ];

    let candidates = detect_topology_diversification_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Should NOT detect insufficient topology when network has multiple hidden layers"
    );
}

// =============================================================================
// 3. Low error: no detection even with simple topology
// =============================================================================

#[test]
fn test_no_detection_for_low_error_direct_path() {
    // Simple topology but low error → problem is solved, no need for more complexity
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            output("output-1", "TANH"),
        ],
        vec![synapse("input-1", "output-1", 0.8)],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-1".to_string(),
            make_records_for("input-1", 40, 0.5, 0.0),
        ),
        (
            "output-1".to_string(),
            // Very low error — the simple topology is adequate
            make_records_for("output-1", 40, 0.4, 0.005),
        ),
    ];

    let candidates = detect_topology_diversification_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Should NOT flag simple topology when error is already low"
    );
}

// =============================================================================
// 4. Insufficient samples returns empty
// =============================================================================

#[test]
fn test_topology_diversification_insufficient_samples_returns_empty() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            output("output-1", "TANH"),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-1".to_string(),
            make_records_for("input-1", 5, 0.5, 0.0),
        ),
        (
            "output-1".to_string(),
            make_records_for("output-1", 5, 0.3, 0.3),
        ),
    ];

    let candidates = detect_topology_diversification_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Should return empty when sample count is insufficient"
    );
}

// =============================================================================
// 5. Coordinated candidate conversion produces addNeuron
// =============================================================================

#[test]
fn test_coordinated_candidates_contain_add_neuron() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            output("output-1", "TANH"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-2", "output-1", -0.3),
        ],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-1".to_string(),
            make_records_for("input-1", 40, 0.5, 0.0),
        ),
        (
            "input-2".to_string(),
            make_records_for("input-2", 40, -0.2, 0.0),
        ),
        (
            "output-1".to_string(),
            (0..40)
                .map(|i| make_record("output-1", i, 0.3, 0.25))
                .collect(),
        ),
    ];

    let candidates = detect_topology_diversification_candidates(&creature, &records);
    assert!(
        !candidates.is_empty(),
        "Precondition: should have candidates"
    );

    let coordinated = topology_diversification_to_coordinated_candidates(&candidates, &creature);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );
    assert!(
        coordinated[0].expected_creature_score_gain > 0.0,
        "Expected score gain should be positive"
    );

    // Should contain addNeuron operation
    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("addNeuron"),
        "Should contain addNeuron operation, got: {ops_json}"
    );
    // Should also contain addSynapse to wire up the new neuron
    assert!(
        ops_json.contains("addSynapse"),
        "Should contain addSynapse operations to wire the neuron, got: {ops_json}"
    );

    // Comment should reference issue #549
    assert!(
        coordinated[0].comment.as_ref().unwrap().contains("549"),
        "Comment should reference issue #549, got: {:?}",
        coordinated[0].comment
    );
}

// =============================================================================
// 6. Multiple outputs with mixed structural needs
// =============================================================================

#[test]
fn test_multiple_outputs_mixed_results() {
    // Two outputs: one has all direct paths with high error, another has hidden layer path
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            hidden("hidden-1", "TANH"),
            output("output-direct", "TANH"),
            output("output-deep", "TANH"),
        ],
        vec![
            // output-direct only has direct input connections
            synapse("input-1", "output-direct", 0.4),
            synapse("input-2", "output-direct", 0.3),
            // output-deep has hidden layer path
            synapse("input-1", "hidden-1", 0.5),
            synapse("input-2", "hidden-1", 0.3),
            synapse("hidden-1", "output-deep", 0.6),
        ],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-1".to_string(),
            make_records_for("input-1", 40, 0.5, 0.0),
        ),
        (
            "input-2".to_string(),
            make_records_for("input-2", 40, -0.2, 0.0),
        ),
        (
            "hidden-1".to_string(),
            make_records_for("hidden-1", 40, 0.4, 0.05),
        ),
        (
            "output-direct".to_string(),
            (0..40)
                .map(|i| make_record("output-direct", i, 0.3, 0.28))
                .collect(),
        ),
        (
            "output-deep".to_string(),
            make_records_for("output-deep", 40, 0.3, 0.15),
        ),
    ];

    let candidates = detect_topology_diversification_candidates(&creature, &records);

    // Should detect the output with only direct paths and high error
    let direct_candidates: Vec<_> = candidates
        .iter()
        .filter(|c| c.output_neuron_uuid == "output-direct")
        .collect();

    assert!(
        !direct_candidates.is_empty(),
        "Should detect insufficient topology for output-direct (only direct input paths)"
    );
}

// =============================================================================
// 7. No false positive when neurons have high individual error variance
// =============================================================================

#[test]
fn test_no_detection_when_individual_neurons_have_issues() {
    // Network with a single hidden layer but the hidden neuron has high individual
    // error variance — the problem is parametric (the hidden neuron needs tuning),
    // not structural.
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("hidden-1", "TANH"),
            output("output-1", "TANH"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.6),
        ],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-1".to_string(),
            make_records_for("input-1", 40, 0.5, 0.0),
        ),
        (
            "hidden-1".to_string(),
            (0..40)
                .map(|i| {
                    // Hidden neuron has wildly varying errors — parametric issue
                    let error = if i % 2 == 0 { 0.5 } else { 0.01 };
                    make_record("hidden-1", i, 0.3 + (i as f32 * 0.01), error)
                })
                .collect(),
        ),
        (
            "output-1".to_string(),
            (0..40)
                .map(|i| make_record("output-1", i, 0.3, 0.2))
                .collect(),
        ),
    ];

    let candidates = detect_topology_diversification_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Should NOT detect topology issue when a hidden neuron already exists on the path \
         and has high error variance (parametric, not structural)"
    );
}

// =============================================================================
// 8. Empty inputs
// =============================================================================

#[test]
fn test_empty_creature_returns_empty() {
    let creature = make_creature(vec![], vec![]);
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let candidates = detect_topology_diversification_candidates(&creature, &records);
    assert!(
        candidates.is_empty(),
        "Empty creature should produce no candidates"
    );
}
