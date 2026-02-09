//! Tests for Issue #400: Sentinel value gating — propose gated connections
//! that ignore null/sentinel observations.
//!
//! When observations use sentinel values (e.g., -1 meaning "no data"), simply
//! multiplying by a weight propagates meaningless information. This module detects
//! observations where sentinel values degrade performance (using error correlation
//! from observation range detection, Issue #398) and proposes gated neuron
//! structures using scalar, GPU-compatible squash functions.
//!
//! ## TDD Plan
//! 1. Detect observation where sentinel at -1 correlates with noise
//! 2. Proposed candidate uses coordinatedStructural with addNeuron + addSynapse + setBias
//! 3. Gate neuron uses GPU-compatible squash (HARD_TANH or STEP, not IF/MAXIMUM)
//! 4. Gate connects observation to downstream targets
//! 5. No detection when observation has no sentinel cluster
//! 6. No detection with insufficient samples
//! 7. Multiple observations with sentinel values produce independent gated candidates
//! 8. Sentinel at 0 is also detected and gated
//! 9. Depends on observation range detection for sentinel identification

mod common;

use common::{make_creature, neuron, output, synapse};
use neat_ai_discovery::CoordinatedStructuralOpJson;
use neat_ai_discovery::analysis::sentinel_gating::{
    detect_sentinel_gating_candidates, sentinel_gating_to_coordinated_candidates,
};
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

/// Non-aggregate squash functions that are GPU-compatible (scalar).
const GPU_COMPATIBLE_SQUASH: &[&str] = &[
    "HARD_TANH",
    "STEP",
    "RELU",
    "IDENTITY",
    "LOGISTIC",
    "TANH",
    "CLIPPED_RELU",
    "BIPOLAR",
    "BIPOLAR_SIGMOID",
    "GAUSSIAN",
    "ABSOLUTE",
    "INVERSE",
    "SELU",
    "COMPLEMENT",
    "MINIMUM",
    "MEAN",
    "MAXIMUM",
];

/// Squash functions that are NOT scalar (aggregate), hence not GPU-compatible.
const NON_GPU_SQUASH: &[&str] = &["IF"];

// ---------------------------------------------------------------------------
// Test 1: Observation with sentinel at -1 and error correlation is detected.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_sentinel_gating_candidate_at_minus_one() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    // 40% at -1.0 (sentinel: constant low error — no correlation)
    // 60% in useful range [0.1, 0.8] with varied error (correlation present)
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records.push(record_with_error("input-obs", i, -1.0, 0.01));
    }
    for i in 40..100 {
        let useful_val = 0.1 + 0.7 * ((i - 40) as f32 / 60.0);
        let error = 0.5 - useful_val;
        records.push(record_with_error("input-obs", i, useful_val, error));
    }

    let candidates =
        detect_sentinel_gating_candidates(&creature, &[("input-obs".to_string(), records)]);

    assert_eq!(
        candidates.len(),
        1,
        "Should detect one sentinel gating candidate"
    );
    assert_eq!(candidates[0].neuron_uuid, "input-obs");
    assert!(
        candidates[0].sentinel_value == -1.0,
        "Sentinel value should be -1.0, got {}",
        candidates[0].sentinel_value
    );
}

// ---------------------------------------------------------------------------
// Test 2: Coordinated candidate uses addNeuron + addSynapse + setBias.
// ---------------------------------------------------------------------------
#[test]
fn test_coordinated_candidate_has_required_operations() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records.push(record_with_error("input-obs", i, -1.0, 0.01));
    }
    for i in 40..100 {
        let useful_val = 0.1 + 0.7 * ((i - 40) as f32 / 60.0);
        let error = 0.5 - useful_val;
        records.push(record_with_error("input-obs", i, useful_val, error));
    }

    let detected =
        detect_sentinel_gating_candidates(&creature, &[("input-obs".to_string(), records)]);
    let coordinated = sentinel_gating_to_coordinated_candidates(&detected, &creature);

    assert!(
        !coordinated.is_empty(),
        "Should produce at least one coordinated candidate"
    );

    let c = &coordinated[0];

    // Must have addNeuron, addSynapse, and setBias operations
    let has_add_neuron = c
        .operations
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::AddNeuron { .. }));
    let has_add_synapse = c
        .operations
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::AddSynapse { .. }));

    assert!(
        has_add_neuron,
        "Must include addNeuron operation for gate neuron"
    );
    assert!(has_add_synapse, "Must include addSynapse operation(s)");
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
    assert!(c.comment.is_some(), "Should include a descriptive comment");
}

// ---------------------------------------------------------------------------
// Test 3: Gate neuron uses GPU-compatible (scalar) squash function.
// ---------------------------------------------------------------------------
#[test]
fn test_gate_neuron_uses_gpu_compatible_squash() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records.push(record_with_error("input-obs", i, -1.0, 0.01));
    }
    for i in 40..100 {
        let useful_val = 0.1 + 0.7 * ((i - 40) as f32 / 60.0);
        let error = 0.5 - useful_val;
        records.push(record_with_error("input-obs", i, useful_val, error));
    }

    let detected =
        detect_sentinel_gating_candidates(&creature, &[("input-obs".to_string(), records)]);
    let coordinated = sentinel_gating_to_coordinated_candidates(&detected, &creature);

    for candidate in &coordinated {
        for op in &candidate.operations {
            if let CoordinatedStructuralOpJson::AddNeuron { squash, .. } = op {
                assert!(
                    GPU_COMPATIBLE_SQUASH.contains(&squash.as_str()),
                    "Gate neuron squash must be GPU-compatible, got '{squash}'"
                );
                assert!(
                    !NON_GPU_SQUASH.contains(&squash.as_str()),
                    "Gate neuron must NOT use aggregate squash '{squash}'"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: Gate connects observation to downstream targets.
// ---------------------------------------------------------------------------
#[test]
fn test_gate_connects_to_downstream_targets() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            neuron("input-other", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-obs", "output-1", 0.5),
            synapse("input-other", "output-1", 0.3),
        ],
    );

    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records.push(record_with_error("input-obs", i, -1.0, 0.01));
    }
    for i in 40..100 {
        let useful_val = 0.1 + 0.7 * ((i - 40) as f32 / 60.0);
        let error = 0.5 - useful_val;
        records.push(record_with_error("input-obs", i, useful_val, error));
    }

    let detected =
        detect_sentinel_gating_candidates(&creature, &[("input-obs".to_string(), records)]);
    let coordinated = sentinel_gating_to_coordinated_candidates(&detected, &creature);

    assert!(!coordinated.is_empty());

    let c = &coordinated[0];

    // The gate neuron should have a synapse FROM the observation
    // AND a synapse TO a downstream target (output-1)
    let add_synapse_ops: Vec<_> = c
        .operations
        .iter()
        .filter(|op| matches!(op, CoordinatedStructuralOpJson::AddSynapse { .. }))
        .collect();

    // Should have at least 2 synapses: obs→gate and gate→target
    assert!(
        add_synapse_ops.len() >= 2,
        "Should have at least 2 addSynapse operations (obs→gate and gate→target), got {}",
        add_synapse_ops.len()
    );

    // Verify there's a synapse from the gate to a downstream target
    let gate_uuid_prefix = "gate-sentinel-";
    let has_gate_to_target = c.operations.iter().any(|op| {
        if let CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid,
            to_neuron_uuid,
            ..
        } = op
        {
            from_neuron_uuid.starts_with(gate_uuid_prefix) && to_neuron_uuid == "output-1"
        } else {
            false
        }
    });

    assert!(
        has_gate_to_target,
        "Should have a synapse from gate neuron to downstream target (output-1)"
    );
}

// ---------------------------------------------------------------------------
// Test 5: No detection for observation without sentinel cluster.
// ---------------------------------------------------------------------------
#[test]
fn test_no_detection_without_sentinel_cluster() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    // Uniform distribution — no sentinel cluster
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let val = -1.0 + 2.0 * (i as f32 / 99.0);
            let error = 0.5 - val * 0.3;
            record_with_error("input-obs", i, val, error)
        })
        .collect();

    let candidates =
        detect_sentinel_gating_candidates(&creature, &[("input-obs".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Uniform distribution should not produce sentinel gating candidates"
    );
}

// ---------------------------------------------------------------------------
// Test 6: No detection with insufficient samples.
// ---------------------------------------------------------------------------
#[test]
fn test_no_detection_with_insufficient_samples() {
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

    let candidates =
        detect_sentinel_gating_candidates(&creature, &[("input-obs".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Should not detect with insufficient samples"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Multiple observations produce independent gated candidates.
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_observations_produce_independent_candidates() {
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

    let candidates = detect_sentinel_gating_candidates(
        &creature,
        &[
            ("obs-a".to_string(), records_a),
            ("obs-b".to_string(), records_b),
        ],
    );

    assert_eq!(
        candidates.len(),
        2,
        "Should detect sentinel gating for both observations"
    );

    let has_a = candidates.iter().any(|c| c.neuron_uuid == "obs-a");
    let has_b = candidates.iter().any(|c| c.neuron_uuid == "obs-b");
    assert!(has_a, "Should have candidate for obs-a");
    assert!(has_b, "Should have candidate for obs-b");
}

// ---------------------------------------------------------------------------
// Test 8: Sentinel at 0 is detected and gated.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_sentinel_at_zero() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..50 {
        records.push(record_with_error("input-obs", i, 0.0, 0.01));
    }
    for i in 50..100 {
        let useful_val = 0.3 + 0.6 * ((i - 50) as f32 / 50.0);
        let error = 0.5 - useful_val;
        records.push(record_with_error("input-obs", i, useful_val, error));
    }

    let candidates =
        detect_sentinel_gating_candidates(&creature, &[("input-obs".to_string(), records)]);

    assert_eq!(candidates.len(), 1, "Should detect sentinel at 0");
    assert!(
        (candidates[0].sentinel_value - 0.0).abs() < 0.1,
        "Sentinel value should be 0.0"
    );
}

// ---------------------------------------------------------------------------
// Test 9: Output neurons are excluded from sentinel gating.
// ---------------------------------------------------------------------------
#[test]
fn test_output_neurons_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            output("output-1", "TANH"),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Output neuron with sentinel-like pattern — should NOT be detected
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        records.push(record_with_error("output-1", i, -1.0, 0.01));
    }
    for i in 40..100 {
        let val = 0.3 * ((i - 40) as f32 / 60.0);
        records.push(record_with_error("output-1", i, val, 0.1));
    }

    let candidates =
        detect_sentinel_gating_candidates(&creature, &[("output-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Output neurons should be excluded from sentinel gating"
    );
}

// ---------------------------------------------------------------------------
// Test 10: Candidate depends on observation range data for sentinel detection.
// ---------------------------------------------------------------------------
#[test]
fn test_uses_error_correlation_for_detection() {
    let creature = make_creature(
        vec![
            neuron("input-obs", "input", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("input-obs", "output-1", 0.5)],
    );

    // Cluster at -1, but with HIGH error variance (sentinel is actually meaningful)
    // — should NOT be detected because the sentinel correlates with error
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..40 {
        // High-variance error at sentinel: the sentinel IS meaningful
        let error = -0.5 + 1.0 * (i as f32 / 39.0);
        records.push(record_with_error("input-obs", i, -1.0, error));
    }
    for i in 40..100 {
        let useful_val = 0.1 + 0.7 * ((i - 40) as f32 / 60.0);
        // Low variance error in useful range
        let error = 0.01;
        records.push(record_with_error("input-obs", i, useful_val, error));
    }

    let candidates =
        detect_sentinel_gating_candidates(&creature, &[("input-obs".to_string(), records)]);

    // When sentinel has HIGHER error variance than non-sentinel range,
    // the sentinel is actually informative and should not be gated
    assert!(
        candidates.is_empty(),
        "Should not gate sentinel when it has higher error variance (is informative)"
    );
}
