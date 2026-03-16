//! Tests for Issue #550: Local minimum escape — weight magnitude reset for stuck
//! synapses.
//!
//! ## TDD Plan
//! 1. Test detection identifies stuck synapses with high error contribution
//! 2. Test generates `setWeight` candidates with exploratory weight values
//! 3. Test weight candidates span a wide range of values (sign flip, magnitude change)
//! 4. Test no candidates when synapses are not stuck (low error, varying weights)
//! 5. Test insufficient samples produce no candidates
//! 6. Test multiple stuck synapses produce ranked candidates
//! 7. Test comment references issue number

use neat_ai_discovery::analysis::detection::weight_magnitude_reset::{
    detect_stuck_synapse_weight_resets, stuck_synapses_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

// =============================================================================
// Helpers
// =============================================================================

fn make_record(uuid: &str, idx: u32, value: f32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(value),
        activation,
        errors: vec![error],
    }
}

fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    CreatureJson {
        neurons,
        synapses,
        input: 2,
        output: 1,
    }
}

fn input_neuron(uuid: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "input".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn hidden_neuron(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "hidden".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

fn output_neuron(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "output".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Create records simulating a stuck synapse pattern:
/// - Source neuron has active (non-zero) activations
/// - Target neuron has persistently high error
/// - Weight is stable (low variance across observations)
fn make_stuck_synapse_records(
    source_uuid: &str,
    target_uuid: &str,
    sample_count: usize,
) -> Vec<(String, Vec<DiscoverRecord>)> {
    let source_records: Vec<DiscoverRecord> = (0..sample_count)
        .map(|i| {
            // Stable non-zero activations — the source is contributing signal
            let activation = 0.5 + (i as f32 * 0.01).sin() * 0.05;
            make_record(source_uuid, i as u32, activation, activation, 0.0)
        })
        .collect();

    let target_records: Vec<DiscoverRecord> = (0..sample_count)
        .map(|i| {
            // Persistently high error with low variance — stuck in local minimum
            let error = 0.4 + (i as f32 * 0.01).sin() * 0.02;
            let activation = 0.3;
            make_record(target_uuid, i as u32, 0.3, activation, error)
        })
        .collect();

    vec![
        (source_uuid.to_string(), source_records),
        (target_uuid.to_string(), target_records),
    ]
}

// =============================================================================
// 1. Detection identifies stuck synapses with high error contribution
// =============================================================================

#[test]
fn test_detects_stuck_synapse_with_high_error() {
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            input_neuron("input-2"),
            hidden_neuron("hidden-1", "TANH"),
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.2),
            synapse("input-2", "hidden-1", 0.8),
            synapse("hidden-1", "output-1", 0.5),
        ],
    );

    let neuron_records = make_stuck_synapse_records("hidden-1", "output-1", 40);

    let candidates = detect_stuck_synapse_weight_resets(&creature, &neuron_records);
    assert!(
        !candidates.is_empty(),
        "Should detect stuck synapse with persistently high error"
    );

    // The candidate should target the synapse from hidden-1 to output-1
    let found = candidates
        .iter()
        .any(|c| c.from_neuron_uuid == "hidden-1" && c.to_neuron_uuid == "output-1");
    assert!(
        found,
        "Should identify the stuck synapse hidden-1 → output-1"
    );
}

// =============================================================================
// 2. Generates setWeight candidates with exploratory weight values
// =============================================================================

#[test]
fn test_generates_set_weight_candidates() {
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            input_neuron("input-2"),
            hidden_neuron("hidden-1", "TANH"),
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.2),
            synapse("input-2", "hidden-1", 0.8),
            synapse("hidden-1", "output-1", 0.5),
        ],
    );

    let neuron_records = make_stuck_synapse_records("hidden-1", "output-1", 40);

    let candidates = detect_stuck_synapse_weight_resets(&creature, &neuron_records);
    assert!(!candidates.is_empty());

    let coordinated = stuck_synapses_to_coordinated_candidates(&candidates);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    // Each candidate should contain a setWeight operation
    for cc in &coordinated {
        let ops_json = serde_json::to_string(&cc.operations).unwrap();
        assert!(
            ops_json.contains("setWeight"),
            "Candidate must contain setWeight operation, got: {ops_json}"
        );
    }
}

// =============================================================================
// 3. Weight candidates span a wide range of values
// =============================================================================

#[test]
fn test_weight_candidates_span_wide_range() {
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            hidden_neuron("hidden-1", "TANH"),
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.0),
            synapse("hidden-1", "output-1", 0.5),
        ],
    );

    let neuron_records = make_stuck_synapse_records("hidden-1", "output-1", 40);

    let candidates = detect_stuck_synapse_weight_resets(&creature, &neuron_records);
    assert!(!candidates.is_empty());

    let coordinated = stuck_synapses_to_coordinated_candidates(&candidates);
    assert!(
        coordinated.len() >= 2,
        "Should produce multiple exploratory weight candidates, got: {}",
        coordinated.len()
    );

    // Extract the proposed weights from all candidates
    let mut weights: Vec<f32> = Vec::new();
    for cc in &coordinated {
        for op in &cc.operations {
            let json = serde_json::to_string(op).unwrap();
            if json.contains("setWeight") {
                // Extract weight from JSON
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json)
                    && let Some(w) = val.get("weight").and_then(serde_json::Value::as_f64)
                {
                    weights.push(w as f32);
                }
            }
        }
    }

    assert!(
        !weights.is_empty(),
        "Should have extracted weights from candidates"
    );

    // Verify the weights span a wide range: should include both positive and negative,
    // or significantly different magnitudes
    let has_sign_variation = weights.iter().any(|w| *w < 0.0) && weights.iter().any(|w| *w > 0.0);
    let min_w = weights.iter().copied().fold(f32::INFINITY, f32::min);
    let max_w = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let has_magnitude_variation = (max_w - min_w).abs() > 0.5;

    assert!(
        has_sign_variation || has_magnitude_variation,
        "Weight candidates should span a wide range (sign flip or magnitude change). Weights: {weights:?}"
    );
}

// =============================================================================
// 4. No candidates when synapses are not stuck
// =============================================================================

#[test]
fn test_no_candidates_for_healthy_synapses() {
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            hidden_neuron("hidden-1", "TANH"),
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.0),
            synapse("hidden-1", "output-1", 0.5),
        ],
    );

    // Low, varying error — the synapse is healthy and converging
    let source_records: Vec<DiscoverRecord> = (0..40)
        .map(|i| {
            let activation = 0.5 + (i as f32 * 0.1).sin() * 0.3;
            make_record("hidden-1", i, activation, activation, 0.0)
        })
        .collect();
    let target_records: Vec<DiscoverRecord> = (0..40)
        .map(|i| {
            // Low error — network is performing well
            let error = 0.005 + (i as f32 * 0.05).sin() * 0.003;
            make_record("output-1", i, 0.3, 0.3, error)
        })
        .collect();

    let neuron_records = vec![
        ("hidden-1".to_string(), source_records),
        ("output-1".to_string(), target_records),
    ];

    let candidates = detect_stuck_synapse_weight_resets(&creature, &neuron_records);
    assert!(
        candidates.is_empty(),
        "Healthy synapses should produce no candidates, got: {}",
        candidates.len()
    );
}

// =============================================================================
// 5. Insufficient samples produce no candidates
// =============================================================================

#[test]
fn test_weight_magnitude_reset_insufficient_samples_no_candidates() {
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            hidden_neuron("hidden-1", "TANH"),
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.0),
            synapse("hidden-1", "output-1", 0.5),
        ],
    );

    // Only 5 samples — far below the minimum threshold
    let neuron_records = make_stuck_synapse_records("hidden-1", "output-1", 5);

    let candidates = detect_stuck_synapse_weight_resets(&creature, &neuron_records);
    assert!(
        candidates.is_empty(),
        "Insufficient samples should produce no candidates"
    );
}

// =============================================================================
// 6. Multiple stuck synapses produce ranked candidates
// =============================================================================

#[test]
fn test_multiple_stuck_synapses_ranked() {
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            input_neuron("input-2"),
            hidden_neuron("hidden-1", "TANH"),
            hidden_neuron("hidden-2", "RELU"),
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.0),
            synapse("input-2", "hidden-2", 0.8),
            synapse("hidden-1", "output-1", 0.5),
            synapse("hidden-2", "output-1", 0.3),
        ],
    );

    // Both hidden→output synapses are stuck, but hidden-1→output-1 has higher error
    let source1_records: Vec<DiscoverRecord> = (0..40)
        .map(|i| {
            let activation = 0.6 + (i as f32 * 0.01).sin() * 0.04;
            make_record("hidden-1", i, activation, activation, 0.0)
        })
        .collect();
    let source2_records: Vec<DiscoverRecord> = (0..40)
        .map(|i| {
            let activation = 0.4 + (i as f32 * 0.01).sin() * 0.03;
            make_record("hidden-2", i, activation, activation, 0.0)
        })
        .collect();
    let target_records: Vec<DiscoverRecord> = (0..40)
        .map(|i| {
            // High persistent error
            let error = 0.5 + (i as f32 * 0.01).sin() * 0.02;
            make_record("output-1", i, 0.3, 0.3, error)
        })
        .collect();

    let neuron_records = vec![
        ("hidden-1".to_string(), source1_records),
        ("hidden-2".to_string(), source2_records),
        ("output-1".to_string(), target_records),
    ];

    let candidates = detect_stuck_synapse_weight_resets(&creature, &neuron_records);
    assert!(
        candidates.len() >= 2,
        "Should detect multiple stuck synapses, got: {}",
        candidates.len()
    );

    // Candidates should be sorted by estimated improvement (descending)
    for window in candidates.windows(2) {
        assert!(
            window[0].estimated_improvement >= window[1].estimated_improvement,
            "Candidates should be sorted by estimated improvement (descending)"
        );
    }
}

// =============================================================================
// 7. Comment references issue number
// =============================================================================

#[test]
fn test_comment_references_issue_number() {
    let creature = make_creature(
        vec![
            input_neuron("input-1"),
            hidden_neuron("hidden-1", "TANH"),
            output_neuron("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.0),
            synapse("hidden-1", "output-1", 0.5),
        ],
    );

    let neuron_records = make_stuck_synapse_records("hidden-1", "output-1", 40);

    let candidates = detect_stuck_synapse_weight_resets(&creature, &neuron_records);
    assert!(!candidates.is_empty());

    let coordinated = stuck_synapses_to_coordinated_candidates(&candidates);
    assert!(!coordinated.is_empty());

    let comment = coordinated[0].comment.as_deref().unwrap_or("");
    assert!(
        comment.contains("Issue #550"),
        "Comment should reference Issue #550, got: {comment}"
    );
}
