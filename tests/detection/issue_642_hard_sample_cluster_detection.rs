//! Tests for Issue #642: Hard sample clustering detector (cross-network high-error observations).
//!
//! When groups of observations (`obs_index` values) are consistently high-error across
//! all output neurons, it indicates a systematic structural gap — the network lacks capacity
//! to handle that region of the input space. This module detects those hard sample clusters
//! and produces targeted structural candidates.
//!
//! ## TDD Plan
//! 1. Create test network with clear easy/hard observation split
//! 2. Verify detection identifies hard sample clusters by `obs_index`
//! 3. Verify targeted structural candidates are produced
//! 4. Test single-output network (degenerate case)
//! 5. Test no hard samples (uniform error)
//! 6. Test insufficient samples

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::hard_sample_cluster::{
    HardSampleCluster, HardSampleClusterConfig, detect_hard_sample_clusters,
    hard_sample_clusters_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a `DiscoverRecord`.
fn record(neuron_uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
    }
}

/// Helper: build a minimal creature.
fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    let input_count = neurons.iter().filter(|n| n.neuron_type == "input").count();
    let output_count = neurons.iter().filter(|n| n.neuron_type == "output").count();
    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: output_count,
    }
}

fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
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

/// Test 1: Clear easy/hard split across multiple output neurons is detected.
///
/// Observations 0-49 have low error; observations 50-99 have high error across
/// all output neurons. The detector should identify obs 50-99 as a hard cluster.
#[test]
fn test_detects_hard_sample_cluster_across_outputs() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-2", "output-2", 0.3),
        ],
    );

    let mut all_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // Both outputs have high error on observations 50-99
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                let error = if i >= 50 { 0.8 } else { 0.05 };
                record(output_id, i, 0.5, vec![error])
            })
            .collect();
        all_records.push((output_id.to_string(), records));
    }

    // Input records for activation pattern analysis
    for input_id in &["input-1", "input-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                let activation = if i >= 50 { 0.9 } else { 0.1 };
                record(input_id, i, activation, vec![])
            })
            .collect();
        all_records.push((input_id.to_string(), records));
    }

    let config = HardSampleClusterConfig::default();
    let clusters = detect_hard_sample_clusters(&creature, &all_records, &config);

    assert!(
        !clusters.is_empty(),
        "Should detect at least one hard sample cluster"
    );

    let cluster = &clusters[0];
    assert!(
        cluster.hard_obs_indices.len() >= 20,
        "Hard cluster should contain a significant number of observations, got {}",
        cluster.hard_obs_indices.len()
    );

    // Most hard observations should be in the 50-99 range
    let high_range_count = cluster
        .hard_obs_indices
        .iter()
        .filter(|&&idx| idx >= 50)
        .count();
    assert!(
        high_range_count > cluster.hard_obs_indices.len() / 2,
        "Most hard observations should be in the high-error range"
    );

    assert!(
        cluster.mean_error > 0.1,
        "Mean error of hard cluster should be significant, got {}",
        cluster.mean_error
    );
}

/// Test 2: Single-output network works as a degenerate case.
#[test]
fn test_single_output_network_degenerate_case() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i >= 70 { 0.9 } else { 0.02 };
            record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let input_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i >= 70 { 0.8 } else { 0.2 };
            record("input-1", i, activation, vec![])
        })
        .collect();

    let all_records = vec![
        ("output-1".to_string(), records),
        ("input-1".to_string(), input_records),
    ];

    let config = HardSampleClusterConfig::default();
    let clusters = detect_hard_sample_clusters(&creature, &all_records, &config);

    assert!(
        !clusters.is_empty(),
        "Single-output network should still detect hard sample clusters"
    );
}

/// Test 3: Uniform error produces no hard clusters.
#[test]
fn test_uniform_error_no_clusters() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
    );

    let mut all_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| record(output_id, i, 0.5, vec![0.1]))
            .collect();
        all_records.push((output_id.to_string(), records));
    }

    let config = HardSampleClusterConfig::default();
    let clusters = detect_hard_sample_clusters(&creature, &all_records, &config);

    assert!(
        clusters.is_empty(),
        "Uniform error should not produce hard sample clusters"
    );
}

/// Test 4: Insufficient samples should not trigger detection.
#[test]
fn test_hard_sample_cluster_insufficient_samples_no_detection() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Only 5 samples
    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| record("output-1", i, 0.5, vec![0.8]))
        .collect();

    let all_records = vec![("output-1".to_string(), records)];

    let config = HardSampleClusterConfig::default();
    let clusters = detect_hard_sample_clusters(&creature, &all_records, &config);

    assert!(
        clusters.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 5: Hard clusters produce coordinated structural candidates with
/// addNeuron and addSynapse operations.
#[test]
fn test_hard_clusters_produce_structural_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-2", "output-1", 0.3),
        ],
    );

    let cluster = HardSampleCluster {
        hard_obs_indices: (50..100).collect(),
        mean_error: 0.7,
        easy_mean_error: 0.05,
        hard_to_easy_ratio: 14.0,
        dominant_input_uuids: vec!["input-1".to_string()],
        output_neuron_count: 1,
        estimated_improvement: 0.03,
    };

    let candidates = hard_sample_clusters_to_coordinated_candidates(&[cluster], &creature);

    assert!(
        !candidates.is_empty(),
        "Should produce at least one coordinated candidate"
    );

    let c = &candidates[0];
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
    assert!(
        c.comment.is_some(),
        "Should have a comment explaining the recommendation"
    );

    // Check operations include addNeuron and addSynapse
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("addNeuron"),
        "Should include addNeuron operation, got: {ops_json}"
    );
    assert!(
        ops_json.contains("addSynapse"),
        "Should include addSynapse operation, got: {ops_json}"
    );
}

/// Test 6: Empty records produce no clusters.
#[test]
fn test_empty_records_no_clusters() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![],
    );

    let config = HardSampleClusterConfig::default();
    let clusters: Vec<HardSampleCluster> = detect_hard_sample_clusters(&creature, &[], &config);

    assert!(
        clusters.is_empty(),
        "Empty records should produce no clusters"
    );
}

/// Test 7: Dominant inputs are identified — input neurons whose activations
/// differ most between hard and easy samples.
#[test]
fn test_dominant_inputs_identified() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("input-3", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-2", "output-1", 0.3),
            synapse("input-3", "output-2", 0.4),
        ],
    );

    let mut all_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // Output neurons: high error on obs 50-99
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                let error = if i >= 50 { 0.8 } else { 0.02 };
                record(output_id, i, 0.5, vec![error])
            })
            .collect();
        all_records.push((output_id.to_string(), records));
    }

    // input-1: activation strongly differs between easy/hard regions
    all_records.push((
        "input-1".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i >= 50 { 0.95 } else { 0.05 };
                record("input-1", i, activation, vec![])
            })
            .collect(),
    ));

    // input-2: constant activation — NOT discriminative
    all_records.push((
        "input-2".to_string(),
        (0..100)
            .map(|i| record("input-2", i, 0.5, vec![]))
            .collect(),
    ));

    // input-3: random — not strongly correlated
    all_records.push((
        "input-3".to_string(),
        (0..100)
            .map(|i| {
                let activation = ((i as f32) * 0.73).sin() * 0.5 + 0.5;
                record("input-3", i, activation, vec![])
            })
            .collect(),
    ));

    let config = HardSampleClusterConfig::default();
    let clusters = detect_hard_sample_clusters(&creature, &all_records, &config);

    assert!(!clusters.is_empty(), "Should detect hard sample cluster");
    let cluster = &clusters[0];
    assert!(
        !cluster.dominant_input_uuids.is_empty(),
        "Should identify dominant inputs"
    );
    assert!(
        cluster
            .dominant_input_uuids
            .contains(&"input-1".to_string()),
        "input-1 should be identified as dominant, got: {:?}",
        cluster.dominant_input_uuids
    );
}

/// Test 8: Estimated improvement is positive for detected clusters.
#[test]
fn test_hard_sample_cluster_estimated_improvement_positive() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
    );

    let mut all_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                let error = if i >= 60 { 0.7 } else { 0.03 };
                record(output_id, i, 0.5, vec![error])
            })
            .collect();
        all_records.push((output_id.to_string(), records));
    }

    let config = HardSampleClusterConfig::default();
    let clusters = detect_hard_sample_clusters(&creature, &all_records, &config);

    assert!(!clusters.is_empty(), "Should detect hard sample cluster");
    for cluster in &clusters {
        assert!(
            cluster.estimated_improvement > 0.0,
            "Estimated improvement should be positive, got {}",
            cluster.estimated_improvement
        );
    }
}

/// Test 9: Multiple error values per record — uses mean absolute error across all outputs.
#[test]
fn test_multi_error_values_handled() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let errors = if i >= 60 {
                vec![0.7, 0.9, 0.6]
            } else {
                vec![0.02, 0.01, 0.03]
            };
            record("output-1", i, 0.5, errors)
        })
        .collect();

    let all_records = vec![("output-1".to_string(), records)];

    let config = HardSampleClusterConfig::default();
    let clusters = detect_hard_sample_clusters(&creature, &all_records, &config);

    assert!(
        !clusters.is_empty(),
        "Should handle multiple error values per record"
    );
}

/// Test 10: Hard-to-easy ratio correctly reflects the disparity.
#[test]
fn test_hard_to_easy_ratio() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
    );

    let mut all_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                // Hard samples have ~10x the error of easy samples
                let error = if i >= 50 { 0.5 } else { 0.05 };
                record(output_id, i, 0.5, vec![error])
            })
            .collect();
        all_records.push((output_id.to_string(), records));
    }

    let config = HardSampleClusterConfig::default();
    let clusters = detect_hard_sample_clusters(&creature, &all_records, &config);

    assert!(!clusters.is_empty(), "Should detect hard sample cluster");
    let cluster = &clusters[0];
    assert!(
        cluster.hard_to_easy_ratio > 2.0,
        "Hard-to-easy ratio should be significant, got {}",
        cluster.hard_to_easy_ratio
    );
}
