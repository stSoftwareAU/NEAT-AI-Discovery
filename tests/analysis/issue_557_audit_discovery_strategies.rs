//! Issue #557: Audit all discovery strategies
//!
//! Verifies that all returned candidates have strictly positive
//! `expected_creature_score_gain`. Candidates that do not predict an
//! improvement in the creature's score waste evaluation budget and should
//! never be returned.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::common::{hidden, hidden_with_bias, make_creature, neuron, output, record, synapse};
use crate::skip_without_gpu;
use neat_ai_discovery::AnalyzeAllInput;
use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use tempfile::tempdir;

/// Create a creature with structure designed to trigger edge-case discovery
/// modules that may produce zero or near-zero gain candidates.
fn create_audit_creature() -> neat_ai_discovery::CreatureJson {
    make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            // Dead neuron: zero activation → removal candidate
            hidden("hidden-dead", "RELU"),
            // Saturated neuron: always near +1 → squash change candidate
            hidden_with_bias("hidden-saturated", "TANH", 5.0),
            // Active neuron: varies meaningfully
            hidden("hidden-active", "IDENTITY"),
            // Dormant synapse source: very small contribution
            hidden("hidden-dormant-src", "IDENTITY"),
            output("output-0", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-dead", 0.0),
            synapse("input-0", "hidden-saturated", 3.0),
            synapse("input-0", "hidden-active", 0.5),
            synapse("input-1", "hidden-active", 0.3),
            synapse("hidden-dead", "output-0", 0.5),
            synapse("hidden-saturated", "output-0", 0.8),
            synapse("hidden-active", "output-0", 0.6),
            // Dormant synapse: extremely small weight
            synapse("hidden-dormant-src", "output-0", 0.0001),
            synapse("input-0", "hidden-dormant-src", 0.0001),
        ],
    )
}

/// Create parquet records that trigger edge-case detection patterns.
fn create_audit_records() -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    let num_samples = 100u32;

    for i in 0..num_samples {
        let t = i as f32 / num_samples as f32;

        // Input neurons with varying activations
        records.push(DiscoverRecord {
            obs_index: i,
            neuron_uuid: "input-0".to_string(),
            value: None,
            activation: (t * std::f32::consts::TAU).sin(),
            errors: Vec::new(),
        });
        records.push(DiscoverRecord {
            obs_index: i,
            neuron_uuid: "input-1".to_string(),
            value: None,
            activation: (t * std::f32::consts::PI).cos(),
            errors: Vec::new(),
        });

        // Dead neuron: always zero activation
        records.push(record("hidden-dead", i, 0.0, Some(0.0)));

        // Saturated neuron: always near +1 (tanh saturation)
        records.push(record("hidden-saturated", i, 0.999, Some(0.999)));

        // Active neuron: varies meaningfully
        let activation = 0.3 + 0.4 * (t * std::f32::consts::TAU).sin();
        records.push(record("hidden-active", i, activation, Some(activation)));

        // Dormant source: extremely small activation
        let dormant_activation = 0.00001 * t;
        records.push(record(
            "hidden-dormant-src",
            i,
            dormant_activation,
            Some(dormant_activation),
        ));

        // Output neuron with error
        let error = 0.1 * (t * std::f32::consts::PI).cos();
        records.push(DiscoverRecord {
            obs_index: i,
            neuron_uuid: "output-0".to_string(),
            value: Some(0.5 + 0.2 * t),
            activation: 0.5 + 0.2 * t,
            errors: vec![error],
        });
    }

    records
}

/// All returned candidates (synapse and neuron) must have strictly positive
/// `expected_creature_score_gain`. This is the core audit requirement from
/// Issue #557.
#[test]
fn all_candidates_have_positive_expected_score_gain() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("audit_records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let records = create_audit_records();
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = create_audit_creature();
    let focus_neurons: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone())
        .collect();

    let input = AnalyzeAllInput {
        parquet_file,
        creature,
        focus_neurons,
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: Some(42),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_all(&input).expect("analyze_all should succeed");

    // Check synapse results: all candidate types must have positive gain
    if let Some(syn) = &result.synapse {
        for (i, c) in syn.coordinated_structural_candidates.iter().enumerate() {
            assert!(
                c.expected_creature_score_gain > 0.0,
                "Coordinated candidate {i} has non-positive gain: {} (comment: {:?})",
                c.expected_creature_score_gain,
                c.comment,
            );
        }

        for (i, c) in syn.helpful_synapses.iter().enumerate() {
            assert!(
                c.expected_creature_score_gain > 0.0,
                "Helpful synapse {i} has non-positive gain: {} (from {} → {})",
                c.expected_creature_score_gain,
                c.from_neuron_uuid,
                c.to_neuron_uuid,
            );
        }

        for (i, c) in syn.harmful_synapses.iter().enumerate() {
            assert!(
                c.expected_creature_score_gain > 0.0,
                "Harmful synapse {i} has non-positive gain: {} (from {} → {})",
                c.expected_creature_score_gain,
                c.from_neuron_uuid,
                c.to_neuron_uuid,
            );
        }
    }

    // Check neuron results: all candidates must have positive gain
    if let Some(neuron_result) = &result.neuron {
        for (i, c) in neuron_result.helpful_neurons.iter().enumerate() {
            assert!(
                c.expected_creature_score_gain > 0.0,
                "Neuron candidate {i} has non-positive gain: {} (squash: {}, source: {}, target: {})",
                c.expected_creature_score_gain,
                c.squash,
                c.source_neuron_uuid,
                c.target_neuron_uuid,
            );
        }
    }
}

/// Metadata `candidates_returned` must match actual candidate count after
/// positive-gain filtering.
#[test]
fn metadata_consistent_after_positive_gain_filtering() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("audit_meta_records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let records = create_audit_records();
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = create_audit_creature();
    let focus_neurons: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone())
        .collect();

    let input = AnalyzeAllInput {
        parquet_file,
        creature,
        focus_neurons,
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: Some(42),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_all(&input).expect("analyze_all should succeed");

    if let Some(syn) = &result.synapse {
        let actual_count = syn.helpful_synapses.len()
            + syn.harmful_synapses.len()
            + syn.coordinated_structural_candidates.len();

        assert_eq!(
            syn.metadata.candidates_returned, actual_count,
            "synapse metadata.candidates_returned ({}) must match actual count ({actual_count})",
            syn.metadata.candidates_returned,
        );
    }

    if let Some(neuron_result) = &result.neuron {
        assert_eq!(
            neuron_result.metadata.candidates_returned,
            neuron_result.helpful_neurons.len(),
            "neuron metadata.candidates_returned ({}) must match actual count ({})",
            neuron_result.metadata.candidates_returned,
            neuron_result.helpful_neurons.len(),
        );
    }
}
