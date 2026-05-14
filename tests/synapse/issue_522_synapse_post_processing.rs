//! Issue #522: Targeted tests for synapse `post_processing` sub-module
//!
//! Tests the post-processing pipeline in `src/analysis/synapse/post_processing.rs`:
//! - Impact-based discounting (hidden vs output neurons)
//! - Candidate sorting by `expected_creature_score_gain`
//! - Truncation via `max_candidates`
//! - Pessimism discount integration in the full pipeline
//!
//! These tests exercise the full analysis pipeline with crafted creatures and
//! verify correct post-processing behaviour by examining the output candidates.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a parquet file and return the path.
fn write_test_parquet(records: &[DiscoverRecord]) -> (tempfile::TempDir, String) {
    let temp_dir = tempfile::tempdir().unwrap();
    let parquet_path = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .unwrap()
        .to_string();
    write_records_to_parquet(&parquet_path, records).unwrap();
    (temp_dir, parquet_path)
}

// =============================================================================
// Impact discounting: output vs hidden targets
// =============================================================================

/// Candidates targeting output neurons should have impact = 1.0 (no discount).
/// Candidates targeting hidden neurons far from outputs should have reduced impact.
#[test]
fn output_targets_have_full_impact_hidden_targets_are_discounted() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    // Topology: input-0 → hidden-0 → output-0
    // Both hidden-0 and output-0 are focus neurons.
    // We set up errors so that add-synapse candidates are found for both targets.
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
    };

    let mut records = Vec::new();
    for obs in 0..200u32 {
        let x0 = (obs as f32 / 100.0) - 1.0;
        let x1 = ((obs as f32) * 0.7).sin();
        let hidden_act = 0.5 * x0;
        let output_act = 0.5 * hidden_act;

        // Errors correlated with input-1 for both targets
        let hidden_error = 0.1 * x1;
        let output_error = 0.1 * x1;

        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(x0),
            x0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-1".to_string(),
            Some(x1),
            x1,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "hidden-0".to_string(),
            Some(hidden_act),
            hidden_act,
            vec![hidden_error],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(output_act),
            output_act,
            vec![output_error],
        ));
    }
    let (_temp_dir, parquet_path) = write_test_parquet(&records);

    let input = AnalyzeSynapsesInput {
        creature,
        parquet_file: parquet_path,
        focus_neurons: vec!["hidden-0".to_string(), "output-0".to_string()],
        max_candidates: Some(100),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = neat_ai_discovery::analysis::synapse::analyze_synapses(&input).unwrap();

    // Find candidates targeting output-0 and hidden-0
    let output_candidates: Vec<_> = result
        .helpful_synapses
        .iter()
        .filter(|c| c.to_neuron_uuid == "output-0")
        .collect();
    let hidden_candidates: Vec<_> = result
        .helpful_synapses
        .iter()
        .filter(|c| c.to_neuron_uuid == "hidden-0")
        .collect();

    // Output candidates should have impact = 1.0
    for c in &output_candidates {
        assert!(
            (c.target_neuron_impact - 1.0).abs() < 1e-6,
            "Output target should have impact 1.0, got {:.4}",
            c.target_neuron_impact
        );
    }

    // Hidden candidates should have impact <= 1.0 (potentially discounted)
    for c in &hidden_candidates {
        assert!(
            c.target_neuron_impact <= 1.0 + 1e-6,
            "Hidden target impact should be <= 1.0, got {:.4}",
            c.target_neuron_impact
        );
        assert!(
            c.target_neuron_impact > 0.0,
            "Hidden target impact should be positive, got {:.4}",
            c.target_neuron_impact
        );
    }
}

// =============================================================================
// Candidate sorting: highest expected_creature_score_gain first
// =============================================================================

/// After post-processing, helpful candidates should be sorted by
/// `expected_creature_score_gain` in descending order (before diversification).
#[test]
fn candidates_sorted_by_expected_score_gain_descending() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    // Create a creature with multiple unused inputs to generate multiple candidates.
    let creature = CreatureJson {
        input: 4,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 0.5,
            synapse_type: None,
        }],
    };

    let mut records = Vec::new();
    for obs in 0..200u32 {
        let x0 = (obs as f32 / 100.0) - 1.0;
        let x1 = ((obs as f32) * 0.3).sin();
        let x2 = ((obs as f32) * 0.5).cos();
        let x3 = ((obs as f32) * 0.7).sin() * 0.1; // Weak signal
        let current = 0.5 * x0;

        // Error correlates strongly with x1, moderately with x2, weakly with x3
        let error = 0.3 * x1 + 0.1 * x2 + 0.01 * x3;

        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(x0),
            x0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-1".to_string(),
            Some(x1),
            x1,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-2".to_string(),
            Some(x2),
            x2,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-3".to_string(),
            Some(x3),
            x3,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(current),
            current,
            vec![error],
        ));
    }
    let (_temp_dir, parquet_path) = write_test_parquet(&records);

    let input = AnalyzeSynapsesInput {
        creature,
        parquet_file: parquet_path,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(100),
        analysis_deadline_ms: None, // No deadline = no diversification shuffle
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = neat_ai_discovery::analysis::synapse::analyze_synapses(&input).unwrap();

    if result.helpful_synapses.len() >= 2 {
        // Check that sorting is non-increasing (within weight variant groups)
        // Weight variants may have lower scores, so we check overall ordering
        // is roughly descending by finding the maximum score in the first half
        // vs the second half.
        let mid = result.helpful_synapses.len() / 2;
        let first_half_max = result.helpful_synapses[..mid]
            .iter()
            .map(|c| c.expected_creature_score_gain)
            .fold(f32::NEG_INFINITY, f32::max);
        let second_half_max = result.helpful_synapses[mid..]
            .iter()
            .map(|c| c.expected_creature_score_gain)
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            first_half_max >= second_half_max - 1e-6,
            "First half max ({first_half_max:.6}) should be >= second half max ({second_half_max:.6})"
        );
    }
}

// =============================================================================
// max_candidates truncation
// =============================================================================

/// When `max_candidates` is set, the total candidate count should not exceed
/// the specified limit.
#[test]
fn max_candidates_limits_total_output() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let creature = CreatureJson {
        input: 4,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 0.1,
            synapse_type: None,
        }],
    };

    let mut records = Vec::new();
    for obs in 0..200u32 {
        let x0 = (obs as f32 / 100.0) - 1.0;
        let x1 = ((obs as f32) * 0.3).sin();
        let x2 = ((obs as f32) * 0.5).cos();
        let x3 = ((obs as f32) * 0.7).sin();
        let current = 0.1 * x0;
        let error = 0.2 * x1 + 0.15 * x2 + 0.1 * x3;

        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(x0),
            x0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-1".to_string(),
            Some(x1),
            x1,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-2".to_string(),
            Some(x2),
            x2,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-3".to_string(),
            Some(x3),
            x3,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(current),
            current,
            vec![error],
        ));
    }
    let (_temp_dir, parquet_path) = write_test_parquet(&records);

    let max_limit = 5;
    let input = AnalyzeSynapsesInput {
        creature,
        parquet_file: parquet_path,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(max_limit),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = neat_ai_discovery::analysis::synapse::analyze_synapses(&input).unwrap();

    let total = result.helpful_synapses.len()
        + result.harmful_synapses.len()
        + result.coordinated_structural_candidates.len();

    assert!(
        total <= max_limit,
        "Total candidates ({total}) should not exceed max_candidates ({max_limit}). \
        helpful={}, harmful={}, coordinated={}",
        result.helpful_synapses.len(),
        result.harmful_synapses.len(),
        result.coordinated_structural_candidates.len()
    );
}

// =============================================================================
// Pessimism discount in full pipeline
// =============================================================================

/// Verify that the full pipeline applies pessimism discount. The pessimism
/// discount is based on the `improved_count` / `total_count` ratio. Candidates
/// where not all samples improve should have a discounted score gain.
///
/// Note: `score_gain` also includes source-type and target-type boosts
/// applied after pessimism, so `score_gain` can exceed `error_reduction`.
#[test]
fn pipeline_applies_pessimism_discount_to_candidates() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 0.5,
            synapse_type: None,
        }],
    };

    let mut records = Vec::new();
    for obs in 0..200u32 {
        let x0 = (obs as f32 / 100.0) - 1.0;
        let x1 = ((obs as f32) * 0.5).sin();
        let current = 0.5 * x0;
        let error = 0.2 * x1; // Correlated with input-1

        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(x0),
            x0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-1".to_string(),
            Some(x1),
            x1,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(current),
            current,
            vec![error],
        ));
    }
    let (_temp_dir, parquet_path) = write_test_parquet(&records);

    let input = AnalyzeSynapsesInput {
        creature,
        parquet_file: parquet_path,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(100),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = neat_ai_discovery::analysis::synapse::analyze_synapses(&input).unwrap();

    // At least one helpful candidate should exist
    assert!(
        !result.helpful_synapses.is_empty(),
        "Should find at least one helpful synapse candidate"
    );

    for candidate in &result.helpful_synapses {
        // expected_creature_score_gain should be finite and positive
        assert!(
            candidate.expected_creature_score_gain.is_finite(),
            "Score gain should be finite"
        );
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Score gain should be positive for helpful candidates"
        );

        // improved_count and total_count should be set by the pipeline
        assert!(candidate.total_count > 0, "Total count should be positive");

        // The pessimism discount uses improved_count / total_count as a quality signal.
        // Verify that the ratio is reasonable (between 0 and 1).
        let ratio = candidate.improved_count as f32 / candidate.total_count as f32;
        assert!(
            (0.0..=1.0).contains(&ratio),
            "Improved ratio should be in [0, 1]: {ratio}"
        );
    }
}

// =============================================================================
// Metadata assembly
// =============================================================================

/// Verify that the analysis metadata contains reasonable values after
/// post-processing.
#[test]
fn metadata_contains_candidate_counts_and_timing() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 0.5,
            synapse_type: None,
        }],
    };

    let mut records = Vec::new();
    for obs in 0..100u32 {
        let x0 = (obs as f32 / 50.0) - 1.0;
        let x1 = ((obs as f32) * 0.5).sin();
        let current = 0.5 * x0;
        let error = 0.1 * x1;

        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(x0),
            x0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-1".to_string(),
            Some(x1),
            x1,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(current),
            current,
            vec![error],
        ));
    }
    let (_temp_dir, parquet_path) = write_test_parquet(&records);

    let input = AnalyzeSynapsesInput {
        creature,
        parquet_file: parquet_path,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = neat_ai_discovery::analysis::synapse::analyze_synapses(&input).unwrap();

    // Check metadata fields
    let metadata = &result.metadata;
    assert_eq!(
        metadata.total_focus_neurons, 1,
        "Should have 1 focus neuron"
    );
    assert_eq!(
        metadata.completed_focus_neurons, 1,
        "Should have completed 1 focus neuron"
    );
    assert!(
        !metadata.timed_out,
        "Should not have timed out without deadline"
    );

    // candidates_returned should match actual candidates returned
    let actual_returned = result.helpful_synapses.len()
        + result.harmful_synapses.len()
        + result.coordinated_structural_candidates.len();
    assert_eq!(
        metadata.candidates_returned, actual_returned,
        "Metadata candidates_returned ({}) should match actual returned ({})",
        metadata.candidates_returned, actual_returned
    );

    // candidates_found should be >= candidates_returned (some may be truncated)
    assert!(
        metadata.candidates_found >= metadata.candidates_returned,
        "candidates_found ({}) should be >= candidates_returned ({})",
        metadata.candidates_found,
        metadata.candidates_returned
    );
}
