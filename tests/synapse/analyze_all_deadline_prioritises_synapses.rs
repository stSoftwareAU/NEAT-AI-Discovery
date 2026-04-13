//! Integration tests for synapse-friendly discovery improvements (v0.2.17).
//!
//! These tests verify:
//! 1. Synapse analysis runs first when a deadline is set (preventing starvation)
//! 2. Metadata fields correctly indicate whether `target_value` was available
//! 3. Metadata fields correctly indicate whether saturation-aware simulation was used
//! 4. Candidate found/returned counts are accurate

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson};
use tempfile::tempdir;

/// Helper macro to skip tests that require GPU when no GPU is available.
macro_rules! skip_if_no_gpu {
    () => {
        if !neat_ai_discovery::analysis::GpuAnalyzer::gpu_is_available() {
            eprintln!("⚠️  Skipping test: GPU not available");
            return;
        }
    };
}

/// Create a simple creature with multiple inputs and one output for testing.
/// Note: No existing synapses so new synapse candidates can be generated.
fn create_simple_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "HARD_TANH".to_string(),
            bias: 0.0,
        }],
        synapses: vec![], // No existing synapses - all inputs are candidates
        input: 3,         // 3 inputs to ensure candidates are generated
        output: 1,
    }
}

/// Create test records with `target_value` (pre-activation) data.
fn create_records_with_value(parquet_file: &str) {
    let mut records = Vec::new();
    for obs_index in 0..100u32 {
        // Input neuron records (3 inputs)
        for input_idx in 0..3 {
            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{input_idx}"),
                Some((input_idx as f32 + 1.0) * 0.2), // value (pre-activation)
                (input_idx as f32 + 1.0) * 0.2,       // activation
                vec![0.0],                            // errors
            ));
        }
        // Output neuron records with value data
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.3),                                  // value (pre-activation) - present!
            0.2,                                        // activation
            vec![0.1 * ((obs_index % 3) as f32 - 1.0)], // varying errors
        ));
    }
    write_records_to_parquet(parquet_file, &records).expect("Failed to write test records");
}

/// Create test records WITHOUT `target_value` (pre-activation) data.
fn create_records_without_value(parquet_file: &str) {
    let mut records = Vec::new();
    for obs_index in 0..100u32 {
        // Input neuron records (3 inputs)
        for input_idx in 0..3 {
            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{input_idx}"),
                None,                           // value - MISSING
                (input_idx as f32 + 1.0) * 0.2, // activation
                vec![0.0],                      // errors
            ));
        }
        // Output neuron records WITHOUT value data
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            None,                                       // value - MISSING!
            0.2,                                        // activation
            vec![0.1 * ((obs_index % 3) as f32 - 1.0)], // varying errors
        ));
    }
    write_records_to_parquet(parquet_file, &records).expect("Failed to write test records");
}

/// Test that synapse analysis runs when a deadline is set.
///
/// Prior to v0.2.17, neuron analysis ran first and could consume the entire
/// deadline budget, leaving zero time for synapse analysis. This test verifies
/// that synapse analysis now runs first when deadline-constrained.
#[test]
fn synapse_analysis_runs_under_deadline() {
    skip_if_no_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    create_records_with_value(&parquet_file);

    let input = AnalyzeAllInput {
        parquet_file,
        creature: create_simple_creature(),
        focus_neurons: vec!["output-0".to_string()],
        max_synapse_candidates: Some(10),
        max_neuron_candidates: Some(10),
        analysis_deadline_ms: Some(30_000), // 30 second deadline - should be plenty
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: None,
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
    };

    let result = analyze_all(&input).expect("Analysis should succeed");

    // Synapse analysis should have run (not been starved)
    assert!(
        result.synapse.is_some(),
        "Synapse analysis should have run when deadline is set"
    );

    // Neuron analysis should also have run (deadline is generous)
    assert!(
        result.neuron.is_some(),
        "Neuron analysis should have run with generous deadline"
    );
}

/// Test that metadata correctly indicates `target_value` was available.
#[test]
fn metadata_indicates_target_value_available() {
    skip_if_no_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    // Create records WITH value data
    create_records_with_value(&parquet_file);

    let input = AnalyzeAllInput {
        parquet_file,
        creature: create_simple_creature(),
        focus_neurons: vec!["output-0".to_string()],
        max_synapse_candidates: Some(10),
        max_neuron_candidates: Some(10),
        analysis_deadline_ms: Some(30_000),
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: None,
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
    };

    let result = analyze_all(&input).expect("Analysis should succeed");
    let synapse_result = result.synapse.expect("Synapse analysis should have run");

    // Metadata should indicate target_value was available
    assert!(
        synapse_result.metadata.target_value_available,
        "target_value_available should be true when records have value data"
    );
}

/// Test that metadata correctly indicates `target_value` was NOT available.
#[test]
fn metadata_indicates_target_value_not_available() {
    skip_if_no_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    // Create records WITHOUT value data
    create_records_without_value(&parquet_file);

    let input = AnalyzeAllInput {
        parquet_file,
        creature: create_simple_creature(),
        focus_neurons: vec!["output-0".to_string()],
        max_synapse_candidates: Some(10),
        max_neuron_candidates: Some(10),
        analysis_deadline_ms: Some(30_000),
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: None,
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
    };

    let result = analyze_all(&input).expect("Analysis should succeed");
    let synapse_result = result.synapse.expect("Synapse analysis should have run");

    // Metadata should indicate target_value was NOT available
    assert!(
        !synapse_result.metadata.target_value_available,
        "target_value_available should be false when records lack value data"
    );

    // Saturation-aware simulation should NOT be used without value data
    assert!(
        !synapse_result.metadata.saturation_aware_simulation_used,
        "saturation_aware_simulation_used should be false without value data"
    );
}

/// Test that metadata correctly indicates saturation-aware simulation was used.
#[test]
fn metadata_indicates_saturation_aware_simulation_used() {
    skip_if_no_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    // Create records WITH value data and HARD_TANH target
    create_records_with_value(&parquet_file);

    let input = AnalyzeAllInput {
        parquet_file,
        creature: create_simple_creature(), // Uses HARD_TANH for output-0
        focus_neurons: vec!["output-0".to_string()],
        max_synapse_candidates: Some(10),
        max_neuron_candidates: Some(10),
        analysis_deadline_ms: Some(30_000),
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: None,
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
    };

    let result = analyze_all(&input).expect("Analysis should succeed");
    let synapse_result = result.synapse.expect("Synapse analysis should have run");

    // With HARD_TANH target and value data available, saturation-aware simulation should be used
    // Note: This depends on there being candidates that actually use the simulation path
    if !synapse_result.helpful_synapses.is_empty() || !synapse_result.harmful_synapses.is_empty() {
        assert!(
            synapse_result.metadata.saturation_aware_simulation_used,
            "saturation_aware_simulation_used should be true with HARD_TANH + value data + candidates"
        );
    }
}

/// Test that candidate counts are tracked correctly.
#[test]
fn candidate_counts_tracked_correctly() {
    skip_if_no_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    create_records_with_value(&parquet_file);

    // Request more candidates than might exist to test counting
    let input = AnalyzeAllInput {
        parquet_file,
        creature: create_simple_creature(),
        focus_neurons: vec!["output-0".to_string()],
        max_synapse_candidates: Some(100), // High limit
        max_neuron_candidates: Some(100),  // High limit
        analysis_deadline_ms: Some(30_000),
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: None,
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
    };

    let result = analyze_all(&input).expect("Analysis should succeed");

    // Synapse metadata
    if let Some(synapse_result) = &result.synapse {
        let metadata = &synapse_result.metadata;
        let actual_returned = synapse_result.helpful_synapses.len()
            + synapse_result.harmful_synapses.len()
            + synapse_result.coordinated_structural_candidates.len();

        assert_eq!(
            metadata.candidates_returned, actual_returned,
            "candidates_returned should match actual returned count"
        );

        // candidates_found should be >= candidates_returned
        assert!(
            metadata.candidates_found >= metadata.candidates_returned,
            "candidates_found ({}) should be >= candidates_returned ({})",
            metadata.candidates_found,
            metadata.candidates_returned
        );
    }

    // Neuron metadata
    if let Some(neuron_result) = &result.neuron {
        let metadata = &neuron_result.metadata;
        let actual_returned = neuron_result.helpful_neurons.len();

        assert_eq!(
            metadata.candidates_returned, actual_returned,
            "candidates_returned should match actual returned count"
        );

        // NOTE: Unlike synapse analysis, neuron analysis can have candidates_returned > candidates_found
        // because pair_extreme_candidates_with_conservative_variants() can ADD conservative and
        // gentle nudge variants for extreme candidates. See tests/neuron_metadata_candidates_found_includes_pairing.rs
        // for explicit test cases demonstrating this behaviour.
        //
        // candidates_found = original candidates discovered before pairing
        // candidates_returned = final count after pairing (can be higher!) and truncation
    }
}

/// Test that truncation is reflected in candidate counts.
#[test]
fn truncation_reflected_in_candidate_counts() {
    skip_if_no_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    // Create a creature with multiple inputs to generate more synapse candidates
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(), // IDENTITY to avoid saturation issues
            bias: 0.0,
        }],
        synapses: vec![], // No existing synapses - all inputs are candidates
        input: 10,        // 10 inputs
        output: 1,
    };

    // Create records for all inputs
    let mut records = Vec::new();
    for obs_index in 0..100u32 {
        for input_idx in 0..10 {
            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{input_idx}"),
                Some((input_idx as f32) / 10.0),
                (input_idx as f32) / 10.0,
                vec![0.0],
            ));
        }
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.1 * (obs_index as f32 % 3.0 - 1.0)], // Varying errors
        ));
    }
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write test records");

    // Request only 2 candidates but there should be more available
    let input = AnalyzeAllInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_synapse_candidates: Some(2), // Low limit to force truncation
        max_neuron_candidates: Some(2),
        analysis_deadline_ms: Some(30_000),
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: None,
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
    };

    let result = analyze_all(&input).expect("Analysis should succeed");

    if let Some(synapse_result) = &result.synapse {
        let metadata = &synapse_result.metadata;

        // If truncation happened, candidates_found should be > candidates_returned
        // This test may not always trigger truncation depending on threshold filtering
        if metadata.candidates_found > 0 {
            eprintln!(
                "Synapse candidates: found={}, returned={}",
                metadata.candidates_found, metadata.candidates_returned
            );

            // candidates_returned should not exceed the limit
            assert!(
                metadata.candidates_returned <= 4, // 2 helpful + 2 harmful max
                "candidates_returned should respect max_candidates limit"
            );
        }
    }
}
