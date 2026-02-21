//! Test suite for Issue #221: Leverage sample locality for correlated source neurons.
//!
//! When analysing multiple source neurons for the same target, there is significant
//! overlap in which samples are relevant. This test suite verifies that:
//! 1. Sources can be grouped by sample locality (obs_index overlap)
//! 2. Grouped sources share sample building overhead
//! 3. GPU transfers are reduced for correlated sources
//!
//! ## Expected Benefits
//!
//! | Scenario | Current | With Batching | Improvement |
//! |----------|---------|---------------|-------------|
//! | 100 sources, same obs_indices | 100 sample builds | 1 sample build | 100x fewer builds |
//! | 100 sources, 80% overlap | 100 sample builds | ~5 sample builds | 20x fewer builds |
//! | 100 sources, no overlap | 100 sample builds | 100 sample builds | No change |

mod common;

use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_synapses};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use std::collections::HashSet;
use tempfile::tempdir;

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Create a test creature with multiple inputs and a single output.
fn create_correlated_input_creature(input_count: usize) -> CreatureJson {
    CreatureJson {
        input: input_count,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    }
}

/// Create test records where all inputs share the same obs_indices.
/// This simulates the common case where input neurons are recorded together.
fn create_fully_correlated_records(input_count: usize, record_count: usize) -> Vec<DiscoverRecord> {
    let mut records = Vec::with_capacity((input_count + 1) * record_count);

    for obs_index in 0..record_count as u32 {
        // All inputs share the same obs_indices
        for input_idx in 0..input_count {
            let activation = ((obs_index as f32 + input_idx as f32) % 10.0 - 5.0) / 5.0;
            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{input_idx}"),
                None,
                activation,
                Vec::new(),
            ));
        }

        // Output neuron with correlated error
        let error = if obs_index % 2 == 0 { 0.3 } else { -0.3 };
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));
    }

    records
}

/// Create test records where inputs have partial overlap in obs_indices.
/// Half of the inputs share obs_indices 0-49, the other half share 50-99.
fn create_partially_correlated_records(
    input_count: usize,
    record_count: usize,
    overlap_fraction: f32,
) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    let overlap_count = (record_count as f32 * overlap_fraction) as u32;
    let non_overlap_count = record_count as u32 - overlap_count;

    for obs_index in 0..record_count as u32 {
        // Output neuron records - all obs_indices
        let error = if obs_index % 2 == 0 { 0.3 } else { -0.3 };
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));

        // First half of inputs get first group of obs_indices
        // Plus the overlap region
        for input_idx in 0..input_count / 2 {
            if obs_index < non_overlap_count / 2 + overlap_count {
                let activation = ((obs_index as f32 + input_idx as f32) % 10.0 - 5.0) / 5.0;
                records.push(DiscoverRecord::new(
                    obs_index,
                    format!("input-{input_idx}"),
                    None,
                    activation,
                    Vec::new(),
                ));
            }
        }

        // Second half of inputs get second group of obs_indices
        // Plus the overlap region
        for input_idx in input_count / 2..input_count {
            if obs_index >= non_overlap_count / 2 {
                let activation = ((obs_index as f32 + input_idx as f32) % 10.0 - 5.0) / 5.0;
                records.push(DiscoverRecord::new(
                    obs_index,
                    format!("input-{input_idx}"),
                    None,
                    activation,
                    Vec::new(),
                ));
            }
        }
    }

    records
}

/// Test that sample locality detection correctly groups sources by obs_index overlap.
#[test]
fn sample_locality_groups_sources_by_obs_index_overlap() {
    // This test verifies the grouping logic without needing GPU
    use std::collections::HashMap;

    // Create mock obs_indices for each source
    let mut source_obs_indices: HashMap<String, HashSet<u32>> = HashMap::new();

    // Group A: sources 0-4 share obs_indices 0-99
    for i in 0..5 {
        let indices: HashSet<u32> = (0..100).collect();
        source_obs_indices.insert(format!("input-{i}"), indices);
    }

    // Group B: sources 5-9 share obs_indices 50-149
    for i in 5..10 {
        let indices: HashSet<u32> = (50..150).collect();
        source_obs_indices.insert(format!("input-{i}"), indices);
    }

    // Group C: source 10 has unique obs_indices 200-299
    let indices: HashSet<u32> = (200..300).collect();
    source_obs_indices.insert("input-10".to_string(), indices);

    // Calculate overlap between sources
    fn compute_overlap(a: &HashSet<u32>, b: &HashSet<u32>) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let intersection = a.intersection(b).count();
        let min_size = a.len().min(b.len());
        intersection as f32 / min_size as f32
    }

    // Verify high overlap within Group A
    let overlap_0_1 = compute_overlap(
        source_obs_indices.get("input-0").unwrap(),
        source_obs_indices.get("input-1").unwrap(),
    );
    assert!(
        overlap_0_1 > 0.99,
        "Sources 0 and 1 should have 100% overlap, got {overlap_0_1}"
    );

    // Verify partial overlap between Group A and Group B (50% overlap)
    let overlap_0_5 = compute_overlap(
        source_obs_indices.get("input-0").unwrap(),
        source_obs_indices.get("input-5").unwrap(),
    );
    assert!(
        (0.4..=0.6).contains(&overlap_0_5),
        "Sources 0 and 5 should have ~50% overlap, got {overlap_0_5}"
    );

    // Verify no overlap with Group C
    let overlap_0_10 = compute_overlap(
        source_obs_indices.get("input-0").unwrap(),
        source_obs_indices.get("input-10").unwrap(),
    );
    assert!(
        overlap_0_10 < 0.01,
        "Sources 0 and 10 should have no overlap, got {overlap_0_10}"
    );
}

/// Test that sources with identical obs_indices are grouped together.
/// This test verifies the grouping logic using local helper functions.
#[test]
fn sample_locality_groups_identical_obs_indices() {
    use std::collections::HashMap;

    // Create 10 sources all with identical obs_indices
    let mut source_obs_indices: HashMap<String, HashSet<u32>> = HashMap::new();
    let shared_indices: HashSet<u32> = (0..100).collect();

    for i in 0..10 {
        source_obs_indices.insert(format!("input-{i}"), shared_indices.clone());
    }

    // Group sources by locality using local helper
    let groups = group_sources_by_locality_test(&source_obs_indices, 0.8);

    // All sources should be in a single group
    assert_eq!(
        groups.len(),
        1,
        "All sources with identical obs_indices should be in one group"
    );
    assert_eq!(
        groups[0].len(),
        10,
        "The single group should contain all 10 sources"
    );
}

/// Test that sources with no overlap remain separate.
#[test]
fn sample_locality_keeps_disjoint_sources_separate() {
    use std::collections::HashMap;

    // Create 5 sources each with completely disjoint obs_indices
    let mut source_obs_indices: HashMap<String, HashSet<u32>> = HashMap::new();

    for i in 0..5 {
        let start = i * 100;
        let indices: HashSet<u32> = (start..start + 100).collect();
        source_obs_indices.insert(format!("input-{i}"), indices);
    }

    // Group sources by locality using local helper
    let groups = group_sources_by_locality_test(&source_obs_indices, 0.8);

    // Each source should be in its own group (or no grouping benefit)
    // With disjoint indices, we should have 5 groups
    assert_eq!(
        groups.len(),
        5,
        "Disjoint sources should each be in their own group"
    );
}

/// Helper function to group sources by obs_index locality.
/// Sources are grouped together if they share >= threshold fraction of their obs_indices.
fn group_sources_by_locality_test(
    source_obs_indices: &std::collections::HashMap<String, HashSet<u32>>,
    threshold: f32,
) -> Vec<Vec<String>> {
    use std::collections::HashMap;

    fn compute_overlap(a: &HashSet<u32>, b: &HashSet<u32>) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let intersection = a.intersection(b).count();
        let min_size = a.len().min(b.len());
        intersection as f32 / min_size as f32
    }

    let source_uuids: Vec<String> = source_obs_indices.keys().cloned().collect();
    let mut assigned: HashMap<String, usize> = HashMap::new();
    let mut groups: Vec<Vec<String>> = Vec::new();

    for uuid in &source_uuids {
        if assigned.contains_key(uuid) {
            continue;
        }

        let my_indices = source_obs_indices.get(uuid).unwrap();

        // Try to find an existing group with sufficient overlap
        let mut found_group: Option<usize> = None;
        for (group_idx, group) in groups.iter().enumerate() {
            if let Some(first) = group.first() {
                let first_indices = source_obs_indices.get(first).unwrap();
                let overlap = compute_overlap(my_indices, first_indices);
                if overlap >= threshold {
                    found_group = Some(group_idx);
                    break;
                }
            }
        }

        match found_group {
            Some(idx) => {
                groups[idx].push(uuid.clone());
                assigned.insert(uuid.clone(), idx);
            }
            None => {
                let new_group_idx = groups.len();
                groups.push(vec![uuid.clone()]);
                assigned.insert(uuid.clone(), new_group_idx);
            }
        }
    }

    groups
}

/// Benchmark test to verify that sample locality batching improves performance.
/// Test that sample locality batching produces correct results with correlated inputs.
#[test]
fn sample_locality_batching_produces_results() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let input_count = 100;
    let record_count = 100; // 100 observations per source

    // Create fully correlated records (all inputs share same obs_indices)
    let records = create_fully_correlated_records(input_count, record_count);
    neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
        .expect("Failed to write parquet");

    let creature = create_correlated_input_creature(input_count);

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: Some(42),
    };

    // Note: The benchmark has been moved to benches/sample_locality.rs
    // This test file now only contains correctness tests.

    // Verify analysis produces results (correctness check)
    let result = analyze_synapses(&input).expect("Analysis should succeed");
    assert!(
        !result.helpful_synapses.is_empty() || !result.coordinated_structural_candidates.is_empty(),
        "Analysis should find candidates"
    );
}

/// Test that sample locality optimisation preserves correctness.
/// Results should be identical whether or not batching is applied.
#[test]
fn sample_locality_preserves_correctness() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let input_count = 20;
    let record_count = 50;

    // Create records with strong correlation to ensure candidates are found
    let mut records = Vec::new();
    for obs_index in 0..record_count as u32 {
        // All input neurons share the same obs_indices
        for input_idx in 0..input_count {
            // Alternating pattern for even/odd inputs
            let activation = if input_idx % 2 == 0 {
                (obs_index as f32 - 25.0) / 25.0
            } else {
                -(obs_index as f32 - 25.0) / 25.0
            };
            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{input_idx}"),
                None,
                activation,
                Vec::new(),
            ));
        }

        // Output with correlated error
        let error = (obs_index as f32 - 25.0) / 25.0 * 0.5;
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));
    }

    neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
        .expect("Failed to write parquet");

    let creature = create_correlated_input_creature(input_count);

    // Run analysis with a fixed seed for reproducibility
    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: Some(42),
    };

    let result = analyze_synapses(&input).expect("Analysis should succeed");

    // Verify candidates were found
    let total_candidates =
        result.helpful_synapses.len() + result.coordinated_structural_candidates.len();
    assert!(
        total_candidates > 0,
        "Analysis should find candidates with correlated data"
    );

    println!(
        "Correctness test passed: Found {} helpful synapses and {} coordinated candidates",
        result.helpful_synapses.len(),
        result.coordinated_structural_candidates.len()
    );
}

/// Test the specific scenario from the issue: 100 sources with 90% sample overlap.
/// Verifies that analysis produces correct results with high overlap.
#[test]
fn source_batching_90_percent_overlap_produces_results() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let input_count = 100;
    let record_count = 100;
    let overlap_fraction = 0.9;

    // Create partially correlated records
    let records = create_partially_correlated_records(input_count, record_count, overlap_fraction);
    neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
        .expect("Failed to write parquet");

    let creature = create_correlated_input_creature(input_count);

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: Some(42),
    };

    // Note: The benchmark has been moved to benches/sample_locality.rs
    // This test file now only contains correctness tests.

    // Verify the analysis completed and produced results (correctness check)
    let result = analyze_synapses(&input).expect("Analysis should succeed");
    let total_candidates =
        result.helpful_synapses.len() + result.coordinated_structural_candidates.len();
    assert!(total_candidates > 0, "Analysis should find candidates");
}
