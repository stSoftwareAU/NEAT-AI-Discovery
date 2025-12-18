//! Integration tests for the visualisation snapshot export feature.
//!
//! These tests verify that the export function correctly:
//! - Reads Parquet recordings
//! - Computes neuron stats and impacts
//! - Computes synapse contributions
//! - Computes reconstruction checks (recorded vs computed values)
//! - Writes a valid JSON snapshot

use neat_ai_discovery::export::{
    export_visualisation_snapshot, ExportOptions, VisualisationSnapshot,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::fs::File;
use std::io::BufReader;
use tempfile::tempdir;

/// Create a minimal test creature with:
/// - 2 inputs (input-0, input-1)
/// - 1 hidden neuron (hidden-0) with IDENTITY squash
/// - 1 output neuron (output-0) with IDENTITY squash
fn create_test_creature() -> CreatureJson {
    CreatureJson {
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
            // input-0 -> hidden-0 (weight 1.0)
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            // input-1 -> hidden-0 (weight 0.5)
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            // hidden-0 -> output-0 (weight 2.0)
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 2.0,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 1,
    }
}

/// Create test Parquet records for the test creature.
///
/// For IDENTITY squash with bias=0:
/// - input-0 activation = 1.0
/// - input-1 activation = 2.0
/// - hidden-0: value = 1.0*1.0 + 2.0*0.5 = 2.0, activation = 2.0 (IDENTITY)
/// - output-0: value = 2.0*2.0 = 4.0, activation = 4.0 (IDENTITY)
fn create_test_records() -> Vec<DiscoverRecord> {
    vec![
        // obsIndex 0: input neurons
        DiscoverRecord::new(0, "input-0".to_string(), Some(1.0), 1.0, vec![0.0]),
        DiscoverRecord::new(0, "input-1".to_string(), Some(2.0), 2.0, vec![0.0]),
        // obsIndex 0: hidden neuron
        // value = 1.0*1.0 + 2.0*0.5 = 2.0
        DiscoverRecord::new(0, "hidden-0".to_string(), Some(2.0), 2.0, vec![0.1]),
        // obsIndex 0: output neuron
        // value = 2.0 * 2.0 = 4.0
        DiscoverRecord::new(0, "output-0".to_string(), Some(4.0), 4.0, vec![0.05]),
        // obsIndex 1: different activations
        DiscoverRecord::new(1, "input-0".to_string(), Some(0.5), 0.5, vec![0.0]),
        DiscoverRecord::new(1, "input-1".to_string(), Some(1.0), 1.0, vec![0.0]),
        // hidden-0: value = 0.5*1.0 + 1.0*0.5 = 1.0
        DiscoverRecord::new(1, "hidden-0".to_string(), Some(1.0), 1.0, vec![0.2]),
        // output-0: value = 1.0 * 2.0 = 2.0
        DiscoverRecord::new(1, "output-0".to_string(), Some(2.0), 2.0, vec![0.1]),
    ]
}

#[test]
fn test_export_creates_valid_json() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_path = temp_dir.path().join("records.parquet");
    let snapshot_path = temp_dir.path().join("snapshot.json");

    // Write test records to Parquet
    let records = create_test_records();
    write_records_to_parquet(parquet_path.to_str().unwrap(), &records)
        .expect("Failed to write Parquet");

    // Create test creature
    let creature = create_test_creature();

    // Export snapshot
    let options = ExportOptions::default();
    let stats = export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &options,
    )
    .expect("Export should succeed");

    // Verify stats
    assert_eq!(stats.obs_count, 2, "Should have 2 observations");
    assert_eq!(stats.synapse_count, 3, "Should have 3 synapses");
    assert_eq!(stats.output_count, 1, "Should have 1 output");

    // Load and parse the JSON
    let file = File::open(&snapshot_path).expect("Failed to open snapshot");
    let reader = BufReader::new(file);
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(reader).expect("Failed to parse snapshot JSON");

    // Verify structure
    assert!(
        !snapshot.meta.exported_at.is_empty(),
        "Should have export timestamp"
    );
    assert!(
        !snapshot.meta.discovery_version.is_empty(),
        "Should have version"
    );
    assert_eq!(
        snapshot.recording.obs_indices.len(),
        2,
        "Should have 2 obsIndices"
    );
    assert!(
        snapshot.recording.neurons.contains_key("output-0"),
        "Should have output-0 in recording"
    );
}

#[test]
fn test_neuron_stats_are_computed_correctly() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_path = temp_dir.path().join("records.parquet");
    let snapshot_path = temp_dir.path().join("snapshot.json");

    let records = create_test_records();
    write_records_to_parquet(parquet_path.to_str().unwrap(), &records)
        .expect("Failed to write Parquet");

    let creature = create_test_creature();
    let options = ExportOptions::default();

    export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &options,
    )
    .expect("Export should succeed");

    let file = File::open(&snapshot_path).expect("Failed to open snapshot");
    let reader = BufReader::new(file);
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(reader).expect("Failed to parse snapshot JSON");

    // Check hidden-0 neuron stats
    let hidden_recording = snapshot
        .recording
        .neurons
        .get("hidden-0")
        .expect("Should have hidden-0");

    // hidden-0 activations: [2.0, 1.0]
    // mean = 1.5, min = 1.0, max = 2.0
    assert!(
        (hidden_recording.stats.mean_activation - 1.5).abs() < 1e-5,
        "hidden-0 mean activation should be 1.5, got {}",
        hidden_recording.stats.mean_activation
    );
    assert!(
        (hidden_recording.stats.activation_min - 1.0).abs() < 1e-5,
        "hidden-0 min activation should be 1.0"
    );
    assert!(
        (hidden_recording.stats.activation_max - 2.0).abs() < 1e-5,
        "hidden-0 max activation should be 2.0"
    );
}

#[test]
fn test_synapse_contributions_are_computed_correctly() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_path = temp_dir.path().join("records.parquet");
    let snapshot_path = temp_dir.path().join("snapshot.json");

    let records = create_test_records();
    write_records_to_parquet(parquet_path.to_str().unwrap(), &records)
        .expect("Failed to write Parquet");

    let creature = create_test_creature();
    let options = ExportOptions::default();

    export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &options,
    )
    .expect("Export should succeed");

    let file = File::open(&snapshot_path).expect("Failed to open snapshot");
    let reader = BufReader::new(file);
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(reader).expect("Failed to parse snapshot JSON");

    // Check hidden-0 -> output-0 synapse (weight 2.0)
    // hidden-0 activations: [2.0, 1.0]
    // contribution = activation * weight = [4.0, 2.0]
    let synapse = snapshot
        .derived
        .synapses
        .get("hidden-0→output-0")
        .expect("Should have hidden-0→output-0 synapse");

    assert!(
        (synapse.weight - 2.0).abs() < 1e-5,
        "Synapse weight should be 2.0"
    );
    assert_eq!(
        synapse.contribution.len(),
        2,
        "Should have 2 contribution values"
    );
    assert!(
        (synapse.contribution[0] - 4.0).abs() < 1e-5,
        "First contribution should be 4.0, got {}",
        synapse.contribution[0]
    );
    assert!(
        (synapse.contribution[1] - 2.0).abs() < 1e-5,
        "Second contribution should be 2.0, got {}",
        synapse.contribution[1]
    );
}

#[test]
fn test_reconstruction_checks_for_identity_network() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_path = temp_dir.path().join("records.parquet");
    let snapshot_path = temp_dir.path().join("snapshot.json");

    let records = create_test_records();
    write_records_to_parquet(parquet_path.to_str().unwrap(), &records)
        .expect("Failed to write Parquet");

    let creature = create_test_creature();
    let options = ExportOptions {
        include_per_synapse_series: true,
        include_reconstruction_checks: true,
        max_obs: None,
        top_k_worst_samples: 10,
    };

    export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &options,
    )
    .expect("Export should succeed");

    let file = File::open(&snapshot_path).expect("Failed to open snapshot");
    let reader = BufReader::new(file);
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(reader).expect("Failed to parse snapshot JSON");

    // For an IDENTITY network with correct recordings, reconstruction deltas should be ~0
    let checks = snapshot
        .derived
        .reconstruction_checks
        .expect("Should have reconstruction checks");

    // Find hidden-0 check
    let hidden_check = checks.iter().find(|c| c.neuron_uuid == "hidden-0");
    assert!(hidden_check.is_some(), "Should have check for hidden-0");
    let hidden_check = hidden_check.unwrap();

    // hidden-0 receives from input-0 (weight 1.0) and input-1 (weight 0.5)
    // obsIndex 0: reconstructed = 0 + 1.0*1.0 + 2.0*0.5 = 2.0, recorded = 2.0
    // obsIndex 1: reconstructed = 0 + 0.5*1.0 + 1.0*0.5 = 1.0, recorded = 1.0
    // So deltas should be 0
    assert!(
        hidden_check.max_value_delta < 0.01,
        "For IDENTITY network, value delta should be ~0, got {}",
        hidden_check.max_value_delta
    );
    assert!(
        hidden_check.max_activation_delta < 0.01,
        "For IDENTITY network, activation delta should be ~0, got {}",
        hidden_check.max_activation_delta
    );
}

#[test]
fn test_impacts_are_computed() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_path = temp_dir.path().join("records.parquet");
    let snapshot_path = temp_dir.path().join("snapshot.json");

    let records = create_test_records();
    write_records_to_parquet(parquet_path.to_str().unwrap(), &records)
        .expect("Failed to write Parquet");

    let creature = create_test_creature();
    let options = ExportOptions::default();

    export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &options,
    )
    .expect("Export should succeed");

    let file = File::open(&snapshot_path).expect("Failed to open snapshot");
    let reader = BufReader::new(file);
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(reader).expect("Failed to parse snapshot JSON");

    // Output neurons should have impact = 1.0
    let output_impact = snapshot
        .derived
        .impacts_by_neuron_uuid
        .get("output-0")
        .expect("Should have output-0 impact");
    assert!(
        (*output_impact - 1.0).abs() < 1e-5,
        "Output neuron should have impact 1.0, got {}",
        output_impact
    );

    // Hidden neuron should have impact > 0 (since it feeds into output)
    let hidden_impact = snapshot
        .derived
        .impacts_by_neuron_uuid
        .get("hidden-0")
        .expect("Should have hidden-0 impact");
    assert!(
        *hidden_impact > 0.0,
        "Hidden neuron should have positive impact, got {}",
        hidden_impact
    );
}

#[test]
fn test_max_obs_limits_output() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_path = temp_dir.path().join("records.parquet");
    let snapshot_path = temp_dir.path().join("snapshot.json");

    let records = create_test_records();
    write_records_to_parquet(parquet_path.to_str().unwrap(), &records)
        .expect("Failed to write Parquet");

    let creature = create_test_creature();

    // Limit to 1 observation
    let options = ExportOptions {
        include_per_synapse_series: true,
        include_reconstruction_checks: true,
        max_obs: Some(1),
        top_k_worst_samples: 5,
    };

    let stats = export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &options,
    )
    .expect("Export should succeed");

    assert_eq!(stats.obs_count, 1, "Should be limited to 1 observation");

    let file = File::open(&snapshot_path).expect("Failed to open snapshot");
    let reader = BufReader::new(file);
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(reader).expect("Failed to parse snapshot JSON");

    assert_eq!(
        snapshot.recording.obs_indices.len(),
        1,
        "Should have only 1 obsIndex"
    );
}

#[test]
fn test_export_without_reconstruction_checks() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_path = temp_dir.path().join("records.parquet");
    let snapshot_path = temp_dir.path().join("snapshot.json");

    let records = create_test_records();
    write_records_to_parquet(parquet_path.to_str().unwrap(), &records)
        .expect("Failed to write Parquet");

    let creature = create_test_creature();

    let options = ExportOptions {
        include_per_synapse_series: true,
        include_reconstruction_checks: false, // Disable
        max_obs: None,
        top_k_worst_samples: 5,
    };

    export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &options,
    )
    .expect("Export should succeed");

    let file = File::open(&snapshot_path).expect("Failed to open snapshot");
    let reader = BufReader::new(file);
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(reader).expect("Failed to parse snapshot JSON");

    assert!(
        snapshot.derived.reconstruction_checks.is_none(),
        "Reconstruction checks should be None when disabled"
    );
}

#[test]
fn test_export_without_synapse_series() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_path = temp_dir.path().join("records.parquet");
    let snapshot_path = temp_dir.path().join("snapshot.json");

    let records = create_test_records();
    write_records_to_parquet(parquet_path.to_str().unwrap(), &records)
        .expect("Failed to write Parquet");

    let creature = create_test_creature();

    let options = ExportOptions {
        include_per_synapse_series: false, // Disable
        include_reconstruction_checks: true,
        max_obs: None,
        top_k_worst_samples: 5,
    };

    export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &options,
    )
    .expect("Export should succeed");

    let file = File::open(&snapshot_path).expect("Failed to open snapshot");
    let reader = BufReader::new(file);
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(reader).expect("Failed to parse snapshot JSON");

    assert!(
        snapshot.derived.synapses.is_empty(),
        "Synapse derived data should be empty when disabled"
    );
}
