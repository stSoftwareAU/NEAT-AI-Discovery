//! Regression test: export snapshot must be JSON-safe with non-finite floats.
//!
//! The snapshot export includes raw per-observation series (eg synapse contribution).
//! If any series contains NaN or ±Infinity, `serde_json` will fail because JSON has
//! no representation for non-finite numbers.
//!
//! This test ensures we sanitise non-finite values during export so the snapshot
//! always serialises successfully.

use neat_ai_discovery::export::{
    export_visualisation_snapshot, ExportOptions, VisualisationSnapshot,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::fs::File;
use std::io::BufReader;
use tempfile::tempdir;

fn create_creature_with_contribution_overflow() -> CreatureJson {
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
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            // Deliberately keep weight finite, but allow contribution overflow via huge activation.
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 2.0,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    }
}

fn create_records_with_non_finite_values() -> Vec<DiscoverRecord> {
    vec![
        // obsIndex 0: include a NaN activation (should be sanitised during export).
        DiscoverRecord::new(0, "input-0".to_string(), Some(1.0), 1.0, vec![0.0]),
        DiscoverRecord::new(
            0,
            "hidden-0".to_string(),
            Some(0.0),
            f32::NAN,
            vec![f32::NAN, 0.1],
        ),
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.0), 0.0, vec![0.0]),
        // obsIndex 1: use a huge finite activation so (activation * weight) would overflow
        // without sanitisation.
        DiscoverRecord::new(1, "input-0".to_string(), Some(1.0), 1.0, vec![0.0]),
        DiscoverRecord::new(
            1,
            "hidden-0".to_string(),
            Some(0.0),
            f32::MAX,
            vec![f32::INFINITY, 0.2],
        ),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.0), 0.0, vec![0.0]),
    ]
}

#[test]
fn export_visualisation_snapshot_sanitises_non_finite_series() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_path = temp_dir.path().join("records.parquet");
    let snapshot_path = temp_dir.path().join("snapshot.json");

    let records = create_records_with_non_finite_values();
    write_records_to_parquet(parquet_path.to_str().unwrap(), &records)
        .expect("Failed to write Parquet");

    let creature = create_creature_with_contribution_overflow();
    let options = ExportOptions::default();

    export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &options,
    )
    .expect("Export should succeed even with NaN/Infinity inputs");

    // If serialisation produced invalid JSON, we would fail to parse it here.
    let file = File::open(&snapshot_path).expect("Failed to open snapshot");
    let reader = BufReader::new(file);
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(reader).expect("Snapshot JSON should be valid");

    // Ensure the derived synapse data is finite and JSON-safe.
    let syn = snapshot
        .derived
        .synapses
        .get("hidden-0→output-0")
        .expect("Expected hidden-0→output-0 derived synapse entry");

    assert!(syn.weight.is_finite(), "Synapse weight must be finite");
    assert!(
        syn.contribution.iter().all(|v| v.is_finite()),
        "Contribution series must contain only finite numbers"
    );
}
