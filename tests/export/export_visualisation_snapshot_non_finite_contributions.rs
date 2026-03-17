//! Regression test: non-finite synapse contributions must not break snapshot export.
//!
//! The snapshot exporter stores per-synapse `contribution` series as:
//!   contribution[obs] = from_activation[obs] * weight
//!
//! Even when both inputs are finite, this multiplication can overflow to ±Infinity.
//! JSON cannot represent non-finite floats, so the exporter must sanitise the raw
//! contribution series before serialisation.

use neat_ai_discovery::export::{
    ExportOptions, VisualisationSnapshot, export_visualisation_snapshot,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::fs::File;
use std::io::BufReader;
use tempfile::tempdir;

fn create_creature_with_single_synapse(weight: f32) -> CreatureJson {
    CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight,
            synapse_type: None,
        }],
        input: 1,
        output: 1,
    }
}

#[test]
fn export_visualisation_snapshot_sanitises_non_finite_contribution_series() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_path = temp_dir.path().join("records.parquet");
    let snapshot_path = temp_dir.path().join("snapshot.json");

    // Finite activation that will overflow when multiplied by weight=2.0.
    // f32::MAX * 2.0 => +Infinity
    let records = vec![
        DiscoverRecord::new(
            0,
            "input-0".to_string(),
            Some(f32::MAX),
            f32::MAX,
            vec![0.0],
        ),
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.0), 0.0, vec![0.0]),
    ];

    write_records_to_parquet(parquet_path.to_str().unwrap(), &records)
        .expect("Failed to write Parquet");

    let creature = create_creature_with_single_synapse(2.0);

    // Keep the test focused on synapse series JSON serialisation.
    let options = ExportOptions {
        include_per_synapse_series: true,
        include_reconstruction_checks: false,
        max_obs: None,
        top_k_worst_samples: 5,
    };

    export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &options,
    )
    .expect("Export should succeed even if contribution overflows to non-finite");

    let file = File::open(&snapshot_path).expect("Failed to open snapshot");
    let reader = BufReader::new(file);
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(reader).expect("Failed to parse snapshot JSON");

    let syn = snapshot
        .derived
        .synapses
        .get("input-0→output-0")
        .expect("Expected derived synapse entry");

    assert_eq!(syn.contribution.len(), 1, "Should have 1 obs entry");
    assert!(
        syn.contribution[0].is_finite(),
        "Contribution series must be JSON-safe (finite), got {}",
        syn.contribution[0]
    );
}
