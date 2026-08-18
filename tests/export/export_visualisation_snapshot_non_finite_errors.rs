//! Regression test: non-finite error values must not break snapshot export.
//!
//! The snapshot exporter computes summary statistics for recorded errors.
//! `compute_stats()` already filters out NaN/±Infinity so summary stats remain valid,
//! but MAE/MSE must apply the same filtering to avoid generating NaN in JSON output.

use neat_ai_discovery::export::{
    ExportOptions, VisualisationSnapshot, export_visualisation_snapshot,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson};
use std::fs::File;
use std::io::BufReader;
use tempfile::tempdir;

fn create_minimal_output_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
        input: 1,
        output: 1,
    }
}

#[test]
fn export_visualisation_snapshot_filters_non_finite_errors_for_mae_and_mse() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_path = temp_dir.path().join("records.parquet");
    let snapshot_path = temp_dir.path().join("snapshot.json");

    // Two observations, each includes non-finite errors plus finite errors [1.0, -2.0].
    // Finite-only MAE = (|1| + |2| + |1| + |2|) / 4 = 1.5
    // Finite-only MSE = (1² + 2² + 1² + 2²) / 4 = 2.5
    let records = vec![
        DiscoverRecord::new(
            0,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![f32::NAN, 1.0, f32::INFINITY, -2.0],
        ),
        DiscoverRecord::new(
            1,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![f32::NEG_INFINITY, 1.0, -2.0, f32::NAN],
        ),
    ];

    write_records_to_parquet(parquet_path.to_str().unwrap(), &records)
        .expect("Failed to write Parquet");

    let creature = create_minimal_output_creature();
    let options = ExportOptions::default();

    export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &options,
    )
    .expect("Export should succeed even with non-finite errors");

    let file = File::open(&snapshot_path).expect("Failed to open snapshot");
    let reader = BufReader::new(file);
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(reader).expect("Failed to parse snapshot JSON");

    let stats = &snapshot
        .recording
        .neurons
        .get("output-0")
        .expect("Should have output-0 recording")
        .stats;

    let errors = &snapshot
        .recording
        .neurons
        .get("output-0")
        .expect("Should have output-0 recording")
        .errors;

    assert_eq!(errors.len(), 2, "Should have errors for 2 observations");
    assert_eq!(
        errors[0].len(),
        2,
        "Non-finite errors should be dropped (obs 0 should keep only finite values)"
    );
    assert_eq!(
        errors[1].len(),
        2,
        "Non-finite errors should be dropped (obs 1 should keep only finite values)"
    );

    assert!(
        stats.mean_absolute_error.is_finite(),
        "MAE should be finite even with non-finite errors, got {}",
        stats.mean_absolute_error
    );
    assert!(
        stats.mean_squared_error.is_finite(),
        "MSE should be finite even with non-finite errors, got {}",
        stats.mean_squared_error
    );

    assert!(
        (stats.mean_absolute_error - 1.5).abs() < 1e-5,
        "Expected finite-only MAE 1.5, got {}",
        stats.mean_absolute_error
    );
    assert!(
        (stats.mean_squared_error - 2.5).abs() < 1e-5,
        "Expected finite-only MSE 2.5, got {}",
        stats.mean_squared_error
    );
}
