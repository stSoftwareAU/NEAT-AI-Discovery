use std::fs;

use neat_ai_discovery::parquet_format::{read_all_records_from_parquet, write_records_to_parquet};
use neat_ai_discovery::record::record_discovery_data;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    CreatureJson, NeuronData, NeuronJson, RecordDiscoveryInput, TrainingRecord,
};
use tempfile::tempdir;

#[test]
fn processing_error_preserves_existing_recording_and_cleans_temporary_file() {
    let dir = tempdir().expect("create isolated temp directory");
    let final_path = dir.path().join("discovery_data.parquet");
    let existing = vec![sample_record("old-neuron", 0.25)];
    write_records_to_parquet(final_path.to_str().expect("UTF-8 path"), &existing)
        .expect("seed valid recording");
    let original_bytes = fs::read(&final_path).expect("read original bytes");

    let mut input = recording_input(dir.path());
    input.training_data[1].neuron_data = None;
    let error = record_discovery_data(&input).expect_err("missing neuron data must fail");

    assert!(
        error
            .to_string()
            .contains("No pre-computed neuron_data provided"),
        "expected the processing error, got: {error:#}"
    );
    assert_eq!(
        fs::read(&final_path).expect("original recording remains readable as bytes"),
        original_bytes,
        "failed processing must preserve the published file byte-for-byte"
    );
    assert_eq!(
        read_all_records_from_parquet(final_path.to_str().expect("UTF-8 path"))
            .expect("original recording remains a valid Parquet file"),
        existing,
    );
    assert!(
        !temporary_path(&final_path).exists(),
        "failed processing must remove the temporary file"
    );
}

#[test]
fn successful_recording_replaces_existing_file_with_readable_data() {
    let dir = tempdir().expect("create isolated temp directory");
    let final_path = dir.path().join("discovery_data.parquet");
    write_records_to_parquet(
        final_path.to_str().expect("UTF-8 path"),
        &[sample_record("old-neuron", 0.25)],
    )
    .expect("seed valid recording");

    let result = record_discovery_data(&recording_input(dir.path()))
        .expect("successful recording should publish");

    assert_eq!(result.file, "discovery_data.parquet");
    let records = read_all_records_from_parquet(final_path.to_str().expect("UTF-8 path"))
        .expect("published recording is readable");
    assert_eq!(records.len(), 4, "two observations each produce two rows");
    assert!(
        records
            .iter()
            .all(|record| record.neuron_uuid != "old-neuron")
    );
    assert!(!temporary_path(&final_path).exists());
}

#[test]
fn processing_error_without_existing_recording_cleans_temporary_file() {
    let dir = tempdir().expect("create isolated temp directory");
    let final_path = dir.path().join("discovery_data.parquet");
    let mut input = recording_input(dir.path());
    input.training_data[0].neuron_data = None;

    let error = record_discovery_data(&input).expect_err("missing neuron data must fail");

    assert!(
        error
            .to_string()
            .contains("No pre-computed neuron_data provided"),
        "expected the processing error, got: {error:#}"
    );
    assert!(
        !final_path.exists(),
        "failed recording must not be published"
    );
    assert!(!temporary_path(&final_path).exists());
}

#[test]
fn rename_error_cleans_temporary_file_and_preserves_destination_directory() {
    let dir = tempdir().expect("create isolated temp directory");
    let final_path = dir.path().join("discovery_data.parquet");
    fs::create_dir(&final_path).expect("block publication with a directory");
    let occupied = final_path.join("occupied.txt");
    fs::write(&occupied, b"keep destination").expect("seed non-empty destination");

    let error = record_discovery_data(&recording_input(dir.path()))
        .expect_err("renaming onto a non-empty directory must fail");

    assert!(
        error
            .to_string()
            .contains("Failed to publish discovery Parquet file"),
        "expected the publication error, got: {error:#}"
    );
    assert_eq!(
        fs::read(&occupied).expect("destination remains"),
        b"keep destination"
    );
    assert!(!temporary_path(&final_path).exists());
}

#[test]
fn writer_construction_error_reports_cleanup_failure() {
    let dir = tempdir().expect("create isolated temp directory");
    let final_path = dir.path().join("discovery_data.parquet");
    let tmp_path = temporary_path(&final_path);
    fs::create_dir(&tmp_path).expect("block temporary-file construction");

    let error = record_discovery_data(&recording_input(dir.path()))
        .expect_err("writer cannot create a file where a directory exists");
    let chain = format!("{error:#}");

    assert!(
        chain.contains("Failed to initialise Parquet writer"),
        "{chain}"
    );
    assert!(
        chain.contains("Failed to remove temporary Parquet file"),
        "{chain}"
    );
    assert!(tmp_path.is_dir(), "cleanup must not remove a directory");
}

fn sample_record(uuid: &str, activation: f32) -> DiscoverRecord {
    DiscoverRecord::new(0, uuid.to_string(), Some(activation), activation, vec![0.0])
}

fn recording_input(temp_dir: &std::path::Path) -> RecordDiscoveryInput {
    let training_data = [0.4, 0.7]
        .into_iter()
        .map(|activation| TrainingRecord {
            input: vec![activation],
            output: vec![activation],
            neuron_data: Some(vec![NeuronData {
                neuron_uuid: "output-0".to_string(),
                activation,
                value: Some(activation),
                errors: vec![0.0],
            }]),
        })
        .collect();

    RecordDiscoveryInput {
        creature: CreatureJson {
            neurons: vec![
                NeuronJson {
                    uuid: "input-0".to_string(),
                    neuron_type: "input".to_string(),
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
            synapses: Vec::new(),
            input: 1,
            output: 1,
        },
        training_data,
        temp_dir: temp_dir.to_string_lossy().into_owned(),
        binary_file_path: None,
        record_indices: None,
        timeout_seconds: None,
        task_descriptor: None,
    }
}

fn temporary_path(final_path: &std::path::Path) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("{}.tmp", final_path.display()))
}
