//! Issue #2067 — the contracts `src/record/`'s module headers now state.
//!
//! `processing.rs` and `validation.rs` opened with a restatement of their file
//! names, so a reader looking for where AGENTS.md's "Atomic Record Writes"
//! invariant is enforced, or which preconditions recording rejects, learnt
//! nothing from the module index. Both headers now state those contracts;
//! these tests drive each documented claim through the public API so the prose
//! cannot drift into fiction.

use std::collections::BTreeMap;

use neat_ai_discovery::parquet_format::read_all_records_from_parquet;
use neat_ai_discovery::record::record_discovery_data;
use neat_ai_discovery::{
    CreatureJson, NeuronData, NeuronJson, RecordDiscoveryInput, TrainingRecord,
};

/// `processing.rs` header: every record for one observation — one per
/// non-input neuron, plus one per input activation — is written together, and
/// data from different training records is never mixed within an observation.
#[test]
fn each_observation_is_written_as_a_complete_group() {
    let input = recording_input("issue-2067-atomic", 3);
    let result = record_discovery_data(&input).expect("recording must succeed");

    let path = format!("{}/{}", result.temp_dir, result.file);
    let records = read_all_records_from_parquet(&path).expect("written file must be readable");
    let _cleanup = TempDirGuard(result.temp_dir);

    let mut by_observation: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for record in &records {
        by_observation
            .entry(record.obs_index)
            .or_default()
            .push(record.neuron_uuid.clone());
    }

    assert_eq!(
        by_observation.keys().copied().collect::<Vec<_>>(),
        vec![0, 1, 2],
        "one group per training observation"
    );
    for (obs_index, mut uuids) in by_observation {
        uuids.sort();
        assert_eq!(
            uuids,
            vec!["input-0".to_string(), "output-0".to_string()],
            "observation {obs_index} must carry its non-input neuron and its input activation"
        );
    }

    // Data from different training records is never mixed: each observation's
    // activation is the one its own training record supplied.
    for record in &records {
        let expected = observation_activation(record.obs_index);
        assert!(
            (record.activation - expected).abs() < f32::EPSILON,
            "observation {} carried activation {} from another training record (expected {expected})",
            record.obs_index,
            record.activation
        );
    }
}

/// `validation.rs` header: a creature with no non-input neuron is rejected,
/// because input neurons are skipped and such a creature yields nothing to
/// record.
#[test]
fn creature_without_non_input_neurons_is_rejected() {
    let mut input = recording_input("issue-2067-no-non-input", 1);
    input.creature.neurons = vec![NeuronJson {
        uuid: "input-0".to_string(),
        neuron_type: "input".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }];

    let err = record_discovery_data(&input).expect_err("no non-input neuron must be rejected");
    assert!(
        err.to_string().contains("no non-input neurons"),
        "error must name the missing non-input neurons: {err}"
    );
}

/// `validation.rs` header: empty training data is an error, not an empty
/// success — no Parquet file is published for it.
#[test]
fn empty_training_data_is_an_error_not_an_empty_file() {
    let mut input = recording_input("issue-2067-empty", 0);
    input.training_data.clear();

    let err = record_discovery_data(&input).expect_err("empty training data must be rejected");
    assert!(
        err.to_string()
            .contains("No discovery records were generated"),
        "error must say no records were derivable: {err}"
    );
}

/// `validation.rs` header: caller-supplied `record_indices` must match the
/// training-data length.
#[test]
fn record_indices_length_must_match_training_data() {
    let mut input = recording_input("issue-2067-length", 2);
    input.record_indices = Some(vec![0]);

    let err = record_discovery_data(&input).expect_err("mismatched length must be rejected");
    assert!(
        err.to_string().contains("must match training_data length"),
        "error must name the length mismatch: {err}"
    );
}

/// `validation.rs` header: caller-supplied `record_indices` must be free of
/// duplicates, so two observations can never share an `obs_index`.
#[test]
fn duplicate_record_indices_are_rejected() {
    let mut input = recording_input("issue-2067-duplicate", 2);
    input.record_indices = Some(vec![7, 7]);

    let err = record_discovery_data(&input).expect_err("duplicate indices must be rejected");
    assert!(
        err.to_string().contains("duplicated in record_indices"),
        "error must name the duplicate index: {err}"
    );
}

/// The activation the training record at `obs_index` supplies. Distinct per
/// observation so a mixed write is detectable.
fn observation_activation(obs_index: u32) -> f32 {
    let small = u8::try_from(obs_index).expect("these tests use small observation counts");
    0.5 + f32::from(small)
}

/// A minimal valid recording input with `observations` training records, each
/// carrying one input and one output neuron with per-observation values.
fn recording_input(tag: &str, observations: u32) -> RecordDiscoveryInput {
    let training_data = (0..observations)
        .map(|obs_index| {
            let activation = observation_activation(obs_index);
            TrainingRecord {
                input: vec![activation],
                output: vec![activation],
                neuron_data: Some(vec![NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation,
                    value: Some(activation),
                    errors: vec![0.0],
                }]),
            }
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
        temp_dir: std::env::temp_dir()
            .join(format!("neat-ai-discovery-{tag}"))
            .to_string_lossy()
            .to_string(),
        binary_file_path: None,
        record_indices: None,
        timeout_seconds: None,
        task_descriptor: None,
    }
}

/// Removes the temporary recording directory when the test ends.
struct TempDirGuard(String);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
