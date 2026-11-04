//! Issue #2515 — the streaming `.tmp` recording must be created exclusively.
//!
//! `start_session` used `File::create` on `<temp_dir>/discovery_data.parquet.tmp`, which
//! truncates an existing file and follows a symlink. These tests drive the public streaming
//! API and assert that a pre-existing occupant at the tmp path is refused and left untouched.

use neat_ai_discovery::parquet_format::read_all_records_from_parquet;
use neat_ai_discovery::streaming::{append_records, cancel_session, finish_session, start_session};
use neat_ai_discovery::{CreatureJson, NeuronData, NeuronJson};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

const TMP_NAME: &str = "discovery_data.parquet.tmp";

fn test_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![],
        input: 2,
        output: 1,
    }
}

/// One observation: two non-input neurons plus two inputs = 4 records.
fn batch(obs_index: u32) -> Vec<(u32, Vec<NeuronData>, Vec<f32>)> {
    vec![(
        obs_index,
        vec![
            NeuronData {
                neuron_uuid: "hidden-1".to_string(),
                activation: 0.5,
                value: Some(0.4),
                errors: vec![0.1],
            },
            NeuronData {
                neuron_uuid: "output-0".to_string(),
                activation: 0.6,
                value: Some(0.6),
                errors: vec![0.0],
            },
        ],
        vec![0.1, 0.2],
    )]
}

const RECORDS_PER_BATCH: u64 = 4;

fn dir_string(dir: &TempDir) -> String {
    dir.path().to_str().unwrap().to_string()
}

fn start(dir: &TempDir) -> anyhow::Result<String> {
    start_session(test_creature(), dir_string(dir))
}

/// A pre-existing regular file at the tmp path must be refused, not truncated.
#[test]
fn existing_regular_file_is_preserved_and_start_session_fails() {
    let dir = TempDir::new().unwrap();
    let tmp_file = dir.path().join(TMP_NAME);
    fs::write(&tmp_file, b"keep me").unwrap();

    let err = start(&dir).expect_err("start_session must refuse an existing tmp file");
    assert!(
        format!("{err:#}").contains(TMP_NAME),
        "error chain should name the tmp path: {err:#}"
    );
    assert_eq!(fs::read(&tmp_file).unwrap(), b"keep me");

    // Positive control: with the occupier removed, the same directory is accepted.
    fs::remove_file(&tmp_file).unwrap();
    let id = start(&dir).expect("start_session should succeed once the occupier is removed");
    cancel_session(&id).unwrap();
}

/// A symlink at the tmp path must not be followed into an outside target.
#[cfg(unix)]
#[test]
fn symlink_at_tmp_path_is_not_written_through() {
    let dir = TempDir::new().unwrap();
    let tmp_file = dir.path().join(TMP_NAME);
    let target = dir.path().join("target.bin");
    fs::write(&target, b"target bytes").unwrap();
    std::os::unix::fs::symlink(&target, &tmp_file).unwrap();

    start(&dir).expect_err("start_session must refuse a symlink at the tmp path");
    assert_eq!(fs::read(&target).unwrap(), b"target bytes");
    assert!(
        fs::symlink_metadata(&tmp_file).unwrap().is_symlink(),
        "the symlink should be left in place"
    );

    // Positive control: with the symlink removed, the same directory is accepted.
    fs::remove_file(&tmp_file).unwrap();
    let id = start(&dir).expect("start_session should succeed once the symlink is removed");
    cancel_session(&id).unwrap();
}

/// A dangling symlink at the tmp path must not cause the target to be created.
#[cfg(unix)]
#[test]
fn dangling_symlink_is_not_created_through() {
    let dir = TempDir::new().unwrap();
    let tmp_file = dir.path().join(TMP_NAME);
    let missing = dir.path().join("missing.bin");
    std::os::unix::fs::symlink(&missing, &tmp_file).unwrap();

    start(&dir).expect_err("start_session must refuse a dangling symlink at the tmp path");
    assert!(
        !missing.exists(),
        "start_session must not create the symlink target"
    );

    // Positive control: with the dangling symlink removed, the same directory is accepted.
    fs::remove_file(&tmp_file).unwrap();
    let id =
        start(&dir).expect("start_session should succeed once the dangling symlink is removed");
    cancel_session(&id).unwrap();
}

/// A second session on the same directory must not clobber the first session's recording.
#[test]
fn second_session_on_same_temp_dir_does_not_corrupt_the_first() {
    let dir = TempDir::new().unwrap();
    let tmp_file = dir.path().join(TMP_NAME);

    let id_a = start(&dir).unwrap();
    let appended = append_records(&id_a, batch(0)).unwrap();
    assert_eq!(appended, RECORDS_PER_BATCH);
    let len_before = fs::metadata(&tmp_file).unwrap().len();

    let err = start(&dir).expect_err("a second session on the same tmp path must be refused");
    assert!(
        format!("{err:#}").contains(TMP_NAME),
        "error chain should name the tmp path: {err:#}"
    );
    assert_eq!(
        fs::metadata(&tmp_file).unwrap().len(),
        len_before,
        "the refused start must not truncate the first session's tmp file"
    );

    // The first session must still finalise with every appended record intact.
    let (result_dir, file, total) = finish_session(&id_a).expect("first session should finish");
    assert_eq!(total, appended);
    let final_path = Path::new(&result_dir).join(&file);
    let records = read_all_records_from_parquet(final_path.to_str().unwrap())
        .expect("finalised recording should be readable");
    assert_eq!(
        records.len() as u64,
        appended,
        "finalised recording should hold every appended record"
    );

    // Positive control: the tmp path is free again once the first session has finished.
    let id_b = start(&dir).expect("start_session should succeed after the first session finished");
    cancel_session(&id_b).unwrap();
}

/// A recording still in progress must survive a refused second start unchanged.
#[test]
fn retained_recording_survives_a_second_start_session() {
    let dir = TempDir::new().unwrap();
    let tmp_file = dir.path().join(TMP_NAME);

    let id_a = start(&dir).unwrap();
    append_records(&id_a, batch(0)).unwrap();
    let retained = fs::read(&tmp_file).unwrap();

    let err = start(&dir).expect_err("a second session on the same tmp path must be refused");
    assert!(
        format!("{err:#}").contains(TMP_NAME),
        "error chain should name the tmp path: {err:#}"
    );
    assert_eq!(
        fs::read(&tmp_file).unwrap(),
        retained,
        "the refused start must leave the retained recording byte-identical"
    );

    // Cancelling the first session removes its tmp file.
    cancel_session(&id_a).unwrap();
    assert!(!tmp_file.exists(), "cancel should remove the tmp file");

    // Positive control: with only the occupier removed, a fresh session is accepted.
    let id_b = start(&dir).expect("start_session should succeed once the occupier is gone");
    cancel_session(&id_b).unwrap();
}
