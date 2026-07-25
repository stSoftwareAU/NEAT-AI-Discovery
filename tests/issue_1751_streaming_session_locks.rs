//! Issue #1751 — per-session locking for streaming recording sessions.
//!
//! `append_records` used to hold the process-global `SESSIONS` mutex for the whole
//! batch, including every Parquet disk write, so unrelated sessions queued behind
//! one another's I/O. These tests exercise the public streaming API concurrently
//! and assert that every session's records are recorded correctly.

use neat_ai_discovery::streaming::{append_records, cancel_session, finish_session, start_session};
use neat_ai_discovery::{CreatureJson, NeuronData, NeuronJson};
use std::path::Path;
use tempfile::TempDir;

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
const BATCHES_PER_SESSION: u32 = 25;

#[test]
fn concurrent_sessions_each_record_their_own_batches() {
    let dirs: Vec<TempDir> = (0..4).map(|_| TempDir::new().unwrap()).collect();
    let session_ids: Vec<String> = dirs
        .iter()
        .map(|dir| {
            start_session(test_creature(), dir.path().to_str().unwrap().to_string()).unwrap()
        })
        .collect();

    // Every session appends concurrently — no session may lose records to, or be
    // starved by, another session's disk writes.
    let workers: Vec<_> = session_ids
        .iter()
        .map(|id| {
            let id = id.clone();
            std::thread::spawn(move || {
                let mut written = 0u64;
                for obs_index in 0..BATCHES_PER_SESSION {
                    written += append_records(&id, batch(obs_index)).unwrap();
                }
                written
            })
        })
        .collect();

    for (worker, id) in workers.into_iter().zip(&session_ids) {
        let written = worker.join().unwrap();
        assert_eq!(
            written,
            u64::from(BATCHES_PER_SESSION) * RECORDS_PER_BATCH,
            "every appended record must be written"
        );

        let (temp_dir, file, total) = finish_session(id).unwrap();
        assert_eq!(total, written, "session record count must match appends");
        assert!(
            Path::new(&temp_dir).join(&file).exists(),
            "expected the finalised parquet file to exist"
        );
    }
}

#[test]
fn cancelling_one_session_leaves_another_usable() {
    let dir_a = TempDir::new().unwrap();
    let dir_b = TempDir::new().unwrap();

    let id_a = start_session(test_creature(), dir_a.path().to_str().unwrap().to_string()).unwrap();
    let id_b = start_session(test_creature(), dir_b.path().to_str().unwrap().to_string()).unwrap();

    append_records(&id_a, batch(0)).unwrap();
    append_records(&id_b, batch(0)).unwrap();

    cancel_session(&id_a).unwrap();
    assert!(
        !dir_a.path().join("discovery_data.parquet.tmp").exists(),
        "expected the cancelled session's partial file to be removed"
    );

    // Session B is unaffected by A's cancellation and its file cleanup
    let written = append_records(&id_b, batch(1)).unwrap();
    assert_eq!(written, RECORDS_PER_BATCH);

    let (temp_dir, file, total) = finish_session(&id_b).unwrap();
    assert_eq!(total, RECORDS_PER_BATCH * 2);
    assert!(Path::new(&temp_dir).join(&file).exists());
}

#[test]
fn appending_to_a_finished_session_fails_loudly() {
    let dir = TempDir::new().unwrap();
    let id = start_session(test_creature(), dir.path().to_str().unwrap().to_string()).unwrap();

    append_records(&id, batch(0)).unwrap();
    finish_session(&id).unwrap();

    let err = append_records(&id, batch(1))
        .expect_err("appending to a finished session must return an error");
    assert!(
        err.to_string().contains("Session not found"),
        "unexpected error: {err}"
    );
}
