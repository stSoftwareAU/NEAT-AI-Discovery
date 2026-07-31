//! Issue #1876 — appending to a cancelled session must never report success.
//!
//! `append_records` returned `Ok(records_in_batch)` for a session that had been
//! concurrently cancelled or TTL-swept, acknowledging records that were about to be
//! discarded with the session's `.tmp` file. These tests drive the public streaming
//! API and assert that every append made after cancellation fails loudly.

use neat_ai_discovery::streaming::{append_records, cancel_session, start_session};
use neat_ai_discovery::{CreatureJson, NeuronData, NeuronJson};
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

/// Either error is acceptable: the session is gone from the map ("not found") or
/// the caller reached a handle that has been tombstoned ("cancelled").
fn assert_session_gone(err: &anyhow::Error) {
    let msg = err.to_string();
    assert!(
        msg.contains("Session not found") || msg.contains("Session cancelled"),
        "unexpected error: {msg}"
    );
}

#[test]
fn appending_to_a_cancelled_session_fails_loudly() {
    let dir = TempDir::new().unwrap();
    let id = start_session(test_creature(), dir.path().to_str().unwrap().to_string()).unwrap();

    append_records(&id, batch(0)).unwrap();
    cancel_session(&id).unwrap();

    let err = append_records(&id, batch(1))
        .expect_err("appending to a cancelled session must return an error");
    assert_session_gone(&err);
}

/// A host appending in a loop must see the cancellation, not a stream of `Ok`s for
/// records that are being discarded.
#[test]
fn concurrent_cancel_stops_appends_reporting_success() {
    const MAX_BATCHES: u32 = 10_000;

    let dir = TempDir::new().unwrap();
    let tmp_file = dir.path().join("discovery_data.parquet.tmp");
    let id = start_session(test_creature(), dir.path().to_str().unwrap().to_string()).unwrap();

    let id_worker = id.clone();
    let worker = std::thread::spawn(move || {
        for obs_index in 0..MAX_BATCHES {
            if let Err(err) = append_records(&id_worker, batch(obs_index)) {
                return Some(err);
            }
        }
        None
    });

    cancel_session(&id).unwrap();

    let err = worker
        .join()
        .expect("append worker should not panic")
        .expect("appends must start failing once the session is cancelled");
    assert_session_gone(&err);

    assert!(
        !tmp_file.exists(),
        "expected the cancelled session's partial file to be removed"
    );
}
