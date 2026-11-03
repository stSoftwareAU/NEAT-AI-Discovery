//! Issue #2452: the shared-decode cache must not serve a stale decode when a
//! parquet file is rewritten in place.
//!
//! The original key was `(path, dev/ino, len, mtime)`. A rewrite that keeps the
//! inode and length, followed by a restored mtime (`cp -p`, `rsync --inplace -t`,
//! `touch -r`), reproduced that key exactly, so the superseded decode was served.
//! The key now also carries `ctime`, which a user cannot set, so the restored
//! mtime no longer masks the rewrite.
//!
//! Two properties are exercised here:
//!
//! 1. An in-place rewrite with the original length and a restored mtime is a
//!    cache miss, and the caller receives the rewritten records.
//! 2. `record_discovery_data` invalidates the shared slot after finalising the
//!    file, so a re-recording never serves its previous decode.
//!
//! These tests share the process-global cache, so they are `#[serial]`.

use neat_ai_discovery::parquet_format::shared_records::{
    SharedGroupedRecords, decodes_for_path, invalidate, load_grouped_records_shared,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::record::record_discovery_data;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    CreatureJson, NeuronData, NeuronJson, RecordDiscoveryInput, TrainingRecord,
};
use serial_test::serial;
use std::sync::Arc;
use std::time::Duration;
use tempfile::NamedTempFile;

/// Upper bound on in-place rewrites when the coarse filesystem clock has not yet
/// ticked past the first load's `ctime`.
#[cfg(unix)]
const MAX_REWRITE_ATTEMPTS: usize = 50;

/// Pause between rewrite attempts. Long enough for a coarse kernel timestamp
/// tick to advance, short enough to keep the test fast.
#[cfg(unix)]
const REWRITE_BACKOFF: Duration = Duration::from_millis(10);

fn records(activation: f32) -> Vec<DiscoverRecord> {
    vec![DiscoverRecord::new(
        0,
        "neuron-1".to_string(),
        Some(0.5),
        activation,
        vec![0.1],
    )]
}

/// The activation of the single record cached under `neuron-1`.
fn first_activation(grouped: &Arc<SharedGroupedRecords>) -> f32 {
    grouped
        .get("neuron-1")
        .expect("neuron-1 present")
        .first()
        .expect("one record for neuron-1")
        .activation
}

/// Rewriting a file in place (same inode, same length) and restoring its mtime
/// must not reproduce the cached key. The `ctime` component changes because the
/// rewrite and the mtime restore both update it, and a user cannot set it back.
#[cfg(unix)]
#[test]
#[serial]
fn inplace_rewrite_with_restored_mtime_is_not_a_cache_hit() {
    use std::os::unix::fs::MetadataExt;

    let tmp = NamedTempFile::new().expect("create temp file");
    let path = tmp.path().to_str().expect("utf-8 path").to_string();
    write_records_to_parquet(&path, &records(1.0)).expect("write original");
    invalidate();

    let first = load_grouped_records_shared(&path, None).expect("first load decodes");
    assert!(
        (first_activation(&first) - 1.0).abs() < f32::EPSILON,
        "the first load must decode the original activation",
    );

    let before = std::fs::metadata(&path).expect("stat original");
    let original_mtime = before.modified().expect("original mtime");

    // Rewrite in place, then restore the original mtime. Each attempt must leave
    // inode, length and mtime unchanged; only ctime may move. If the coarse
    // filesystem clock has not ticked since the first load, back off and rewrite
    // again until it does.
    let mut ctime_changed = false;
    for _ in 0..MAX_REWRITE_ATTEMPTS {
        write_records_to_parquet(&path, &records(9.0)).expect("rewrite in place");
        std::fs::File::options()
            .write(true)
            .open(&path)
            .expect("open for mtime restore")
            .set_modified(original_mtime)
            .expect("restore original mtime");

        let after = std::fs::metadata(&path).expect("stat rewritten");
        assert_eq!(after.ino(), before.ino(), "the rewrite must keep the inode");
        assert_eq!(
            after.len(),
            before.len(),
            "the rewrite must keep the length"
        );
        assert_eq!(
            after.modified().expect("rewritten mtime"),
            original_mtime,
            "the restored mtime must equal the original",
        );

        if (after.ctime(), after.ctime_nsec()) != (before.ctime(), before.ctime_nsec()) {
            ctime_changed = true;
            break;
        }
        std::thread::sleep(REWRITE_BACKOFF);
    }
    assert!(
        ctime_changed,
        "ctime never changed across {MAX_REWRITE_ATTEMPTS} in-place rewrites; \
         the restored-mtime case cannot be exercised on this filesystem",
    );

    let second = load_grouped_records_shared(&path, None).expect("second load decodes");
    assert!(
        !Arc::ptr_eq(&first, &second),
        "an in-place rewrite with a restored mtime must not be served from the cache",
    );
    assert!(
        (first_activation(&second) - 9.0).abs() < f32::EPSILON,
        "the second load must return the rewritten activation",
    );
    assert_eq!(
        decodes_for_path(&path),
        1,
        "the rewritten file decodes fresh"
    );

    invalidate();
}

/// A minimal valid recording input with one observation, one input neuron and
/// one output neuron, recorded into `temp_dir`.
fn recording_input(temp_dir: &str) -> RecordDiscoveryInput {
    let activation = 0.5_f32;
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
        training_data: vec![TrainingRecord {
            input: vec![activation],
            output: vec![activation],
            neuron_data: Some(vec![NeuronData {
                neuron_uuid: "output-0".to_string(),
                activation,
                value: Some(activation),
                errors: vec![0.0],
            }]),
        }],
        temp_dir: temp_dir.to_string(),
        binary_file_path: None,
        record_indices: None,
        timeout_seconds: None,
        task_descriptor: None,
    }
}

/// Re-recording into a directory whose parquet is resident in the shared cache
/// must invalidate that slot, so the previous decode is never served again.
#[test]
#[serial]
fn record_discovery_data_invalidates_the_shared_decode_cache() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp_dir = dir.path().to_string_lossy().to_string();
    let path = dir
        .path()
        .join("discovery_data.parquet")
        .to_string_lossy()
        .to_string();
    invalidate();

    let input = recording_input(&temp_dir);
    record_discovery_data(&input).expect("first recording");

    load_grouped_records_shared(&path, None).expect("load recorded parquet");
    assert_eq!(
        decodes_for_path(&path),
        1,
        "the recorded parquet should be resident in the shared cache",
    );

    record_discovery_data(&input).expect("re-recording");
    assert_eq!(
        decodes_for_path(&path),
        0,
        "re-recording must invalidate the shared cache slot",
    );

    invalidate();
}
