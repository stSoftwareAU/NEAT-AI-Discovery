//! Issue #2005: one shared decoder for discovery Parquet record batches.
//!
//! The rule for turning a discovery Parquet batch into `DiscoverRecord`s —
//! resolve the five columns by name, downcast each to its Arrow type, then per
//! row read the nullable `value` and decode the `errors` list — used to be
//! copy-pasted into four readers, and the copies had diverged. These tests
//! assert the reconciled behaviour:
//!
//! 1. Every reader decodes the same file into the same records (nullable
//!    `value` and variable-length `errors` lists included).
//! 2. The streaming block loader charges the shared [`DecodeBudget`] just like
//!    the three `reader.rs` paths (Issue #1869), so it is no longer the one
//!    decode path where an oversized file materialises unbounded.

use neat_ai_discovery::analysis::cache::StreamingRecordCache;
use neat_ai_discovery::ffi_types::DiscoveryErrorKind;
use neat_ai_discovery::parquet_format::{
    ColumnProfile, read_all_records_from_parquet, read_all_records_grouped_by_neuron,
    read_records_from_parquet, read_records_from_parquet_with_limit_and_profile,
    write_records_to_parquet,
};
use neat_ai_discovery::types::DiscoverRecord;
use serial_test::serial;
use tempfile::NamedTempFile;

const NEURON: &str = "11111111-2222-3333-4444-555555555555";

/// Write a parquet whose rows exercise the decode rule: a null `value`, an
/// empty `errors` list, and lists of differing lengths.
fn write_mixed_parquet() -> NamedTempFile {
    let file = NamedTempFile::with_suffix(".parquet").expect("temp file");
    let records = vec![
        DiscoverRecord::new(0, NEURON.to_string(), Some(0.5), 0.75, vec![0.1, 0.2]),
        DiscoverRecord::new(1, NEURON.to_string(), None, -0.25, vec![]),
        DiscoverRecord::new(2, NEURON.to_string(), Some(-1.5), 0.0, vec![0.3, 0.4, 0.5]),
        DiscoverRecord::new(3, "other-neuron".to_string(), Some(0.25), 0.5, vec![0.6]),
    ];
    write_records_to_parquet(file.path().to_str().expect("utf-8 path"), &records).expect("write");
    file
}

/// A low-entropy recording large enough that a single block exceeds a 1 MB
/// decode budget once materialised.
fn write_bulky_parquet(rows: u32, errors_per_row: usize) -> NamedTempFile {
    let file = NamedTempFile::with_suffix(".parquet").expect("temp file");
    let records: Vec<DiscoverRecord> = (0..rows)
        .map(|i| {
            DiscoverRecord::new(
                i,
                NEURON.to_string(),
                Some(0.5),
                0.5,
                vec![0.25f32; errors_per_row],
            )
        })
        .collect();
    write_records_to_parquet(file.path().to_str().expect("utf-8 path"), &records).expect("write");
    file
}

fn sorted_for(records: &[DiscoverRecord], uuid: &str) -> Vec<DiscoverRecord> {
    let mut filtered: Vec<DiscoverRecord> = records
        .iter()
        .filter(|r| r.neuron_uuid == uuid)
        .cloned()
        .collect();
    filtered.sort_by_key(|r| r.obs_index);
    filtered
}

fn assert_same(expected: &[DiscoverRecord], actual: &[DiscoverRecord], path_name: &str) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{path_name} returned a different record count"
    );
    for (want, got) in expected.iter().zip(actual.iter()) {
        assert_eq!(want.obs_index, got.obs_index, "{path_name} obs_index");
        assert_eq!(want.neuron_uuid, got.neuron_uuid, "{path_name} neuron_uuid");
        assert_eq!(want.value, got.value, "{path_name} value (nullable)");
        assert_eq!(want.activation, got.activation, "{path_name} activation");
        assert_eq!(want.errors, got.errors, "{path_name} errors list");
    }
}

#[test]
fn every_reader_decodes_a_batch_identically() {
    let file = write_mixed_parquet();
    let path = file.path().to_str().expect("utf-8 path");

    let all = read_all_records_from_parquet(path).expect("read all");
    let expected = sorted_for(&all, NEURON);
    assert_eq!(expected.len(), 3, "three rows belong to the target neuron");
    assert_eq!(expected[1].value, None, "the null value decodes as None");
    assert!(expected[1].errors.is_empty(), "an empty list stays empty");

    let grouped = read_all_records_grouped_by_neuron(path).expect("grouped read");
    let mut grouped_records = grouped.get(NEURON).cloned().unwrap_or_default();
    grouped_records.sort_by_key(|r| r.obs_index);
    assert_same(&expected, &grouped_records, "grouped reader");

    let mut filtered = read_records_from_parquet(path, NEURON).expect("filtered read");
    filtered.sort_by_key(|r| r.obs_index);
    assert_same(&expected, &filtered, "uuid-filtered reader");

    let limited = read_records_from_parquet_with_limit_and_profile(path, None, ColumnProfile::Full)
        .expect("limited read");
    assert_same(&expected, &sorted_for(&limited, NEURON), "limited reader");

    let cache = StreamingRecordCache::new(path, None, None).expect("streaming cache");
    let mut streamed = (*cache.get(NEURON).expect("streaming read")).clone();
    streamed.sort_by_key(|r| r.obs_index);
    assert_same(&expected, &streamed, "streaming block loader");
}

#[test]
fn the_without_errors_profile_still_yields_empty_error_lists() {
    let file = write_mixed_parquet();
    let path = file.path().to_str().expect("utf-8 path");

    let records =
        read_records_from_parquet_with_limit_and_profile(path, None, ColumnProfile::WithoutErrors)
            .expect("projected read");

    assert_eq!(records.len(), 4, "every row is still returned");
    assert!(
        records.iter().all(|r| r.errors.is_empty()),
        "the projected-away errors column decodes as an empty vec"
    );
    let with_null = records
        .iter()
        .find(|r| r.obs_index == 1)
        .expect("row with a null value");
    assert_eq!(with_null.value, None, "nullability survives the projection");
}

/// Issue #1869 landed on the three `reader.rs` copies but not on the streaming
/// block loader. With one shared decoder the streaming path charges the same
/// budget, so an oversized block fails loud instead of materialising.
#[test]
#[serial]
fn the_streaming_block_loader_charges_the_decode_budget() {
    let file = write_bulky_parquet(20_000, 32);
    let path = file.path().to_str().expect("utf-8 path");

    // SAFETY: `#[serial]` guarantees no other test runs concurrently, so no
    // other thread can read the environment while it is being mutated.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB", "1") };
    let bounded = StreamingRecordCache::new(path, None, None)
        .expect("the block index builds without materialising records")
        .get(NEURON);
    // SAFETY: as above — `#[serial]` serialises this test against all others.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB") };

    // Match rather than `expect_err`: the success value is 10k records, and
    // debug-printing them would bury the assertion in megabytes of output.
    let err = match bounded {
        Ok(records) => panic!(
            "a 1 MB budget cannot hold a 10k-record block, but {} records were returned",
            records.len()
        ),
        Err(err) => err,
    };
    let kind = neat_ai_discovery::ffi_types::classify_anyhow_error(&err);
    assert_eq!(
        kind,
        DiscoveryErrorKind::MemoryExhausted,
        "budget aborts are typed as memory exhaustion, got {kind:?}: {err}"
    );
    assert!(
        err.to_string().contains("decode budget"),
        "the error should name the decode budget: {err}"
    );
}

#[test]
#[serial]
fn the_streaming_block_loader_succeeds_within_a_generous_budget() {
    let file = write_bulky_parquet(2_000, 4);
    let path = file.path().to_str().expect("utf-8 path");

    // SAFETY: `#[serial]` guarantees no other test runs concurrently.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB", "4096") };
    let loaded = StreamingRecordCache::new(path, None, None)
        .expect("streaming cache")
        .get(NEURON);
    // SAFETY: as above.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB") };

    let records = loaded.expect("a 4 GB budget comfortably holds the block");
    assert!(
        !records.is_empty(),
        "records are still returned when the budget fits"
    );
}
