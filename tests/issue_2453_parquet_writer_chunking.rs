//! Issue #2453: round-trip and rejection tests for `ParquetRecordWriter` batch
//! chunking under the per-batch UUID byte and error-value limits.
//!
//! Chunk boundaries are not visible in the written file: the writer coalesces
//! batches, so these tests guard that every record survives chunking (no loss,
//! duplication or reordering). The exact chunk boundaries are pinned by the unit
//! tests of `determine_chunk_end` in `src/parquet_format/writer.rs`.

use neat_ai_discovery::parquet_format::{ParquetRecordWriter, read_all_records_from_parquet};
use neat_ai_discovery::types::DiscoverRecord;
use std::path::Path;
use tempfile::TempDir;

/// Effectively unlimited per-batch limit (`i32::MAX`), so only the probed limit is active.
const UNLIMITED: usize = i32::MAX as usize;

/// Error-list lengths cycled through the fixtures; includes an empty list.
const ERROR_LENS: [usize; 6] = [2, 3, 1, 4, 5, 0];

/// Deterministic fixture record `i`: non-sequential `obs_index`, a mix of `Some`
/// and `None` values, UUIDs of varied byte length, and error lists of varied
/// length (including an empty list).
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Fixture indices are tiny, exact in u32 and f32.
fn fixture_record(i: usize) -> DiscoverRecord {
    let errors = (0..ERROR_LENS[i % ERROR_LENS.len()])
        .map(|j| (i * 10 + j) as f32 * 0.125)
        .collect();
    DiscoverRecord::new(
        (i as u32 * 7919) % 10_000,
        format!("{}{i}", "h".repeat(i % 6)),
        i.is_multiple_of(2).then_some(i as f32 * 0.5),
        i as f32 * 0.25 - 1.0,
        errors,
    )
}

fn fixture(count: usize) -> Vec<DiscoverRecord> {
    (0..count).map(fixture_record).collect()
}

/// A path inside `dir` for a file the writer has not created yet.
fn fresh_path(dir: &TempDir, name: &str) -> String {
    let path = dir.path().join(name).to_string_lossy().into_owned();
    assert!(
        !Path::new(&path).exists(),
        "{path} must not exist before the writer runs"
    );
    path
}

/// Construct a writer with `capacity`, write `records` in one call and finish.
fn write_with_capacity(
    path: &str,
    max_uuid_bytes: usize,
    max_error_values: usize,
    capacity: usize,
    records: &[DiscoverRecord],
) -> anyhow::Result<()> {
    let mut writer = ParquetRecordWriter::new(path, max_uuid_bytes, max_error_values, capacity)?;
    writer.write_records(records)?;
    writer.finish()
}

/// As [`write_with_capacity`], with capacity equal to `records.len()`.
fn write_in_one_call(
    path: &str,
    max_uuid_bytes: usize,
    max_error_values: usize,
    records: &[DiscoverRecord],
) -> anyhow::Result<()> {
    write_with_capacity(
        path,
        max_uuid_bytes,
        max_error_values,
        records.len(),
        records,
    )
}

/// Read the file back and compare every field of every row against `expected`.
fn assert_round_trip(path: &str, expected: &[DiscoverRecord]) {
    assert!(Path::new(path).exists(), "{path} should exist after finish");
    let actual = read_all_records_from_parquet(path).expect("read back written Parquet file");
    assert_eq!(actual.len(), expected.len(), "row count mismatch");
    for (row, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(got.obs_index, want.obs_index, "row {row}: obs_index");
        assert_eq!(got.neuron_uuid, want.neuron_uuid, "row {row}: neuron_uuid");
        assert_eq!(got.value, want.value, "row {row}: value");
        assert_eq!(got.activation, want.activation, "row {row}: activation");
        assert_eq!(got.errors, want.errors, "row {row}: errors");
    }
}

// Fixture UUID byte lengths for `fixture(10)` sum to 31 bytes, so an 8-byte
// per-batch limit forces several chunks.
#[test]
fn uuid_byte_limit_chunking_round_trips_every_record() {
    let dir = TempDir::new().expect("temp dir");
    let path = fresh_path(&dir, "uuid_chunked.parquet");
    let records = fixture(10);

    write_in_one_call(&path, 8, UNLIMITED, &records).expect("chunked write should succeed");

    assert_round_trip(&path, &records);

    // A UUID byte limit above Arrow's `i32` offset ceiling would let one chunk
    // overflow the offsets (a panic, not an error), so it is rejected up front
    // and before the file is created. The ceiling itself (`UNLIMITED`, used
    // above) is the accepted sibling.
    #[cfg(target_pointer_width = "64")]
    {
        let rejected = fresh_path(&dir, "uuid_limit_over_ceiling.parquet");
        let err = ParquetRecordWriter::new(&rejected, UNLIMITED + 1, UNLIMITED, records.len())
            .err()
            .expect("a UUID byte limit above i32::MAX must be rejected");
        assert!(
            err.to_string().contains("byte limit per batch"),
            "unexpected error: {err}"
        );
        assert!(
            !Path::new(&rejected).exists(),
            "a rejected writer must not create the file"
        );
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn error_value_limit_above_arrow_offset_ceiling_is_rejected() {
    let dir = TempDir::new().expect("temp dir");
    let records = fixture(3);

    let rejected = fresh_path(&dir, "error_limit_over_ceiling.parquet");
    let err = ParquetRecordWriter::new(&rejected, UNLIMITED, UNLIMITED + 1, records.len())
        .err()
        .expect("an error value limit above i32::MAX must be rejected");
    assert!(
        err.to_string().contains("error value limit per batch"),
        "unexpected error: {err}"
    );
    assert!(
        !Path::new(&rejected).exists(),
        "a rejected writer must not create the file"
    );

    let accepted = fresh_path(&dir, "error_limit_at_ceiling.parquet");
    write_in_one_call(&accepted, UNLIMITED, UNLIMITED, &records)
        .expect("the same input must be accepted with the limit at i32::MAX");
    assert_round_trip(&accepted, &records);
}

#[test]
fn uuid_byte_limit_chunking_round_trips_across_write_calls() {
    let dir = TempDir::new().expect("temp dir");
    let path = fresh_path(&dir, "uuid_multi_call.parquet");
    let records = fixture(10);

    let mut writer = ParquetRecordWriter::new(&path, 8, UNLIMITED, records.len()).expect("writer");
    for slice in records.chunks(3) {
        writer
            .write_records(slice)
            .expect("each write_records call should succeed");
    }
    writer.finish().expect("finish should succeed");

    assert_round_trip(&path, &records);
}

#[test]
fn error_value_limit_chunking_round_trips_every_record() {
    let dir = TempDir::new().expect("temp dir");
    let path = fresh_path(&dir, "error_chunked.parquet");
    // Error lengths cycle 2,3,1,4,5,0; a limit of 5 forces several chunks.
    let records = fixture(12);

    write_in_one_call(&path, UNLIMITED, 5, &records).expect("chunked write should succeed");

    assert_round_trip(&path, &records);
}

#[test]
fn both_batch_limits_small_round_trips_every_record() {
    let dir = TempDir::new().expect("temp dir");
    let path = fresh_path(&dir, "both_limits.parquet");
    let records = fixture(15);

    write_in_one_call(&path, 8, 5, &records).expect("chunked write should succeed");

    assert_round_trip(&path, &records);
}

#[test]
fn exact_boundary_uuid_and_error_counts_are_accepted_and_round_trip() {
    let dir = TempDir::new().expect("temp dir");
    let path = fresh_path(&dir, "exact_boundary.parquet");
    let records = vec![
        DiscoverRecord::new(
            3,
            "abcdef".to_string(),
            Some(1.5),
            0.5,
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
        ),
        DiscoverRecord::new(9, "g".to_string(), None, -2.0, vec![]),
        DiscoverRecord::new(1, "hhhhhh".to_string(), Some(0.25), 0.75, vec![1.0, 2.0]),
    ];

    // The 6-byte UUIDs equal the UUID limit and the 5-value record equals the
    // error limit, so both must be accepted.
    write_in_one_call(&path, 6, 5, &records).expect("boundary-sized input should be accepted");

    assert_round_trip(&path, &records);
}

#[test]
fn uuid_longer_than_limit_is_rejected_and_accepted_once_limit_is_raised() {
    let dir = TempDir::new().expect("temp dir");
    let records = vec![
        DiscoverRecord::new(0, "ab".to_string(), Some(0.5), 0.1, vec![0.2]),
        DiscoverRecord::new(1, "abcdefg".to_string(), None, 0.3, vec![]),
    ];

    let rejected = fresh_path(&dir, "uuid_over_limit.parquet");
    let err = write_in_one_call(&rejected, 6, UNLIMITED, &records)
        .expect_err("a 7-byte UUID must be rejected under a 6-byte limit");
    assert!(
        err.to_string().contains("exceeds configured limit"),
        "unexpected error: {err}"
    );

    let accepted = fresh_path(&dir, "uuid_at_limit.parquet");
    write_in_one_call(&accepted, 7, UNLIMITED, &records)
        .expect("the same input must be accepted once the UUID limit is 7");
    assert_round_trip(&accepted, &records);
}

#[test]
fn error_list_longer_than_limit_is_rejected_and_accepted_once_limit_is_raised() {
    let dir = TempDir::new().expect("temp dir");
    let records = vec![
        DiscoverRecord::new(0, "ab".to_string(), Some(0.5), 0.1, vec![0.0, 0.5]),
        DiscoverRecord::new(
            1,
            "cd".to_string(),
            None,
            0.3,
            vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        ),
    ];

    let rejected = fresh_path(&dir, "errors_over_limit.parquet");
    let err = write_in_one_call(&rejected, UNLIMITED, 5, &records)
        .expect_err("six error values must be rejected under a limit of 5");
    assert!(
        err.to_string().contains("error value(s)"),
        "unexpected error: {err}"
    );

    let accepted = fresh_path(&dir, "errors_at_limit.parquet");
    write_in_one_call(&accepted, UNLIMITED, 6, &records)
        .expect("the same input must be accepted once the error limit is 6");
    assert_round_trip(&accepted, &records);
}

#[test]
fn zero_uuid_byte_limit_is_rejected_and_accepted_at_one() {
    let dir = TempDir::new().expect("temp dir");
    let records = vec![DiscoverRecord::new(
        0,
        "a".to_string(),
        Some(0.5),
        0.5,
        vec![0.5],
    )];

    let rejected = fresh_path(&dir, "zero_uuid_limit.parquet");
    let err = write_in_one_call(&rejected, 0, UNLIMITED, &records)
        .expect_err("a zero UUID byte limit must be rejected");
    assert!(
        err.to_string()
            .contains("Configured neuron UUID byte limit per batch must be greater than zero"),
        "unexpected error: {err}"
    );

    let accepted = fresh_path(&dir, "uuid_limit_one.parquet");
    write_in_one_call(&accepted, 1, UNLIMITED, &records)
        .expect("the same input must be accepted with a UUID limit of 1");
    assert_round_trip(&accepted, &records);
}

#[test]
fn zero_error_value_limit_is_rejected_and_accepted_at_one() {
    let dir = TempDir::new().expect("temp dir");
    let records = vec![
        DiscoverRecord::new(0, "ab".to_string(), Some(0.5), 0.5, vec![]),
        DiscoverRecord::new(1, "cd".to_string(), None, 0.25, vec![0.75]),
    ];

    let rejected = fresh_path(&dir, "zero_error_limit.parquet");
    let err = write_in_one_call(&rejected, UNLIMITED, 0, &records)
        .expect_err("a zero error value limit must be rejected");
    assert!(
        err.to_string()
            .contains("Configured error value limit per batch must be greater than zero"),
        "unexpected error: {err}"
    );

    let accepted = fresh_path(&dir, "error_limit_one.parquet");
    write_in_one_call(&accepted, UNLIMITED, 1, &records)
        .expect("the same input must be accepted with an error limit of 1");
    assert_round_trip(&accepted, &records);
}

#[test]
fn zero_capacity_is_rejected_before_file_creation_and_accepted_at_record_count() {
    let dir = TempDir::new().expect("temp dir");
    let records = fixture(1);

    let rejected = fresh_path(&dir, "zero_capacity.parquet");
    let err = write_with_capacity(&rejected, UNLIMITED, UNLIMITED, 0, &records)
        .expect_err("a zero capacity must be rejected");
    assert!(
        err.to_string().contains("No records to write"),
        "unexpected error: {err}"
    );
    assert!(
        !Path::new(&rejected).exists(),
        "a rejected writer must not create the file"
    );

    let accepted = fresh_path(&dir, "capacity_of_one.parquet");
    write_with_capacity(&accepted, UNLIMITED, UNLIMITED, records.len(), &records)
        .expect("the same input must be accepted with capacity equal to the record count");
    assert_round_trip(&accepted, &records);
}

#[test]
fn writing_more_records_than_capacity_is_rejected_and_accepted_at_capacity() {
    let dir = TempDir::new().expect("temp dir");
    let records = fixture(3);

    let rejected = fresh_path(&dir, "over_capacity.parquet");
    let mut writer = ParquetRecordWriter::new(&rejected, UNLIMITED, UNLIMITED, 2)
        .expect("writer with capacity 2 should construct");
    let err = writer
        .write_records(&records)
        .expect_err("writing 3 records into capacity 2 must be rejected");
    assert!(
        err.to_string().contains("exceeding remaining capacity"),
        "unexpected error: {err}"
    );

    let accepted = fresh_path(&dir, "at_capacity.parquet");
    write_with_capacity(&accepted, UNLIMITED, UNLIMITED, 3, &records)
        .expect("the same input must be accepted with capacity 3");
    assert_round_trip(&accepted, &records);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn capacity_above_arrow_row_limit_is_rejected_before_file_creation() {
    let dir = TempDir::new().expect("temp dir");
    let records = fixture(3);

    let rejected = fresh_path(&dir, "over_row_limit.parquet");
    let err = ParquetRecordWriter::new(&rejected, UNLIMITED, UNLIMITED, i32::MAX as usize + 1)
        .err()
        .expect("a capacity above i32::MAX must be rejected");
    assert!(
        err.to_string()
            .contains("exceeds maximum supported row count"),
        "unexpected error: {err}"
    );
    assert!(
        !Path::new(&rejected).exists(),
        "a rejected writer must not create the file"
    );

    let accepted = fresh_path(&dir, "at_row_limit.parquet");
    write_with_capacity(&accepted, UNLIMITED, UNLIMITED, UNLIMITED, &records)
        .expect("the same input must be accepted with capacity i32::MAX");
    assert_round_trip(&accepted, &records);
}
