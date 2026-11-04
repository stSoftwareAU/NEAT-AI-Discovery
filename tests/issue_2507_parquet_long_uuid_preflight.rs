//! Footer preflight must charge the decoded `neuron_uuid` strings of a
//! dictionary-encoded column. Arrow copies the value into every row of a batch,
//! so a tiny file with one long repeated identifier can decode to tens of
//! megabytes (Issue #2507).

use arrow::array::{ArrayRef, Float32Array, ListArray, RecordBatch, StringArray, UInt32Array};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field};
use neat_ai_discovery::analysis::streaming::StreamingRecordCache;
use neat_ai_discovery::ffi_types::{DiscoveryErrorKind, classify_anyhow_error};
use neat_ai_discovery::parquet_format::footer::read_footer_stats;
use neat_ai_discovery::parquet_format::{
    ColumnProfile, merge_parquet_files, read_all_records_grouped_by_neuron_bounded,
    read_mean_abs_activation_by_neuron, read_records_from_parquet_with_profile,
};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serial_test::serial;
use std::collections::HashSet;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

/// Environment variable that sets the decode ceiling for readers without a caller budget.
const DECODE_LIMIT_ENV: &str = "NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB";

/// Rows per fixture. Every row repeats the same identifier, so the file is tiny.
const ROWS: usize = 2_000;

/// Identifier length. `ROWS * LONG_IDENTIFIER_BYTES` (40 MB) is decoded, but the
/// file stores the value once in its dictionary.
const LONG_IDENTIFIER_BYTES: usize = 20_000;

/// Decoded identifier bytes of the long-identifier fixture.
const DECODED_IDENTIFIER_BYTES: u64 = (ROWS * LONG_IDENTIFIER_BYTES) as u64;

#[cfg(target_os = "linux")]
fn process_high_water_kib() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").expect("read process status");
    let line = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .expect("VmHWM in process status");
    line.split_whitespace()
        .nth(1)
        .expect("VmHWM value")
        .parse()
        .expect("numeric VmHWM")
}

/// Writes the five-column discovery schema as one row group with dictionary
/// encoding enabled, one row per identifier.
fn write_fixture(path: &std::path::Path, identifiers: &[&str]) {
    let schema = Arc::new(neat_ai_discovery::parquet_format::create_schema());
    let rows = identifiers.len();
    let errors = Arc::new(
        ListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            OffsetBuffer::<i32>::from_lengths(vec![0; rows]),
            Arc::new(Float32Array::from(Vec::<f32>::new())),
            None,
        )
        .expect("empty errors list"),
    );
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt32Array::from_iter_values(
            0..u32::try_from(rows).expect("row count fits u32"),
        )),
        Arc::new(StringArray::from_iter_values(identifiers.iter().copied())),
        Arc::new(Float32Array::from(vec![Some(0.5_f32); rows])),
        Arc::new(Float32Array::from(vec![0.7_f32; rows])),
        errors,
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).expect("discovery batch");
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(true)
        .build();
    let file = std::fs::File::create(path).expect("create fixture parquet");
    let mut writer = ArrowWriter::try_new(file, schema, Some(properties)).expect("fixture writer");
    writer.write(&batch).expect("write fixture batch");
    writer.close().expect("close fixture parquet");
}

/// Writes the long-identifier fixture and returns its path and identifier. The
/// file is checked to be small on disk, so only its decoded size is large.
fn long_identifier_fixture(dir: &TempDir) -> (PathBuf, String) {
    let path = dir.path().join("long-identifier.parquet");
    let long = "n".repeat(LONG_IDENTIFIER_BYTES);
    let identifiers = vec![long.as_str(); ROWS];
    write_fixture(&path, &identifiers);
    let file_bytes = std::fs::metadata(&path).expect("fixture metadata").len();
    assert!(
        file_bytes < 1024 * 1024,
        "the fixture must be small on disk, got {file_bytes} bytes"
    );
    (path, long)
}

/// Sets `NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB` and restores the prior value on drop.
/// Tests that use it are `#[serial]`.
struct DecodeLimitGuard {
    prior: Option<OsString>,
}

impl DecodeLimitGuard {
    fn set(value: &str) -> Self {
        let prior = std::env::var_os(DECODE_LIMIT_ENV);
        set_decode_limit(value);
        Self { prior }
    }
}

impl Drop for DecodeLimitGuard {
    fn drop(&mut self) {
        // SAFETY: the owning test is `#[serial]`, so no other test reads or writes
        // the environment concurrently.
        unsafe {
            match &self.prior {
                Some(value) => std::env::set_var(DECODE_LIMIT_ENV, value),
                None => std::env::remove_var(DECODE_LIMIT_ENV),
            }
        }
    }
}

fn set_decode_limit(value: &str) {
    // SAFETY: callers are `#[serial]` tests, so no other test reads or writes
    // the environment concurrently.
    unsafe { std::env::set_var(DECODE_LIMIT_ENV, value) };
}

fn assert_read_without_large_allocation<T, E>(
    read: impl FnOnce() -> Result<T, E>,
    verify: impl FnOnce(Result<T, E>),
    path: &str,
) {
    #[cfg(target_os = "linux")]
    {
        std::fs::write("/proc/self/clear_refs", "5").expect("reset process high-water mark");
        let before = process_high_water_kib();
        verify(read());
        let increase = process_high_water_kib().saturating_sub(before);
        assert!(
            increase < 8 * 1024,
            "{path} high-water mark grew by {increase} KiB during the read; Arrow appears to have materialised the identifiers"
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        verify(read());
    }
}

macro_rules! assert_memory_exhausted {
    ($result:expr, $path:expr) => {{
        let error = $result
            .err()
            .expect("oversized decode must be refused before batch construction");
        assert_eq!(
            classify_anyhow_error(&error),
            DiscoveryErrorKind::MemoryExhausted,
            "{} must reject through the decode budget: {error:#}",
            $path
        );
        assert!(
            format!("{error:#}").contains("footer preflight before Arrow batch construction"),
            "{} must refuse during preflight: {error:#}",
            $path
        );
    }};
}

#[test]
fn footer_statistic_carries_the_decoded_identifier_heap() {
    let dir = TempDir::new().expect("temporary directory");
    let (path, _long) = long_identifier_fixture(&dir);
    let stats = read_footer_stats(path.to_str().expect("utf-8 path")).expect("fixture footer");
    assert_eq!(
        stats.uuid_heap_bytes, DECODED_IDENTIFIER_BYTES,
        "the footer size statistic must carry the decoded identifier bytes"
    );
    assert!(
        stats.projected_in_memory_bytes() >= DECODED_IDENTIFIER_BYTES,
        "the pre-load projection must include the decoded identifier bytes"
    );
}

#[test]
#[serial]
fn eager_grouped_read_refuses_long_identifiers_before_arrow_batches() {
    let dir = TempDir::new().expect("temporary directory");
    let (path, _long) = long_identifier_fixture(&dir);
    let path_str = path.to_str().expect("utf-8 path");
    let _limit = DecodeLimitGuard::set("1");

    assert_read_without_large_allocation(
        || read_all_records_grouped_by_neuron_bounded(path_str, None, ColumnProfile::Full, Some(1)),
        |result| assert_memory_exhausted!(result, "eager grouped read"),
        "eager grouped read",
    );
}

#[test]
#[serial]
fn lazy_neuron_read_refuses_long_identifiers_before_arrow_batches() {
    let dir = TempDir::new().expect("temporary directory");
    let (path, long) = long_identifier_fixture(&dir);
    let path_str = path.to_str().expect("utf-8 path");
    let _limit = DecodeLimitGuard::set("1");

    assert_read_without_large_allocation(
        || read_records_from_parquet_with_profile(path_str, &long, ColumnProfile::Full),
        |result| assert_memory_exhausted!(result, "lazy neuron read"),
        "lazy neuron read",
    );
}

#[test]
#[serial]
fn activation_summary_refuses_long_identifiers_before_arrow_batches() {
    let dir = TempDir::new().expect("temporary directory");
    let (path, long) = long_identifier_fixture(&dir);
    let path_str = path.to_str().expect("utf-8 path");
    let _limit = DecodeLimitGuard::set("1");

    let wanted = HashSet::from([long.as_str()]);
    assert_read_without_large_allocation(
        || read_mean_abs_activation_by_neuron(path_str, &wanted, None),
        |result| assert_memory_exhausted!(result, "activation summary"),
        "activation summary",
    );
}

#[test]
#[serial]
fn streaming_index_refuses_long_identifiers_before_arrow_batches() {
    let dir = TempDir::new().expect("temporary directory");
    let (path, _long) = long_identifier_fixture(&dir);
    let path_str = path.to_str().expect("utf-8 path");
    let _limit = DecodeLimitGuard::set("1");

    assert_read_without_large_allocation(
        || StreamingRecordCache::new(path_str, Some(1), Some(0)),
        |result| assert_memory_exhausted!(result, "streaming UUID index"),
        "streaming UUID index",
    );
}

#[test]
#[serial]
fn streaming_block_load_refuses_long_identifiers_before_arrow_batches() {
    let dir = TempDir::new().expect("temporary directory");
    let (path, long) = long_identifier_fixture(&dir);
    let path_str = path.to_str().expect("utf-8 path");
    let _limit = DecodeLimitGuard::set("64");

    // The index must read the identifier heap once, so it builds under a 64 MB
    // ceiling. The block load is then refused under the 1 MB ceiling.
    let cache = StreamingRecordCache::new(path_str, Some(1), Some(0))
        .expect("the index builds under a 64 MB ceiling");
    set_decode_limit("1");
    assert_read_without_large_allocation(
        || cache.get(&long),
        |result| assert_memory_exhausted!(result, "streaming block read"),
        "streaming block read",
    );
}

#[test]
#[serial]
fn merge_refuses_long_identifiers_before_arrow_batches() {
    let dir = TempDir::new().expect("temporary directory");
    let (path, _long) = long_identifier_fixture(&dir);
    let path_str = path.to_str().expect("utf-8 path");
    let output = dir.path().join("merged.parquet");
    let _limit = DecodeLimitGuard::set("1");

    assert_read_without_large_allocation(
        || {
            merge_parquet_files(
                output.to_str().expect("output path"),
                &[path_str.to_owned()],
            )
        },
        |result| assert_memory_exhausted!(result, "merge input read"),
        "merge input read",
    );
    assert!(
        !output.exists(),
        "a refused merge must not create the destination"
    );
}

#[test]
#[serial]
fn short_repeated_identifier_of_the_same_shape_is_accepted_under_the_same_budget() {
    let dir = TempDir::new().expect("temporary directory");
    let path = dir.path().join("short-identifier.parquet");
    let short = "n".repeat(36);
    let identifiers = vec![short.as_str(); ROWS];
    write_fixture(&path, &identifiers);
    let path_str = path.to_str().expect("utf-8 path");

    let _limit = DecodeLimitGuard::set("1");

    let records =
        read_all_records_grouped_by_neuron_bounded(path_str, None, ColumnProfile::Full, Some(1))
            .expect("36-character identifiers fit the 1 MB ceiling in the eager reader");
    assert_eq!(records[&short].len(), ROWS);

    let lazy = read_records_from_parquet_with_profile(path_str, &short, ColumnProfile::Full)
        .expect("36-character identifiers fit the 1 MB ceiling in the lazy reader");
    assert_eq!(lazy.len(), ROWS);

    let wanted = HashSet::from([short.as_str()]);
    let summary = read_mean_abs_activation_by_neuron(path_str, &wanted, None)
        .expect("36-character identifiers fit the 1 MB ceiling in the activation summary");
    assert_eq!(summary.len(), 1);

    let cache = StreamingRecordCache::new(path_str, Some(1), Some(0))
        .expect("36-character identifiers fit the 1 MB ceiling in the streaming index");
    let block = cache
        .get(&short)
        .expect("36-character identifiers fit the 1 MB ceiling in the block load");
    assert_eq!(block.len(), ROWS);

    let output = dir.path().join("merged.parquet");
    merge_parquet_files(
        output.to_str().expect("output path"),
        &[path_str.to_owned()],
    )
    .expect("36-character identifiers fit the 1 MB ceiling in the merge");
    let merged = read_footer_stats(output.to_str().expect("output path")).expect("merged footer");
    assert_eq!(merged.num_rows, u64::try_from(ROWS).unwrap());
}

#[test]
#[serial]
fn distinct_36_character_identifiers_still_read_under_the_same_budget() {
    let dir = TempDir::new().expect("temporary directory");
    let path = dir.path().join("distinct-identifiers.parquet");
    let identifiers: Vec<String> = (0..ROWS).map(|index| format!("{index:036}")).collect();
    let refs: Vec<&str> = identifiers.iter().map(String::as_str).collect();
    write_fixture(&path, &refs);
    let path_str = path.to_str().expect("utf-8 path");
    assert!(identifiers.iter().all(|id| id.len() == 36));

    let _limit = DecodeLimitGuard::set("1");

    let records =
        read_all_records_grouped_by_neuron_bounded(path_str, None, ColumnProfile::Full, Some(1))
            .expect("distinct 36-character identifiers fit the 1 MB ceiling");
    assert_eq!(records.len(), ROWS);

    let cache = StreamingRecordCache::new(path_str, Some(1), Some(0))
        .expect("distinct identifiers build the streaming index");
    let block = cache
        .get(&identifiers[7])
        .expect("one distinct identifier loads its block");
    assert_eq!(block.len(), 1);
}
