//! Footer preflight must reject hostile decoded sizes before Arrow allocates
//! their list values (Issue #2446).

use arrow::array::{
    ArrayRef, Float32Array, Int64Array, ListArray, RecordBatch, StringArray, UInt32Array,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use neat_ai_discovery::analysis::streaming::StreamingRecordCache;
use neat_ai_discovery::ffi_types::{DiscoveryErrorKind, classify_anyhow_error};
use neat_ai_discovery::parquet_format::{
    ColumnProfile, merge_parquet_files, read_all_records_grouped_by_neuron_bounded,
    read_all_records_grouped_by_neuron_with_profile, read_mean_abs_activation_by_neuron,
    read_records_from_parquet_with_limit_and_budget, read_records_from_parquet_with_profile,
    write_records_to_parquet,
};
use neat_ai_discovery::types::DiscoverRecord;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serial_test::serial;
use std::collections::HashSet;
use std::sync::Arc;
use tempfile::TempDir;

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

fn hostile_file(path: &std::path::Path, error_count: usize) {
    hostile_file_with_rows(path, &[(0, "neuron-a", error_count)]);
}

fn hostile_two_row_group_file(path: &std::path::Path, error_count: usize) {
    hostile_file_with_rows(path, &[(0, "neuron-a", 0), (1, "neuron-b", error_count)]);
}

fn hostile_file_with_rows(path: &std::path::Path, rows: &[(u32, &str, usize)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("obs_index", DataType::UInt32, false),
        Field::new("neuron_uuid", DataType::Utf8, false),
        Field::new("value", DataType::Float32, true),
        Field::new("activation", DataType::Float32, false),
        Field::new(
            "errors",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
    ]));
    let file = std::fs::File::create(path).expect("create hostile parquet");
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(true)
        .build();
    let mut writer =
        ArrowWriter::try_new(file, Arc::clone(&schema), Some(properties)).expect("writer");
    for &(obs, neuron, count) in rows {
        let errors = Arc::new(
            ListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                OffsetBuffer::<i32>::from_lengths([count]),
                Arc::new(Float32Array::from(vec![0.0_f32; count])),
                None,
            )
            .expect("errors list"),
        );
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(vec![obs])),
            Arc::new(StringArray::from(vec![neuron])),
            Arc::new(Float32Array::from(vec![Some(0.5_f32)])),
            Arc::new(Float32Array::from(vec![0.7_f32])),
            errors,
        ];
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).expect("hostile batch");
        writer.write(&batch).expect("write hostile batch");
        writer.flush().expect("finish row group");
    }
    writer.close().expect("close hostile parquet");
    if let Ok(seed_path) = std::env::var("ISSUE_2446_FUZZ_SEED_PATH") {
        std::fs::copy(path, seed_path).expect("copy Parquet fuzz seed");
    }
}

fn malformed_hostile_file(path: &std::path::Path, error_count: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new(
            "errors",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
    ]));
    let file = std::fs::File::create(path).expect("create malformed Parquet");
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(true)
        .build();
    let mut writer =
        ArrowWriter::try_new(file, Arc::clone(&schema), Some(properties)).expect("writer");
    let errors = Arc::new(
        ListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            OffsetBuffer::<i32>::from_lengths([error_count]),
            Arc::new(Float32Array::from(vec![0.0_f32; error_count])),
            None,
        )
        .expect("errors list"),
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(UInt32Array::from(vec![0])), errors],
    )
    .expect("malformed batch");
    writer.write(&batch).expect("write malformed batch");
    writer.close().expect("close malformed Parquet");
}

fn write_uuid_schema_file(
    path: &std::path::Path,
    uuid_type: DataType,
    uuid_nullable: bool,
    uuid_values: ArrayRef,
) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "neuron_uuid",
        uuid_type,
        uuid_nullable,
    )]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![uuid_values]).expect("UUID schema batch");
    let file = std::fs::File::create(path).expect("create UUID schema Parquet");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("UUID schema writer");
    writer.write(&batch).expect("write UUID schema batch");
    writer.close().expect("close UUID schema Parquet");
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
#[serial]
fn oversized_error_list_is_rejected_before_arrow_materialises_it() {
    let dir = TempDir::new().expect("temporary directory");
    let hostile = dir.path().join("hostile.parquet");
    let legal = dir.path().join("legal.parquet");

    // Create the large fixture before tracking the reader's allocations.
    hostile_file(&hostile, 20_000_000);
    #[cfg(target_os = "linux")]
    {
        std::fs::write("/proc/self/clear_refs", "5").expect("reset process high-water mark");
    }
    #[cfg(target_os = "linux")]
    let high_water_before = process_high_water_kib();
    let result = read_all_records_grouped_by_neuron_bounded(
        hostile.to_str().expect("hostile path"),
        None,
        ColumnProfile::Full,
        Some(1),
    );
    #[cfg(target_os = "linux")]
    let allocated_kib = process_high_water_kib().saturating_sub(high_water_before);

    let error = result.expect_err("20 million errors exceed the 1 MB ceiling");
    assert_eq!(
        classify_anyhow_error(&error),
        DiscoveryErrorKind::MemoryExhausted,
        "the refusal must be the decode-budget preflight: {error:#}"
    );
    #[cfg(target_os = "linux")]
    assert!(
        allocated_kib < 8 * 1024,
        "reader high-water mark grew by {allocated_kib} KiB; Arrow appears to have materialised the list: {error:#}"
    );
    assert!(
        format!("{error:#}").contains("footer preflight before Arrow batch construction"),
        "the error must come from footer preflight before record charging: {error:#}"
    );

    let near_budget = (0..1_000)
        .map(|index| DiscoverRecord::new(index, "neuron-a".into(), Some(0.5), 0.7, vec![0.1; 100]))
        .collect::<Vec<_>>();
    write_records_to_parquet(legal.to_str().expect("legal path"), &near_budget)
        .expect("write legal near-budget file");
    let records = read_all_records_grouped_by_neuron_bounded(
        legal.to_str().expect("legal path"),
        None,
        ColumnProfile::Full,
        Some(1),
    )
    .expect("legal file near the 1 MB budget remains readable");
    assert_eq!(records["neuron-a"].len(), 1_000);
    assert_eq!(records["neuron-a"][999].errors.len(), 100);
}

#[test]
#[serial]
fn streaming_index_rejects_malformed_schema_before_reading_unprojected_errors() {
    let dir = TempDir::new().expect("temporary directory");
    let malformed = dir.path().join("malformed-hostile.parquet");
    malformed_hostile_file(&malformed, 20_000_000);

    #[cfg(target_os = "linux")]
    {
        std::fs::write("/proc/self/clear_refs", "5").expect("reset process high-water mark");
    }
    #[cfg(target_os = "linux")]
    let high_water_before = process_high_water_kib();

    let result = StreamingRecordCache::new(
        malformed.to_str().expect("malformed path"),
        Some(1),
        Some(0),
    );
    let error = match result {
        Ok(_) => panic!("missing neuron_uuid must be rejected from the schema"),
        Err(error) => error,
    };

    assert!(
        format!("{error:#}").contains("missing 'neuron_uuid' column"),
        "schema failure should identify the missing index column: {error:#}"
    );
    #[cfg(target_os = "linux")]
    {
        let allocated_kib = process_high_water_kib().saturating_sub(high_water_before);
        assert!(
            allocated_kib < 8 * 1024,
            "malformed schema read grew the high-water mark by {allocated_kib} KiB"
        );
    }
}

#[test]
fn streaming_index_rejects_nullable_uuid_column() {
    let dir = TempDir::new().expect("temporary directory");
    assert_streaming_index_rejects_uuid_schema(
        &dir,
        "nullable",
        DataType::Utf8,
        true,
        Arc::new(StringArray::from(vec!["neuron-a"])),
    );
}

#[test]
fn streaming_index_rejects_non_utf8_uuid_column() {
    let dir = TempDir::new().expect("temporary directory");
    assert_streaming_index_rejects_uuid_schema(
        &dir,
        "wrong-type",
        DataType::Int64,
        false,
        Arc::new(Int64Array::from(vec![1])),
    );
}

fn assert_streaming_index_rejects_uuid_schema(
    dir: &TempDir,
    name: &str,
    uuid_type: DataType,
    uuid_nullable: bool,
    values: ArrayRef,
) {
    let path = dir.path().join(format!("{name}.parquet"));
    write_uuid_schema_file(&path, uuid_type, uuid_nullable, values);
    let result =
        StreamingRecordCache::new(path.to_str().expect("UUID schema path"), Some(1), Some(0));
    let error = match result {
        Ok(_) => panic!("{name} neuron_uuid column must be refused before reading"),
        Err(error) => error,
    };
    assert!(
        format!("{error:#}").contains("must be non-null Utf8"),
        "{name} UUID schema error should identify the required type: {error:#}"
    );
}

#[test]
#[serial]
fn every_production_reader_preflights_before_filtering_and_respects_projection() {
    let dir = TempDir::new().expect("temporary directory");
    let hostile = dir.path().join("hostile.parquet");
    let activation_oversized = dir.path().join("activation-oversized.parquet");
    let output = dir.path().join("merged.parquet");
    hostile_two_row_group_file(&hostile, 20_000_000);
    let metadata = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
        std::fs::File::open(&hostile).expect("open multi-row-group fixture"),
    )
    .expect("read multi-row-group footer");
    assert_eq!(metadata.metadata().row_groups().len(), 2);
    let many_records = (0..12_000)
        .map(|index| {
            DiscoverRecord::new(
                index,
                format!("neuron-{index:032}"),
                Some(0.5),
                0.7,
                Vec::new(),
            )
        })
        .collect::<Vec<_>>();
    write_records_to_parquet(
        activation_oversized.to_str().expect("activation path"),
        &many_records,
    )
    .expect("write oversized activation-only selection");
    std::fs::write(&output, b"existing merge destination").expect("seed merge destination");
    let hostile_path = hostile.to_str().expect("hostile path");
    let prior_limit = std::env::var_os("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB");
    // SAFETY: tests that mutate this environment variable use `#[serial]`.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB", "1") };

    assert_read_without_large_allocation(
        || {
            read_records_from_parquet_with_profile(
                hostile_path,
                "another-neuron",
                ColumnProfile::Full,
            )
        },
        |result| assert_memory_exhausted!(result, "UUID-filtered read of an unrelated neuron"),
        "UUID-filtered read",
    );

    assert_read_without_large_allocation(
        || {
            read_records_from_parquet_with_limit_and_budget(
                hostile_path,
                Some(1),
                ColumnProfile::Full,
                Some(1),
            )
        },
        |result| assert_memory_exhausted!(result, "observation-limited read"),
        "observation-limited read",
    );

    let wanted = HashSet::from(["another-neuron"]);
    let summary = read_mean_abs_activation_by_neuron(hostile_path, &wanted, None);
    assert!(
        summary
            .expect("activation summary projects away the oversized errors list")
            .is_empty()
    );
    let wanted = HashSet::from(["neuron-00000000000000000000000000000000"]);
    assert_read_without_large_allocation(
        || {
            read_mean_abs_activation_by_neuron(
                activation_oversized.to_str().expect("activation path"),
                &wanted,
                None,
            )
        },
        |result| assert_memory_exhausted!(result, "activation summary's selected columns"),
        "activation summary",
    );

    assert_read_without_large_allocation(
        || {
            StreamingRecordCache::new(
                activation_oversized.to_str().expect("activation path"),
                Some(1),
                Some(0),
            )
        },
        |result| assert_memory_exhausted!(result, "streaming UUID index"),
        "streaming UUID index",
    );

    let cache = StreamingRecordCache::new(hostile_path, Some(1), Some(0)).expect("build index");
    assert_read_without_large_allocation(
        || cache.get("neuron-a"),
        |result| assert_memory_exhausted!(result, "streaming block read"),
        "streaming block read",
    );

    let projected =
        read_all_records_grouped_by_neuron_with_profile(hostile_path, ColumnProfile::WithoutErrors)
            .expect("projection that skips errors must not charge the skipped list");
    assert!(projected["neuron-a"][0].errors.is_empty());

    assert_read_without_large_allocation(
        || {
            merge_parquet_files(
                output.to_str().expect("output path"),
                &[hostile_path.to_owned()],
            )
        },
        |result| assert_memory_exhausted!(result, "merge input read"),
        "merge input read",
    );
    assert_eq!(
        std::fs::read(&output).expect("destination remains readable"),
        b"existing merge destination",
        "preflight failure must preserve the existing merge destination"
    );
    if let Some(value) = prior_limit {
        // SAFETY: this test is serialised with the other environment mutator.
        unsafe { std::env::set_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB", value) };
    } else {
        // SAFETY: this test is serialised with the other environment mutator.
        unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB") };
    }
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
            "{path} high-water mark grew by {increase} KiB during the read"
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        verify(read());
    }
}

#[test]
#[serial]
fn combined_row_groups_exceed_budget_even_when_each_group_fits() {
    let dir = TempDir::new().unwrap();
    for (name, rows) in [
        ("first", vec![(0, "neuron-a", 150_000)]),
        ("second", vec![(1, "neuron-b", 150_000)]),
        (
            "combined",
            vec![(0, "neuron-a", 150_000), (1, "neuron-b", 150_000)],
        ),
    ] {
        let path = dir.path().join(format!("{name}.parquet"));
        hostile_file_with_rows(&path, &rows);
        let read = || {
            read_all_records_grouped_by_neuron_bounded(
                path.to_str().unwrap(),
                None,
                ColumnProfile::Full,
                Some(1),
            )
        };
        if name == "combined" {
            assert_read_without_large_allocation(
                read,
                |result| assert_memory_exhausted!(result, "combined row groups"),
                "combined row groups",
            );
        } else {
            assert_eq!(read().expect("each individual group fits 1 MB").len(), 1);
        }
    }
}

#[test]
#[serial]
fn streaming_projects_away_oversized_extra_columns() {
    extra_column_projection(false);
}

#[test]
#[serial]
fn merge_projects_away_oversized_extra_columns() {
    extra_column_projection(true);
}

fn extra_column_projection(merge: bool) {
    let dir = TempDir::new().unwrap();
    let source = dir.path().join("extra.parquet");
    let output = dir.path().join("merged.parquet");
    let mut fields = neat_ai_discovery::parquet_format::create_schema()
        .fields()
        .to_vec();
    fields.push(Arc::new(Field::new(
        "extra",
        DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
        false,
    )));
    let schema = Arc::new(Schema::new(fields));
    let list = |count| {
        Arc::new(
            ListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                OffsetBuffer::<i32>::from_lengths([count]),
                Arc::new(Float32Array::from(vec![0.0; count])),
                None,
            )
            .unwrap(),
        ) as ArrayRef
    };
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(UInt32Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["neuron-a"])),
            Arc::new(Float32Array::from(vec![Some(0.5)])),
            Arc::new(Float32Array::from(vec![0.7])),
            list(1),
            list(20_000_000),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(true)
        .build();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&source).unwrap(), schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    drop(batch);
    let prior_limit = std::env::var_os("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB");
    // SAFETY: environment-mutating tests in this suite are serialised.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB", "1") };
    let source_path = source.to_str().unwrap();
    if !merge {
        let cache = StreamingRecordCache::new(source_path, Some(1), Some(0)).unwrap();
        assert_read_without_large_allocation(
            || cache.get("neuron-a"),
            |result| {
                let records = result.expect("streaming must skip extra columns");
                assert_eq!(records.len(), 1);
                assert_eq!(records[0].errors, vec![0.0]);
            },
            "streaming extra column",
        );
    } else {
        assert_read_without_large_allocation(
            || merge_parquet_files(output.to_str().unwrap(), &[source_path.to_owned()]),
            |result| result.expect("merge must skip extra columns"),
            "merge extra column",
        );
        let records = read_all_records_grouped_by_neuron_bounded(
            output.to_str().unwrap(),
            None,
            ColumnProfile::Full,
            Some(1),
        )
        .unwrap();
        assert_eq!(records["neuron-a"][0].errors, vec![0.0]);
    }
    // SAFETY: environment-mutating tests in this suite are serialised.
    unsafe {
        if let Some(value) = prior_limit {
            std::env::set_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB", value);
        } else {
            std::env::remove_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB");
        }
    }
}
