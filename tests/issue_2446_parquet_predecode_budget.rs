//! Footer preflight must reject hostile decoded sizes before Arrow allocates
//! their list values (Issue #2446).

use arrow::array::{ArrayRef, Float32Array, ListArray, RecordBatch, StringArray, UInt32Array};
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
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(properties))
        .expect("writer");
    for (obs, neuron, count) in [(0_u32, "neuron-a", 0), (1, "neuron-b", error_count)] {
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
    }
    writer.close().expect("close hostile parquet");
    if let Ok(seed_path) = std::env::var("ISSUE_2446_FUZZ_SEED_PATH") {
        std::fs::copy(path, seed_path).expect("copy Parquet fuzz seed");
    }
}

#[cfg(target_os = "linux")]
#[test]
fn oversized_error_list_is_rejected_before_arrow_materialises_it() {
    let dir = TempDir::new().expect("temporary directory");
    let hostile = dir.path().join("hostile.parquet");
    let legal = dir.path().join("legal.parquet");

    // Create the large fixture before tracking the reader's allocations.
    hostile_file(&hostile, 20_000_000);
    write_records_to_parquet(
        legal.to_str().expect("legal path"),
        &[DiscoverRecord::new(
            0,
            "neuron-a".into(),
            Some(0.5),
            0.7,
            vec![0.1],
        )],
    )
    .expect("write legal file");

    std::fs::write("/proc/self/clear_refs", "5").expect("reset process high-water mark");
    let high_water_before = process_high_water_kib();
    let result = read_all_records_grouped_by_neuron_bounded(
        hostile.to_str().expect("hostile path"),
        None,
        ColumnProfile::Full,
        Some(1),
    );
    let allocated_kib = process_high_water_kib().saturating_sub(high_water_before);

    let error = result.expect_err("20 million errors exceed the 1 MB ceiling");
    assert_eq!(
        classify_anyhow_error(&error),
        DiscoveryErrorKind::MemoryExhausted,
        "the refusal must be the decode-budget preflight: {error:#}"
    );
    assert!(
        allocated_kib < 8 * 1024,
        "reader high-water mark grew by {allocated_kib} KiB; Arrow appears to have materialised the list"
    );

    let records = read_all_records_grouped_by_neuron_bounded(
        legal.to_str().expect("legal path"),
        None,
        ColumnProfile::Full,
        Some(1),
    )
    .expect("legal file below the budget remains readable");
    assert_eq!(records["neuron-a"].len(), 1);
}

#[cfg(target_os = "linux")]
#[test]
#[serial]
fn every_production_reader_preflights_before_filtering_and_respects_projection() {
    let dir = TempDir::new().expect("temporary directory");
    let hostile = dir.path().join("hostile.parquet");
    let output = dir.path().join("merged.parquet");
    hostile_file(&hostile, 20_000_000);
    std::fs::write(&output, b"existing merge destination").expect("seed merge destination");
    let hostile_path = hostile.to_str().expect("hostile path");
    let prior_limit = std::env::var_os("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB");
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB", "1") };

    let by_uuid =
        read_records_from_parquet_with_profile(hostile_path, "another-neuron", ColumnProfile::Full);
    assert_memory_exhausted(by_uuid, "UUID-filtered read of an unrelated neuron");

    let by_observation = read_records_from_parquet_with_limit_and_budget(
        hostile_path,
        Some(1),
        ColumnProfile::Full,
        Some(1),
    );
    assert_memory_exhausted(by_observation, "observation-limited read");

    let wanted = HashSet::from(["another-neuron"]);
    let summary = read_mean_abs_activation_by_neuron(hostile_path, &wanted, None);
    assert!(
        summary
            .expect("activation summary projects away the oversized errors list")
            .is_empty()
    );

    let cache = StreamingRecordCache::new(hostile_path, Some(1), Some(0)).expect("build index");
    assert_memory_exhausted(cache.get("neuron-a"), "streaming block read");

    let projected =
        read_all_records_grouped_by_neuron_with_profile(hostile_path, ColumnProfile::WithoutErrors)
            .expect("projection that skips errors must not charge the skipped list");
    assert!(projected["neuron-a"][0].errors.is_empty());

    let merge = merge_parquet_files(
        output.to_str().expect("output path"),
        &[hostile_path.to_owned()],
    );
    assert_memory_exhausted(merge, "merge input read");
    assert_eq!(
        std::fs::read(&output).expect("destination remains readable"),
        b"existing merge destination",
        "preflight failure must preserve the existing merge destination"
    );
    if let Some(value) = prior_limit {
        unsafe { std::env::set_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB", value) };
    } else {
        unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB") };
    }
}

fn assert_memory_exhausted<T: std::fmt::Debug>(result: anyhow::Result<T>, path: &str) {
    let error = result.expect_err("oversized decode must be refused before batch construction");
    assert_eq!(
        classify_anyhow_error(&error),
        DiscoveryErrorKind::MemoryExhausted,
        "{path} must reject through the decode budget: {error:#}"
    );
}
