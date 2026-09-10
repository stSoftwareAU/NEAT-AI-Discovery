//! Issue #2066 — the `src/parquet_format/` module headers must state a
//! contract, not restate their own file path.
//!
//! `reader.rs`, `schema.rs` and `writer.rs` each opened with a header fully
//! derivable from the file name ("Parquet reading and deserialisation for
//! discovery records"), so the module index taught a reader nothing about the
//! properties that actually matter when integrating with the on-disk format.
//! These tests drive the three contracts the new headers claim, so the prose
//! cannot drift into fiction:
//!
//!   1. `schema.rs` — the five fixed columns, in order, with `value` the only
//!      nullable one.
//!   2. `reader.rs` — reads are column-projected, and a profile that omits
//!      `errors` returns an empty `errors` vec rather than the stored values.
//!   3. `writer.rs` — the destination file is not a readable Parquet file
//!      until `finish()` writes the footer.
//!
//! A light check that each header no longer *is* the old paraphrase guards the
//! regression itself; the behavioural tests above are what keep it honest.

use arrow::datatypes::DataType;
use neat_ai_discovery::parquet_format::{
    ColumnProfile, ParquetRecordWriter, create_schema, read_all_records_from_parquet,
    read_records_from_parquet_with_profile, write_records_to_parquet,
};
use neat_ai_discovery::types::DiscoverRecord;
use tempfile::TempDir;

fn sample_records() -> Vec<DiscoverRecord> {
    vec![
        DiscoverRecord {
            obs_index: 0,
            neuron_uuid: "neuron-a".to_string(),
            value: Some(0.25),
            activation: 0.5,
            errors: vec![0.1, 0.2, 0.3],
        },
        DiscoverRecord {
            obs_index: 1,
            neuron_uuid: "neuron-a".to_string(),
            value: None,
            activation: -0.75,
            errors: vec![0.4],
        },
    ]
}

/// `schema.rs` claims five fixed columns in a fixed order, with `value` the
/// only nullable one.
#[test]
fn schema_declares_five_columns_with_value_the_only_nullable_one() {
    let schema = create_schema();

    let expected: [(&str, DataType, bool); 5] = [
        ("obs_index", DataType::UInt32, false),
        ("neuron_uuid", DataType::Utf8, false),
        ("value", DataType::Float32, true),
        ("activation", DataType::Float32, false),
        (
            "errors",
            DataType::List(std::sync::Arc::new(arrow::datatypes::Field::new(
                "item",
                DataType::Float32,
                false,
            ))),
            false,
        ),
    ];

    assert_eq!(
        schema.fields().len(),
        expected.len(),
        "documented schema has exactly {} columns",
        expected.len()
    );

    for (index, (name, data_type, nullable)) in expected.iter().enumerate() {
        let field = schema.field(index);
        assert_eq!(field.name(), name, "column {index} name");
        assert_eq!(field.data_type(), data_type, "column {name} type");
        assert_eq!(field.is_nullable(), *nullable, "column {name} nullability");
    }
}

/// `reader.rs` claims a projection that omits `errors` returns records with an
/// empty `errors` vec — "not read", not "no errors".
#[test]
fn without_errors_profile_returns_empty_errors_while_full_returns_the_values() {
    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join("discovery_data.parquet");
    let path = path.to_str().expect("utf-8 path");

    let records = sample_records();
    write_records_to_parquet(path, &records).expect("write records");

    let full = read_records_from_parquet_with_profile(path, "neuron-a", ColumnProfile::Full)
        .expect("full read");
    assert_eq!(full.len(), 2, "full profile returns every row");
    assert_eq!(
        full.iter().map(|r| r.errors.clone()).collect::<Vec<_>>(),
        vec![vec![0.1, 0.2, 0.3], vec![0.4]],
        "full profile decodes the stored errors"
    );

    let projected =
        read_records_from_parquet_with_profile(path, "neuron-a", ColumnProfile::WithoutErrors)
            .expect("projected read");
    assert_eq!(
        projected.len(),
        2,
        "the projection changes columns, not rows"
    );
    assert!(
        projected.iter().all(|r| r.errors.is_empty()),
        "WithoutErrors leaves errors empty: {projected:?}"
    );
    assert_eq!(
        projected
            .iter()
            .map(|r| (r.obs_index, r.value, r.activation))
            .collect::<Vec<_>>(),
        full.iter()
            .map(|r| (r.obs_index, r.value, r.activation))
            .collect::<Vec<_>>(),
        "every projected column still decodes identically"
    );
}

/// `writer.rs` claims the destination is not a readable Parquet file until
/// `finish()` writes the footer — the reason callers must write to a
/// `.parquet.tmp` sibling and rename it into place.
#[test]
fn file_is_only_readable_after_finish_writes_the_footer() {
    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join("in_progress.parquet");
    let path = path.to_str().expect("utf-8 path").to_string();

    let records = sample_records();
    let mut writer =
        ParquetRecordWriter::new(&path, 1_000, 1_000, records.len()).expect("create writer");
    writer.write_records(&records).expect("write records");

    assert!(
        std::path::Path::new(&path).exists(),
        "the destination file exists while the writer is still open"
    );
    assert!(
        read_all_records_from_parquet(&path).is_err(),
        "a footer-less file must not read back as valid Parquet"
    );

    writer.finish().expect("finish writer");

    let read_back = read_all_records_from_parquet(&path).expect("read after finish");
    assert_eq!(
        read_back.len(),
        records.len(),
        "every record is readable once the footer is written"
    );
}

/// The regression itself: no header may be the file-name paraphrase again.
#[test]
fn module_headers_are_no_longer_the_file_name_paraphrase() {
    let paraphrases = [
        (
            "reader.rs",
            include_str!("../src/parquet_format/reader.rs"),
            "Parquet reading and deserialisation for discovery records",
        ),
        (
            "schema.rs",
            include_str!("../src/parquet_format/schema.rs"),
            "Schema definitions and validation for discovery Parquet records",
        ),
        (
            "writer.rs",
            include_str!("../src/parquet_format/writer.rs"),
            "Parquet writing and serialisation for discovery records",
        ),
    ];

    for (name, source, paraphrase) in paraphrases {
        let header = module_doc(source);
        assert!(
            !header.starts_with(paraphrase),
            "{name} still opens with the file-name paraphrase: {header}"
        );
        assert!(
            header.len() > paraphrase.len(),
            "{name} header must carry more contract than the paraphrase it replaced"
        );
    }
}

/// The leading `//!` block of a Rust source file, with the markers stripped.
fn module_doc(source: &str) -> String {
    source
        .lines()
        .take_while(|line| line.trim_start().starts_with("//!") || line.trim().is_empty())
        .filter_map(|line| line.trim_start().strip_prefix("//!"))
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}
