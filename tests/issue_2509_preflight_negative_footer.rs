//! Decode preflight must refuse negative footer counts and sizes (Issue #2509).
//!
//! The parquet crate accepts negative `num_rows`, `uncompressed_size` and
//! `num_values` at parse time, so the preflight projection must reject them
//! rather than count them as zero.

use neat_ai_discovery::parquet_format::decode_budget::DecodeBudget;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::ParquetMetaData;
use std::fs;
use tempfile::tempdir;

const ALL_ROOTS: [&str; 5] = ["obs_index", "neuron_uuid", "value", "activation", "errors"];

fn valid_bytes() -> Vec<u8> {
    let dir = tempdir().expect("temporary directory");
    let path = dir.path().join("records.parquet");
    let records: Vec<_> = (0..50)
        .map(|row| DiscoverRecord::new(row, "neuron-a".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]))
        .collect();
    write_records_to_parquet(path.to_str().expect("UTF-8 path"), &records)
        .expect("write valid Parquet file");
    fs::read(&path).expect("read Parquet file")
}

fn zigzag_i64(value: i64) -> Vec<u8> {
    let mut encoded = ((value << 1) ^ (value >> 63)).cast_unsigned();
    let mut bytes = Vec::new();
    while encoded >= 0x80 {
        bytes.push(u8::try_from(encoded & 0x7f).expect("varint byte fits") | 0x80);
        encoded >>= 7;
    }
    bytes.push(u8::try_from(encoded).expect("final varint byte fits"));
    bytes
}

/// Replace the `occurrence`-th compact-protocol `header, old_value` pair in the
/// footer with `header, -1`, and fix up the footer length.
fn patch_negative(bytes: &[u8], header: u8, old_value: i64, occurrence: usize) -> Vec<u8> {
    let mut bytes = bytes.to_vec();
    let trailer_at = bytes.len() - 8;
    let footer_length = u32::from_le_bytes(bytes[trailer_at..trailer_at + 4].try_into().unwrap());
    let start = trailer_at - footer_length as usize;
    let old = zigzag_i64(old_value);
    let replacement = zigzag_i64(-1);
    let offset = (start..trailer_at)
        .filter(|&at| {
            bytes[at] == header && bytes.get(at + 1..at + 1 + old.len()) == Some(old.as_slice())
        })
        .nth(occurrence)
        .expect("compact field header/value present");
    bytes.splice(
        offset + 1..offset + 1 + old.len(),
        replacement.iter().copied(),
    );
    let new_length = footer_length - u32::try_from(old.len() - replacement.len()).unwrap_or(0);
    let trailer_at = bytes.len() - 8;
    bytes[trailer_at..trailer_at + 4].copy_from_slice(&new_length.to_le_bytes());
    bytes
}

fn metadata_of(bytes: &[u8], dir: &std::path::Path) -> ParquetMetaData {
    let path = dir.join("patched.parquet");
    fs::write(&path, bytes).expect("write patched Parquet file");
    ParquetRecordBatchReaderBuilder::try_new(fs::File::open(&path).expect("open patched file"))
        .expect("the parquet crate accepts the negative footer value")
        .metadata()
        .as_ref()
        .clone()
}

/// Patch the first `header, old_value` occurrence for which `is_target` holds
/// on the parsed footer, so the test never depends on which field it hit.
fn patched_footer(
    header: u8,
    old_value: i64,
    is_target: impl Fn(&ParquetMetaData) -> bool,
) -> ParquetMetaData {
    let bytes = valid_bytes();
    let dir = tempdir().expect("temporary directory");
    for occurrence in 0..16 {
        let patched = patch_negative(&bytes, header, old_value, occurrence);
        let path = dir.path().join(format!("probe-{occurrence}.parquet"));
        fs::write(&path, &patched).expect("write probe file");
        let Ok(builder) = ParquetRecordBatchReaderBuilder::try_new(fs::File::open(&path).unwrap())
        else {
            continue;
        };
        let metadata = builder.metadata().as_ref().clone();
        if is_target(&metadata) {
            return metadata;
        }
    }
    panic!("no occurrence of the footer field produced the intended negative value");
}

fn column_index(metadata: &ParquetMetaData, root: &str) -> usize {
    metadata
        .row_group(0)
        .columns()
        .iter()
        .position(|column| column.column_path().parts().first().map(String::as_str) == Some(root))
        .expect("column present")
}

fn negative_row_group_rows() -> ParquetMetaData {
    patched_footer(0x16, 50, |m| m.row_group(0).num_rows() == -1)
}

fn negative_uncompressed_size() -> ParquetMetaData {
    let valid = valid_bytes();
    let dir = tempdir().unwrap();
    let valid_metadata = metadata_of(&valid, dir.path());
    let size = valid_metadata
        .row_group(0)
        .column(column_index(&valid_metadata, "obs_index"))
        .uncompressed_size();
    patched_footer(0x16, size, |m| {
        m.row_group(0)
            .column(column_index(m, "obs_index"))
            .uncompressed_size()
            == -1
    })
}

fn negative_error_values() -> ParquetMetaData {
    patched_footer(0x16, 100, |m| {
        m.row_group(0)
            .column(column_index(m, "errors"))
            .num_values()
            == -1
    })
}

fn assert_refused(metadata: &ParquetMetaData, roots: &[&str], field: &str) {
    // No ceiling: the refusal must come from the footer value, not the limit.
    let budget = DecodeBudget::new(None);
    for (stage, result) in [
        (
            "preflight_columns",
            budget.preflight_columns(metadata, roots, "x.parquet"),
        ),
        (
            "preflight_row_groups",
            budget.preflight_row_groups(metadata, roots, "x.parquet"),
        ),
    ] {
        let message = format!("{:#}", result.expect_err(&format!("{stage} must refuse")));
        assert!(
            message.contains(&format!("Invalid Parquet footer {field}: -1")),
            "{stage} error must name {field}: {message}"
        );
    }
}

fn assert_accepted(metadata: &ParquetMetaData, roots: &[&str]) {
    let budget = DecodeBudget::new(None);
    budget
        .preflight_columns(metadata, roots, "x.parquet")
        .expect("preflight_columns accepts");
    budget
        .preflight_row_groups(metadata, roots, "x.parquet")
        .expect("preflight_row_groups accepts");
}

#[test]
fn valid_footer_passes_both_preflights() {
    let dir = tempdir().unwrap();
    assert_accepted(&metadata_of(&valid_bytes(), dir.path()), &ALL_ROOTS);
}

#[test]
fn negative_row_group_num_rows_is_refused() {
    assert_refused(&negative_row_group_rows(), &ALL_ROOTS, "row-group num_rows");
}

#[test]
fn negative_selected_uncompressed_size_is_refused() {
    assert_refused(
        &negative_uncompressed_size(),
        &ALL_ROOTS,
        "uncompressed_size",
    );
}

#[test]
fn negative_errors_num_values_is_refused() {
    assert_refused(&negative_error_values(), &ALL_ROOTS, "num_values");
}

#[test]
fn negative_values_outside_the_selection_are_not_charged() {
    // `obs_index` is not selected, so its size is not part of the projection.
    assert_accepted(&negative_uncompressed_size(), &["neuron_uuid"]);
    // The error-value term is only read when the `errors` column is selected.
    assert_accepted(&negative_error_values(), &["obs_index"]);
}
