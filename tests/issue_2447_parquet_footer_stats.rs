//! Regression coverage for malformed Parquet footer statistics (Issue #2447).

use neat_ai_discovery::parquet_format::footer::read_footer_stats;
use neat_ai_discovery::parquet_format::{create_schema, write_records_to_parquet};
use neat_ai_discovery::types::DiscoverRecord;
use parquet::arrow::ArrowWriter;
use std::fs;
use std::sync::Arc;
use tempfile::tempdir;

fn valid_parquet() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("temporary directory");
    let path = dir.path().join("records.parquet");
    let records: Vec<_> = (0..50)
        .map(|row| DiscoverRecord::new(row, "neuron-a".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]))
        .collect();
    write_records_to_parquet(path.to_str().expect("UTF-8 path"), &records)
        .expect("write valid Parquet file");
    (dir, path)
}

fn read_bytes(path: &std::path::Path) -> Vec<u8> {
    fs::read(path).expect("read Parquet file")
}

fn footer_start(bytes: &[u8]) -> usize {
    let footer_length_at = bytes.len() - 8;
    let footer_length = u32::from_le_bytes(
        bytes[footer_length_at..footer_length_at + 4]
            .try_into()
            .expect("footer length"),
    ) as usize;
    let magic = &bytes[footer_length_at + 4..];
    assert_eq!(magic, b"PAR1", "valid Parquet trailer");
    footer_length_at - footer_length
}

fn zigzag_i64(value: i64) -> Vec<u8> {
    let mut encoded = ((value << 1) ^ (value >> 63)) as u64;
    let mut bytes = Vec::new();
    while encoded >= 0x80 {
        bytes.push((encoded as u8) | 0x80);
        encoded >>= 7;
    }
    bytes.push(encoded as u8);
    bytes
}

fn patch_first_compact_field(
    bytes: &mut Vec<u8>,
    field_header: u8,
    old_value: i64,
    new_value: i64,
    occurrence: usize,
) {
    let start = footer_start(bytes);
    let old = zigzag_i64(old_value);
    let replacement = zigzag_i64(new_value);
    let matches: Vec<_> = (start..bytes.len() - 8)
        .filter(|&offset| {
            bytes[offset] == field_header
                && bytes.get(offset + 1..offset + 1 + old.len()) == Some(old.as_slice())
        })
        .collect();
    let offset = *matches
        .get(occurrence)
        .unwrap_or_else(|| panic!("compact field header/value found {} times", matches.len()));
    bytes.splice(
        offset + 1..offset + 1 + old.len(),
        replacement.iter().copied(),
    );
    let trailer_at = bytes.len() - 8;
    let footer_length = u32::from_le_bytes(
        bytes[trailer_at..trailer_at + 4]
            .try_into()
            .expect("footer length"),
    );
    let new_footer_length = if replacement.len() >= old.len() {
        footer_length + u32::try_from(replacement.len() - old.len()).expect("small delta")
    } else {
        footer_length - u32::try_from(old.len() - replacement.len()).expect("small delta")
    };
    bytes[trailer_at..trailer_at + 4].copy_from_slice(&new_footer_length.to_le_bytes());
}

fn expect_footer_error(bytes: &[u8], field: &str, diagnostic: &str) {
    let dir = tempdir().expect("temporary directory");
    let path = dir.path().join("patched.parquet");
    fs::write(&path, bytes).expect("write patched Parquet file");
    let error = read_footer_stats(path.to_str().expect("UTF-8 path"))
        .expect_err("malformed footer must be rejected");
    let message = format!("{error:#}");
    assert!(
        message.contains(field),
        "error must identify {field}: {message}"
    );
    assert!(
        message.contains(diagnostic),
        "error must identify {diagnostic}: {message}"
    );
}

#[test]
fn valid_footer_stats_match_written_records() {
    let (_dir, path) = valid_parquet();
    let stats = read_footer_stats(path.to_str().expect("UTF-8 path")).expect("valid footer");
    assert_eq!(stats.num_rows, 50);
    assert_eq!(stats.error_values, 100);
    assert!(stats.uncompressed_bytes > 0);
}

#[test]
fn empty_footer_stats_remain_valid() {
    let dir = tempdir().expect("temporary directory");
    let path = dir.path().join("empty.parquet");
    let writer = ArrowWriter::try_new(
        fs::File::create(&path).expect("create empty Parquet file"),
        Arc::new(create_schema()),
        None,
    )
    .expect("create Parquet writer");
    writer.close().expect("write empty footer");

    let stats = read_footer_stats(path.to_str().expect("UTF-8 path")).expect("valid empty footer");
    assert_eq!(stats.num_rows, 0);
    assert_eq!(stats.uncompressed_bytes, 0);
    assert_eq!(stats.error_values, 0);
}

#[test]
fn negative_file_num_rows_is_rejected() {
    let (_dir, path) = valid_parquet();
    let mut bytes = read_bytes(&path);
    // FileMetaData field 3 follows field 2, so compact-protocol delta/type is 0x16.
    patch_first_compact_field(&mut bytes, 0x16, 50, -1, 0);
    expect_footer_error(&bytes, "num_rows", "Invalid Parquet footer num_rows");
}

#[test]
fn fuzz_seed_negative_file_num_rows_is_rejected() {
    let seed = include_bytes!("../fuzz/seeds/parquet_footer_stats/negative_file_num_rows");
    expect_footer_error(seed, "num_rows", "Invalid Parquet footer num_rows");
}

#[test]
fn understated_file_num_rows_is_rejected() {
    let (_dir, path) = valid_parquet();
    let mut bytes = read_bytes(&path);
    patch_first_compact_field(&mut bytes, 0x16, 50, 1, 0);
    expect_footer_error(&bytes, "num_rows", "does not match row-group num_rows sum");
}

#[test]
fn negative_row_group_num_rows_is_rejected() {
    let (_dir, path) = valid_parquet();
    let mut bytes = read_bytes(&path);
    // The final matching field 3 is the row-group count after column metadata.
    patch_first_compact_field(&mut bytes, 0x16, 50, -1, 5);
    expect_footer_error(
        &bytes,
        "num_rows",
        "Invalid Parquet footer row-group num_rows",
    );
}

#[test]
fn negative_total_byte_size_is_rejected() {
    let (_dir, path) = valid_parquet();
    let stats = read_footer_stats(path.to_str().expect("UTF-8 path")).expect("valid footer");
    let mut bytes = read_bytes(&path);
    // RowGroup field 2 follows its column list and is an I64 delta field (0x16).
    patch_first_compact_field(
        &mut bytes,
        0x16,
        i64::try_from(stats.uncompressed_bytes).expect("small fixture size"),
        -1,
        0,
    );
    expect_footer_error(
        &bytes,
        "total_byte_size",
        "Invalid Parquet footer total_byte_size",
    );
}

#[test]
fn negative_num_values_is_rejected() {
    let (_dir, path) = valid_parquet();
    let mut bytes = read_bytes(&path);
    // The errors leaf column has 100 values; compact header 0x16 is field 5.
    patch_first_compact_field(&mut bytes, 0x16, 100, -1, 0);
    expect_footer_error(&bytes, "num_values", "Invalid Parquet footer num_values");
}
