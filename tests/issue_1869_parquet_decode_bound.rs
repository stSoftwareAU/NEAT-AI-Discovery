//! Issue #1869: bound the Parquet decode instead of predicting it.
//!
//! The pre-load admission decision used to be `compressed_file_size × 3`, and
//! the only budget check ran *after* the allocation had already happened. Two
//! behaviours are asserted here:
//!
//! 1. [`estimate_parquet_in_memory_bytes`] projects from the **footer** (exact
//!    decompressed row count and error-value count), so a highly compressible
//!    file is no longer projected at three times its on-disk size.
//! 2. The batch loop in the reader charges every materialised record against a
//!    decode budget and aborts with a typed `MemoryExhausted` error *during*
//!    the decode, rather than letting the allocation complete (or abort the
//!    process) and reporting it afterwards.

use neat_ai_discovery::analysis::utils::estimate_parquet_in_memory_bytes;
use neat_ai_discovery::export::dense_bound::{
    ensure_dense_snapshot_fits, projected_dense_snapshot_bytes,
};
use neat_ai_discovery::ffi_types::DiscoveryErrorKind;
use neat_ai_discovery::parquet_format::decode_budget::{
    DecodeBudget, default_decode_limit_bytes, estimated_record_bytes,
};
use neat_ai_discovery::parquet_format::footer::read_footer_stats;
use neat_ai_discovery::parquet_format::{
    ColumnProfile, read_all_records_grouped_by_neuron_bounded,
    read_records_from_parquet_with_limit, read_records_from_parquet_with_limit_and_budget,
    write_records_to_parquet,
};
use neat_ai_discovery::types::DiscoverRecord;
use serial_test::serial;
use tempfile::NamedTempFile;

/// A deliberately low-entropy recording: every row repeats the same UUID and
/// the same error values, which Parquet's dictionary + RLE encodings compress
/// far better than the 3:1 average the old estimator assumed.
fn write_compressible_parquet(rows: usize, errors_per_row: usize) -> NamedTempFile {
    let file = NamedTempFile::with_suffix(".parquet").expect("temp file");
    let records: Vec<DiscoverRecord> = (0..rows)
        .map(|i| {
            DiscoverRecord::new(
                u32::try_from(i % 8).unwrap_or(0),
                "11111111-2222-3333-4444-555555555555".to_string(),
                Some(0.5),
                0.5,
                vec![0.25f32; errors_per_row],
            )
        })
        .collect();
    write_records_to_parquet(file.path().to_str().expect("utf-8 path"), &records).expect("write");
    file
}

#[test]
fn footer_stats_report_decompressed_row_count() {
    let file = write_compressible_parquet(20_000, 8);
    let path = file.path().to_str().expect("utf-8 path");

    let stats = read_footer_stats(path).expect("footer readable");
    assert_eq!(stats.num_rows, 20_000, "row count comes from the footer");
    assert!(
        stats.uncompressed_bytes > 0,
        "row groups report an uncompressed size"
    );
}

#[test]
fn projection_is_not_derived_from_compressed_size_alone() {
    let file = write_compressible_parquet(20_000, 8);
    let path = file.path().to_str().expect("utf-8 path");
    let on_disk = std::fs::metadata(path).expect("metadata").len();

    let projected = estimate_parquet_in_memory_bytes(path);

    // The old estimator returned exactly on_disk × 3. A low-entropy file
    // decodes to far more than that, so the footer-derived projection must
    // exceed the compressed heuristic.
    assert!(
        projected > on_disk.saturating_mul(3),
        "projection {projected} should exceed the compressed heuristic {}",
        on_disk.saturating_mul(3)
    );

    // It must also cover the real cost of the materialised records.
    let minimum_real_cost = 20_000 * estimated_record_bytes(36, 8);
    assert!(
        projected >= minimum_real_cost,
        "projection {projected} should cover the materialised cost {minimum_real_cost}"
    );
}

#[test]
fn grouped_decode_aborts_when_the_budget_is_exhausted() {
    let file = write_compressible_parquet(50_000, 16);
    let path = file.path().to_str().expect("utf-8 path");

    let err = read_all_records_grouped_by_neuron_bounded(path, None, ColumnProfile::Full, Some(1))
        .expect_err("a 1 MB budget cannot hold 50k records");

    let kind = neat_ai_discovery::ffi_types::classify_anyhow_error(&err);
    assert_eq!(
        kind,
        DiscoveryErrorKind::MemoryExhausted,
        "budget aborts are typed as memory exhaustion, got {kind:?}: {err}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("decode budget"),
        "error should name the decode budget: {msg}"
    );
}

#[test]
fn grouped_decode_succeeds_within_a_generous_budget() {
    let file = write_compressible_parquet(5_000, 4);
    let path = file.path().to_str().expect("utf-8 path");

    let grouped =
        read_all_records_grouped_by_neuron_bounded(path, None, ColumnProfile::Full, Some(4_096))
            .expect("4 GB budget comfortably holds 5k records");

    let total: usize = grouped.values().map(Vec::len).sum();
    assert_eq!(
        total, 5_000,
        "every record is returned when the budget fits"
    );
}

#[test]
fn limited_read_aborts_when_the_budget_is_exhausted() {
    let file = write_compressible_parquet(50_000, 16);
    let path = file.path().to_str().expect("utf-8 path");

    // `max_obs` caps distinct observations, not rows: all 50k rows share eight
    // obs_index values, so the observation limit lets every row through. The
    // decode budget is the bound that actually holds.
    let err = read_records_from_parquet_with_limit_and_budget(
        path,
        Some(8),
        ColumnProfile::Full,
        Some(1),
    )
    .expect_err("a 1 MB budget cannot hold 50k records");

    assert_eq!(
        neat_ai_discovery::ffi_types::classify_anyhow_error(&err),
        DiscoveryErrorKind::MemoryExhausted
    );
}

#[test]
fn a_budget_charges_every_record_and_reports_consumption() {
    let mut budget = DecodeBudget::new(Some(estimated_record_bytes(36, 4) * 2));

    assert!(budget.charge_record(36, 4, "file.parquet").is_ok());
    assert!(budget.charge_record(36, 4, "file.parquet").is_ok());
    let err = budget
        .charge_record(36, 4, "file.parquet")
        .expect_err("third record exceeds the two-record budget");
    assert!(err.to_string().contains("file.parquet"));
    assert!(budget.consumed_bytes() > 0);
}

#[test]
fn an_unlimited_budget_never_aborts() {
    let mut budget = DecodeBudget::new(None);
    for _ in 0..1_000 {
        budget
            .charge_record(36, 64, "file.parquet")
            .expect("no limit means no abort");
    }
    assert_eq!(budget.limit_bytes(), None);
}

#[test]
fn default_limit_is_half_of_total_ram_and_unknown_ram_is_unbounded() {
    assert_eq!(
        default_decode_limit_bytes(8 * 1024 * 1024 * 1024),
        Some(4 * 1024 * 1024 * 1024)
    );
    assert_eq!(
        default_decode_limit_bytes(0),
        None,
        "an unknown RAM figure cannot produce a meaningful ceiling"
    );
}

/// Paths with no caller budget (the snapshot export, focus gradients) are
/// bounded by the environment override.
#[test]
#[serial]
fn the_environment_override_bounds_budget_free_decode_paths() {
    let file = write_compressible_parquet(50_000, 16);
    let path = file.path().to_str().expect("utf-8 path");

    // SAFETY: `#[serial]` guarantees no other test runs concurrently, so no
    // other thread can read the environment while it is being mutated.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB", "1") };
    let bounded = read_records_from_parquet_with_limit(path, None);
    // SAFETY: as above — `#[serial]` serialises this test against all others.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB") };

    let err = bounded.expect_err("the 1 MB override bounds a budget-free decode");
    assert_eq!(
        neat_ai_discovery::ffi_types::classify_anyhow_error(&err),
        DiscoveryErrorKind::MemoryExhausted
    );

    // Without the override the same read completes (the default ceiling is
    // half of total RAM).
    let records = read_records_from_parquet_with_limit(path, None)
        .expect("unbounded by the override, the decode fits the host default");
    assert_eq!(records.len(), 50_000);
}

#[test]
fn dense_snapshot_grid_is_bounded_before_allocation() {
    // A sparse-but-wide recording: the decoded records are few, but a dense
    // neurons × observations grid is not.
    let projected = projected_dense_snapshot_bytes(200_000, 200_000);
    assert!(
        projected > 1024 * 1024 * 1024 * 1024,
        "a 200k × 200k grid projects into terabytes, got {projected}"
    );

    let err = ensure_dense_snapshot_fits(200_000, 200_000)
        .expect_err("a terabyte-scale grid cannot fit any host budget");
    assert_eq!(
        neat_ai_discovery::ffi_types::classify_anyhow_error(&err),
        DiscoveryErrorKind::MemoryExhausted
    );

    ensure_dense_snapshot_fits(10, 100).expect("a small grid is allowed through");
}

#[test]
fn estimated_record_bytes_grows_with_uuid_and_errors() {
    let base = estimated_record_bytes(0, 0);
    assert!(estimated_record_bytes(36, 0) > base, "UUID heap is charged");
    assert!(
        estimated_record_bytes(36, 10) > estimated_record_bytes(36, 0),
        "errors payload is charged"
    );
    assert_eq!(
        estimated_record_bytes(36, 10) - estimated_record_bytes(36, 0),
        40,
        "ten f32 errors cost 40 bytes"
    );
}
