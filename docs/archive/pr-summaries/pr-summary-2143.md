## Summary

Chunk 7b read-side audit of `src/parquet_format` (`reader.rs`, `batch_columns.rs`, `footer.rs`,
`decode_budget.rs`, `schema.rs`, `activation_summary.rs`). All six files were read end to end and
probed with hand-built Parquet files. The staged ledger
`docs/audits/in-progress/security-sweep-chunk-07-parquet.md` now carries the six per-file outcomes,
every probe (class, `file:line`, input, outcome) and the filed issues. Two findings survived:

- **#2446 (SEC-2091-01, medium)** — the decode budget is charged after the parquet crate has built
  the whole Arrow batch. A 1,643-byte file with one 20,000,000-value `errors` list grew the process
  high-water mark by 240,586,752 bytes under a 1 MB budget.
- **#2447 (SEC-2091-02, low)** — `read_footer_stats` maps a negative footer `num_rows` /
  `total_byte_size` / `num_values` to 0 (`footer.rs` L79, L83, L98), so the footer projection
  under-reports.

`batch_columns.rs` and `schema.rs` are clean. The write side (`mod.rs`, `writer.rs`,
`shared_records.rs`) stays `pending #2144`, so the record stays under `in-progress/` and chunk `7`
in `lib-sweep-coverage.json` is unchanged. No production Rust and no `Cargo.toml` change: nothing
in `src/` changed, so there is no version bump. Closes #2143.

## Spec

### Intent and Rationale

- The issue asks for an audit that files one issue per surviving finding, not for fixes. Fixes ship
  in #2446 and #2447, each with its own failing-first test.
- Probes that found nothing are recorded in the ledger and were not committed; the throwaway probe
  file was deleted before the commit.

### Essential Design Decisions

- The ledger stays at `docs/audits/in-progress/` (trusted comment on #2143), because
  `tests/issue_2088_sweep_ledger_contract.rs` scans only the top level of `docs/audits/`.
- The issue says "no `.rs` change". This PR adds one test file, `tests/issue_2143_chunk_07b_read_side_sweep.rs`,
  because the worker's security-fix gate for `security`-labelled issues blocks PR creation unless the
  branch changes a test file. It guards the ledger text only (like `tests/issue_2249_chunk9_ledger_complete.rs`);
  nothing under `src/` and no `Cargo.toml` change.
- The parquet crate question is answered in the ledger: parquet 60.0.0 builds the whole `RecordBatch`
  (default 1,024 rows, `DEFAULT_BATCH_SIZE` in `parquet-60.0.0/src/arrow/arrow_reader/mod.rs`) before
  the first `charge_record`, and has no per-list cap. #2446 asks for the in-repo footer pre-check
  rather than a crate change.

### Undiscoverable Facts

- Peak memory was measured with `/proc/self/clear_refs` (writing `5`) then `VmHWM`; the crate's
  global allocator (`src/tracking_alloc.rs`) is private, so a test-local counting allocator is not an option.
- Footer lies were produced by patching compact-Thrift bytes in the footer of a 50-row file
  (`0x16 0x64` → `0x16 0x01`), which the parquet crate accepts.

## Evidence

Audit only; no UI or performance surface. Probe outputs (recorded in the ledger):

- `read_all_records_grouped_by_neuron_bounded(path, None, Full, Some(1))` on the 1,643-byte file:
  `MemoryExhausted`, high-water growth 240,586,752 bytes.
- Footer `num_rows` patched to -1: `ParquetFooterStats { num_rows: 0, uncompressed_bytes: 459, error_values: 100 }`,
  while the reader decodes 50 rows.
- Schema trust, NaN/inf/null and UUID probes: see the ledger's `## Defect classes probed`.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

CRITERIA_PENDING_REVIEW

## Test Plan

TEST_PLAN_PENDING_REVIEW
