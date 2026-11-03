## Summary

The Parquet writer tests in `src/parquet_format/mod.rs` could not go red when the behaviour they are named for broke. They asserted `Path::exists()` on a `NamedTempFile` (which exists before the writer runs), asserted only `is_ok()`, and never read the file back. This PR makes them read every row back, adds a round-trip and rejection suite in `tests/issue_2453_parquet_writer_chunking.rs`, and pins the exact chunk boundaries of the private `determine_chunk_end` with unit tests in `src/parquet_format/writer.rs`. One small production hardening: `ParquetRecordWriter::new` now rejects a per-batch UUID byte or error-value limit above `i32::MAX` (Arrow's offset ceiling) up front, before creating the file. Version bump: `Cargo.toml` (and the matching `Cargo.lock` entry) updated for this change; the merged commit records the final version. Closes #2453.

## Spec

### Intent and Rationale

- The writer's chunking and validation paths were not guarded: a `determine_chunk_end` that never split, or a writer that dropped a row, kept every test green.
- Tests assert on what the reader returns (row count, `obs_index`, UUID, value, activation, `errors` per row) and on the specific rejection message, with a legal sibling input accepted as the positive control.
- Chunk boundaries are **not observable in the written file** (`ArrowWriter` coalesces batches into one row group), so a round-trip test alone cannot see a missing split. The boundaries are pinned by direct unit tests of `determine_chunk_end` inside `writer.rs`, where the private function is reachable without widening its visibility.

### Essential Design Decisions

- `determine_chunk_end` stays private; its tests live in `writer.rs`'s own `mod tests`. The in-module tests in `mod.rs` keep their names, as the issue requires.
- The `end > start &&` guard is unreachable through `write_records` (`validate_records` rejects any single record above either limit first), so no round-trip test can kill its removal. Only `determine_chunk_end_oversized_first_uuid_still_makes_progress` / `..._error_count_...` do, by calling the function directly.
- The i32 overflow path is exercised through the real ceiling: a capacity of `i32::MAX + 1` is rejected before the file is created (gated to 64-bit targets), with `i32::MAX` accepted.

### Undiscoverable Facts

- The audit (#2144) found the chunk arithmetic correct, but `new()` documented validating the batch limits and did not bound them: a limit above `i32::MAX` let one chunk overflow Arrow's `i32` offsets, which panics in `StringArray`/`OffsetBuffer` rather than returning an error. All in-tree callers pass `i32::MAX`, so none changes behaviour. The other red evidence is mutation of production code (below).
- The worker's periodic WIP checkpoint snapshotted one of my mutation probes into commit `762b8f8`; the following commit (`94e18b8`) restores the source, so `determine_chunk_end`, `write_records` and `write_chunk` match the base. The only production change at the head is in `ParquetRecordWriter::new`, which now also rejects a per-batch UUID byte or error-value limit above `i32::MAX` (commit `545055c`).
- `bump-deps.sh` added a `chore: bump dependencies` commit (`cc` 1.10.0 → 1.11.0 in `Cargo.lock`); it is unrelated to the tests. Unit tests: `cargo test --lib parquet_format` 45 passed after merging `origin/Develop` (which added further `writer.rs` tests).

## Evidence

Backend-only change; no UI. Tests were run on the final head.

- `cargo test --test issue_2453_parquet_writer_chunking`: 13 passed.
- `cargo test --lib parquet_format`: 45 passed (after merging `origin/Develop`) (includes the 5 strengthened `mod.rs` tests and 8 new `determine_chunk_end` tests).
- `cargo clippy --all-targets -- -D warnings`: clean. `cargo fmt --all --check`: clean.

## Reproduction

- **symptom** — the writer tests stay green when `determine_chunk_end` never splits, drops a record, or the capacity ceiling moves, because they assert vacuous conditions
- **status** — `verified` — `uuid_byte_limit_chunking_round_trips_every_record` fails against the base `writer.rs` (the oversized UUID byte limit is accepted: "a UUID byte limit above i32::MAX must be rejected") and passes at the head (re-observed on the final head by swapping in the base `writer.rs`, running the test, then restoring). The vacuous-assertion half is shown red under deliberate production mutations (Test Plan).
- **regression test** — `tests/issue_2453_parquet_writer_chunking.rs::uuid_byte_limit_chunking_round_trips_every_record`

The original trigger (assertions that cannot fail) is closed: the vacuous `exists()`/`is_ok()` checks are replaced by per-row read-back and specific-message assertions, and each is shown red under mutation. No equivalent bypass remains for the chunk-boundary logic, since the boundary vectors are asserted exactly, including exact-fit (`>` vs `>=`) and each limit operand independently.

## Test Plan

Added (all exist at head):

- `tests/issue_2453_parquet_writer_chunking.rs`: `uuid_byte_limit_chunking_round_trips_every_record` (also asserts the over-ceiling UUID limit is rejected), `error_value_limit_above_arrow_offset_ceiling_is_rejected`, `uuid_byte_limit_chunking_round_trips_across_write_calls`, `error_value_limit_chunking_round_trips_every_record`, `both_batch_limits_small_round_trips_every_record`, `exact_boundary_uuid_and_error_counts_are_accepted_and_round_trip`, `uuid_longer_than_limit_is_rejected_and_accepted_once_limit_is_raised`, `error_list_longer_than_limit_is_rejected_and_accepted_once_limit_is_raised`, `zero_uuid_byte_limit_is_rejected_and_accepted_at_one`, `zero_error_value_limit_is_rejected_and_accepted_at_one`, `zero_capacity_is_rejected_before_file_creation_and_accepted_at_record_count`, `writing_more_records_than_capacity_is_rejected_and_accepted_at_capacity`, `capacity_above_arrow_row_limit_is_rejected_before_file_creation`.
- `src/parquet_format/writer.rs`: `determine_chunk_end_uuid_limit_binds_only`, `determine_chunk_end_error_limit_binds_only`, `determine_chunk_end_mixed_error_counts_exact_vector`, `determine_chunk_end_both_limits_bind_at_different_records`, `determine_chunk_end_oversized_first_uuid_still_makes_progress`, `determine_chunk_end_oversized_first_error_count_still_makes_progress`, `determine_chunk_end_respects_nonzero_start`, `determine_chunk_end_unlimited_limits_yield_single_chunk`.

Edited in `src/parquet_format/mod.rs` (names unchanged): `test_write_records_to_parquet`, `test_write_records_prevents_i32_overflow`, `test_write_records_validates_error_count_overflow`, `test_write_records_chunks_when_uuid_bytes_exceed_limit`, `test_write_records_chunks_when_error_value_count_exceeds_limit`.

- Removed from `src/parquet_format/mod.rs`: `assert!(std::path::Path::new(file_path).exists())` in `test_write_records_prevents_i32_overflow`
  - Obsolete: the path came from `NamedTempFile::new()`, so it existed before the writer ran and the assertion could not fail. Replaced by: `assert_records_round_trip(&file_path, &records)`. Still covered by: `src/parquet_format/mod.rs::test_write_records_prevents_i32_overflow`.
- Kept (now meaningful) in `test_write_records_to_parquet` and the two chunking tests: the post-write `exists()` assertion, preceded by `fresh_parquet_path`, which asserts the path does **not** exist before the writer runs, and followed by `assert_records_round_trip`.
- Removed from `src/parquet_format/mod.rs`: `assert!(writer.write_records(&records).is_ok(), ...)` in the error-value chunking test
  - Replaced by: `.expect(...)` on the same call, plus `assert_records_round_trip`. The success check is kept; read-back is added.

Red checks (production `determine_chunk_end` / `write_records` / `new` mutated one at a time, then restored; head re-run green):

- never split (limit comparison replaced with `false`): integration suite stays **green** (12 passed — boundaries are invisible in the file, by design); 7 `determine_chunk_end_*` unit tests go red.
- remove `end > start &&`: red only in the two `..._oversized_first_..._still_makes_progress` tests (unreachable through `write_records`).
- drop the uuid operand / drop the error operand / `>` to `>=` on each / delete `uuid_bytes += uuid_len` / delete `error_value_count += errors_len`: each reds at least one `determine_chunk_end_*` test (`>=` on the uuid comparison reds `determine_chunk_end_uuid_limit_binds_only`'s exact-fit case).
- `write_records` skips a record after each chunk (`start = end + 1`): 7 of 12 integration tests and the 3 `mod.rs` chunking/error-count tests go red.
- `write_chunk` corrupts `activation`: all 5 strengthened `mod.rs` tests go red (including `test_write_records_to_parquet`).
- capacity ceiling `> MAX_ARROW_OFFSET + 1`: `test_write_records_prevents_i32_overflow` and `capacity_above_arrow_row_limit_is_rejected_before_file_creation` go red.

Callers/siblings: no production signature changed. Related existing rules/tests checked: the other `mod.rs` writer tests (UUID rejection, single-record limits) are untouched and still pass.

<!-- vibe-quality-gate-skipped reason="./quality.sh passed bash-syntax, shellcheck, cargo-install-pinning, PR-summary-layout, cargo deny, build, clippy and check stages, then hit the 595s foreground cap partway through the full cargo test run; targeted tests above were run on the head. CI runs the full gate." -->
