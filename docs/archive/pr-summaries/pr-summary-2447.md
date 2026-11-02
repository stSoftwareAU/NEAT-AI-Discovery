## Summary

Reject malformed Parquet footer counts and sizes instead of treating negative values as zero, and reject file row counts that disagree with the row-group total. This keeps the memory estimator on its existing unreadable-footer warning and fallback path. Added malformed-footer regression coverage and a bounded footer fuzz target. Closes #2447.

Version bump: `Cargo.toml` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- Negative or inconsistent footer statistics can understate decoded memory, weakening the advisory pre-load estimate.
- Reject invalid signed metadata at the footer reader so the estimator uses its existing warning and compressed-size fallback.

### Essential Design Decisions

- Validate every column's `num_values`, while only accumulating values for the `errors` column.
- Sum row-group rows with checked arithmetic and compare that total with file metadata; valid zero-row files remain accepted.
- The footer fuzz target caps input at 65,536 bytes and replays the patched negative-row-count seed.

### Undiscoverable Facts

- The baseline regression run returned `num_rows: 0` for the negative file count and `num_rows: 1` for the understated count; both files still decoded 50 rows.
- The isolated fuzz lockfile was refreshed to versions already present in the root lockfile so the fuzz target can run with `--locked`.

## Evidence

- `tests/issue_2447_parquet_footer_stats.rs` builds real 50-row Parquet files, patches compact-Thrift footer fields, checks exact valid statistics, and asserts field-specific errors.
- The original trigger is closed: negative counts and sizes fail conversion, and understated or overstated file row counts fail the row-group consistency check; equivalent malformed footer values cannot be accepted as zero.
- `read_footer_stats` is used by `estimate_parquet_in_memory_bytes`; its existing unreadable-footer warning and compressed-size fallback remain unchanged.
- Footer fuzz target and seed: `fuzz/fuzz_targets/fuzz_parquet_footer_stats.rs` and `fuzz/seeds/parquet_footer_stats/negative_file_num_rows`.
- `cargo check --manifest-path fuzz/Cargo.toml --locked --offline --bin fuzz_parquet_footer_stats` passed. A bounded direct libFuzzer smoke run completed 100 executions and replayed the seed; the stable binary reported that sanitizer coverage callbacks were unavailable, so this was not a coverage-guided `cargo-fuzz` run.
- `gh issue view 2447 --json title,labels` confirmed the issue title and `bug` / `security` labels used for this record.

## Reproduction

- **symptom** — negative footer values became zero and an understated file row count was accepted, under-reporting the projected decoded size.
- **status** — `verified` — `negative_file_num_rows_is_rejected` and `understated_file_num_rows_is_rejected` failed against unchanged production code and passed after the fix.
- **regression test** — `tests/issue_2447_parquet_footer_stats.rs::negative_file_num_rows_is_rejected`

## Test Plan

- Added `tests/issue_2447_parquet_footer_stats.rs::valid_footer_stats_match_written_records` — passed; confirms 50 rows and 100 error values.
- Added `tests/issue_2447_parquet_footer_stats.rs::negative_file_num_rows_is_rejected` — baseline failed because the negative value was reported as zero; passed after the fix.
- Added `tests/issue_2447_parquet_footer_stats.rs::understated_file_num_rows_is_rejected` — baseline accepted the count of one; passed after the fix.
- Added `tests/issue_2447_parquet_footer_stats.rs::negative_row_group_num_rows_is_rejected` — passed; identifies `num_rows`.
- Added `tests/issue_2447_parquet_footer_stats.rs::negative_total_byte_size_is_rejected` — passed; identifies `total_byte_size`.
- Added `tests/issue_2447_parquet_footer_stats.rs::negative_num_values_is_rejected` and `tests/issue_2447_parquet_footer_stats.rs::negative_non_error_column_num_values_is_rejected` — passed; both reject negative `num_values`, including on a non-errors column.
- Added `tests/issue_2447_parquet_footer_stats.rs::fuzz_seed_negative_file_num_rows_is_rejected` — passed; the checked-in fuzz seed is rejected with a `num_rows` diagnostic.
- Added `src/parquet_format/footer.rs::tests::row_group_row_sum_rejects_overflow` and `src/parquet_format/footer.rs::tests::empty_footer_stats_remain_valid` — passed in the full test run.
- Mutation checks: reverting each of the file-row conversion, file/group consistency, row-group conversion, byte-size conversion, errors-column value conversion, and non-errors-column validation made its corresponding regression test fail. Replacing checked row summation with saturation made `row_group_row_sum_rejects_overflow` fail.
- `tests/issue_1869_parquet_decode_bound.rs` — 11 passed.
- `timeout 900s ./quality.sh < /dev/null` — passed, including formatting, Clippy, type checks, all library and integration tests, docs, and release build.
- Documentation sweep covered the footer reader and its estimate/fallback behavior; updated `README.md`'s Fuzz Testing section and `docs/CACHE_TUNING.md`'s memory-estimate section.
- No existing test assertions were removed or changed.

## Remaining Work and Risk

Coverage-guided `cargo-fuzz` under nightly was not run because nightly tooling was unavailable. The bounded stable libFuzzer smoke run completed, but without coverage instrumentation. No other known work remains.
