## Summary

Add defence-in-depth integrity validation to the Parquet reading path. Corrupted, truncated, or schema-mismatched files now produce clear `DiscoveryError::Io` errors instead of panicking with unhelpful index-out-of-bounds messages. Incomplete `.parquet.tmp` files are detected and rejected with a warning. Closes #1085.

### Changes

- **Schema validation on file open** (`src/parquet_format/reader.rs`): Added `validate_parquet_schema()` that checks column count, column indices, and column names match the expected discovery schema before applying projection masks. This prevents panics in the `ProjectionMask::roots` call when files have mismatched schemas.
- **`.parquet.tmp` rejection** (`src/parquet_format/reader.rs`): `open_parquet_file()` now detects `.parquet.tmp` suffix and returns a typed `DiscoveryError::Io` with a clear message about incomplete writes, with a `tracing::warn` log.
- **Graceful corruption handling**: Truncated and corrupted files already produced errors from the parquet library, but schema mismatches caused panics. All paths now produce `Result::Err` rather than panicking.
- **No performance regression**: Validation is metadata-only (checks Arrow schema fields), adding negligible overhead to valid file reads.

## Evidence

This is a backend/library change with no UI. Evidence is provided by the test suite:

- `test_truncated_parquet_file_returns_error_not_panic` — truncated file produces error
- `test_empty_file_returns_error_not_panic` — empty file produces error
- `test_random_bytes_file_returns_error_not_panic` — garbage bytes produce error
- `test_schema_mismatch_returns_clear_error` — wrong schema produces error mentioning schema
- `test_schema_mismatch_with_filtered_read_returns_clear_error` — wrong column types produce error
- `test_tmp_parquet_file_is_rejected_with_warning` — `.parquet.tmp` file rejected with clear message
- `test_tmp_parquet_file_detected_by_all_read_functions` — all read functions reject `.tmp` files
- `test_valid_parquet_file_still_reads_correctly` — no regression on valid files

## Test Plan

- Added `tests/parquet_integrity_validation.rs` with 9 integration tests covering all acceptance criteria
- All existing tests continue to pass (171 integration tests + existing parquet tests)
- `./quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)
