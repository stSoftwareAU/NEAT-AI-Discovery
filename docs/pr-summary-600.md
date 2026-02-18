## Summary

Split `parquet_format.rs` (~1,085 lines) into focused sub-modules under `src/parquet_format/`. Closes #600.

The monolithic file has been split into:
- `parquet_format/mod.rs` — Public API, re-exports for backward compatibility
- `parquet_format/schema.rs` — Schema definitions (`create_schema`), validation (`validate_neuron_uuid`), and shared utilities (`truncate_utf8`)
- `parquet_format/writer.rs` — Parquet writing and serialisation (`ParquetRecordWriter`, `write_records_to_parquet`, `merge_parquet_files`)
- `parquet_format/reader.rs` — Parquet reading and deserialisation (`read_records_from_parquet`, `read_all_records_grouped_by_neuron`, `read_records_from_parquet_with_limit`)

All public API items are re-exported from `parquet_format/mod.rs`, so no downstream code changes are required.

## Evidence

This is a pure refactoring change with no visual output. All existing tests pass unchanged, and `quality.sh` passes cleanly (fmt, clippy, check, test, release build).

## Test Plan

- All 20 existing unit tests in `parquet_format::tests` pass without modification
- All integration tests referencing `parquet_format::` imports continue to work via re-exports
- `cargo clippy` passes with no warnings
- `./quality.sh` passes cleanly
