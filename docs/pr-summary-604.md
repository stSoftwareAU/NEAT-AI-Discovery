## Summary

Split `record.rs` into focused sub-modules under `src/record/` for better separation of concerns. Closes #604.

- `record/mod.rs` — public API, `RecordResult` type, orchestration of the recording pipeline
- `record/validation.rs` — input validation and observation index resolution
- `record/processing.rs` — record building from training data and Parquet writing

Public API remains unchanged (`record::record_discovery_data`, `record::RecordResult`).

## Evidence

This is a pure refactoring with no UI or performance changes. All existing tests pass without modification. `quality.sh` passes cleanly.

## Test Plan

- All 11 existing `record::tests::*` unit tests pass unchanged
- All 509 unit tests and 97 integration test suites pass
- `cargo clippy`, `cargo fmt`, and `cargo check` pass cleanly
- `cargo build --release` succeeds
