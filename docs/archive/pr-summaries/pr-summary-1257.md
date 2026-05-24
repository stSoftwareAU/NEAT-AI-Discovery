## Summary

Added one-line `///` item-level summaries to the public FFI request/response
structs and enums that ship with documented fields but no item-level rustdoc
header. `cargo doc` and IDE hover now show a meaningful description for each
type instead of a bare name. Closes #1257.

Items documented:

- `src/ffi_types/responses/analysis.rs` — `AnalyzeParallelOutput`,
  `SynapseDiagnosticJson`, `SynapseDiagnosticReasonJson`,
  `SynapseDiagnosticDetailJson`, `NeuronDiagnosticJson`,
  `NeuronDiagnosticReasonJson`, `NeuronDiagnosticDetailJson`.
- `src/ffi_types/responses/export.rs` — `MergeParquetOutput`.
- `src/ffi_types/responses/mod.rs` — `GetVersionOutput`,
  `RankFocusNeuronsOutput`.
- `src/ffi_types/responses/gpu.rs` — `CheckGpuOutput`.
- `src/ffi_types/mod.rs` — `NeuronJson`, `SynapseJson`, `NeuronStatsJson`.
- `src/ffi_types/requests.rs` — `AnalyzeParallelInput`.
- `src/parquet_format/writer.rs` — `ParquetRecordWriter` and its `pub fn new`.

This is a pure doc-only change — no behaviour or wire format changes.

## Evidence

CLI/library change with no UI to screenshot. Verification is via the docs
build, which compiles all `///` text as rustdoc:

- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` succeeds, proving every
  added summary parses cleanly and all intra-doc links resolve.
- `./quality.sh` passed end-to-end: `cargo build`, `cargo clippy
  --all-targets --all-features -- -D warnings`, `cargo check`, the full test
  suite (`cargo test --lib --tests --all-features`), `cargo doc`, and the
  release build all completed successfully.

## Test Plan

- Existing unit and integration tests continue to pass (`cargo test --lib
  --tests --all-features`). No tests were modified — the change is
  documentation-only and is validated by the doc build inside `quality.sh`.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` exercises every added
  `///` block; any malformed rustdoc or broken intra-doc link would fail it.
