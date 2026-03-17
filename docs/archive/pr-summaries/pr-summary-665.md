## Summary

Split the monolithic `src/ffi_internal.rs` (964 lines) into a `src/ffi_internal/` directory with per-category sub-modules mirroring the existing `src/ffi/` structure. Closes #665.

### New structure

| Module | Functions | Lines |
|--------|-----------|-------|
| `ffi_internal/recording.rs` | `record_discovery_internal` | 63 |
| `ffi_internal/analysis.rs` | `analyze_parallel_internal`, `build_analyze_all_input_from_parallel`, `rank_focus_neurons_internal`, `get_calibration_summary_internal` | 260 |
| `ffi_internal/gpu.rs` | `check_gpu_available_internal`, `get_library_version_internal` | 48 |
| `ffi_internal/utilities.rs` | `merge_discovery_parquet_internal`, `export_visualisation_snapshot_internal`, `read_discovery_records` | 183 |
| `ffi_internal/mod.rs` | Re-exports + all unit tests | ~290 |

All public API signatures remain unchanged. No individual file exceeds ~300 lines — well under the ~1,500 line target.

## Evidence

This is a pure refactoring task with no UI or performance changes. Evidence:
- `./quality.sh` passes cleanly (fmt, clippy, check, tests, doc build, release build)
- All existing tests pass without modification — they are preserved in `ffi_internal/mod.rs`

## Test Plan

- All existing unit tests moved from the old `ffi_internal.rs` `#[cfg(test)]` module to `ffi_internal/mod.rs` without modification
- No tests were added, removed, or modified — this is a structure-only refactoring
- All integration tests continue to work via the `pub use ffi_internal::*` re-export in `lib.rs`
