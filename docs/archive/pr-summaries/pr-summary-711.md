## Summary

Improve FFI safety by replacing all 12 `#[allow(clippy::not_unsafe_ptr_arg_deref)]`
suppressions across `src/ffi/` with properly marked `unsafe extern "C"` functions
and `// SAFETY:` comments on every `unsafe` block. The `ffi` module is now public
so integration tests can exercise the FFI boundary directly. Closes #711.

## Changes

- **`src/ffi/analysis.rs`**: Removed 2 clippy suppressions; marked `rank_focus_neurons`
  and `analyze_parallel` as `unsafe extern "C"` with `# Safety` doc comments and
  `// SAFETY:` annotations on internal unsafe blocks.
- **`src/ffi/recording.rs`**: Removed 5 clippy suppressions; marked `record_discovery`,
  `start_discovery_session`, `append_discovery_records`, `finish_discovery_session`, and
  `cancel_discovery_session` as `unsafe extern "C"` with safety documentation.
- **`src/ffi/mod.rs`**: Removed 1 clippy suppression; marked `free_discovery_result` as
  `unsafe extern "C"` with safety documentation.
- **`src/ffi/utilities.rs`**: Removed 4 clippy suppressions; marked `merge_discovery_parquet`,
  `read_discovery_records_ffi`, `export_visualisation_snapshot`, and
  `get_calibration_summary` as `unsafe extern "C"` with safety documentation.
- **`src/lib.rs`**: Made `ffi` module public to allow integration tests to call FFI
  functions directly.

## Evidence

This is a backend/FFI safety change with no visual output. Evidence is provided
by the test results below and a clean `quality.sh` run (fmt, clippy, check, tests,
doc build, release build all pass).

## Test Plan

- Added `tests/issue_711_ffi_safety_unsafe_markers.rs` with 16 tests:
  - 12 null-pointer tests — one for each FFI function that accepts a raw pointer,
    verifying it returns a well-formed JSON error response with `success: false`.
  - 4 invalid JSON tests — verifying graceful error handling for malformed input.
