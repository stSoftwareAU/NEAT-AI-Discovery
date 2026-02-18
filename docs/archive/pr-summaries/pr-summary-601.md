## Summary

Split `src/ffi.rs` (1,075 lines) into focused sub-modules under `src/ffi/`, grouped by FFI concern. Closes #601.

### New structure

| File | Purpose | Lines |
|------|---------|-------|
| `ffi/mod.rs` | Public API, re-exports, `free_discovery_result` | ~38 |
| `ffi/gpu.rs` | GPU probe (`check_gpu_available`) | ~62 |
| `ffi/recording.rs` | Recording — single-call and streaming (5 functions) | ~380 |
| `ffi/analysis.rs` | Analysis (`rank_focus_neurons`, `analyze_parallel`) | ~160 |
| `ffi/utilities.rs` | Utilities — merge, read, export, version (4 functions) | ~310 |

All 13 FFI symbols remain visible in the compiled `.dylib`:
`record_discovery`, `start_discovery_session`, `append_discovery_records`,
`finish_discovery_session`, `cancel_discovery_session`, `merge_discovery_parquet`,
`rank_focus_neurons`, `analyze_parallel`, `check_gpu_available`,
`get_library_version`, `read_discovery_records_ffi`,
`export_visualisation_snapshot`, `free_discovery_result`.

## Evidence

This is a pure refactoring (code moved, no logic changed). Verified by:
- All 13 `#[no_mangle]` symbols present in `nm -gU` output of release dylib
- All existing tests pass without modification
- `cargo clippy` and `./quality.sh` pass cleanly

## Test Plan

- No new tests required — this is a structural refactor with no behaviour change
- All existing integration and unit tests pass unmodified
- FFI symbol visibility verified via `nm -gU` on the release dylib
