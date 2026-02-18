## Summary

Add fuzz testing for the FFI JSON boundary using cargo-fuzz. This introduces
two fuzz targets that exercise deserialisation and business-logic entry points
with arbitrary inputs, plus 12 deterministic integration tests covering
malformed, truncated, and extreme-value JSON edge cases.

Both fuzz targets ran for 60+ seconds without discovering any panics
(679,549 + 732,451 = 1,411,000 total iterations), confirming the existing
`panic::catch_unwind` + `serde_json` error handling is robust.

Closes #574.

## Evidence

This is a purely additive testing infrastructure change with no UI or
performance impact. Evidence is from test and fuzz execution:

- `fuzz_ffi_deserialisation`: 679,549 runs in 61 seconds, zero crashes
- `fuzz_ffi_entry_points`: 732,451 runs in 61 seconds, zero crashes
- All 12 new integration tests pass
- All existing tests pass (`cargo test --test-threads=1`)
- `quality.sh` passes cleanly

## Test Plan

- Added `tests/issue_574_ffi_json_fuzz_edge_cases.rs` with 12 tests:
  - `deserialise_malformed_json_never_panics` — 30+ malformed inputs across all FFI types
  - `internal_entry_points_return_error_json_for_malformed_input` — verifies error JSON structure
  - `internal_entry_points_handle_extreme_values_gracefully` — extreme values, NaN, negatives
  - `truncated_json_never_panics` — byte-by-byte truncation of valid JSON
  - `long_string_values_do_not_panic` — 100KB string values
  - `extreme_float_values_do_not_panic` — f32 boundary values
  - `merge_parquet_edge_cases_do_not_panic` — empty/nonexistent file paths
  - `rank_focus_neurons_nonexistent_file_does_not_panic`
  - `analyze_parallel_nonexistent_file_does_not_panic`
  - `export_visualisation_nonexistent_file_does_not_panic`
  - `creature_json_defaults_applied_correctly` — serde default values
  - `creature_json_round_trip` — serialise/deserialise consistency
- Added `fuzz/fuzz_targets/fuzz_ffi_deserialisation.rs` — fuzzes all FFI input type deserialisation
- Added `fuzz/fuzz_targets/fuzz_ffi_entry_points.rs` — fuzzes internal business-logic functions
- Updated README.md with fuzz testing documentation
