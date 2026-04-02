## Summary

The FFI boundary production code already has proper error handling — all `unwrap()` calls
at the line numbers referenced in the issue are in `#[cfg(test)]` test blocks, not in
production code. Every FFI entry point uses `catch_unwind`, validates null pointers and
UTF-8, handles JSON parse errors gracefully with `match`, and returns structured JSON
error responses with `success: false`, `error`, and `errorKind` fields.

This PR adds comprehensive malformed-input test coverage for the two internal entry points
that were not previously tested for malformed input: `analyze_parallel_internal` and
`get_calibration_summary_internal`. It also adds tests verifying that all seven internal
FFI entry points return structured error responses with `errorKind` fields for diagnostic
classification. Closes #975.

## Evidence

- All FFI entry points in `src/ffi/*.rs` use `catch_unwind` + `match` error handling
- No `unwrap()` or `expect()` on external data exists in production code under `src/ffi/`
  or `src/ffi_internal/` (only in `#[cfg(test)]` blocks)
- All quality checks pass including the new tests

## Test Plan

- Added `tests/ffi/issue_975_ffi_boundary_error_handling.rs` with 4 tests:
  - `analyze_parallel_internal_returns_error_json_for_malformed_input` — 9 malformed inputs
  - `get_calibration_summary_internal_returns_error_json_for_malformed_input` — 8 malformed inputs
  - `all_internal_entry_points_return_structured_error_for_garbage_input` — all 7 entry points
  - `error_responses_include_error_kind_field` — verifies `errorKind` in all 7 entry points
