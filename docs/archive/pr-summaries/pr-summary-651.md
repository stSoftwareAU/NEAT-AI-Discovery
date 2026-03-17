## Summary

Add structured error classification and retry strategy for transient failures. Closes #651.

All FFI error responses now include two new optional fields:
- `errorKind` — classifies the error (e.g., `gpu_transient`, `data_validation`, `timeout`, `memory_exhausted`, `io_error`, `gpu_permanent`, `internal_panic`, `unknown`)
- `retryable` — boolean indicating whether the controller should retry the operation

These fields are additive and backward-compatible: existing callers that do not read them are unaffected, while new callers can use them to make informed retry decisions.

### Error classification categories

| Kind | Retryable | Examples |
|------|-----------|----------|
| `gpu_transient` | Yes | Device lost, driver unresponsive, command buffer overflow |
| `gpu_permanent` | No | No GPU available, unsupported hardware |
| `data_validation` | No | Invalid JSON, null input, missing fields |
| `timeout` | Yes | Deadline exceeded, operation timed out |
| `memory_exhausted` | Yes | Out of memory, allocation failed |
| `io_error` | Yes | Parquet read failure, file not found |
| `internal_panic` | No | Caught panic at FFI boundary |
| `unknown` | No | Unclassified errors |

## Evidence

This is a backend/FFI change with no visual output. Correctness is verified through 26 integration tests and all existing unit tests in the `error_classification` module.

## Test Plan

- Added `tests/issue_651_error_classification.rs` — 26 integration tests covering:
  - All 8 error kind classifications
  - Retryable/non-retryable correctness for each kind
  - `error_fields()` and `no_error_fields()` helper functions
  - JSON serialisation of error kinds (snake_case format)
  - FFI response shape verification (error fields present on failure, absent on success)
  - Backward compatibility (existing `success` and `error` fields preserved)
  - Case-insensitive error message classification
- Added unit tests in `src/ffi_types/error_classification.rs`
- Updated `tests/issue_337_candidate_type_contract.rs` to include new fields
- All existing tests continue to pass
- `./quality.sh` passes cleanly
