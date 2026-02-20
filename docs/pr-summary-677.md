## Summary

Introduce typed error enums using `thiserror` for domain errors instead of relying
solely on string matching for error classification. Closes #677.

### What changed

- Added `thiserror` dependency to `Cargo.toml`
- Defined `DiscoveryError` enum in `src/ffi_types/error_classification.rs` with
  typed variants: `GpuUnavailable`, `GpuDeviceLost`, `InvalidInput`, `Timeout`,
  `MemoryExhausted`, and `Io`
- Each variant maps to its `DiscoveryErrorKind` via pattern matching (`error_kind()`)
- Added `classify_anyhow_error()` which attempts downcast to `DiscoveryError`
  before falling back to string-based classification
- Added `error_fields_from_anyhow()` convenience function for FFI error handling
- Updated all FFI internal functions (`ffi_internal/`) and FFI boundary functions
  (`ffi/`) to use typed errors for known categories (e.g., JSON parse errors use
  `DiscoveryError::InvalidInput`) and `error_fields_from_anyhow` for `anyhow::Error`
- String-based `classify_error()` and `error_fields()` remain available as
  backward-compatible fallback for third-party errors

### Backward compatibility

The external JSON error response format is unchanged — `errorKind` and `retryable`
fields are serialised identically. The `classify_error()` string-based function
is preserved for errors from third-party crates (wgpu, parquet, etc.).

## Evidence

This is a backend/library change with no visual output. Evidence is provided by
the test suite:

- All 13 new integration tests pass (`tests/issue_677_typed_error_enums.rs`)
- All existing error classification tests pass (`tests/issue_651_error_classification.rs`)
- Full quality gate (`./quality.sh`) passes cleanly

## Test Plan

- Added `tests/issue_677_typed_error_enums.rs` with 13 tests covering:
  - Each `DiscoveryError` variant maps to correct `DiscoveryErrorKind`
  - `classify_anyhow_error` downcasts typed errors before string fallback
  - `classify_anyhow_error` falls back to string matching for non-typed errors
  - `DiscoveryError::Display` messages are human-readable
  - String-based `classify_error` backward compatibility preserved
  - Typed errors produce same JSON classification as string-matched errors
  - FFI response shape unchanged for existing error scenarios
