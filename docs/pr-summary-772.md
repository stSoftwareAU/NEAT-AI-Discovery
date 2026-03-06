## Summary

Replace panicking `expect()` and `unwrap()` calls in GPU evaluation and FFI paths with proper error handling. Closes #772.

### GPU evaluation changes (6 sites)
- Replaced `.expect("GPU device must be available...")` with `.context("GPU device unavailable for ... analysis")?` in all 5 GPU evaluation files (helpful, harmful, ReLU, bias, activation — 6 call sites total).
- These now return `Err` through the existing `Result` return type instead of panicking.

### FFI serialisation changes (~40 sites)
- Created `src/ffi/helpers.rs` with three shared helpers:
  - `to_ffi_json<T: Serialize>(&T) -> *mut c_char` — serialises and converts to C string without panicking
  - `ffi_error_literal(&str) -> *mut c_char` — converts static error strings safely
  - `panic_to_ffi_json(Box<dyn Any + Send>) -> *mut c_char` — builds panic error response safely
- Replaced all `serde_json::to_string(&output).unwrap()` + `CString::new(json).unwrap().into_raw()` patterns with `to_ffi_json(&output)`
- Replaced all `CString::new(error_literal).unwrap().into_raw()` patterns with `ffi_error_literal(error_literal)`
- Replaced all duplicated panic handler blocks with `panic_to_ffi_json`
- All FFI functions retain `panic::catch_unwind` at the boundary as a safety net

### Result
- Net deletion of ~280 lines of duplicated boilerplate
- Zero `unwrap()` calls in non-test FFI code
- Zero `expect()` calls for GPU device availability in evaluation code
- All existing tests pass, `quality.sh` passes

## Evidence

This is a backend/library change with no visual output. Evidence is provided by:
- All 6 unit tests in `src/ffi/helpers.rs` covering `to_ffi_json`, `ffi_error_literal`, `panic_to_ffi_json`
- All existing integration and unit tests pass (verified via `quality.sh`)

## Test Plan

- Added 6 unit tests in `src/ffi/helpers.rs`:
  - `test_to_ffi_json_success` — verifies successful serialisation round-trip
  - `test_ffi_error_literal_returns_valid_string` — verifies literal conversion
  - `test_panic_to_ffi_json_with_string_message` — String panic message
  - `test_panic_to_ffi_json_with_str_message` — &str panic message
  - `test_panic_to_ffi_json_with_unknown_type` — non-string panic type
  - `test_panic_to_ffi_json_escapes_special_chars` — JSON escaping of special characters
- All existing tests continue to pass unchanged
