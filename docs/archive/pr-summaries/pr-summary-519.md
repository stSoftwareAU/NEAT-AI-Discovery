## Summary

Split `src/lib.rs` (3,042 lines) into four focused modules following the
Single Responsibility Principle. Closes #519.

| File | Lines | Purpose |
|------|-------|---------|
| `src/lib.rs` | 49 | Module declarations, re-exports, version init |
| `src/ffi_types.rs` | 1,187 | JSON request/response structs for the FFI boundary |
| `src/ffi.rs` | 1,075 | `#[no_mangle] pub extern "C"` FFI entry points |
| `src/ffi_internal.rs` | 829 | Internal business-logic functions + unit tests |

**No public API changes.** All types and functions are re-exported from the
crate root via `pub use ffi_types::*` and `pub use ffi_internal::*`, so
existing `crate::TypeName` and `neat_ai_discovery::TypeName` paths continue
to resolve exactly as before.

## Evidence

This is a pure refactoring with no behavioural changes — no UI or performance
impact. Evidence of correctness:

- `./quality.sh` passes: fmt, clippy, check, all 474 unit tests, all 97
  integration test files, release build
- All FFI symbols remain exported (verified by successful build of the
  `cdylib` target)
- `AGENTS.md` source layout updated to reflect the new file structure

## Test Plan

- No tests were added, modified, or removed
- All 474 existing unit tests pass
- All existing integration tests pass
- The existing test suite comprehensively covers the FFI boundary types,
  internal functions, and `extern "C"` entry points that were moved
