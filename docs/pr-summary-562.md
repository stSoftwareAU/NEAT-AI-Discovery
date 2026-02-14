## Summary

Split `analysis/mod.rs` (~1,830 lines) into focused orchestration sub-modules while preserving the existing public API. Closes #562.

The file has been decomposed into three new sub-modules:

- **`orchestration.rs`** (400 lines) — `analyze_all()` entry point, deadline ordering, and optional analysis phase execution
- **`candidate_aggregation.rs`** (198 lines) — Candidate merging, neuron-to-coordinated conversion, and post-processing
- **`module_dispatch_specs.rs`** (1,085 lines) — All 32 discovery module spec builders, cross-module deduplication, and candidate clustering

The original `mod.rs` is reduced to 266 lines (module declarations, re-exports, and sub-module wiring). All sub-modules are well under the 1,500 line target.

## Evidence

This is a pure refactoring change with no UI or performance impact. All existing tests pass without modification:

- `quality.sh` passes cleanly (fmt, clippy, check, 502 unit tests + 95 integration tests, release build)
- No public API changes - all FFI consumers are unaffected
- Re-exports maintain backward compatibility at the `analysis::` level

## Test Plan

- No new tests required - this is a structure-preserving refactoring
- All 502 unit tests pass (including `mod_tests.rs` which directly tests the extracted helper functions)
- All 95 integration tests pass
- The only test file change was adding `use anyhow::Result;` to `mod_tests.rs` since the `anyhow` import is no longer brought in via `super::*` from mod.rs
