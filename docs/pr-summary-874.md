## Summary

Refactored three large source files (each >600 lines) into focused submodules following the Single Responsibility Principle. Closes #874.

### Files split

| Original file | Submodules | Lines |
|---|---|---|
| `src/analysis/shared.rs` (661 lines) | `shared/timing.rs` (300), `shared/metadata.rs` (253), `shared/gpu_info.rs` (124), `shared/mod.rs` (16) | |
| `src/observability.rs` (632 lines) | `observability/phase_timer.rs` (104), `observability/gpu_metrics.rs` (154), `observability/profile.rs` (219), `observability/mod.rs` (165) | |
| `src/ffi_types/responses.rs` (657 lines) | `responses/analysis.rs` (250), `responses/gpu.rs` (139), `responses/export.rs` (112), `responses/mod.rs` (208) | |

### Key properties

- **No public API changes** — all types remain accessible via the same import paths through `mod.rs` re-exports
- **No new files exceed 300 lines**
- **All existing tests pass without modification**
- **Module-level `//!` doc comments** added to each new submodule

## Evidence

- `./quality.sh` passes cleanly (fmt, clippy, check, tests, doc build, release build)
- All 12 new submodule files are under 300 lines
- Backward-compatible imports verified by dedicated integration test

## Test Plan

- Added `tests/infrastructure/issue_874_module_split_backward_compat.rs` — verifies all public types from all three split modules remain importable via the original paths
- All existing tests pass unmodified (no import path changes needed)
