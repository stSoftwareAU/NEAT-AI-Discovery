## Summary

Split `src/analysis/samples.rs` (1,323 lines) into a directory with focused sub-modules. Closes #597.

The monolithic file is now organised as:

| File | Lines | Responsibility |
|------|-------|---------------|
| `samples/mod.rs` | ~370 | Public API, re-exports, core `HelpfulSample` type, unit tests |
| `samples/gpu_types.rs` | ~170 | GPU-compatible `#[repr(C)]` data formats (bytemuck Pod/Zeroable) |
| `samples/statistics.rs` | ~310 | `NeuronStats`, `HelpfulStats`, `HarmfulStats`, `ReluStats` |
| `samples/thresholds.rs` | ~200 | Threshold computation, source variance analysis |

Public API is unchanged — all items are re-exported from `samples/mod.rs`, so existing `use crate::analysis::samples::*` and `use crate::analysis::*` paths continue to work.

## Evidence

This is a backend code reorganisation with no UI or performance changes.

- `./quality.sh` passes cleanly (fmt, clippy, check, 509 unit tests, all integration tests, release build)
- No test modifications — all existing tests pass without changes
- No public API changes — all re-exports preserved

## Test Plan

- All 509 existing unit tests pass unchanged
- All integration tests pass unchanged
- Verified via `./quality.sh` (includes fmt, clippy, check, test, release build)
