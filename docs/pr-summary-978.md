## Summary

Extract shared GPU compute pipeline builder to eliminate duplicated boilerplate across five GPU evaluation modules. Closes #978.

A new `pipeline_builder.rs` module provides a reusable `build_compute_pipeline()` function that accepts shader source, binding specifications, and labels — replacing ~200 lines of near-identical pipeline construction code that was previously copy-pasted across `helpful_evaluation.rs`, `harmful_evaluation.rs`, `activation_evaluation.rs`, `relu_evaluation.rs`, and `bias_evaluation.rs`.

Two predefined binding layouts are provided:
- `STANDARD_BINDINGS` (3 bindings: storage RO, storage RW, uniform) — used by 7 of 8 pipelines
- `BIAS_BINDINGS` (4 bindings: 2x storage RO, storage RW, uniform) — used by the bias pipeline

No functional changes — all GPU compute results remain identical.

## Evidence

- All 158 tests pass
- All quality checks pass (clippy, fmt, doc build, release build)
- The refactored pipeline builders delegate to the shared helper with identical labels, shader sources, and binding types

## Test Plan

- Added unit tests in `pipeline_builder.rs` verifying:
  - `STANDARD_BINDINGS` has correct count (3) and binding types
  - `BIAS_BINDINGS` has correct count (4) and binding types
- All existing tests continue to pass unchanged (158 tests)
