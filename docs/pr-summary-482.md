## Summary

Split the monolithic `src/analysis/synapse.rs` (4,375 lines) into focused submodules under `src/analysis/synapse/` (Issue #482). This is a pure structural refactoring — no logic changes, no public API changes.

### New Module Structure

| File | Lines | Responsibility |
|------|------:|----------------|
| `synapse/mod.rs` | 2,485 | Core pipeline (`analyze_synapses_with_cache_impl`), public API, re-exports, tests |
| `synapse/gpu_evaluation.rs` | 975 | ReLU/activation candidate evaluation, batched GPU processing |
| `synapse/scoring.rs` | 532 | Improvement calculation, saturation-aware simulation, source/target boosting |
| `synapse/filtering.rs` | 243 | Candidate truncation, deduplication, deterministic UUID generation |
| `synapse/candidate_generation.rs` | 220 | Sample locality grouping (Issue #221), ordered neuron building |
| **Total** | **4,455** | |

### Design Constraints

- NEAT-AI does not need to change — all public API functions (`analyze_synapses`, `analyze_synapses_with_cache_and_gpu_queue`) remain at the same path
- All `pub(crate)` items used by `analysis/mod.rs` and `neuron.rs` are re-exported from `synapse/mod.rs`
- All existing candidate types are reused unchanged
- All 463+ existing tests continue to pass without modification

### Key Decisions

- **`mod.rs` retains the core pipeline**: The 1,946-line `analyze_synapses_with_cache_impl` function stays in `mod.rs` because it is deeply intertwined with parallel iteration, closures, and local struct definitions that make further extraction impractical without major refactoring
- **`#[path]` directive updated**: The `implementation_tests` path attribute changed from `"implementation_tests/mod.rs"` to `"../implementation_tests/mod.rs"` to reflect the new directory depth
- **`#[cfg(test)]` imports isolated**: Test-only functions (`get_target_simulation_fn`, `build_samples`, etc.) use conditional `#[cfg(test)]` imports and re-exports to avoid unused-import warnings in non-test builds

## Evidence

This is a backend/CLI change with no visual UI. Evidence is provided via:
- All 463+ existing unit and integration tests continue to pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)
- Zero compilation warnings
- No logic changes — byte-for-byte equivalent behaviour

## Test Plan

No new tests required — this is a pure structural refactoring. The existing test suite validates that the split preserves all behaviour:

- 13 inline unit tests in `synapse/mod.rs` (Issue #413 prediction accuracy)
- 11 implementation test files in `implementation_tests/` (GPU batch, diagnostics, prediction accuracy, etc.)
- Integration tests across `tests/` directory (synapse analysis, impact discounting, target map optimisation, etc.)
