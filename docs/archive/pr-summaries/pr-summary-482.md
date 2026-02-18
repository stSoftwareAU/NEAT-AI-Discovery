## Summary

Further split the synapse analysis submodules to bring all files under the ~1,500-line target (Issue #482). The previous PR (#495) split the monolithic `synapse.rs` into submodules but left `mod.rs` at 2,485 lines. This PR extracts three new focused modules, reducing `mod.rs` from 2,485 to 860 lines — a 65% reduction.

### New Module Structure

| File | Lines | Responsibility |
|------|------:|----------------|
| `synapse/mod.rs` | 860 | Orchestration, public API, re-exports, tests |
| `synapse/target_analysis.rs` | 1,020 | Per-target analysis loop (helpful, harmful, coordinated) |
| `synapse/gpu_evaluation.rs` | 975 | ReLU/activation candidate evaluation, batched GPU processing |
| `synapse/scoring.rs` | 532 | Improvement calculation, saturation-aware simulation, boosting |
| `synapse/structural_patterns.rs` | 411 | Noisy vs trusted input folding, collapse hidden neurons |
| `synapse/post_processing.rs` | 315 | Impact discounting, sorting, diversification, metadata |
| `synapse/filtering.rs` | 243 | Candidate truncation, deduplication |
| `synapse/candidate_generation.rs` | 220 | Sample locality grouping, ordered neuron building |
| **Total** | **4,576** | All files under 1,500-line target |

### Extractions in this PR

1. **`target_analysis.rs`** — Extracted the 1,200-line per-target parallel loop body into `analyse_single_target()`. Uses a `TargetAnalysisContext` struct to share pre-computed data across targets, and returns `TargetAnalysisResults` for the orchestrator to merge.

2. **`structural_patterns.rs`** — Extracted coordinated structural discovery:
   - Noisy vs trusted input folding (Issue #165)
   - Collapse 1-in/1-out hidden neurons into direct synapses (Issue #425)

3. **`post_processing.rs`** — Extracted all post-analysis processing:
   - Impact-based discounting for helpful, harmful, and coordinated candidates
   - Source-type and target-type boosting (Issues #467, #468)
   - Sorting, diversification, truncation
   - Metadata assembly via `MetadataParams` struct

### Design Constraints

- No changes to the public API — NEAT-AI does not need to change
- All `pub(crate)` items used by `analysis/mod.rs` and `neuron.rs` are re-exported from `synapse/mod.rs`
- All existing candidate types are reused unchanged
- All existing tests continue to pass without modification
- Updated AGENTS.md source layout to reflect new module structure

## Evidence

This is a backend/CLI change with no visual UI. Evidence is provided via:
- All existing unit and integration tests continue to pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)
- Zero compilation warnings
- No logic changes — behaviour is preserved

## Test Plan

No new tests required — this is a pure structural refactoring. The existing test suite validates that the split preserves all behaviour:

- 13 inline unit tests in `synapse/mod.rs` (Issue #413 prediction accuracy)
- 11 implementation test files in `implementation_tests/` (GPU batch, diagnostics, prediction accuracy, etc.)
- Integration tests across `tests/` directory (synapse analysis, impact discounting, target map optimisation, etc.)
