## Summary

Refactor the two largest files in `src/analysis/synapse/` for maintainability,
following the Single Responsibility Principle. Closes #802.

### gpu_evaluation.rs (988 → 244 lines)

- Extracted `evaluate_relu_candidates_split` and `SplitReluResult` into `relu_evaluation.rs` (168 lines)
- Extracted `evaluate_activation_for_subset` and `SubsetEvalParams` into `activation_subset_evaluation.rs` (204 lines)
- Extracted `evaluate_activation_candidate` and `ActivationEvalParams` into `activation_evaluation.rs` (434 lines)
- Kept `evaluate_all_activation_specs_batched` and sequential fallback in `gpu_evaluation.rs` as thin orchestration with re-exports

### mod.rs (945 → 155 lines)

- Extracted `analyze_synapses_with_cache_impl` into `orchestration.rs` (152 lines)
- Extracted `AtomicMetadata` and `MergedResults` into `metadata.rs` (99 lines)
- Extracted `FinaliseParams` and `finalise_synapse_results` into `results.rs` (99 lines)
- Extracted unit tests into `tests.rs` (493 lines)
- Kept `mod.rs` as a thin re-export layer with public API functions

### File sizes after refactoring

All files that were refactored are now under 500 lines. The pre-existing
`scoring.rs` (690 lines) was not in scope for this issue.

## Evidence

- All 577+ existing tests pass without modification
- `quality.sh` passes cleanly (fmt, clippy, check, tests, doc, release build)
- Public API unchanged — all re-exports preserved in `mod.rs`

## Test Plan

- No new tests needed — this is a pure refactoring with no behaviour changes
- All existing tests verified passing: `cargo test --lib --tests --all-features -- --test-threads=2`
- All tests in `analysis::synapse::tests` and `analysis::synapse::implementation_tests` pass
