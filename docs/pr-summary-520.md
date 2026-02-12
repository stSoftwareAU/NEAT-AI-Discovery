## Summary

Split the monolithic `gpu/analyzer.rs` (2,625 lines) into per-evaluation modules, reducing the core file to 492 lines. Closes #520.

Each evaluation pipeline (builder + batch method) is now in its own self-contained module:

| Module | Lines | Contents |
|--------|------:|---------|
| `analyzer.rs` | 492 | Core `GpuAnalyzer` struct, `GpuEvaluator` trait, initialisation, helpers |
| `helpful_evaluation.rs` | 596 | Helpful synapse pipeline + batch evaluation + merge results |
| `harmful_evaluation.rs` | 528 | Harmful synapse pipeline + batch evaluation |
| `relu_evaluation.rs` | 272 | ReLU activation pipeline + evaluation |
| `activation_evaluation.rs` | 474 | Activation function pipeline + single/batched evaluation |
| `bias_evaluation.rs` | 277 | Bias pipeline + grid search evaluation |

This follows the project's established pattern of splitting large files (synapse/ in #482, focus/ in #491, lib.rs in #519).

## Evidence

This is a pure refactoring change with no visual or behavioural changes. All existing tests pass unchanged — `./quality.sh` passes cleanly (474 unit tests + 97 integration test files).

## Test Plan

- All existing GPU evaluation tests pass unchanged (no tests added, removed, or modified)
- `merge_batch_results` tests moved from `analyzer.rs` to `helpful_evaluation.rs` (where the function now lives)
- `./quality.sh` passes: fmt, clippy, check, all tests, release build
