## Summary

Split `analysis/neuron.rs` (~928 lines) into focused sub-modules under `analysis/neuron/`. Closes #598.

The original monolithic file contained the entire neuron analysis pipeline. It has been split by logical concern:

- **`neuron/mod.rs`** (344 lines) — Public API (`analyze_neurons`, `analyze_neurons_with_cache`), orchestration, and the parallel processing loop
- **`neuron/preparation.rs`** (445 lines) — Focus target filtering, neuron type maps, source ordering, and record loading
- **`neuron/evaluation.rs`** (206 lines) — GPU-based candidate evaluation (ReLU split, activation specs batched)
- **`neuron/post_processing.rs`** (184 lines) — Impact discounting, pessimism discount, sorting, filtering, and result assembly

All files are well under the ~1,500 line target. The public API remains unchanged — `analyze_neurons` and `analyze_neurons_with_cache` are still exported at `analysis::neuron::`.

## Evidence

This is a purely structural refactoring with no UI changes. Evidence is provided by:
- All existing tests pass without modification (`cargo test` — 0 failures)
- `cargo clippy` passes cleanly
- `./quality.sh` passes all checks including release build

## Test Plan

- No new tests needed — this is a pure structural refactoring
- All existing tests pass without modification, confirming backward compatibility
- Verified via `./quality.sh` (fmt, clippy, check, test, release build)
