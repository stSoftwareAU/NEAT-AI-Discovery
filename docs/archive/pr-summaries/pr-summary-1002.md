## Summary

Run synapse and neuron analyses concurrently via `rayon::join` with a shared
`GpuWorkQueue`, reducing wall-clock time when both analyses are enabled. Closes #1002.

### Changes

- **`src/analysis/orchestration.rs`**: `dispatch_analyses()` now uses `rayon::join`
  when both analyses are enabled. A shared `GpuWorkQueue` is created once in
  `analyze_all()` and passed to both analyses. Removed `choose_deadline_order_synapse_first()`
  and the randomised ordering logic — both analyses now get the full time budget.
- **`src/analysis/neuron/mod.rs`**: Added `analyze_neurons_with_cache_and_gpu_queue()`
  to accept an externally created GPU queue, matching the existing synapse API.
  `analyze_neurons_with_cache()` now delegates to this new function.
- **`src/analysis/mod.rs`**: Re-exported `analyze_neurons_with_cache_and_gpu_queue`
  for use by benchmarks and external tests.
- **Removed tests**: Three `choose_deadline_order_synapse_first` unit tests
  removed as the function was deleted (no longer needed with concurrent execution).

## Evidence

- `cargo build` — compiles cleanly
- `cargo fmt --all -- --check` — no formatting issues
- `cargo clippy --all-targets --all-features -- -D warnings` — no warnings
- `cargo test --lib --tests --all-features -- --test-threads=2` — 158 tests pass
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` — documentation builds cleanly
- `cargo build --release --lib` — release build succeeds

## Test Plan

- Added `tests/infrastructure/issue_1002_concurrent_analysis.rs` with tests verifying:
  - `analyze_neurons_with_cache_and_gpu_queue` is publicly accessible with correct signature
  - `analyze_synapses_with_cache_and_gpu_queue` remains publicly accessible
  - Both analysis functions accept the same `Arc<GpuWorkQueue>` type (shared queue contract)
- All existing integration tests continue to pass (158/158)
- Removed 3 tests for `choose_deadline_order_synapse_first` (function deleted)
