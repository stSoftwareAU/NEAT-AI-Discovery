## Summary

Overlap CPU analysis with GPU computation using async pipeline. Closes #568.

The synapse analysis pipeline previously executed helpful GPU evaluation and harmful
sample preparation sequentially. This change introduces a non-blocking GPU submission
pattern (`GpuFuture`) that allows harmful sample preparation (cache lookups and sample
building — pure CPU work) to run while the GPU processes the helpful batch.

### Changes

- **`src/analysis/gpu/queue.rs`**: Added `GpuFuture<T>` type and `submit_helpful_batch()`
  method for non-blocking GPU work submission
- **`src/analysis/synapse/target_analysis.rs`**: Restructured per-target analysis to overlap
  helpful GPU evaluation with harmful sample preparation:
  - `submit_helpful_gpu_work()` — submits helpful batch without blocking
  - `prepare_harmful_samples()` — builds harmful samples on CPU (overlaps with GPU)
  - `collect_and_process_helpful_results()` — collects GPU results and processes
  - `process_harmful_batch_from_prepared()` — submits and processes pre-built harmful data
- **`benches/async_pipeline.rs`**: New benchmark suite measuring pipeline throughput
- **`Cargo.toml`**: Added `async_pipeline` bench entry
- **`tests/issue_568_async_pipeline.rs`**: Integration tests for correctness and determinism
- **`tests/issue_576_benchmark_regression_tracking.rs`**: Updated expected benchmark list

### No public API changes

The `GpuFuture` type is `pub(crate)` only. All public APIs remain unchanged.

## Evidence

### Benchmark Results

| Workload | Baseline | After | Change | Significant |
|----------|----------|-------|--------|-------------|
| 10 hidden, 200 records | 748.20 ms | 710.99 ms | **-5.0%** | Yes (p=0.00) |
| 30 hidden, 300 records | 1893.9 ms | 1908.6 ms | +0.8% | No (p=0.45) |

The improvement is most visible with fewer focus neurons (10 hidden) where Rayon
parallelism alone doesn't fully saturate the CPU — intra-neuron CPU/GPU overlap
provides additional benefit. With many focus neurons (30 hidden), Rayon already
provides natural overlap across neurons, so the additional intra-neuron overlap
has negligible effect.

This is a backend/CLI performance change with no visual output — no screenshots applicable.

## Test Plan

- `tests/issue_568_async_pipeline.rs`:
  - `async_pipeline_produces_valid_analysis_results` — verifies the pipeline produces
    valid candidates after the overlap refactoring
  - `async_pipeline_is_deterministic_with_fixed_seed` — verifies deterministic results
    across multiple runs with the same seed
- All existing tests pass (`cargo test --test-threads=1`)
- `quality.sh` passes cleanly
