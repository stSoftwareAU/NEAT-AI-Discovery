## Summary

Refactored the synapse analysis pipeline to use thread-local collection with a final single-threaded merge instead of shared `Mutex<Vec<...>>` collectors. This eliminates mutex contention during the parallel phase where rayon threads previously competed for locks to push individual results. Closes #744.

### What changed

- **Removed** `SharedResultCollectors` struct with its 6 `Arc<Mutex<...>>` fields
- **Added** `AtomicMetadata` struct for lock-free atomic metadata updates during parallel processing
- **Added** `MergedResults` struct that collects per-target results via rayon's `map` + `collect` pattern and merges them in a single-threaded pass
- **Changed** the parallel loop from `par_iter().try_for_each()` (locking per target) to `par_iter().map().collect()` (lock-free per target)
- The `analysis_timed_out` flag changed from `Mutex<bool>` to `AtomicBool`

### What stayed the same

- All existing tests pass without modification
- The observable behaviour (candidates produced, metadata, post-processing) is identical
- Atomic metadata fields (target_value_seen, saturation_aware_used, input min/max) were already lock-free and remain so

## Evidence

### Benchmark results (`cargo bench --bench parallel_discovery`)

| Benchmark | Baseline (ms) | After (ms) | Change | p-value |
|-----------|--------------|------------|--------|---------|
| 5h_100r | 160.43 | 145.52 | **-9.3%** | < 0.05 |
| 20h_200r | 127.51 | 119.31 | **-6.4%** | < 0.05 |
| 50h_200r | 321.07 | 306.29 | **-4.6%** | < 0.05 |

All three configurations show statistically significant improvement (p < 0.05). The improvement is largest for smaller creatures where mutex contention represents a larger fraction of total runtime.

## Test Plan

- All existing tests pass (no test modifications required)
- `quality.sh` passes cleanly
- Benchmark results demonstrate meaningful improvement across all creature sizes
