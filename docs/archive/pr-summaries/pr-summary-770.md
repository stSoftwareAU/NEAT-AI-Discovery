## Summary

Pass `CreatureTopologyCache` to weight coherence detection functions, eliminating redundant `HashMap`/`HashSet` construction on every call. Closes #770.

The three weight coherence detection functions (`detect_incoherent_weight_ratios`, `detect_near_constant_paths`, `detect_symmetric_cancellation`) now accept `Option<&CreatureTopologyCache>`. When provided, they use the cache's `fan_in`, `fan_out`, `hidden_uuids`, and `synapse_weight()` data instead of rebuilding local structures. When `None` is passed, they fall back to local construction for backward compatibility.

This follows the same pattern used by `detect_dead_neurons` and `detect_bottleneck_neurons`.

## Evidence

### Benchmark Results

| Creature Size | Without Cache | With Cache | Speedup |
|--------------|--------------|------------|---------|
| 50 neurons   | 34.9 µs      | 8.3 µs     | 4.2×    |
| 100 neurons  | 74.0 µs      | 17.5 µs    | 4.2×    |
| 200 neurons  | 167.7 µs     | 42.0 µs    | 4.0×    |
| 500 neurons  | 437.2 µs     | 116.2 µs   | 3.8×    |

Consistent ~4× improvement from eliminating redundant topology map construction.

## Test Plan

- Added `tests/issue_770_weight_coherence_topology_cache.rs` with 5 tests:
  - `test_incoherent_weight_ratios_with_topology_cache` — verifies identical results with/without cache
  - `test_near_constant_paths_with_topology_cache` — verifies identical results with/without cache
  - `test_symmetric_cancellation_with_topology_cache` — verifies identical results with/without cache
  - `test_multiple_hidden_neurons_with_cache` — multi-neuron network
  - `test_no_hidden_neurons_with_cache_returns_empty` — edge case
- All 16 existing tests in `tests/issue_437_weight_coherence_validation.rs` updated to pass `None` and continue passing
- Added `benches/weight_coherence_cache.rs` benchmark suite
- `quality.sh` passes cleanly
