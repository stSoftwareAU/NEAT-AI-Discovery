## Summary

Eliminate per-call `String` heap allocations in `CreatureTopologyCache::synapse_exists()` and `synapse_weight()` by switching from flat `HashSet<(String, String)>` / `HashMap<(String, String), f32>` to nested `HashMap<String, HashSet<String>>` / `HashMap<String, HashMap<String, f32>>`. This allows zero-copy lookups with `&str` keys since `String: Borrow<str>`. Closes #769.

## Changes

- **`src/analysis/detection/topology_cache.rs`**: Replaced `existing_synapses: HashSet<(String, String)>` with `existing_synapse_set: HashMap<String, HashSet<String>>` and `synapse_weights: HashMap<(String, String), f32>` with `synapse_weight_map: HashMap<String, HashMap<String, f32>>`. Fields are now private. Added `synapse_count()` accessor.
- **`tests/issue_769_zero_alloc_synapse_lookup.rs`**: New integration test covering exists/weight lookups, missing synapses, multiple in/out, and empty creatures.
- **`benches/synapse_lookup.rs`**: New benchmark measuring synapse lookup performance with and without string pre-allocation.
- **`tests/issue_576_benchmark_regression_tracking.rs`**: Added `synapse_lookup` to expected benchmarks list.

## Evidence

Benchmark results (500 neurons, pre-allocated keys isolate pure lookup cost):

| Method | Before | After | Improvement |
|--------|--------|-------|-------------|
| `synapse_exists` (prealloc) | 18.59 µs | 10.75 µs | **42% faster** |
| `synapse_weight` (prealloc) | 19.99 µs | 14.93 µs | **25% faster** |
| `synapse_exists` (with format) | 36.71 µs | 27.38 µs | **25% faster** |
| `synapse_weight` (with format) | 38.51 µs | 33.35 µs | **13% faster** |

The "with format" benchmarks include `format!()` overhead for creating test lookup strings, which masks the improvement. The "prealloc" results show the true lookup improvement.

No visual changes — backend-only performance optimisation.

## Test Plan

- Added `tests/issue_769_zero_alloc_synapse_lookup.rs` (8 tests):
  - `synapse_exists_returns_true_for_existing_synapses`
  - `synapse_exists_returns_false_for_missing_synapses`
  - `synapse_weight_returns_correct_values`
  - `synapse_weight_returns_none_for_missing`
  - `synapse_count_matches_creature_synapses`
  - `multiple_outgoing_from_same_source`
  - `multiple_incoming_to_same_target`
  - `empty_creature_has_no_synapses`
- All existing tests pass including `issue_754_topology_cache` (6 tests)
- `quality.sh` passes cleanly
