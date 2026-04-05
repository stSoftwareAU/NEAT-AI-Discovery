## Summary

Audit and reduce unnecessary `clone()` calls in the candidate analysis pipeline. Closes #943.

### Changes

1. **`post_processing.rs`**: Changed `neuron_type_map` from `HashMap<String, String>` to `HashMap<&str, &str>`, borrowing UUID and neuron type strings from input instead of cloning. Similarly changed `error_sq_map` from `HashMap<String, f32>` to `HashMap<&str, f32>`.

2. **`scoring.rs`**: Updated `apply_target_type_boost` signature to accept `HashMap<&str, &str>` (consistent with borrowed neuron type maps throughout the pipeline).

3. **`candidate_aggregation.rs`**: Changed `direct_synapse_weight` map from `HashMap<(String, String), f32>` to `HashMap<(&str, &str), f32>`, eliminating 2 string clones per synapse when building the lookup map, plus 2 per candidate when constructing lookup keys.

4. **`sample_weighted.rs`**: Replaced `sort_by` (O(n log n)) with `select_nth_unstable_by` (O(n)) for median computation in `stratify_samples`, reducing both CPU time and allocation overhead.

### Clone reduction count

| Location | Clones eliminated per call |
|----------|---------------------------|
| `neuron_type_map` construction | 2 x N (uuid + type per neuron) |
| `error_sq_map` construction | N (uuid per neuron) |
| `direct_synapse_weight` construction | 2 x S (from + to uuid per synapse) |
| `direct_synapse_weight` lookup | 2 per candidate |
| `stratify_samples` sort | Replaced O(n log n) with O(n) |

Where N = number of neurons and S = number of synapses.

## Evidence

### Benchmark results (criterion, `candidate_pipeline_clones`)

**Neuron type map construction** (clone vs borrow):

| Scale | Cloned | Borrowed | Improvement |
|-------|--------|----------|-------------|
| 20 neurons | 647 ns | 228 ns | **64.8% faster** |
| 100 neurons | 3,336 ns | 1,103 ns | **66.9% faster** |
| 500 neurons | 17,214 ns | 5,484 ns | **68.1% faster** |

**Synapse weight map construction** (clone vs borrow):

| Scale | Cloned | Borrowed | Improvement |
|-------|--------|----------|-------------|
| 20 synapses | 1,104 ns | 479 ns | **56.6% faster** |
| 100 synapses | 4,152 ns | 1,907 ns | **54.1% faster** |
| 500 synapses | 19,213 ns | 8,492 ns | **55.8% faster** |

**Stratify samples** (sort vs `select_nth_unstable`):

| Scale | Before (sort) | After (select_nth) | Improvement |
|-------|---------------|---------------------|-------------|
| 50 records | 483 ns | 338 ns | **30.0% faster** |
| 200 records | 1,425 ns | 825 ns | **42.1% faster** |
| 1000 records | 7,275 ns | 3,163 ns | **56.5% faster** |

## Test Plan

- All 158 existing tests pass unchanged (verified via `./quality.sh`)
- Updated test files to match new `HashMap<&str, &str>` signature:
  - `tests/analysis/issue_468_target_type_prioritisation.rs` (mechanical: `to_string()` removed from HashMap construction)
  - `tests/synapse/issue_522_synapse_scoring.rs` (mechanical: same)
- New benchmark: `benches/candidate_pipeline_clones.rs` measuring all three optimisation categories
