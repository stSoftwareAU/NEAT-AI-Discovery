## Summary

Fixed performance issue in `rank_focus_neurons` where synapse counting had O(n×m) complexity. The `count_synapses_for_neuron` function was scanning all synapses twice (incoming + outgoing) for each neuron being ranked.

**Solution**: Pre-compute synapse counts into HashMaps during initialisation, reducing complexity to O(n+m).

### Changes

1. **Added `SynapseCounts` struct** (`src/focus.rs:658-705`)
   - Pre-computes incoming and outgoing synapse counts in O(m) time
   - Provides O(1) lookup via `get(neuron_uuid)` method
   - Used in three places within `rank_focus_neurons`:
     - Removal candidates filtering
     - High-error exploratory ablation candidates
     - Constant neuron removal candidates

2. **Removed deprecated `count_synapses_for_neuron` function**
   - Was performing O(m) iteration per call
   - All usages replaced with `SynapseCounts::get()`

3. **Updated `quality.sh`**
   - Exclude benchmarks from test runs (criterion benchmarks use custom harness)

## Evidence

### Benchmark Results

```
synapse_counts/old_O(n×m)/100n_500s
                        time:   [155.72 µs 155.93 µs 156.14 µs]
synapse_counts/new_O(n+m)/100n_500s
                        time:   [27.854 µs 27.898 µs 27.944 µs]

synapse_counts/old_O(n×m)/500n_10000s
                        time:   [14.089 ms 14.115 ms 14.140 ms]
synapse_counts/new_O(n+m)/500n_10000s
                        time:   [486.51 µs 487.53 µs 488.57 µs]

synapse_counts/old_O(n×m)/1000n_50000s
                        time:   [147.67 ms 147.81 ms 147.94 ms]
synapse_counts/new_O(n+m)/1000n_50000s
                        time:   [2.4810 ms 2.4890 ms 2.4974 ms]
```

| Creature Size | Old O(n×m) | New O(n+m) | Speedup |
|---------------|------------|------------|---------|
| 100 neurons, 500 synapses | 155.9 µs | 27.9 µs | **5.6x** |
| 500 neurons, 10,000 synapses | 14.1 ms | 487.5 µs | **29x** |
| 1000 neurons, 50,000 synapses | 147.8 ms | 2.49 ms | **59x** |

The improvement scales with creature size as expected from the complexity analysis.

## Test Plan

### New Tests Added
- `tests/issue_208_synapse_counts.rs`:
  - `test_synapse_counts_basic` - Basic incoming/outgoing counts
  - `test_synapse_counts_multiple_connections` - Multiple connections per neuron
  - `test_synapse_counts_neuron_with_no_synapses` - Orphan neuron handling
  - `test_synapse_counts_self_loop` - Self-referential synapses
  - `test_synapse_counts_empty_creature` - Empty creature edge case
  - `test_synapse_counts_nonexistent_neuron` - Missing neuron lookup
  - `test_synapse_counts_complex_network` - Complex topology verification
  - `test_synapse_counts_matches_original_implementation` - Correctness against original

### Benchmarks Added
- `benches/synapse_counts.rs`:
  - Compares old O(n×m) vs new O(n+m) approach
  - Tests three creature sizes: 100/500, 500/10000, 1000/50000 neurons/synapses
  - Run with: `cargo bench --bench synapse_counts`

### Existing Tests
All 200+ existing tests continue to pass, verifying no regressions in functionality.
