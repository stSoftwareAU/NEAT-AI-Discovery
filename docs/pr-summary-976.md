## Summary

Reduce UUID string cloning in `src/focus/impact.rs` hot loops by avoiding unnecessary
heap allocations when building HashMap keys from synapse pairs. Closes #976.

### Changes

1. **Hoisted key creation outside inner record loops** (`compute_min_stats`, `compute_max_stats`):
   The synapse key `(from_uuid, to_uuid)` was cloned once per observation record. Now it is
   cloned once per synapse (outside the inner loop) and reused via `clone()` from a local variable.

2. **Avoided cloning for existing keys in `win_counts`**: Replaced `entry(key.clone()).or_insert(0)`
   with a `get_mut` check that only clones and inserts when the key is new.

3. **Avoided cloning `from_uuid` in `build_adjacency`**: Used `get_mut` to check if the key exists
   before cloning, eliminating clones for synapses sharing the same source neuron.

4. **Merged `inbound_count` and `total_inbound_weight` loops**: Combined two separate synapse
   iteration passes into one, halving the number of `to_uuid` clones and using `get_mut` to
   avoid cloning when the key already exists.

## Evidence

Benchmark results (`cargo bench --bench impact_uuid_cloning`):

| Benchmark | Baseline | Optimised | Change |
|---|---|---|---|
| dense_50h_3o_impacts | 54.7 us | 58.9 us | ~same (noise) |
| dense_200h_5o_impacts | 241.6 us | 211.6 us | **-12.4%** |
| selection_20_inputs | 31.5 us | 31.1 us | ~same |
| selection_100_inputs | 64.5 us | 64.3 us | ~same |
| fanout_50h_10edges | 124.2 us | 98.3 us | **-20.9%** |
| fanout_100h_20edges | 307.8 us | 242.7 us | **-21.2%** |
| fanout_200h_5edges | 255.5 us | 223.7 us | **-12.5%** |

Networks with many synapses sharing the same source/target UUIDs show 12-21% improvement.
Small networks with unique keys show no change (expected, since the optimisation targets
repeated key lookups).

## Test Plan

- All 15 existing impact-related tests pass (regression tests, impact discounting, normalisation)
- All 158 tests in the full test suite pass
- `./quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)
- Added `benches/impact_uuid_cloning.rs` benchmark for ongoing performance tracking
