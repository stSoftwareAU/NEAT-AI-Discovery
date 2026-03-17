## Summary

Replace `Mutex<HashMap<String, f32>>` with `DashMap<String, f32>` in the impact computation shared cache (`src/focus/impact.rs`). The previous implementation acquired a global mutex lock on every recursive call during parallel impact computation, causing significant contention with deep neuron graphs and many threads. DashMap provides internally-sharded concurrent reads and writes, eliminating the lock bottleneck. Closes #835.

## Evidence

Benchmark results (`cargo bench --bench impact_cache_contention`):

| Scenario | Before (Mutex) | After (DashMap) | Change |
|----------|---------------|-----------------|--------|
| deep_100_layers | 311.94 us | 193.24 us | -39% |
| mesh_10x20 | 1.6182 ms | 1.2313 ms | -27% |
| wide_200_neurons | 375.77 us | ~375 us | ~0% (minimal recursion) |

The deep chain and diamond mesh scenarios (which represent the contention-heavy recursive patterns described in the issue) show significant improvement. The wide scenario has minimal recursion so contention was never the bottleneck there.

## Test Plan

- Added `tests/focus/issue_835_impact_cache_contention.rs` with 4 tests:
  - `issue_835_wide_network_impact_values_are_deterministic` — 50 neurons to single output, verifies exact values across 5 runs
  - `issue_835_deep_chain_impact_values_are_deterministic` — 30-layer chain, verifies exact values across 5 runs
  - `issue_835_diamond_mesh_impact_values_are_deterministic` — 5x10 mesh, verifies exact values across 5 runs
  - `issue_835_cyclic_network_produces_finite_positive_impacts` — cyclic network, verifies finite positive results across 10 runs
- Added `benches/impact_cache_contention.rs` benchmark with wide, deep, and mesh scenarios
- All existing impact tests continue to pass unchanged
- `quality.sh` passes cleanly
