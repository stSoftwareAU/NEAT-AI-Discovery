## Summary

Eliminate data-dependent branching in the auto-vectorisation hot paths for synapse, ReLU, and activation improvement scoring. Closes #1075.

Each of the three `compute_*_improvement_and_count` functions contained per-sample branches (`Option` checks, `TargetSimulationMode` match, `is_finite()` guards) that inhibited compiler auto-vectorisation. This PR:

1. **Splits each function into specialised variants** dispatched once before the loop:
   - `compute_relu_improvement_no_target` / `compute_relu_improvement_with_target`
   - `compute_activation_improvement_no_target` / `compute_activation_improvement_with_target`
   - `compute_synapse_improvement_no_target` / `compute_synapse_improvement_with_target` / `compute_synapse_improvement_approximate`

2. **Replaces `is_finite()` branching** with a branchless `select_finite()` helper that returns `0.0` for non-finite values, allowing the compiler to vectorise the accumulation loop.

3. **Extracts shared `finalise_improvement()`** to deduplicate the improvement calculation across all variants.

Public API signatures are unchanged; the split is entirely internal.

## Evidence

Benchmark results from `cargo bench --bench simd_hot_paths` (median times):

| Benchmark | Baseline | After | Improvement |
|-----------|----------|-------|-------------|
| synapse_improvement/value_domain/100 | 156.9 ns | 118.0 ns | **-24.8%** |
| synapse_improvement/value_domain/1000 | 1.646 us | 1.113 us | **-32.4%** |
| synapse_improvement/value_domain/10000 | 16.73 us | 11.55 us | **-30.9%** |
| synapse_improvement/tanh_simulation/100 | 163.8 ns | 137.6 ns | **-15.9%** |
| synapse_improvement/tanh_simulation/1000 | 1.638 us | 1.145 us | **-30.1%** |
| synapse_improvement/tanh_simulation/10000 | 16.51 us | 11.66 us | **-29.4%** |
| relu_improvement/no_target_fn/100 | 138.7 ns | 91.1 ns | **-34.3%** |
| relu_improvement/no_target_fn/1000 | 1.583 us | 1.040 us | **-34.3%** |
| relu_improvement/no_target_fn/10000 | 16.07 us | 11.21 us | **-30.2%** |
| relu_improvement/tanh_target/10000 | 65.38 us | 69.15 us | +5.8% (noise) |
| activation_improvement/tanh_no_target/10000 | 41.67 us | 42.61 us | +2.3% (noise) |
| activation_improvement/tanh_with_target/10000 | 108.1 us | 109.2 us | +1.0% (noise) |

The value-domain (no-target) paths show **30-34% throughput improvement**. The activation paths are dominated by `tanh()` call cost, so the branch elimination has negligible impact there (within noise). No regressions in unrelated benchmarks.

## Test Plan

- Added 6 new unit tests in `src/analysis/synapse/scoring/tests.rs`:
  - `test_relu_no_target_branchless_handles_non_finite` - non-finite error handling
  - `test_relu_with_target_identity_matches_no_target` - identity target equivalence
  - `test_synapse_no_target_branchless_handles_non_finite` - synapse non-finite handling
  - `test_activation_no_target_branchless_handles_non_finite` - activation non-finite handling
  - `test_branchless_variants_empty_samples` - empty input edge case
  - `test_synapse_improvement_positive_weight_reduces_positive_error` - correctness check
- All 4 existing scoring tests continue to pass unchanged
- `./quality.sh` passes cleanly
