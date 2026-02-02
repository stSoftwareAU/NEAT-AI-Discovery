## Summary

Add inline unit tests (`#[cfg(test)]`) to all 9 discovery detection modules in `src/analysis/`. These tests exercise internal helper functions and private logic that cannot be tested through the public API alone, complementing the existing integration tests in `tests/`.

Closes #376.

## Changes

Added `#[cfg(test)] mod tests` blocks to each of the 9 detection modules:

| Module | Unit tests added | Coverage areas |
|--------|-----------------|----------------|
| `saturation.rs` | 21 | `is_bounded_squash`, `check_bounded_saturation`, `compute_saturation_severity`, `compute_input_std_dev`, `recommend_fix`, detection thresholds, conversion |
| `bottleneck.rs` | 8 | `bottleneck_parallel_neuron_uuid` determinism, fan-in/fan-out ratio filtering, output exclusion, conversion |
| `dead_neuron.rs` | 9 | `compute_removal_confidence`, `find_connected_outputs` (direct + transitive), active/output exclusion, conversion |
| `correlated_error.rs` | 11 | `compute_pearson_correlation` (positive/negative/zero-variance/no-overlap), `cluster_correlated_outputs`, `count_shared_error_samples`, `shared_neuron_uuid` |
| `multi_hop.rs` | 9 | `compute_activation_error_correlation`, `compute_activation_activation_correlation`, `compute_mean_abs_error`, `multi_hop_relay_uuid` determinism |
| `oscillating_neuron.rs` | 12 | `recommend_squash_for_oscillation`, `recommend_bias_for_oscillation`, stable/dead/insufficient exclusion, conversion |
| `dormant_synapse.rs` | 5 | Near-zero weight detection, active weight exclusion, sole connection protection, insufficient samples, conversion |
| `opposing_synapse.rs` | 7 | `pearson_correlation` (positive/negative/single/zero-variance), hidden target exclusion, removal vs weight flip conversion |
| `output_bias_drift.rs` | 6 | Positive/balanced error detection, hidden neuron exclusion, noise threshold, insufficient samples, conversion bias value |

**Total: 88 new unit tests** (test count increased from 434 to 522).

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

All 522 tests pass and `./quality.sh` completes successfully (fmt, clippy, check, test, release build).

## Test Plan

- 88 new `#[cfg(test)]` unit tests across the 9 detection modules
- Each module has at least 5 unit tests covering:
  - **Detection criteria**: Verify neurons/synapses meeting threshold criteria are detected
  - **Exclusion criteria**: Verify neurons/synapses that should NOT be detected are excluded
  - **Edge cases**: Insufficient samples, boundary threshold values, zero-variance inputs
  - **Conversion**: Verify `*_to_coordinated_candidates()` produces correct operation types
- All tests use Australian English in names and comments
- Tests verify outcomes, not implementation details
- `./quality.sh` passes cleanly
