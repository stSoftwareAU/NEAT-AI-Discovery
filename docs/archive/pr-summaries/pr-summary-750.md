## Summary

Deduplicate Pearson correlation and statistical utility functions across detection modules into a single shared `stats.rs` module, following the DRY principle. Closes #750.

**Changes:**
- Created `src/analysis/detection/stats.rs` with shared `pearson_correlation`, `compute_mean`, and `compute_variance` functions
- Replaced 4 separate Pearson correlation implementations across `opposing_synapse.rs`, `co_adaptation.rs`, and `correlated_error.rs` (2 variants)
- Replaced duplicated `compute_mean` / `compute_variance` in `input_sensitivity.rs` and `noise_signal.rs`
- **Bug fix**: The `opposing_synapse.rs` implementation was missing `.clamp(-1.0, 1.0)`, which could produce values slightly outside [-1.0, 1.0] due to floating-point rounding. The shared implementation includes the clamp.

**Net effect**: Removed ~151 lines of duplicated code, replaced with 14 lines of imports plus the single 62-line shared module.

## Evidence

This is a backend refactoring with no UI changes. Correctness is verified by:
- 15 new unit tests covering all edge cases (empty, single element, constant, perfect correlation, clamping)
- All existing tests continue to pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

- Added `tests/issue_750_stats_deduplication.rs` with 15 tests:
  - `pearson_empty_inputs_return_zero`
  - `pearson_single_element_returns_zero`
  - `pearson_constant_sequences_return_zero_not_nan`
  - `pearson_perfect_positive_correlation`
  - `pearson_perfect_negative_correlation`
  - `pearson_identical_sequences_return_exactly_one`
  - `pearson_result_always_clamped`
  - `pearson_mismatched_lengths_uses_shorter`
  - `mean_empty_returns_zero`
  - `mean_single_element`
  - `mean_multiple_elements`
  - `variance_empty_returns_zero`
  - `variance_single_element_returns_zero`
  - `variance_constant_values_returns_zero`
  - `variance_known_values`
