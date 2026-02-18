## Summary

Refactor: Extract weight calculation functions from `implementation.rs` to dedicated `src/analysis/weights.rs` module.

This PR addresses Issue #270 by extracting ~300 lines of weight calculation logic into a dedicated module, improving code organisation and maintainability.

### Functions Extracted

**Weight Calculation Functions:**
- `calculate_optimal_outgoing_weight()` - Compute optimal weight from samples using least squares formula
- `calculate_optimal_identity_outgoing_and_bias()` - Joint weight/bias calculation for IDENTITY neurons (affine correction fitting)
- `calculate_optimal_bias()` - Find optimal bias via grid search (CPU or GPU accelerated)

**Weight Utility Functions:**
- `clamp_weight_update_delta()` - Constrain weight updates to valid ranges
- `coordinated_structural_activation_delta()` - Compute delta for coordinated structural candidates

**Related Constants:**
- `MAX_OUTGOING_WEIGHT` (0.1) - Maximum allowed outgoing weight for add-neuron/synapse candidates
- `MIN_WEIGHT_RATIO` (50.0) - Minimum incoming/outgoing weight ratio for reliable predictions
- `MIN_NEURON_SAMPLE_COUNT` (10) - Minimum samples required for neuron evaluation

### Changes Made

1. **New file: `src/analysis/weights.rs`**
   - Contains all weight calculation functions with comprehensive documentation
   - Includes ~25 unit tests covering various edge cases
   - Uses Australian English spelling in comments

2. **Updated `src/analysis/mod.rs`**
   - Added `pub mod weights;` declaration
   - Added re-exports for public API accessibility

3. **Updated `src/analysis/implementation.rs`**
   - Removed extracted functions and constants
   - Added imports from the new weights module
   - Removed duplicate tests now covered by weights module

## Evidence

Unable to generate screenshot: This is a CLI-only Rust library with no visual interface.

## Test Plan

- Added 24 new unit tests in `src/analysis/weights.rs`:
  - `test_optimal_weight_returns_none_for_insufficient_activation`
  - `test_optimal_weight_returns_none_for_very_small_activation`
  - `test_optimal_weight_returns_none_for_non_finite_results`
  - `test_optimal_weight_returns_none_for_near_zero_weights`
  - `test_optimal_weight_is_clamped_to_max_outgoing_weight`
  - `test_optimal_weight_accounts_for_incoming_weight`
  - `test_optimal_weight_normal_calculation`
  - `test_optimal_weight_ratio_validation`
  - `test_optimal_weight_rejects_poor_ratio`
  - `test_identity_returns_none_for_empty_samples`
  - `test_identity_returns_none_for_invalid_samples`
  - `test_identity_with_low_variance_returns_zero_bias`
  - `test_identity_computes_valid_weight_and_bias`
  - `test_optimal_bias_returns_zero_for_empty_samples`
  - `test_optimal_bias_returns_zero_for_zero_baseline_error`
  - `test_optimal_bias_with_tanh`
  - `test_optimal_bias_with_relu`
  - `test_clamp_delta_returns_none_for_tiny_delta`
  - `test_clamp_delta_within_bounds`
  - `test_clamp_delta_exceeds_upper_bound`
  - `test_clamp_delta_exceeds_lower_bound`
  - `test_clamp_delta_already_at_max`
  - `test_coordinated_delta_returns_none_for_zero_noisy_weight`
  - `test_coordinated_delta_computes_correctly`
  - `test_coordinated_delta_with_negative_weights`
  - `test_hard_tanh_clamps_correctly`

- All 330 existing tests continue to pass
- `./quality.sh` passes cleanly with no warnings
