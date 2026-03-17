## Summary

Added comprehensive test coverage for the `filter_candidates_to_sensible_ranges` function
and pairing function edge cases, completing the DRY variant generation refactoring.
The core consolidation (parameterised `NeuronVariantConfig`/`SynapseVariantConfig` structs
replacing three near-identical functions) was completed in PRs #819/#820. This PR adds the
missing test coverage identified during review. Addresses #806.

### Changes

- Added 16 new tests to `tests/issue_806_dry_variant_generation.rs`:
  - 10 tests for `filter_candidates_to_sensible_ranges` covering boundary values,
    NaN/infinity rejection, excessive weights/bias, negative values, mixed candidate lists,
    and empty input
  - 6 tests for pairing function edge cases (zero limits, empty inputs, non-extreme
    pass-through, variant generation with/without limits)

## Evidence

All 34 tests in `issue_806_dry_variant_generation.rs` pass. Full `quality.sh` passes cleanly.
All existing tests (40+ variant-related tests across 6 other test files) continue to pass
unchanged.

## Test Plan

- Extended `tests/issue_806_dry_variant_generation.rs` from 18 to 34 tests:
  - `filter_sensible_ranges_passes_candidates_within_bounds`
  - `filter_sensible_ranges_rejects_excessive_incoming_weight`
  - `filter_sensible_ranges_rejects_excessive_bias`
  - `filter_sensible_ranges_rejects_excessive_outgoing_weight`
  - `filter_sensible_ranges_rejects_nan_values`
  - `filter_sensible_ranges_rejects_infinity`
  - `filter_sensible_ranges_accepts_boundary_values`
  - `filter_sensible_ranges_keeps_valid_from_mixed_list`
  - `filter_sensible_ranges_returns_empty_for_empty_input`
  - `filter_sensible_ranges_handles_negative_values_within_bounds`
  - `pair_extreme_neuron_zero_limit_returns_empty`
  - `pair_extreme_neuron_non_extreme_passes_through_unmodified`
  - `pair_synapse_zero_limit_returns_empty`
  - `pair_synapse_empty_input_returns_empty`
  - `pair_synapse_generates_variants_with_no_limit`
  - `pair_synapse_limit_truncates_variants`
