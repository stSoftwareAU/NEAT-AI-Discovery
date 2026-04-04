## Summary

Make weight constraints activation-aware so that non-linear activation candidates
(TANH, GELU, ReLU, etc.) are not systematically rejected by constraints calibrated
on IDENTITY-dominated success data. Closes #905.

The tightened constraints from Issue #888 (`MAX_OUTGOING_WEIGHT = 0.01`,
`MIN_WEIGHT_RATIO = 50.0`) were over-fitted to IDENTITY candidates. Non-linear
activations compress their output range and operate in different weight regimes,
requiring relaxed constraints:

- **`MAX_OUTGOING_WEIGHT_NON_LINEAR = 0.03`** (3x the IDENTITY ceiling of 0.01)
- **`MIN_WEIGHT_RATIO_NON_LINEAR = 10.0`** (5x more permissive than IDENTITY's 50.0)

New public functions `max_outgoing_weight_for_activation()` and
`min_weight_ratio_for_activation()` select the appropriate constraint based on
the activation name. `calculate_activation_aware_outgoing_weight()` wraps the
core least-squares weight calculation with activation-aware constraints.

All non-IDENTITY callers in the activation evaluation pipeline now use the
activation-aware function. IDENTITY candidates are unchanged (no regression).

## Evidence

- 13 new unit tests verify activation-aware behaviour
- All existing tests pass (including GPU shader tests)
- `quality.sh` passes cleanly

## Test Plan

- Added `tests/scoring/issue_905_activation_aware_weight_constraints.rs` (13 tests):
  - `test_identity_gets_tight_max_outgoing_weight` — IDENTITY uses 0.01 ceiling
  - `test_tanh_gets_relaxed_max_outgoing_weight` — TANH gets relaxed ceiling
  - `test_gelu_gets_relaxed_max_outgoing_weight` — GELU gets relaxed ceiling
  - `test_relu_gets_relaxed_max_outgoing_weight` — ReLU gets relaxed ceiling
  - `test_identity_gets_tight_min_weight_ratio` — IDENTITY uses 50.0 ratio
  - `test_non_linear_gets_relaxed_min_weight_ratio` — Non-linear activations use relaxed ratio
  - `test_activation_aware_identity_matches_default` — IDENTITY path matches default function
  - `test_activation_aware_tanh_allows_larger_weight` — TANH weight not clamped to 0.01
  - `test_activation_aware_non_linear_passes_ratio_with_smaller_incoming` — Hidden-source passes
  - `test_activation_aware_still_rejects_invalid_weights` — Invalid inputs still rejected
  - `test_activation_aware_no_regression_for_identity` — No regression for IDENTITY
  - `test_tanh_candidate_not_systematically_rejected` — TANH with small incoming passes
  - `test_gelu_candidate_with_moderate_weight_accepted` — GELU keeps unclamped weight
