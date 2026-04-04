## Summary

Fixed the saturation-aware simulation fallback for non-linear activations. When `target_value`
is missing but `target_activation` is available, the system now uses approximate inverse
functions to recover `target_value` from `target_activation` for common monotonic activations
(TANH, LOGISTIC, GELU, SOFTSIGN, etc.). Previously, only HARD_TANH/CLIPPED received
saturation-aware scoring in this path; all other non-linear activations fell back to the
linear model which ignores saturation effects, causing unreliable improvement predictions.

Closes #906.

## Changes

### `src/activations.rs`
- Added `approximate_inverse_fn()` returning inverse functions for 19 activations:
  - Closed-form inverses: TANH (`atanh`), LOGISTIC (logit), SOFTSIGN, BIPOLAR_SIGMOID,
    ISRU, ARCTAN, LOGSIGMOID, SOFTPLUS
  - Piecewise-linear inverses: HARD_TANH, RELU6, ELU, LEAKYRELU, SELU
  - Newton's method inverses (4 iterations): GELU, MISH, SWISH
  - Trivial: IDENTITY, COMPLEMENT
- Added unit tests for round-trip accuracy of all inverse functions

### `src/analysis/activation/simulation.rs`
- Changed `ApproximateValueFromActivation` from a tuple variant `(fn(f32) -> f32)` to a
  struct variant with both `activation_fn` and `inverse_fn` fields
- Extended `get_target_simulation_mode()` to use `approximate_inverse_fn()` for all supported
  activations instead of only HARD_TANH/CLIPPED

### `src/analysis/synapse/scoring.rs`
- Updated both match arms for `ApproximateValueFromActivation` to use `inverse_fn` to compute
  `target_value` when it is missing, instead of using `target_activation` directly

### Updated match patterns
- `src/analysis/activation/mod.rs` — updated tests for struct variant pattern
- `src/analysis/synapse/tests.rs` — updated test pattern
- `tests/unit/analysis_implementation.rs` — updated test pattern

## Evidence

All 151 tests pass including 14 new integration tests and 5 new unit tests that verify:
- Mode selection returns `ApproximateValueFromActivation` for TANH, LOGISTIC, SOFTSIGN,
  GELU, BIPOLAR_SIGMOID, ELU (and still for HARD_TANH)
- Non-invertible activations (SINE, GAUSSIAN, SQUARE) still fall back to None
- Full mode still used when `target_value` is present
- Round-trip accuracy of inverse functions (forward then inverse ≈ identity)
- Saturation-aware estimates do not overestimate vs linear model near saturation
- Approximate and Full mode predictions are comparable for non-saturated regions

## Test Plan

- `tests/synapse/issue_906_saturation_aware_fallback_nonlinear.rs` — 14 integration tests:
  - Mode selection tests for TANH, LOGISTIC, SOFTSIGN, GELU, BIPOLAR_SIGMOID, ELU, HARD_TANH
  - Non-invertible activations fall back to None
  - Full mode regression test
  - None squash regression test
  - End-to-end saturation tests via epistatic detection (TANH, LOGISTIC, GELU)
  - Approximate vs Full mode comparison for TANH
- `src/activations.rs` unit tests — 5 new tests:
  - `inverse_fn_exists_for_monotonic_activations` — 19 activations verified
  - `inverse_fn_none_for_non_monotonic` — 6 activations verified
  - `inverse_round_trip_closed_form` — TANH, LOGISTIC, SOFTSIGN, ELU
  - `inverse_round_trip_newton` — GELU, MISH, SWISH
