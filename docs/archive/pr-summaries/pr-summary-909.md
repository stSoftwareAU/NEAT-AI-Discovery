## Summary

Recalibrated the IDENTITY activation boost to a penalty, discouraging its dominance in the candidate pool. IDENTITY's apparent 14.9% success rate was inflated by having the most candidates (274) of any activation. GRQ-sampler evidence (commit 7f15429) confirms IDENTITY neurons are frequently substituted with non-linear activations like SINE for improvement, indicating IDENTITY acts as a placeholder rather than an optimal choice.

Closes #909.

## Changes

- **IDENTITY penalty**: Changed from 1.03× boost to 0.85× penalty — IDENTITY now ranks below TANH (baseline), BIPOLAR, and all genuinely better non-linear activations
- **SINE activation added**: New 1.15× boost based on substitution evidence; SINUSOID alias also supported
- **Boost ordering recalibrated**: Updated the activation boost table documentation to reflect normalised success rates and candidate-pool dominance correction
- **Updated existing tests**: Adjusted boost ordering test to reflect IDENTITY's new position; added IDENTITY to the penalty test group; added SINE to compile-time range checks

## Evidence

All 238 scoring tests pass, including 10 new tests verifying:
- IDENTITY receives a penalty below baseline
- All non-linear activations (GELU, Mish, ELU, Softplus, SINE, ArcTan, SOFTSIGN, TANH) outscore IDENTITY with equal raw gain
- SINE lookup works correctly and SINUSOID maps to the same boost
- IDENTITY's penalty reduces score gain below the raw value
- Ordering integrity: IDENTITY sits between HARD_TANH (0.80) and BIPOLAR (0.95)

## Test Plan

- Added `tests/scoring/issue_909_identity_activation_penalty.rs` with 10 tests:
  - `identity_has_penalty_below_baseline` — verifies IDENTITY < 1.0 and < BIPOLAR
  - `identity_penalty_is_not_too_severe` — verifies IDENTITY >= 0.5 and >= HARD_TANH
  - `non_linear_activations_outscore_identity_with_equal_raw_gain` — verifies 8 non-linear activations all outscore IDENTITY
  - `identity_penalty_reduces_score_gain` — verifies boosted gain < raw gain
  - `sine_activation_has_boost_above_baseline` — verifies SINE > 1.0
  - `sine_lookup_returns_correct_value` — verifies function lookup
  - `sinusoid_maps_to_sine_boost` — verifies alias mapping
  - `sine_outscores_identity_with_equal_raw_gain` — verifies SINE > IDENTITY with ratio check
  - `identity_ranked_below_tanh_after_recalibration` — verifies IDENTITY < TANH
  - `identity_ranked_between_hard_tanh_and_bipolar` — verifies ordering
- Modified `tests/scoring/issue_887_activation_neuron_boost.rs`:
  - Added IDENTITY to `low_success_activations_get_penalty_below_baseline`
  - Updated `boost_ordering_reflects_success_rates` for new ordering with SINE and repositioned IDENTITY
  - Added SINE compile-time range validation
