## Summary

Apply saturation-aware prediction discount for neuron candidates targeting near-saturated neurons. Production data shows predictions of ~0.01 for candidates targeting `HARD_TANH` at full activation range, while actual results are ~-0.00005 (100-200x overestimation beyond existing calibration). The discount is applied after pessimism discounting and before logistic calibration, linearly interpolating from 1.0 (at saturation threshold 0.9) to the aggressive floor of 0.15 (at full saturation). Closes #1112.

## Changes

- **`src/analysis/constants/candidate_scoring.rs`**: Added `SATURATION_DISCOUNT_AGGRESSIVE` constant (0.15) as the discount floor for fully saturated targets.
- **`src/analysis/synapse/scoring/discounting.rs`**: Added `saturation_discount()` helper and public `apply_saturation_prediction_discount()` function.
- **`src/analysis/neuron/post_processing.rs`**: Applied saturation discount in `apply_impact_discounting()` between pessimism discount and logistic calibration.
- **`src/analysis/synapse/scoring/mod.rs`** and **`src/analysis/synapse/mod.rs`**: Re-exported the new function.

## Evidence

Backend/scoring change with no web interface. Verified via unit tests:

- All 7 new tests pass (`cargo test --test scoring -- issue_1112`)
- `./quality.sh` passes cleanly (fmt, clippy, check, tests, doc build, release build)

## Test Plan

- `tests/scoring/issue_1112_saturation_prediction_discount.rs` — 7 tests:
  - `saturated_hard_tanh_target_gets_heavy_discount`: HARD_TANH at [-1, 1] (factor=1.0) reduces prediction by >= 60%
  - `non_saturated_identity_target_unchanged`: IDENTITY (None) receives no discount
  - `threshold_boundary_gets_no_discount`: factor=0.9 (threshold) returns unchanged gain
  - `below_threshold_gets_no_discount`: factor=0.5 returns unchanged gain
  - `discount_is_proportional_to_saturation`: factor=0.95 retains more gain than factor=1.0
  - `zero_gain_remains_zero`: zero input produces zero output
  - `negative_gain_preserves_sign`: negative gains remain negative after discount
