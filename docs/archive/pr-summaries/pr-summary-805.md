## Summary

Add numeric safety guards for division-by-zero and overflow-prone casts. Closes #805.

### Changes

1. **`src/analysis/gpu/bias_evaluation.rs`**: Added guard for `step < f32::EPSILON` before the bias grid division `((max_bias - min_bias) / step).ceil() as i32`. This prevents division-by-zero producing infinity which causes undefined behaviour when cast to `i32`.

2. **`src/activations.rs`**: Added `.is_finite()` belt-and-suspenders checks after all four `f64 as f32` casts in EXPONENTIAL and SOFTPLUS (both `apply_scalar_squash` and `target_simulation_fn`). The existing cutoff guards already prevent non-finite results in practice, but these additional checks provide defence-in-depth against future changes to the cutoff values.

3. **`src/discovery_history.rs`**: Already safe — the `compute_calibration_factor` function has an existing `if ratio_count == 0 { return 1.0; }` guard, and the else branch always increments `ratio_count` for near-zero predictions (capping the ratio at `CALIBRATION_FACTOR_MAX`).

4. **`src/record/mod.rs` and `src/streaming.rs`**: Already safe — production code uses `u32::try_from()` with proper error handling. The bare `as u32` casts mentioned in the issue exist only in test code with known small values.

## Evidence

- `quality.sh` passes with all checks (fmt, clippy, check, test, doc, release build)
- All 9 new edge-case tests pass
- All existing tests pass

## Test Plan

New test file `tests/issue_805_numeric_safety.rs` with 9 tests:
- `exponential_returns_finite_for_extreme_negative` — f32::MIN input
- `exponential_target_sim_returns_finite_for_extreme_negative` — target simulation with f32::MIN
- `exponential_returns_finite_for_non_finite_inputs` — NaN, +inf, -inf inputs
- `softplus_returns_finite_for_large_positive` — x=700 (near cutoff)
- `softplus_target_sim_returns_finite_for_large_positive` — target simulation with x=700
- `softplus_returns_finite_for_non_finite_inputs` — NaN, +inf, -inf inputs
- `calibration_factor_unknown_module_returns_one` — unknown module returns 1.0
- `calibration_factor_near_zero_predicted_no_div_by_zero` — zero predicted value
- `calibration_factor_all_near_zero_predictions` — multiple near-zero predictions
