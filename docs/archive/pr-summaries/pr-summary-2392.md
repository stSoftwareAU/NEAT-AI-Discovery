## Summary

Calibration summaries now remain numeric for extreme finite observations: invalid Rust-side inputs are skipped, error metrics saturate safely, and overflowing ratios cannot poison the clamped factor. The FFI and history format remain unchanged. Closes #2392.

Version bump: package version updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- Prevent finite history values from becoming `null` metrics under `success: true` through overflowing subtraction, ratio, or sum arithmetic.
- Scale error terms before summing so cancellation is retained before an unrepresentable result is saturated.

### Essential Design Decisions

- Skip and warn on Rust-side observations with non-finite `predicted` or `actual` values.
- Exclude non-finite ratios and use `1.0` when none are usable; keep the existing near-zero prediction cap and `[0.1, 10.0]` clamp.
- Preserve the discovery-history JSON format. Empty observation buckets report zero errors and the neutral factor.

### Undiscoverable Facts

- The regression was observed returning `success: true` with `meanAbsoluteError: null` before the arithmetic changes.

## Evidence

- `tests/issue_2392_calibration_metrics_finite.rs::extreme_finite_history_returns_numeric_metrics` failed against the unchanged production code: the FFI response contained `meanAbsoluteError: null` with `success: true`. The same test passed after the fix.
- The original finite trigger and equivalent subtraction, ratio, and sum variants are closed: non-finite Rust inputs are skipped, finite error terms are scaled before aggregation and saturated at `f64::MAX`, and overflowing ratios are excluded before aggregation. No equivalent calibration arithmetic path remains that can serialise a non-finite metric.
- Backend-only Rust change; no visual surface changed.

## Test Plan

- `tests/issue_2392_calibration_metrics_finite.rs::extreme_finite_history_returns_numeric_metrics` — passed on the final implementation. Its three FFI histories cover subtraction overflow, opposite-sign overflowing ratios, and overflowing sums of finite error terms. It asserts success, numeric metrics and the factor clamp.
- `src/discovery_history.rs::tests::calibration_skips_non_finite_input_fields_and_keeps_valid_samples` — passed. Covers NaN, positive infinity and negative infinity independently in both input fields, then a mixed invalid/valid history.
- `src/discovery_history.rs::tests::calibration_summary_handles_empty_persisted_observation_bucket` and `src/discovery_history.rs::tests::calibration_summary_handles_zero_valued_observations` — passed; pin empty-bucket and zero-ratio behaviour.
- `src/discovery_history.rs::tests::calibration_errors_saturate_after_preserving_bias_cancellation` — passed; checks finite saturation and cancellation.
- `src/discovery_history.rs::tests::calibration_factor_discards_overflowing_ratios_and_uses_neutral_fallback` and `src/discovery_history.rs::tests::calibration_factor_respects_both_clamp_boundaries` — passed.
- Enabled `src/discovery_history.rs::tests::test_calibration_summary_stays_finite_for_extreme_finite_observations` and `src/discovery_history.rs::tests::test_calibration_factor_is_never_nan_for_finite_observations`; both passed.
- `tests/scoring/issue_605_calibration_tracking.rs` — all 13 existing calibration tests passed, including ordinary calibration accuracy and near-zero prediction behaviour.
- Mutation checks: removing either finite-input operand independently made `calibration_skips_non_finite_input_fields_and_keeps_valid_samples` fail; removing the ratio-finiteness filter made `calibration_factor_discards_overflowing_ratios_and_uses_neutral_fallback` fail with NaN; removing saturation made `calibration_errors_saturate_after_preserving_bias_cancellation` fail with infinity; removing the zero-scale guards made the zero-value and empty-bucket tests fail with NaN.
- `cargo fmt --all --check`, `cargo clippy --lib --tests --all-features -- -D warnings`, and `git diff --check` passed.
- `./quality.sh < /dev/null` was bounded at 900 seconds and timed out while compiling the full integration-test targets; it produced no full-suite result. The targeted calibration suites above passed.

## Merge verification

- Full `./quality.sh` result: pending; the bounded local run timed out during integration-test compilation.
- CI and code review: pending.
