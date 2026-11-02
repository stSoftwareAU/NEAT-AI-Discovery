## Summary

Calibration history now produces finite numeric error metrics and a calibration factor within `[0.1, 10.0]` for finite inputs. Non-finite Rust observations are skipped with a warning, and the discovery-history JSON format is unchanged. Closes #2392.

Version bump: package version updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- Prevent finite observations from overflowing subtraction, ratio, or accumulation and becoming JSON `null` under `success: true`.
- Scale values before aggregation, preserve cancellation, and saturate only results that cannot be represented as `f64`.

### Essential Design Decisions

- Skip and warn on non-finite values supplied through the Rust recording API.
- Exclude overflowing ratios and average remaining ratios with scaled compensated summation; retain the near-zero prediction rule, neutral fallback `1.0`, and `[0.1, 10.0]` clamp.
- Keep the persisted discovery-history JSON shape and ordinary-value results unchanged.

### Undiscoverable Facts

- Develop was fetched at `21371f75faeef91dc42dd5bdf0cc7e70bc8fbdcc`; the package version was resolved to `0.74.324` over its `0.74.323`.
- The baseline regression was run in a detached Develop worktree with only the new integration test copied in.

## Evidence

- `tests/issue_2392_calibration_metrics_finite.rs::extreme_finite_history_returns_numeric_metrics` failed on Develop with `success: true`, `meanAbsoluteError: null`, and `bias: null`; it passes on this branch.
- The original trigger and equivalent arithmetic variants are closed without a trivial bypass: finite inputs remain accepted, error differences and sums are scaled before they can overflow, unrepresentable final errors saturate to signed `f64::MAX`, overflowing ratios are excluded, and usable ratios are accumulated without an overflowing intermediate sum. The FFI summary uses this same calculation path.
- Calibration behaviour is documented in the `Calibration Summary` section of `docs/FFI_API.md`. This is a backend-only change; no visual surface changed.

## Test Plan

- `tests/issue_2392_calibration_metrics_finite.rs::extreme_finite_history_returns_numeric_metrics` — base red: `cargo test --test issue_2392_calibration_metrics_finite extreme_finite_history_returns_numeric_metrics -- --exact` failed because the successful FFI response contained a null `meanAbsoluteError`; final head: `cargo test --test issue_2392_calibration_metrics_finite` passed.
- `src/discovery_history.rs::tests::test_calibration_summary_stays_finite_for_extreme_finite_observations` and `src/discovery_history.rs::tests::test_calibration_factor_is_never_nan_for_finite_observations` — enabled and passed.
- `src/discovery_history.rs::tests::calibration_skips_non_finite_input_fields_and_keeps_valid_samples` — passed for NaN and both infinities in each input field, then a valid observation. Removing either recorder finiteness check alone made it fail at the empty-summary assertion.
- `src/discovery_history.rs::tests::calibration_factor_discards_overflowing_ratios_and_uses_neutral_fallback` — passed for all-overflowing ratios, mixed overflowing and usable ratios, and non-finite helper inputs. Removing either helper input-finiteness operand alone made it fail; removing the finite-ratio filter alone made it return NaN instead of `1.0`.
- `src/discovery_history.rs::tests::calibration_factor_sums_finite_ratios_without_overflow` — passed with overflowing intermediate sums and cancelling finite ratios, yielding approximately `0.4`; restoring direct summation made it fail.
- `src/discovery_history.rs::tests::calibration_errors_saturate_after_preserving_bias_cancellation` — passed for positive and negative saturation and cancellation; removing saturation made it fail with infinity.
- `src/discovery_history.rs::tests::calibration_summary_handles_empty_persisted_observation_bucket`, `src/discovery_history.rs::tests::calibration_summary_handles_zero_valued_observations`, and `src/discovery_history.rs::tests::calibration_factor_respects_both_clamp_boundaries` — passed. Removing either zero-scale guard alone made its corresponding empty or zero case fail with NaN.
- `cargo test --lib discovery_history::tests::calibration_` — 7 passed.
- `cargo test --test scoring issue_605_calibration_tracking::` — 13 passed, including ordinary metrics, empty FFI history, and near-zero predictions.
- `cargo fmt --all --check`, `cargo clippy --lib --tests --all-features -- -D warnings`, and `git diff --check` — passed.
- `timeout 900 ./quality.sh < /dev/null` — passed all checks, including the full test targets, documentation build, and release build.
- No existing test assertions were removed; the two existing calibration regressions had only their `#[ignore]` attributes removed.

## Merge verification

- Quality gate: passed on the final code and test changes.
- CI and code review: pending.
