## Summary

`benchmark_compare.sh` now compares the Criterion point estimate (the middle value of `change: [lower point upper]`) against the threshold, as `docs/BENCHMARKS.md` ("Interpreting Results") documents. The old greedy `.*` captured the upper confidence bound, so a statistically insignificant change such as `[-1.2% +2.0% +6.0%]` failed the gate at the 5% threshold. Closes #2584.

Version bump: `Cargo.toml` updated for this change (`Cargo.lock` and `fuzz/Cargo.lock` follow); the merged commit records the final version.

## Spec

### Intent and Rationale

- The parse regex now matches all three bracketed values and captures only the middle one, so both the regression and improvement checks use the point estimate.
- Each value accepts Criterion 0.8's U+2212 minus sign as well as ASCII `-`/`+`, and an optional fractional part (Criterion prints values of 10000% or more without one). The captured point estimate is normalised to ASCII `-` before the threshold check and `bc`.

### Essential Design Decisions

- A `change: [` line the parser cannot read exits 1 with an error naming the suite (as #2554/#2572 do) rather than being dropped, which would have counted the suite as a GPU skip and exited 0.

### Undiscoverable Facts

- None.

## Evidence

Backend/CLI-only change. The sandbox tests drive the real script with a stub `cargo` that emits Criterion 0.8's real format (U+2212 for negative values): a noisy line `[−1.2000% +2.0000% +6.0000%]` is not a regression, `[−9.0000% −6.5000% −1.0000%]` counts as an improvement, `[−0.5000% +7.0000% +14.000%]` is a `+7.0000%` regression that is counted, `[+9000.0000% +12000% +15000%]` (no fractional part) is parsed, and an unreadable `change: [` line exits 1.

## Test Plan

- Added in `tests/issue_2554_benchmark_compare_failure_is_loud.rs`: `comparison_uses_point_estimate_not_upper_bound`, `negative_lower_bound_with_regressing_point_estimate_is_a_regression`, `change_values_of_ten_thousand_percent_without_fraction_are_parsed` and `unreadable_change_line_fails_loudly`. The stub's negative values now use U+2212 as Criterion 0.8.2 does, so the fixtures match real output.
- Edited expectations in the same file: `+14.0000%` → `+12.5000%` and `+18.0000%` → `+16.5000%` (point estimates of the unchanged stub lines, as the issue specifies) in `comparison_reports_wrapped_criterion_benchmark_id`, `comparison_reports_regression_and_later_suite_failure`, `comparison_uses_suite_name_when_criterion_name_is_missing` and `comparison_reports_multiple_regressions_and_continues_to_later_suites`. Obsolete base behaviour: the script reported the upper confidence bound. Remaining behaviour: the point estimate is reported and compared. Covering tests: those four, in `tests/issue_2554_benchmark_compare_failure_is_loud.rs`.
- `cargo test --test issue_2554_benchmark_compare_failure_is_loud`: 17 passed. `issue_2572_benchmark_compare_requires_bc` and `issue_1918_benchmark_threshold_validation` passed. `cargo fmt --check`, `cargo clippy --tests` and `shellcheck benchmark_compare.sh` are clean.
- `./quality.sh` was not run end to end here: this host has no `bc`, so `regression::issue_576_benchmark_regression_tracking::benchmark_compare_without_baseline_reports_error` stops at the #2572 "bc is not installed" error. It is unrelated to the parse change; CI, which has `bc`, runs the full gate.
