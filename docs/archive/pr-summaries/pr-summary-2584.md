## Summary

`benchmark_compare.sh` now compares the Criterion point estimate (the middle value of `change: [lower point upper]`) against the threshold, as `docs/BENCHMARKS.md` ("Interpreting Results") documents. The old greedy `.*` captured the upper confidence bound, so a statistically insignificant change such as `[-1.2% +2.0% +6.0%]` failed the gate at the 5% threshold. Closes #2584.

Version bump: left to CI's `version-increment` job (AGENTS.md "Version Bumps"); not changed here.

## Spec

### Intent and Rationale

- The parse regex now matches all three bracketed values and captures only the middle one, so both the regression and improvement checks use the point estimate.

### Essential Design Decisions

- A `change:` line that is not a three-value bracket no longer counts as a comparison; real Criterion output always has three values.

### Undiscoverable Facts

- None.

## Evidence

Backend/CLI-only change. New sandbox test `comparison_uses_point_estimate_not_upper_bound` drives the real script with a stub `cargo`: a noisy line `[-1.2000% +2.0000% +6.0000%]` is not a regression, and `[-9.0000% -6.5000% -1.0000%]` counts as an improvement.

## Test Plan

- Added `comparison_uses_point_estimate_not_upper_bound` in `tests/issue_2554_benchmark_compare_failure_is_loud.rs`; it goes red with the old regex restored (verified).
- Edited expectations in the same file: `+14.0000%` → `+12.5000%` and `+18.0000%` → `+16.5000%` (point estimates of the unchanged stub lines, as the issue specifies); with the old regex restored the four edited tests fail, so they still guard the parse.
- `cargo test --test issue_2554_benchmark_compare_failure_is_loud`, `issue_2572_benchmark_compare_requires_bc`, `issue_1918_benchmark_threshold_validation`: passed.
- `./quality.sh` on the final head stopped at `regression::issue_576_benchmark_regression_tracking::benchmark_compare_without_baseline_reports_error`: this host has no `bc` (`command -v bc` finds nothing), so the script exits with the #2572 "bc is not installed" error before that older test's expected text. It is unrelated to the parse change and is not touched by it; the later stages of the gate were not reached here. CI, which has `bc`, runs the full gate.
