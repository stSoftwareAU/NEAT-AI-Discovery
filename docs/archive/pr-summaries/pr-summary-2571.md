## Summary

Fixed regression name lookup in `benchmark_compare.sh` and added a suite-name
fallback, so regressions and later suite results appear in the final summary
before the script exits 1. Added regression-path coverage and updated the
benchmark guide. Closes #2571.

Version bump: `Cargo.toml` updated for this change; the merged commit records
the final version.

## Spec

### Intent and Rationale

- Criterion's `change:` line contains regex metacharacters. Tracking the name
  while reading timings avoids a second lookup and reports each regression.

### Essential Design Decisions

- A missing benchmark name falls back to the suite name, while a failed
  `cargo bench` remains counted as a failure. Comparison continues through
  later suites and returns 1 after printing the summary when regressions or
  failures occurred.
- Criterion puts IDs longer than 23 characters on the previous non-blank line;
  those IDs are retained for the subsequent change report.

### Undiscoverable Facts

- The three new regression-path tests failed against `origin/Develop`'s
  `benchmark_compare.sh` and passed after the fix.

## Evidence

- Backend/CLI change; verified through the sandbox tests, Bash syntax and
  ShellCheck gates, and the full quality gate. The regression tests provide
  Criterion-style `time:` and `change:` output and use a numeric `bc` stub.
- The merged tree records version `0.74.368`; `fuzz/Cargo.lock` is aligned
  with `Cargo.toml` after merging Develop.
- Execution was on Linux; macOS/BSD userland was not run.

## Test Plan

- `tests/issue_2554_benchmark_compare_failure_is_loud.rs::comparison_reports_multiple_regressions_and_continues_to_later_suites`: failed against `origin/Develop` production code, passed with the fix.
- `tests/issue_2554_benchmark_compare_failure_is_loud.rs::comparison_uses_suite_name_when_criterion_name_is_missing`: failed against `origin/Develop` production code, passed with the fix.
- `tests/issue_2554_benchmark_compare_failure_is_loud.rs::comparison_reports_regression_and_later_suite_failure`: failed against `origin/Develop` production code, passed with the fix.
- `tests/issue_2554_benchmark_compare_failure_is_loud.rs::comparison_reports_wrapped_criterion_benchmark_id` covers a long ID on its own line, including the final summary.
- The former lookup of names only on `time:` lines is replaced by tracking
  timing names and the preceding non-blank ID. Inline names and missing-name
  fallback remain covered by `tests/issue_2554_benchmark_compare_failure_is_loud.rs::comparison_reports_multiple_regressions_and_continues_to_later_suites` and `tests/issue_2554_benchmark_compare_failure_is_loud.rs::comparison_uses_suite_name_when_criterion_name_is_missing`.
- `cargo test --test issue_2554_benchmark_compare_failure_is_loud < /dev/null`: passed, 12 tests.
- `cargo fmt --all -- --check`, `bash -n benchmark_compare.sh`, and `shellcheck benchmark_compare.sh`: passed.
- `./quality.sh < /dev/null`: passed, including dependency audit, all tests, documentation build, and release build.
- Removed assertions: none.
