## Summary

Fixed regression name lookup in `benchmark_compare.sh` and added a suite-name
fallback, so regressions and later suite results appear in the final summary
before the script exits 1. Added regression-path coverage and updated the
benchmark guide. Closes #2571.

Version bump: `Cargo.toml` updated for this change; the merged commit records
the final version.

## Spec

### Intent and Rationale

- Criterion's `change:` line contains regex metacharacters; fixed-string matching
  lets the comparison report the benchmark instead of stopping at the first
  regression.

### Essential Design Decisions

- A missing benchmark name falls back to the suite name, while a failed
  `cargo bench` remains counted as a failure. Comparison continues through
  later suites and returns 1 after printing the summary when regressions or
  failures occurred.

### Undiscoverable Facts

- The three new regression-path tests failed against `origin/Develop`'s
  `benchmark_compare.sh` and passed after the fix.

## Evidence

- Backend/CLI change; verified through the sandbox tests, Bash syntax and
  ShellCheck gates, and the full quality gate. The regression tests provide
  Criterion-style `time:` and `change:` output and use a numeric `bc` stub.
- Version `0.74.366` is recorded in `Cargo.toml`, `Cargo.lock`, and
  `fuzz/Cargo.lock`.
- Execution was on Linux; macOS/BSD userland was not run.

## Test Plan

- `tests/issue_2554_benchmark_compare_failure_is_loud.rs::comparison_reports_multiple_regressions_and_continues_to_later_suites`: failed against `origin/Develop` production code, passed with the fix.
- `tests/issue_2554_benchmark_compare_failure_is_loud.rs::comparison_uses_suite_name_when_criterion_name_is_missing`: failed against `origin/Develop` production code, passed with the fix.
- `tests/issue_2554_benchmark_compare_failure_is_loud.rs::comparison_reports_regression_and_later_suite_failure`: failed against `origin/Develop` production code, passed with the fix.
- `cargo test --test issue_2554_benchmark_compare_failure_is_loud < /dev/null`: passed, 12 tests.
- `cargo fmt --all -- --check`, `bash -n benchmark_compare.sh`, and `shellcheck benchmark_compare.sh`: passed.
- `./quality.sh < /dev/null`: passed, including dependency audit, all tests, documentation build, and release build.
- Removed assertions: none.
