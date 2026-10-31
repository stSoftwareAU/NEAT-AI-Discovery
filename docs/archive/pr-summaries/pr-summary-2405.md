## Summary

Capped `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS` at 24 hours and added a warning when a larger value is clamped. The cap keeps the watchdog's millisecond threshold representable without truncation. Closes #2405.

Version bump: `Cargo.toml` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- A very large stall timeout could wrap when `watchdog_loop` cast its millisecond value to `u64`; the accessor now bounds the input before constructing the duration.

### Essential Design Decisions

- Positive values above 86,400 seconds clamp to the named 24-hour maximum and log the environment variable, supplied value and cap.
- Unset, zero, whitespace handling and invalid-input behaviour remain unchanged. Issue #2404's abort-delay configuration is untouched.

### Undiscoverable Facts

- `tests/issue_2405_watchdog_stall_timeout_truncation.rs::oversized_values_keep_the_effective_millisecond_threshold_within_u64` failed against `origin/Develop` with all three oversized inputs and passed at the fixed head. Removing the clamp also made it fail.
- The test reads the real accessor and does not start a watchdog thread; process-abort behaviour was not exercised because it would abort the test process.

## Evidence

- Regression test: `tests/issue_2405_watchdog_stall_timeout_truncation.rs::oversized_values_keep_the_effective_millisecond_threshold_within_u64` covers `18446744073709552`, `2305843009213693952` and `u64::MAX`. Against the base accessor it reported thresholds of 18446744073709552000, 2305843009213693952000 and 18446744073709551615000 milliseconds, all above the expected 86,400-second cap. It passed on the fixed head.
- Trigger closure: every parsed positive `u64` is clamped before `Duration::from_secs`; therefore the original input and larger values reach `watchdog_loop` only as the 86,400-second maximum. The resulting 86,400,000 ms threshold fits in `u64`.
- The `docs/GPU_GUIDE.md` Hang Watchdog section and the `WATCHDOG_STALL_SECS` configuration row now state the maximum. The security-sweep archive distinguishes its historical observation from the current fix.

## Test Plan

- `timeout 900 cargo test --test issue_2405_watchdog_stall_timeout_truncation -- --test-threads=1 < /dev/null` — passed, 4 tests. Includes the base-code red run and the temporary clamp-removal red run described above.
- `timeout 900 cargo test --test issue_2006_numeric_env_override_trim -- --test-threads=1 < /dev/null` — passed, 14 tests; whitespace, zero and invalid-value contracts remain intact.
- `timeout 900 cargo test --test issue_2096_config_user_facing_a -- --test-threads=1 < /dev/null` — passed, 16 tests, including invalid stall-timeout inputs.
- `timeout 900 cargo test --test infrastructure config_watchdog_stall -- --test-threads=1 < /dev/null` — passed, 2 watchdog configuration tests.
- `timeout 120 cargo fmt --all -- --check < /dev/null` — passed after formatting.
- `timeout 900 cargo clippy --test issue_2405_watchdog_stall_timeout_truncation -- -D warnings < /dev/null` — passed.
- `timeout 900 cargo check --test issue_2405_watchdog_stall_timeout_truncation < /dev/null` — passed.
- `git diff --check` — passed. No assertions were removed from existing tests.
- `timeout 900 ./quality.sh < /dev/null` — not completed. Two attempts exited 124 at the test stage: the first during test-target compilation; the second after library tests began (`running 1745 tests`) and while they were still passing. A third attempt reached test-target compilation, then was stopped when the run-budget file reported 342 seconds remaining. Resume with `timeout 900 ./quality.sh < /dev/null` from this branch; the test targets are now built.
