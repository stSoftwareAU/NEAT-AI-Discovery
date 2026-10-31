## Summary

Capped positive `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS` values at 24 hours, logged the supplied value when clamping, and added regression coverage for the millisecond truncation inputs. Closes #2405.

Version bump: `Cargo.toml` matches current `Develop` at `0.74.314`; the PR diff carries no separate version change.

## Spec

### Intent and Rationale

- The watchdog compared milliseconds via `as_millis() as u64`; bounding the accessor prevents oversized values from wrapping to a short timeout.

### Essential Design Decisions

- Values above `MAX_WATCHDOG_STALL_SECS` (86,400 seconds) clamp before `Duration::from_secs` and produce a warning with the environment variable, supplied value and cap.
- Unset, zero, whitespace and invalid values retain their existing behaviour.

### Undiscoverable Facts

- `WatchdogConfig::from_env` is the production reader and gets the stall timeout only from `config::watchdog_stall_timeout`; its result is capped before reaching `watchdog_loop`.
- Current `origin/Develop` was merged into the branch; its version `0.74.314` was retained after resolving the manifest conflict.

## Evidence

- Regression test: `tests/issue_2405_watchdog_stall_timeout_truncation.rs::oversized_values_keep_the_effective_millisecond_threshold_within_u64`.
- On the unfixed `origin/Develop` accessor, the regression test failed for `18446744073709552` (the reported 384 ms wraparound input), `2305843009213693952` (the zero-wrap input) and `u64::MAX`; the unbounded millisecond values exceeded `u64`. The same test passed with the clamp and failed again with the clamp removed.
- Trigger closure: all production watchdog configuration reaches the capped accessor through `WatchdogConfig::from_env`. Every parsed positive value above 86,400 is clamped before duration construction, so `watchdog_loop` receives at most 86,400,000 milliseconds, which fits in `u64`. No production watchdog configuration path bypasses that accessor.
- `docs/CONFIGURATION.md`, `docs/GPU_GUIDE.md` and the Issue #2405 row in `docs/audits/security-sweep-chunk-13-config-env.md` describe the current cap while preserving the audit's historical finding.
- Backend-only change; no browser evidence applies.

## Test Plan

- `timeout 900s cargo test --test issue_2405_watchdog_stall_timeout_truncation -- --test-threads=1 < /dev/null` — passed, 4 tests. Covers the reported input, zero-wrap input, `u64::MAX`, the 86,400/86,401 boundary, warning fields and unchanged disabled behaviour.
- The same test file against unfixed `origin/Develop` — failed, exit 101; three clamp-specific tests failed and the unchanged disabled-behaviour test passed. Removing the clamp reproduced the failure.
- `timeout 900s cargo test --test issue_2006_numeric_env_override_trim -- --test-threads=1 < /dev/null` — passed, 14 tests.
- `timeout 900s cargo test --test issue_2096_config_user_facing_a -- --test-threads=1 < /dev/null` — passed, 16 tests.
- `timeout 900s cargo test --test infrastructure config_watchdog_stall -- --test-threads=1 < /dev/null` — passed, 2 tests.
- `timeout 120s cargo fmt --all -- --check < /dev/null` — passed.
- `timeout 900s cargo clippy --test issue_2405_watchdog_stall_timeout_truncation -- -D warnings < /dev/null` — passed.
- `timeout 900s cargo build --release --lib < /dev/null` — passed.
- `timeout 900s ./quality.sh < /dev/null` — not completed within the command bound in three attempts. The first ended during test-target compilation; the second returned exit 124 during release compilation after tests and documentation passed; the third ended during integration tests without a complete result. The full gate remains unverified as a single successful run.
- `git diff --check` — passed. No test assertions were removed from existing tests; the edit in `tests/issue_2096_config_user_facing_a.rs` only updates a comment.
