## Summary

Capped positive `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS` values at 24 hours (`MAX_WATCHDOG_STALL_SECS = 86_400` in `src/config/user_facing.rs`), logged a warning naming the variable, supplied value and cap when clamping, documented the cap, and added regression coverage for the millisecond truncation inputs. Closes #2405.

Version bump: patch version `0.74.324` → `0.74.325` in `Cargo.toml` and the `neat_ai_discovery` entry of `Cargo.lock`, per AGENTS.md.

## Spec

### Intent and Rationale

- `watchdog_loop` compares against `stall_timeout.as_millis() as u64`; a huge seconds value wrapped to a short threshold (384 ms for `18446744073709552`, 0 ms for `2305843009213693952`) and aborted a healthy process. Clamping at the accessor (the issue's preferred option) keeps the millisecond value far inside `u64` and keeps both watchdog fixes in one accessor file.

### Essential Design Decisions

- Values above `MAX_WATCHDOG_STALL_SECS` (86,400 s) clamp before `Duration::from_secs` and emit one `WARN` with fields `env`, `value` and `max`.
- Unset, zero, whitespace and invalid values keep their existing behaviour (disabled); the silent-opt-out class is #2213's and is not changed here.
- `watchdog_loop` is left unchanged: its `as_millis() as u64` is now bounded by the accessor (at most 86,400,000).

### Undiscoverable Facts

- `WatchdogConfig::from_env` in `src/watchdog.rs` is the production reader and takes the stall timeout only from `config::watchdog_stall_timeout`, so the cap reaches `watchdog_loop`.

## Evidence

- Regression test: `tests/issue_2405_watchdog_stall_timeout_truncation.rs::oversized_values_keep_the_effective_millisecond_threshold_within_u64`.
- Against the unfixed accessor (base `src/config/user_facing.rs`), that test failed for `18446744073709552` (the reported 384 ms wraparound input), `2305843009213693952` (zero wrap) and `u64::MAX`; it passes with the clamp and failed again when the clamp was removed.
- Original trigger closed, no trivial bypass: the only production path from the environment variable to `watchdog_loop` is `WatchdogConfig::from_env` → `watchdog_stall_timeout`. Every parsed positive value above 86,400 is clamped to 86,400 before `Duration::from_secs`, so the threshold is at most 86,400,000 ms, which fits in `u64`; no value, including `u64::MAX`, can wrap to a short threshold, and no other path builds a stall timeout.
- Docs updated: `docs/CONFIGURATION.md`, `docs/GPU_GUIDE.md`, the module doc in `src/watchdog.rs`, and the Issue #2405 row in `docs/audits/security-sweep-chunk-13-config-env.md` (historical finding preserved).
- Review correction: restored `live` as the stall-timeout accessor read mode in the 13b audit table and retained the current guard-weakening finding (a typo silently disables the watchdog), alongside the historical truncation finding and its 24-hour cap fix.
- Review correction: marked #2405 fixed in the 13b findings list and updated the documentation-reconciliation row to record both watchdog caps.
- Backend-only change; no browser evidence applies.

## Test Plan

- `cargo test --test issue_2405_watchdog_stall_timeout_truncation --test issue_2096_config_user_facing_a` — passed (20 tests, run on the final head). The 2405 file covers the reported input, zero-wrap input, `u64::MAX`, the 86,400/86,401 boundary, warning fields and unchanged disabled behaviour.
- Red check: removing the clamp makes the clamp-specific 2405 tests fail; the disabled-behaviour test passes either way (it pins current behaviour).
- `tests/issue_2096_config_user_facing_a.rs`: only a comment changed; no assertions removed.
- Earlier runs on this branch: `issue_2006_numeric_env_override_trim`, `infrastructure config_watchdog_stall`, `cargo fmt --check`, `cargo clippy` on the new test and `cargo build --release --lib` passed.
- `./quality.sh`: the formatting, lint and type-check stages passed on the final head; the unit-test stage reported no failure but was cut off at the 595 s command bound (exit 124) before finishing, so the full gate is unverified as a single run.

<!-- vibe-quality-gate-skipped reason="full gate exceeds the 595 s foreground command bound on this host; formatting, lint and type checks passed, targeted tests passed; CI runs the full gate" -->

- Mutation check: it reported "baseline tests fail before mutation"; at the head the targeted suites (`issue_2405_watchdog_stall_timeout_truncation`, `issue_2096_config_user_facing_a`) pass (20 tests) after the version bump, so the baseline failure is not reproducible here and is likely the checker's own environment (the full unit stage exceeds the host's command bound).
