## Summary

Config readers that switch a guard off on `0`, `false`/`no`, or an unparsable value now log one `tracing::warn!` naming the variable, the raw value and the guard disabled, so the default `warn` filter shows a disabled watchdog, reserve, memory gate and so on. Behaviour (return values) is unchanged. Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version. Closes #2468.

Sites covered (all 23 cases in the issue's two tables, plus the second `CONSTANT_SOURCE_EFFECT_THRESHOLD` reader): `FOCUS_RANKING_BUDGET_MS`, `ANALYSIS_RESERVE_MS`, `GPU_STALL_WINDOW_SECS` (`0`), `INSUFFICIENT_RECORDING_FRACTION`, `MIN_AVAILABLE_MEMORY_GB`, `DROUGHT_RESET_AFTER_EPOCHS`, `DROUGHT_ALARM_EPOCHS`, `CPU_PRE_REJECT`, `HIDDEN_SQUASH_PRUNE`, `STRICT_CANDIDATE_RECONCILIATION`, `WATCHDOG_STALL_SECS` (invalid value), `MAX_SOURCES_PER_TARGET`, `MAX_ACTIVATION_CONFIGS_PER_TARGET`, `CONSTANT_SOURCE_EFFECT_THRESHOLD` (both readers), `MIN_EXPECTED_GAIN`, `MIN_BYPASS_WEIGHT_FOR_COLLAPSE`, `REMOVE_LOW_IMPACT_NOISE_FLOOR` and `_UNITS`.

## Spec

### Intent and Rationale

- A guard turned off with no trace in the default log is invisible during an incident; one `WARN` per opt-out at the read site is the issue's suggested fix.
- Warnings sit in the readers (not a startup sweep) because the readers are the only place that knows the value was an opt-out rather than a default.

### Essential Design Decisions

- Warn only when the variable is explicitly set; unset (the default state) and valid non-opt-out values stay silent.
- Dedupe per `(variable, raw value)` per process (`warn_guard_disabled` / `warn_guard_invalid` in `src/config/helpers.rs`), because several readers run on every pass or per candidate. A changed value warns again.
- `WATCHDOG_STALL_SECS=0` does not warn: the watchdog is off by default, so `0` is the default state; only an unparsable value (a typo such as `3O`) warns, as the issue specifies.
- Existing `debug!` lines for opt-outs were replaced by the `WARN`; the existing `debug!` for an invalid `MIN_AVAILABLE_MEMORY_GB` is kept (it falls back to the armed default, so no guard is lost).

### Undiscoverable Facts

- None.

## Evidence

Security-fix evidence:

- Test file: `tests/issue_2468_guard_opt_out_warnings.rs`.
- Test identifier: `tests/issue_2468_guard_opt_out_warnings.rs::every_opt_out_logs_one_warn_naming_variable_value_and_guard`.
- Base-fail / head-pass: with `src/` restored to the base commit `39d1100`, the test failed for all 23 cases (`expected exactly one WARN ... got 0`, three with only a `DEBUG` event), and `repeated_opt_out_warns_once_and_a_new_value_warns_again` failed with no warnings; at head both pass.
- Original trigger closed: `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS=3O`, `NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS=0` and `NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB=0` each now reach `warn_guard_invalid` / `warn_guard_disabled` before the guard is left off, and `WARN` passes the default `EnvFilter::new("warn")`. The only silent variants left are the unset default and `WATCHDOG_STALL_SECS=0` (the default state, by design above); every non-zero spelling of a zero (`0.0`, `" 0 "`) is parsed by the same arm and warns.

## Acceptance Criteria

The issue states no acceptance-criteria heading; nothing it asks is left undone.

## Test Plan

- Added `tests/issue_2468_guard_opt_out_warnings.rs` (4 tests: `every_opt_out_logs_one_warn_naming_variable_value_and_guard`, `unset_variables_log_nothing`, `valid_and_empty_values_log_no_warn_and_keep_their_value`, `repeated_opt_out_warns_once_and_a_new_value_warns_again`): passed at head (`cargo test --test issue_2468_guard_opt_out_warnings`, 4 passed).
- Added two dedupe unit tests in `src/config/helpers.rs`.
- Red check: base `src/` restored, new test run: 2 of 4 failed as above (the other two are negative controls, expected green on base).
- Negative controls (`unset_variables_log_nothing`, `valid_and_empty_values_log_no_warn_and_keep_their_value`) were not mutation-checked against a deliberately over-eager warning.
- No existing test was edited or removed. Existing config tests (`issue_2006_numeric_env_override_trim`, `issue_2096_config_user_facing_a`/`_b`, `issue_2096_env_readers_outside_config`, `issue_2272_config_small_modules`, `issue_2405_watchdog_stall_timeout_truncation`) pass.
- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` clean.
- `cargo test --lib`: 1770 passed, 1 failed (`analysis::utils::platform::tests::test_apply_xdg_runtime_dir_sets_when_unset`); it passes when run alone, so it is an environment-variable race with other lib tests, unrelated to this change.
- Docs checked: `docs/CONFIGURATION.md` (note added; watchdog row no longer says "no log"), doc comments on every changed reader. Left as-is: `docs/DROUGHT_PLAYBOOK.md:393` (a different variable).
- Callers checked: the readers are called from existing code paths whose return values did not change; the helper warns only after the decision is made.

## Quality gate

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeds the 600 s foreground cap; its bash-syntax, shellcheck, cargo-deny, fmt and clippy stages and the lib tests were run individually; left to CI for the integration binaries" -->

`./quality.sh` ran past the bash-syntax, shellcheck, install-pinning, layout and licence stages into the lib tests before the 595 s cap cut it off, with no failure printed. The integration test binaries were not run in full.

Not done: full integration suite; a mutation check of the negative controls.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
