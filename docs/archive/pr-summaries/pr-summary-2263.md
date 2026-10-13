# PR Summary — Issue #2263

## Summary

Closes #2263. Chunk 13c-1 of #2123 (baseline `b85a551`): adds
`tests/issue_2096_config_user_facing_b.rs`, 29 tests that pin how the
`src/config/user_facing.rs` accessors from the focus-ranking budget to the
drought reset treat hostile environment values. Test-only; nothing in `src/`
changes, so no version bump is owed (CI's `version-increment` job handles it).

- [x] Survey accessors and resolvers, write the table-driven tests
- [x] Red-check the opt-out guards
- [x] `./quality.sh` passes
- [x] Independent spec and standards review

### Behaviour that differs from the parent's expectations (for 13c-3)

Pinned as the code does it today, not changed:

- **(a) Non-literal zero spellings disable a guard.** The `u64`/`u32` guards
  (`FOCUS_RANKING_BUDGET_MS`, `ANALYSIS_RESERVE_MS`, `DROUGHT_ALARM_EPOCHS`,
  `DROUGHT_RESET_AFTER_EPOCHS`) accept `0`, `00` and `+0` as opt-outs, and fall
  back to the armed default for `0.0`, `-0` and `0e5`. The `f64` guards
  (`INSUFFICIENT_RECORDING_FRACTION`, `MIN_AVAILABLE_MEMORY_GB`) accept all six
  spellings (`0`, `00`, `+0`, `0.0`, `-0`, `0e5`) as opt-outs.
- **(b) `sample_program_override` returns a padded value untrimmed.** Empty and
  whitespace-only values give `None`, but `"  x  "` comes back with its
  padding.
- **(c) Two accessors do not fall back where the name suggests.**
  `focus_ranking_perf_cliff_ms` accepts `u64::MAX` unclamped, and
  `remove_neuron_drought_factor` clamps `0` and `-1` up to `0.001` instead of
  falling back to its default.

## Spec

### Intent and Rationale

- Garbage in an env var must never silently disable a guard that `0` opts out of.
- Pin current behaviour so 13c-3 can record the differences above.

### Essential Design Decisions

- One `env_table_test!` macro generates one `#[serial]` table test per accessor.
- A file-local `EnvGuard` sets, unsets and restores on `Drop`.
- OnceLock-cached accessors are tested through their pure resolvers.

### Undiscoverable Facts

- `u64`/`u32` parsing accepts `00` and `+0` but rejects `0.0`, `-0` and `0e5`; `f64` accepts all six.

## Evidence

- `tests/issue_2096_config_user_facing_b.rs`: 495 lines, 29 tests, all passing.
- `timeout 900 ./quality.sh < /dev/null` exited 0 ("All quality checks passed!") on head `175ba22`.
- **Docs sweep:** section: none changed (test-only; the module doc lists what is deliberately not tested). siblings: `tests/issue_2096_config_user_facing_a.rs`, `tests/issue_1422_drought_mitigation_config.rs`, `tests/issue_3172_focus_ranking_budget_scaling.rs`, `tests/issue_1934_sample_fallback.rs`; no docs name the new file.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- File exists and every listed accessor and resolver has a hostile-input assertion. reviewer: met (the reviewer noted the `scale_default_focus_ranking_budget_ms` test pins the ceiling only; the lazy base is already above the minimum, so the floor cannot bite, as the test's comment says).
- Each of the six `0` opt-outs: garbage keeps the guard active, asserted on guard state. reviewer: met.
- Zero-spelling behaviour pinned per opt-out. reviewer: met.
- `sample_program_override`: empty, whitespace-only, untrimmed padded value. reviewer: met.
- No `src/` changes, env mutation `#[serial]` and restored on drop. reviewer: met.
- `./quality.sh` passes. reviewer: met on the stated evidence (not re-run by the reviewer).
- Unrequested: `sample_program_override_is_none_when_unset` (overlaps `tests/issue_1934_sample_fallback.rs`, kept as a cheap guard for the same table). reviewer: minor, harmless.
- Unrequested: extra range pins (`"2"` for the recording fraction, `"65"` for the memory ceiling, `5` for the budget minimum, and in-range scale cases). reviewer: minor, harmless.
- PR-summary list of differing behaviour. reviewer: not verifiable from the diff; supplied above as (a), (b), (c).

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

reviewer: no violations. `CODING-STANDARDS.md` does not exist in this repository, so the diff was checked against `AGENTS.md` and `CONTRIBUTING.md`. Named test files exist, no stubs are added, no workflow is touched, every env-mutating test is `#[serial]` with restore on drop, and `unsafe` blocks carry SAFETY comments.

## Test Plan

Branch outcomes: none added

The pin tests are expected green on base because no production change was made.

Red-checks (each mutation was applied to `src/`, one test went red, then the file was restored byte-identical):

- Inverting `focus_ranking_budget_ms().is_some()` went red in `garbage_never_disables_any_zero_opt_out`.
- Inverting `drought_reset_after_epochs().is_some()` went red in `garbage_never_disables_any_zero_opt_out`.
- Inverting `min_available_memory_gb() > 0.0` went red in `garbage_never_disables_any_zero_opt_out`.
- Flipping `("00", true)` to `false` went red in `integer_opt_outs_only_accept_integer_zero_spellings`.
- Changing the padded expectation to the trimmed value went red in `sample_program_override_keeps_a_padded_value_untrimmed`.

Caveat: the first three mutations land in the same test, so this does not show each assertion is individually load-bearing.

Run: `cargo test --test issue_2096_config_user_facing_b` (29 passed), clippy and fmt clean, `./quality.sh` green.
