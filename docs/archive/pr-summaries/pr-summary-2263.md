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

- **met** — `tests/issue_2096_config_user_facing_b.rs` exists, and every listed accessor and resolver has at least one hostile-input assertion — evidence: `tests/issue_2096_config_user_facing_b.rs::focus_ranking_budget_ms_clamps_disables_and_falls_back`, `tests/issue_2096_config_user_facing_b.rs::effective_focus_ranking_budget_ms_scales_only_the_default`, `tests/issue_2096_config_user_facing_b.rs::analysis_reserve_ms_clamps_and_falls_back`, `tests/issue_2096_config_user_facing_b.rs::analysis_reserve_fraction_clamps_and_falls_back`, `tests/issue_2096_config_user_facing_b.rs::insufficient_recording_fraction_disables_on_zero_only`, `tests/issue_2096_config_user_facing_b.rs::focus_ranking_perf_cliff_ms_accepts_huge_unclamped`, `tests/issue_2096_config_user_facing_b.rs::session_ttl_secs_clamps_and_falls_back`, `tests/issue_2096_config_user_facing_b.rs::max_wall_clock_minutes_clamps_and_falls_back`, `tests/issue_2096_config_user_facing_b.rs::low_success_rate_threshold_ignores_out_of_range`, `tests/issue_2096_config_user_facing_b.rs::conservative_mode_max_epochs_ignores_zero_and_garbage`, `tests/issue_2096_config_user_facing_b.rs::conservative_gain_multiplier_floors_at_one`, `tests/issue_2096_config_user_facing_b.rs::drought_log_threshold_ignores_zero_and_garbage`, `tests/issue_2096_config_user_facing_b.rs::drought_alarm_epochs_disables_on_zero_only`, `tests/issue_2096_config_user_facing_b.rs::remove_neuron_drought_factor_clamps_into_the_valid_band`, `tests/issue_2096_config_user_facing_b.rs::novelty_suppression_ratio_ignores_zero_and_garbage`, `tests/issue_2096_config_user_facing_b.rs::novelty_gain_relaxation_ignores_zero_and_garbage`, `tests/issue_2096_config_user_facing_b.rs::min_available_memory_gb_disables_on_zero_only`, `tests/issue_2096_config_user_facing_b.rs::drought_reset_after_epochs_disables_on_zero_only`, `tests/issue_2096_config_user_facing_b.rs::resolve_insufficient_recording_fraction_matches_the_table`, `tests/issue_2096_config_user_facing_b.rs::resolve_drought_alarm_epochs_threads_the_default_through`, `tests/issue_2096_config_user_facing_b.rs::resolve_remove_neuron_drought_factor_threads_the_default_through`, `tests/issue_2096_config_user_facing_b.rs::resolve_min_available_memory_gb_threads_the_default_through`, `tests/issue_2096_config_user_facing_b.rs::scale_default_focus_ranking_budget_ms_saturates_and_clamps` (and the `sample_program_override` tests below) — reviewer: met (minor gap noted: the scale test reaches the `u64::MAX` saturation and the upper clamp only; the lazy base is already above `FOCUS_RANKING_BUDGET_MIN_MS`, so the floor cannot be reached through the public API, as the test's comment says)
- **met** — For each of the six `0` opt-outs, garbage input keeps the guard active, asserted on the guard's state — evidence: `tests/issue_2096_config_user_facing_b.rs::garbage_never_disables_any_zero_opt_out` (asserts `focus_ranking_budget_ms().is_some()`, `analysis_reserve_ms() != 0`, `insufficient_recording_fraction().is_some()`, `drought_alarm_epochs().is_some()`, `drought_reset_after_epochs().is_some()`, `min_available_memory_gb() > 0.0`) — reviewer: met
- **met** — Zero-spelling behaviour is pinned for each opt-out — evidence: `tests/issue_2096_config_user_facing_b.rs::integer_opt_outs_only_accept_integer_zero_spellings` (the four integer guards), `tests/issue_2096_config_user_facing_b.rs::float_opt_outs_accept_every_zero_spelling` (the two `f64` guards) — reviewer: met
- **met** — `sample_program_override` has tests for empty, whitespace-only and untrimmed padded input — evidence: `tests/issue_2096_config_user_facing_b.rs::sample_program_override_drops_blank_values`, `tests/issue_2096_config_user_facing_b.rs::sample_program_override_keeps_a_padded_value_untrimmed` — reviewer: met
- **met** — No `src/` changes; every env mutation is inside a `#[serial]` test and restored on drop — evidence: `git diff --stat cde7d18..HEAD -- src/` is empty; `tests/issue_2096_config_user_facing_b.rs::EnvGuard` `impl Drop` restores the prior value; every env-touching test is `#[serial]` via `env_table_test!` or explicitly — reviewer: met
- **met** — `./quality.sh` passes — evidence: `timeout 900 ./quality.sh < /dev/null` exited 0 ("All quality checks passed!") on head `175ba22`; `cargo test --test issue_2096_config_user_facing_b` 29 passed — reviewer: not re-run by the reviewer (it re-ran the targeted test binary, 29 of 29 passed)
- **met** — PR summary lists every place behaviour differs from the parent's expectations, for 13c-3 — evidence: `docs/archive/pr-summaries/pr-summary-2263.md` § "Behaviour that differs from the parent's expectations" items (a), (b), (c) — reviewer: met (documentation only, records the differences for 13c-3)
- **unrequested** — `tests/issue_2096_config_user_facing_b.rs::sample_program_override_is_none_when_unset` — reviewer: unrequested, harmless — reason: a sanity baseline for the unset case under `EnvGuard::unset` and `#[serial]`, so the blank-value assertions are not vacuous
- **unrequested** — Extra range pins: `"2"` → `Some(1.0)` in `recording_fraction_cases`, `"65"` → default in `min_memory_cases`, `"5"` → `FOCUS_RANKING_BUDGET_MIN_MS` in `focus_ranking_budget_ms_clamps_disables_and_falls_back`, and the in-range cases in `scale_default_focus_ranking_budget_ms_saturates_and_clamps` — reviewer: unrequested, harmless — reason: they pin the documented clamp bounds the issue's Failure Detection names ("escape an accessor's documented clamp"); the in-range scale cases mildly overlap `tests/issue_3172_focus_ranking_budget_scaling.rs` and are kept to make the saturation arithmetic readable

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — `CODING-STANDARDS.md` does not exist in this repository, so the reviewer checked `tests/issue_2096_config_user_facing_b.rs` against `AGENTS.md` and `CONTRIBUTING.md`: Australian English, `#[serial]` on every env-mutating test with `EnvGuard` restore on drop (pure-resolver tests touch no env), SAFETY comments on all three `unsafe` blocks, no stubs, file size (495 lines), symbols cited rather than line numbers, outcome-only assertions with no timing, positive precondition for the garbage test via the zero-spelling tables; `cargo test` 29/29, `rustfmt --check` and `cargo clippy -D warnings` clean. No violations.
- **clean** — Non-blocking nits noted, not changed: `480_000`/`500_000` are hard-coded beside a constant-built value, `FOCUS_RANKING_BUDGET_MS_PER_PROJECTED_MB == 20` pins a tuning constant, and `EnvGuard` is another file-local copy following the existing pattern.

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
