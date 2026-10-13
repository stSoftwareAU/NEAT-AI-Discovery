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

## Acceptance Criteria

<․!-- vibe-spec-review inputs="diff+issue-body" --․>

- **met** — tests/issue 2096 config user facing b.rs exists, and every accessor and resolver listed above has at least one hostile-input assertion. — evidence: `tests/issue 2096 config user facing b.rs: one env table test! per accessor (focus ranking budget ms clamps disables and falls back, effective focus ranking budget ms scales only the default, analysis reserve ms clamps and falls back, analysis reserve fraction clamps and falls back, focus ranking per` — reviewer: met
- **met** — For each of the six 0 opt-outs, a test asserts that garbage input keeps the guard active, asserted on the guard's state. — evidence: `tests/issue 2096 config user facing b.rs::garbage never disables any zero opt out asserts focus ranking budget ms().is some(), analysis reserve ms() != 0, insufficient recording fraction().is some(), drought alarm epochs().is some(), drought reset after epochs().is some() and min available memory gb` — reviewer: met
- **met** — Zero-spelling behaviour is pinned for each opt-out. — evidence: `tests/issue 2096 config user facing b.rs::integer opt outs only accept integer zero spellings (FOCUS RANKING BUDGET MS, ANALYSIS RESERVE MS, DROUGHT ALARM EPOCHS, DROUGHT RESET AFTER EPOCHS over 0/00/+0/0.0/-0/0e5) and tests/issue 2096 config user facing b.rs::float opt outs accept every zero spelli` — reviewer: met
- **met** — sample program override has tests for empty input, whitespace-only input and an untrimmed padded value. — evidence: `tests/issue 2096 config user facing b.rs::sample program override drops blank values ("" and " " give None) and tests/issue 2096 config user facing b.rs::sample program override keeps a padded value untrimmed` — reviewer: met
- **met** — No src/ changes. Every env mutation happens inside a [serial] test and is restored on drop. — evidence: `git diff --stat origin/milestone/2083-security-scan-overflow-8-chunks-not-reached...HEAD lists three files, none under src/: tests/issue 2096 config user facing b.rs, docs/archive/pr-summaries/pr-summary-2263.md, and scripts/runlib.sh (synced from NEAT-AI-core Develop in 381156e). Every env-touching test is [serial] (via env table test! or explicitly) and mutates only through tests/issue` — reviewer: met
- **met** — ./quality.sh passes. — evidence: `met. The reviewer re-ran timeout 1500 ./quality.sh < /dev/null on head b800653, which exited 0 with "All quality checks passed!". tests/issue 2096 config user facing b.rs ran in that gate: 29 passed, 0 failed. The working tree was clean afterwards. RUSTUP TOOLCHAIN was pinned to the only installed t` — reviewer: met
- **unrequested** — Extra test: tests/issue 2096 config user facing b.rs::sample program override is none when unset — evidence: `tests/issue 2096 config user facing b.rs::sample program override is none when unset` — reviewer: unrequested — reason: The issue asks only for empty, whitespace-only and padded cases. This unset baseline is harmless, [serial], and restored through EnvGuard::unset.

## Standards Review

<․!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" --․>

- **clean** — CODING-STANDARDS.md does not exist in this repository, so the diff was checked against AGENTS.md and CONTRIBUTING.md. Checked: Australian English in comments and identifiers; [serial] on every env-mutating test, with restore-on-Drop; SAFETY comments on all three unsafe env blocks; symbols cited inst
