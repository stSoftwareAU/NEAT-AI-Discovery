## Summary

Adds `tests/issue_2096_config_user_facing_a.rs`: `#[serial]` hostile-value probe tests for every un-cached numeric accessor in the first half of `src/config/user_facing.rs` (top of file through `focus_ranking_memory_margin_mb`). Each accessor is fed `u64::MAX` (built from the constant), `-1`, `NaN`, `inf`, `abc` and, where it matters, `0`, and the value it returns today is pinned. No `src/` change. Closes #2258.

Every observed value matched the expectations listed in the issue, so there is **no divergence to record for #2260**. The huge-value `f32` accessors (`focus_reconstruction_mismatch_weight`, `focus_impact_gate_threshold`, both constant-source thresholds) accept `u64::MAX` as exactly 2^64, a finite `f32`, and that is now pinned.

Deliberately left out (documented in the module doc): the `OnceLock`-cached accessors, the boolean accessors, whitespace trimming (already in `issue_2006`), and the huge-value watchdog cases (owned by #2259).

## Evidence

Backend/config tests only, so there is no UI to screenshot.

- `cargo test --test issue_2096_config_user_facing_a`: 16 passed.
- `cargo clippy --test issue_2096_config_user_facing_a -- -D warnings`: clean.
- `./quality.sh < /dev/null`: "All quality checks passed!"

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — Every un-cached numeric accessor listed has a `#[serial]` test covering the hostile set with an exact value or explicit range — evidence: `tests/issue_2096_config_user_facing_a.rs` (16 tests, one per accessor) — reviewer: met
- **met** — Every test goes through `with_env`, so no variable leaks — evidence: `tests/issue_2096_config_user_facing_a.rs::with_env`, used by `assert_table` and `assert_constant_source_huge` — reviewer: met
- **met** — `cargo test --test issue_2096_config_user_facing_a` passes, and so does `./quality.sh` — evidence: 16 passed; full `./quality.sh` run after the final edit passed — reviewer: partial — reason: the reviewer confirmed the targeted test but could not run the gate; it was run here and passed
- **unrequested** — `inf` also probed on the integer accessors, and sanity asserts that the stall-window constants are 600 s / 30 s — reviewer: unrequested — reason: `inf` is in the issue's hostile set; the constant asserts pin the 600 s / 30 s the issue states, so a changed constant cannot hide behind the symbolic comparison
- **unrequested** — 64-bit-target note on the `usize::MAX` expectations in the module doc — reviewer: unrequested — reason: records the assumption behind `u64::MAX` parsing as `usize::MAX`

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — PR summary file missing (CONTRIBUTING.md "PR Summary File") — evidence: `docs/archive/pr-summaries/pr-summary-2258.md` — reason: fixed here, this file adds it
- **clean** — `#[serial]` plus a restoring `with_env` for all env mutation, assertions on returned values (no timing checks, no vacuous asserts), scope limited to un-cached numeric accessors with the exclusions justified, Australian English, no dependency/CI/FFI/manifest changes. The reviewer used CONTRIBUTING.md and AGENTS.md, since this repository has no CODING-STANDARDS.md. Optional note: `NEAT_AI_DISCOVERY_PREFETCH_DEPTH` has no production reader (`get_streaming_config_from_env` has no caller). That is a dead-lever candidate predating this diff and outside this issue's scope.

## Test Plan

- Added `tests/issue_2096_config_user_facing_a.rs` with one table-driven `#[serial]` test for each of: `gpu_stall_window`, `watchdog_stall_timeout`, `watchdog_abort_delay`, `max_cached_blocks`, `prefetch_depth`, `block_size`, `focus_reconstruction_mismatch_weight`, `focus_impact_gate_threshold`, `max_activation_configs_per_target`, `source_input_index_bias`, `max_sources_per_target`, `constant_source_effect_threshold`, `constant_source_threshold_with_dynamic(None)`, `focus_ranking_memory_budget_mb`, `max_parquet_decode_mb`, `focus_ranking_memory_margin_mb`.
