## Summary

Documents that benchmark timings must be collected on a quiet host: an
otherwise idle machine on an exclusive runner, with no quality gates or parallel
worker jobs running while timings are collected. The requirement is now stated
before the Quick Start commands and in the Tier 2 CI text of
`docs/BENCHMARKS.md`, and in the `--help` text of `benchmark_compare.sh` and
`scripts/benchmark-ci.sh`. Documentation and help text only; no benchmark
behaviour changed. Version bump: `Cargo.toml` (and the lockfiles) updated for
this change; the merged commit records the final version. Closes #2557.

## Spec

### Intent and Rationale

- Consistent hardware alone does not make timings comparable: a baseline taken idle and a comparison taken under load can report a regression that is only contention.

### Essential Design Decisions

- The note applies to timing runs (`--save-baseline`, comparison, `--compare`). `--compile-only` stays suitable for ordinary PR runners, as the issue asks.
- The wording is shared across the three surfaces so a single phrase set ("idle machine", "exclusive runner", "quality gates") is pinned by tests.

### Undiscoverable Facts

- None.

## Evidence

Docs and script-help change only. Tests drive the real scripts' `--help` and read the real `docs/BENCHMARKS.md`.

Review correction: restored the unrelated #2556 archived summary to its Develop
content, preserving its historical fuzz lockfile version (0.74.361).

## Test Plan

- Added `tests/issue_2557_benchmark_quiet_host_requirement.rs` (3 tests): `benchmark_compare_help_states_the_quiet_host_requirement`, `benchmark_ci_help_states_the_quiet_host_requirement`, `benchmarks_doc_states_the_requirement_before_the_quick_start_commands`. All pass.
- Red-checks (each piece removed alone, matching test went red, then restored): `benchmark_compare.sh` help paragraph → compare help test red; `benchmark-ci.sh` `print_usage` lines → ci help test red; Quick Start note → doc test red; Tier 2 clause → doc test red.
- Existing `issue_2554_benchmark_compare_failure_is_loud` (9), `issue_1918_benchmark_threshold_validation` (9) and the `regression` target's `benchmark_compare` tests (5) pass. `shellcheck` and `rustfmt --check` are clean.
- No existing test was edited, so no assertions were removed.
- Twinned copies: the requirement text appears in the three places above; all were changed together and each is pinned by a test.
- Siblings read in `docs/BENCHMARKS.md`: the threshold-section "stable hardware" sentence and the GPU Benchmarks note also gained the idle-host clause; the Disk Space section makes no timing claim.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh not run in this run; targeted checks above passed, CI runs the full gate" -->
