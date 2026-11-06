## Summary

`benchmark_compare.sh` no longer turns a failed `cargo bench` run into success. Every suite under `benches/` already exits 0 when no GPU is available, so any non-zero exit is a genuine failure (compile error, panic, missing `cargo`, missing baseline) and is now reported as one. Closes #2554.

Version bump: `Cargo.toml` (and the lockfiles) updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- The save path discarded every failure and announced "Baseline saved"; the compare path counted every non-zero exit as a GPU skip, dropped the captured output, and exited 0 when all suites failed. A regression gate cannot infer a clean verdict from the absence of recorded regressions.
- The issue suggests a separate "explicit GPU-unavailable result". It is not needed: all 47 suites return normally on a GPU-less host (`grep` of `benches/` for `gpu_is_available` / `GpuAnalyzer::new()` shows each guarded with an early `return`), so "exit 0 and no `change:` line" already is that signal.

### Essential Design Decisions

- Save mode streams `cargo bench` output (it is the diagnostic), finishes the remaining suites, then exits 1 naming the failed ones and without a "saved" line.
- Compare mode prints the captured output to stderr, counts `Failed suites`, and exits 1 even with zero regressions or when every suite failed.
- A suite that exits 0 but records no `change:` line is counted as skipped. A run that compared nothing still exits 0 (documented GPU-less behaviour) but prints "No comparisons were recorded; no regression verdict." instead of "No regressions detected."
- `scripts/benchmark-ci.sh` is unchanged: under `set -euo pipefail` its `"$COMPARE_SCRIPT" ...` calls already propagate a non-zero status.

### Undiscoverable Facts

- None.

## Evidence

Backend/CLI shell change, no UI. Tests drive the real script in a sandbox with stub `cargo` and `bc`.

Red check: the original script restored (`git checkout HEAD -- benchmark_compare.sh`) with the new test kept — the four failure-case tests (`save_baseline_with_a_failing_suite_fails_and_is_not_reported_saved`, `save_baseline_with_every_suite_failing_fails`, `compare_with_a_failing_suite_fails_loudly`, `compare_with_every_suite_failing_fails`) go red; the two success-wording tests also differ on the new summary lines. With the new script all 7 pass.

## Acceptance Criteria

No acceptance-criteria list; the suggested fix is covered as follows: GPU skip preserved (`compare_with_gpu_less_suites_keeps_the_documented_skip`), diagnostic printed and non-zero exit on failure (`compare_with_a_failing_suite_fails_loudly`), no "saved" or clean verdict without evidence, failing-stub coverage for save and compare including all-failed.

## Test Plan

- Added `tests/issue_2554_benchmark_compare_failure_is_loud.rs` (7 tests: save one/all failing and all passing; compare one/all failing, all passing, GPU-less skip). No existing test edited or assertion removed.
- `cargo test --test issue_2554_benchmark_compare_failure_is_loud --test issue_2140_benchmark_failure_is_loud --test issue_1918_benchmark_threshold_validation`: 19 passed.
- `cargo fmt --check` clean; `./quality/shellcheck.sh`: 17 scripts passed.
- Docs: `docs/BENCHMARKS.md` (GPU note, Initial Setup, Detecting Regressions, CI Tier 2, GPU Benchmarks) and `CHANGELOG.md` updated; `README.md` and `CONTRIBUTING.md` have no `benchmark_compare` mentions.
- Full `./quality.sh` was not run (about 15 minutes, which exceeds the foreground cap): <!-- vibe-quality-gate-skipped reason="full gate exceeds the foreground command cap; targeted tests, rustfmt and shellcheck passed; CI runs the full gate" -->

Not done: `scripts/benchmark-ci.sh` has no dedicated test of its status propagation (it relies on `set -e`).
