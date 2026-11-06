## Summary

`benchmark_compare.sh` comparison mode now checks for `bc` up front and exits 1 with `Error: bc is not installed ...` before any suite runs. Previously a missing `bc` made every `(( $(echo ... | bc -l) ))` condition false, so every comparison counted as "Within threshold" and a real slowdown printed `No regressions detected.` with exit 0. Closes #2572.

Version bump: not changed here; CI's `version-increment` job owns it.

## Spec

### Intent and Rationale

- A missing tool must not become a clean regression verdict. `set -e` does not fire inside an `if`/`elif` condition, so the only reliable fix is to check for the tool before any comparison.
- Chose the issue's first suggestion (`command -v bc`) over rewriting the comparison in `awk`: it is the smallest change and matches the existing check in `benchmark.sh` (Issue #2141).

### Essential Design Decisions

- The check sits in comparison mode only, after the `--save-baseline` and `--list` branches, because those modes do no arithmetic and must keep working on a host without `bc`.
- It runs before the banner and before any `cargo bench`, so no benchmark time is spent on a run that cannot give a verdict.

### Undiscoverable Facts

- Unrelated and left alone: in the regression branch, `grep -B5 "$line"` uses the Criterion `change: [...]` line as a regular expression, so the brackets never match. Under `set -e` and `pipefail` the script then exits 1 before printing `REGRESSIONS DETECTED`. It still fails the gate, but the regression is not listed. The new tests deliberately use an in-threshold measurement so they do not depend on that path.

## Evidence

Backend/CLI change only, so there is no screenshot. `tests/issue_2572_benchmark_compare_requires_bc.rs` runs the real script in a sandbox whose `PATH` holds only a stub `cargo` and symlinks to the few system tools it needs, never the host `bc`.

Observed on this host: `command -v bc` finds nothing, so the missing-tool path was run for real, not only under a stub.

## Acceptance Criteria

The issue states no acceptance-criteria heading. Its suggested fix (up-front `command -v bc` check, exit 1 with a clear error, and a sandbox test with no `bc` on `PATH` asserting a non-zero exit and no clean verdict) is delivered in full, so nothing is left undone and no reviewer sub-agents were dispatched.

## Test Plan

- `tests/issue_2572_benchmark_compare_requires_bc.rs` (new, 4 tests): `compare_without_bc_fails_loudly_and_gives_no_clean_verdict` (non-zero exit, stderr names `bc`, no `No regressions detected`, no `Within threshold`), `compare_without_bc_stops_before_running_any_suite` (no `Comparing:` line), `compare_with_bc_reaches_a_verdict` (positive control: same sandbox plus a stub `bc` reaches `Total comparisons: 1` and `Within threshold:  1`), and `save_baseline_and_list_do_not_need_bc`.
- Red check: with `benchmark_compare.sh` restored to the base version, the first two tests fail and the other two pass; with the fix, all four pass.
- Not mutation-checked: the placement of the check after the save-baseline branch (`save_baseline_and_list_do_not_need_bc`) was not broken on purpose.
- Existing tests still pass: `issue_1918_benchmark_threshold_validation`, `issue_2532_shell_script_coverage`, `issue_2554_benchmark_compare_failure_is_loud`, `issue_2557_benchmark_quiet_host_requirement`. No existing test was edited.
- `cargo fmt --check`, `cargo clippy --test issue_2572_benchmark_compare_requires_bc -- -D warnings`, `shellcheck benchmark_compare.sh`, `bash -n`: clean.
- Docs: `docs/BENCHMARKS.md` gains a "Prerequisite — `bc`" paragraph in Quick Start, and the `benchmark_compare.sh` header comment documents the requirement. I read the surrounding Quick Start text; the `bc` mention in the header's `BENCHMARK_THRESHOLD` entry is about an expression being rejected and is still true.
- Not run: the full `./quality.sh`. It did not finish within the 10-minute foreground limit on the previous run on this repository.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeds the 10-minute foreground limit on this repository; fmt, clippy, shellcheck and the five benchmark-script test files were run and pass; CI runs the same checks" -->

## Not done / risks

- The full `./quality.sh` was not run locally; CI will run it.
- The unrelated `grep -B5 "$line"` bug described above is not fixed here.
