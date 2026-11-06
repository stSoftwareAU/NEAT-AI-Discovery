## Summary

`docs/ci-fuzzing-workflow.yml` ran only the two FFI fuzz targets, so an operator who copied it never fuzzed the Parquet footer-statistics and pre-decode budget paths. The template now also runs `fuzz_parquet_footer_stats` (with the `fuzz/seeds/parquet_footer_stats` seed corpus and `-max_len=65536`, as `scripts/fuzz-ci.sh` does) and `fuzz_parquet_predecode_budget`. A new test compares the `[[bin]]` names in `fuzz/Cargo.toml`, the `FUZZ_TARGETS` array in `scripts/fuzz-ci.sh` and the template's `fuzz run` steps, so the next new target cannot be missed. Closes #2527.

Version bump: `Cargo.toml` and the root `Cargo.lock` updated for this change, and `fuzz/Cargo.lock` restamped with `quality/fuzz_lockfile.sh sync`; the merged commit records the final version.

## Spec

### Intent and Rationale

- The issue offers two fixes: call `./scripts/fuzz-ci.sh`, or add the missing steps. Adding steps was chosen because `tests/issue_2506_fuzz_lockfile_sync.rs` and `tests/issue_1939_documented_commands.rs` pin the template's own `cargo +nightly fuzz run --locked` text and the sync-before-first-run order; calling the script would have meant rewriting those contracts.
- The drift test supplies the "one list" guarantee instead: three lists must name the same targets.

### Essential Design Decisions

- The existing two steps and the lockfile sync step are unchanged byte-for-byte.
- The footer-stats step passes the same corpus directory, seed directory and `-max_len` as `fuzz-ci.sh`; the test derives those strings from the script, so a change there turns the test red.

### Undiscoverable Facts

- `fuzz/Cargo.lock` was already at 0.74.349 while the root was 0.74.350; the sync step corrected both.

## Evidence

Backend/CI-template change, no UI. `tests/issue_2527_fuzz_template_targets.rs` has helper tests for each text extractor (positive and negative, including a `\`-continued command, a commented-out command and a non-`run` `fuzz build`) and three contract tests: `fuzz_ci_script_runs_every_declared_fuzz_target`, `ci_template_runs_every_declared_fuzz_target` and `ci_template_gives_the_footer_target_its_seeds_and_max_len`.

## Test Plan

- Added `tests/issue_2527_fuzz_template_targets.rs` (6 tests, all pass on the final tree).
- Red checks against the base template (`git show HEAD:docs/ci-fuzzing-workflow.yml`): `ci_template_runs_every_declared_fuzz_target` and `ci_template_gives_the_footer_target_its_seeds_and_max_len` fail; the other four pass, as expected.
- Single-break checks on the new template: dropping only ` -max_len=65536` → `ci_template_gives_the_footer_target_its_seeds_and_max_len` red; dropping only the seed directory → same test red; renaming only the pre-decode target → `ci_template_runs_every_declared_fuzz_target` red.
- Existing tests unchanged and still passing: `issue_2506_fuzz_lockfile_sync` (9), `issue_1939_documented_commands` (11), `issue_2478_fuzzing_template_hardening` (8). No assertion removed from any existing test.
- `cargo fmt --all -- --check` clean; `cargo clippy --locked --tests -- -D warnings` clean.
- Docs: `CHANGELOG.md` entry added; grep of `README.md`, `CONTRIBUTING.md`, `docs/*.md` and `fuzz/` found no sentence claiming the template runs only two targets (`README.md` already documents the parquet targets).
- Full gate: `timeout 595 ./quality.sh` hit the cap (exit 124) on a cold build in this fresh worktree; not run to completion here.

Not done / risk: the full quality gate did not finish; the new template steps were not executed in a real GitHub Actions run (the template is not a live workflow).

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the 595 s foreground cap (exit 124) after a cold build in a fresh worktree; targeted tests, fmt and clippy were run separately; CI runs the full gate" -->
