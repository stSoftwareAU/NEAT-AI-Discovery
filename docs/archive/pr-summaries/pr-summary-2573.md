## Summary

The CI `version-increment` job now restamps `fuzz/Cargo.lock` after it bumps `Cargo.toml`, so Develop's fuzz lock no longer lags one version behind and `cargo fuzz run --locked` / `quality/fuzz_lockfile.sh check` keep passing after merges. Closes #2573.

Version bump: not made by hand; the CI `version-increment` job bumps `Cargo.toml` on the PR, and this change is CI/test/docs only (no library code). `fuzz/Cargo.lock` was restamped to the branch's current `Cargo.toml` version with `./quality/fuzz_lockfile.sh sync`.

## Spec

### Intent and Rationale

- The job bumped the patch version and guarded only the root `Cargo.lock`, so the path-package stamp in the separate `fuzz/` workspace stayed stale (Develop: `Cargo.toml` 0.74.356, fuzz lock 0.74.355).
- The existing `quality/fuzz_lockfile.sh sync` already rewrites exactly that one line with no re-resolve and no network, so the job reuses it rather than adding a second implementation.

### Essential Design Decisions

- The `fuzz/Cargo.lock` guard allows exactly two diff lines: `-version = "<any>"` and `+version = "<new crate version>"`. The old stamp may be any value, because it can lag by more than one release; anything else fails the job loudly, matching the Cargo.lock guard (Issue #1878).
- `.github/workflows/ci.yml` was edited because the issue is an approved work item asking for exactly this change; AGENTS.md's approval requirement is met by the issue itself.
- The "already ahead of base" early exit is left alone: the sync runs only when the job performs the bump, as the issue asks.

### Undiscoverable Facts

- None.

## Evidence

Backend/CI only, so no screenshot. `tests/issue_1878_ci_version_increment_no_reresolve.rs` runs the real extracted `run:` script in a throwaway git repo with a stubbed `cargo`, a copy of the real `quality/fuzz_lockfile.sh`, and fuzz fixtures.

## Test Plan

- Updated the Issue #2476 version-increment fixture to include the real fuzz sync helper and a fuzz workspace, and assert that a bump restamps only the path-package version. This repairs the coverage failure caused by the incomplete fixture.
- Added `version_increment_step_restamps_the_fuzz_lockfile_version_only`, `version_increment_step_restamps_a_fuzz_lockfile_stale_by_more_than_one_release` and `version_increment_step_fails_loud_when_the_fuzz_lockfile_moves_beyond_its_stamp` in `tests/issue_1878_ci_version_increment_no_reresolve.rs`. The two existing tests keep all their assertions; only their call form changed (a `Setup` struct replaces the mutation argument).
- Red checks: deleting the new `ci.yml` block turned the three new tests red while the two existing ones stayed green. Changing the fuzz guard's `exit 1` to `true` turned only the fail-loud test red.
- Passed: `cargo test --test issue_1878_ci_version_increment_no_reresolve --test issue_2506_fuzz_lockfile_sync --test issue_1681_doc_staleness --test issue_1683_agents_consolidation --test issue_1685_doc_link_integrity`.
- `./quality.sh` ran cargo deny, build, format, lint and type checks cleanly, then hit this run's 590 s foreground cap during its test stage, so the full test suite was not completed here. <!-- vibe-quality-gate-skipped reason="foreground command cap of 600s is shorter than the ~15m gate; fmt, clippy, type checks and the touched tests passed; CI runs the full suite" -->
- Docs updated: `README.md` (fuzz lock paragraph), `CONTRIBUTING.md` (`version-increment` bullet), `CHANGELOG.md` (new Unreleased entry). The Issue #2506 changelog entry that says the job is "deliberately unchanged" is history and stays as written.
- Merge resolution: retained the fuzz lock restamp description alongside Develop's quiet `cargo build -q --lib` documentation in `CONTRIBUTING.md`.
- Twinned copies: the workflow template `docs/ci-fuzzing-workflow.yml` and `scripts/fuzz-ci.sh` already run `sync` before fuzzing and need no change.

Not done / risk: the full test suite did not complete locally; the workflow change itself can only be exercised on a real PR run.
