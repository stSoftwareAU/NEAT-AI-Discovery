## Summary

The CI `version-increment` step no longer skips the bump just because an earlier
`chore: auto-increment versions for changed projects` commit exists on the PR
branch. A PR that merged Develop and ended up level with it is now bumped again.
Closes #2476.

Version bump: `Cargo.toml` is not edited by hand here; CI's `version-increment` job bumps it, and the merged commit records the final version.

## Spec

### Intent and Rationale

- The commit-exists check (old SECOND) ran before the version comparison, so a PR level with Develop after a merge exited early and shipped code with no version change.
- The version comparison (now SECOND) already decides correctly: the FIRST guard rejects downgrades, so `version != base` means strictly ahead (skip) and `version == base` falls through to a bump. The commit-exists check was therefore redundant, and the fix deletes it rather than reordering it.

### Essential Design Decisions

- Deleted rather than folded in: with the downgrade guard first, the commit-exists test can only wrongly skip, never usefully skip. Duplicate bumps remain impossible because a bumped PR is strictly ahead of base.
- `.github/workflows/ci.yml` was edited because the issue asks for exactly that change; AGENTS.md requires approval for CI edits, and the issue supplies it. Only the one block and its comment labels changed.

### Undiscoverable Facts

- Issue comments report the same failure on PRs #2426, #2481, #2485 and #2482 (versions 0.74.295, .309, .313, .316/.317), all level with Develop after a merge.

## Evidence

- Docs sweep: `README.md` (Distributed Build & Versioning, first bullet) and `CONTRIBUTING.md` (Version Management) said CI bumps "unless the PR branch already carries a bump"; both now say "unless the PR's version is already ahead of `origin/Develop`". Greps for `auto-increment`, `increment commit`, `already carries` found no other hit outside `docs/archive/`.
- Regression tests run the real `run:` script extracted from `ci.yml` in a throwaway git repo with a stubbed `cargo` and an `origin/Develop` ref.

## Acceptance Criteria

The issue states no acceptance-criteria heading and nothing it asks is left undone.

## Test Plan

- Added `tests/issue_2476_ci_version_increment_level_with_base.rs` (3 tests, passed on the head):
  - `level_with_base_is_bumped_despite_earlier_auto_increment_commit`: with the old SECOND block restored in `ci.yml`, red (the step exits early, `Cargo.toml` stays 0.1.0); green with the fix.
  - `ahead_of_base_with_earlier_auto_increment_commit_is_not_bumped_again`: red on the old block, because it never prints "already ahead of base branch"; green with the fix. It guards that a strictly-ahead PR is still skipped.
  - `level_with_base_without_auto_increment_commit_is_bumped`: control for the existing behaviour, green on both.
- Existing workflow tests passed unchanged: `issue_1878_ci_version_increment_no_reresolve`, `issue_2015_version_no_downgrade`, `issue_1286_workflow_permissions`, `issue_1287_workflow_timeouts`, `issue_1288_workflow_concurrency`, `issue_1868_actions_push_credential_persistence`, `issue_2036_setup_rust_composite`, `issue_2072_canonical_runlib`, `issue_1564_ci_no_push_trigger`, `issue_1651_ci_milestone_filter`, `issue_1992_ci_doc_accuracy`, `issue_1681_doc_staleness`. No assertion was removed from any existing test.
- Twinned rule: the "bump unless already ahead" rule is stated in `ci.yml`, `README.md` and `CONTRIBUTING.md`; all three now agree.

<!-- vibe-quality-gate-skipped reason="the full ./quality.sh takes about 15 minutes, beyond the 10-minute foreground cap; the targeted workflow tests above were run and CI runs the full gate on the PR" -->

## Not checked

- Full `./quality.sh` was not run (see the note above).
- The step was exercised against a stubbed `cargo` and a local `origin/Develop` ref, not on a real GitHub runner.
