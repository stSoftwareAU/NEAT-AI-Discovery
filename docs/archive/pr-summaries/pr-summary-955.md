## Summary

Fix PR actions not triggering reliably by moving version auto-increment from CI
to the local `quality.sh` pre-commit workflow. Closes #955.

### Root Cause

The CI `version-increment` job pushes a commit using `GITHUB_TOKEN`. By GitHub
design, pushes made with `GITHUB_TOKEN` do **not** re-trigger workflows. This
means:

1. PR opened/updated — CI triggers on commit SHA1
2. `version-increment` pushes a new commit (SHA2) — CI does **not** re-trigger
3. GitHub updates the PR head to SHA2, but check results are reported on SHA1
4. SHA2 (the actual PR head) has **no check runs**
5. User must manually push another commit to trigger CI on the latest SHA

This pattern was observed on every recent merged PR (#944–#956), each requiring
a manual "Update Cargo.toml" or "Bump version" follow-up commit.

### Fix

Add `scripts/auto-version.sh` which auto-increments the `Cargo.toml` patch
version when `src/` has changed compared to the base branch. This script is
called by `quality.sh` (which must be run before every commit per project
guidelines). When the version is already incremented locally, the CI
`version-increment` job detects the difference and **skips** — no commit is
pushed, and CI checks run on the actual PR head SHA.

### Changes

- **`scripts/auto-version.sh`** — New script that compares the local version
  against `origin/Develop`, increments the patch version if `src/` has changed,
  and is idempotent (safe to run multiple times).
- **`quality.sh`** — Calls `auto-version.sh` early in the quality gate pipeline
  (non-fatal if it fails).
- **`AGENTS.md`** — Updated version management documentation to describe the
  local-first increment approach and the CI fallback.
- **`tests/scripts/auto_version_test.sh`** — Integration tests for the
  auto-version script (7 test cases).

## Evidence

The NEAT-AI repository (which does not have this problem) uses a PAT
(`secrets.ACTIONS_PUSH`) for CI pushes, which re-triggers workflows. This fix
achieves the same result without requiring a PAT by preventing the CI push
entirely.

Test results from `tests/scripts/auto_version_test.sh`:
- Test 1: Increment when src/ has changed — PASS
- Test 2: Skip when version already incremented — PASS
- Test 3: Skip when no src/ changes — PASS
- Test 4: Skip when on base branch — PASS
- Test 5: Idempotent (no double-increment, first run) — PASS
- Test 6: Idempotent (no double-increment, second run) — PASS
- Test 7: Handles unstaged src/ changes — PASS

## Test Plan

- `tests/scripts/auto_version_test.sh` — 7 integration tests covering:
  - Version increment when `src/` has changes
  - Skip when version already differs from base branch
  - Skip when no `src/` changes exist
  - Skip when on the base branch (Develop)
  - Idempotency (running twice does not double-increment)
  - Detection of unstaged `src/` changes
