# Pin `ludeeus/action-shellcheck` to a commit SHA

## Summary

`.github/workflows/shellcheck.yml` referenced `ludeeus/action-shellcheck@master`,
a mutable branch. Any commit landed upstream on `master` would execute in this
repository's CI on the next PR with access to `GITHUB_TOKEN`. The coding
guidelines mandate commit-SHA pinning for third-party GitHub Actions.

Pinned the action to `00cae500b08a931fb5698e11e79bfbd38e612a38` (tag `2.0.0`,
published 2023-01-29) with a trailing version comment for auditability.

Closes #1215.

## Evidence

This is a CI/workflow change with no UI or runtime behaviour to screenshot.

Before — `.github/workflows/shellcheck.yml` line 20:

```yaml
        uses: ludeeus/action-shellcheck@master
```

After:

```yaml
        # Pinned to a 40-char commit SHA (Issue #1215) to mitigate
        # supply-chain risk from mutable upstream branches. Update via
        # Renovate/Dependabot or by manually resolving the latest release.
        uses: ludeeus/action-shellcheck@00cae500b08a931fb5698e11e79bfbd38e612a38 # 2.0.0
```

The SHA was resolved from the upstream `2.0.0` release tag via
`gh api repos/ludeeus/action-shellcheck/git/refs/tags/2.0.0`.

## Test Plan

- Added `tests/test_shellcheck_workflow_pinning.sh` which asserts the
  workflow file:
  1. does **not** pin the action to `@master` or `@main`,
  2. **does** pin to a 40-character commit SHA,
  3. carries a trailing comment recording the human-readable version.
- The test fails against the pre-fix workflow (verified: 0 passed, 3
  failed) and passes after the fix (3 passed, 0 failed).
- Bash syntax check (`bash -n`) and `shellcheck` clean across the
  repository, including the new test script.

Full Rust quality gate (`cargo build`, `clippy`, `test`, `doc`) was
not re-run because no Rust source was touched; project CI exercises
those gates on PR.
