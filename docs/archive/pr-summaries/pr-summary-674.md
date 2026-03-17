## Summary

Move the dependency upgrade workflow from running on every pull request to a
weekly schedule (Monday 06:00 UTC) with manual trigger support. Instead of
committing dependency updates directly to unrelated PR branches, the workflow
now creates a dedicated PR targeting `Develop` using the
`peter-evans/create-pull-request` action. This ensures dependency updates are
isolated, reviewed independently, and run through the full CI quality gate
before merging. Closes #674.

## Evidence

This is a CI/workflow-only change with no Rust code modifications. The full
quality gate (`quality.sh`) passes cleanly — no regressions introduced.

### Key changes to `.github/workflows/upgrade-dependencies.yml`:

- **Trigger**: Changed from `on: pull_request` to `on: schedule` (weekly cron)
  and `on: workflow_dispatch` (manual)
- **Checkout**: Now checks out `Develop` branch directly instead of
  `github.head_ref`
- **PR creation**: Uses `peter-evans/create-pull-request@v7` to create a
  dedicated `chore/upgrade-dependencies` PR targeting `Develop`, which
  automatically triggers the full CI pipeline
- **Permissions**: Added `pull-requests: write` for PR creation

### Acceptance criteria met:

1. Dependency updates no longer modify unrelated PR branches
2. A dedicated PR is created for dependency updates on a regular schedule
3. The dependency update PR runs the full quality gate (CI triggers on PRs to Develop)
4. Existing PRs are not affected by dependency upgrade commits

## Test Plan

- Verified `quality.sh` passes with no regressions
- Workflow YAML syntax validated (bash syntax check in quality.sh)
- The workflow can be tested via manual trigger (`workflow_dispatch`) after merge
