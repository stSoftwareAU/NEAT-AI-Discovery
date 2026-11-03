## Summary

The fuzzing CI template `docs/ci-fuzzing-workflow.yml` now carries the hardening the live workflows have: a top-level `permissions: contents: read`, `persist-credentials: false` on its `actions/checkout` step, `timeout-minutes: 30` on the job and a `concurrency` group. A new test, `tests/issue_2478_fuzzing_template_hardening.rs`, keeps the template from drifting again. Closes #2478.

## Spec

### Intent and Rationale

- Someone who copies the template into `.github/workflows/fuzzing.yml` otherwise gets a job that stores the `GITHUB_TOKEN` on disk and runs `cargo install` and fuzz builds with the default token scope.

### Essential Design Decisions

- The test reads the template as plain text (no YAML parser is in the dependency tree, matching `tests/issue_1286_workflow_permissions.rs`). It ignores comment lines, and each helper is checked against inline positive and negative YAML.
- The issue's optional `timeout-minutes` and `concurrency` items are included; the 30-minute cap is a judgement (two 30-second fuzz runs plus a `cargo-fuzz` install).
- No `Cargo.toml` version bump: no library code changed (a docs template and a test file only); CI's `version-increment` job covers the PR.

### Undiscoverable Facts

- None.

## Evidence

Backend/CI-template change, no UI. Tests: `tests/issue_2478_fuzzing_template_hardening.rs` (8 tests, all passed); `tests/issue_1939_documented_commands.rs` (references the template, 11 passed); `cargo fmt --check` and `cargo clippy --test issue_2478_fuzzing_template_hardening -- -D warnings` clean.

Review correction: restored the historical version in `pr-summary-2363.md` to the Develop record, removing an unrelated merge edit.

## Acceptance Criteria

The issue has no acceptance-criteria heading; every requested item is done (permissions, persist-credentials, optional timeout and concurrency, optional drift test).

## Test Plan

- Added `tests/issue_2478_fuzzing_template_hardening.rs`: real-file tests for permissions, checkout `persist-credentials`, job `timeout-minutes` and `concurrency`, plus helper tests with negatives (commented-out directive, setting in a different step, job-level block, empty timeout).
- Red checks on the template: removing `persist-credentials: false` fails `every_checkout_step_sets_persist_credentials_false`; removing `permissions:` fails `template_declares_read_only_top_level_permissions`; removing `timeout-minutes` fails `fuzzing_job_declares_timeout_minutes`. Each was restored.
- No existing assertions were removed.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the foreground time cap in this environment on earlier runs; targeted tests, fmt and clippy were run; CI runs the full gate" -->
