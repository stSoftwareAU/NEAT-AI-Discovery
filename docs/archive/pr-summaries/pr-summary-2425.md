## Summary

The `semgrep` check was red on the default branch because the fuzzing workflow template `docs/ci-fuzzing-workflow.yml` used mutable action tags (`actions/checkout@v4`, `actions/cache@v4`), tripping `yaml.github-actions.security.github-actions-mutable-action-tag`. Both are now pinned to commit SHAs with the version in a trailing comment. Closes #2425.

- Version bump: not needed; only a docs template changed, no compiled code.

## Spec

### Intent and Rationale

- The scan runs over the whole tree, so a template under `docs/` fails it like a live workflow would. Pinning (rather than excluding the file) keeps the template safe to copy into `.github/workflows/`.

### Essential Design Decisions

- The SHAs are the ones already used in `.github/workflows/ci.yml` (`actions/checkout` v6.0.2, `actions/cache` v5.0.5), so the template matches the repo's resolved pins and is not typed from memory.

### Undiscoverable Facts

- `quality.sh` does not run semgrep; the red check comes from the worker's SAST stage, which mirrors `.github/workflows/semgrep.yml` (`semgrep ci --config p/default` with the two renovate rule exclusions).

## Evidence

Backend/config only; no UI files touched.

- Before: `semgrep scan --config p/default` (same exclusions as `.github/workflows/semgrep.yml`) reported 2 findings, both `github-actions-mutable-action-tag` in `docs/ci-fuzzing-workflow.yml` (lines 33 and 53).
- After: the same command over the whole tree (including `docs/archive`) reports no findings.

## Test Plan

- `semgrep scan --config p/default --metrics=off` with the workflow's exclusions: 2 findings before, 0 after.
- `cargo test --test issue_1939_documented_commands` (this test `include_str!`s the template): 11 passed.
- No test added or edited: the change is a pin in a docs template, and the semgrep scan is the check that covers it. `./quality.sh` was not run: it does not include semgrep and no Rust code changed; CI runs it on the PR.
