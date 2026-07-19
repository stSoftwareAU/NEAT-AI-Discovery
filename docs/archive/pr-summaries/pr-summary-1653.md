## Summary

The Markdown Lint CI gate (`.github/workflows/markdown-lint.yml`) triggered on
`pull_request` with a `branches: ["*"]` filter. In GitHub Actions the
single-star glob `*` matches only slash-free branch names, so milestone feature
branches (`milestone/<slug>`) never matched. Milestone sub-issue PRs target a
shared `milestone/<name>` branch, so the lint gate silently skipped them —
every intermediate sub-issue PR merged into the milestone branch unchecked,
with the gap only caught later by the single rollup PR into the default branch.

Added `milestone/*` to the filter (`branches: ["*", "milestone/*"]`) so the gate
now runs on milestone PRs too. This matches the existing convention already
applied to `ci.yml` (Issue #1651) and `gitleaks.yml`.

Closes #1653.

## Evidence

Purely a CI workflow-configuration change — no web interface to screenshot.
Validation performed:

- `actionlint .github/workflows/markdown-lint.yml` → **PASS**.
- YAML parses cleanly (`yaml.safe_load`).
- Change mirrors the established milestone-branch pattern already present in
  `.github/workflows/ci.yml` and `.github/workflows/gitleaks.yml`.

```mermaid
flowchart LR
    A[Sub-issue PR into milestone/clean-up-v2] -->|filter "*" only| B[Gate SKIPPED — merges unchecked]
    A -->|filter "*", "milestone/*"| C[Markdown Lint runs — PR gated]
```

## Test Plan

Workflow branch-filter behaviour is enforced by GitHub Actions itself and is not
unit-testable in this Rust crate. Verification:

- `actionlint` static analysis of the workflow passes.
- Confirmed the glob semantics: `*` excludes slash-containing refs, so
  `milestone/*` is required for `milestone/<slug>` branches.
- Cross-checked consistency with `ci.yml` (Issue #1651) and `gitleaks.yml`.
