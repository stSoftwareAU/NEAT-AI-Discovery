## Summary

The ShellCheck CI quality workflow (`.github/workflows/shellcheck.yml`) declared
its `pull_request` branch filter as `["*"]`. In GitHub Actions branch filters the
`*` glob does **not** span `/`, so `"*"` matches single-level names like `Develop`
but never `milestone/<slug>`. As a result the ShellCheck lint gate never ran on
milestone sub-issue PRs, letting shell-script regressions merge into the shared
`milestone/<name>` branch unchecked until the single rollup PR into the default
branch.

Added `milestone/*` alongside the existing glob so milestone PRs are gated too.
Milestone branch names are `milestone/<slug>` with no nested slashes, so the
single-level `milestone/*` glob is sufficient. This matches the filter already in
place on the sibling workflows (`semgrep.yml`, `markdown-lint.yml`,
`gitleaks.yml`, `actionlint.yml`, `cargo-quality.yml`).

Closes #1655.

## Evidence

This is a CI configuration change with no web interface to screenshot. Validation
performed:

- `python3 -c "yaml.safe_load(...)"` — workflow parses as valid YAML.
- `actionlint .github/workflows/shellcheck.yml` — passes with no findings.
- Confirmed the new filter matches the pattern already used by all sibling
  quality workflows in `.github/workflows/`.

```mermaid
flowchart LR
    A[Milestone sub-issue PR<br/>base: milestone/clean-up-v2] -->|before: branches ["*"]| B[ShellCheck skipped ❌]
    A -->|after: branches ["*", "milestone/*"]| C[ShellCheck runs ✅]
```

## Test Plan

- YAML validity check (`yaml.safe_load`) — passes.
- `actionlint` on the modified workflow — passes.

No Rust source or bash scripts were changed, so the Rust quality gate
(`./quality.sh`) is not exercised by this change; the relevant gate for a workflow
YAML edit is `actionlint`, which passes.
