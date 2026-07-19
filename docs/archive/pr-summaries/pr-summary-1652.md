## Summary

The Gitleaks CI quality workflow (`.github/workflows/gitleaks.yml`) used a
`pull_request` branch filter of `["*"]`. In GitHub Actions branch-filter globs,
`*` matches zero or more characters but **does not cross `/`**, so `["*"]`
matches single-level branches (`Develop`, `main`) but never a
`milestone/<slug>` branch. Milestone sub-issue PRs target a shared
`milestone/<name>` branch, so the secret scan never ran on them — every
intermediate sub-issue PR merged into the milestone branch unscanned, with the
gap only caught later by the single rollup PR into the default branch.

Added `milestone/*` to the filter (`branches: ["*", "milestone/*"]`) so the
Gitleaks gate runs on milestone PRs too, matching the established pattern already
applied to `actionlint.yml` (Issue #1650) and `cargo-quality.yml` (Issue #1656).
Milestone branch names carry no nested slashes, so a single-level `milestone/*`
glob is sufficient.

Closes #1652.

## Evidence

Backend/CI-config change only — no web interface to screenshot.

Validation performed:
- `python3 -c "import yaml; yaml.safe_load(...)"` — YAML parses cleanly.
- `actionlint .github/workflows/gitleaks.yml` — passes with no findings.

The Rust `quality.sh` suite (fmt/clippy/check/test/release build) exercises
`src/`, not `.github/workflows/`, so it is not the relevant gate for this
workflow-YAML-only change; `actionlint` is.

```mermaid
flowchart LR
    subgraph Before["Before — branches: [\"*\"]"]
        A1[PR → Develop/main] -->|matched| G1[Gitleaks runs]
        A2[PR → milestone/slug] -.->|not matched| S1[Scan skipped]
    end
    subgraph After["After — branches: [\"*\", \"milestone/*\"]"]
        B1[PR → Develop/main] -->|matched| G2[Gitleaks runs]
        B2[PR → milestone/slug] -->|matched| G3[Gitleaks runs]
    end
```

## Test Plan

Workflow trigger filters are declarative GitHub Actions configuration; they
cannot be exercised by the repository's Rust unit/integration tests. Validation
was done via `actionlint` (static workflow linter) and a YAML parse check, both
of which pass. The fix mirrors the identical, already-merged change in
`actionlint.yml` (#1650) and `cargo-quality.yml` (#1656).
