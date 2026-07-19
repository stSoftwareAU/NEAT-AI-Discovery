## Summary

The Coverage workflow (`.github/workflows/cargo-quality.yml`) declared a
`pull_request: branches: ["*"]` filter. In GitHub Actions branch filters the
`*` glob does **not** cross `/`, so `"*"` matches single-level branch names
(e.g. `Develop`) but never `milestone/<slug>`. Milestone sub-issue PRs target a
shared `milestone/<name>` branch, so the coverage gate never ran on those PRs —
coverage regressions accumulated on the milestone branch unobserved and only
surfaced on the single rollup PR into the default branch.

The fix adds `milestone/*` alongside the existing glob so milestone PRs are
gated too. Milestone branch names carry no nested slashes, so the single-level
`milestone/*` glob is sufficient. The `"*"` glob is retained so non-milestone
PRs (Develop and friends) stay gated.

Closes #1656.

## Evidence

Backend/CI-config change — no web interface to screenshot.

Branch-filter behaviour before and after:

```mermaid
flowchart LR
    subgraph before["Before: branches: [\"*\"]"]
        A1[PR to Develop] -->|matches| C1[Coverage runs]
        A2["PR to milestone/foo"] -->|no match| S1[Coverage skipped]
    end
    subgraph after["After: branches: [\"*\", \"milestone/*\"]"]
        B1[PR to Develop] -->|matches| C2[Coverage runs]
        B2["PR to milestone/foo"] -->|matches milestone/*| C3[Coverage runs]
    end
```

Verified:
- `actionlint .github/workflows/cargo-quality.yml` — clean (exit 0).
- New tests fail against the unfixed workflow (`git stash` of the YAML change
  reproduces the failure) and pass with the fix.
- `./quality.sh` passes cleanly (fmt, clippy, check, test, docs, release build).

## Test Plan

Added `tests/issue_1656_cargo_quality_milestone_filter.rs`, mirroring the
sibling text-parsing workflow tests (no YAML parser is in the dependency tree):

- `cargo_quality_pull_request_filter_matches_milestone_branches` — asserts the
  `pull_request` `branches:` filter includes `milestone/*` (reproduces #1656;
  fails without the fix).
- `cargo_quality_pull_request_filter_keeps_wildcard` — asserts the additive
  change retains the single-level `"*"` glob so non-milestone PRs stay gated.
