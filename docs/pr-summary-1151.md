## Summary

Replaced the manual `gitleaks` install/scan steps in `.github/workflows/gitleaks.yml` with the official `gitleaks/gitleaks-action@v2` action so the workflow matches the standard template and satisfies the workflow-sync detection. Closes #1151.

## Evidence

This is a CI workflow change with no UI or runtime code surface. Verification:

- The workflow file parses as valid YAML (`python3 -c "import yaml; yaml.safe_load(...)"`).
- The job uses `gitleaks/gitleaks-action@v2` with `GITHUB_TOKEN` per the suggested template.
- The repository is public, so `gitleaks-action@v2` runs without a `GITLEAKS_LICENSE`.

```mermaid
flowchart LR
    PR[Pull Request] --> CO[actions/checkout@v4<br/>fetch-depth: 0]
    CO --> GL[gitleaks/gitleaks-action@v2]
    GL -->|GITHUB_TOKEN| Result[Secrets scan result]
```

## Test Plan

- [x] `.github/workflows/gitleaks.yml` validated as YAML.
- [x] Workflow contains both required patterns: `gitleaks` and `gitleaks/gitleaks-action`.
- [ ] On the next PR, the `Gitleaks` job runs `gitleaks/gitleaks-action@v2` and reports a clean scan.
