## Summary

Added the Gitleaks Secrets Detection workflow at `.github/workflows/gitleaks.yml` so every pull request is scanned for accidentally committed secrets. The workflow uses the official `gitleaks/gitleaks-action@v2` action and requests only read access to repository contents. Closes #1119.

## Evidence

This is a CI configuration change with no web interface to screenshot.

- **YAML validated**: parsed successfully with `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/gitleaks.yml'))"` — top-level keys `name`, `on.pull_request`, `permissions.contents: read`, and the `gitleaks` job with `actions/checkout@v4` (`fetch-depth: 0`) and `gitleaks/gitleaks-action@v2` all parse as expected.
- **Runtime verification** happens in CI: once this workflow is on the default branch, subsequent pull requests will trigger the `Gitleaks` check automatically.

## Test Plan

- [x] YAML syntax validated locally.
- [ ] Merge PR and confirm the `Gitleaks` check runs on the next pull request against `Develop`.

## Notes

- Scope is strictly limited to adding the new workflow file; no Rust sources, tests, or existing workflows were touched.
- Permissions are pinned to `contents: read`, following the principle of least privilege recommended in AGENTS.md.
