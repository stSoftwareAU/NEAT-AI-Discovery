## Summary
Added the Semgrep SAST scanning GitHub Actions workflow at `.github/workflows/semgrep.yml`. The workflow runs on every pull request using the official `semgrep/semgrep` container and the `p/default` ruleset, improving the repository's security posture by catching common vulnerability patterns before merge. Closes #1120.

## Evidence
This is a CI configuration change with no application code or UI impact. Validation performed:

- **YAML syntax validated** with `python3 -c "import yaml; yaml.safe_load(...)"` — parses cleanly.
- **Quality gate** — `./quality.sh` passes (171 unit + 5 integration tests, fmt, clippy, docs, release build).
- **Style consistency** — trigger and `permissions: contents: read` match the sibling security workflows (`gitleaks.yml`, `shellcheck.yml`).
- **Template fidelity** — matches the YAML template provided in the issue verbatim; no deviations.

The workflow itself will execute once merged and a PR is opened against the repository. It requires an optional `SEMGREP_APP_TOKEN` secret for Semgrep Cloud integration; without it, Semgrep CI still runs against the public `p/default` ruleset.

## Test Plan
- [x] `.github/workflows/semgrep.yml` parses as valid YAML.
- [x] `./quality.sh` passes cleanly.
- [x] Workflow file matches the template specified in the issue.
- [ ] After merge: confirm the Semgrep check appears on subsequent PRs.
