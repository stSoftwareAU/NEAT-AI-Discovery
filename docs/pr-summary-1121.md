## Summary
Added `.github/workflows/shellcheck.yml` to lint all bash scripts with ShellCheck on every pull request, filling the gap left by the existing `shell-checks` CI job that only performs `bash -n` syntax validation. Closes #1121.

## Evidence
Ran `shellcheck --severity=warning` locally against every `*.sh` file in the repository (`quality.sh`, `benchmark.sh`, `benchmark_compare.sh`, and the scripts under `scripts/` and `tests/`). All files pass cleanly, so the new workflow will be green on the current codebase.

This is a CI-only change; there is no UI to screenshot and no performance impact.

## Test Plan
- [x] Verified all existing shell scripts pass `shellcheck --severity=warning` with no findings
- [x] Validated `shellcheck.yml` parses as valid YAML
- [ ] Confirm the `ShellCheck` job runs and passes on this PR
