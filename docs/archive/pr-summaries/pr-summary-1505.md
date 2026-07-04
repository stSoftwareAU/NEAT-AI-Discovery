## Summary

Added `set -euo pipefail` as the first line of the `Install gitleaks` step's
multi-line `run:` block in `.github/workflows/gitleaks.yml`. This
security-sensitive step downloads and integrity-verifies a release binary that
later runs with the runner's `GITHUB_TOKEN`, so the download-and-verify chain
should fail fast and unambiguously at the first error. `set -euo pipefail`
ensures a mid-sequence failure aborts at its origin (`-e`), an unset variable
aborts rather than expanding to an empty string (`-u`), and a failing command
on the left of a pipe is not masked (`-o pipefail`). This brings the block into
line with the rest of the repository's substantial `run:` blocks (see
`ci.yml`), which already open with it.

Closes #1505.

## Evidence

Backend/CI-only change — no web interface to screenshot. Verification performed:

- YAML parses cleanly (`python3 -c "import yaml; yaml.safe_load(...)"` → `YAML OK`).
- The `run:` block's bash body passes `bash -n` syntax checking with the new
  `set -euo pipefail` line prepended.

The change, before → after:

```diff
         run: |
+          set -euo pipefail
           GITLEAKS_VERSION="8.24.3"
           EXPECTED_SHA256="9991e0b2903da4c8f6122b5c3186448b927a5da4deef1fe45271c3793f4ee29c"
```

## Test Plan

This is a single-line hardening change to a GitHub Actions workflow; there is
no Rust code path to unit-test. Validation:

- `.github/workflows/gitleaks.yml` remains valid YAML.
- The gitleaks install step's shell body remains syntactically valid bash.
- The `gitleaks.yml` workflow itself continues to run on pull requests and will
  exercise the modified step, which now fails fast on any download/verify error.
