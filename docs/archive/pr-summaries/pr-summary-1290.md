## Summary

Added `set -euo pipefail` (and an explicit `shell: bash`) to every
multi-line `run: |` block in `.github/workflows/ci.yml` that was
previously running unguarded. Without these guards a failed `sed` /
`grep` / piped command silently swallows its exit code, which is
especially dangerous in the `version-increment` step that rewrites
`Cargo.toml` in place. Closes #1290.

The eight steps now guarded:

- `version-increment` → "Check for source changes and increment version"
- `version-increment` → "Check if there are changes"
- `quality` → "Free up runner disk space"
- `quality` → "Clean intermediate artefacts before tests"
- `validation` → "Check for required files"
- `validation` → "Validate Cargo.toml"
- `validation` → "Check documentation"
- `auto-format` → "Ensure bash scripts are executable"

Existing intentional-failure tolerances are preserved — e.g.
`cargo outdated -R || true` continues to suppress that single command's
exit code under `set -e`, and `find ... 2>/dev/null || true` and
`grep -c "///" src/lib.rs || echo "0"` keep their fallbacks.

## Evidence

This is a CI workflow change with no UI or runtime behaviour to
screenshot. Verification is via the new tests in
`tests/issue_1290_workflow_set_euo_pipefail.rs`, which parse `ci.yml`
and assert each named step contains `set -euo pipefail` and declares
`shell: bash`. Both tests went from FAILED → ok after the edits, and
the full `./quality.sh` gate passes (fmt, clippy, check, test, doc,
release build).

```mermaid
flowchart LR
    A[run: \|] -->|previously unguarded| B[silent failure of sed/grep/pipe]
    A2[run: \|<br/>set -euo pipefail<br/>shell: bash] -->|after #1290| C[fail loudly on<br/>unset var, pipe error,<br/>intermediate non-zero]
```

## Test Plan

- Added `tests/issue_1290_workflow_set_euo_pipefail.rs` with two tests:
  - `ci_yml_guarded_steps_contain_set_euo_pipefail` — asserts each of
    the eight steps contains `set -euo pipefail`.
  - `ci_yml_guarded_steps_declare_shell_bash` — asserts each declares
    `shell: bash`.
- Existing workflow tests (`issue_1216_workflow_sha_pins`,
  `issue_1286_workflow_permissions`, `issue_1287_workflow_timeouts`,
  `issue_1288_workflow_concurrency`) continue to pass.
- `./quality.sh` passes cleanly.
