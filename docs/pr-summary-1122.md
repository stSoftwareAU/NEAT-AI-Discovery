## Summary
Completed the Cargo Security Audit workflow by adding the missing
`rustsec/audit-check` detection pattern. The reusable workflow at
`.github/workflows/security.yml` now runs the official RustSec GitHub
Action alongside the existing `cargo install cargo-audit` + `cargo audit`
steps, and is granted the `issues: write` / `checks: write` permissions
the action needs to annotate PRs and open advisory issues. Closes #1122.

A small pre-existing clippy lint (`map(...).unwrap_or(...)` on a `Result`)
surfaced by the newer toolchain in `src/debug.rs` was also fixed so
`quality.sh` passes cleanly — the change is a one-line rewrite to
`map_or(...)`.

## Evidence
This is a CI/workflow change with no user interface. Verification:

- New TDD test `tests/test_security_workflow_validation.sh` asserts the
  workflow contains all three canonical patterns: `cargo audit`,
  `cargo-audit`, and `rustsec/audit-check`. Red→green confirmed:
  - Before the workflow edit: `1 failed` (missing `rustsec/audit-check`).
  - After the workflow edit: `3 passed, 0 failed`.
- Full `./quality.sh` run passes end-to-end (fmt, clippy, check, tests,
  docs, release build).

## Test Plan
- Added `tests/test_security_workflow_validation.sh` covering the three
  required cargo-audit patterns in `.github/workflows/security.yml`.
- Ran `./quality.sh < /dev/null` — all checks pass.
- Ran the new test directly — 3 passed, 0 failed.
