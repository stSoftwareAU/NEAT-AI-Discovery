# Add `timeout-minutes:` to every job across `.github/workflows/`

## Summary

Every job across the seven workflows in `.github/workflows/` was missing a
`timeout-minutes:` field, so a wedged step could hold a runner for the GitHub
default of 6 hours (360 minutes). This PR adds an explicit per-job
`timeout-minutes:` to each job — light lint/scan jobs at 10 minutes,
medium toolchain jobs at 15–20 minutes, and the heavy cargo build/test
jobs at 45 minutes — so hangs surface as fast failures rather than
tying up runner capacity for hours.

Closes #1287.

### Timeouts applied

| Workflow | Job | timeout-minutes |
| --- | --- | --- |
| `ci.yml` | `version-increment` | 15 |
| `ci.yml` | `quality` | 45 |
| `ci.yml` | `validation` | 10 |
| `ci.yml` | `auto-format` | 15 |
| `ci.yml` | `spell-check` | 10 |
| `cargo-quality.yml` | `quality` | 45 |
| `gitleaks.yml` | `gitleaks` | 10 |
| `markdown-lint.yml` | `markdownlint` | 10 |
| `security.yml` | `security` | 20 |
| `semgrep.yml` | `semgrep` | 10 |
| `shellcheck.yml` | `shellcheck` | 10 |

The `security` job in `ci.yml` is a reusable-workflow caller
(`uses: ./.github/workflows/security.yml`) — it inherits its cap from
the `timeout-minutes:` set inside `security.yml`, so no separate
entry is needed at the call site.

## Evidence

Backend/CI-only change with no UI to screenshot. Verified by:

1. Added a new test suite `tests/issue_1287_workflow_timeouts.rs` that
   reads each workflow file and asserts every job block contains a
   `timeout-minutes:` line at the expected per-job indent.
2. Ran the test suite before the YAML edits — all 7 tests failed as
   expected (TDD red).
3. Applied the YAML edits — all 7 tests pass (TDD green).
4. Confirmed adjacent workflow tests (`issue_1216_workflow_sha_pins`,
   `issue_1258_container_image_digest_pins`,
   `issue_1286_workflow_permissions`) still pass — the timeout additions
   do not disturb SHA pinning, container-digest pinning, or the
   top-level `permissions:` block.

## Test Plan

- [x] `cargo test --test issue_1287_workflow_timeouts` — 7 tests pass.
- [x] `cargo test --test issue_1286_workflow_permissions` — 2 tests
      pass (no regression in `permissions:` test).
- [x] `cargo test --test issue_1216_workflow_sha_pins` — passes (SHA
      pin test unaffected).
- [x] `cargo test --test issue_1258_container_image_digest_pins` —
      6 tests pass (container-digest test unaffected).
- [x] `cargo fmt --all -- --check` — clean.
- [x] `cargo clippy --tests --all-features -- -D warnings` — clean.
