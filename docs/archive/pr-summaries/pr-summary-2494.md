## Summary

`tests/test_gitleaks_workflow_verification.sh` was run by no gate, so nothing enforced the SHA-256 check on the downloaded gitleaks binary in `.github/workflows/gitleaks.yml`. Its assertions now live in a Rust integration test that runs under `cargo test`, and the script is deleted. The other four unrun `tests/test_*.sh` scripts were each checked and removed or ported. Closes #2494.

Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- Same gap Issue #2390 closed for the coverage workflow: a script no gate runs enforces nothing. A Rust test under `tests/` runs in `cargo test` and in CI.
- Checks are pure functions over the workflow text with a real-file test each and negative fixtures, following `tests/issue_2390_coverage_workflow_guards.rs`.

### Essential Design Decisions

- The gitleaks test is stricter than the script: it ignores comments, joins backslash-continued lines, requires `sha256sum -c` to use `${EXPECTED_SHA256}` and the path `curl` wrote, requires `tar -xzf` to extract that same verified path, and requires `set -euo pipefail` before the check (otherwise a failed check in a pipe would not abort).
- The published-checksum pin is a table (`PUBLISHED_CHECKSUMS`) keyed by version. A version bump without a table entry fails with a message asking for the published checksum, so a bump cannot skip verification.
- `test_cargo_toml_validation.sh` tested a copy of the CI grep. The Rust port extracts the real `Validate Cargo.toml` pipeline from `ci.yml` and runs it with `bash` against fixtures, failing closed if the step or pipeline is missing.
- Deleted without a port, as already covered or not a repo property: `test_security_workflow_validation.sh` (covered by `tests/issue_1657_workflow_dedup_audit.rs`), `test_shellcheck_workflow_pinning.sh` (covered by `tests/issue_1898_shellcheck_gate.rs::shellcheck_workflow_actions_stay_sha_pinned` and `no_workflow_depends_on_the_orphaned_shellcheck_wrapper`), `test_quality_upgrade.sh` (asserted host `cargo-upgrade` availability and network; `bump-deps.sh` skips gracefully without cargo-edit; `Cargo.lock` presence is covered by `tests/issue_1266_cargo_lock_committed.rs`).

### Undiscoverable Facts

- Dropping the comment-filter `grep -vE` from the CI Cargo.toml pipeline changes no behavioural verdict (the anchored second grep already rejects comment lines), so only the pipeline-shape test `real_ci_step_and_pipeline_are_extractable` goes red for it.

## Evidence

Backend/test-only change; no UI. Tests:

- `tests/issue_2494_gitleaks_workflow_guards.rs` (47 tests, passed)
- `tests/issue_2494_cargo_toml_validation.rs` (12 tests, passed)

## Test Plan

- Ran `cargo test --test issue_2494_gitleaks_workflow_guards --test issue_2494_cargo_toml_validation`: 47 and 12 passed. `cargo fmt --check` clean; `cargo clippy` on each new test target with `-D warnings` clean.
- Red checks against the real workflow: deleting the `sha256sum -c` line → `real_workflow_verifies_download_with_sha256sum`, `real_workflow_verifies_before_extraction`, `real_workflow_sets_pipefail_before_verification` red; moving it after `tar -xzf` → `real_workflow_verifies_before_extraction` red; changing the last hash digit → `real_workflow_matches_published_checksum` red. Dropping the `^` anchor in the `ci.yml` field grep → `build_edition_does_not_match_edition` and `rust_version_does_not_match_version` red; removing the comment filter → `real_ci_step_and_pipeline_are_extractable` red. Workflows restored afterwards (`git diff --stat .github/workflows` empty).
- Removed from `tests/test_gitleaks_workflow_verification.sh` (script deleted): all six assertions. Moved to: `tests/issue_2494_gitleaks_workflow_guards.rs`.
- Removed from `tests/test_cargo_toml_validation.sh` (script deleted): all cases. Moved to: `tests/issue_2494_cargo_toml_validation.rs`.
- Removed from `tests/test_security_workflow_validation.sh`: cargo audit / cargo-audit present, rustsec/audit-check absent. Still covered by: `tests/issue_1657_workflow_dedup_audit.rs::security_workflow_keeps_pinned_cargo_audit_path`, `security_workflow_has_no_rustsec_audit_check`.
- Removed from `tests/test_shellcheck_workflow_pinning.sh`: SHA-pin, version-comment and ludeeus-absent checks. Still covered by: `tests/issue_1898_shellcheck_gate.rs::shellcheck_workflow_actions_stay_sha_pinned`, `no_workflow_depends_on_the_orphaned_shellcheck_wrapper`.
- Removed from `tests/test_quality_upgrade.sh`: `cargo-upgrade` installed, `cargo upgrade --dry-run` succeeds, `Cargo.toml` has `[dependencies]`, `Cargo.lock` exists. Obsolete: host-tool and network checks, not a repository property; `bump-deps.sh` handles a missing `cargo-upgrade`. Still covered by: `tests/issue_1266_cargo_lock_committed.rs::cargo_lock_exists_at_repo_root`.
- Docs sweep: grepped README, CONTRIBUTING and `docs/` (excluding `docs/archive/`) for the five deleted script names; no live reference. Archived PR summaries mention them as history and are left unchanged.
- Twinned copies: the CI `Validate Cargo.toml` pipeline was previously copied into the script; now the test runs the real one. No other copy found.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the 580 s foreground cap while running the whole test suite; shellcheck, pinning and type-check stages passed before the cap, fmt/clippy and the targeted tests were run separately; CI runs the full gate" -->

## Not checked

- Full `./quality.sh` test suite did not complete locally (see note above).
- No Spec/Standards reviewer sub-agents were dispatched: the issue has no acceptance-criteria section.
