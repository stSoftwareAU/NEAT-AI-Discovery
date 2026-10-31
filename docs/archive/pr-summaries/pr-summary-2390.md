## Summary

Ported the assertions of `tests/test_cargo_quality_workflow.sh` into a Rust integration test, `tests/issue_2390_coverage_workflow_guards.rs`, which `cargo test` (and therefore `./quality.sh` and CI) runs, and deleted the script. The stale `dtolnay/rust-toolchain` check is replaced by a check for `./.github/actions/setup-rust`. Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version. Closes #2390.

## Spec

### Intent and Rationale

- The only guard for the Coverage workflow's `persist-credentials: false` (Issue #1567) was a shell script no gate ran and that already failed on Develop. A Rust test under `tests/` is run by the existing gate, matching `issue_1656_cargo_quality_milestone_filter.rs` and `issue_1287_workflow_timeouts.rs`.
- The script was deleted rather than wired into `quality.sh`, so there is one copy of the rule and it cannot go stale again.

### Essential Design Decisions

- Each check is a pure function over the workflow text, so it is tested both against the real file and against fixtures that must fail; the `persist-credentials`, `continue-on-error` and step lookups are scoped to the step's own block, and a missing step name or `steps:` key is an error (fail closed), never a pass.
- The old `dtolnay/rust-toolchain` presence check is inverted: it is now forbidden (non-comment lines) and `./.github/actions/setup-rust` is required, consistent with `tests/issue_1891_rust_toolchain_install.rs`.
- The `cargo-llvm-cov` presence check keeps the original script's loose substring match (the step name alone satisfies it). Tightening it was not asked for.

### Undiscoverable Facts

- Develop's `./quality.sh` failed at the `cargo doc` stage (`RUSTDOCFLAGS="-D warnings"`) on two doc links in `src/analysis/detection/compound_degradation.rs` (a link to the private `OutputBitset`, and a redundant explicit link target on `ScanTruncation`). This PR fixes both doc lines so the gate can pass; no code changed there.
- Running `bash tests/test_cargo_quality_workflow.sh` on Develop printed `Results: 8 passed, 1 failed` (the stale `dtolnay` assertion). Other `tests/*.sh` scripts are also unrun by the gate; they are out of scope here.

## Evidence

Backend/CLI-only change (no UI files). Evidence is the test run below. Checks ported: `pull_request` trigger, checkout `persist-credentials: false`, `contents: read`, no write permission, `setup-rust` used and `dtolnay/rust-toolchain` absent, `cargo-llvm-cov` and `codecov/codecov-action@` present, no `cargo fmt`/`cargo clippy`, Codecov step `continue-on-error: true`, coverage-generation step without `continue-on-error`.

## Test Plan

- Added `tests/issue_2390_coverage_workflow_guards.rs`: 11 real-file tests plus 21 fixture tests (positive baseline and one negative per check; a fixture that has `persist-credentials: false` only in another step must fail). `cargo test --test issue_2390_coverage_workflow_guards`: 32 passed.
- Red checks against the real workflow (each restored afterwards; workflow has no diff): removing `persist-credentials: false` → `real_workflow_checkout_does_not_persist_credentials` red; removing `continue-on-error: true` from the Codecov step → `real_workflow_codecov_upload_continues_on_error` red; adding a `cargo clippy` run step → `real_workflow_runs_no_cargo_fmt_or_clippy` red.
- Removed from `tests/test_cargo_quality_workflow.sh` (deleted file): all nine assertions. Moved to `tests/issue_2390_coverage_workflow_guards.rs`, except `dtolnay/rust-toolchain reference present`, which the issue says is stale and is replaced by the `setup-rust` check.
- `cargo fmt -- --check` and `cargo clippy --test issue_2390_coverage_workflow_guards -- -D warnings`: clean.
- Twinned rules: the sibling tests `issue_1656_...`, `issue_1287_...`, `issue_1891_...` and `issue_2036_...` also read `cargo-quality.yml` for different properties (branch filter, timeouts, toolchain action); none checks the same property, so no disagreement is possible.
