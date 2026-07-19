//! Issue #1657: `.github/workflows/security.yml` ran the same `RustSec`
//! advisory audit over the repository's `Cargo.lock` twice per pull request:
//!
//!   1. An explicit `cargo install --locked --version 0.22.1 cargo-audit`
//!      followed by `cargo audit` (pinned for `CVSS` 4.0 support, Issue #1223).
//!   2. The `rustsec/audit-check` GitHub Action, which wraps cargo-audit over
//!      the same lockfile with its own bundled cargo-audit version.
//!
//! Both consult the same advisory database against the same dependency tree in
//! the same run, so a newly disclosed advisory fails the job at the first
//! `cargo audit` before the action ever adds signal — redundant work on every
//! PR, and a drift risk (the explicit path is version-pinned for `CVSS` 4.0
//! while the action resolves its own cargo-audit version).
//!
//! The fix keeps the version-pinned explicit path and removes the
//! `rustsec/audit-check` step. With the action gone, the `checks: write`
//! permission it required is no longer needed and is dropped for least
//! privilege; `issues: write` stays because `dependency-review-action` posts
//! its PR summary comment through the Issues API.
//!
//! This test parses `security.yml` as plain text (no YAML parser is in the
//! dependency tree) and asserts the single audit path is retained and the
//! duplicate is gone.

use std::fs;
use std::path::{Path, PathBuf};

fn read_security_workflow() -> String {
    let path: PathBuf =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/security.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn security_workflow_has_no_rustsec_audit_check() {
    let body = read_security_workflow();
    assert!(
        !body.contains("rustsec/audit-check"),
        "security.yml must not use `rustsec/audit-check` — it duplicates the \
         explicit `cargo install --locked cargo-audit` + `cargo audit` path, \
         auditing the same Cargo.lock against the same advisory database in the \
         same run for no added coverage (Issue #1657)",
    );
}

#[test]
fn security_workflow_keeps_pinned_cargo_audit_path() {
    let body = read_security_workflow();
    // The retained single audit path is the version-pinned explicit install
    // (CVSS 4.0 support, Issue #1223) followed by the `cargo audit` invocation.
    assert!(
        body.contains("cargo install --locked --version 0.22.1 cargo-audit"),
        "security.yml must retain the version-pinned `cargo install --locked \
         --version 0.22.1 cargo-audit` install as the single audit path \
         (Issue #1657)",
    );
    assert!(
        body.contains("run: cargo audit"),
        "security.yml must retain the `cargo audit` invocation as the single \
         RustSec advisory gate (Issue #1657)",
    );
}

#[test]
fn security_workflow_drops_unused_checks_write_permission() {
    let body = read_security_workflow();
    // `checks: write` was requested solely for `rustsec/audit-check`'s check
    // annotations. With the action removed it has no consumer, so it is dropped
    // (least privilege). `issues: write` remains for dependency-review's PR
    // summary comment, which is posted via the Issues API.
    assert!(
        !body.contains("checks: write"),
        "security.yml must drop the now-unused `checks: write` permission — it \
         was needed only by the removed `rustsec/audit-check` action \
         (Issue #1657)",
    );
    assert!(
        body.contains("issues: write"),
        "security.yml must keep `issues: write` — dependency-review-action \
         posts its PR summary comment through the Issues API (Issue #1657)",
    );
}
