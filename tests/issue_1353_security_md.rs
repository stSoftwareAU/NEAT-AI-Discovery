//! Issue #1353 (SCR-RUNBOOK): the repository must ship a `SECURITY.md` at its
//! root that closes the two supply-chain *readiness* gaps the audit flagged:
//!
//!   1. A **disclosure contact** so external researchers have a non-public
//!      channel instead of falling back to public issues (premature
//!      disclosure) or giving up.
//!   2. A brief **emergency dependency-bump procedure** that ties together the
//!      machinery that already exists — Renovate's security fast-lane
//!      (`vulnerabilityAlerts` with no quarantine wait) and the manual
//!      `VIBE_BUMP_QUARANTINE_HOURS=0 ./bump-deps.sh` override — plus the
//!      standing verification gate (`cargo audit` / `cargo deny check`).
//!
//! These tests assert on the real committed artefact (existence + the
//! content elements the runbook must name), not on source-code patterns.

use std::fs;
use std::path::Path;

fn read_security_md() -> String {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join("SECURITY.md");
    assert!(
        path.is_file(),
        "SECURITY.md must exist at the repo root (Issue #1353); expected {path:?}"
    );
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

#[test]
fn security_md_exists_at_repo_root() {
    // read_security_md asserts existence.
    let contents = read_security_md();
    assert!(
        !contents.trim().is_empty(),
        "SECURITY.md must not be empty (Issue #1353)"
    );
}

#[test]
fn security_md_names_disclosure_contact() {
    let contents = read_security_md();
    assert!(
        contents.contains("security@stsoftware.com.au"),
        "SECURITY.md must name a private disclosure contact \
         (security@stsoftware.com.au) so reporters have a non-public channel \
         (Issue #1353)"
    );
}

#[test]
fn security_md_documents_emergency_bump_procedure() {
    let contents = read_security_md();

    // The Renovate security fast-lane that bypasses the quarantine window.
    assert!(
        contents.contains("vulnerabilityAlerts"),
        "SECURITY.md must point at Renovate's security fast-lane \
         (vulnerabilityAlerts) for the emergency-bump procedure (Issue #1353)"
    );

    // The manual emergency override.
    assert!(
        contents.contains("VIBE_BUMP_QUARANTINE_HOURS=0"),
        "SECURITY.md must document the manual emergency override \
         (VIBE_BUMP_QUARANTINE_HOURS=0 ./bump-deps.sh) (Issue #1353)"
    );
    assert!(
        contents.contains("bump-deps.sh"),
        "SECURITY.md must reference bump-deps.sh for the manual emergency \
         bump (Issue #1353)"
    );
}

#[test]
fn security_md_names_verification_gate() {
    let contents = read_security_md();
    assert!(
        contents.contains("cargo audit"),
        "SECURITY.md must name `cargo audit` as part of the verification gate \
         (Issue #1353)"
    );
    assert!(
        contents.contains("cargo deny check"),
        "SECURITY.md must name `cargo deny check` as part of the verification \
         gate (Issue #1353)"
    );
}
