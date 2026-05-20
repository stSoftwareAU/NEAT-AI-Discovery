//! Issue #1234: Dependency-update quarantine declared but never enforced.
//!
//! `bump-deps.sh` documents a `VIBE_BUMP_QUARANTINE_HOURS` window (default
//! 24h) intended to "dodge fast-flagged supply-chain attacks", and
//! `.github/workflows/upgrade-dependencies.yml` runs the weekly Cargo
//! upgrade. Previously neither tool actually filtered upgrades by publish
//! age — a poisoned version published shortly before the cron fired would
//! land in `Cargo.toml` immediately.
//!
//! This test enforces the configuration-level contract for the fix:
//!
//!   1. A Renovate config (`renovate.json`) exists at the repo root, and
//!      configures `minimumReleaseAge` of at least 24 hours for cargo
//!      dependencies — defence in depth in case Renovate is enabled.
//!   2. The scheduled upgrade workflow invokes `bump-deps.sh` rather than
//!      calling `cargo upgrade` directly. `bump-deps.sh` is the single
//!      place where the quarantine gate is enforced.
//!   3. `bump-deps.sh` has wiring that fetches publish times from
//!      crates.io and reverts in-quarantine bumps — the helper is no
//!      longer dead code.

use std::fs;
use std::path::Path;

#[test]
fn renovate_json_configures_minimum_release_age() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let renovate = root.join("renovate.json");
    assert!(
        renovate.is_file(),
        "renovate.json must exist at the repository root (Issue #1234) — \
         it is the defence-in-depth gate that delays externally-published \
         cargo crates by at least 24h"
    );
    let contents = fs::read_to_string(&renovate).expect("read renovate.json");
    assert!(
        contents.contains("minimumReleaseAge"),
        "renovate.json must set `minimumReleaseAge` (Issue #1234)"
    );
    // The window must be at least 24h. We accept the canonical literal
    // form "24h", "48h", … or "1 day".
    let has_window = contents.contains("\"24h\"")
        || contents.contains("\"48h\"")
        || contents.contains("\"72h\"")
        || contents.contains("\"1 day\"")
        || contents.contains("\"2 days\"")
        || contents.contains("\"3 days\"");
    assert!(
        has_window,
        "renovate.json must set `minimumReleaseAge` to at least 24h \
         (Issue #1234) — found: {contents}"
    );
    // The exception for internal stSoftwareAU/* deps must be documented.
    // Either the config has a packageRules entry that mentions
    // stSoftwareAU, or the file documents that there are no internal
    // deps in this repository.
    assert!(
        contents.contains("stSoftwareAU") || contents.contains("internal"),
        "renovate.json must document the stSoftwareAU/* internal-dep \
         exception (Issue #1234)"
    );
}

#[test]
fn upgrade_workflow_invokes_bump_deps_script() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workflow = root.join(".github/workflows/upgrade-dependencies.yml");
    let contents = fs::read_to_string(&workflow).expect("read upgrade-dependencies.yml");
    assert!(
        contents.contains("bump-deps.sh"),
        "upgrade-dependencies.yml must invoke ./bump-deps.sh so the \
         quarantine gate applies on the scheduled upgrade path \
         (Issue #1234) — direct `cargo upgrade` bypasses the gate"
    );
}

#[test]
fn bump_deps_script_enforces_quarantine() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script = root.join("bump-deps.sh");
    let contents = fs::read_to_string(&script).expect("read bump-deps.sh");
    // The helper must be invoked from the bump phase (not only defined
    // in the header). We require an actual crates.io API lookup helper
    // plus a use site that reverts an in-quarantine bump.
    assert!(
        contents.contains("crates.io/api/v1/crates"),
        "bump-deps.sh must query crates.io for publish time \
         (Issue #1234) — the existing quarantine plumbing currently \
         never reaches the API"
    );
    assert!(
        contents.contains("fetch_publish_epoch")
            || contents.contains("publish_epoch")
            || contents.contains("published_epoch"),
        "bump-deps.sh must define and call a publish-time fetch helper \
         (Issue #1234)"
    );
    assert!(
        contents.contains("revert") || contents.contains("rollback"),
        "bump-deps.sh must revert in-quarantine bumps rather than just \
         logging them (Issue #1234)"
    );
}
