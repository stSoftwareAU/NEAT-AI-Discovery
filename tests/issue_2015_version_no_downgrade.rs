//! Issue #2015: refuse a `Cargo.toml` version downgrade vs the PR base.
//!
//! The CI `version-increment` job used to treat any `CURRENT != BASE` as
//! "already bumped" and skip. A merge conflict that silently took Develop's
//! older version therefore looked like a deliberate bump and shipped a
//! **downgrade** — remote `runlib.sh` rebuilds key off this crate version.
//!
//! These WHAT-tests drive `scripts/check-version-no-downgrade.sh` (the same
//! guard wired into `.github/workflows/ci.yml`) and assert the observable
//! policy:
//!
//!   * head < base  → fail
//!   * head == base → pass (CI may still auto-patch-bump)
//!   * head > base  → pass (already ahead; no forced second bump)
//!
//! They also pin that the version-increment step invokes the script before
//! the skip-on-difference logic, so a downgrade cannot be waved through.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script_path() -> PathBuf {
    repo_root().join("scripts/check-version-no-downgrade.sh")
}

fn run_guard(base: &str, head: &str) -> (bool, String) {
    let output = Command::new("bash")
        .arg(script_path())
        .arg(base)
        .arg(head)
        .stdin(Stdio::null())
        .output()
        .expect("run check-version-no-downgrade.sh");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.success(), combined)
}

#[test]
fn behind_base_fails() {
    let (ok, msg) = run_guard("0.74.218", "0.74.217");
    assert!(
        !ok,
        "head strictly behind base must fail (Issue #2015) — got:\n{msg}"
    );
    assert!(
        msg.contains("downgrad") || msg.contains("backwards"),
        "failure must name the downgrade so the cause is obvious (Issue #2015) — got:\n{msg}"
    );
}

#[test]
fn equal_to_base_passes() {
    let (ok, msg) = run_guard("0.74.218", "0.74.218");
    assert!(
        ok,
        "equal versions must pass so CI can still auto-patch-bump (Issue #2015) — got:\n{msg}"
    );
}

#[test]
fn ahead_of_base_passes() {
    let (ok, msg) = run_guard("0.74.218", "0.74.219");
    assert!(
        ok,
        "head ahead of base must pass without forcing another bump (Issue #2015) — got:\n{msg}"
    );
}

#[test]
fn major_and_minor_ordering_is_numeric() {
    // Patch-only comparison is not enough: 0.75.0 must beat 0.74.999, and
    // 1.0.0 must beat 0.99.99.
    assert!(
        run_guard("0.74.999", "0.75.0").0,
        "minor bump must count as ahead (Issue #2015)"
    );
    assert!(
        !run_guard("0.75.0", "0.74.999").0,
        "minor downgrade must fail (Issue #2015)"
    );
    assert!(
        run_guard("0.99.99", "1.0.0").0,
        "major bump must count as ahead (Issue #2015)"
    );
    assert!(
        !run_guard("1.0.0", "0.99.99").0,
        "major downgrade must fail (Issue #2015)"
    );
}

#[test]
fn malformed_versions_fail() {
    let (ok, msg) = run_guard("not-a-version", "0.1.0");
    assert!(!ok, "malformed base must fail — got:\n{msg}");
    let (ok, msg) = run_guard("0.1.0", "1.2");
    assert!(!ok, "two-part head must fail — got:\n{msg}");
}

/// The version-increment step must call the no-downgrade guard before it
/// treats `CURRENT != BASE` as "already bumped".
#[test]
fn version_increment_step_invokes_no_downgrade_guard() {
    let body = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/ci.yml"),
    )
    .expect("read ci.yml");
    let header = "    - name: Check for source changes and increment version\n";
    let start = body
        .find(header)
        .expect("version-increment step present in ci.yml");
    // Bound the search to this step's `run: |` body (until the next top-level step).
    let after = &body[start..];
    let next = after[header.len()..]
        .find("\n    - name: ")
        .map(|i| header.len() + i)
        .unwrap_or(after.len());
    let step = &after[..next];

    assert!(
        step.contains("scripts/check-version-no-downgrade.sh"),
        "version-increment step must invoke scripts/check-version-no-downgrade.sh (Issue #2015)"
    );

    let guard_at = step
        .find("check-version-no-downgrade.sh")
        .expect("guard invocation");
    let skip_at = step
        .find("already ahead of base branch")
        .or_else(|| step.find("already different from base branch"))
        .expect("skip-on-ahead message present");
    assert!(
        guard_at < skip_at,
        "no-downgrade guard must run before the skip-on-difference logic \
         so a downgrade cannot be waved through (Issue #2015)"
    );
}
