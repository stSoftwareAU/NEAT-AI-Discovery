//! Issue #2054: the `bump-deps.sh` audit gate must report the cause
//! `cargo deny check` actually gave.
//!
//! A stale `[advisories] ignore` in `deny.toml` — `RUSTSEC-2024-0436` for
//! `paste`, whose reason recorded that it would die once parquet migrated to
//! `pastey` — became unmatched the moment `parquet 59.3.0` dropped `paste`
//! from the graph. `unused-ignored-advisory = "deny"` (Issue #1917) then
//! failed the whole `cargo deny check`, so `bump-deps.sh` exited 7 on every
//! run and the worker reverted every bump: dependency bumps were disabled for
//! the repository.
//!
//! Triage took three runs because the gate lied about the cause. It scraped
//! the first `<name> vX.Y.Z` token out of the cargo-deny log and called it
//! "the offending crate" — but cargo-deny prints its inclusion graph and its
//! yanked/duplicate warnings *before* the failing diagnostic, so the token was
//! `getrandom v0.2.17`, a crate with nothing to do with the failure.
//!
//! These tests drive `bump_deps::describe_deny_failure` against recorded
//! cargo-deny logs and assert it names the failing check, the diagnostic and
//! the config location — and never a bystander crate.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn bump_deps_script() -> PathBuf {
    repo_root().join("bump-deps.sh")
}

/// Source `bump-deps.sh` in helper-only mode and run `describe_deny_failure`
/// against `log_body`, returning its stdout.
fn describe(log_body: &str) -> String {
    let dir = tempfile::tempdir().expect("temp dir");
    let log = dir.path().join("deny.log");
    std::fs::write(&log, log_body).expect("write deny log fixture");
    describe_path(&log.display().to_string())
}

/// Run `describe_deny_failure` against an arbitrary path (which need not
/// exist), returning its stdout.
fn describe_path(log_path: &str) -> String {
    let script = format!(
        "source {} && bump_deps::describe_deny_failure {}",
        shell_quote(&bump_deps_script().display().to_string()),
        shell_quote(log_path),
    );
    let output = Command::new("bash")
        .env("BUMP_DEPS_SOURCE_ONLY", "1")
        .arg("-c")
        .arg(&script)
        .output()
        .expect("run bump-deps.sh helper");
    assert!(
        output.status.success(),
        "describe_deny_failure must exit 0 (it only formats a message); \
         stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

/// The log that disabled bumps for this repo: a long inclusion graph naming
/// unrelated crates, then the real diagnostic.
const STALE_IGNORE_LOG: &str = r#"warning[yanked]: detected yanked crate (try `cargo update -p chacha20`)
   ┌─ /repo/Cargo.lock:47:1
   │
47 │ chacha20 0.10.1 registry+https://github.com/rust-lang/crates.io-index
   │
   ├ getrandom v0.2.17
     └── rand_core v0.6.4
         ├── wgpu-hal v30.0.1 (*)
         └── wgpu-types v30.0.1 (*)

error[advisory-not-detected]: advisory was not encountered
   ┌─ /repo/deny.toml:62:13
   │
62 │     { id = "RUSTSEC-2024-0436", reason = "transitive dep from parquet" },
   │             no crate matched advisory criteria

advisories FAILED, bans ok, licenses ok, sources ok
"#;

/// A different failing check, so the helper is not fitted to one advisory.
const LICENCE_FAILURE_LOG: &str = r#"    Checking neat_ai_discovery v0.74.232
   ├ serde v1.0.228
     └── neat_ai_discovery v0.74.232

error[rejected]: failed to satisfy license requirements
   ┌─ /repo/Cargo.lock:912:1
   │
912 │ some-crate 1.2.3 registry+https://github.com/rust-lang/crates.io-index
   │

advisories ok, bans ok, licenses FAILED, sources ok
"#;

// ── The real cause is reported ────────────────────────────────────────

#[test]
fn names_the_check_that_failed() {
    let described = describe(STALE_IGNORE_LOG);
    assert!(
        described.contains("failing checks: advisories"),
        "the audit gate must name the cargo-deny check that failed \
         (Issue #2054); got:\n{described}"
    );
}

#[test]
fn quotes_the_diagnostic_cargo_deny_emitted() {
    let described = describe(STALE_IGNORE_LOG);
    assert!(
        described.contains("error[advisory-not-detected]: advisory was not encountered"),
        "the audit gate must quote cargo-deny's own diagnostic headline \
         (Issue #2054); got:\n{described}"
    );
}

#[test]
fn points_at_the_config_line_that_failed() {
    let described = describe(STALE_IGNORE_LOG);
    assert!(
        described.contains("/repo/deny.toml:62:13"),
        "the audit gate must point at the file:line cargo-deny blamed, so the \
         next reader lands on the stale ignore (Issue #2054); got:\n{described}"
    );
    assert!(
        !described.contains("/repo/Cargo.lock:47:1"),
        "the location under a `warning[yanked]` is not why the gate failed and \
         must not be reported as such (Issue #2054); got:\n{described}"
    );
}

#[test]
fn does_not_blame_a_bystander_crate_from_the_inclusion_graph() {
    let described = describe(STALE_IGNORE_LOG);
    assert!(
        !described.contains("getrandom"),
        "the audit gate must not report the first crate in cargo-deny's \
         inclusion graph as the offender — `getrandom v0.2.17` had nothing to \
         do with the stale deny.toml ignore (Issue #2054); got:\n{described}"
    );
    assert!(
        !described.contains("chacha20"),
        "a yanked-crate warning is not the failure either (Issue #2054); \
         got:\n{described}"
    );
}

// ── The same treatment for a different failing check ──────────────────

#[test]
fn reports_a_licence_rejection_by_its_own_check_and_diagnostic() {
    let described = describe(LICENCE_FAILURE_LOG);
    assert!(
        described.contains("failing checks: licenses"),
        "a licence rejection must be reported as the `licenses` check failing, \
         not the `advisories` one (Issue #2054); got:\n{described}"
    );
    assert!(
        described.contains("error[rejected]: failed to satisfy license requirements"),
        "the licence diagnostic must be quoted (Issue #2054); got:\n{described}"
    );
    assert!(
        !described.contains("serde"),
        "the first crate in the graph is not the offender here either \
         (Issue #2054); got:\n{described}"
    );
}

// ── Unrecognisable and missing logs fail loud, not silently ───────────

#[test]
fn an_unrecognisable_log_says_so_instead_of_inventing_an_offender() {
    let described = describe("cargo-deny exploded in some new way\nfoo v1.2.3\n");
    assert!(
        described.contains("cargo deny check rejected the tree"),
        "an unparsable log must still produce a plain statement that the gate \
         rejected the tree (Issue #2054); got:\n{described}"
    );
    assert!(
        !described.contains("foo v1.2.3"),
        "an unparsable log must not be mined for a crate name to blame \
         (Issue #2054); got:\n{described}"
    );
}

#[test]
fn a_missing_log_is_reported_rather_than_swallowed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let missing = dir.path().join("never-written.log");
    let described = describe_path(&missing.display().to_string());
    assert!(
        described.contains("could not be read"),
        "an unreadable log must be reported — absence of a diagnostic is not \
         success (Issue #2054); got:\n{described}"
    );
}
