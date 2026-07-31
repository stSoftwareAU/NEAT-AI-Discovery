//! Issue #1870: the declared cargo-deny supply-chain policy must actually be
//! enforced.
//!
//! `deny.toml` declares a strict `[sources]` policy (crates.io only, no git
//! remotes, no alternate registries) plus `[bans]` and `[licenses]` tables.
//! Nothing in `.github/workflows/` consulted it: the security workflow ran
//! `cargo audit` and `dependency-review-action`, both of which check published
//! *advisories* only and never read `deny.toml`. A pull request adding
//! `foo = { git = "https://github.com/attacker/foo" }` — bypassing crates.io
//! checksums entirely — passed every required check.
//!
//! The two paths that did run `cargo deny check` were honour-system:
//! `quality.sh` (a local pre-commit gate CI never runs) and `bump-deps.sh`
//! phase 4, which *skipped* the gate with a warning when `cargo-deny` was not
//! installed — a silent pass on any host without the tool.
//!
//! The fix wires an enforced `cargo deny check` step into
//! `.github/workflows/security.yml` (version-pinned and `--locked`, matching
//! the cargo-audit install) and makes `bump-deps.sh` hard-fail when
//! `cargo-deny` is missing.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn read_security_workflow() -> String {
    let path: PathBuf = repo_root().join(".github/workflows/security.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Write an executable stub script into `dir`.
fn write_stub(dir: &Path, name: &str, body: &str) {
    let path = dir.join(name);
    fs::write(&path, body).expect("write stub");
    let mut perms = fs::metadata(&path).expect("stat stub").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&path, perms).expect("chmod stub");
}

// ── CI enforcement ────────────────────────────────────────────────────

#[test]
fn security_workflow_installs_cargo_deny_pinned_and_locked() {
    let body = read_security_workflow();
    let install = body
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("run: cargo install") && line.contains("cargo-deny"))
        .unwrap_or_else(|| {
            panic!(
                "security.yml must install cargo-deny so the declared deny.toml \
                 policy is enforced in CI (Issue #1870)"
            )
        });
    assert!(
        install.contains("--locked"),
        "the cargo-deny install must pass --locked so the plugin's own \
         Cargo.lock is honoured — without it a poisoned transitive release \
         executes via build.rs in this workflow (Issue #1870): {install}"
    );
    assert!(
        install.contains("--version "),
        "the cargo-deny install must pin an explicit --version, matching the \
         cargo-audit install (Issue #1870): {install}"
    );
}

#[test]
fn security_workflow_runs_cargo_deny_check() {
    let body = read_security_workflow();
    assert!(
        body.lines()
            .map(str::trim)
            .any(|line| line.starts_with("run:") && line.contains("cargo deny check")),
        "security.yml must run `cargo deny check` — without it the [sources], \
         [bans] and [licenses] tables in deny.toml are advisory only, and a PR \
         adding a git or alternate-registry dependency passes every required \
         check (Issue #1870)"
    );
}

// ── Local path: bump-deps.sh must not silently skip the audit gate ────

/// Run `bump-deps.sh --no-network` with a stubbed `PATH`. `cargo_deny_stub`
/// selects whether a `cargo-deny` executable is visible to the script.
fn run_bump_deps(cargo_deny_stub: bool) -> std::process::Output {
    let tmp = tempfile::tempdir().expect("temp dir");
    let bin = tmp.path().join("bin");
    fs::create_dir_all(&bin).expect("create stub bin dir");

    // Stub `cargo`: succeed, mutate nothing. `--no-network` keeps the run off
    // crates.io, so the only cargo call reached is the audit gate itself.
    write_stub(&bin, "cargo", "#!/bin/bash\nexit 0\n");
    if cargo_deny_stub {
        write_stub(&bin, "cargo-deny", "#!/bin/bash\nexit 0\n");
    }

    // HOME points at the temp dir so bump-deps.sh does not source the real
    // ~/.cargo/env and prepend a real cargo-deny ahead of the stub PATH.
    Command::new("bash")
        .arg("bump-deps.sh")
        .arg("--no-network")
        .current_dir(repo_root())
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("HOME", tmp.path())
        .stdin(Stdio::null())
        .output()
        .expect("run bump-deps.sh")
}

#[test]
fn bump_deps_hard_fails_when_cargo_deny_is_missing() {
    let output = run_bump_deps(false);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "bump-deps.sh must fail loud when cargo-deny is absent — skipping the \
         audit gate with a warning silently passes the dependency-source \
         policy on any host without the tool (Issue #1870)\n\
         stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("cargo-deny"),
        "the failure must name cargo-deny so the operator knows what to \
         install (Issue #1870)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        !stdout.contains("audit gate skipped"),
        "the audit gate must never report itself as skipped — absence of a \
         failure is not a pass (Issue #1870)\nstdout:\n{stdout}"
    );
}

#[test]
fn bump_deps_runs_the_audit_gate_when_cargo_deny_is_present() {
    let output = run_bump_deps(true);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "bump-deps.sh must succeed when cargo-deny is installed and the audit \
         gate passes (Issue #1870)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("audit gate") && stdout.contains("audit_run=1"),
        "the run must positively confirm the audit gate executed (Issue \
         #1870)\nstdout:\n{stdout}"
    );
}
