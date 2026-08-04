//! Issue #1994: the `--no-network` notice was unreachable without cargo-edit.
//!
//! The notice lived inside the `command -v cargo-upgrade` branch of
//! `bump-deps.sh`, so on a host with no cargo-edit installed the run took the
//! "external bumps skipped" branch and never logged that it was offline —
//! `tests/bump_deps_test.sh` Test 11 failed purely because of the host's
//! toolchain. Offline is a mode of the whole run, so it must be reported
//! regardless of which branch runs.
//!
//! This test drives the real script with a sanitised `PATH`/`HOME` that
//! guarantees the cargo-edit-absent branch is taken.

use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Directories that must stay on `PATH` for the script's coreutils
/// (`awk`, `sed`, `date`, `mktemp`, `git`).
const BASE_PATH: &str = "/usr/bin:/bin";

fn resolves_on(path: &str, program: &str) -> bool {
    path.split(':')
        .any(|dir| Path::new(dir).join(program).is_file())
}

#[test]
fn no_network_notice_prints_without_cargo_edit() {
    let bin_dir = tempfile::tempdir().expect("temp bin dir");
    // An empty HOME stops the script sourcing ~/.cargo/env, which would put
    // the real ~/.cargo/bin (and cargo-upgrade) back on PATH.
    let fake_home = tempfile::tempdir().expect("temp home");

    // Expose cargo — and only cargo — from the real toolchain.
    let cargo = PathBuf::from(env!("CARGO"));
    assert!(cargo.is_file(), "CARGO must point at the cargo binary");
    symlink(&cargo, bin_dir.path().join("cargo")).expect("symlink cargo");

    let path = format!("{}:{BASE_PATH}", bin_dir.path().display());
    assert!(
        !resolves_on(&path, "cargo-upgrade"),
        "the test needs a cargo-edit-free PATH to exercise the branch \
         (cargo-upgrade resolves on {path})"
    );

    let out = Command::new("bash")
        .arg(repo_root().join("bump-deps.sh"))
        .arg("--dry-run")
        .arg("--no-network")
        .current_dir(repo_root())
        .env("PATH", &path)
        .env("HOME", fake_home.path())
        .stdin(Stdio::null())
        .output()
        .expect("run bump-deps.sh");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        out.status.success(),
        "an offline dry-run must exit 0 without cargo-edit (Issue #1994). \
         stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("cargo-edit not installed"),
        "the test must actually exercise the cargo-edit-absent branch \
         (Issue #1994). stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("--no-network set"),
        "the offline mode must be reported even when cargo-edit is absent \
         (Issue #1994). stdout:\n{stdout}"
    );
}

#[test]
fn run_without_no_network_does_not_claim_to_be_offline() {
    let bin_dir = tempfile::tempdir().expect("temp bin dir");
    let fake_home = tempfile::tempdir().expect("temp home");
    let cargo = PathBuf::from(env!("CARGO"));
    symlink(&cargo, bin_dir.path().join("cargo")).expect("symlink cargo");

    // Same cargo-edit-free environment, so this dry-run touches no network.
    let path = format!("{}:{BASE_PATH}", bin_dir.path().display());
    let out = Command::new("bash")
        .arg(repo_root().join("bump-deps.sh"))
        .arg("--dry-run")
        .current_dir(repo_root())
        .env("PATH", &path)
        .env("HOME", fake_home.path())
        .stdin(Stdio::null())
        .output()
        .expect("run bump-deps.sh");

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "a dry-run must exit 0 without cargo-edit. stdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !stdout.contains("--no-network set"),
        "a run without --no-network must not report offline mode. \
         stdout:\n{stdout}"
    );
}
