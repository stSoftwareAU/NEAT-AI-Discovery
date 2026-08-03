//! Issue #1912: enforce the Issue #1223 `cargo install` pinning invariant
//! repository-wide.
//!
//! #1223 pinned the three workflow call sites but nothing enforced the rule, so
//! `scripts/fuzz-ci.sh` kept running an unpinned `cargo install cargo-fuzz`.
//! `quality/cargo_install_pinning.sh` is the committed gate that closes that
//! hole; these tests execute it for real — against purpose-built fixture trees
//! and against this repository — and assert on exit codes and reported paths,
//! never on the gate's source text.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn gate() -> PathBuf {
    repo_root().join("quality/cargo_install_pinning.sh")
}

/// Runs the gate over `root` and returns its raw output.
fn run_gate(root: &Path) -> Output {
    Command::new("bash")
        .arg(gate())
        .arg(root)
        .current_dir(repo_root())
        .output()
        .expect("run cargo_install_pinning.sh")
}

/// Builds a temp tree containing a single file and scans it.
fn scan_fixture(name: &str, contents: &str) -> (Output, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("create temp dir");
    fs::write(dir.path().join(name), contents).expect("write fixture");
    let output = run_gate(dir.path());
    (output, dir)
}

fn combined(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn this_repository_pins_every_cargo_install() {
    let output = run_gate(&repo_root());
    assert!(
        output.status.success(),
        "the repository has an unpinned `cargo install`:\n{}",
        combined(&output)
    );
}

#[test]
fn unpinned_install_in_a_shell_script_fails() {
    let (output, _dir) = scan_fixture(
        "fuzz-ci.sh",
        "#!/bin/bash\ncargo +nightly install cargo-fuzz\n",
    );
    assert!(
        !output.status.success(),
        "unpinned install must fail the gate"
    );
    let report = combined(&output);
    assert!(
        report.contains("fuzz-ci.sh") && report.contains("--locked"),
        "the failure must name the file and the missing flag: {report}"
    );
}

#[test]
fn install_without_a_version_pin_fails() {
    let (output, _dir) = scan_fixture("install.sh", "cargo install --locked cargo-fuzz\n");
    assert!(
        !output.status.success(),
        "a missing --version pin must fail the gate"
    );
    assert!(
        combined(&output).contains("--version"),
        "the failure must name the missing --version pin"
    );
}

#[test]
fn unpinned_install_in_a_workflow_fails() {
    let (output, _dir) = scan_fixture(
        "fuzz.yml",
        "jobs:\n  fuzz:\n    steps:\n      - run: cargo install cargo-deny\n",
    );
    assert!(
        !output.status.success(),
        "unpinned workflow install must fail the gate"
    );
    assert!(combined(&output).contains("fuzz.yml"));
}

#[test]
fn fully_pinned_install_passes() {
    let (output, _dir) = scan_fixture(
        "fuzz-ci.sh",
        "#!/bin/bash\ncargo +nightly install --locked --version 0.13.2 cargo-fuzz\n",
    );
    assert!(
        output.status.success(),
        "a --locked --version install must pass: {}",
        combined(&output)
    );
}

/// Advisory messages and comments describe an install, they do not run one —
/// flagging them would make the gate unusable (`bump-deps.sh` prints install
/// hints, and every call site carries an explanatory comment).
#[test]
fn mentions_in_messages_and_comments_are_not_invocations() {
    let (output, _dir) = scan_fixture(
        "hints.sh",
        concat!(
            "#!/bin/bash\n",
            "# Install it: cargo install cargo-deny\n",
            "echo \"not installed (install: cargo install cargo-edit)\" >&2\n",
            "cat <<'USAGE'\n",
            "  9  cargo-deny missing (install: cargo install --locked cargo-deny).\n",
            "USAGE\n",
        ),
    );
    assert!(
        output.status.success(),
        "documentation mentions must not be treated as invocations: {}",
        combined(&output)
    );
}

/// A gate that scans nothing must fail loudly rather than report success.
#[test]
fn empty_and_missing_roots_fail_loudly() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let empty = run_gate(dir.path());
    assert!(
        !empty.status.success(),
        "an empty tree must not report success"
    );

    let missing = run_gate(&dir.path().join("does-not-exist"));
    assert!(!missing.status.success(), "a missing root must fail");
    assert!(combined(&missing).contains("not found"));
}
