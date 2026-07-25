//! Behavioural tests for Issue #1755: committed `bash -n` syntax gate.
//!
//! The repository ships `quality/bash_syntax.sh` — a committed gate script that
//! runs `bash -n` over every tracked shell script. It is invoked both by
//! `./quality.sh` (local gate) and by a CI workflow (pull-request gate), so a
//! syntax error can never reach the default branch.
//!
//! Every test here runs the real script against real fixture trees and asserts
//! on its exit code and output — nothing greps the script's source.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn gate_script() -> PathBuf {
    repo_root().join("quality/bash_syntax.sh")
}

/// Run the gate over `root`, returning the raw process output.
fn run_gate(root: &Path) -> Output {
    Command::new("bash")
        .arg(gate_script())
        .arg(root)
        .current_dir(repo_root())
        .output()
        .expect("failed to run quality/bash_syntax.sh")
}

fn combined(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn write_script(dir: &Path, name: &str, body: &str) {
    if let Some(parent) = dir.join(name).parent() {
        fs::create_dir_all(parent).expect("create fixture directory");
    }
    fs::write(dir.join(name), body).expect("write fixture script");
}

const VALID: &str = "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n";
// Unterminated `if` — bash rejects this at parse time, before execution.
const BROKEN: &str = "#!/usr/bin/env bash\nif [ 1 -eq 1 ]; then\n  echo oops\n";

#[test]
fn gate_passes_over_the_repository_itself() {
    let output = run_gate(&repo_root());
    assert!(
        output.status.success(),
        "quality/bash_syntax.sh must pass over this repository:\n{}",
        combined(&output)
    );
}

#[test]
fn gate_accepts_a_tree_of_valid_scripts() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_script(tmp.path(), "top.sh", VALID);
    write_script(tmp.path(), "nested/deep.sh", VALID);

    let output = run_gate(tmp.path());
    assert!(
        output.status.success(),
        "gate should accept syntactically valid scripts:\n{}",
        combined(&output)
    );
}

#[test]
fn gate_fails_loudly_on_a_syntax_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_script(tmp.path(), "good.sh", VALID);
    write_script(tmp.path(), "broken.sh", BROKEN);

    let output = run_gate(tmp.path());
    assert!(
        !output.status.success(),
        "gate must exit non-zero when a script has a syntax error:\n{}",
        combined(&output)
    );
    assert!(
        combined(&output).contains("broken.sh"),
        "gate must name the offending script:\n{}",
        combined(&output)
    );
}

#[test]
fn gate_reports_every_broken_script_not_just_the_first() {
    let tmp = tempfile::tempdir().expect("tempdir");
    // Names chosen so `first` sorts before `second` in any traversal order.
    write_script(tmp.path(), "first_broken.sh", BROKEN);
    write_script(tmp.path(), "second_broken.sh", "#!/bin/bash\ncase x in\n");

    let output = run_gate(tmp.path());
    let text = combined(&output);
    assert!(!output.status.success(), "gate must fail:\n{text}");
    assert!(
        text.contains("first_broken.sh") && text.contains("second_broken.sh"),
        "gate must report all failing scripts, not stop at the first:\n{text}"
    );
}

#[test]
fn gate_scans_scripts_with_spaces_in_their_path() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_script(tmp.path(), "a dir/my broken script.sh", BROKEN);

    let output = run_gate(tmp.path());
    assert!(
        !output.status.success(),
        "gate must not skip paths containing spaces:\n{}",
        combined(&output)
    );
}

#[test]
fn gate_ignores_build_artefacts_and_git_internals() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_script(tmp.path(), "real.sh", VALID);
    write_script(tmp.path(), "target/debug/build/vendored.sh", BROKEN);
    write_script(tmp.path(), ".git/hooks/sample.sh", BROKEN);

    let output = run_gate(tmp.path());
    assert!(
        output.status.success(),
        "gate must skip ./target and ./.git so vendored scripts cannot fail the build:\n{}",
        combined(&output)
    );
}

#[test]
fn gate_fails_when_it_finds_nothing_to_scan() {
    // A gate that silently scans zero scripts looks green while checking
    // nothing — absence of a failure is not success.
    let tmp = tempfile::tempdir().expect("tempdir");
    let output = run_gate(tmp.path());
    assert!(
        !output.status.success(),
        "gate must fail when no scripts are found:\n{}",
        combined(&output)
    );
}

#[test]
fn gate_fails_on_a_missing_root() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let missing = tmp.path().join("does-not-exist");

    let output = run_gate(&missing);
    assert!(
        !output.status.success(),
        "gate must fail when the scan root does not exist:\n{}",
        combined(&output)
    );
}

#[test]
fn quality_gate_invokes_the_committed_bash_syntax_script() {
    let quality = fs::read_to_string(repo_root().join("quality.sh")).expect("read quality.sh");
    assert!(
        quality.contains("quality/bash_syntax.sh"),
        "./quality.sh must invoke the committed bash syntax gate (Issue #1755)"
    );
}

/// The gate only protects the default branch if CI actually runs it on pull
/// requests. Verify some workflow invokes the committed script.
#[test]
fn a_ci_workflow_invokes_the_bash_syntax_gate_on_pull_requests() {
    let wf_dir = repo_root().join(".github/workflows");
    let mut invoking = Vec::new();
    for entry in fs::read_dir(&wf_dir).expect("read workflows dir").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yml") {
            continue;
        }
        let body = fs::read_to_string(&path).expect("read workflow");
        if body.contains("quality/bash_syntax.sh") {
            assert!(
                body.contains("pull_request"),
                "{} invokes the bash syntax gate but is not triggered on pull requests",
                path.display()
            );
            invoking.push(path);
        }
    }
    assert!(
        !invoking.is_empty(),
        "no CI workflow invokes quality/bash_syntax.sh — invalid bash could land on the \
         default branch (Issue #1755)"
    );
}
