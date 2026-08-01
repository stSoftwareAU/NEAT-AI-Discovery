//! Behavioural tests for Issue #1898: committed `ShellCheck` lint gate.
//!
//! The unmaintained `ludeeus/action-shellcheck` wrapper action (last release
//! January 2023, last push June 2024) has been removed. In its place the
//! repository ships `quality/shellcheck.sh` — a committed gate script that runs
//! the koalaman `ShellCheck` binary over every shell script. It is invoked both by
//! `./quality.sh` (local gate) and by a CI workflow (pull-request gate), so both
//! enforce exactly the same rules.
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
    repo_root().join("quality/shellcheck.sh")
}

fn shellcheck_available() -> bool {
    Command::new("shellcheck")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Run the gate over `root`, returning the raw process output.
fn run_gate(root: &Path) -> Output {
    Command::new("bash")
        .arg(gate_script())
        .arg(root)
        .current_dir(repo_root())
        .output()
        .expect("failed to run quality/shellcheck.sh")
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

const CLEAN: &str = "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n";
// SC2086: unquoted expansion — a ShellCheck "info" finding is not enough, so
// this uses an error-level defect that ShellCheck always reports.
const DIRTY: &str = "#!/usr/bin/env bash\nset -euo pipefail\nrm $1\n";

fn load_shellcheck_yml() -> String {
    let path = repo_root().join(".github/workflows/shellcheck.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn gate_passes_over_the_repository_itself() {
    if !shellcheck_available() {
        eprintln!("skipping: shellcheck is not installed");
        return;
    }
    let output = run_gate(&repo_root());
    assert!(
        output.status.success(),
        "quality/shellcheck.sh must pass over this repository:\n{}",
        combined(&output)
    );
}

#[test]
fn gate_accepts_a_tree_of_clean_scripts() {
    if !shellcheck_available() {
        eprintln!("skipping: shellcheck is not installed");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    write_script(tmp.path(), "top.sh", CLEAN);
    write_script(tmp.path(), "nested/deep.sh", CLEAN);

    let output = run_gate(tmp.path());
    assert!(
        output.status.success(),
        "gate should accept ShellCheck-clean scripts:\n{}",
        combined(&output)
    );
}

#[test]
fn gate_fails_loudly_on_a_lint_violation() {
    if !shellcheck_available() {
        eprintln!("skipping: shellcheck is not installed");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    write_script(tmp.path(), "clean.sh", CLEAN);
    write_script(tmp.path(), "dirty.sh", DIRTY);

    let output = run_gate(tmp.path());
    assert!(
        !output.status.success(),
        "gate must exit non-zero when a script fails ShellCheck:\n{}",
        combined(&output)
    );
    assert!(
        combined(&output).contains("dirty.sh"),
        "gate must name the offending script:\n{}",
        combined(&output)
    );
}

#[test]
fn gate_reports_every_failing_script_not_just_the_first() {
    if !shellcheck_available() {
        eprintln!("skipping: shellcheck is not installed");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    write_script(tmp.path(), "first_dirty.sh", DIRTY);
    write_script(tmp.path(), "second_dirty.sh", DIRTY);

    let output = run_gate(tmp.path());
    let text = combined(&output);
    assert!(!output.status.success(), "gate must fail:\n{text}");
    assert!(
        text.contains("first_dirty.sh") && text.contains("second_dirty.sh"),
        "gate must report all failing scripts, not stop at the first:\n{text}"
    );
}

#[test]
fn gate_scans_scripts_with_spaces_in_their_path() {
    if !shellcheck_available() {
        eprintln!("skipping: shellcheck is not installed");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    write_script(tmp.path(), "a dir/my dirty script.sh", DIRTY);

    let output = run_gate(tmp.path());
    assert!(
        !output.status.success(),
        "gate must not skip paths containing spaces:\n{}",
        combined(&output)
    );
}

#[test]
fn gate_ignores_build_artefacts_and_git_internals() {
    if !shellcheck_available() {
        eprintln!("skipping: shellcheck is not installed");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    write_script(tmp.path(), "real.sh", CLEAN);
    write_script(tmp.path(), "target/debug/build/vendored.sh", DIRTY);
    write_script(tmp.path(), ".git/hooks/sample.sh", DIRTY);

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
    if !shellcheck_available() {
        eprintln!("skipping: shellcheck is not installed");
        return;
    }
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
    if !shellcheck_available() {
        eprintln!("skipping: shellcheck is not installed");
        return;
    }
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
fn gate_fails_loudly_when_shellcheck_is_not_installed() {
    // Run with an empty PATH so `command -v shellcheck` cannot resolve. The gate
    // must fail rather than quietly skipping the lint. bash is invoked by
    // absolute path since PATH lookup is deliberately broken.
    let tmp = tempfile::tempdir().expect("tempdir");
    write_script(tmp.path(), "clean.sh", CLEAN);

    let output = Command::new("/bin/bash")
        .arg(gate_script())
        .arg(tmp.path())
        .env("PATH", "")
        .current_dir(repo_root())
        .output()
        .expect("failed to run quality/shellcheck.sh");

    assert!(
        !output.status.success(),
        "gate must fail when shellcheck is missing rather than reporting success:\n{}",
        combined(&output)
    );
    assert!(
        combined(&output).contains("shellcheck"),
        "gate must explain that shellcheck is missing:\n{}",
        combined(&output)
    );
}

#[test]
fn quality_gate_invokes_the_committed_shellcheck_script() {
    let quality = fs::read_to_string(repo_root().join("quality.sh")).expect("read quality.sh");
    assert!(
        quality.contains("quality/shellcheck.sh"),
        "./quality.sh must invoke the committed ShellCheck gate (Issue #1898)"
    );
}

/// The gate only protects the default branch if CI actually runs it on pull
/// requests. Verify some workflow invokes the committed script.
#[test]
fn a_ci_workflow_invokes_the_shellcheck_gate_on_pull_requests() {
    let wf_dir = repo_root().join(".github/workflows");
    let mut invoking = Vec::new();
    for entry in fs::read_dir(&wf_dir).expect("read workflows dir").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yml") {
            continue;
        }
        let body = fs::read_to_string(&path).expect("read workflow");
        if body.contains("quality/shellcheck.sh") {
            assert!(
                body.contains("pull_request"),
                "{} invokes the ShellCheck gate but is not triggered on pull requests",
                path.display()
            );
            invoking.push(path);
        }
    }
    assert!(
        !invoking.is_empty(),
        "no CI workflow invokes quality/shellcheck.sh — unlinted bash could land on the \
         default branch (Issue #1898)"
    );
}

/// The unmaintained wrapper must not come back — in any workflow, under any pin.
#[test]
fn no_workflow_depends_on_the_orphaned_shellcheck_wrapper() {
    let wf_dir = repo_root().join(".github/workflows");
    for entry in fs::read_dir(&wf_dir).expect("read workflows dir").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yml") {
            continue;
        }
        let body = fs::read_to_string(&path).expect("read workflow");
        for line in body.lines() {
            let trimmed = line.trim_start().trim_start_matches("- ");
            assert!(
                !trimmed.starts_with("uses:") || !trimmed.contains("ludeeus/action-shellcheck"),
                "{} still uses the unmaintained ludeeus/action-shellcheck wrapper (Issue #1898)",
                path.display()
            );
        }
    }
}

/// Whatever actions remain in the `ShellCheck` workflow must still be pinned to a
/// 40-character commit SHA with an auditable trailing version comment
/// (Issue #1215 — preserved across this migration).
#[test]
fn shellcheck_workflow_actions_stay_sha_pinned() {
    let body = load_shellcheck_yml();
    let mut checked = 0;
    for line in body.lines() {
        let Some((_, reference)) = line.split_once("uses:") else {
            continue;
        };
        let reference = reference.trim();
        if reference.starts_with('#') {
            continue;
        }
        let (action, rest) = reference
            .split_once('@')
            .unwrap_or_else(|| panic!("action reference is not pinned at all: {reference}"));
        let (pin, comment) = rest.split_once('#').unwrap_or((rest, ""));
        let pin = pin.trim();
        assert!(
            pin.len() == 40 && pin.chars().all(|c| c.is_ascii_hexdigit()),
            "{action} must be pinned to a 40-character commit SHA, found `{pin}` (Issue #1215)"
        );
        assert!(
            !comment.trim().is_empty(),
            "{action}'s SHA pin lacks a trailing version comment (Issue #1215)"
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "shellcheck.yml declares no actions — expected at least actions/checkout"
    );
}

/// The `ShellCheck` binary itself must come from the actively-maintained upstream
/// (koalaman/shellcheck) at a pinned version, not an unpinned wrapper.
#[test]
fn shellcheck_workflow_installs_a_pinned_shellcheck_version() {
    let body = load_shellcheck_yml();
    let pinned = body
        .lines()
        .filter_map(|l| l.trim().strip_prefix("tool:"))
        .any(|tool| {
            let tool = tool.trim();
            tool.starts_with("shellcheck@") && tool.len() > "shellcheck@".len()
        });
    assert!(
        pinned,
        "shellcheck.yml must install a version-pinned `shellcheck@<version>` so CI lints with a \
         known binary (Issue #1898)"
    );
}
