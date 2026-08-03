//! Issue #1974: `bump_deps::list_manifests` swallowed a failed `cd "$root"`.
//!
//! The `find` fallback ran as `cd "$root" && find … | sed …` with the whole
//! chain's failure mapped to an empty result. An unreadable or missing root
//! therefore reported "no manifests" instead of failing — the quarantine gate
//! would age-check nothing and call that success. The `cd` failure must be
//! loud; only a `find`/`sed` hiccup may degrade to "no manifests".
//!
//! These tests drive the shell helper directly (sourceable via
//! `BUMP_DEPS_SOURCE_ONLY=1`) and assert on real exit codes and output.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Source `bump-deps.sh` in helper-only mode and run `snippet`, returning the
/// raw `Output` so tests can assert on the exit status as well as the streams.
fn run_helper(snippet: &str) -> Output {
    let script = repo_root().join("bump-deps.sh");
    Command::new("bash")
        .arg("-c")
        .arg(format!("source '{}' && {snippet}", script.display()))
        .current_dir(repo_root())
        .env("BUMP_DEPS_SOURCE_ONLY", "1")
        .stdin(Stdio::null())
        .output()
        .expect("run bump-deps.sh helper")
}

// ── A root that cannot be entered must fail loud ──────────────────────

#[test]
fn missing_root_fails_loud_instead_of_reporting_no_manifests() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let missing = tmp.path().join("definitely-not-here");

    let out = run_helper(&format!(
        "bump_deps::list_manifests '{}'",
        missing.to_string_lossy()
    ));

    assert!(
        !out.status.success(),
        "a root that cannot be entered must fail loud, not report an empty \
         manifest set (Issue #1974). stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(&*missing.to_string_lossy()),
        "the failure must name the unreadable root so the operator can see \
         what went wrong (Issue #1974). stderr:\n{stderr}"
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).trim().is_empty(),
        "a failed root must emit no manifests (Issue #1974). stdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
}

// ── A readable root with no manifests is a quiet, successful empty ─────

#[test]
fn readable_root_without_manifests_succeeds_empty() {
    let tmp = tempfile::tempdir().expect("temp dir");
    fs::write(tmp.path().join("README.md"), "no manifests here\n").expect("write file");

    let out = run_helper(&format!(
        "bump_deps::list_manifests '{}'",
        tmp.path().to_string_lossy()
    ));

    assert!(
        out.status.success(),
        "a readable root that simply holds no Cargo.toml must succeed — only \
         an unenterable root fails (Issue #1974). stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).trim().is_empty(),
        "no manifests means no output (Issue #1974). stdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
}

// ── The find fallback still enumerates a non-git tree ──────────────────

#[test]
fn non_git_tree_lists_manifests_and_skips_target() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path();
    fs::write(root.join("Cargo.toml"), "[package]\nname = \"demo\"\n").expect("root manifest");
    fs::create_dir_all(root.join("fuzz")).expect("fuzz dir");
    fs::write(root.join("fuzz/Cargo.toml"), "[package]\nname = \"fuzz\"\n").expect("fuzz manifest");
    fs::create_dir_all(root.join("target/debug")).expect("target dir");
    fs::write(root.join("target/debug/Cargo.toml"), "[package]\n").expect("vendored manifest");

    let out = run_helper(&format!(
        "bump_deps::list_manifests '{}'",
        root.to_string_lossy()
    ));

    assert!(
        out.status.success(),
        "listing a readable non-git tree must succeed (Issue #1974). \
         stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let listed: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    let root_manifest = root.join("Cargo.toml").to_string_lossy().into_owned();
    let fuzz_manifest = root.join("fuzz/Cargo.toml").to_string_lossy().into_owned();
    assert!(
        listed.contains(&root_manifest.as_str()) && listed.contains(&fuzz_manifest.as_str()),
        "the find fallback must still enumerate every manifest in a non-git \
         tree (Issue #1974). Got:\n{stdout}"
    );
    assert!(
        !listed.iter().any(|l| l.contains("/target/")),
        "vendored manifests under target/ must stay out (Issue #1974). \
         Got:\n{stdout}"
    );
}
