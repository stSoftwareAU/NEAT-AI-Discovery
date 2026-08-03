//! Issue #1974: `bump_deps::list_manifests` swallowed a failed `cd "$root"`.
//!
//! The `find` fallback ran as `found="$(cd "$root" && find …)" || found=""`, so
//! an unreadable or missing root collapsed to "no manifests" and the helper
//! still returned success — the quarantine gate would then age-check nothing
//! and report a clean run. A `cd` failure must be loud (non-zero exit plus a
//! diagnostic on stderr); only a `find`/`sed` hiccup is tolerated.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Source `bump-deps.sh` in helper-only mode and run `snippet`, returning the
/// raw `Output` so tests can assert on the exit status as well as the streams.
fn run_helper_raw(snippet: &str) -> Output {
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

#[test]
fn missing_root_fails_loudly_instead_of_reporting_no_manifests() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let missing = tmp.path().join("does-not-exist");

    let out = run_helper_raw(&format!(
        "bump_deps::list_manifests '{}'",
        missing.display()
    ));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        !out.status.success(),
        "list_manifests must fail loudly when the root cannot be entered \
         (Issue #1974) — got exit {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code()
    );
    assert!(
        stdout.trim().is_empty(),
        "a root that cannot be entered must not emit manifests (Issue #1974) — \
         got:\n{stdout}"
    );
    assert!(
        stderr.contains(&*missing.to_string_lossy()),
        "the failure must name the root that could not be entered \
         (Issue #1974) — got stderr:\n{stderr}"
    );
}

#[cfg(unix)]
#[test]
fn unreadable_root_fails_loudly() {
    use std::os::unix::fs::PermissionsExt;

    // root ignores the permission bits, so `cd` would still succeed there.
    let euid = Command::new("id")
        .arg("-u")
        .stdin(Stdio::null())
        .output()
        .expect("run id -u");
    if String::from_utf8_lossy(&euid.stdout).trim() == "0" {
        eprintln!("skipping: running as root, permission bits are not enforced");
        return;
    }

    let tmp = tempfile::tempdir().expect("temp dir");
    let locked = tmp.path().join("locked");
    fs::create_dir(&locked).expect("create locked dir");
    fs::write(locked.join("Cargo.toml"), "[package]\nname = \"demo\"\n").expect("write manifest");
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).expect("lock dir");

    let out = run_helper_raw(&format!("bump_deps::list_manifests '{}'", locked.display()));

    // Restore before asserting so the temp dir can always be cleaned up.
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).expect("unlock dir");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "an unreadable root must fail loudly, not age-check nothing and pass \
         (Issue #1974) — got exit {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code()
    );
    assert!(
        stdout.trim().is_empty(),
        "an unreadable root must not emit manifests (Issue #1974) — got:\n{stdout}"
    );
}

#[test]
fn find_fallback_lists_manifests_outside_a_git_tree() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path();
    fs::write(root.join("Cargo.toml"), "[package]\nname = \"demo\"\n").expect("root manifest");
    fs::create_dir(root.join("fuzz")).expect("create fuzz dir");
    fs::write(
        root.join("fuzz/Cargo.toml"),
        "[package]\nname = \"demo-fuzz\"\n",
    )
    .expect("fuzz manifest");
    // Build output must stay out of the gated set.
    fs::create_dir(root.join("target")).expect("create target dir");
    fs::write(
        root.join("target/Cargo.toml"),
        "[package]\nname = \"vendored\"\n",
    )
    .expect("target manifest");

    let out = run_helper_raw(&format!("bump_deps::list_manifests '{}'", root.display()));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "a readable non-git root must still enumerate manifests \
         (Issue #1974)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    let listed: Vec<String> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim_start_matches(&*root.to_string_lossy())
                .trim_start_matches('/')
                .to_string()
        })
        .collect();
    assert!(
        listed.iter().any(|m| m == "Cargo.toml") && listed.iter().any(|m| m == "fuzz/Cargo.toml"),
        "the find fallback must list every manifest outside target/ \
         (Issue #1974) — got {listed:?}"
    );
    assert!(
        !listed.iter().any(|m| m.starts_with("target/")),
        "manifests under target/ must stay out of the gated set \
         (Issue #1974) — got {listed:?}"
    );
}

#[test]
fn empty_readable_root_is_not_an_error() {
    let tmp = tempfile::tempdir().expect("temp dir");

    let out = run_helper_raw(&format!(
        "bump_deps::list_manifests '{}'",
        tmp.path().display()
    ));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "a readable root with no manifests is a legitimate empty result, not a \
         failure (Issue #1974)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.trim().is_empty(),
        "an empty root must emit nothing (Issue #1974) — got:\n{stdout}"
    );
}
