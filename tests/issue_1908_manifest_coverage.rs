//! Issue #1908: the quarantine gate in `bump-deps.sh` age-checked only the
//! `[dependencies]` and `[dev-dependencies]` tables of the root `Cargo.toml`.
//!
//! Everything else was bumped without a publish-age check:
//!
//!   * `[build-dependencies]` — the highest-value target, because `build.rs`
//!     executes at compile time with the developer's/CI's privileges.
//!   * `[target.'cfg(unix)'.dependencies]` — live in this repo since #1904.
//!   * `[dependencies.<name>]` sub-tables — live in `fuzz/Cargo.toml`.
//!   * `fuzz/Cargo.toml` itself, which was never passed to the gate.
//!
//! These tests drive the shell helpers directly (they are sourceable via
//! `BUMP_DEPS_SOURCE_ONLY=1`) and assert on real behaviour: what
//! `extract_dep_versions` returns, which manifests `list_manifests`
//! enumerates, and what `apply_manifest_quarantine` leaves on disk.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Source `bump-deps.sh` in helper-only mode and run `snippet`, returning
/// stdout. Panics with both streams when the snippet exits non-zero.
fn run_helper(snippet: &str, extra_env: &[(&str, &str)]) -> String {
    let script = repo_root().join("bump-deps.sh");
    let mut command = Command::new("bash");
    command
        .arg("-c")
        .arg(format!("source '{}' && {snippet}", script.display()))
        .current_dir(repo_root())
        .env("BUMP_DEPS_SOURCE_ONLY", "1")
        .stdin(Stdio::null());
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let output = command.output().expect("run bump-deps.sh helper");
    assert!(
        output.status.success(),
        "helper snippet failed: {snippet}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Write a crates.io publish-time fixture for `name@version`, `age_hours`
/// before the stubbed "now" of 2025-06-01T00:00:00Z.
fn write_publish_fixture(dir: &Path, name: &str, version: &str, age_hours: i64) {
    const NOW_EPOCH: i64 = 1_748_736_000; // 2025-06-01T00:00:00Z
    let published = NOW_EPOCH - age_hours * 3600;
    // Format the epoch as ISO-8601 without pulling in a date crate: the
    // fixture only needs to round-trip through the script's `date` parsing.
    let iso = Command::new("date")
        .args(["-u", "-r", &published.to_string(), "+%Y-%m-%dT%H:%M:%S"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .or_else(|| {
            Command::new("date")
                .args(["-u", "-d", &format!("@{published}"), "+%Y-%m-%dT%H:%M:%S"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        })
        .expect("format publish timestamp with date(1)");
    fs::write(
        dir.join(format!("{name}-{version}.json")),
        format!(r#"{{"version":{{"num":"{version}","created_at":"{iso}.000000+00:00"}}}}"#),
    )
    .expect("write publish fixture");
}

/// "now" in epoch seconds — the unit the quarantine helpers take (Issue #1909).
const NOW_EPOCH: i64 = 1_748_736_000;

// ── Acceptance criterion 1: every dependency table is parsed ──────────

#[test]
fn extract_covers_all_dependency_tables() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let manifest = tmp.path().join("Cargo.toml");
    fs::write(
        &manifest,
        r#"[package]
name = "demo"

[dependencies]
serde = "1.0.225"
arrow = { version = "59.1", default-features = false }

[dev-dependencies]
tempfile = "3.27"

[build-dependencies]
cc = "1.2.0"

[target.'cfg(unix)'.dependencies]
libc = "0.2.180"

[target.'cfg(windows)'.build-dependencies]
winres = "0.1.12"

[dependencies.foo]
version = "4.5.6"
features = ["full"]

[dev-dependencies.mockito]
version = "1.7.0"
"#,
    )
    .expect("write fixture manifest");

    let out = run_helper(
        &format!(
            "bump_deps::extract_dep_versions '{}'",
            manifest.path_display()
        ),
        &[],
    );
    let entries: Vec<&str> = out.lines().filter(|l| !l.trim().is_empty()).collect();

    for expected in [
        "serde\t1.0.225",
        "arrow\t59.1",
        "tempfile\t3.27",
        "cc\t1.2.0",
        "libc\t0.2.180",
        "winres\t0.1.12",
        "foo\t4.5.6",
        "mockito\t1.7.0",
    ] {
        assert!(
            entries.contains(&expected),
            "extract_dep_versions must cover every dependency table — missing \
             `{}` (Issue #1908). Got:\n{out}",
            expected.replace('\t', " => ")
        );
    }
    assert_eq!(
        entries.len(),
        8,
        "extract_dep_versions must emit exactly one entry per declared \
         dependency (Issue #1908). Got:\n{out}"
    );
}

// ── Acceptance criterion 2: every tracked manifest is scanned ─────────

#[test]
fn gate_scans_all_tracked_manifests() {
    let out = run_helper(
        &format!(
            "bump_deps::list_manifests '{}'",
            repo_root().to_string_lossy()
        ),
        &[],
    );
    let root = repo_root();
    let mut scanned: Vec<String> = out
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim_start_matches(&*root.to_string_lossy())
                .trim_start_matches('/')
                .to_string()
        })
        .collect();
    scanned.sort();

    // The tracked set, straight from git — the gate must not fall behind it.
    let tracked = Command::new("git")
        .args(["ls-files", "--", "Cargo.toml", "*/Cargo.toml"])
        .current_dir(&root)
        .stdin(Stdio::null())
        .output()
        .expect("git ls-files");
    let mut expected: Vec<String> = String::from_utf8_lossy(&tracked.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_string)
        .collect();
    expected.sort();

    assert_eq!(
        scanned, expected,
        "the quarantine gate must scan every tracked Cargo.toml, not just the \
         root manifest (Issue #1908)"
    );
    assert!(
        scanned.iter().any(|m| m == "fuzz/Cargo.toml"),
        "fuzz/Cargo.toml must be inside the gated manifest set (Issue #1908) — \
         got {scanned:?}"
    );
}

// ── Acceptance criterion 3: a build-dep bump inside the window reverts ─

#[test]
fn build_dependency_bump_inside_window_is_reverted() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let manifest = tmp.path().join("Cargo.toml");
    // Post-upgrade state: cc (build-dep) and serde (normal dep) both bumped.
    fs::write(
        &manifest,
        r#"[package]
name = "demo"

[dependencies]
serde = "1.0.225"

[build-dependencies]
cc = "1.2.0"
"#,
    )
    .expect("write manifest");

    let before = tmp.path().join("before.tsv");
    let after = tmp.path().join("after.tsv");
    fs::write(&before, "serde\t1.0.190\ncc\t1.1.0\n").expect("write before");
    fs::write(&after, "serde\t1.0.225\ncc\t1.2.0\n").expect("write after");

    let fixtures = tmp.path().join("fixtures");
    fs::create_dir_all(&fixtures).expect("fixture dir");
    write_publish_fixture(&fixtures, "serde", "1.0.225", 300); // outside window
    write_publish_fixture(&fixtures, "cc", "1.2.0", 1); // inside window

    let verdicts = run_helper(
        &format!(
            "bump_deps::apply_manifest_quarantine '{}' '{}' '{}' {NOW_EPOCH} 24",
            manifest.path_display(),
            before.path_display(),
            after.path_display()
        ),
        &[("BUMP_DEPS_TEST_FIXTURE", &fixtures.to_string_lossy())],
    );

    assert!(
        verdicts
            .lines()
            .any(|l| l == "revert\tcc\t1.1.0\t1.2.0\t1h"),
        "a [build-dependencies] bump published inside the window must be \
         reverted, exactly as a [dependencies] bump is (Issue #1908). \
         Verdicts:\n{verdicts}"
    );
    assert!(
        verdicts.lines().any(|l| l.starts_with("keep\tserde\t")),
        "a bump published outside the window must be kept (Issue #1908). \
         Verdicts:\n{verdicts}"
    );

    let rewritten = fs::read_to_string(&manifest).expect("read manifest");
    assert!(
        rewritten.contains(r#"cc = "1.1.0""#),
        "the in-quarantine build-dependency must be pinned back on disk \
         (Issue #1908):\n{rewritten}"
    );
    assert!(
        rewritten.contains(r#"serde = "1.0.225""#),
        "the aged dependency must keep its bump (Issue #1908):\n{rewritten}"
    );
}

#[test]
fn target_dependency_bump_inside_window_is_reverted() {
    // This repo declares libc under [target.'cfg(unix)'.dependencies]
    // (Issue #1904), so the target table is a live gap, not a hypothetical.
    let tmp = tempfile::tempdir().expect("temp dir");
    let manifest = tmp.path().join("Cargo.toml");
    fs::write(
        &manifest,
        r#"[package]
name = "demo"

[target.'cfg(unix)'.dependencies]
libc = "0.2.180"  # geteuid/umask

[build-dependencies.cc]
version = "1.2.0"
"#,
    )
    .expect("write manifest");

    let before = tmp.path().join("before.tsv");
    let after = tmp.path().join("after.tsv");
    fs::write(&before, "libc\t0.2.179\ncc\t1.1.0\n").expect("write before");
    fs::write(&after, "libc\t0.2.180\ncc\t1.2.0\n").expect("write after");

    let fixtures = tmp.path().join("fixtures");
    fs::create_dir_all(&fixtures).expect("fixture dir");
    write_publish_fixture(&fixtures, "libc", "0.2.180", 2); // inside window
    // No fixture for cc@1.2.0: an undatable version fails closed.

    let verdicts = run_helper(
        &format!(
            "bump_deps::apply_manifest_quarantine '{}' '{}' '{}' {NOW_EPOCH} 24",
            manifest.path_display(),
            before.path_display(),
            after.path_display()
        ),
        &[("BUMP_DEPS_TEST_FIXTURE", &fixtures.to_string_lossy())],
    );

    assert!(
        verdicts
            .lines()
            .any(|l| l == "revert\tlibc\t0.2.179\t0.2.180\t2h"),
        "a [target.'cfg(unix)'.dependencies] bump inside the window must be \
         reverted (Issue #1908). Verdicts:\n{verdicts}"
    );
    assert!(
        verdicts
            .lines()
            .any(|l| l == "revert\tcc\t1.1.0\t1.2.0\tunknown"),
        "an undatable sub-table bump must fail closed and revert (Issue \
         #1908). Verdicts:\n{verdicts}"
    );

    let rewritten = fs::read_to_string(&manifest).expect("read manifest");
    assert!(
        rewritten.contains(r#"libc = "0.2.179""#),
        "the target-table dependency must be pinned back (Issue \
         #1908):\n{rewritten}"
    );
    assert!(
        rewritten.contains(r#"version = "1.1.0""#),
        "the sub-table dependency must be pinned back (Issue \
         #1908):\n{rewritten}"
    );
    assert!(
        rewritten.contains("# geteuid/umask"),
        "reverting must preserve the rest of the line, comments included \
         (Issue #1908):\n{rewritten}"
    );
}

/// Small convenience so the format! calls above stay readable.
trait PathDisplay {
    fn path_display(&self) -> String;
}

impl PathDisplay for PathBuf {
    fn path_display(&self) -> String {
        self.to_string_lossy().into_owned()
    }
}
