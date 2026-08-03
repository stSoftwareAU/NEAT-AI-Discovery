//! Issue #1910: `bump-deps.sh` must not use fixed, predictable temp paths.
//!
//! Five log paths were hard-coded under the shared temp directory while the
//! same script allocated everything else with `mktemp`. On a shared build
//! host every user resolved them to the same file: two concurrent runs
//! clobbered each other's logs, and two of those logs are read back to decide
//! what the run reports — the audit verdict named whichever crate a stale or
//! foreign file happened to mention. `>` and `tee` also follow symlinks, so a
//! pre-created file at a predictable name captured the redirect.
//!
//! These tests drive the real script with a recording `mktemp` shim on
//! `PATH`, asserting that every temp path is freshly allocated, unique per
//! run, and removed on both the success and the failure exit path.

use std::collections::HashSet;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Write an executable stub script into `dir`.
fn write_stub(dir: &Path, name: &str, body: &str) {
    let path = dir.join(name);
    fs::write(&path, body).expect("write stub");
    let mut perms = fs::metadata(&path).expect("stat stub").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&path, perms).expect("chmod stub");
}

/// A prepared sandbox: a stub `bin` on `PATH` and a log recording every path
/// the script allocated with `mktemp`.
struct Sandbox {
    _tmp: tempfile::TempDir,
    bin: PathBuf,
    home: PathBuf,
    alloc_log: PathBuf,
}

impl Sandbox {
    /// `with_cargo` false simulates a host without cargo, which makes the
    /// script fail after it has already allocated its temp files.
    fn new(with_cargo: bool) -> Self {
        let tmp = tempfile::tempdir().expect("temp dir");
        let bin = tmp.path().join("bin");
        let home = tmp.path().join("home");
        let alloc_log = tmp.path().join("mktemp-allocations.log");
        for dir in [&bin, &home] {
            fs::create_dir_all(dir).expect("create sandbox dir");
        }

        // Recording shim: delegate to the real mktemp and append every
        // allocated path to the log.
        write_stub(
            &bin,
            "mktemp",
            &format!(
                "#!/bin/bash\n\
                 real=/usr/bin/mktemp\n\
                 [[ -x \"$real\" ]] || real=/bin/mktemp\n\
                 out=\"$(\"$real\" \"$@\")\" || exit 1\n\
                 printf '%s\\n' \"$out\" >> '{}'\n\
                 printf '%s\\n' \"$out\"\n",
                alloc_log.display()
            ),
        );
        if with_cargo {
            // `--no-network --dry-run` only probes for cargo; it never runs it.
            write_stub(&bin, "cargo", "#!/bin/bash\nexit 0\n");
        }

        Self {
            _tmp: tmp,
            bin,
            home,
            alloc_log,
        }
    }

    fn run(&self) -> Output {
        Command::new("bash")
            .arg("bump-deps.sh")
            .args(["--dry-run", "--no-network"])
            .current_dir(repo_root())
            .env("PATH", format!("{}:/usr/bin:/bin", self.bin.display()))
            // HOME points at the sandbox so the script does not source the
            // real ~/.cargo/env and prepend the real cargo ahead of the stubs.
            .env("HOME", &self.home)
            .stdin(Stdio::null())
            .output()
            .expect("run bump-deps.sh")
    }

    /// Paths allocated so far, then reset for the next run.
    fn take_allocations(&self) -> Vec<String> {
        let recorded = fs::read_to_string(&self.alloc_log).unwrap_or_default();
        let paths: Vec<String> = recorded.lines().map(str::to_owned).collect();
        fs::write(&self.alloc_log, "").expect("reset allocation log");
        paths
    }
}

/// Allocated paths that still exist — i.e. the cleanup trap missed them.
fn surviving(paths: &[String]) -> Vec<&String> {
    paths
        .iter()
        .filter(|path| Path::new(path).exists())
        .collect()
}

/// The script must allocate every temp path with `mktemp`, so two runs never
/// share one — and must leave nothing behind when it succeeds.
#[test]
fn concurrent_runs_never_share_a_temp_path() {
    let sandbox = Sandbox::new(true);

    let first = sandbox.run();
    assert!(
        first.status.success(),
        "bump-deps.sh --dry-run --no-network failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&first.stdout),
        String::from_utf8_lossy(&first.stderr)
    );
    let run_a = sandbox.take_allocations();

    let second = sandbox.run();
    assert!(second.status.success(), "second run failed");
    let run_b = sandbox.take_allocations();

    // The five formerly-fixed log paths plus the lockfile snapshots and the
    // manifest snapshot directory.
    assert!(
        run_a.len() >= 6,
        "expected every temp path to come from mktemp; recorded only {run_a:?}"
    );

    let set_a: HashSet<&String> = run_a.iter().collect();
    assert_eq!(
        set_a.len(),
        run_a.len(),
        "one run reused a temp path: {run_a:?}"
    );

    let set_b: HashSet<&String> = run_b.iter().collect();
    let shared: Vec<&&String> = set_a.intersection(&set_b).collect();
    assert!(
        shared.is_empty(),
        "two runs shared temp paths — concurrent runs on a shared build host \
         would clobber each other (Issue #1910): {shared:?}"
    );

    let left = surviving(&run_a);
    assert!(
        left.is_empty(),
        "temp files survived a successful run: {left:?}"
    );
}

/// The cleanup trap must also fire when the run aborts part way through.
#[test]
fn temp_files_are_removed_on_the_failure_path() {
    // No cargo stub: the script allocates its temp files, then exits 3.
    let sandbox = Sandbox::new(false);

    let output = sandbox.run();
    assert!(
        !output.status.success(),
        "expected a failure without cargo on PATH"
    );
    let allocated = sandbox.take_allocations();
    assert!(
        !allocated.is_empty(),
        "the run must have allocated temp files before failing"
    );
    let left = surviving(&allocated);
    assert!(
        left.is_empty(),
        "temp files survived a failed run: {left:?}"
    );
}

/// Cheap regression guard: no fixed temp literal may return to the script.
#[test]
fn script_carries_no_fixed_temp_log_path() {
    let script = fs::read_to_string(repo_root().join("bump-deps.sh")).expect("read bump-deps.sh");
    assert!(
        !script.contains("/tmp/bump-deps"),
        "bump-deps.sh must allocate temp paths with mktemp, not fixed /tmp names (Issue #1910)"
    );
}
