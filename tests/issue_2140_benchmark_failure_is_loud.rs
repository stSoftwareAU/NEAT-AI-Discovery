//! Issue #2140: `benchmark.sh` must not time a run that never completed.
//!
//! `run_benchmark` used to invoke its command as `eval "$cmd" > /dev/null 2>&1
//! || true`. Under `set -euo pipefail` that turned a compile error, a panicking
//! test, or a missing `cargo` into a *timing*: the elapsed time of a suite that
//! aborted in two seconds was fed to `calc_improvement` and printed under
//! "Improvement", so the script reported a large speed-up for work that did not
//! happen. Both streams were discarded, so nothing said the suite had failed.
//!
//! These tests drive the real script in a sandbox with stub `cargo`, `git` and
//! `bc` on `PATH`. The stub `git` keeps the script away from any real
//! repository (it checks out a baseline commit and stashes the working tree),
//! and the stub `bc` keeps the result independent of whether the host has `bc`
//! installed.

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

/// Stub `git`: answers the handful of queries `benchmark.sh` makes and makes
/// every mutation a no-op, so the script never touches a real repository.
/// Anything unexpected fails loud rather than silently succeeding.
const GIT_STUB: &str = r#"#!/bin/bash
case "$*" in
    "rev-parse --short HEAD") echo "abc1234" ;;
    "rev-parse --abbrev-ref HEAD") echo "main" ;;
    "status --porcelain") ;;
    checkout*|stash*) ;;
    *) echo "unexpected git invocation: $*" >&2; exit 1 ;;
esac
exit 0
"#;

/// Stub `bc`: the arithmetic is not under test, and pinning the answer keeps
/// the assertions identical on a host without `bc` installed.
const BC_STUB: &str = "#!/bin/bash\ncat > /dev/null\necho '1.5'\n";

/// A prepared sandbox holding a copy of the real `benchmark.sh`.
struct Sandbox {
    _tmp: tempfile::TempDir,
    bin: PathBuf,
    work: PathBuf,
}

impl Sandbox {
    /// `cargo_test_succeeds` false makes the stub `cargo` fail the *test*
    /// sub-command while still succeeding for `build`, which is what a
    /// panicking suite looks like to the script.
    fn new(cargo_test_succeeds: bool) -> Self {
        let tmp = tempfile::tempdir().expect("temp dir");
        let bin = tmp.path().join("bin");
        let work = tmp.path().join("work");
        for dir in [&bin, &work] {
            fs::create_dir_all(dir).expect("create sandbox dir");
        }

        fs::copy(repo_root().join("benchmark.sh"), work.join("benchmark.sh"))
            .expect("copy benchmark.sh into the sandbox");

        write_stub(&bin, "git", GIT_STUB);
        write_stub(&bin, "bc", BC_STUB);
        write_stub(
            &bin,
            "cargo",
            if cargo_test_succeeds {
                "#!/bin/bash\nexit 0\n"
            } else {
                "#!/bin/bash\n\
                 [[ \"${1:-}\" == \"build\" ]] && exit 0\n\
                 echo 'error: test suite panicked' >&2\n\
                 exit 1\n"
            },
        );

        Self {
            _tmp: tmp,
            bin,
            work,
        }
    }

    fn run(&self) -> Output {
        Command::new("bash")
            .arg("benchmark.sh")
            .current_dir(&self.work)
            .env("PATH", format!("{}:/usr/bin:/bin", self.bin.display()))
            // HOME points at the sandbox so the run cannot read or write the
            // developer's own environment.
            .env("HOME", self.work.as_path())
            .stdin(Stdio::null())
            .output()
            .expect("run benchmark.sh")
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn a_failing_benchmark_command_fails_the_script_loudly() {
    let output = Sandbox::new(false).run();
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        !output.status.success(),
        "benchmark.sh must exit non-zero when a benchmarked command fails — a \
         run that never completed cannot be timed (Issue #2140)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("Unit tests"),
        "the failure must name the benchmark step that failed (Issue #2140) — stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("cargo test"),
        "the failure must name the command that failed (Issue #2140) — stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("error: test suite panicked"),
        "the failing command's own diagnostics must reach the operator rather than \
         being discarded (Issue #2140) — stderr:\n{stderr}"
    );
    assert!(
        !stdout.contains("Improvement"),
        "no improvement figure may be printed for a run that did not complete \
         (Issue #2140) — stdout:\n{stdout}"
    );
    assert!(
        !stdout.contains("Benchmark complete"),
        "the script must not report completion after a failed step (Issue #2140) — stdout:\n{stdout}"
    );
}

#[test]
fn a_successful_run_still_prints_the_summary() {
    let output = Sandbox::new(true).run();
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        output.status.success(),
        "benchmark.sh must still succeed when every benchmarked command \
         succeeds (Issue #2140)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("PERFORMANCE SUMMARY") && stdout.contains("Improvement"),
        "the happy path must still print the summary (Issue #2140) — stdout:\n{stdout}"
    );
    assert!(
        stderr.contains("Duration:"),
        "the happy path must still report each step's duration (Issue #2140) — stderr:\n{stderr}"
    );
}

/// A one-line guard against the construct returning: `eval` as a command word
/// on a live (non-comment) line. Prose mentioning `eval` in a comment is not a
/// match, and `eval\t"$cmd"` is.
#[test]
fn benchmark_script_uses_no_eval() {
    let script = fs::read_to_string(repo_root().join("benchmark.sh")).expect("read benchmark.sh");
    let offending: Vec<&str> = script
        .lines()
        .map(|line| line.split('#').next().unwrap_or(""))
        .filter(|code| {
            code.split(|c: char| c.is_whitespace() || c == ';' || c == '|' || c == '&')
                .any(|word| word == "eval")
        })
        .collect();
    assert!(
        offending.is_empty(),
        "benchmark.sh must invoke its commands as `\"$@\"`, never re-parse them \
         through `eval` (Issue #2140) — found:\n{}",
        offending.join("\n")
    );
}
