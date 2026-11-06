//! Issue #2554: `benchmark_compare.sh` must not turn a failed benchmark run into
//! a success.
//!
//! Every suite under `benches/` exits 0 when no GPU is available (it prints
//! "Skipping ... no GPU available" and returns), so a non-zero exit from
//! `cargo bench` is a genuine failure: a compile error, a panic, a missing
//! `cargo`, or a missing baseline. The script used to treat every non-zero exit
//! as a GPU skip. In save mode a failed suite printed a warning and the script
//! still reported "Baseline ... saved". In comparison mode a failed suite was
//! counted as skipped, its compiler output was discarded, and when every suite
//! failed the script printed "All benchmark suites were skipped" and exited 0.
//!
//! These tests drive the real script in a sandbox with stub `cargo` and `bc` on
//! `PATH`. The stub `cargo` fails or skips suites named in environment variables,
//! and the stub `bc` keeps the result independent of whether the host has `bc`
//! installed. Every file the script reads (`Cargo.toml`, the threshold helper,
//! `target/criterion/`) is a copy inside the temporary directory.

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

/// Stub `bc`: the arithmetic is not under test, and pinning the answer keeps
/// the assertions identical on a host without `bc` installed.
const BC_STUB: &str = "#!/bin/bash\ncat > /dev/null\necho '0'\n";

/// Stub `cargo`, invoked as `cargo bench --bench <suite> -- <args>`, so the
/// suite name is `$3`. Suites named in `STUB_FAIL_SUITES` die the way a
/// non-compiling suite does. Suites named in `STUB_SKIP_SUITES` behave like a
/// GPU-less suite: they print the skip notice and exit 0 with no change line.
/// Every other suite reports one comparison.
const CARGO_STUB: &str = r#"#!/bin/bash
suite="$3"
case " ${STUB_FAIL_SUITES:-} " in
    *" $suite "*)
        echo "error: could not compile $suite" >&2
        exit 101
        ;;
esac
case " ${STUB_SKIP_SUITES:-} " in
    *" $suite "*)
        echo "Skipping $suite benchmarks: no GPU available" >&2
        exit 0
        ;;
esac
echo "$suite  time:   [1 µs 1 µs 1 µs]"
echo "                    change: [-0.1000% +0.0000% +0.1000%] (p = 0.50 > 0.05)"
echo "                    No change in performance detected."
exit 0
"#;

/// `Cargo.toml` with two suites. The script discovers suites by reading
/// `./Cargo.toml` from the working directory.
const CARGO_TOML: &str = "\
[package]
name = \"sandbox\"
version = \"0.0.0\"
edition = \"2021\"

[[bench]]
name = \"suite_a\"

[[bench]]
name = \"suite_b\"
";

/// A prepared sandbox holding a copy of the real `benchmark_compare.sh`, its
/// threshold helper, a two-suite `Cargo.toml`, and a `target/criterion/`
/// directory so comparison mode finds a baseline.
struct Sandbox {
    _tmp: tempfile::TempDir,
    bin: PathBuf,
    work: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let tmp = tempfile::tempdir().expect("temp dir");
        let bin = tmp.path().join("bin");
        let work = tmp.path().join("work");
        fs::create_dir_all(&bin).expect("create stub bin dir");
        fs::create_dir_all(work.join("scripts")).expect("create work/scripts");
        fs::create_dir_all(work.join("target").join("criterion")).expect("create baseline dir");

        fs::copy(
            repo_root().join("benchmark_compare.sh"),
            work.join("benchmark_compare.sh"),
        )
        .expect("copy benchmark_compare.sh into the sandbox");
        fs::copy(
            repo_root().join("scripts").join("benchmark_threshold.sh"),
            work.join("scripts").join("benchmark_threshold.sh"),
        )
        .expect("copy benchmark_threshold.sh into the sandbox");
        fs::write(work.join("Cargo.toml"), CARGO_TOML).expect("write sandbox Cargo.toml");

        write_stub(&bin, "cargo", CARGO_STUB);
        write_stub(&bin, "bc", BC_STUB);

        Self {
            _tmp: tmp,
            bin,
            work,
        }
    }

    /// Run the script with `args`. Suites in `failing` exit non-zero; suites in
    /// `gpu_less` exit 0 without recording a comparison.
    fn run(&self, args: &[&str], failing: &[&str], gpu_less: &[&str]) -> Output {
        Command::new("bash")
            .arg("benchmark_compare.sh")
            .args(args)
            .current_dir(&self.work)
            .env("PATH", format!("{}:/usr/bin:/bin", self.bin.display()))
            .env("STUB_FAIL_SUITES", failing.join(" "))
            .env("STUB_SKIP_SUITES", gpu_less.join(" "))
            // HOME points at the sandbox so the run cannot read or write the
            // developer's own environment.
            .env("HOME", self.work.as_path())
            .stdin(Stdio::null())
            .output()
            .expect("run benchmark_compare.sh")
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn save_baseline_with_a_failing_suite_fails_and_is_not_reported_saved() {
    let output = Sandbox::new().run(&["--save-baseline"], &["suite_b"], &[]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        !output.status.success(),
        "benchmark_compare.sh --save-baseline must exit non-zero when a suite fails — an \
         incomplete baseline must not look complete (Issue #2554)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("error: could not compile"),
        "the failing suite's compiler diagnostic must reach the operator rather than be \
         discarded (Issue #2554) — stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("suite_b failed") && stderr.contains("baseline NOT saved"),
        "the failure must name the suite and say its baseline was not saved (Issue #2554) \
         — stderr:\n{stderr}"
    );
    assert!(
        !stdout.contains("saved for") && !stdout.contains("Baseline 'saved' saved"),
        "no saved-baseline report may be printed when a suite failed (Issue #2554) \
         — stdout:\n{stdout}"
    );
}

#[test]
fn save_baseline_with_every_suite_passing_reports_the_saved_baseline() {
    let output = Sandbox::new().run(&["--save-baseline"], &[], &[]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        output.status.success(),
        "--save-baseline must still succeed when every suite succeeds (Issue #2554)\n\
         stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("Baseline 'saved' saved for 2 suites"),
        "the happy path must still report the saved baseline (Issue #2554) — stdout:\n{stdout}"
    );
}

#[test]
fn save_baseline_with_every_suite_failing_fails() {
    let output = Sandbox::new().run(&["--save-baseline"], &["suite_a", "suite_b"], &[]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        !output.status.success(),
        "--save-baseline must exit non-zero when every suite fails (Issue #2554)\n\
         stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("2 of 2 benchmark suites failed"),
        "the failure count must be reported (Issue #2554) — stderr:\n{stderr}"
    );
    assert!(
        !stdout.contains("saved for"),
        "no saved-baseline report may be printed when every suite failed (Issue #2554) \
         — stdout:\n{stdout}"
    );
}

#[test]
fn compare_with_a_failing_suite_fails_loudly() {
    let output = Sandbox::new().run(&[], &["suite_b"], &[]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        !output.status.success(),
        "benchmark_compare.sh must exit non-zero when a suite fails — a failed run gives \
         no regression verdict (Issue #2554)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("error: could not compile"),
        "the failing suite's compiler diagnostic must reach the operator (Issue #2554) \
         — stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("suite_b failed"),
        "the failure must name the failing suite (Issue #2554) — stderr:\n{stderr}"
    );
    assert!(
        stdout.contains("Failed suites:     1"),
        "the summary must count the failed suite (Issue #2554) — stdout:\n{stdout}"
    );
    assert!(
        !stdout.contains("No regressions detected"),
        "no clean verdict may be printed when a suite failed (Issue #2554) — stdout:\n{stdout}"
    );
}

#[test]
fn compare_with_every_suite_failing_fails() {
    let output = Sandbox::new().run(&[], &["suite_a", "suite_b"], &[]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        !output.status.success(),
        "benchmark_compare.sh must exit non-zero when every suite fails — previously this \
         exited 0 (Issue #2554)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("error: could not compile"),
        "the compiler diagnostics must reach the operator (Issue #2554) — stderr:\n{stderr}"
    );
    assert!(
        !stdout.contains("No regressions detected"),
        "no clean verdict may be printed when every suite failed (Issue #2554) — stdout:\n{stdout}"
    );
}

#[test]
fn compare_with_every_suite_passing_reports_no_regressions() {
    let output = Sandbox::new().run(&[], &[], &[]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        output.status.success(),
        "benchmark_compare.sh must still succeed when every suite passes with no regression \
         (Issue #2554)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("Total comparisons: 2") && stdout.contains("No regressions detected."),
        "the happy path must still report the clean verdict (Issue #2554) — stdout:\n{stdout}"
    );
}

#[test]
fn compare_with_gpu_less_suites_keeps_the_documented_skip() {
    let output = Sandbox::new().run(&[], &[], &["suite_a", "suite_b"]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        output.status.success(),
        "a GPU-less suite (exit 0, no change line) is a documented skip and must not fail \
         the script (Issue #2554)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("Skipped suites:    2") && stdout.contains("Failed suites:     0"),
        "GPU-less suites must be counted as skipped, not failed (Issue #2554) — stdout:\n{stdout}"
    );
    assert!(
        !stdout.contains("No regressions detected"),
        "no clean verdict may be printed when no comparison was recorded (Issue #2554) \
         — stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("No comparisons were recorded; no regression verdict."),
        "the empty run must say that no verdict was given (Issue #2554) — stdout:\n{stdout}"
    );
}
