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

/// Stub `bc` with real numeric comparisons, keeping the tests independent of
/// whether the host has `bc` installed.
const BC_STUB: &str = r#"#!/bin/bash
expression=$(cat)
if [[ "$expression" =~ ^(-?[0-9]+([.][0-9]+)?)\ \>\ (-?[0-9]+([.][0-9]+)?)$ ]]; then
    awk -v left="${BASH_REMATCH[1]}" -v right="${BASH_REMATCH[3]}" 'BEGIN { print (left > right) ? 1 : 0 }'
elif [[ "$expression" =~ ^(-?[0-9]+([.][0-9]+)?)\ \<\ (-?[0-9]+([.][0-9]+)?)$ ]]; then
    awk -v left="${BASH_REMATCH[1]}" -v right="${BASH_REMATCH[3]}" 'BEGIN { print (left < right) ? 1 : 0 }'
else
    echo "unsupported bc expression: $expression" >&2
    exit 1
fi
"#;

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
case " ${STUB_REGRESS_SUITES:-} " in
    *" $suite "*)
        if [[ " ${STUB_WRAPPED_SUITES:-} " == *" $suite "* ]]; then
            echo "synapse_counts/old_O(n×m)/100n_500s"
            echo "                        time:   [1.000 µs 1.100 µs 1.200 µs]"
            echo "                        change: [+10.0000% +12.5000% +14.0000%] (p = 0.01 < 0.05)"
        elif [[ " ${STUB_MULTI_REGRESS_SUITES:-} " == *" $suite "* ]]; then
            echo "suite_a::sample[one]  time:   [1.000 µs 1.100 µs 1.200 µs]"
            echo "                    change: [+10.0000% +12.5000% +14.0000%] (p = 0.01 < 0.05)"
            echo "                    Performance has regressed."
            echo "suite_a::sample_two  time:   [2.000 µs 2.100 µs 2.200 µs]"
            echo "                    change: [+15.0000% +16.5000% +18.0000%] (p = 0.01 < 0.05)"
            echo "                    Performance has regressed."
        elif [[ " ${STUB_MISSING_NAME_SUITES:-} " == *" $suite "* ]]; then
            echo "                    change: [+10.0000% +12.5000% +14.0000%] (p = 0.01 < 0.05)"
            echo "                    Performance has regressed."
        else
            echo "$suite::sample  time:   [1.000 µs 1.100 µs 1.200 µs]"
            echo "                    change: [+10.0000% +12.5000% +14.0000%] (p = 0.01 < 0.05)"
            echo "                    Performance has regressed."
        fi
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
        self.run_with_regressions(args, failing, gpu_less, &[], &[], &[])
    }

    fn run_with_regressions(
        &self,
        args: &[&str],
        failing: &[&str],
        gpu_less: &[&str],
        regressing: &[&str],
        multiple: &[&str],
        missing_name: &[&str],
    ) -> Output {
        Command::new("bash")
            .arg("benchmark_compare.sh")
            .args(args)
            .current_dir(&self.work)
            .env("PATH", format!("{}:/usr/bin:/bin", self.bin.display()))
            .env("STUB_FAIL_SUITES", failing.join(" "))
            .env("STUB_WRAPPED_SUITES", if args.contains(&"--bench") { "suite_a" } else { "" })
            .env("STUB_SKIP_SUITES", gpu_less.join(" "))
            .env("STUB_REGRESS_SUITES", regressing.join(" "))
            .env("STUB_MULTI_REGRESS_SUITES", multiple.join(" "))
            .env("STUB_MISSING_NAME_SUITES", missing_name.join(" "))
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
fn save_baseline_counts_gpu_less_suite_as_skipped() {
    let output = Sandbox::new().run(&["--save-baseline"], &[], &["suite_b"]);
    let stdout = text(&output.stdout);
    assert!(
        output.status.success(),
        "mixed save should succeed: {stdout}"
    );
    assert!(stdout.contains("saved for 1 suites"), "{stdout}");
    assert!(stdout.contains("Skipped suites: 1"), "{stdout}");
}

#[test]
fn save_baseline_with_every_suite_skipped_reports_no_baseline() {
    let output = Sandbox::new().run(&["--save-baseline"], &[], &["suite_a", "suite_b"]);
    let stdout = text(&output.stdout);
    assert!(
        output.status.success(),
        "skipped save should succeed: {stdout}"
    );
    assert!(stdout.contains("Skipped suites: 2"), "{stdout}");
    assert!(stdout.contains("No baseline was recorded"), "{stdout}");
    assert!(!stdout.contains("saved for"), "{stdout}");
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
fn comparison_reports_multiple_regressions_and_continues_to_later_suites() {
    let output =
        Sandbox::new().run_with_regressions(&[], &[], &[], &["suite_a"], &["suite_a"], &[]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert_eq!(
        output.status.code(),
        Some(1),
        "regressions must fail the comparison after reporting all suites\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("REGRESSION: suite_a::sample[one] (+14.0000%)")
            && stdout.contains("REGRESSION: suite_a::sample_two (+18.0000%)"),
        "each regression must identify its Criterion benchmark and measured change\nstdout:\n{stdout}"
    );
    assert!(
        stdout.contains("Comparing: suite_b"),
        "the suite after a regression must still run\nstdout:\n{stdout}"
    );
    assert!(
        stdout.contains("Total comparisons: 3")
            && stdout.contains("Regressions:       2 (exceeding 5% threshold)")
            && stdout.contains("Improvements:      0")
            && stdout.contains("Within threshold:  1")
            && stdout.contains("Skipped suites:    0")
            && stdout.contains("Failed suites:     0"),
        "summary counts must include the regressing and later passing suites\nstdout:\n{stdout}"
    );
}

#[test]
fn comparison_reports_wrapped_criterion_benchmark_id() {
    let output = Sandbox::new().run_with_regressions(
        &["--bench", "suite_a"],
        &[],
        &[],
        &["suite_a"],
        &[],
        &[],
    );
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(1), "{stdout}");
    assert!(
        stdout.contains("REGRESSION: synapse_counts/old_O(n×m)/100n_500s (+14.0000%)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("REGRESSION: synapse_counts/old_O(n×m)/100n_500s changed by +14.0000%"),
        "{stdout}"
    );
}

#[test]
fn comparison_uses_suite_name_when_criterion_name_is_missing() {
    let output =
        Sandbox::new().run_with_regressions(&[], &[], &[], &["suite_a"], &[], &["suite_a"]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert_eq!(
        output.status.code(),
        Some(1),
        "a missing benchmark name must fall back without aborting\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("REGRESSION: suite_a (+14.0000%)")
            && stdout.contains("Total comparisons: 2")
            && stdout.contains("Regressions:       1 (exceeding 5% threshold)"),
        "the fallback must preserve regression reporting and the final summary\nstdout:\n{stdout}"
    );
}

#[test]
fn comparison_reports_regression_and_later_suite_failure() {
    let output =
        Sandbox::new().run_with_regressions(&[], &["suite_b"], &[], &["suite_a"], &[], &[]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert_eq!(
        output.status.code(),
        Some(1),
        "a regression and later suite failure must both be reported\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("REGRESSION: suite_a::sample (+14.0000%)")
            && stdout.contains("Comparing: suite_b")
            && stdout.contains("Total comparisons: 1")
            && stdout.contains("Regressions:       1 (exceeding 5% threshold)")
            && stdout.contains("Failed suites:     1")
            && stderr.contains("suite_b failed"),
        "the regression summary must include the later suite failure\nstdout:\n{stdout}\nstderr:\n{stderr}"
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
