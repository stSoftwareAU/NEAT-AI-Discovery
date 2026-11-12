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
case " ${STUB_NOISY_SUITES:-} " in
    *" $suite "*)
        echo "$suite::noisy_regress  time:   [1.000 µs 1.100 µs 1.200 µs]"
        echo "                    change: [−1.2000% +2.0000% +6.0000%] (p = 0.30 > 0.05)"
        echo "                    No change in performance detected."
        echo "$suite::real_improvement  time:   [1.000 µs 1.100 µs 1.200 µs]"
        echo "                    change: [−9.0000% −6.5000% −1.0000%] (p = 0.01 < 0.05)"
        echo "                    Performance has improved."
        exit 0
        ;;
esac
case " ${STUB_STRADDLE_SUITES:-} " in
    *" $suite "*)
        echo "$suite::straddle  time:   [1.000 µs 1.100 µs 1.200 µs]"
        echo "                    change: [−0.5000% +7.0000% +14.000%] (p = 0.07 > 0.05)"
        echo "                    Change within noise threshold."
        exit 0
        ;;
esac
case " ${STUB_LARGE_SUITES:-} " in
    *" $suite "*)
        echo "$suite::large  time:   [1.000 µs 1.100 µs 1.200 µs]"
        echo "                    change: [+9000.0000% +12000% +15000%] (p = 0.00 < 0.05)"
        echo "                    Performance has regressed."
        exit 0
        ;;
esac
case " ${STUB_MALFORMED_SUITES:-} " in
    *" $suite "*)
        echo "$suite::malformed  time:   [1.000 µs 1.100 µs 1.200 µs]"
        echo "                    change: [garbage] (p = 0.07 > 0.05)"
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
echo "                    change: [−0.1000% +0.0000% +0.1000%] (p = 0.50 > 0.05)"
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
        self.run_stubbed(
            args,
            failing,
            gpu_less,
            regressing,
            multiple,
            missing_name,
            &[],
        )
    }

    /// Run the script with `suite_a` reporting a noisy change (upper bound above
    /// the threshold, point estimate within it) and a real improvement (point
    /// estimate below the negative threshold). All other stub suites are unset.
    fn run_noisy(&self) -> Output {
        self.run_stubbed(&[], &[], &[], &[], &[], &[], &["suite_a"])
    }

    /// Run the script with the stub variable `key` (for example
    /// `STUB_STRADDLE_SUITES`) naming `suite_a`; all other stub suites are unset.
    fn run_with_stub(&self, key: &str) -> Output {
        Command::new("bash")
            .arg("benchmark_compare.sh")
            .current_dir(&self.work)
            .env("PATH", format!("{}:/usr/bin:/bin", self.bin.display()))
            .env(key, "suite_a")
            .env("HOME", self.work.as_path())
            .stdin(Stdio::null())
            .output()
            .expect("run benchmark_compare.sh")
    }

    #[allow(clippy::too_many_arguments)]
    fn run_stubbed(
        &self,
        args: &[&str],
        failing: &[&str],
        gpu_less: &[&str],
        regressing: &[&str],
        multiple: &[&str],
        missing_name: &[&str],
        noisy: &[&str],
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
            .env("STUB_NOISY_SUITES", noisy.join(" "))
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
        stdout.contains("REGRESSION: suite_a::sample[one] (+12.5000%)")
            && stdout.contains("REGRESSION: suite_a::sample_two (+16.5000%)"),
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
        stdout.contains("REGRESSION: synapse_counts/old_O(n×m)/100n_500s (+12.5000%)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("REGRESSION: synapse_counts/old_O(n×m)/100n_500s changed by +12.5000%"),
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
        stdout.contains("REGRESSION: suite_a (+12.5000%)")
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
        stdout.contains("REGRESSION: suite_a::sample (+12.5000%)")
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

#[test]
fn comparison_uses_point_estimate_not_upper_bound() {
    // suite_a's noisy_regress has change [−1.2000% +2.0000% +6.0000%]: the upper
    // bound (6.0) exceeds the 5% threshold but the point estimate (2.0) does not,
    // so it must not be reported as a regression (Issue #2584). real_improvement
    // has change [−9.0000% −6.5000% −1.0000%] (Criterion 0.8 signs negative values
    // with U+2212, not ASCII '-'): the point estimate (-6.5) is past
    // the negative threshold, so it must be counted as an improvement.
    let output = Sandbox::new().run_noisy();
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        output.status.success(),
        "a noisy upper bound above the threshold must not fail the comparison when the \
         point estimate is within it (Issue #2584)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("No regressions detected."),
        "the point estimate is within threshold, so the clean verdict must be printed \
         (Issue #2584) — stdout:\n{stdout}"
    );
    assert!(
        !stdout.contains("REGRESSION:"),
        "the upper bound must not be reported as a regression (Issue #2584) — stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("Regressions:       0"),
        "the summary must count no regressions (Issue #2584) — stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("Improvements:      1"),
        "the point estimate past the negative threshold must count as an improvement \
         (Issue #2584) — stdout:\n{stdout}"
    );
}

#[test]
fn negative_lower_bound_with_regressing_point_estimate_is_a_regression() {
    // Criterion 0.8 prints `[−0.5000% +7.0000% +14.000%]` with U+2212. The point
    // estimate (+7) is above the 5% threshold, so it must be counted and fail
    // the run rather than be dropped as an unreadable line (Issue #2584).
    let output = Sandbox::new().run_with_stub("STUB_STRADDLE_SUITES");
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        !output.status.success(),
        "a +7% point estimate must fail the comparison (Issue #2584)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("REGRESSION: suite_a::straddle (+7.0000%)"),
        "the regression must report the point estimate (Issue #2584) — stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("Total comparisons: 2"),
        "the U+2212 line must be counted, not silently dropped (Issue #2584) — stdout:\n{stdout}"
    );
    assert!(
        !stdout.contains("Skipped (no comparison recorded"),
        "a readable change line must not make the suite look GPU-skipped (Issue #2584) — stdout:\n{stdout}"
    );
}

#[test]
fn change_values_of_ten_thousand_percent_without_fraction_are_parsed() {
    // Criterion prints values of 10000% or more with no decimal point.
    let output = Sandbox::new().run_with_stub("STUB_LARGE_SUITES");
    let stdout = text(&output.stdout);

    assert!(
        !output.status.success(),
        "a +12000% point estimate must fail the comparison (Issue #2584) — stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("REGRESSION: suite_a::large (+12000%)"),
        "a fraction-less point estimate must be parsed (Issue #2584) — stdout:\n{stdout}"
    );
}

#[test]
fn unreadable_change_line_fails_loudly() {
    let output = Sandbox::new().run_with_stub("STUB_MALFORMED_SUITES");
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        !output.status.success(),
        "a change line the parser cannot read must not exit 0 (Issue #2584)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("unparsable Criterion change line") && stderr.contains("suite_a"),
        "the error must name the suite (Issue #2584) — stderr:\n{stderr}"
    );
    assert!(
        !stdout.contains("Skipped (no comparison recorded"),
        "an unreadable change line must not be reported as a GPU skip (Issue #2584) — stdout:\n{stdout}"
    );
}
