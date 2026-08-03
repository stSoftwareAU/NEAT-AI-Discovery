//! Issue #1918: the regression threshold must be validated before it reaches `bc`.
//!
//! `benchmark_compare.sh` interpolated `$THRESHOLD` straight into a `bc`
//! expression, so anyone who controlled `BENCHMARK_THRESHOLD` (or `--threshold`)
//! could pass a `bc` expression such as `10^9` and force a "No regressions
//! detected" verdict, or a non-numeric value such as `abc` that made `bc` fail
//! and the comparison silently misbehave.
//!
//! These tests drive the real scripts and assert on exit codes and output. They
//! live in the Rust suite because `tests/benchmark_ci_test.sh` is not run by the
//! standard `cargo test` CI gate, so only a Rust test blocks a regression.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn compare_script() -> PathBuf {
    repo_root().join("benchmark_compare.sh")
}

fn ci_script() -> PathBuf {
    repo_root().join("scripts").join("benchmark-ci.sh")
}

/// Run a script from the repo root with the given args and environment.
fn run(script: &Path, args: &[&str], threshold_env: Option<&str>) -> Output {
    let mut cmd = Command::new("bash");
    cmd.arg(script)
        .args(args)
        .current_dir(repo_root())
        .env_remove("BENCHMARK_THRESHOLD")
        .stdin(Stdio::null());
    if let Some(value) = threshold_env {
        cmd.env("BENCHMARK_THRESHOLD", value);
    }
    cmd.output().expect("run benchmark script")
}

fn combined_output(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Assert the script rejected the threshold loudly and ran no comparison.
fn assert_rejected(output: &Output, value: &str) {
    let text = combined_output(output);
    assert!(
        !output.status.success(),
        "threshold '{value}' must be rejected with a non-zero exit, got success. Output:\n{text}"
    );
    assert!(
        text.contains("Threshold must be"),
        "threshold '{value}' must produce a clear validation message. Output:\n{text}"
    );
    assert!(
        !text.contains("Comparing:"),
        "threshold '{value}' must be rejected before any comparison runs. Output:\n{text}"
    );
}

#[test]
fn environment_threshold_rejects_non_numeric() {
    assert_rejected(&run(&compare_script(), &[], Some("abc")), "abc");
}

#[test]
fn environment_threshold_rejects_bc_expression() {
    // Valid `bc`, silently defeats the gate — must not be accepted.
    assert_rejected(&run(&compare_script(), &[], Some("10^9")), "10^9");
}

#[test]
fn cli_threshold_is_validated_on_the_same_path() {
    for value in ["abc", "10^9", "-5", "5;echo pwned", "1e9", ""] {
        assert_rejected(
            &run(&compare_script(), &["--threshold", value], None),
            value,
        );
    }
}

#[test]
fn cli_threshold_requires_a_value() {
    let output = run(&compare_script(), &["--threshold"], None);
    let text = combined_output(&output);
    assert!(
        !output.status.success(),
        "a bare --threshold must exit non-zero. Output:\n{text}"
    );
    assert!(
        text.contains("requires a value"),
        "a bare --threshold must say it requires a value. Output:\n{text}"
    );
}

#[test]
fn valid_integer_threshold_is_accepted() {
    let output = run(&compare_script(), &["--threshold", "10", "--list"], None);
    let text = combined_output(&output);
    assert!(
        output.status.success(),
        "a valid integer threshold must be accepted. Output:\n{text}"
    );
    assert!(
        text.contains("Available benchmark suites"),
        "listing must still work with a valid threshold. Output:\n{text}"
    );
}

#[test]
fn valid_decimal_threshold_is_accepted() {
    let output = run(&compare_script(), &["--threshold", "2.5", "--list"], None);
    assert!(
        output.status.success(),
        "a fractional threshold must be accepted. Output:\n{}",
        combined_output(&output)
    );
}

#[test]
fn valid_environment_threshold_is_accepted() {
    let output = run(&compare_script(), &["--list"], Some("999999"));
    assert!(
        output.status.success(),
        "a large but numeric threshold is still a number and must be accepted. Output:\n{}",
        combined_output(&output)
    );
}

#[test]
fn help_still_works_with_an_invalid_threshold() {
    let output = run(&compare_script(), &["--help"], Some("abc"));
    let text = combined_output(&output);
    assert!(
        output.status.success(),
        "--help must be handled before threshold validation. Output:\n{text}"
    );
    assert!(
        text.contains("Usage"),
        "--help must print usage. Output:\n{text}"
    );
}

#[test]
fn benchmark_ci_shares_the_same_validation() {
    // The sibling script must reject the same values, so the two cannot drift.
    for value in ["abc", "10^9", "-5"] {
        assert_rejected(&run(&ci_script(), &["--threshold", value], None), value);
    }
    let output = run(&ci_script(), &["--list"], Some("7"));
    assert!(
        output.status.success(),
        "a valid threshold must still be accepted by benchmark-ci.sh. Output:\n{}",
        combined_output(&output)
    );
}
