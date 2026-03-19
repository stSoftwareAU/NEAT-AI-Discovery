//! Integration tests for Issue #576: Benchmark regression tracking with Criterion comparison.
//!
//! These tests verify that the `benchmark_compare.sh` script behaves correctly
//! by running it and checking its outputs — not by inspecting file existence.
//! Converted to behavioural tests as part of Issue #813 audit.

use std::path::Path;
use std::process::Command;

#[test]
fn benchmark_compare_script_has_valid_syntax() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmark_compare.sh");

    // Verify it passes bash syntax check (behavioural: does bash accept this script?)
    let output = Command::new("bash")
        .arg("-n")
        .arg(&script)
        .output()
        .expect("Failed to run bash syntax check");
    assert!(
        output.status.success(),
        "benchmark_compare.sh has syntax errors: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn benchmark_compare_list_discovers_suites() {
    let project_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("bash")
        .arg("benchmark_compare.sh")
        .arg("--list")
        .current_dir(project_dir)
        .output()
        .expect("Failed to run benchmark_compare.sh --list");

    assert!(
        output.status.success(),
        "benchmark_compare.sh --list should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verify key benchmark suites are discovered (spot-check, not exhaustive)
    assert!(
        stdout.contains("synapse_counts"),
        "benchmark_compare.sh --list should include 'synapse_counts'"
    );
    assert!(
        stdout.contains("neuron_interning"),
        "benchmark_compare.sh --list should include 'neuron_interning'"
    );
}

#[test]
fn benchmark_compare_help_shows_usage() {
    let project_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("bash")
        .arg("benchmark_compare.sh")
        .arg("--help")
        .current_dir(project_dir)
        .output()
        .expect("Failed to run benchmark_compare.sh --help");

    assert!(
        output.status.success(),
        "benchmark_compare.sh --help should succeed"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--save-baseline"),
        "Help should document --save-baseline"
    );
    assert!(
        stdout.contains("--threshold"),
        "Help should document --threshold"
    );
}

#[test]
fn benchmark_compare_rejects_unknown_bench() {
    let project_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("bash")
        .arg("benchmark_compare.sh")
        .arg("--bench")
        .arg("nonexistent_benchmark")
        .current_dir(project_dir)
        .output()
        .expect("Failed to run benchmark_compare.sh");

    assert!(
        !output.status.success(),
        "Should fail for unknown benchmark suite"
    );

    let stderr_stdout = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stderr_stdout.contains("Unknown benchmark") || stderr_stdout.contains("nonexistent"),
        "Should report the unknown benchmark name"
    );
}

#[test]
fn benchmark_compare_without_baseline_reports_error() {
    let project_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Running comparison without a saved baseline should fail gracefully
    // (unless a baseline already exists from a previous run)
    let output = Command::new("bash")
        .arg("benchmark_compare.sh")
        .arg("--bench")
        .arg("synapse_counts")
        .current_dir(project_dir)
        .output()
        .expect("Failed to run benchmark_compare.sh");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Either it reports no baseline, or it runs the comparison (if baseline exists)
    // Both are valid outcomes — the script should not crash
    assert!(
        combined.contains("No baseline found")
            || combined.contains("baseline")
            || combined.contains("Comparing")
            || combined.contains("Summary"),
        "Script should handle missing baseline gracefully, got:\n{combined}"
    );
}
