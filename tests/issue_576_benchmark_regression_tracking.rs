//! Integration tests for Issue #576: Benchmark regression tracking with Criterion comparison.
//!
//! These tests verify that:
//! - All benchmark suites are correctly declared in Cargo.toml
//! - The benchmark_compare.sh script discovers benchmarks correctly
//! - The script handles argument parsing and modes properly

use std::path::Path;
use std::process::Command;

/// All expected benchmark suite names from Cargo.toml.
const EXPECTED_BENCHMARKS: &[&str] = &[
    "synapse_counts",
    "neuron_interning",
    "batched_activation",
    "cache_locality",
    "zero_copy_buffer",
    "sample_locality",
    "tiered_loading",
    "parallel_discovery",
    "memory_streaming",
    "clone_reduction",
    "upsert_candidate",
    "gpu_buffer_transfers",
    "async_pipeline",
    "gpu_shader_workgroup",
    "uuid_hashing",
];

#[test]
fn benchmark_suites_have_source_files() {
    let benches_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("benches");
    for name in EXPECTED_BENCHMARKS {
        let source_file = benches_dir.join(format!("{name}.rs"));
        assert!(
            source_file.exists(),
            "Missing benchmark source file: benches/{name}.rs"
        );
    }
}

#[test]
fn benchmark_compare_script_exists_and_is_executable() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmark_compare.sh");
    assert!(script.exists(), "benchmark_compare.sh should exist");

    // Verify it passes bash syntax check
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
fn benchmark_compare_list_discovers_all_suites() {
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

    // Verify all 12 benchmark suites are discovered
    for name in EXPECTED_BENCHMARKS {
        assert!(
            stdout.contains(name),
            "benchmark_compare.sh --list should include '{name}' but output was:\n{stdout}"
        );
    }

    // Verify the count line
    assert!(
        stdout.contains(&format!("({})", EXPECTED_BENCHMARKS.len())),
        "Should report correct count of benchmark suites"
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
