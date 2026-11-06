//! Issue #2557: benchmark timings must be collected on a quiet host (an
//! otherwise idle machine, on an exclusive runner), and that requirement must
//! be stated where an operator reads about running the benchmarks.
//!
//! Another job competing for CPU, GPU or memory bandwidth skews the timings. A
//! baseline taken on an idle host compared against a run taken under load can
//! report a regression that is only contention. The requirement was undocumented
//! in the scripts and in `docs/BENCHMARKS.md`, so these tests pin the wording at
//! the three places an operator reads it: the `--help` text of
//! `benchmark_compare.sh`, the `--help` text of `scripts/benchmark-ci.sh`, and
//! the Quick Start and CI Workflow sections of `docs/BENCHMARKS.md`.
//!
//! The help tests drive the real scripts. The documentation tests read the real
//! `docs/BENCHMARKS.md`.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Run `bash <script> --help` from the repository root and return its stdout.
fn help_stdout(script: &str) -> String {
    let output = Command::new("bash")
        .arg(script)
        .arg("--help")
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("failed to run bash {script} --help: {e}"));
    assert!(
        output.status.success(),
        "bash {script} --help should succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The phrases that together state the quiet-host requirement.
fn assert_help_states_quiet_host(script: &str, help: &str) {
    for phrase in ["idle machine", "exclusive runner", "quality gates"] {
        assert!(
            help.contains(phrase),
            "{script} --help must state the quiet-host requirement; missing {phrase:?}\n\
             --- help output ---\n{help}"
        );
    }
}

#[test]
fn benchmark_compare_help_states_the_quiet_host_requirement() {
    let help = help_stdout("benchmark_compare.sh");
    assert_help_states_quiet_host("benchmark_compare.sh", &help);
}

#[test]
fn benchmark_ci_help_states_the_quiet_host_requirement() {
    let help = help_stdout("scripts/benchmark-ci.sh");
    assert_help_states_quiet_host("scripts/benchmark-ci.sh", &help);
}

#[test]
fn benchmarks_doc_states_the_requirement_before_the_quick_start_commands() {
    let doc = fs::read_to_string(repo_root().join("docs").join("BENCHMARKS.md"))
        .expect("read docs/BENCHMARKS.md");

    // The Quick Start note must sit between its heading and the first bash
    // fence, so an operator reads the requirement before the commands.
    let heading = "## 🚀 Quick Start";
    let heading_at = doc
        .find(heading)
        .expect("docs/BENCHMARKS.md must have a Quick Start heading");
    let after_heading = &doc[heading_at + heading.len()..];
    let fence_at = after_heading
        .find("```bash")
        .expect("Quick Start must contain a bash block");
    let before_commands = &after_heading[..fence_at];
    assert!(
        before_commands.contains("idle machine"),
        "Quick Start must state the quiet-host requirement before its bash block \
         (missing \"idle machine\")"
    );
    assert!(
        before_commands.contains("exclusive runner"),
        "Quick Start must state the quiet-host requirement before its bash block \
         (missing \"exclusive runner\")"
    );

    // Tier 2 is the comparison tier, so it must say the runner is idle and
    // exclusive while timings are collected.
    let tier2_start = doc
        .find("**Tier 2")
        .expect("CI Workflow must contain the Tier 2 paragraph");
    let tier2_end = doc[tier2_start..]
        .find("### Regression Threshold")
        .map(|offset| tier2_start + offset)
        .expect("Tier 2 paragraph must be followed by the Regression Threshold section");
    let tier2 = &doc[tier2_start..tier2_end];
    assert!(
        tier2.contains("idle"),
        "Tier 2 must state the runner is otherwise idle while timings are collected \
         (missing \"idle\")"
    );
    assert!(
        tier2.contains("exclusive"),
        "Tier 2 must state the runner is exclusive while timings are collected \
         (missing \"exclusive\")"
    );
}
