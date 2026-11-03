//! Boolean env-toggle parsing for the six presence-only accessors (Issue #2467).
//!
//! `verbose`, `gpu_timing`, `neuron_targets_output_only`, `quiet_gpu`, `timing`
//! and `gpu_metrics` used to turn on for any set value, so `=0`, `=false` and
//! `=no` enabled them. The documented contract is that `1`, `true` and `yes`
//! (trimmed, case-insensitive) enable a toggle and every other value, including
//! unset or empty, leaves it off.
//!
//! Four of the accessors cache their result in a process-wide `OnceLock`, so one
//! process can only observe a single value. Each case therefore re-executes this
//! test binary as a child process with the six variables set to the same value.
//! The child runs only `probe_toggles_in_child`, which is a no-op unless
//! `ISSUE_2467_CHILD_EXPECT` is set. The parent passes its environment to the
//! `Command` only and never mutates its own.

use neat_ai_discovery::config;
use std::process::{Command, Stdio};

const TOGGLES: [&str; 6] = [
    "NEAT_AI_DISCOVERY_VERBOSE",
    "NEAT_AI_DISCOVERY_GPU_TIMING",
    "NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY",
    "NEAT_AI_DISCOVERY_QUIET_GPU",
    "NEAT_AI_DISCOVERY_TIMING",
    "NEAT_AI_DISCOVERY_GPU_METRICS",
];

const CHILD_EXPECT_ENV: &str = "ISSUE_2467_CHILD_EXPECT";
const PROBE_TEST: &str = "probe_toggles_in_child";
const PROBED_MARKER: &str = "ISSUE_2467_PROBED";

const OFF_VALUES: [&str; 5] = ["0", "false", "no", "FALSE", ""];
const ON_VALUES: [&str; 4] = ["1", "true", "yes", " YES "];

/// Child-side probe. Does nothing unless the parent set `ISSUE_2467_CHILD_EXPECT`.
#[test]
fn probe_toggles_in_child() {
    let Ok(expect) = std::env::var(CHILD_EXPECT_ENV) else {
        return;
    };
    println!("{PROBED_MARKER}");

    let expected = match expect.as_str() {
        "on" => true,
        "off" => false,
        other => panic!("unknown {CHILD_EXPECT_ENV} value {other:?}, expected on or off"),
    };

    let accessors: [(&str, bool); 6] = [
        ("verbose", config::verbose()),
        ("gpu_timing", config::gpu_timing()),
        (
            "neuron_targets_output_only",
            config::neuron_targets_output_only(),
        ),
        ("quiet_gpu", config::quiet_gpu()),
        ("timing", config::timing()),
        ("gpu_metrics", config::gpu_metrics()),
    ];
    for (name, actual) in accessors {
        assert_eq!(
            actual, expected,
            "accessor `{name}` returned {actual} but expected {expected}"
        );
    }
}

/// Re-executes this test binary, running only the probe, with all six toggles
/// set to `value` (or unset when `value` is `None`).
fn run_child(value: Option<&str>, expect: &str) {
    let exe = std::env::current_exe().expect("resolve current test binary");
    let mut cmd = Command::new(exe);
    cmd.args([PROBE_TEST, "--exact", "--nocapture", "--test-threads=1"])
        .stdin(Stdio::null())
        .env(CHILD_EXPECT_ENV, expect);

    for name in TOGGLES {
        cmd.env_remove(name);
        if let Some(v) = value {
            cmd.env(name, v);
        }
    }

    let output = cmd.output().expect("spawn child test binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let context = format!(
        "value {value:?}, expect {expect}\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}"
    );

    assert!(
        output.status.success(),
        "child probe failed ({})\n{context}",
        output.status
    );
    assert!(
        stdout.contains(PROBED_MARKER),
        "child probe was skipped, marker {PROBED_MARKER} missing\n{context}"
    );
}

#[test]
fn off_values_disable_all_six_toggles() {
    for value in OFF_VALUES {
        run_child(Some(value), "off");
    }
}

#[test]
fn on_values_enable_all_six_toggles() {
    for value in ON_VALUES {
        run_child(Some(value), "on");
    }
}

#[test]
fn unset_toggles_are_off() {
    run_child(None, "off");
}
