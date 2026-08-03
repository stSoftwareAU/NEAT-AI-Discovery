//! Issue #1912: `scripts/fuzz-ci.sh` must install cargo-fuzz with both
//! `--locked` and an explicit `--version`.
//!
//! Without `--version` the script fetched whatever crates.io served at run
//! time; without `--locked` `cargo install` re-resolved the whole transitive
//! graph, so a freshly published dependency's `build.rs` executed on the
//! runner — exactly the gap Issue #1223 closed for the workflow call sites.
//!
//! These tests run the real script with stub `cargo` and `rustup` binaries on
//! `PATH`, so the assertions are on the commands actually invoked and the exit
//! code, not on the script's source text.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A sandbox holding stub `cargo`/`rustup` binaries and their invocation log.
struct Sandbox {
    dir: tempfile::TempDir,
}

impl Sandbox {
    /// `fuzz_installed` makes the `cargo +nightly fuzz --version` probe succeed
    /// (so no install is needed); `failing_target` is a fuzz target whose run
    /// the stub fails.
    fn new(fuzz_installed: bool, failing_target: &str) -> Self {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bin = dir.path().join("bin");
        fs::create_dir_all(&bin).expect("create bin dir");
        let log = dir.path().join("commands.log");

        let cargo = format!(
            r#"#!/bin/bash
echo "$*" >> "{log}"
if [[ "$*" == *"fuzz --version"* ]]; then
    [[ "{fuzz_installed}" == "true" ]] && exit 0
    exit 1
fi
if [[ "$*" == *"fuzz run {failing_target} "* ]]; then
    exit 1
fi
exit 0
"#,
            log = log.display(),
            fuzz_installed = fuzz_installed,
            failing_target = failing_target,
        );
        let rustup = format!(
            r#"#!/bin/bash
echo "rustup $*" >> "{log}"
exit 0
"#,
            log = log.display()
        );

        for (name, body) in [("cargo", cargo), ("rustup", rustup)] {
            let path = bin.join(name);
            fs::write(&path, body).expect("write stub");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod stub");
            }
        }

        Self { dir }
    }

    fn run(&self, max_time: &str) -> Output {
        let path = format!(
            "{}:{}",
            self.dir.path().join("bin").display(),
            std::env::var("PATH").unwrap_or_default()
        );
        Command::new("bash")
            .arg(repo_root().join("scripts/fuzz-ci.sh"))
            .arg(max_time)
            .current_dir(repo_root())
            .env("PATH", path)
            .output()
            .expect("run fuzz-ci.sh")
    }

    fn log(&self) -> String {
        fs::read_to_string(self.dir.path().join("commands.log")).unwrap_or_default()
    }
}

#[test]
fn cargo_fuzz_is_installed_with_locked_and_a_version_pin() {
    let sandbox = Sandbox::new(false, "");
    let output = sandbox.run("7");
    assert!(
        output.status.success(),
        "fuzz-ci.sh failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let install = sandbox
        .log()
        .lines()
        .find(|line| line.contains("install") && line.contains("cargo-fuzz"))
        .map(str::to_owned)
        .expect("cargo-fuzz install was never attempted");

    assert!(
        install.contains("--locked"),
        "install must pass --locked: {install}"
    );
    assert!(
        install.contains("--version 0.13.2"),
        "install must pin an explicit version: {install}"
    );
}

#[test]
fn both_fuzz_targets_still_run_with_the_requested_budget() {
    let sandbox = Sandbox::new(false, "");
    let output = sandbox.run("7");
    assert!(
        output.status.success(),
        "fuzz-ci.sh must succeed when targets pass"
    );

    let log = sandbox.log();
    for target in ["fuzz_ffi_deserialisation", "fuzz_ffi_entry_points"] {
        assert!(
            log.contains(&format!("+nightly fuzz run {target} -- -max_total_time=7")),
            "target {target} did not run with the requested budget:\n{log}"
        );
    }
}

#[test]
fn an_already_installed_cargo_fuzz_is_not_reinstalled() {
    let sandbox = Sandbox::new(true, "");
    let output = sandbox.run("3");
    assert!(output.status.success());
    assert!(
        !sandbox.log().contains("install --locked"),
        "cargo-fuzz must not be reinstalled when the probe succeeds:\n{}",
        sandbox.log()
    );
}

/// A crashing fuzz target must fail the script loudly, not be swallowed.
#[test]
fn a_failing_target_exits_non_zero() {
    let sandbox = Sandbox::new(true, "fuzz_ffi_entry_points");
    let output = sandbox.run("3");
    assert!(
        !output.status.success(),
        "a failing fuzz target must fail the run"
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("fuzz_ffi_entry_points failed"),
        "the failing target must be named in the output"
    );
}
