//! Issue #1891: harden Rust toolchain setup against codeload download timeouts.
//!
//! `dtolnay/rust-toolchain` is fetched from `codeload.github.com` during the
//! runner's *Prepare all required actions* phase. That fetch has a fixed 100 s
//! `HttpClient` timeout and a 3-attempt retry policy, neither of which a workflow
//! can tune — so a codeload stall fails the job before any repository code
//! runs (run 30665006027 on PR #1890).
//!
//! The fix replaces the action with `scripts/install-rust-toolchain.sh`, a
//! committed script that drives the runner's preinstalled `rustup`. These tests
//! exercise that script for real: a stub `rustup` is placed on `PATH` and the
//! script is executed, so the assertions are on observable behaviour (exit
//! codes, the commands actually invoked, stderr) rather than on source text.
//!
//! The workflow assertions at the end guard the regression itself: the action
//! must not reappear at any of the five call sites. Those call sites now reach
//! the script through the `.github/actions/setup-rust` composite action rather
//! than invoking it inline (Issue #2036), so the assertions follow that one
//! step of indirection; what they enforce is unchanged.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script_path() -> PathBuf {
    repo_root().join("scripts/install-rust-toolchain.sh")
}

/// A sandbox holding a stub `rustup`, its invocation log, and a fake `HOME`.
struct Sandbox {
    dir: tempfile::TempDir,
}

impl Sandbox {
    /// `install_failures` is the number of leading `toolchain install`
    /// invocations the stub fails before succeeding; `fail_verify` makes the
    /// post-install `rustup run … rustc --version` probe fail.
    fn new(install_failures: u32, fail_verify: bool) -> Self {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bin = dir.path().join("bin");
        fs::create_dir_all(&bin).expect("create bin dir");
        fs::create_dir_all(dir.path().join("home/.cargo/bin")).expect("create fake cargo bin");

        let log = dir.path().join("rustup.log");
        let budget = dir.path().join("failures");
        fs::write(&budget, format!("{install_failures}\n")).expect("write failure budget");

        let stub = format!(
            r#"#!/bin/bash
echo "$*" >> "{log}"
if [[ "${{1:-}}" == "toolchain" && "${{2:-}}" == "install" ]]; then
    remaining=$(cat "{budget}")
    if [[ "$remaining" -gt 0 ]]; then
        echo $((remaining - 1)) > "{budget}"
        echo "stub rustup: simulated transient network failure" >&2
        exit 1
    fi
fi
if [[ "${{1:-}}" == "run" && "{fail_verify}" == "true" ]]; then
    echo "stub rustup: simulated broken toolchain" >&2
    exit 1
fi
exit 0
"#,
            log = log.display(),
            budget = budget.display(),
            fail_verify = fail_verify,
        );
        let stub_path = bin.join("rustup");
        fs::write(&stub_path, stub).expect("write stub rustup");
        make_executable(&stub_path);

        Self { dir }
    }

    fn log(&self) -> String {
        fs::read_to_string(self.dir.path().join("rustup.log")).unwrap_or_default()
    }

    /// Run the script with the stub on `PATH`. `on_path` false omits the stub
    /// directory so `rustup` cannot be found at all.
    fn run(&self, args: &[&str], on_path: bool) -> Output {
        let mut path = String::new();
        if on_path {
            path.push_str(&self.dir.path().join("bin").display().to_string());
            path.push(':');
        }
        path.push_str("/usr/bin:/bin");

        Command::new("bash")
            .arg(script_path())
            .args(args)
            .env_clear()
            .env("PATH", path)
            .env("HOME", self.dir.path().join("home"))
            .env("RUST_TOOLCHAIN_RETRY_DELAY", "0")
            .env("RUST_TOOLCHAIN_MAX_ATTEMPTS", "3")
            .output()
            .expect("run install-rust-toolchain.sh")
    }
}

fn make_executable(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path).expect("stat").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).expect("chmod");
    }
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

#[test]
fn script_is_committed_and_executable() {
    let path = script_path();
    assert!(path.is_file(), "{} must exist", path.display());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&path)
            .expect("stat script")
            .permissions()
            .mode();
        assert!(
            mode & 0o111 != 0,
            "{} must be executable (mode {mode:o})",
            path.display()
        );
    }
}

#[test]
fn help_flag_prints_usage_and_exits_zero() {
    let sandbox = Sandbox::new(0, false);
    let out = sandbox.run(&["--help"], true);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    assert!(
        stdout_of(&out).contains("Usage:"),
        "expected usage text, got: {}",
        stdout_of(&out)
    );
}

#[test]
fn installs_stable_by_default_and_sets_it_as_the_default_toolchain() {
    let sandbox = Sandbox::new(0, false);
    let out = sandbox.run(&[], true);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    let log = sandbox.log();
    assert!(
        log.contains("toolchain install stable --profile minimal --no-self-update"),
        "install invocation missing from log:\n{log}"
    );
    assert!(
        log.lines().any(|l| l == "default stable"),
        "`rustup default stable` missing from log:\n{log}"
    );
}

#[test]
fn verifies_the_toolchain_after_installing_it() {
    // Absence of a failure is not success — the script must positively confirm
    // the toolchain runs before reporting a clean install.
    let sandbox = Sandbox::new(0, false);
    let out = sandbox.run(&[], true);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    let log = sandbox.log();
    assert!(
        log.contains("run stable rustc --version"),
        "rustc verification missing from log:\n{log}"
    );
    assert!(
        log.contains("run stable cargo --version"),
        "cargo verification missing from log:\n{log}"
    );
}

#[test]
fn fails_loud_when_the_installed_toolchain_does_not_run() {
    let sandbox = Sandbox::new(0, true);
    let out = sandbox.run(&[], true);
    assert!(
        !out.status.success(),
        "a broken toolchain must not be reported as success"
    );
}

#[test]
fn passes_requested_components_through_to_rustup() {
    let sandbox = Sandbox::new(0, false);
    let out = sandbox.run(&["stable", "rustfmt", "clippy"], true);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    let log = sandbox.log();
    assert!(
        log.contains("--component rustfmt --component clippy"),
        "components missing from log:\n{log}"
    );
}

#[test]
fn accepts_components_as_a_single_comma_separated_argument() {
    let sandbox = Sandbox::new(0, false);
    let out = sandbox.run(&["stable", "rustfmt, clippy"], true);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    let log = sandbox.log();
    assert!(
        log.contains("--component rustfmt --component clippy"),
        "comma-separated components missing from log:\n{log}"
    );
}

#[test]
fn retries_a_transient_install_failure() {
    // Two failures then success, within the 3-attempt budget.
    let sandbox = Sandbox::new(2, false);
    let out = sandbox.run(&[], true);
    assert!(
        out.status.success(),
        "transient failures must be retried; stderr: {}",
        stderr_of(&out)
    );

    let attempts = sandbox
        .log()
        .lines()
        .filter(|l| l.starts_with("toolchain install"))
        .count();
    assert_eq!(
        attempts,
        3,
        "expected 3 install attempts, log:\n{}",
        sandbox.log()
    );
}

#[test]
fn fails_loud_when_the_retry_budget_is_exhausted() {
    let sandbox = Sandbox::new(99, false);
    let out = sandbox.run(&[], true);
    assert!(
        !out.status.success(),
        "an unrecoverable install must exit non-zero"
    );

    let attempts = sandbox
        .log()
        .lines()
        .filter(|l| l.starts_with("toolchain install"))
        .count();
    assert_eq!(attempts, 3, "expected the 3-attempt budget to be honoured");

    let stderr = stderr_of(&out);
    assert!(
        stderr.contains("3 attempt"),
        "error must name the attempt budget, got: {stderr}"
    );
}

#[test]
fn fails_loud_when_rustup_is_missing() {
    let sandbox = Sandbox::new(0, false);
    let out = sandbox.run(&[], false);
    assert!(!out.status.success(), "a missing rustup must exit non-zero");
    assert!(
        stderr_of(&out).contains("rustup"),
        "error must name rustup, got: {}",
        stderr_of(&out)
    );
}

#[test]
fn rejects_toolchain_and_component_names_that_are_not_plain_identifiers() {
    for args in [
        vec!["stable; rm -rf /tmp/nope"],
        vec!["stable", "clippy && curl evil.example"],
        vec!["$(whoami)"],
    ] {
        let sandbox = Sandbox::new(0, false);
        let out = sandbox.run(&args, true);
        assert!(
            !out.status.success(),
            "{args:?} must be rejected as an invalid name"
        );
        assert!(
            sandbox.log().is_empty(),
            "rustup must not be invoked for rejected input {args:?}, log:\n{}",
            sandbox.log()
        );
    }
}

#[test]
fn exports_cargo_bin_to_github_path_when_running_on_a_runner() {
    let sandbox = Sandbox::new(0, false);
    let github_path = sandbox.dir.path().join("github_path");
    fs::write(&github_path, "").expect("seed GITHUB_PATH file");

    let out = Command::new("bash")
        .arg(script_path())
        .env_clear()
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", sandbox.dir.path().join("bin").display()),
        )
        .env("HOME", sandbox.dir.path().join("home"))
        .env("RUST_TOOLCHAIN_RETRY_DELAY", "0")
        .env("GITHUB_PATH", &github_path)
        .output()
        .expect("run install-rust-toolchain.sh");
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    let written = fs::read_to_string(&github_path).expect("read GITHUB_PATH");
    assert!(
        written.contains(".cargo/bin"),
        "cargo bin dir must be exported to GITHUB_PATH, got: {written:?}"
    );
}

// --- Workflow regression guards -------------------------------------------

fn workflow(name: &str) -> String {
    let path = repo_root().join(".github/workflows").join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The composite action every call site bootstraps Rust through (Issue #2036).
fn setup_rust_action() -> String {
    let path = repo_root().join(".github/actions/setup-rust/action.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

const WORKFLOWS_WITH_TOOLCHAIN: [(&str, usize); 3] =
    [("ci.yml", 3), ("security.yml", 1), ("cargo-quality.yml", 1)];

#[test]
fn no_workflow_downloads_the_rust_toolchain_action() {
    for (name, _) in WORKFLOWS_WITH_TOOLCHAIN {
        let body = workflow(name);
        assert!(
            !body.contains("dtolnay/rust-toolchain"),
            "{name} still downloads dtolnay/rust-toolchain from codeload (Issue #1891)"
        );
    }
    assert!(
        !setup_rust_action().contains("dtolnay/rust-toolchain"),
        "the setup-rust composite action must not reintroduce dtolnay/rust-toolchain \
         (Issue #1891)"
    );
}

#[test]
fn every_toolchain_call_site_uses_the_committed_script() {
    // The call sites reference the composite action, which is the single caller
    // of the committed script (Issue #2036).
    for (name, expected) in WORKFLOWS_WITH_TOOLCHAIN {
        let body = workflow(name);
        let sites = body.matches("uses: ./.github/actions/setup-rust").count();
        assert_eq!(
            sites, expected,
            "{name} must bootstrap Rust at {expected} site(s) via the setup-rust action"
        );
    }
    assert!(
        setup_rust_action().contains("./scripts/install-rust-toolchain.sh"),
        "the setup-rust composite action must invoke the committed script (Issue #1891)"
    );
}

#[test]
fn call_sites_that_need_rustfmt_and_clippy_still_request_them() {
    // The quality and auto-format jobs in ci.yml previously asked the action
    // for `components: rustfmt, clippy`; the composite action must be given the
    // same, and pass it to the script (Issue #2036).
    let body = workflow("ci.yml");
    let with_components = body.matches("components: rustfmt, clippy").count();
    assert_eq!(
        with_components, 2,
        "ci.yml must request rustfmt+clippy at the quality and auto-format jobs"
    );
    assert!(
        setup_rust_action().contains("components:"),
        "the setup-rust composite action must accept a `components` input"
    );
}
