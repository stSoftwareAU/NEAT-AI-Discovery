//! Issue #2036: the Rust bootstrap step is defined once, in a composite action.
//!
//! Five jobs across three workflows — `ci.yml` (`version-increment`, `quality`,
//! `auto-format`), `cargo-quality.yml` (`coverage`) and `security.yml`
//! (`security`) — each hand-maintained the same invocation of
//! `scripts/install-rust-toolchain.sh` plus its rationale comment. Adding a
//! required rustup component meant five edits, and missing one silently left
//! that job on the old toolchain.
//!
//! `.github/actions/setup-rust/action.yml` now owns that invocation and every
//! call site references it. The checkout deliberately stays in each job: GitHub
//! loads a local action (`uses: ./…`) from the runner's workspace, so the
//! repository must already be checked out before the action exists on disk.
//!
//! The behavioural tests below execute the action's own `run:` body against a
//! stub `rustup`, so the assertions are on the commands rustup actually
//! receives — not on source text.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

const ACTION_REF: &str = "uses: ./.github/actions/setup-rust";
const ACTION_PATH: &str = ".github/actions/setup-rust/action.yml";

/// Call sites: workflow file, and how many jobs in it bootstrap Rust.
const CALL_SITES: [(&str, usize); 3] =
    [("ci.yml", 3), ("security.yml", 1), ("cargo-quality.yml", 1)];

fn action_yml() -> String {
    let path = repo_root().join(ACTION_PATH);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn workflow(name: &str) -> String {
    let path = repo_root().join(".github/workflows").join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The shell body of the action's single `run: |` block, dedented.
fn action_run_body() -> String {
    let body = action_yml();
    let mut lines = body.lines();
    let run_line = lines
        .by_ref()
        .find(|line| line.trim_start().starts_with("run: |"))
        .expect("the composite action must carry a `run: |` block");
    let run_indent = run_line.len() - run_line.trim_start().len();

    let mut script = Vec::new();
    for line in lines {
        let indent = line.len() - line.trim_start().len();
        if !line.trim().is_empty() && indent <= run_indent {
            break;
        }
        script.push(line.get(run_indent + 2..).unwrap_or(""));
    }
    assert!(!script.is_empty(), "the `run: |` block must not be empty");
    script.join("\n")
}

/// A sandbox holding a stub `rustup`, its invocation log, and a fake `HOME`.
struct Sandbox {
    dir: tempfile::TempDir,
}

impl Sandbox {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bin = dir.path().join("bin");
        fs::create_dir_all(&bin).expect("create bin dir");
        fs::create_dir_all(dir.path().join("home/.cargo/bin")).expect("create fake cargo bin");

        let log = dir.path().join("rustup.log");
        let stub = format!(
            r#"#!/bin/bash
echo "$*" >> "{log}"
exit 0
"#,
            log = log.display(),
        );
        let stub_path = bin.join("rustup");
        fs::write(&stub_path, stub).expect("write stub rustup");
        make_executable(&stub_path);

        Self { dir }
    }

    fn log(&self) -> String {
        fs::read_to_string(self.dir.path().join("rustup.log")).unwrap_or_default()
    }

    /// Execute the action's `run:` body with the inputs GitHub would export as
    /// environment variables, from the repository root.
    fn run_action(&self, toolchain: &str, components: &str) -> Output {
        Command::new("bash")
            .arg("-c")
            .arg(action_run_body())
            .current_dir(repo_root())
            .env_clear()
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.dir.path().join("bin").display()),
            )
            .env("HOME", self.dir.path().join("home"))
            .env("TOOLCHAIN", toolchain)
            .env("COMPONENTS", components)
            .env("RUST_TOOLCHAIN_RETRY_DELAY", "0")
            .output()
            .expect("run the composite action body")
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

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

// --- Behaviour of the composite action's bootstrap step -------------------

#[test]
fn installs_the_default_toolchain_when_no_inputs_are_supplied() {
    // GitHub exports the declared defaults, so an empty components input is
    // the common case: it must not reach rustup as an empty component name.
    let sandbox = Sandbox::new();
    let out = sandbox.run_action("stable", "");
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    let log = sandbox.log();
    assert!(
        log.contains("toolchain install stable --profile minimal --no-self-update"),
        "install invocation missing from log:\n{log}"
    );
    assert!(
        !log.contains("--component"),
        "no components were requested, log:\n{log}"
    );
}

#[test]
fn passes_the_components_input_through_to_rustup() {
    let sandbox = Sandbox::new();
    let out = sandbox.run_action("stable", "rustfmt, clippy");
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    let log = sandbox.log();
    assert!(
        log.contains("--component rustfmt --component clippy"),
        "components missing from log:\n{log}"
    );
}

#[test]
fn honours_a_non_default_toolchain_input() {
    let sandbox = Sandbox::new();
    let out = sandbox.run_action("nightly", "");
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    let log = sandbox.log();
    assert!(
        log.contains("toolchain install nightly"),
        "requested toolchain missing from log:\n{log}"
    );
    assert!(
        log.lines().any(|line| line == "default nightly"),
        "`rustup default nightly` missing from log:\n{log}"
    );
}

#[test]
fn fails_loud_when_an_input_is_not_a_plain_identifier() {
    // The bootstrap script's allowlist must still be reached: a caller-supplied
    // component must never be interpreted as shell.
    let sandbox = Sandbox::new();
    let out = sandbox.run_action("stable", "clippy && curl evil.example");
    assert!(
        !out.status.success(),
        "an invalid component name must exit non-zero"
    );
    assert!(
        sandbox.log().is_empty(),
        "rustup must not be invoked for rejected input, log:\n{}",
        sandbox.log()
    );
}

// --- The action's contract -------------------------------------------------

#[test]
fn action_declares_the_toolchain_and_components_inputs() {
    let body = action_yml();
    assert!(
        body.contains("using: composite"),
        "{ACTION_PATH} must be a composite action"
    );
    for input in ["toolchain:", "components:"] {
        assert!(
            body.contains(input),
            "{ACTION_PATH} must declare the `{input}` input (Issue #2036)"
        );
    }
}

#[test]
fn action_inputs_reach_the_shell_through_env_not_interpolation() {
    // `${{ }}` pasted into a `run:` body is a script-injection surface; the
    // inputs must be exported as environment variables instead.
    let run_body = action_run_body();
    assert!(
        !run_body.contains("${{"),
        "the action's run body must not interpolate expressions directly:\n{run_body}"
    );
    for variable in ["$TOOLCHAIN", "$COMPONENTS"] {
        assert!(
            run_body.contains(variable),
            "the action's run body must consume `{variable}` from the environment:\n{run_body}"
        );
    }
}

// --- Every call site goes through the action -------------------------------

#[test]
fn every_bootstrap_call_site_uses_the_composite_action() {
    for (name, expected) in CALL_SITES {
        let sites = workflow(name).matches(ACTION_REF).count();
        assert_eq!(
            sites, expected,
            "{name} must bootstrap Rust through `{ACTION_REF}` at {expected} site(s) (Issue #2036)"
        );
    }
}

#[test]
fn no_workflow_invokes_the_bootstrap_script_directly() {
    // Single source of truth: the script is called from the action alone, so a
    // change to the bootstrap sequence is a one-file edit.
    for (name, _) in CALL_SITES {
        assert!(
            !workflow(name).contains("scripts/install-rust-toolchain.sh"),
            "{name} must reach the bootstrap script through {ACTION_PATH}, not directly \
             (Issue #2036)"
        );
    }
    assert!(
        action_yml().contains("scripts/install-rust-toolchain.sh"),
        "{ACTION_PATH} must be the one caller of the bootstrap script"
    );
}

#[test]
fn the_call_sites_that_need_rustfmt_and_clippy_still_request_them() {
    // The quality and auto-format jobs format and lint; losing the components
    // input would fail them at the first `cargo fmt`.
    let body = workflow("ci.yml");
    let with_components = body
        .split(ACTION_REF)
        .skip(1)
        .filter(|block| {
            block
                .lines()
                .take_while(|line| !line.trim_start().starts_with("- "))
                .any(|line| line.trim() == "components: rustfmt, clippy")
        })
        .count();
    assert_eq!(
        with_components, 2,
        "ci.yml must request rustfmt+clippy at the quality and auto-format call sites \
         (Issue #2036)"
    );
}

#[test]
fn each_calling_job_still_checks_out_before_using_the_local_action() {
    // A local action is loaded from the workspace, so the checkout cannot move
    // into it — every call site must be preceded by a checkout in the same file.
    for (name, _) in CALL_SITES {
        let body = workflow(name);
        for (index, _) in body.match_indices(ACTION_REF) {
            let preceding = &body[..index];
            assert!(
                preceding.contains("uses: actions/checkout@"),
                "{name}: every `{ACTION_REF}` must follow a checkout step (Issue #2036)"
            );
        }
    }
}
