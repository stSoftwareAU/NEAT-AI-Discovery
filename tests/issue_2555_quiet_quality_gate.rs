//! Issue #2555: a passing quality gate stays quiet, and a failing one keeps its
//! full output.
//!
//! A successful run used to print every `Compiling` and `Running` line. The
//! cargo steps of `quality.sh` now run with `-q`, the coverage test binaries in
//! `cargo-quality.yml` receive libtest `--quiet`, the `cargo install` steps in
//! `security.yml` run with `-q`, and the `npm install` steps run with
//! `--loglevel=error`. None of these flags hides a test name, an assertion
//! message or a compiler error, so a failure stays fully diagnosable.
//!
//! The assertions parse the `run:` command lines (never comments) and inspect
//! their tokens, so a flag mentioned only in a comment cannot satisfy a check.
//! Each detector below has a negative case, so the checks can fail on the wrong
//! input.

use std::fs;
use std::path::{Path, PathBuf};

const CARGO_SUBCOMMANDS_QUIET: [&str; 5] = ["build", "clippy", "check", "test", "doc"];

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn read_repo_file(relative: &str) -> String {
    let path: PathBuf = repo_root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The command of every non-comment `run:` key in a workflow, with the key
/// removed. A list-item dash (`- run:`) is accepted.
fn workflow_run_commands(body: &str) -> Vec<String> {
    body.lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| line.trim_start_matches("- ").strip_prefix("run:"))
        .map(|command| command.trim().to_string())
        .collect()
}

/// The command of every non-comment line of `quality.sh`, with a leading
/// `RUSTDOCFLAGS="-D warnings"` assignment removed so the program comes first.
fn quality_sh_commands() -> Vec<String> {
    read_repo_file("quality.sh")
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            line.strip_prefix("RUSTDOCFLAGS=\"-D warnings\" ")
                .unwrap_or(line)
                .to_string()
        })
        .collect()
}

fn tokens(command: &str) -> Vec<&str> {
    command.split_whitespace().collect()
}

/// The cargo's own arguments: everything before the `--` separator.
fn cargo_args(command: &str) -> Vec<&str> {
    tokens(command)
        .into_iter()
        .take_while(|token| *token != "--")
        .collect()
}

/// The arguments handed on to the test binary: everything after `--`.
fn passthrough_args(command: &str) -> Vec<&str> {
    tokens(command)
        .into_iter()
        .skip_while(|token| *token != "--")
        .skip(1)
        .collect()
}

/// The cargo subcommand (`build`, `test`, ...) when the command is a cargo call.
fn cargo_subcommand(command: &str) -> Option<&str> {
    let mut parts = tokens(command).into_iter();
    if parts.next()? != "cargo" {
        return None;
    }
    parts.next()
}

/// True when cargo itself is told to be quiet: `-q` or `--quiet` before `--`.
fn has_cargo_quiet(command: &str) -> bool {
    cargo_args(command)
        .iter()
        .any(|token| matches!(*token, "-q" | "--quiet"))
}

/// True when the test binary is told to be quiet: `-q` or `--quiet` after `--`.
fn has_libtest_quiet(command: &str) -> bool {
    passthrough_args(command)
        .iter()
        .any(|token| matches!(*token, "-q" | "--quiet"))
}

/// True when a `cargo install` is quiet, `--locked`, and pinned with `--version X`.
fn is_quiet_locked_pinned_install(command: &str) -> bool {
    let args = cargo_args(command);
    let pinned = args
        .windows(2)
        .any(|pair| pair[0] == "--version" && !pair[1].starts_with('-'));
    has_cargo_quiet(command) && args.contains(&"--locked") && pinned
}

/// True when an `npm install` reports only errors and skips lifecycle scripts.
fn is_quiet_npm_install(command: &str) -> bool {
    let args = tokens(command);
    args.contains(&"--loglevel=error") && args.contains(&"--ignore-scripts")
}

// ============================================================================
// 1. quality.sh: every cargo build/clippy/check/test/doc call runs quiet.
// ============================================================================

#[test]
fn quality_sh_cargo_build_clippy_check_test_and_doc_run_quiet() {
    let commands = quality_sh_commands();
    let mut checked: Vec<&str> = Vec::new();
    for command in &commands {
        let Some(subcommand) = cargo_subcommand(command) else {
            continue;
        };
        if !CARGO_SUBCOMMANDS_QUIET.contains(&subcommand) {
            continue;
        }
        checked.push(subcommand);
        assert!(
            has_cargo_quiet(command),
            "quality.sh runs `{command}` without `-q` or `--quiet` before `--` \
             (Issue #2555): a passing gate must stay quiet"
        );
    }
    // Sanity: each named subcommand was actually seen, so the loop above
    // cannot pass vacuously.
    for subcommand in CARGO_SUBCOMMANDS_QUIET {
        assert!(
            checked.contains(&subcommand),
            "quality.sh must run `cargo {subcommand}` so the quiet check is not vacuous \
             (Issue #2555)"
        );
    }
}

#[test]
fn quality_sh_test_command_keeps_test_threads_two() {
    let test = quality_sh_commands()
        .into_iter()
        .find(|command| cargo_subcommand(command) == Some("test"))
        .expect("quality.sh must run cargo test");
    assert!(
        passthrough_args(&test).contains(&"--test-threads=2"),
        "the quality.sh test command must still pass `--test-threads=2` after `--` \
         (Issue #2555): {test}"
    );
}

// ============================================================================
// 2. cargo-quality.yml: the coverage test binaries receive libtest --quiet.
// ============================================================================

#[test]
fn coverage_llvm_cov_passes_quiet_after_the_separator() {
    let body = read_repo_file(".github/workflows/cargo-quality.yml");
    let llvm_cov = workflow_run_commands(&body)
        .into_iter()
        .find(|command| command.starts_with("cargo llvm-cov"))
        .expect("cargo-quality.yml must run cargo llvm-cov");
    assert!(
        has_libtest_quiet(&llvm_cov),
        "the cargo-quality.yml llvm-cov command must pass `--quiet` after `--` \
         (Issue #2555): {llvm_cov}"
    );
    assert!(
        passthrough_args(&llvm_cov).contains(&"--test-threads=2"),
        "the cargo-quality.yml llvm-cov command must keep `--test-threads=2` after `--` \
         (Issue #2555): {llvm_cov}"
    );
}

// ============================================================================
// 3. security.yml: both cargo install steps run quiet, locked and pinned.
// ============================================================================

#[test]
fn security_workflow_cargo_installs_are_quiet_locked_and_pinned() {
    let body = read_repo_file(".github/workflows/security.yml");
    let installs: Vec<String> = workflow_run_commands(&body)
        .into_iter()
        .filter(|command| command.starts_with("cargo install"))
        .collect();
    assert!(
        installs.len() >= 2,
        "security.yml must install cargo-audit and cargo-deny; found {} install step(s) \
         (Issue #2555)",
        installs.len()
    );
    for install in &installs {
        assert!(
            is_quiet_locked_pinned_install(install),
            "every security.yml `cargo install` must carry `-q`, `--locked` and \
             `--version X` (Issue #2555): {install}"
        );
    }
}

// ============================================================================
// 4. markdown-lint.yml and renovate-validate.yml: npm installs are quiet and
//    keep --ignore-scripts.
// ============================================================================

#[test]
fn npm_installs_are_quiet_and_skip_lifecycle_scripts() {
    for relative in [
        ".github/workflows/markdown-lint.yml",
        ".github/workflows/renovate-validate.yml",
    ] {
        let body = read_repo_file(relative);
        let install = workflow_run_commands(&body)
            .into_iter()
            .find(|command| command.starts_with("npm install"))
            .unwrap_or_else(|| panic!("{relative} must install a tool via npm (Issue #2555)"));
        assert!(
            is_quiet_npm_install(&install),
            "the npm install in {relative} must carry `--loglevel=error` and \
             `--ignore-scripts` (Issue #2555): {install}"
        );
    }
}

// ============================================================================
// Negative capability checks: each detector must reject the wrong input, so a
// check that is always true cannot pass unnoticed.
// ============================================================================

#[test]
fn cargo_quiet_detector_rejects_a_loud_command() {
    assert!(
        has_cargo_quiet("cargo build -q --release --lib"),
        "positive control"
    );
    assert!(!has_cargo_quiet("cargo build --release --lib"));
    assert!(
        !has_cargo_quiet("cargo test --lib -- --quiet"),
        "`--quiet` after `--` is for the test binary, not cargo"
    );
}

#[test]
fn libtest_quiet_detector_rejects_a_loud_command() {
    assert!(
        has_libtest_quiet("cargo llvm-cov -- --test-threads=2 --quiet"),
        "positive control"
    );
    assert!(!has_libtest_quiet(
        "cargo llvm-cov --all-features -- --test-threads=2"
    ));
    assert!(
        !has_libtest_quiet("cargo test -q --lib"),
        "`-q` before `--` is cargo's flag, not the test binary's"
    );
}

#[test]
fn install_detector_rejects_an_unquiet_or_unpinned_install() {
    assert!(
        is_quiet_locked_pinned_install("cargo install -q --locked --version 0.22.1 cargo-audit"),
        "positive control"
    );
    assert!(!is_quiet_locked_pinned_install(
        "cargo install --locked --version 0.22.1 cargo-audit"
    ));
    assert!(
        !is_quiet_locked_pinned_install("cargo install -q cargo-audit"),
        "no --locked and no --version pin"
    );
}

#[test]
fn npm_detector_rejects_a_loud_or_unguarded_install() {
    assert!(
        is_quiet_npm_install("npm install -g --loglevel=error --ignore-scripts renovate@44.7.2"),
        "positive control"
    );
    assert!(!is_quiet_npm_install(
        "npm install -g --ignore-scripts renovate@44.7.2"
    ));
    assert!(
        !is_quiet_npm_install("npm install -g --loglevel=error renovate@44.7.2"),
        "no --ignore-scripts"
    );
}

#[test]
fn workflow_parser_ignores_commented_out_commands() {
    let body = "      # run: cargo llvm-cov -- --quiet\n      run: cargo llvm-cov\n";
    assert_eq!(
        workflow_run_commands(body),
        vec!["cargo llvm-cov".to_string()],
        "a commented-out `run:` line must not count as a step command"
    );
}
