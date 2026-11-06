//! Issue #2555: a passing quality gate stays quiet, and a failing one keeps its
//! full output.
//!
//! A successful run used to print every `Compiling` and `Running` line. The
//! cargo steps of `quality.sh` now run with `-q`, the coverage workflow sets
//! `CARGO_TERM_QUIET=true` for Cargo and its test reporter, the `cargo install`
//! steps in `security.yml` run with `-q`, and the `npm install` steps run with
//! `--loglevel=error`. These settings suppress successful progress without
//! hiding a test name, assertion message or compiler error.
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
/// removed. Inline and indented block commands are accepted.
fn workflow_run_commands(body: &str) -> Vec<String> {
    let mut commands = Vec::new();
    let mut block_indent = None;
    let mut block_command = String::new();

    for raw_line in body.lines() {
        let indent = raw_line.len() - raw_line.trim_start().len();
        let line = raw_line.trim();
        if let Some(base_indent) = block_indent {
            if line.is_empty() {
                continue;
            }
            if indent > base_indent {
                if !block_command.is_empty() {
                    block_command.push('\n');
                }
                block_command.push_str(line);
                continue;
            }
            commands.push(std::mem::take(&mut block_command));
            block_indent = None;
        }

        if line.starts_with('#') {
            continue;
        }
        let Some(command) = line
            .strip_prefix("- run:")
            .or_else(|| line.strip_prefix("run:"))
        else {
            continue;
        };
        let command = command.trim();
        if command.starts_with('|') || command.starts_with('>') {
            block_indent = Some(indent);
        } else {
            commands.push(command.to_string());
        }
    }

    if block_indent.is_some() {
        commands.push(block_command);
    }
    commands
}

fn required_workflow_command(body: &str, file: &str, prefix: &str) -> Result<String, String> {
    workflow_run_commands(body)
        .into_iter()
        .find(|command| command.starts_with(prefix))
        .ok_or_else(|| format!("{file} must run {prefix}"))
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
// ci.yml: quiet progress without weakening test selection or install safety.
// ============================================================================

#[test]
fn ci_tests_are_quiet_and_keep_targets_features_and_thread_limit() {
    let body = read_repo_file(".github/workflows/ci.yml");
    let test = required_workflow_command(&body, "ci.yml", "cargo test")
        .expect("ci.yml must run cargo test");
    let args = cargo_args(&test);
    assert!(
        has_cargo_quiet(&test),
        "ci.yml tests must run quiet: {test}"
    );
    assert!(
        !args.contains(&"--verbose"),
        "remove verbose reporting: {test}"
    );
    for flag in ["--lib", "--tests", "--bins", "--all-features"] {
        assert!(
            args.contains(&flag),
            "ci.yml tests must retain {flag}: {test}"
        );
    }
    assert!(passthrough_args(&test).contains(&"--test-threads=2"));
}

#[test]
fn ci_lint_and_build_are_quiet_and_keep_existing_checks() {
    let body = read_repo_file(".github/workflows/ci.yml");
    let lint = required_workflow_command(&body, "ci.yml", "cargo clippy")
        .expect("ci.yml must run cargo clippy");
    assert!(has_cargo_quiet(&lint), "ci.yml lint must run quiet: {lint}");
    for flag in ["--all-targets", "--all-features"] {
        assert!(cargo_args(&lint).contains(&flag));
    }
    assert_eq!(
        passthrough_args(&lint),
        [
            "-D",
            "warnings",
            "-D",
            "clippy::uninlined_format_args",
            "-D",
            "clippy::filter_next",
            "-D",
            "clippy::collapsible_if"
        ]
    );
    let build = required_workflow_command(&body, "ci.yml", "cargo build")
        .expect("ci.yml must run cargo build");
    assert!(
        has_cargo_quiet(&build),
        "ci.yml build must run quiet: {build}"
    );
    assert!(cargo_args(&build).contains(&"--lib"));
}

#[test]
fn ci_cargo_install_is_quiet_locked_and_keeps_the_existing_pin() {
    let body = read_repo_file(".github/workflows/ci.yml");
    let install = required_workflow_command(&body, "ci.yml", "cargo install")
        .expect("ci.yml must install cargo-outdated");
    assert!(
        is_quiet_locked_pinned_install(&install),
        "ci.yml tool install must be quiet, locked and pinned: {install}"
    );
    let args = cargo_args(&install);
    assert!(args.contains(&"cargo-outdated"));
    assert!(args.windows(2).any(|pair| pair == ["--version", "0.16.0"]));
}

#[test]
fn ci_pip_install_is_quiet_and_keeps_hash_checked_requirements() {
    let body = read_repo_file(".github/workflows/ci.yml");
    let install = required_workflow_command(&body, "ci.yml", "pip install")
        .expect("ci.yml must install codespell");
    let args = tokens(&install);
    assert!(
        args.contains(&"--quiet"),
        "ci.yml pip install must run quiet: {install}"
    );
    assert!(args.contains(&"--user"));
    assert!(args.contains(&"--require-hashes"));
    assert!(
        args.windows(2)
            .any(|pair| pair == ["-r", ".github/requirements/codespell-requirements.txt"])
    );
}

// ============================================================================
// 2. cargo-quality.yml: Cargo and the coverage test reporter run quiet.
// ============================================================================

#[test]
fn coverage_llvm_cov_quiets_cargo_and_keeps_test_threads_two() {
    let body = read_repo_file(".github/workflows/cargo-quality.yml");
    let llvm_cov = required_workflow_command(
        &body,
        "cargo-quality.yml",
        "CARGO_TERM_QUIET=true cargo llvm-cov",
    )
    .expect("coverage command must be present");
    assert!(
        tokens(&llvm_cov).contains(&"CARGO_TERM_QUIET=true"),
        "the cargo-quality.yml llvm-cov command must suppress Cargo progress \
         (Issue #2555): {llvm_cov}"
    );
    assert!(
        passthrough_args(&llvm_cov).contains(&"--test-threads=2"),
        "the cargo-quality.yml llvm-cov command must keep `--test-threads=2` after `--` \
         (Issue #2555): {llvm_cov}"
    );
    assert!(
        !has_libtest_quiet(&llvm_cov),
        "CARGO_TERM_QUIET already quiets libtest; a second `--quiet` is rejected: {llvm_cov}"
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

#[test]
fn workflow_parser_reads_multiline_run_blocks_and_rejects_a_missing_command() {
    let body = "steps:\n  - name: coverage\n    run: |\n      CARGO_TERM_QUIET=true cargo llvm-cov --all-features \\\n        -- --test-threads=2\n  - name: next\n    run: echo done\n";
    let commands = workflow_run_commands(body);
    assert_eq!(commands.len(), 2);
    assert!(commands[0].contains("CARGO_TERM_QUIET=true cargo llvm-cov"));
    assert!(commands[0].contains("--test-threads=2"));
    assert_eq!(commands[1], "echo done");

    let missing = required_workflow_command(body, "workflow.yml", "cargo test");
    assert!(matches!(missing, Err(message) if message.contains("cargo test")));
}
