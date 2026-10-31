//! Issue #2390: `tests/test_cargo_quality_workflow.sh` was never run by any
//! gate and failed on Develop, so its assertions are ported here as a Rust
//! integration test that runs under `cargo test`, and the script is removed.
//!
//! The assertions guard the Coverage workflow (`cargo-quality.yml`) and carry
//! forward the intent of Issues #1180 and #1289 (coverage-only workflow, no
//! duplicate fmt/clippy gates), #1567 (checkout must not persist credentials)
//! and #2389 (the Codecov upload must never fail the job, while the
//! coverage-generation step must still fail on real test failures).
//!
//! Each check is a pure function over the workflow text (`&str`), so it can be
//! exercised against fixtures. The `real_workflow_*` tests run every check
//! against `.github/workflows/cargo-quality.yml`. The workflow is parsed as
//! plain text (no YAML crate is in the dependency tree), matching the sibling
//! workflow tests such as `issue_1656_cargo_quality_milestone_filter.rs`.
//!
//! Every check fails closed: a missing key or step name is an error, not a
//! pass.

use std::fs;
use std::path::Path;

const WORKFLOW_RELATIVE_PATH: &str = ".github/workflows/cargo-quality.yml";

/// Indentation of a step-list item (`      - `) inside `steps:`.
const STEP_ITEM_PREFIX: &str = "      - ";

type Check = fn(&str) -> Result<(), String>;

fn is_comment_line(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

/// Collapses runs of whitespace so checks do not depend on spacing.
fn normalise(line: &str) -> String {
    line.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Normalised non-comment lines. Comment-only lines are dropped so that
/// documentation mentioning a removed step does not trip a check.
fn code_lines(body: &str) -> Vec<String> {
    body.lines()
        .filter(|line| !is_comment_line(line))
        .map(normalise)
        .collect()
}

/// Fails unless some non-comment line contains `needle`.
fn require_code_line(body: &str, needle: &str) -> Result<(), String> {
    if code_lines(body).iter().any(|line| line.contains(needle)) {
        Ok(())
    } else {
        Err(format!("no non-comment line contains `{needle}`"))
    }
}

/// Fails if any non-comment line contains `needle`.
fn forbid_code_line(body: &str, needle: &str) -> Result<(), String> {
    match code_lines(body)
        .into_iter()
        .find(|line| line.contains(needle))
    {
        Some(line) => Err(format!("non-comment line contains `{needle}`: `{line}`")),
        None => Ok(()),
    }
}

/// Splits the `steps:` list into step blocks. A step starts at a line with
/// exactly six spaces followed by `- ` and runs until the next such line or
/// until the list ends (a non-blank line indented less than six spaces).
/// Steps may open with `uses:` rather than `name:`.
fn step_blocks(body: &str) -> Result<Vec<Vec<&str>>, String> {
    let lines: Vec<&str> = body.lines().collect();
    let steps_at = lines
        .iter()
        .position(|line| line.trim() == "steps:")
        .ok_or_else(|| "workflow has no `steps:` key".to_string())?;

    let mut blocks: Vec<Vec<&str>> = Vec::new();
    for &line in &lines[steps_at + 1..] {
        if line.trim().is_empty() || is_comment_line(line) {
            if let Some(block) = blocks.last_mut() {
                block.push(line);
            }
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if indent < 6 {
            break;
        }
        if line.starts_with(STEP_ITEM_PREFIX) {
            blocks.push(vec![line]);
        } else if let Some(block) = blocks.last_mut() {
            block.push(line);
        }
    }
    Ok(blocks)
}

/// Returns the step whose header line is exactly `- name: <name>`. A missing
/// name is an error, so a renamed step cannot silently skip its check.
fn named_step<'a>(blocks: &[Vec<&'a str>], name: &str) -> Result<Vec<&'a str>, String> {
    let header = format!("- name: {name}");
    blocks
        .iter()
        .find(|block| {
            block
                .first()
                .is_some_and(|first| normalise(first) == header)
        })
        .cloned()
        .ok_or_else(|| format!("workflow has no step named `{name}`"))
}

/// True when a non-comment line is `uses: <action_prefix>...`, with or without
/// a leading `- ` list marker.
fn uses_action(line: &str, action_prefix: &str) -> bool {
    if is_comment_line(line) {
        return false;
    }
    let norm = normalise(line);
    let rest = norm.strip_prefix("- ").unwrap_or(norm.as_str());
    rest.strip_prefix("uses: ")
        .is_some_and(|action| action.starts_with(action_prefix))
}

/// Entries of the top-level `permissions:` block, whitespace-normalised.
fn top_level_permissions(body: &str) -> Result<Vec<String>, String> {
    let lines: Vec<&str> = body.lines().collect();
    let start = lines
        .iter()
        .position(|line| line.starts_with("permissions:"))
        .ok_or_else(|| "workflow has no top-level `permissions:` key".to_string())?;

    let mut entries = Vec::new();
    for line in &lines[start + 1..] {
        if line.trim().is_empty() || is_comment_line(line) {
            continue;
        }
        if !line.starts_with(' ') {
            break;
        }
        entries.push(normalise(line));
    }
    Ok(entries)
}

fn check_pull_request_trigger(body: &str) -> Result<(), String> {
    require_code_line(body, "pull_request:")
}

fn check_checkout_disables_persist_credentials(body: &str) -> Result<(), String> {
    let blocks = step_blocks(body)?;
    let checkouts: Vec<&Vec<&str>> = blocks
        .iter()
        .filter(|block| {
            block
                .iter()
                .any(|line| uses_action(line, "actions/checkout@"))
        })
        .collect();
    if checkouts.is_empty() {
        return Err("no step uses `actions/checkout@`".to_string());
    }
    // The setting is checked within each checkout step's own block, so a
    // `persist-credentials: false` in some other step does not satisfy it.
    for block in checkouts {
        let disabled = code_lines(&block.join("\n"))
            .iter()
            .any(|line| line == "persist-credentials: false");
        if !disabled {
            return Err(
                "the `actions/checkout@` step must set `persist-credentials: false` in its own \
                 `with:` block (Issue #1567)"
                    .to_string(),
            );
        }
    }
    Ok(())
}

fn check_top_level_permissions_contents_read(body: &str) -> Result<(), String> {
    let entries = top_level_permissions(body)?;
    if entries.iter().any(|entry| entry == "contents: read") {
        Ok(())
    } else {
        Err("top-level `permissions:` must include `contents: read`".to_string())
    }
}

fn check_no_write_permission(body: &str) -> Result<(), String> {
    match code_lines(body)
        .into_iter()
        .find(|line| line.contains("write"))
    {
        Some(line) => Err(format!("non-comment line grants write access: `{line}`")),
        None => Ok(()),
    }
}

fn check_rust_installed_via_setup_rust(body: &str) -> Result<(), String> {
    require_code_line(body, "uses: ./.github/actions/setup-rust")
}

fn check_no_dtolnay_rust_toolchain(body: &str) -> Result<(), String> {
    forbid_code_line(body, "dtolnay/rust-toolchain")
}

fn check_cargo_llvm_cov_installed(body: &str) -> Result<(), String> {
    require_code_line(body, "cargo-llvm-cov")
}

fn check_codecov_action_used(body: &str) -> Result<(), String> {
    require_code_line(body, "codecov/codecov-action@")
}

/// Issue #1289: the fmt and clippy gates are owned by `ci.yml/quality` and must
/// not be re-run here.
fn check_no_cargo_fmt_or_clippy(body: &str) -> Result<(), String> {
    forbid_code_line(body, "cargo fmt")?;
    forbid_code_line(body, "cargo clippy")
}

/// Issue #2389: the Codecov upload step must be non-blocking.
fn check_codecov_upload_continues_on_error(body: &str) -> Result<(), String> {
    let block = named_step(&step_blocks(body)?, "Upload coverage to Codecov")?;
    let continues = code_lines(&block.join("\n"))
        .iter()
        .any(|line| line == "continue-on-error: true");
    if continues {
        Ok(())
    } else {
        Err(
            "step `Upload coverage to Codecov` must set `continue-on-error: true` (Issue #2389)"
                .to_string(),
        )
    }
}

/// Issue #2389: the coverage-generation step must still fail the job on a real
/// test failure, so `continue-on-error` must not appear on it.
fn check_generate_coverage_fails_job_on_error(body: &str) -> Result<(), String> {
    let block = named_step(&step_blocks(body)?, "Generate coverage (lcov)")?;
    match code_lines(&block.join("\n"))
        .into_iter()
        .find(|line| line.starts_with("continue-on-error:"))
    {
        Some(line) => Err(format!(
            "step `Generate coverage (lcov)` must not set `continue-on-error` (real test failures \
             must fail the job): `{line}`"
        )),
        None => Ok(()),
    }
}

fn all_checks() -> Vec<(&'static str, Check)> {
    vec![
        ("pull_request trigger", check_pull_request_trigger as Check),
        (
            "checkout disables persist-credentials",
            check_checkout_disables_persist_credentials as Check,
        ),
        (
            "top-level permissions contents: read",
            check_top_level_permissions_contents_read as Check,
        ),
        ("no write permission", check_no_write_permission as Check),
        (
            "Rust installed via setup-rust",
            check_rust_installed_via_setup_rust as Check,
        ),
        (
            "no dtolnay/rust-toolchain",
            check_no_dtolnay_rust_toolchain as Check,
        ),
        (
            "cargo-llvm-cov installed",
            check_cargo_llvm_cov_installed as Check,
        ),
        ("codecov action used", check_codecov_action_used as Check),
        (
            "no cargo fmt or clippy",
            check_no_cargo_fmt_or_clippy as Check,
        ),
        (
            "Codecov upload continues on error",
            check_codecov_upload_continues_on_error as Check,
        ),
        (
            "coverage generation fails the job",
            check_generate_coverage_fails_job_on_error as Check,
        ),
    ]
}

fn read_real_workflow() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(WORKFLOW_RELATIVE_PATH);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn assert_ok(result: Result<(), String>) {
    if let Err(message) = result {
        panic!("{message}");
    }
}

// --- Real-file tests -------------------------------------------------------

#[test]
fn real_workflow_triggers_on_pull_request() {
    assert_ok(check_pull_request_trigger(&read_real_workflow()));
}

#[test]
fn real_workflow_checkout_does_not_persist_credentials() {
    assert_ok(check_checkout_disables_persist_credentials(
        &read_real_workflow(),
    ));
}

#[test]
fn real_workflow_grants_contents_read() {
    assert_ok(check_top_level_permissions_contents_read(
        &read_real_workflow(),
    ));
}

#[test]
fn real_workflow_grants_no_write_permission() {
    assert_ok(check_no_write_permission(&read_real_workflow()));
}

#[test]
fn real_workflow_installs_rust_via_setup_rust_action() {
    assert_ok(check_rust_installed_via_setup_rust(&read_real_workflow()));
}

#[test]
fn real_workflow_does_not_use_dtolnay_rust_toolchain() {
    assert_ok(check_no_dtolnay_rust_toolchain(&read_real_workflow()));
}

#[test]
fn real_workflow_installs_cargo_llvm_cov() {
    assert_ok(check_cargo_llvm_cov_installed(&read_real_workflow()));
}

#[test]
fn real_workflow_uses_codecov_action() {
    assert_ok(check_codecov_action_used(&read_real_workflow()));
}

#[test]
fn real_workflow_runs_no_cargo_fmt_or_clippy() {
    assert_ok(check_no_cargo_fmt_or_clippy(&read_real_workflow()));
}

#[test]
fn real_workflow_codecov_upload_continues_on_error() {
    assert_ok(check_codecov_upload_continues_on_error(
        &read_real_workflow(),
    ));
}

#[test]
fn real_workflow_coverage_generation_still_fails_job_on_error() {
    assert_ok(check_generate_coverage_fails_job_on_error(
        &read_real_workflow(),
    ));
}

// --- Fixture tests ---------------------------------------------------------

/// A minimal workflow that satisfies every check. Negative tests mutate it one
/// property at a time, so each failure is attributable to a single edit.
const COMPLIANT: &str = r#"name: Coverage

on:
  pull_request:
    branches: ["*"]

permissions:
  contents: read

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          persist-credentials: false

      - name: Install Rust toolchain
        uses: ./.github/actions/setup-rust

      - name: Install cargo-llvm-cov
        uses: taiki-e/install-action@7be9fd86bd1707236395105d6e9329dd1511a7e1 # v2.79.0
        with:
          tool: cargo-llvm-cov

      # Comments may mention cargo clippy and dtolnay/rust-toolchain without tripping a check.
      - name: Generate coverage (lcov)
        run: cargo llvm-cov --all-features --lcov --output-path lcov.info -- --test-threads=2

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@0f8570b1a125f4937846a11fcfa3bcd548bd8c97 # v4.6.0
        continue-on-error: true
        with:
          files: lcov.info
          fail_ci_if_error: false
"#;

const CHECKOUT_LINE: &str =
    "      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2\n";
const CHECKOUT_WITH: &str = "        with:\n          persist-credentials: false\n";
const SETUP_RUST_LINE: &str = "        uses: ./.github/actions/setup-rust\n";
const GENERATE_RUN_PREFIX: &str = "        run: cargo llvm-cov";
const CODECOV_HEADER: &str = "      - name: Upload coverage to Codecov\n";
const CODECOV_CONTINUE: &str = "        continue-on-error: true\n";
const PERMISSIONS_BLOCK: &str = "permissions:\n  contents: read\n";

/// Replaces exactly one occurrence, failing loudly if the fixture drifted so a
/// no-op mutation can never make a negative test pass vacuously.
fn replace_once(body: &str, from: &str, to: &str) -> String {
    assert!(body.contains(from), "fixture no longer contains {from:?}");
    body.replacen(from, to, 1)
}

#[test]
fn compliant_fixture_passes_every_check() {
    for (name, check) in all_checks() {
        assert!(
            check(COMPLIANT).is_ok(),
            "baseline fixture must pass `{name}`: {:?}",
            check(COMPLIANT)
        );
    }
}

#[test]
fn checkout_without_persist_false_fails_even_when_another_step_sets_it() {
    let checkout_unset = replace_once(COMPLIANT, CHECKOUT_WITH, "");
    let leaked = replace_once(
        &checkout_unset,
        SETUP_RUST_LINE,
        &format!("{SETUP_RUST_LINE}{CHECKOUT_WITH}"),
    );
    assert!(check_checkout_disables_persist_credentials(&leaked).is_err());
}

#[test]
fn checkout_with_persist_false_passes_for_the_same_fixture() {
    let checkout_unset = replace_once(COMPLIANT, CHECKOUT_WITH, "");
    let leaked = replace_once(
        &checkout_unset,
        SETUP_RUST_LINE,
        &format!("{SETUP_RUST_LINE}{CHECKOUT_WITH}"),
    );
    // Make only the probed value legal: the checkout step itself disables it.
    let fixed = replace_once(
        &leaked,
        CHECKOUT_LINE,
        &format!("{CHECKOUT_LINE}{CHECKOUT_WITH}"),
    );
    assert_ok(check_checkout_disables_persist_credentials(&fixed));
}

#[test]
fn checkout_persist_credentials_true_fails() {
    let body = replace_once(
        COMPLIANT,
        "persist-credentials: false",
        "persist-credentials: true",
    );
    assert!(check_checkout_disables_persist_credentials(&body).is_err());
}

#[test]
fn missing_checkout_step_fails_closed() {
    let body = replace_once(COMPLIANT, "uses: actions/checkout@", "uses: actions/cache@");
    assert!(check_checkout_disables_persist_credentials(&body).is_err());
}

#[test]
fn cargo_clippy_in_run_line_fails() {
    let body = replace_once(
        COMPLIANT,
        CODECOV_HEADER,
        &format!(
            "      - name: Lint\n        run: cargo clippy -- -D warnings\n\n{CODECOV_HEADER}"
        ),
    );
    let err = check_no_cargo_fmt_or_clippy(&body).expect_err("clippy in a run line must fail");
    assert!(err.contains("cargo clippy"), "unexpected message: {err}");
}

#[test]
fn cargo_fmt_check_with_irregular_spacing_fails() {
    let body = replace_once(
        COMPLIANT,
        CODECOV_HEADER,
        &format!("      - name: Format\n        run: cargo   fmt --check\n\n{CODECOV_HEADER}"),
    );
    assert!(check_no_cargo_fmt_or_clippy(&body).is_err());
}

#[test]
fn cargo_clippy_mentioned_only_in_comment_passes() {
    let body = replace_once(
        COMPLIANT,
        CODECOV_HEADER,
        &format!("      # cargo clippy now runs in ci.yml/quality.\n{CODECOV_HEADER}"),
    );
    assert_ok(check_no_cargo_fmt_or_clippy(&body));
}

#[test]
fn codecov_step_without_continue_on_error_fails() {
    let body = replace_once(COMPLIANT, CODECOV_CONTINUE, "");
    assert!(check_codecov_upload_continues_on_error(&body).is_err());
}

#[test]
fn codecov_step_with_continue_on_error_false_fails() {
    let body = replace_once(
        COMPLIANT,
        CODECOV_CONTINUE,
        "        continue-on-error: false\n",
    );
    assert!(check_codecov_upload_continues_on_error(&body).is_err());
}

#[test]
fn continue_on_error_on_generate_step_fails() {
    let body = replace_once(
        COMPLIANT,
        GENERATE_RUN_PREFIX,
        &format!("        continue-on-error: true\n{GENERATE_RUN_PREFIX}"),
    );
    assert!(check_generate_coverage_fails_job_on_error(&body).is_err());
}

#[test]
fn dtolnay_rust_toolchain_present_fails() {
    let body = replace_once(
        COMPLIANT,
        SETUP_RUST_LINE,
        "        uses: dtolnay/rust-toolchain@stable\n",
    );
    assert!(check_no_dtolnay_rust_toolchain(&body).is_err());
}

#[test]
fn missing_setup_rust_action_fails() {
    let body = replace_once(COMPLIANT, SETUP_RUST_LINE, "");
    assert!(check_rust_installed_via_setup_rust(&body).is_err());
}

#[test]
fn renamed_codecov_step_fails_closed() {
    let body = replace_once(
        COMPLIANT,
        CODECOV_HEADER,
        "      - name: Upload to Codecov\n",
    );
    let err = check_codecov_upload_continues_on_error(&body)
        .expect_err("a missing step name must fail closed");
    assert!(
        err.contains("Upload coverage to Codecov"),
        "unexpected message: {err}"
    );
}

#[test]
fn renamed_generate_step_fails_closed() {
    let body = replace_once(
        COMPLIANT,
        "      - name: Generate coverage (lcov)\n",
        "      - name: Generate coverage\n",
    );
    let err = check_generate_coverage_fails_job_on_error(&body)
        .expect_err("a missing step name must fail closed");
    assert!(
        err.contains("Generate coverage (lcov)"),
        "unexpected message: {err}"
    );
}

#[test]
fn missing_steps_key_fails_closed() {
    let body = replace_once(COMPLIANT, "    steps:\n", "    step-list:\n");
    assert!(check_checkout_disables_persist_credentials(&body).is_err());
    assert!(check_codecov_upload_continues_on_error(&body).is_err());
}

#[test]
fn pull_request_trigger_absent_fails() {
    let body = replace_once(COMPLIANT, "  pull_request:\n", "  push:\n");
    assert!(check_pull_request_trigger(&body).is_err());
}

#[test]
fn write_permission_fails() {
    let body = replace_once(
        COMPLIANT,
        PERMISSIONS_BLOCK,
        "permissions:\n  contents: read\n  issues: write\n",
    );
    assert!(check_no_write_permission(&body).is_err());
    assert_ok(check_top_level_permissions_contents_read(&body));
}

#[test]
fn contents_read_absent_fails() {
    let body = replace_once(
        COMPLIANT,
        PERMISSIONS_BLOCK,
        "permissions:\n  contents: write\n",
    );
    assert!(check_top_level_permissions_contents_read(&body).is_err());
}

#[test]
fn top_level_permissions_absent_fails_closed() {
    let body = replace_once(COMPLIANT, PERMISSIONS_BLOCK, "");
    assert!(check_top_level_permissions_contents_read(&body).is_err());
}

#[test]
fn cargo_llvm_cov_and_codecov_absent_fail() {
    // The step name also contains `cargo-llvm-cov`, so both occurrences must go.
    let no_llvm_cov = replace_once(COMPLIANT, "tool: cargo-llvm-cov", "tool: grcov");
    let no_llvm_cov = replace_once(
        &no_llvm_cov,
        "name: Install cargo-llvm-cov",
        "name: Install grcov",
    );
    assert!(check_cargo_llvm_cov_installed(&no_llvm_cov).is_err());
    let no_codecov = replace_once(
        COMPLIANT,
        "uses: codecov/codecov-action@",
        "uses: codecov/other-action@",
    );
    assert!(check_codecov_action_used(&no_codecov).is_err());
}
