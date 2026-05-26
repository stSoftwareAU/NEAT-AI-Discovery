//! Issue #1287: every job declared across `.github/workflows/` must set
//! an explicit `timeout-minutes:` field. Without one, a wedged step
//! holds a runner for the GitHub default (6 hours / 360 minutes),
//! tying up scarce CI capacity. An explicit cap surfaces hangs as
//! fast failures.
//!
//! This test parses each workflow file as plain text (no YAML parser
//! is in the dependency tree), locates every job block under `jobs:`,
//! and asserts that the block contains a `timeout-minutes:` field
//! before the next sibling job or end of file.
//!
//! Reusable workflows (the `security` job in `ci.yml` that delegates
//! to `security.yml` via `uses:`) are excluded — they have no
//! `runs-on:` and `timeout-minutes:` is set inside the reusable file.

use std::fs;
use std::path::{Path, PathBuf};

fn workflows_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows")
}

fn read_workflow(file_name: &str) -> String {
    let path = workflows_dir().join(file_name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Locate a job block by name (`  <job>:` at two-space indent under
/// `jobs:`) and return the block body up to the next sibling job or end
/// of file.
fn job_block<'a>(body: &'a str, job_name: &str) -> Option<&'a str> {
    let header = format!("  {job_name}:");
    let start = body.find(&header)?;
    let after = &body[start..];
    let bytes = after.as_bytes();
    let mut search_from = header.len();
    while search_from < bytes.len() {
        if let Some(nl) = after[search_from..].find('\n') {
            let line_start = search_from + nl + 1;
            if line_start >= bytes.len() {
                return Some(&after[..bytes.len()]);
            }
            let line_end = after[line_start..]
                .find('\n')
                .map_or(bytes.len(), |n| line_start + n);
            let line = &after[line_start..line_end];
            // Sibling job: exactly two spaces then a non-space char and
            // ends with `:`.
            if line.starts_with("  ")
                && line.as_bytes().get(2).is_some_and(|b| *b != b' ')
                && line.trim_end().ends_with(':')
            {
                return Some(&after[..line_start]);
            }
            search_from = line_start;
        } else {
            return Some(after);
        }
    }
    Some(after)
}

/// Returns true if `block` contains a `timeout-minutes:` line at the
/// expected per-job indent (four spaces).
fn has_timeout_minutes(block: &str) -> bool {
    block.lines().any(|line| {
        line.trim_start() == line.trim_start_matches(' ')
            && line.starts_with("    timeout-minutes:")
    })
}

#[test]
fn ci_yml_jobs_declare_timeout_minutes() {
    let body = read_workflow("ci.yml");
    // The `security` job in ci.yml is a reusable-workflow caller
    // (no runs-on); its timeout-minutes lives in security.yml.
    for job_name in [
        "version-increment",
        "quality",
        "validation",
        "auto-format",
        "spell-check",
    ] {
        let block = job_block(&body, job_name)
            .unwrap_or_else(|| panic!("job `{job_name}` not found in ci.yml"));
        assert!(
            has_timeout_minutes(block),
            "job `{job_name}` in ci.yml must declare `timeout-minutes:` \
             to cap exposure to wedged runners (Issue #1287)",
        );
    }
}

#[test]
fn cargo_quality_yml_job_declares_timeout_minutes() {
    let body = read_workflow("cargo-quality.yml");
    let block = job_block(&body, "quality").expect("job `quality` not found in cargo-quality.yml");
    assert!(
        has_timeout_minutes(block),
        "job `quality` in cargo-quality.yml must declare \
         `timeout-minutes:` (Issue #1287)",
    );
}

#[test]
fn gitleaks_yml_job_declares_timeout_minutes() {
    let body = read_workflow("gitleaks.yml");
    let block = job_block(&body, "gitleaks").expect("job `gitleaks` not found in gitleaks.yml");
    assert!(
        has_timeout_minutes(block),
        "job `gitleaks` in gitleaks.yml must declare \
         `timeout-minutes:` (Issue #1287)",
    );
}

#[test]
fn markdown_lint_yml_job_declares_timeout_minutes() {
    let body = read_workflow("markdown-lint.yml");
    let block = job_block(&body, "markdownlint")
        .expect("job `markdownlint` not found in markdown-lint.yml");
    assert!(
        has_timeout_minutes(block),
        "job `markdownlint` in markdown-lint.yml must declare \
         `timeout-minutes:` (Issue #1287)",
    );
}

#[test]
fn security_yml_job_declares_timeout_minutes() {
    let body = read_workflow("security.yml");
    let block = job_block(&body, "security").expect("job `security` not found in security.yml");
    assert!(
        has_timeout_minutes(block),
        "job `security` in security.yml must declare \
         `timeout-minutes:` (Issue #1287)",
    );
}

#[test]
fn semgrep_yml_job_declares_timeout_minutes() {
    let body = read_workflow("semgrep.yml");
    let block = job_block(&body, "semgrep").expect("job `semgrep` not found in semgrep.yml");
    assert!(
        has_timeout_minutes(block),
        "job `semgrep` in semgrep.yml must declare \
         `timeout-minutes:` (Issue #1287)",
    );
}

#[test]
fn shellcheck_yml_job_declares_timeout_minutes() {
    let body = read_workflow("shellcheck.yml");
    let block =
        job_block(&body, "shellcheck").expect("job `shellcheck` not found in shellcheck.yml");
    assert!(
        has_timeout_minutes(block),
        "job `shellcheck` in shellcheck.yml must declare \
         `timeout-minutes:` (Issue #1287)",
    );
}
