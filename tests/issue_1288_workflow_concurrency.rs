//! Issue #1288: every PR / branch workflow in `.github/workflows/` must
//! declare a workflow-level `concurrency:` group that cancels superseded
//! runs on the same ref. Without one, force-pushing a PR or landing
//! several commits to `Develop` in quick succession leaves older runs
//! churning while newer runs start — wasting runner capacity and, in
//! `ci.yml`'s case, racing the `version-increment` job.
//!
//! The canonical group is `${{ github.workflow }}-${{ github.ref }}`
//! with `cancel-in-progress: true`.
//!
//! Reusable workflows (`security.yml`, invoked via `workflow_call`) are
//! excluded — concurrency must live on the caller workflow, not the
//! reusable callee.
//!
//! This test parses each workflow file as plain text (no YAML parser
//! is in the dependency tree) and asserts that a top-level
//! `concurrency:` block exists with the expected `group:` expression
//! and `cancel-in-progress: true`.

use std::fs;
use std::path::{Path, PathBuf};

fn workflows_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows")
}

fn read_workflow(file_name: &str) -> String {
    let path = workflows_dir().join(file_name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Locate the top-level `concurrency:` block and return its body up to
/// the next top-level key or end of file. Top-level keys start at
/// column zero and end with `:`.
fn concurrency_block(body: &str) -> Option<&str> {
    let header = "\nconcurrency:";
    let start = body.find(header)? + 1; // skip the leading newline
    let after = &body[start..];
    // Find the next top-level key (line starting with a non-space and
    // ending with `:`), skipping the `concurrency:` header line itself.
    let mut search_from = "concurrency:".len();
    while let Some(nl) = after[search_from..].find('\n') {
        let line_start = search_from + nl + 1;
        if line_start >= after.len() {
            return Some(after);
        }
        let line_end = after[line_start..]
            .find('\n')
            .map_or(after.len(), |n| line_start + n);
        let line = &after[line_start..line_end];
        // A sibling top-level key: non-space first char, ends with `:`,
        // and is not a list item or comment.
        if !line.is_empty()
            && !line.starts_with(' ')
            && !line.starts_with('#')
            && line.trim_end().ends_with(':')
        {
            return Some(&after[..line_start]);
        }
        search_from = line_start;
    }
    Some(after)
}

const CANONICAL_GROUP: &str = "${{ github.workflow }}-${{ github.ref }}";

fn assert_canonical_concurrency(file_name: &str) {
    let body = read_workflow(file_name);
    let block = concurrency_block(&body).unwrap_or_else(|| {
        panic!(
            "{file_name} must declare a top-level `concurrency:` block \
             (Issue #1288)"
        )
    });
    assert!(
        block.contains(CANONICAL_GROUP),
        "{file_name} concurrency block must use the canonical group \
         expression `{CANONICAL_GROUP}` so superseded runs on the same \
         ref are cancelled (Issue #1288). Found:\n{block}",
    );
    assert!(
        block.contains("cancel-in-progress: true"),
        "{file_name} concurrency block must set \
         `cancel-in-progress: true` so superseded runs are actually \
         cancelled (Issue #1288). Found:\n{block}",
    );
}

#[test]
fn ci_yml_declares_concurrency_group() {
    assert_canonical_concurrency("ci.yml");
}

#[test]
fn cargo_quality_yml_declares_concurrency_group() {
    assert_canonical_concurrency("cargo-quality.yml");
}

#[test]
fn gitleaks_yml_declares_concurrency_group() {
    assert_canonical_concurrency("gitleaks.yml");
}

#[test]
fn markdown_lint_yml_declares_concurrency_group() {
    assert_canonical_concurrency("markdown-lint.yml");
}

#[test]
fn semgrep_yml_declares_concurrency_group() {
    assert_canonical_concurrency("semgrep.yml");
}

#[test]
fn shellcheck_yml_declares_concurrency_group() {
    assert_canonical_concurrency("shellcheck.yml");
}
