//! Issue #1565: `markdown-lint.yml` is a test/lint workflow. It must gate the
//! pull request only — it must NOT trigger on `push:` to the default branch
//! (`Develop`). A post-merge push run only re-runs the lint that already
//! passed on the PR, wasting runner capacity and risking a stray red tick on
//! `Develop`.
//!
//! Deploy/publish/release workflows are different — they must keep firing on
//! push — but a checker gates the PR only. This mirrors the sibling checker
//! workflows (actionlint #1563, ci #1564) which are `pull_request:` only.
//!
//! This test parses `markdown-lint.yml` as plain text (no YAML parser is in
//! the dependency tree, matching the sibling workflow tests) and asserts the
//! top-level `on:` block declares neither a `push:` trigger nor the `Develop`
//! branch under one, while still keeping the `pull_request:` trigger.

use std::fs;
use std::path::{Path, PathBuf};

fn workflows_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows")
}

fn read_workflow(file_name: &str) -> String {
    let path = workflows_dir().join(file_name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Return the top-level `on:` block body, from the `on:` header up to the
/// next top-level key (a line starting at column zero and ending with `:`)
/// or end of file.
fn on_block(body: &str) -> Option<String> {
    let header = "\non:";
    let start = body.find(header)? + 1; // skip the leading newline
    let after = &body[start..];
    let mut search_from = "on:".len();
    while let Some(nl) = after[search_from..].find('\n') {
        let line_start = search_from + nl + 1;
        if line_start >= after.len() {
            return Some(after.to_string());
        }
        let line_end = after[line_start..]
            .find('\n')
            .map_or(after.len(), |n| line_start + n);
        let line = &after[line_start..line_end];
        // A sibling top-level key: non-space first char, ends with `:`,
        // and is neither a list item nor a comment.
        if !line.is_empty()
            && !line.starts_with(' ')
            && !line.starts_with('#')
            && line.trim_end().ends_with(':')
        {
            return Some(after[..line_start].to_string());
        }
        search_from = line_start;
    }
    Some(after.to_string())
}

#[test]
fn markdown_lint_yml_has_no_push_trigger() {
    let body = read_workflow("markdown-lint.yml");
    let block = on_block(&body).expect("markdown-lint.yml must declare a top-level `on:` block");
    let has_push = block.lines().any(|l| l.trim_start().starts_with("push:"));
    assert!(
        !has_push,
        "markdown-lint.yml is a checker workflow and must not trigger on `push:` — \
         it should gate the pull request only (Issue #1565). `on:` block:\n{block}",
    );
}

#[test]
fn markdown_lint_yml_on_block_does_not_reference_develop_push() {
    // Guard against a `push:` sneaking back in that targets `Develop`.
    let body = read_workflow("markdown-lint.yml");
    let block = on_block(&body).expect("markdown-lint.yml must declare a top-level `on:` block");
    let mut in_push = false;
    for line in block.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("push:") {
            in_push = true;
            continue;
        }
        // A new top-level (2-space) trigger key ends the push section.
        if in_push && line.starts_with("  ") && !line.starts_with("    ") && trimmed.ends_with(':')
        {
            in_push = false;
        }
        if in_push {
            assert!(
                !trimmed.contains("Develop"),
                "markdown-lint.yml must not trigger on push to `Develop` (Issue #1565). \
                 Offending line: {line}",
            );
        }
    }
}

#[test]
fn markdown_lint_yml_keeps_pull_request_trigger() {
    let body = read_workflow("markdown-lint.yml");
    let block = on_block(&body).expect("markdown-lint.yml must declare a top-level `on:` block");
    assert!(
        block
            .lines()
            .any(|l| l.trim_start().starts_with("pull_request:")),
        "markdown-lint.yml must keep its `pull_request:` trigger so PRs are still gated \
         (Issue #1565). `on:` block:\n{block}",
    );
}
