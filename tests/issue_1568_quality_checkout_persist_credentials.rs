//! Issue #1568: the `quality` job in `ci.yml` runs `actions/checkout`
//! without `persist-credentials: false`. By default checkout writes the
//! workflow's `GITHUB_TOKEN` into `.git/config`, where any later step in
//! the job — including a compromised dependency — can read it and act as
//! the token. The quality job only reads the tree, builds, and runs the
//! test suite; it never pushes back or fetches a private submodule, so the
//! persisted credential is unnecessary and only widens the blast radius of
//! a compromised step.
//!
//! This test parses `ci.yml` as plain text (no YAML parser is in the
//! dependency tree, matching the sibling workflow tests) and asserts the
//! `quality` job's checkout step sets `persist-credentials: false`.

use std::fs;
use std::path::{Path, PathBuf};

fn workflows_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows")
}

fn read_workflow(file_name: &str) -> String {
    let path = workflows_dir().join(file_name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Return the body of a named job (2-space indented key under `jobs:`),
/// from the job header up to the next sibling job or end of file.
fn job_block(body: &str, job_name: &str) -> Option<String> {
    let header = format!("\n  {job_name}:");
    let start = body.find(&header)? + 1; // skip the leading newline
    let after = &body[start..];
    let mut search_from = header.len() - 1; // past the header line content
    while let Some(nl) = after[search_from..].find('\n') {
        let line_start = search_from + nl + 1;
        if line_start >= after.len() {
            return Some(after.to_string());
        }
        let line_end = after[line_start..]
            .find('\n')
            .map_or(after.len(), |n| line_start + n);
        let line = &after[line_start..line_end];
        // A sibling job: exactly 2-space indent, ends with `:`, not deeper.
        if line.starts_with("  ") && !line.starts_with("   ") && line.trim_end().ends_with(':') {
            return Some(after[..line_start].to_string());
        }
        search_from = line_start;
    }
    Some(after.to_string())
}

#[test]
fn quality_job_checkout_disables_credential_persistence() {
    let body = read_workflow("ci.yml");
    let block = job_block(&body, "quality").expect("ci.yml must declare a `quality` job");
    assert!(
        block.contains("actions/checkout@"),
        "the quality job must still check out the repository (Issue #1568)",
    );
    assert!(
        block
            .lines()
            .any(|l| l.trim() == "persist-credentials: false"),
        "the quality job's checkout must set `persist-credentials: false` so the \
         GITHUB_TOKEN is not written to .git/config (Issue #1568). quality job:\n{block}",
    );
}
