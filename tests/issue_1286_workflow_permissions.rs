//! Issue #1286: `.github/workflows/ci.yml` must declare an explicit
//! top-level `permissions:` block that narrows the default `GITHUB_TOKEN`
//! scope to read-only. GitHub's default token scope is read/write across
//! many surfaces, so unprivileged jobs (cargo build/test, file-presence
//! checks, codespell) inherit unnecessary write capabilities unless the
//! workflow tightens them.
//!
//! Reference: GitHub Actions security-hardening guide — every workflow
//! should declare a minimal top-level `permissions:` block and only
//! upgrade per-job where needed.
//!
//! This test reads `ci.yml` as plain text (no YAML parser is in the
//! dependency tree) and asserts:
//!   1. A top-level `permissions:` block exists at column 0.
//!   2. The block sets `contents: read`.
//!   3. Issue #2102: the jobs that push commits (`version-increment` and
//!      `auto-format` in ci.yml, `family-sync` in family-sync.yml) grant the
//!      `GITHUB_TOKEN` no write scope. They push with the `ACTIONS_PUSH` PAT
//!      through an explicit remote URL and check out with
//!      `persist-credentials: false`, so a write grant would be unused
//!      privilege.

use std::fs;
use std::path::Path;

fn load_workflow(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".github/workflows")
        .join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn load_ci_yml() -> String {
    load_workflow("ci.yml")
}

/// Returns true if `body` contains a top-level (column-0) `permissions:`
/// block whose immediate child mapping declares `contents: read`.
fn has_top_level_contents_read(body: &str) -> bool {
    let lines: Vec<&str> = body.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if *line == "permissions:" {
            // Scan subsequent indented lines (2-space indent) until we
            // leave the block.
            for follow in lines.iter().skip(i + 1) {
                let trimmed = follow.trim_end();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                if !follow.starts_with(' ') {
                    break; // Left the block — back to column 0.
                }
                if follow.trim_start() == "contents: read" {
                    return true;
                }
            }
        }
    }
    false
}

#[test]
fn ci_yml_declares_minimal_top_level_permissions() {
    let body = load_ci_yml();
    assert!(
        has_top_level_contents_read(&body),
        "ci.yml must declare a top-level `permissions:` block with \
         `contents: read` to narrow the default GITHUB_TOKEN scope \
         (Issue #1286)",
    );
}

/// Locate a job block by name (`<job>:` at four-space indent under
/// `jobs:`) and return the block body up to the next sibling job or end
/// of file.
fn job_block<'a>(body: &'a str, job_name: &str) -> Option<&'a str> {
    let header = format!("  {job_name}:");
    let start = body.find(&header)?;
    let after = &body[start..];
    // The next sibling job header begins with `  ` and ends with `:`
    // at the same indent. Find the next line that starts with two
    // spaces followed by a non-space character (sibling indent), past
    // the first line.
    let mut search_from = header.len();
    let bytes = after.as_bytes();
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

/// Returns the scopes a job block grants `write`, e.g. `["contents"]`.
fn write_scopes(block: &str) -> Vec<&str> {
    block
        .lines()
        .filter_map(|line| line.trim().strip_suffix(": write"))
        .collect()
}

#[test]
fn pat_pushing_jobs_grant_github_token_no_write() {
    let jobs = [
        ("ci.yml", "version-increment"),
        ("ci.yml", "auto-format"),
        ("family-sync.yml", "family-sync"),
    ];
    for (file, job_name) in jobs {
        let body = load_workflow(file);
        let block = job_block(&body, job_name)
            .unwrap_or_else(|| panic!("job `{job_name}` not found in {file}"));
        // Precondition: the push goes through the PAT, which is what makes a
        // GITHUB_TOKEN write grant redundant.
        assert!(
            block.contains("secrets.ACTIONS_PUSH") && block.contains("persist-credentials: false"),
            "job `{job_name}` in {file} must push via the ACTIONS_PUSH PAT with \
             `persist-credentials: false` (Issue #1868)",
        );
        assert!(
            write_scopes(block).is_empty(),
            "job `{job_name}` in {file} pushes with the ACTIONS_PUSH PAT, so it \
             must not grant the GITHUB_TOKEN write scopes {:?} (Issue #2102)",
            write_scopes(block),
        );
    }
}

#[test]
fn write_scopes_detects_write_grants() {
    let block = "  job:\n    permissions:\n      contents: write\n      pull-requests: write\n      issues: read\n";
    assert_eq!(write_scopes(block), vec!["contents", "pull-requests"]);
    assert!(write_scopes("  job:\n    permissions:\n      contents: read\n").is_empty());
    assert!(write_scopes("").is_empty());
}
