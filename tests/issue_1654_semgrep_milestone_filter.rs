//! Issue #1654: the Semgrep workflow (`semgrep.yml`) declared a
//! `pull_request: branches: ["*"]` filter. In GitHub Actions branch
//! filters the `*` glob does not cross `/`, so `"*"` matches single-level
//! branch names (e.g. `Develop`) but never `milestone/<slug>`. Milestone
//! sub-issue PRs target a shared `milestone/<name>` branch, so the SAST
//! gate never ran on those PRs and they merged into the milestone branch
//! unscanned.
//!
//! The fix adds `milestone/*` alongside the existing glob so milestone PRs
//! are gated too. This test parses `semgrep.yml` as plain text (no YAML
//! parser is in the dependency tree, matching the sibling workflow tests)
//! and asserts the `pull_request:` branch filter both keeps the original
//! `"*"` glob and includes `milestone/*`.

use std::fs;
use std::path::{Path, PathBuf};

fn workflows_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows")
}

fn read_workflow(file_name: &str) -> String {
    let path = workflows_dir().join(file_name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Return the value of the `branches:` line inside the `pull_request:`
/// trigger of the given workflow body, trimmed of surrounding whitespace.
fn pull_request_branches(body: &str) -> Option<String> {
    let mut in_pull_request = false;
    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("pull_request:") {
            in_pull_request = true;
            continue;
        }
        // A comment or blank line inside the block does not end it.
        if in_pull_request && (trimmed.is_empty() || trimmed.starts_with('#')) {
            continue;
        }
        if in_pull_request {
            if let Some(rest) = trimmed.strip_prefix("branches:") {
                return Some(rest.trim().to_string());
            }
            // Any other same- or lower-indent key that is not `branches:`
            // is fine to skip; keep scanning within the trigger. A new
            // top-level trigger (2-space indent, ends with `:`) ends it.
            if line.starts_with("  ")
                && !line.starts_with("    ")
                && trimmed.ends_with(':')
                && !trimmed.starts_with("pull_request:")
            {
                in_pull_request = false;
            }
        }
    }
    None
}

#[test]
fn semgrep_pull_request_filter_matches_milestone_branches() {
    let body = read_workflow("semgrep.yml");
    let branches = pull_request_branches(&body)
        .expect("semgrep.yml must declare a `pull_request:` `branches:` filter");
    assert!(
        branches.contains("milestone/*"),
        "semgrep.yml `pull_request` branch filter must include \
         `milestone/*` so milestone sub-issue PRs are SAST-gated \
         (Issue #1654). Found: {branches}",
    );
}

#[test]
fn semgrep_pull_request_filter_keeps_wildcard() {
    // The milestone glob is additive — single-level branches (Develop and
    // friends) must still be gated.
    let body = read_workflow("semgrep.yml");
    let branches = pull_request_branches(&body)
        .expect("semgrep.yml must declare a `pull_request:` `branches:` filter");
    assert!(
        branches.contains("\"*\"") || branches.contains("'*'") || branches.contains("\"**\""),
        "semgrep.yml must keep the single-level `\"*\"` glob (or `\"**\"`) \
         so non-milestone PRs stay gated (Issue #1654). Found: {branches}",
    );
}
