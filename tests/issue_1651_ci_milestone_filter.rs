//! Issue #1651: the CI quality workflow (`ci.yml`) declared a
//! `pull_request:` `branches:` filter listing only `Develop`. Milestone
//! sub-issue PRs target a shared `milestone/<slug>` branch, so with that
//! filter the CI gate never ran on those PRs and they merged into the
//! milestone branch unchecked — the gap was only caught later by the single
//! rollup PR into `Develop`.
//!
//! The fix adds `milestone/*` to the filter so milestone PRs are gated too.
//! Unlike the sibling milestone-filter tests (Issues #1650, #1656), `ci.yml`
//! declares its branches as a YAML block sequence (`- Develop` on its own
//! line), so this test collects the list items rather than reading an inline
//! array. No YAML parser is in the dependency tree, matching the sibling
//! workflow tests, so the parse is plain text.

use std::fs;
use std::path::{Path, PathBuf};

fn workflows_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows")
}

fn read_workflow(file_name: &str) -> String {
    let path = workflows_dir().join(file_name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Collect the branch entries of the `branches:` filter inside the
/// `pull_request:` trigger. Handles both an inline array
/// (`branches: [Develop, milestone/*]`) and a block sequence:
///
/// ```yaml
///   pull_request:
///     branches:
///       - Develop
///       - milestone/*
/// ```
fn pull_request_branches(body: &str) -> Vec<String> {
    let mut in_pull_request = false;
    let mut in_branches = false;
    let mut out = Vec::new();

    for line in body.lines() {
        let trimmed = line.trim_start();

        if !in_pull_request {
            if trimmed.starts_with("pull_request:") {
                in_pull_request = true;
            }
            continue;
        }

        // Blank lines and comments never terminate a block we are scanning.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if in_branches {
            // Block-sequence item belonging to `branches:`.
            if let Some(item) = trimmed.strip_prefix("- ") {
                out.push(item.trim().to_string());
                continue;
            }
            // Any non-list line ends the branches block (and, being a new
            // key, also ends our interest in this trigger).
            break;
        }

        if let Some(rest) = trimmed.strip_prefix("branches:") {
            let rest = rest.trim();
            if rest.is_empty() {
                // Block-sequence form: items follow on subsequent lines.
                in_branches = true;
            } else {
                // Inline-array form: split the bracketed list.
                let inner = rest.trim_start_matches('[').trim_end_matches(']');
                out.extend(
                    inner
                        .split(',')
                        .map(|s| s.trim().trim_matches(['"', '\'']).to_string())
                        .filter(|s| !s.is_empty()),
                );
            }
            continue;
        }

        // A new top-level trigger (2-space indent key) ends the pull_request
        // block before we reached its branches filter.
        if line.starts_with("  ") && !line.starts_with("   ") && trimmed.ends_with(':') {
            break;
        }
    }

    out
}

#[test]
fn ci_pull_request_filter_matches_milestone_branches() {
    let branches = pull_request_branches(&read_workflow("ci.yml"));
    assert!(
        !branches.is_empty(),
        "ci.yml must declare a `pull_request:` `branches:` filter",
    );
    assert!(
        branches.iter().any(|b| b == "milestone/*"),
        "ci.yml `pull_request` branch filter must include `milestone/*` so \
         milestone sub-issue PRs are CI-gated (Issue #1651). Found: {branches:?}",
    );
}

#[test]
fn ci_pull_request_filter_keeps_develop() {
    // The milestone glob is additive — the existing `Develop` gate must stay.
    let branches = pull_request_branches(&read_workflow("ci.yml"));
    assert!(
        branches.iter().any(|b| b == "Develop"),
        "ci.yml must keep the `Develop` branch filter so non-milestone PRs \
         stay gated (Issue #1651). Found: {branches:?}",
    );
}
