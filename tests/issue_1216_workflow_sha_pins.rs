//! Issue #1216: All third-party GitHub Actions across this repository's
//! workflows must be pinned to immutable 40-character commit SHAs, not
//! mutable tags or branches (e.g. `@v4`, `@stable`,
//! `@cargo-llvm-cov`).
//!
//! Mutable refs are silently re-pointable by an attacker that compromises
//! the upstream action repository (cf. the `tj-actions/changed-files`
//! incident, March 2025) and would execute in this repository's CI with
//! access to `GITHUB_TOKEN` and the `ACTIONS_PUSH` PAT.
//!
//! This test reads every `*.yml` file under `.github/workflows/` and
//! asserts that the ref after `@` on each `uses:` line is a 40-character
//! lower-case hexadecimal SHA. Local action references (`./<path>`) are
//! ignored.

use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn all_workflow_actions_are_pinned_to_commit_sha() {
    let workflows_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows");
    assert!(
        workflows_dir.is_dir(),
        "expected .github/workflows directory at {workflows_dir:?}"
    );

    let mut yml_files: Vec<PathBuf> = fs::read_dir(&workflows_dir)
        .expect("read .github/workflows")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("yml"))
        .collect();
    yml_files.sort();
    assert!(!yml_files.is_empty(), "no workflow files found");

    let mut violations: Vec<String> = Vec::new();

    for path in &yml_files {
        let contents = fs::read_to_string(path).expect("read workflow file");
        for (idx, raw_line) in contents.lines().enumerate() {
            let lineno = idx + 1;
            // Strip everything from the first '#' that is not inside the
            // value before the comment so we ignore commented-out
            // examples.
            let mut after = raw_line.trim_start();
            if after.starts_with('#') {
                continue;
            }
            // Allow leading "- " for list items.
            if let Some(rest) = after.strip_prefix("- ") {
                after = rest.trim_start();
            }
            let Some(rest) = after.strip_prefix("uses:") else {
                continue;
            };
            // Strip trailing comment for parsing the value.
            let value_with_comment = rest.trim();
            let value = value_with_comment
                .split('#')
                .next()
                .unwrap_or("")
                .trim()
                .trim_matches('"')
                .trim_matches('\'');
            if value.is_empty() {
                continue;
            }
            // Local references (reusable workflows in this repo) are
            // not subject to SHA pinning — they are resolved from the
            // committed file tree.
            if value.starts_with("./") {
                continue;
            }
            let Some((_repo, reference)) = value.rsplit_once('@') else {
                violations.push(format!(
                    "{}:{}: malformed `uses:` value `{}` — no `@<ref>` suffix",
                    path.display(),
                    lineno,
                    value
                ));
                continue;
            };
            if !is_commit_sha(reference) {
                violations.push(format!(
                    "{}:{}: `{}` is not pinned to a 40-character commit SHA \
                     (Issue #1216)",
                    path.display(),
                    lineno,
                    value
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "GitHub Actions must be pinned to immutable commit SHAs \
         (Issue #1216):\n{}",
        violations.join("\n")
    );
}

fn is_commit_sha(reference: &str) -> bool {
    reference.len() == 40 && reference.bytes().all(|b| b.is_ascii_hexdigit())
}
