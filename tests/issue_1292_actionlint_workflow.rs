//! Issue #1292: A CI lint gate for GitHub Actions workflows must invoke
//! `actionlint` (upstream: rhysd/actionlint —
//! <https://github.com/rhysd/actionlint>) on every push and pull request
//! so that workflow lint regressions fail the build.
//!
//! This test reads `.github/workflows/actionlint.yml` and asserts:
//!   1. The workflow file exists.
//!   2. It triggers on both `pull_request` and `push`.
//!   3. It declares a top-level minimal `permissions:` block with
//!      `contents: read` (Issue #1286).
//!   4. It pins the third-party `raven-actions/actionlint` step to a
//!      40-character commit SHA (Issue #1216).
//!   5. It declares a per-job `timeout-minutes:` (Issue #1287).
//!   6. It declares a `concurrency:` block that cancels superseded runs
//!      on the same ref (Issue #1288).

use std::fs;
use std::path::Path;

fn load_workflow() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/actionlint.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn actionlint_workflow_exists() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/actionlint.yml");
    assert!(
        path.is_file(),
        "expected actionlint CI workflow at {} (Issue #1292)",
        path.display()
    );
}

#[test]
fn actionlint_workflow_triggers_on_pull_request_and_push() {
    let body = load_workflow();
    // The `on:` mapping must include both pull_request and push so the
    // gate fires on PRs (catching regressions) and on direct pushes to
    // protected branches (catching anything that bypasses PR review).
    assert!(
        body.contains("pull_request:"),
        "actionlint workflow must trigger on pull_request (Issue #1292)"
    );
    assert!(
        body.contains("push:"),
        "actionlint workflow must trigger on push (Issue #1292)"
    );
}

#[test]
fn actionlint_workflow_has_minimal_top_level_permissions() {
    let body = load_workflow();
    // Top-level `permissions:` block at column 0 with `contents: read`.
    let mut in_block = false;
    let mut found = false;
    for line in body.lines() {
        if line == "permissions:" {
            in_block = true;
            continue;
        }
        if in_block {
            if line.starts_with(' ') {
                if line.trim() == "contents: read" {
                    found = true;
                    break;
                }
            } else if !line.trim().is_empty() && !line.starts_with('#') {
                break;
            }
        }
    }
    assert!(
        found,
        "actionlint workflow must declare top-level `permissions:` with \
         `contents: read` (Issue #1286 / #1292)"
    );
}

#[test]
fn actionlint_workflow_pins_third_party_actions_to_sha() {
    let body = load_workflow();
    // Every `uses:` line that references an external action must point
    // at a 40-char lower-case hex SHA (Issue #1216).
    let sha_re_ok = |reference: &str| -> bool {
        reference.len() == 40 && reference.chars().all(|c| c.is_ascii_hexdigit())
    };
    let mut saw_external = false;
    for (idx, raw) in body.lines().enumerate() {
        let lineno = idx + 1;
        let mut after = raw.trim_start();
        if after.starts_with('#') {
            continue;
        }
        if let Some(rest) = after.strip_prefix("- ") {
            after = rest.trim_start();
        }
        let Some(rest) = after.strip_prefix("uses:") else {
            continue;
        };
        let value = rest
            .split('#')
            .next()
            .unwrap_or("")
            .trim()
            .trim_matches('"')
            .trim_matches('\'');
        if value.is_empty() || value.starts_with("./") {
            continue;
        }
        saw_external = true;
        let Some((_repo, reference)) = value.rsplit_once('@') else {
            panic!("line {lineno}: malformed `uses:` value `{value}` — no `@<ref>` suffix");
        };
        assert!(
            sha_re_ok(reference),
            "line {lineno}: `uses: {value}` must pin to a 40-char SHA, \
             not a mutable tag/branch (Issue #1216)"
        );
    }
    assert!(
        saw_external,
        "actionlint workflow must use an external action (e.g. \
         raven-actions/actionlint or actions/checkout) pinned to a SHA \
         (Issue #1292)"
    );
}

#[test]
fn actionlint_workflow_declares_job_timeout() {
    let body = load_workflow();
    assert!(
        body.contains("timeout-minutes:"),
        "actionlint workflow must declare `timeout-minutes:` on its job \
         to bound runner exposure (Issue #1287 / #1292)"
    );
}

#[test]
fn actionlint_workflow_declares_concurrency() {
    let body = load_workflow();
    assert!(
        body.contains("concurrency:"),
        "actionlint workflow must declare a `concurrency:` block that \
         cancels superseded runs (Issue #1288 / #1292)"
    );
    assert!(
        body.contains("cancel-in-progress: true"),
        "actionlint workflow's concurrency block must set \
         `cancel-in-progress: true` (Issue #1288 / #1292)"
    );
}

#[test]
fn actionlint_workflow_invokes_actionlint() {
    let body = load_workflow();
    // Sanity: ensure the workflow actually runs actionlint, either via
    // the raven-actions/actionlint action or by invoking the binary in
    // a `run:` step. Without this assertion, the workflow file could
    // satisfy every structural check above and still not lint anything.
    let mentions_action =
        body.contains("raven-actions/actionlint@") || body.contains("rhysd/actionlint@");
    let mentions_binary = body.contains("actionlint ")
        || body.contains("actionlint\n")
        || body.contains("./actionlint");
    assert!(
        mentions_action || mentions_binary,
        "actionlint workflow must invoke actionlint (via action or \
         binary) — Issue #1292"
    );
}
