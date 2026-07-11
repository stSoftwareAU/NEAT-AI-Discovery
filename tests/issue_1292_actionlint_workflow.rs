//! Issue #1292: A CI lint gate for GitHub Actions workflows must invoke
//! `actionlint` (upstream: rhysd/actionlint —
//! <https://github.com/rhysd/actionlint>) on every push and pull request
//! so that workflow lint regressions fail the build.
//!
//! This test reads `.github/workflows/actionlint.yml` and asserts:
//!   1. The workflow file exists.
//!   2. It triggers on `pull_request` and does NOT re-run on `push` to
//!      the default branch `Develop` (Issue #1563): as a checker it
//!      gates the pull request only, so a duplicate post-merge run is
//!      not declared. (This point relaxes the original Issue #1292
//!      requirement of a `push:` trigger — see the note on
//!      `actionlint_workflow_triggers_on_pull_request_not_push_develop`.)
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

/// Extract the top-level `on:` block: from the `on:` line to the next
/// top-level key (a non-space, non-comment line ending in `:`).
fn on_block(body: &str) -> String {
    let mut lines = body.lines();
    let mut block = String::new();
    let mut in_block = false;
    for line in lines.by_ref() {
        if line == "on:" || line.starts_with("on:") && !line.starts_with(' ') {
            in_block = true;
            block.push_str(line);
            block.push('\n');
            continue;
        }
        if in_block {
            // A sibling top-level key ends the `on:` block.
            if !line.is_empty()
                && !line.starts_with(' ')
                && !line.starts_with('#')
                && line.trim_end().ends_with(':')
            {
                break;
            }
            block.push_str(line);
            block.push('\n');
        }
    }
    block
}

#[test]
fn actionlint_workflow_triggers_on_pull_request_not_push_develop() {
    // Issue #1563: as a checker (not a deploy/publish workflow) this
    // gate must fire on the pull request but must NOT re-run on `push`
    // to the default branch `Develop`. A post-merge push run would
    // duplicate the run that already gated the PR — wasting CI minutes
    // and risking a red tick on Develop for a check that already
    // passed. This assertion intentionally supersedes the original
    // Issue #1292 requirement that the workflow trigger on `push`.
    let body = load_workflow();
    let on = on_block(&body);
    assert!(
        on.contains("pull_request:"),
        "actionlint workflow must trigger on pull_request (Issue #1292 / #1563)"
    );
    // No `push:` trigger may reach the default branch `Develop`. The
    // finding permits either dropping `push:` entirely or narrowing its
    // `branches:` to exclude `Develop`; both satisfy this check because
    // `Develop` must not appear anywhere inside a push trigger.
    if let Some(push_idx) = on.find("push:") {
        let push_section = &on[push_idx..];
        assert!(
            !push_section.contains("Develop"),
            "actionlint workflow must not re-run on push to the default \
             branch `Develop` — drop `push:` or exclude `Develop` from \
             its branches filter (Issue #1563). Found:\n{on}"
        );
    }
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
fn actionlint_workflow_checkout_disables_persist_credentials() {
    // Issue #1566: the `actions/checkout` step must set
    // `persist-credentials: false` so the workflow's GITHUB_TOKEN is not
    // written to `.git/config`, where a later compromised step could read
    // it. This job only reads the checkout to lint workflow files — it
    // never pushes back or fetches a private submodule — so it does not
    // need the persisted credential.
    let body = load_workflow();
    let mut checkout_line: Option<usize> = None;
    for (idx, line) in body.lines().enumerate() {
        if line.contains("uses:") && line.contains("actions/checkout@") {
            checkout_line = Some(idx);
            break;
        }
    }
    let checkout_idx =
        checkout_line.expect("actionlint workflow must use actions/checkout (Issue #1566)");

    // Scan the checkout step's `with:` block — the indented lines that
    // follow the `uses:` line until the next step (`- ...`) or a
    // dedent to the step-list column.
    let lines: Vec<&str> = body.lines().collect();
    let uses_indent = lines[checkout_idx].len() - lines[checkout_idx].trim_start().len();
    let mut found = false;
    for line in &lines[checkout_idx + 1..] {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent = line.len() - trimmed.len();
        // A new list item at or below the `uses:` indent ends this step.
        if indent <= uses_indent {
            break;
        }
        if trimmed == "persist-credentials: false" {
            found = true;
            break;
        }
    }
    assert!(
        found,
        "actionlint workflow's actions/checkout step must set \
         `persist-credentials: false` so the GITHUB_TOKEN is not written \
         to .git/config (Issue #1566)"
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
