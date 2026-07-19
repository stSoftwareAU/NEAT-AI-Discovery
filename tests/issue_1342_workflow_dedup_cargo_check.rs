//! Issue #1342: `.github/workflows/ci.yml` compiled the crate with
//! `cargo check` twice per pull request — once in the `quality` job
//! ("Check types", `--all-targets --all-features`) and once in the
//! `validation` job ("Validate Cargo.toml", default target).
//!
//! The `validation` invocation compiled a strict subset of the `quality`
//! invocation's scope, so it added no coverage while doubling the most
//! expensive part of the `validation` job. The redundant `cargo check`
//! is removed from the "Validate Cargo.toml" step, leaving only the
//! field-presence validation unique to that step. The single compile gate
//! is the `quality` job's "Run linter" (clippy) step — clippy is a superset
//! of `cargo check` and drives the same compilation, so it proves manifest
//! resolvability on every pull request (the redundant standalone "Check types"
//! `cargo check` step was removed in Issue #1658).
//!
//! This test parses `ci.yml` as plain text (no YAML parser is in the
//! dependency tree) and asserts:
//!   1. The "Validate Cargo.toml" step contains no `cargo check`.
//!   2. The "Validate Cargo.toml" step still performs field-presence
//!      validation (its genuinely distinct work is preserved).
//!   3. The "Run linter" step retains the broad compile gate so the
//!      manifest is still proven resolvable on every pull request.

use std::fs;
use std::path::{Path, PathBuf};

fn workflows_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows")
}

fn read_workflow(file_name: &str) -> String {
    let path = workflows_dir().join(file_name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Locate a step by its `- name: <step_name>` marker and return the block
/// of the step up to the next sibling step or new job header.
fn step_block<'a>(body: &'a str, step_name: &str) -> Option<&'a str> {
    let header = format!("    - name: {step_name}\n");
    let start = body.find(&header)?;
    let after = &body[start..];
    let mut idx = header.len();
    let bytes = after.as_bytes();
    while idx < bytes.len() {
        if let Some(nl) = after[idx..].find('\n') {
            let line_start = idx + nl + 1;
            if line_start >= bytes.len() {
                return Some(after);
            }
            let line_end = after[line_start..]
                .find('\n')
                .map_or(bytes.len(), |n| line_start + n);
            let line = &after[line_start..line_end];
            if line.starts_with("    - name:")
                || (line.starts_with("  ")
                    && line.as_bytes().get(2).is_some_and(|b| *b != b' ')
                    && line.trim_end().ends_with(':'))
            {
                return Some(&after[..line_start]);
            }
            idx = line_start;
        } else {
            return Some(after);
        }
    }
    Some(after)
}

#[test]
fn validate_cargo_toml_step_has_no_cargo_check() {
    let body = read_workflow("ci.yml");
    let block = step_block(&body, "Validate Cargo.toml")
        .expect("step `Validate Cargo.toml` not found in ci.yml");
    assert!(
        !block.contains("cargo check"),
        "step `Validate Cargo.toml` must not run `cargo check` — it duplicates the \
         broader compile gate in the `quality` job's `Run linter` step \
         (`cargo clippy --all-targets --all-features`), doubling the crate's \
         compilation per pull request for no added coverage (Issue #1342)",
    );
}

#[test]
fn validate_cargo_toml_step_keeps_field_presence_validation() {
    let body = read_workflow("ci.yml");
    let block = step_block(&body, "Validate Cargo.toml")
        .expect("step `Validate Cargo.toml` not found in ci.yml");
    // The genuinely distinct work — the name/version/edition field grep —
    // must remain so the step still validates the manifest's required fields.
    for field in ["name", "version", "edition"] {
        assert!(
            block.contains(&format!("\"{field}\"")),
            "step `Validate Cargo.toml` must still validate the presence of the \
             `{field}` field in Cargo.toml (Issue #1342)",
        );
    }
}

#[test]
fn linter_step_retains_broad_compile_gate() {
    let body = read_workflow("ci.yml");
    let block = step_block(&body, "Run linter").expect("step `Run linter` not found in ci.yml");
    assert!(
        block.contains("cargo clippy --all-targets --all-features"),
        "the `Run linter` step must retain `cargo clippy --all-targets --all-features` \
         as the single compile gate that proves manifest resolvability on every \
         pull request — clippy is a superset of `cargo check` (Issue #1342, Issue #1658)",
    );
}
