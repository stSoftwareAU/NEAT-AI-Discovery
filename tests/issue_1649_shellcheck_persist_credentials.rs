//! Issue #1649: the `shellcheck` job's `actions/checkout` step in
//! `.github/workflows/shellcheck.yml` must set `persist-credentials: false`.
//!
//! By default `actions/checkout` writes the workflow's `GITHUB_TOKEN` into
//! `.git/config` as an auth header, where any later step in the job — including
//! a compromised dependency — can read it and act as the token. The
//! `shellcheck` job only lints bash scripts over the tree; it never pushes back
//! to the repository nor fetches a private submodule, so it does not need the
//! persisted credential. Disabling persistence narrows the blast radius of a
//! compromised step.
//!
//! This test reads `shellcheck.yml` as plain text (no YAML parser is in the
//! dependency tree) and asserts the checkout step inside the `shellcheck` job
//! carries `persist-credentials: false`.

use std::fs;
use std::path::Path;

fn load_shellcheck_yml() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/shellcheck.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Locate a job block by name (`<job>:` at two-space indent under `jobs:`) and
/// return the block body up to the next sibling job or end of file.
fn job_block<'a>(body: &'a str, job_name: &str) -> Option<&'a str> {
    let header = format!("  {job_name}:");
    let start = body.find(&header)?;
    let after = &body[start..];
    let bytes = after.as_bytes();
    let mut search_from = header.len();
    while search_from < bytes.len() {
        if let Some(nl) = after[search_from..].find('\n') {
            let line_start = search_from + nl + 1;
            if line_start >= bytes.len() {
                return Some(after);
            }
            let line_end = after[line_start..]
                .find('\n')
                .map_or(bytes.len(), |n| line_start + n);
            let line = &after[line_start..line_end];
            // Sibling job: exactly two spaces then a non-space char, ends `:`.
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

#[test]
fn shellcheck_checkout_disables_credential_persistence() {
    let body = load_shellcheck_yml();
    let block = job_block(&body, "shellcheck")
        .expect("`shellcheck` job not found in shellcheck.yml (Issue #1649)");

    assert!(
        block.contains("uses: actions/checkout@"),
        "`shellcheck` job should still check out the repository (Issue #1649)",
    );
    assert!(
        block.contains("persist-credentials: false"),
        "`shellcheck` job's checkout step must set `persist-credentials: false` \
         so the GITHUB_TOKEN is not written to disk (Issue #1649)",
    );
}
