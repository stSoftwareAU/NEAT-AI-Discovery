//! Issue #1574: the `semgrep` job's `actions/checkout` step in
//! `.github/workflows/semgrep.yml` must set `persist-credentials: false`.
//!
//! By default `actions/checkout` writes the workflow's `GITHUB_TOKEN` into
//! `.git/config` as an auth header, where any later step in the job — including
//! a compromised dependency or an injected script — can read it and act as the
//! token. The `semgrep` job only checks out the tree to run the `semgrep ci`
//! SAST scan; it never pushes back to the repository nor fetches a private
//! submodule, so it does not need the persisted credential. Disabling
//! persistence narrows the blast radius of a compromised step.
//!
//! This test reads `semgrep.yml` as plain text (no YAML parser is in the
//! dependency tree) and asserts the checkout step inside the `semgrep` job
//! carries `persist-credentials: false`.

use std::fs;
use std::path::Path;

fn load_semgrep_yml() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/semgrep.yml");
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
fn semgrep_checkout_disables_credential_persistence() {
    let body = load_semgrep_yml();
    let block =
        job_block(&body, "semgrep").expect("`semgrep` job not found in semgrep.yml (Issue #1574)");

    assert!(
        block.contains("uses: actions/checkout@"),
        "`semgrep` job should still check out the repository (Issue #1574)",
    );
    assert!(
        block.contains("persist-credentials: false"),
        "`semgrep` job's checkout step must set `persist-credentials: false` \
         so the GITHUB_TOKEN is not written to disk (Issue #1574)",
    );
}
