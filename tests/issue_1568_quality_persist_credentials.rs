//! Issue #1568: the `quality` job's `actions/checkout` step in
//! `.github/workflows/ci.yml` must set `persist-credentials: false`.
//!
//! By default `actions/checkout` writes the workflow `GITHUB_TOKEN` into
//! `.git/config` as an auth header, where any later step in the job — including
//! a compromised dependency — can read it and act as the token. The `quality`
//! job only reads the tree (fmt, clippy, check, build, test) and never pushes
//! back, so it does not need the persisted credential. Disabling persistence
//! narrows the blast radius of a compromised step.
//!
//! This test reads `ci.yml` as plain text (no YAML parser is in the dependency
//! tree) and asserts that the checkout `with:` block inside the `quality` job
//! declares `persist-credentials: false`.

use std::fs;
use std::path::Path;

fn load_ci_yml() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/ci.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Extracts the lines belonging to the named top-level job (2-space indent).
/// The block runs from the `  <job>:` header until the next column-2 job
/// header or the end of the file.
fn extract_job<'a>(body: &'a str, job: &str) -> Vec<&'a str> {
    let header = format!("  {job}:");
    let lines: Vec<&str> = body.lines().collect();
    let mut out = Vec::new();
    let mut inside = false;
    for line in lines {
        if line == header {
            inside = true;
            continue;
        }
        if inside {
            // A new column-2 job header (`  something:`) ends the block.
            let is_job_header = line.starts_with("  ")
                && !line.starts_with("   ")
                && line.trim_end().ends_with(':')
                && !line.trim_start().starts_with('-');
            if is_job_header {
                break;
            }
            out.push(line);
        }
    }
    out
}

/// Returns true if the checkout step's `with:` block within `job_lines`
/// declares `persist-credentials: false`.
fn checkout_disables_persist_credentials(job_lines: &[&str]) -> bool {
    let mut in_checkout_with = false;
    for line in job_lines {
        let trimmed = line.trim_start();
        // Enter the with: block that follows a checkout `uses:` line.
        if trimmed.starts_with("uses: actions/checkout@") {
            in_checkout_with = false; // reset; wait for its `with:`
            continue;
        }
        if trimmed == "with:" {
            in_checkout_with = true;
            continue;
        }
        // A new step (`- name:`/`- uses:`) ends the current with: block.
        if in_checkout_with && trimmed.starts_with('-') {
            in_checkout_with = false;
        }
        if in_checkout_with && trimmed == "persist-credentials: false" {
            return true;
        }
    }
    false
}

#[test]
fn quality_checkout_disables_persist_credentials() {
    let body = load_ci_yml();
    let quality = extract_job(&body, "quality");
    assert!(
        !quality.is_empty(),
        "ci.yml must define a `quality:` job (Issue #1568)"
    );
    assert!(
        checkout_disables_persist_credentials(&quality),
        "the `quality` job's actions/checkout step must set \
         `persist-credentials: false` so the GITHUB_TOKEN is not written to \
         disk (Issue #1568)"
    );
}
