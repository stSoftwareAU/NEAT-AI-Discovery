//! Issue #1484: the `markdown-lint` workflow installed `markdownlint-cli2`
//! with an unpinned `npm install -g markdownlint-cli2`, resolving the
//! latest release (and its full transitive tree) fresh on every PR. That
//! made it the only tool install in the repository that was neither
//! version-pinned nor integrity-checked — the weakest link in an
//! otherwise fully-pinned supply chain (class A03, Software Supply Chain
//! Failures).
//!
//! This test enforces the configuration-level contract for the fix:
//!
//!   1. The install step in `.github/workflows/markdown-lint.yml` pins an
//!      exact `markdownlint-cli2@<x.y.z>` version.
//!   2. That install disables lifecycle scripts (`--ignore-scripts`) so a
//!      poisoned release cannot run a `postinstall` on the runner.
//!   3. `renovate.json` manages the pinned version via a custom manager so
//!      it still receives quarantined (>= 24h) bumps like every other
//!      external dependency.

use std::fs;
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn workflow_path() -> PathBuf {
    root().join(".github/workflows/markdown-lint.yml")
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Return the `run:` line that installs markdownlint-cli2 via npm.
fn npm_install_line(workflow: &str) -> &str {
    workflow
        .lines()
        .find(|line| line.contains("npm install") && line.contains("markdownlint-cli2"))
        .unwrap_or_else(|| {
            panic!("markdown-lint.yml must install markdownlint-cli2 via npm (Issue #1484)")
        })
}

/// True when `s` contains `markdownlint-cli2@` followed by a semver
/// (`x.y.z`).
fn has_pinned_version(s: &str) -> bool {
    let Some(idx) = s.find("markdownlint-cli2@") else {
        return false;
    };
    let after = &s[idx + "markdownlint-cli2@".len()..];
    // First token must look like `NUMBER.NUMBER.NUMBER`.
    let token: String = after
        .chars()
        .take_while(|c| !c.is_whitespace())
        .collect();
    let parts: Vec<&str> = token.split('.').collect();
    parts.len() >= 3
        && parts[..3]
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

#[test]
fn markdownlint_install_is_version_pinned() {
    let workflow = read(&workflow_path());
    let line = npm_install_line(&workflow);
    assert!(
        has_pinned_version(line),
        "the markdownlint-cli2 install in markdown-lint.yml must pin an \
         exact `markdownlint-cli2@<x.y.z>` version (Issue #1484) — found: {line}"
    );
    // The bare, unpinned form must be gone.
    assert!(
        !workflow.contains("markdownlint-cli2\n") || has_pinned_version(line),
        "markdown-lint.yml must not install an unpinned markdownlint-cli2 \
         (Issue #1484)"
    );
}

#[test]
fn markdownlint_install_disables_lifecycle_scripts() {
    let workflow = read(&workflow_path());
    let line = npm_install_line(&workflow);
    assert!(
        line.contains("--ignore-scripts"),
        "the markdownlint-cli2 install must pass `--ignore-scripts` so a \
         poisoned release cannot run install/postinstall scripts on the \
         runner (Issue #1484) — found: {line}"
    );
}

#[test]
fn renovate_manages_the_pinned_markdownlint_version() {
    let contents = read(&root().join("renovate.json"));
    assert!(
        contents.contains("markdownlint-cli2"),
        "renovate.json must reference markdownlint-cli2 so the pinned \
         workflow version still receives managed, quarantined bumps \
         (Issue #1484)"
    );
    assert!(
        contents.contains("customManagers"),
        "renovate.json must use a customManagers entry to track the \
         markdownlint-cli2 version embedded in the workflow (Issue #1484)"
    );
    assert!(
        contents.contains("\"npm\""),
        "renovate.json must declare the npm datasource for the \
         markdownlint-cli2 custom manager (Issue #1484)"
    );
    // The npm bump must inherit the >= 24h quarantine window.
    assert!(
        contents.contains("minimumReleaseAge"),
        "renovate.json must apply a minimumReleaseAge quarantine to the \
         markdownlint-cli2 npm dependency (Issue #1484)"
    );
}
