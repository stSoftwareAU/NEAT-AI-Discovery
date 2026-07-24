//! Code and config files must not cite the private policy repository by name
//! (Issue #1727).
//!
//! This is check 3 of the private-repo reference audit: textual private-repo
//! name mentions in code and configuration. Sibling gates already cover shipped
//! Rust source (Issue #1724/#1725), active documentation (Issue #1723), and the
//! archived PR summaries (Issue #1726). This suite closes the remaining surface:
//! shell scripts, JSON/TOML config, and GitHub workflow YAML.
//!
//! Three files cited the private policy repository under the `stSoftwareAU`
//! organisation by issue slug — `bump-deps.sh`, `renovate.json`, and
//! `.github/workflows/semgrep.yml`. The cited issue is unreadable to the public,
//! so the pointer explained nothing to public readers while naming a private
//! repository (the `renovate.json` one even surfaces in dependency-dashboard
//! UIs). The policy is now stated inline at concept level, and this suite is the
//! regression gate that keeps the private name out. It fails loudly (Issue
//! #3234) the moment the name reappears.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// File extensions for the code/config surface this gate scans.
const SCANNED_EXTENSIONS: [&str; 5] = ["sh", "json", "toml", "yml", "yaml"];

fn has_scanned_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| SCANNED_EXTENSIONS.contains(&ext))
}

/// Every code/config file that could carry a policy citation: the repository
/// root scripts and config files, plus everything under `.github/`.
fn scanned_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("failed to read directory {}: {e}", dir.display()));
        for entry in entries {
            let path = entry
                .unwrap_or_else(|e| panic!("failed to read entry in {}: {e}", dir.display()))
                .path();
            if path.is_dir() {
                walk(&path, out);
            } else if has_scanned_extension(&path) {
                out.push(path);
            }
        }
    }

    let root = repo_root();
    let mut files = Vec::new();

    // Repository-root scripts and config files (bump-deps.sh, renovate.json, ...).
    let root_entries = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("failed to read repository root {}: {e}", root.display()));
    for entry in root_entries {
        let path = entry
            .unwrap_or_else(|e| panic!("failed to read entry in {}: {e}", root.display()))
            .path();
        if path.is_file() && has_scanned_extension(&path) {
            files.push(path);
        }
    }

    // GitHub workflows and config.
    let github_dir = root.join(".github");
    if github_dir.is_dir() {
        walk(&github_dir, &mut files);
    }

    files.sort();
    files
}

/// Name of the private policy repository this gate keeps out of code and config.
///
/// The needle is assembled at runtime from fragments so this guard does not
/// itself commit the private repository name it is written to exclude. Matching
/// on this token alone (not the shared `stSoftwareAU` org prefix) leaves the
/// legitimate `stSoftwareAU/NEAT-AI-Discovery` and `stSoftwareAU/*` internal-dep
/// references untouched.
fn private_repo_marker() -> String {
    format!("{}{}", "Vibe", "Coding")
}

/// Report every offending line so one run lists the full remaining set rather
/// than failing one file at a time. Matching is case-insensitive so no casing
/// variant of the private name slips through.
fn offending_lines(text: &str, marker: &str) -> Vec<usize> {
    let needle = marker.to_lowercase();
    text.lines()
        .enumerate()
        .filter(|(_, line)| line.to_lowercase().contains(&needle))
        .map(|(index, _)| index + 1)
        .collect()
}

/// No code or config file may name the private policy repository.
#[test]
fn no_code_or_config_names_the_private_policy_repository() {
    let marker = private_repo_marker();
    let root = repo_root();
    let mut offenders = Vec::new();

    for path in scanned_files() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        let display = path.strip_prefix(&root).unwrap_or(&path).display();
        for line in offending_lines(&text, &marker) {
            offenders.push(format!(
                "{display}:{line} names the private policy repository"
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "code and config must state policy at concept level, not by private repository name, \
         but {} line(s) still do:\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}

/// Harness-integrity guard: the walk must actually find the files it is meant to
/// police. An empty or mis-scoped set would make the guard above pass vacuously.
#[test]
fn config_walk_covers_the_expected_files() {
    let files = scanned_files();
    let root = repo_root();
    let relative: Vec<String> = files
        .iter()
        .map(|p| {
            p.strip_prefix(&root)
                .unwrap_or(p)
                .display()
                .to_string()
                .replace('\\', "/")
        })
        .collect();

    for expected in [
        "bump-deps.sh",
        "renovate.json",
        ".github/workflows/semgrep.yml",
    ] {
        assert!(
            relative.iter().any(|p| p == expected),
            "config walk missed {expected}; found {} files",
            relative.len()
        );
    }
}
