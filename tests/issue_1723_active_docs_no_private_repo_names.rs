//! Active documentation must not name private repositories (Issue #1723).
//!
//! This is a public repository. Its front-line documentation — `README.md`,
//! `CHANGELOG.md`, and everything under `docs/` — was citing private
//! `stSoftware` repositories by name as the evidence behind tuning decisions.
//! Every such mention points a public reader at a repository they cannot open:
//! dead weight for them, and a standing advertisement of private infrastructure
//! names for everyone else.
//!
//! The evidence itself is not the problem — the *names* are. Active docs
//! describe the same evidence at concept level ("the production discovery
//! cache", "a large production creature"), which is both readable by the public
//! and accurate.
//!
//! This suite is the regression gate. It fails loudly (Issue #3234) the moment
//! an active documentation page reintroduces a private repository name.
//!
//! Scope note: `docs/archive/` is deliberately excluded — archived PR summaries
//! are a historical record and are cleaned separately (Issue #1726).

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every active (non-archived) Markdown page: the repository-root pages plus
/// everything under `docs/` except the `docs/archive/` historical record.
fn active_doc_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("failed to read docs dir {}: {e}", dir.display()));
        for entry in entries {
            let path = entry
                .unwrap_or_else(|e| panic!("failed to read entry in {}: {e}", dir.display()))
                .path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "archive") {
                    continue;
                }
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "md") {
                out.push(path);
            }
        }
    }

    let root = repo_root();
    let mut files = Vec::new();

    // Root-level Markdown pages (README, CHANGELOG, CONTRIBUTING, ...).
    let root_entries = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("failed to read repository root {}: {e}", root.display()));
    for entry in root_entries {
        let path = entry
            .unwrap_or_else(|e| panic!("failed to read entry in {}: {e}", root.display()))
            .path();
        if path.is_file() && path.extension().is_some_and(|e| e == "md") {
            files.push(path);
        }
    }

    walk(&root.join("docs"), &mut files);
    files.sort();
    files
}

/// Marker naming a private repository or one of its deployments.
///
/// The needle is assembled at runtime from fragments so this guard does not
/// itself commit the private name it is written to keep out. A single token
/// covers the whole family (the bare repository, its cluster/discovery/sampler
/// siblings, and the numbered production deployments), because no public
/// artefact in this project carries that token.
fn private_repo_marker() -> String {
    format!("{}{}", "GR", "Q")
}

/// No active documentation page may name a private repository.
#[test]
fn no_active_doc_names_a_private_repository() {
    let marker = private_repo_marker();
    let root = repo_root();
    let mut offenders = Vec::new();

    for path in active_doc_files() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read doc {}: {e}", path.display()));
        for (index, line) in text.lines().enumerate() {
            if line.contains(marker.as_str()) {
                let shown = path.strip_prefix(&root).unwrap_or(&path).display();
                offenders.push(format!("{shown}:{}", index + 1));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "active documentation must describe production evidence at concept level, but {} \
         line(s) name a private repository:\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}

/// Harness-integrity guard: the walk must actually find the front-line pages,
/// and must not reach into the archived history it deliberately excludes. An
/// empty or mis-scoped set would make the guard above pass vacuously.
#[test]
fn active_doc_walk_covers_the_front_line_pages_only() {
    let files = active_doc_files();
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
        "README.md",
        "CHANGELOG.md",
        "docs/FOCUS_SELECTION.md",
        "docs/CANDIDATE_PIPELINE_MCMC_AUDIT.md",
        "docs/analysis/snapshot-mining-1631.md",
    ] {
        assert!(
            relative.iter().any(|p| p == expected),
            "active doc walk missed {expected}; found {} files",
            relative.len()
        );
    }

    assert!(
        !relative.iter().any(|p| p.starts_with("docs/archive/")),
        "active doc walk must exclude the archived PR summaries (Issue #1726)"
    );
}
