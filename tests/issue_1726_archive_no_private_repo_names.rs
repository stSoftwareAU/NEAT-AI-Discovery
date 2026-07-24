//! Archived PR summaries must not name private repositories (Issue #1726).
//!
//! This is a public repository. The historical PR-summary archive under
//! `docs/archive/pr-summaries/` was citing private `stSoftware` repositories by
//! name — and in several places linked directly to private issues, commits, and
//! checkout paths. Archived or not, every file here ships in every public clone
//! and is indexed by search engines: the direct links point the public at
//! resources that 404 for them, and the sheer volume of name-level mentions made
//! the private repository names a pervasive fixture of a public repository.
//!
//! The evidence itself is not the problem — the *names* are. The archive now
//! describes the same evidence at concept level ("the production discovery
//! cache", "a large production creature"), which is both readable by the public
//! and accurate.
//!
//! This suite is the regression gate. It fails loudly (Issue #3234) the moment
//! an archived PR summary reintroduces a private repository name. It is the
//! archive counterpart to the active-docs gate in
//! `issue_1723_active_docs_no_private_repo_names.rs`.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn archive_dir() -> PathBuf {
    repo_root()
        .join("docs")
        .join("archive")
        .join("pr-summaries")
}

/// Every Markdown page under the archived PR-summary directory.
fn archived_doc_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("failed to read archive dir {}: {e}", dir.display()));
        for entry in entries {
            let path = entry
                .unwrap_or_else(|e| panic!("failed to read entry in {}: {e}", dir.display()))
                .path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "md") {
                out.push(path);
            }
        }
    }

    let mut files = Vec::new();
    walk(&archive_dir(), &mut files);
    files.sort();
    files
}

/// Markers naming a private repository or one of its deployments.
///
/// The needles are assembled at runtime from fragments so this guard does not
/// itself commit the private names it is written to keep out. The short
/// three-letter prefix covers the whole family (the bare repository, its
/// cluster/discovery/sampler siblings, and the numbered production deployments)
/// because no public artefact in this project carries that token; the workflow
/// marker matches the private CI repository without matching the public
/// "Vibe Coder" product name.
fn private_repo_markers() -> Vec<String> {
    vec![
        format!("{}{}", "GR", "Q"),
        format!("{}{}", "Vibe", "Coding"),
    ]
}

/// No archived PR summary may name a private repository.
#[test]
fn no_archived_summary_names_a_private_repository() {
    let markers = private_repo_markers();
    let root = repo_root();
    let mut offenders = Vec::new();

    for path in archived_doc_files() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read doc {}: {e}", path.display()));
        let lower = text.to_lowercase();
        // Cheap short-circuit: skip line scan unless a marker is present.
        if !markers.iter().any(|m| lower.contains(&m.to_lowercase())) {
            continue;
        }
        for (index, line) in text.lines().enumerate() {
            let line_lower = line.to_lowercase();
            for marker in &markers {
                if line_lower.contains(&marker.to_lowercase()) {
                    let shown = path.strip_prefix(&root).unwrap_or(&path).display();
                    offenders.push(format!("{shown}:{}", index + 1));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "archived PR summaries must describe production evidence at concept level, but {} \
         line(s) name a private repository:\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}

/// Harness-integrity guard: the walk must actually find the archived summaries.
/// An empty set would make the guard above pass vacuously (Issue #3234).
#[test]
fn archive_walk_covers_the_pr_summaries() {
    let files = archived_doc_files();
    assert!(
        files.len() > 50,
        "archive walk found only {} PR summaries; expected the full historical set",
        files.len()
    );

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

    assert!(
        relative
            .iter()
            .all(|p| p.starts_with("docs/archive/pr-summaries/")),
        "archive walk must stay within the archived PR-summary directory"
    );
    assert!(
        relative
            .iter()
            .any(|p| p == "docs/archive/pr-summaries/pr-summary-432.md"),
        "archive walk missed a known historical summary"
    );
}
