//! Recovering wiped cache records from the production discovery cache's git history.
//!
//! "Clean up OLD discovery caches" commits delete every record belonging to a
//! superseded model hash. Those records are still reachable in history, so the
//! study corpus is widened by reading each deleted blob back out of the commit
//! that removed it (Issue #1920).

use anyhow::{Context, Result, bail};
use std::path::Path;
use std::process::Command;

use super::corpus::{CorpusEntry, RecordSource, parse_entry};

/// A record path together with the commit that deleted it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeletedRecord {
    /// Commit that performed the deletion.
    pub commit: String,
    /// Repository-relative path of the deleted record.
    pub path: String,
}

/// Parses `git log --diff-filter=D --name-only` output in the
/// `commit <sha>` / path-per-line format produced by [`deleted_records`].
#[must_use]
pub fn parse_deleted_log(output: &str) -> Vec<DeletedRecord> {
    let mut records = Vec::new();
    let mut commit = String::new();
    for line in output.lines() {
        let line = line.trim_end();
        if let Some(sha) = line.strip_prefix("commit ") {
            commit = sha.to_string();
        } else if !line.is_empty() && !commit.is_empty() {
            records.push(DeletedRecord {
                commit: commit.clone(),
                path: line.to_string(),
            });
        }
    }
    records
}

/// Lists every candidate record deleted anywhere in the repository's history.
///
/// # Errors
/// Returns an error if `git` is missing or the command fails — a history sweep
/// that silently returns nothing would understate the corpus.
pub fn deleted_records(root: &Path) -> Result<Vec<DeletedRecord>> {
    let output = run_git(
        root,
        &[
            "log",
            "--all",
            "--diff-filter=D",
            "--name-only",
            "--pretty=format:commit %H",
            "--",
            "success",
            "failures",
        ],
    )?;
    Ok(parse_deleted_log(&output))
}

/// Recovers the content of every deleted record and parses it into entries.
///
/// # Errors
/// Returns an error if `git` is unavailable, a blob cannot be read, or a
/// recovered record is malformed.
pub fn load_from_history(root: &Path) -> Result<Vec<CorpusEntry>> {
    let mut entries = Vec::new();
    for deleted in deleted_records(root)? {
        // The blob still exists in the deleting commit's parent.
        let spec = format!("{}^:{}", deleted.commit, deleted.path);
        let contents = run_git(root, &["show", &spec])?;
        if let Some(entry) = parse_entry(&deleted.path, &contents, RecordSource::GitHistory)? {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn run_git(root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .with_context(|| format!("running `git {}` in {}", args.join(" "), root.display()))?;
    if !output.status.success() {
        bail!(
            "`git {}` failed in {} with {}: {}",
            args.join(" "),
            root.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout)
        .with_context(|| format!("`git {}` produced non-UTF-8 output", args.join(" ")))
}
