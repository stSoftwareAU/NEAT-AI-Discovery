//! Loading the candidates corpus from a production discovery cache checkout.
//!
//! The live tree only ever holds the *current* model hash — the periodic
//! "Clean up OLD discovery caches" commits delete every earlier hash. The
//! corpus is therefore widened from git history (see [`super::git_history`]).

use anyhow::{Context, Result, bail};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use super::record::CandidateRecord;

/// Which side of the cache a record was filed under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// The controller kept the change.
    Success,
    /// The controller rejected the change.
    Failure,
}

impl Outcome {
    /// The cache directory name for this outcome.
    #[must_use]
    pub fn dir_name(self) -> &'static str {
        match self {
            Outcome::Success => "success",
            Outcome::Failure => "failures",
        }
    }
}

/// Where a corpus entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordSource {
    /// Present in the working tree of the checked-out branch.
    Live,
    /// Recovered from a git commit that deleted it.
    GitHistory,
}

/// One cache record plus the classification carried by its path.
#[derive(Debug, Clone)]
pub struct CorpusEntry {
    /// Success or failure side of the cache.
    pub outcome: Outcome,
    /// Model hash directory the record was filed under.
    pub model_hash: String,
    /// Strategy directory the record was filed under.
    pub strategy: String,
    /// Live tree or recovered from history.
    pub source: RecordSource,
    /// The parsed record.
    pub record: CandidateRecord,
}

/// The classification encoded in a cache path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachePath {
    /// Success or failure side of the cache.
    pub outcome: Outcome,
    /// Model hash directory.
    pub model_hash: String,
    /// Strategy directory.
    pub strategy: String,
}

/// Parses a repository-relative cache path of the form
/// `success|failures/<model-hash>/<strategy>/<key>.json`.
///
/// Returns `None` for any path that is not a candidate record, so callers can
/// skip README files and the like without treating them as corrupt records.
#[must_use]
pub fn parse_cache_path(relative: &str) -> Option<CachePath> {
    let parts: Vec<&str> = relative.split('/').filter(|p| !p.is_empty()).collect();
    let [top, model_hash, strategy, file] = parts.as_slice() else {
        return None;
    };
    if !file.ends_with(".json") {
        return None;
    }
    let outcome = match *top {
        "success" => Outcome::Success,
        "failures" => Outcome::Failure,
        _ => return None,
    };
    Some(CachePath {
        outcome,
        model_hash: (*model_hash).to_string(),
        strategy: (*strategy).to_string(),
    })
}

/// Parses one record, attaching the path so a malformed record fails loudly
/// with the file that caused it rather than being skipped.
pub fn parse_entry(
    relative: &str,
    contents: &str,
    source: RecordSource,
) -> Result<Option<CorpusEntry>> {
    let Some(path) = parse_cache_path(relative) else {
        return Ok(None);
    };
    let record: CandidateRecord = serde_json::from_str(contents)
        .with_context(|| format!("malformed candidate record at {relative}"))?;
    Ok(Some(CorpusEntry {
        outcome: path.outcome,
        model_hash: path.model_hash,
        strategy: path.strategy,
        source,
        record,
    }))
}

/// Loads every candidate record present in the working tree of `root`.
///
/// # Errors
/// Returns an error if `root` has neither cache directory, or if any record
/// under them is unreadable or malformed.
pub fn load_live(root: &Path) -> Result<Vec<CorpusEntry>> {
    let mut entries = Vec::new();
    let mut found_any_dir = false;
    for outcome in [Outcome::Success, Outcome::Failure] {
        let dir = root.join(outcome.dir_name());
        if !dir.is_dir() {
            continue;
        }
        found_any_dir = true;
        let mut files = Vec::new();
        collect_json_files(&dir, &mut files)?;
        for file in files {
            let relative = relative_path(root, &file)?;
            let contents = fs::read_to_string(&file)
                .with_context(|| format!("reading cache record {}", file.display()))?;
            if let Some(entry) = parse_entry(&relative, &contents, RecordSource::Live)? {
                entries.push(entry);
            }
        }
    }
    if !found_any_dir {
        bail!(
            "{} contains neither a success/ nor a failures/ directory — is it a discovery cache checkout?",
            root.display()
        );
    }
    Ok(entries)
}

/// Appends history-recovered entries, skipping any that duplicate one already
/// held. Returns the number of entries actually added.
pub fn merge_deduplicated(entries: &mut Vec<CorpusEntry>, extra: Vec<CorpusEntry>) -> usize {
    let mut seen: HashSet<(Outcome, String, String, String)> =
        entries.iter().map(identity_key).collect();
    let before = entries.len();
    for entry in extra {
        if seen.insert(identity_key(&entry)) {
            entries.push(entry);
        }
    }
    entries.len() - before
}

fn identity_key(entry: &CorpusEntry) -> (Outcome, String, String, String) {
    (
        entry.outcome,
        entry.model_hash.clone(),
        entry.record.key.clone(),
        entry.record.timestamp.clone(),
    )
}

fn relative_path(root: &Path, file: &Path) -> Result<String> {
    let relative = file
        .strip_prefix(root)
        .with_context(|| format!("{} is not under {}", file.display(), root.display()))?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

fn collect_json_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let read = fs::read_dir(dir).with_context(|| format!("reading directory {}", dir.display()))?;
    let mut children: Vec<PathBuf> = Vec::new();
    for entry in read {
        let entry =
            entry.with_context(|| format!("reading directory entry in {}", dir.display()))?;
        children.push(entry.path());
    }
    children.sort();
    for path in children {
        if path.is_dir() {
            collect_json_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "json") {
            out.push(path);
        }
    }
    Ok(())
}
