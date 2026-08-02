//! Repeatable study of the production discovery candidates cache (Issue #1920).
//!
//! The cache holds one JSON record per evaluated candidate, filed under
//! `success/<model-hash>/<strategy>/` or `failures/<model-hash>/<strategy>/`.
//! This module loads that corpus — widening it with records that the periodic
//! "Clean up OLD discovery caches" commits deleted — and aggregates it into a
//! Markdown report covering the two questions the issue asks:
//!
//! - **Volume** — how many candidates are evaluated and cached per day, model
//!   hash and strategy, and how lopsided the strategy mix is.
//! - **Gain size** — which record fields correlate with a bigger `scoreDelta`.
//!
//! Run it with the `study_candidates_cache` example:
//!
//! ```text
//! cargo run --example study_candidates_cache -- /path/to/discovery-cache-checkout
//! ```

pub mod corpus;
pub mod git_history;
pub mod record;
pub mod report;
pub mod stats;

use anyhow::Result;
use std::path::Path;

pub use corpus::{CorpusEntry, Outcome, RecordSource};
pub use record::CandidateRecord;
pub use report::render_markdown;
pub use stats::{CacheStudy, GroupStats, Predictor};

/// Loads the corpus from a production discovery cache checkout.
///
/// When `include_history` is set, records deleted by cache-cleanup commits are
/// recovered from git history and merged in, skipping duplicates of live records.
///
/// # Errors
/// Returns an error if `root` is not a cache checkout, if any record is
/// unreadable or malformed, or if the git history sweep fails.
pub fn load_corpus(root: &Path, include_history: bool) -> Result<Vec<CorpusEntry>> {
    let mut entries = corpus::load_live(root)?;
    if include_history {
        let recovered = git_history::load_from_history(root)?;
        corpus::merge_deduplicated(&mut entries, recovered);
    }
    Ok(entries)
}

/// Loads the corpus and renders the Markdown study report in one step.
///
/// # Errors
/// Propagates any error from [`load_corpus`].
pub fn study_checkout(root: &Path, include_history: bool) -> Result<String> {
    let entries = load_corpus(root, include_history)?;
    Ok(render_markdown(&stats::study(&entries)))
}
