//! Studies the production discovery candidates cache and prints a Markdown report.
//!
//! ```text
//! cargo run --example study_candidates_cache -- <discovery cache checkout> [--no-history]
//! ```
//!
//! By default the corpus is widened with records that "Clean up OLD discovery
//! caches" commits deleted; pass `--no-history` to study the working tree only
//! (Issue #1920).

use anyhow::{Result, bail};
use neat_ai_discovery::analysis::cache_study::study_checkout;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut root: Option<PathBuf> = None;
    let mut include_history = true;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--no-history" => include_history = false,
            other if other.starts_with("--") => bail!("unknown option {other}"),
            other if root.is_none() => root = Some(PathBuf::from(other)),
            other => bail!("unexpected extra argument {other}"),
        }
    }
    let Some(root) = root else {
        bail!("usage: study_candidates_cache <discovery cache checkout> [--no-history]");
    };

    print!("{}", study_checkout(&root, include_history)?);
    Ok(())
}
