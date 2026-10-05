//! Generic Markdown ledger helpers (section lookup, table-row extraction),
//! used today by
//! `tests/issue_2253_chunk_11c2_discovery_history_test.rs` (Issue #2253).
//! They take the section name as a parameter rather than a file-specific
//! constant, so another ledger test can call them too.
#![allow(dead_code)]

use std::path::PathBuf;

/// The repository root, derived from the test crate's manifest directory.
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Read a repository-relative file to a `String`, panicking with its path on failure.
pub fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// The text of a Markdown section, from its heading line to the next heading
/// of the same or a higher level.
pub fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let start = doc
        .lines()
        .scan(0usize, |offset, line| {
            let at = *offset;
            *offset += line.len() + 1;
            Some((at, line))
        })
        .find(|(_, line)| line.trim_end() == heading)
        .map_or_else(
            || panic!("the document must carry the heading `{heading}`"),
            |(at, _)| at,
        );
    let level = heading.chars().take_while(|c| *c == '#').count();
    let body_start = start + heading.len();
    let mut cursor = body_start;
    let end = loop {
        let Some(offset) = doc[cursor..].find("\n#") else {
            break doc.len();
        };
        let at = cursor + offset + 1;
        let depth = doc[at..].chars().take_while(|c| *c == '#').count();
        if depth <= level {
            break at;
        }
        cursor = at;
    };
    &doc[body_start..end]
}

/// The `|`-prefixed table rows within a section body.
pub fn table_rows(body: &str) -> Vec<&str> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .collect()
}
