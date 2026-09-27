//! Chunk 11c-1 ledger contract: the `src/watchdog.rs` and
//! `src/tracking_alloc.rs` rows of the chunk 11 record (Issue #2252).
//!
//! The scaffold (Issue #2233) cut every row as `pending`. This file pins the
//! swept state of those two rows only — `src/discovery_history.rs` shares the
//! section but is owned by sibling sub-issue 11c-2 and may still
//! read `pending`:
//!
//! * both rows carry `audited — <reason>`;
//! * the section records the SIGUSR1 verdict, cross-references the watchdog
//!   configuration findings (#2122, #2259) rather than refiling them, and names
//!   both decision consumers of `TrackingAlloc::allocated`;
//! * the section's mutation region states that neither file has a filesystem
//!   mutation site;
//! * every `file.rs::symbol` the section cites still exists, and no citation
//!   uses a line number (Issue #1942).

use std::path::PathBuf;

const RECORD: &str = "docs/audits/security-sweep-chunk-11-filesystem-lifecycle.md";
const SECTION: &str = "watchdog + tracking_alloc + discovery_history";

const AUDITED: [&str; 2] = ["src/watchdog.rs", "src/tracking_alloc.rs"];

/// Terms the probe dispositions must carry: the signal verdict, the
/// cross-referenced (not refiled) config findings, and both consumers.
const REQUIRED_TERMS: [&str; 6] = [
    "SIGUSR1",
    "#2122",
    "#2259",
    "#2234",
    "ffi/utilities.rs::discovery_memory_usage_bytes",
    "analysis/utils/memory.rs::is_memory_budget_exceeded",
];

/// `(cited site, source file, definition the symbol must still match)`.
const CITED: [(&str, &str, &str); 11] = [
    ("watchdog.rs::watchdog_loop", "src/watchdog.rs", "fn watchdog_loop("),
    ("watchdog.rs::Watchdog::start", "src/watchdog.rs", "fn start(config: WatchdogConfig)"),
    ("watchdog.rs::Drop::drop", "src/watchdog.rs", "impl Drop for Watchdog"),
    ("watchdog.rs::heartbeat_snapshot", "src/watchdog.rs", "fn heartbeat_snapshot("),
    ("watchdog.rs::WatchdogConfig::from_env", "src/watchdog.rs", "fn from_env() -> Option<Self>"),
    ("tracking_alloc.rs::TrackingAlloc::allocated", "src/tracking_alloc.rs", "pub fn allocated(&self)"),
    ("ffi/utilities.rs::discovery_memory_usage_bytes", "src/ffi/utilities.rs", "fn discovery_memory_usage_bytes("),
    ("analysis/utils/memory.rs::is_memory_budget_exceeded", "src/analysis/utils/memory.rs", "fn is_memory_budget_exceeded("),
    ("analysis/orchestration.rs::analyze_all", "src/analysis/orchestration.rs", "fn analyze_all("),
    ("config/user_facing.rs::watchdog_abort_delay", "src/config/user_facing.rs", "fn watchdog_abort_delay("),
    ("debug.rs::install_signal_handler", "src/debug.rs", "fn install_signal_handler("),
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// The text of a Markdown section, from its heading line to the next heading
/// of the same or a higher level.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let start = doc
        .lines()
        .scan(0usize, |offset, line| {
            let at = *offset;
            *offset += line.len() + 1;
            Some((at, line))
        })
        .find(|(_, line)| line.trim_end() == heading)
        .map_or_else(
            || panic!("{RECORD} must carry the heading `{heading}`"),
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

/// This section's marked region of a finding table: from its marker to the
/// next `<!-- section:` marker.
fn marker_region(table: &str) -> &str {
    let marker = format!("<!-- section: {SECTION} -->");
    let start = table
        .find(&marker)
        .unwrap_or_else(|| panic!("the table must carry `{marker}`"))
        + marker.len();
    let end = table[start..]
        .find("<!-- section:")
        .map_or(table.len(), |offset| start + offset);
    &table[start..end]
}

fn table_rows(body: &str) -> Vec<&str> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .collect()
}

/// `(path, outcome)` for each three-cell `src/` row of an inventory table.
fn file_rows(body: &str) -> Vec<(String, String)> {
    table_rows(body)
        .into_iter()
        .filter_map(|line| {
            let cells: Vec<&str> = line.trim().trim_matches('|').split('|').collect();
            if cells.len() != 3 {
                return None;
            }
            let path = cells[0].trim().trim_matches('`').to_string();
            path.starts_with("src/")
                .then(|| (path, cells[2].trim().to_string()))
        })
        .collect()
}

#[test]
fn watchdog_and_tracking_alloc_rows_are_audited_with_a_reason() {
    let doc = read(RECORD);
    let files_swept = section(&doc, "## Files swept");
    let rows = file_rows(section(files_swept, &format!("### {SECTION}")));

    for path in AUDITED {
        let (_, outcome) = rows
            .iter()
            .find(|(row_path, _)| row_path == path)
            .unwrap_or_else(|| panic!("the {SECTION} section must carry a {path} row"));
        let Some((verdict, reason)) = outcome.split_once(" — ") else {
            panic!("{path} must read `audited — <reason>`, got: {outcome}");
        };
        assert_eq!(verdict.trim(), "audited", "{path} verdict: {outcome}");
        assert!(!reason.trim().is_empty(), "{path} must give a reason: {outcome}");
    }
    assert!(
        rows.iter().any(|(path, _)| path == "src/discovery_history.rs"),
        "the discovery_history.rs row (11c-2) must stay in the section"
    );
}

#[test]
fn the_section_records_the_signal_verdict_cross_references_and_consumers() {
    let doc = read(RECORD);
    let body = section(section(&doc, "## Files swept"), &format!("### {SECTION}"));
    for term in REQUIRED_TERMS {
        assert!(body.contains(term), "the {SECTION} section must mention `{term}`");
    }
}

#[test]
fn the_mutation_region_states_neither_file_has_a_site() {
    let doc = read(RECORD);
    let region = marker_region(section(&doc, "## Filesystem mutation sites"));
    let row = table_rows(region)
        .into_iter()
        .find(|row| row.contains("`src/watchdog.rs`") && row.contains("`src/tracking_alloc.rs`"))
        .unwrap_or_else(|| {
            panic!("the {SECTION} mutation region must carry a row for watchdog.rs and tracking_alloc.rs")
        });
    assert!(row.contains("none"), "the row must state there is no site: {row}");
}

#[test]
fn every_cited_symbol_exists_and_no_citation_uses_a_line_number() {
    let doc = read(RECORD);
    let body = section(section(&doc, "## Files swept"), &format!("### {SECTION}"));
    for (site, file, definition) in CITED {
        assert!(body.contains(site), "the {SECTION} section must cite `{site}`");
        // Issue #1942: a symbol citation is only worth something while the
        // symbol still exists.
        assert!(
            read(file).contains(definition),
            "`{site}` is cited but `{definition}` is no longer in {file}"
        );
    }
    let bytes = body.as_bytes();
    let line_cited = body
        .match_indices(".rs:")
        .any(|(at, _)| bytes.get(at + 4).is_some_and(u8::is_ascii_digit));
    assert!(!line_cited, "the {SECTION} section must cite by symbol, not `file.rs:<line>`");
}
