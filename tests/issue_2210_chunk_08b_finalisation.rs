//! Finalisation contract for the chunk 8b sweep record (Issue #2210).
//!
//! Issues #2103–#2109 each swept one section of a shared record. This file
//! pins the finished state, so the record cannot quietly slide back to a
//! partial one:
//!
//! * no row reads `pending`, every file row carries an outcome and a reason,
//!   and the status heading no longer says the sweep is in progress;
//! * every in-scope file with a production capacity or comparator site is
//!   cited in the matching finding table (detector pinned first, Issue #1799);
//! * no `output_competition.rs` row calls the module unreachable while
//!   `scoring_specs.rs` dispatches it (PR #2188), and the finding the
//!   re-verdict filed is listed;
//! * the index's `8b` entry equals the record's sweep date and baseline.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::Value;

const RECORD: &str = "docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md";
const INDEX: &str = "docs/audits/lib-sweep-coverage.json";
const DISPATCH: &str = "src/analysis/module_dispatch_specs/scoring_specs.rs";

const SCOPE: [&str; 4] = [
    "src/analysis/synapse",
    "src/analysis/scoring",
    "src/analysis/recommendation",
    "src/analysis/shared",
];

/// Issue #2210 says 59 rows; `issue_2169_structural_patterns_cancellation_test.rs`
/// (Issue #2221) landed after it was planned, so the table now has 60 — the
/// same count `tests/issue_2103_chunk_08b_ledger_scaffold.rs` pins.
const EXPECTED_FILE_COUNT: usize = 60;

/// The security finding the `output_competition.rs` re-verdict filed.
const OUTPUT_COMPETITION_FINDING: &str = "#2236";

/// Phrases that claim the module has no production caller.
const UNREACHABLE_PHRASES: [&str; 3] = ["unreachable", "not reachable", "no production caller"];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// The text of a Markdown section, from its heading to the next heading of the
/// same or a higher level.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let start = doc
        .find(heading)
        .unwrap_or_else(|| panic!("{RECORD} must carry the heading `{heading}`"));
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

fn rust_files_under(root: &str) -> Vec<String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", dir.display()));
        for entry in entries {
            let path = entry.expect("directory entry must be readable").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    let base = repo_root();
    let mut found = Vec::new();
    walk(&base.join(root), &mut found);
    found
        .iter()
        .map(|p| {
            p.strip_prefix(&base)
                .expect("walked path sits under the repo root")
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect()
}

#[derive(Debug)]
struct FileRow {
    path: String,
    outcome: String,
}

/// The three-cell path / line-count / outcome rows of the per-file table.
fn file_rows(body: &str) -> Vec<FileRow> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .filter_map(|line| {
            let cells: Vec<&str> = line.trim().trim_matches('|').split('|').collect();
            if cells.len() != 3 {
                return None;
            }
            let path = cells[0].trim().trim_matches('`').to_string();
            path.starts_with("src/").then(|| FileRow {
                path,
                outcome: cells[2].trim().to_string(),
            })
        })
        .collect()
}

/// The source up to the first `#[cfg(test)]` — the chunk 8b sweeps' shared
/// definition of "production" (Issues #2104–#2109).
fn production_source(rel: &str) -> String {
    let body = read(rel);
    match body.find("#[cfg(test)]") {
        Some(at) => body[..at].to_string(),
        None => body,
    }
}

/// `with_capacity\(|vec!\[.*;|\.reserve\(` — with the `;` required inside the
/// brackets, so a literal `vec![x];` is not a sized allocation.
fn is_capacity_site(line: &str) -> bool {
    if line.contains("with_capacity(") || line.contains(".reserve(") {
        return true;
    }
    let Some(at) = line.find("vec![") else {
        return false;
    };
    let after = &line[at + "vec![".len()..];
    after
        .rfind(']')
        .is_some_and(|close| after[..close].contains(';'))
}

/// `partial_cmp\(|\.sort_by\(|\.sort_unstable_by\(|\.total_cmp\(|\.max_by\(|\.min_by\(`.
fn is_comparator_site(line: &str) -> bool {
    [
        "partial_cmp(",
        ".sort_by(",
        ".sort_unstable_by(",
        ".total_cmp(",
        ".max_by(",
        ".min_by(",
    ]
    .iter()
    .any(|needle| line.contains(needle))
}

fn basename(rel: &str) -> &str {
    rel.rsplit('/').next().unwrap_or(rel)
}

/// Every in-scope file whose production half matches `detector`.
fn scope_files_matching(detector: fn(&str) -> bool) -> BTreeSet<String> {
    SCOPE
        .iter()
        .flat_map(|root| rust_files_under(root))
        .filter(|rel| production_source(rel).lines().any(detector))
        .collect()
}

/// The Markdown table rows (lines opening with `|`) anywhere in `body`.
fn table_rows(body: &str) -> Vec<&str> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .collect()
}

#[test]
fn no_row_reads_pending_and_every_file_row_has_an_outcome_and_a_reason() {
    let doc = read(RECORD);
    let rows = file_rows(section(&doc, "## Files swept"));
    assert_eq!(
        rows.len(),
        EXPECTED_FILE_COUNT,
        "{RECORD} must carry one row per in-scope file"
    );
    for row in &rows {
        assert!(
            !row.outcome.contains("pending"),
            "{} still reads `pending` — the chunk is not finished: {}",
            row.path,
            row.outcome
        );
        let Some((outcome, reason)) = row.outcome.split_once(" — ") else {
            panic!(
                "{} must read `<outcome> — <reason>`, got: {}",
                row.path, row.outcome
            );
        };
        assert!(
            !outcome.trim().is_empty() && !reason.trim().is_empty(),
            "{} must carry both an outcome and a reason: {}",
            row.path,
            row.outcome
        );
    }
    let pending_anywhere = table_rows(&doc)
        .into_iter()
        .filter(|row| row.contains("| pending |") || row.ends_with("| pending |"))
        .count();
    assert_eq!(pending_anywhere, 0, "no table cell may read `pending`");
}

#[test]
fn the_status_heading_reads_complete_and_the_placeholder_is_gone() {
    let doc = read(RECORD);
    assert!(
        !doc.contains("IN PROGRESS"),
        "{RECORD} still carries an `IN PROGRESS` status although no row reads `pending`"
    );
    assert!(
        doc.contains("### Sweep status — COMPLETE"),
        "{RECORD} must carry a `### Sweep status — COMPLETE` heading"
    );
    assert!(
        !section(&doc, "## Issues filed").contains("as they are swept"),
        "the `## Issues filed` placeholder must be deleted once every section is swept"
    );
}

#[test]
fn every_production_capacity_and_comparator_file_is_cited_in_its_table() {
    let capacity_files = scope_files_matching(is_capacity_site);
    let comparator_files = scope_files_matching(is_comparator_site);

    // Issue #1799: pin the detectors on known production sites first, so an
    // empty sweep cannot pass the citation check vacuously.
    for (set, known) in [
        (
            &capacity_files,
            "src/analysis/recommendation/sample_weighted.rs",
        ),
        (
            &capacity_files,
            "src/analysis/synapse/target_analysis/statistics.rs",
        ),
        (&comparator_files, "src/analysis/recommendation/fan_in.rs"),
        (
            &comparator_files,
            "src/analysis/recommendation/epistatic/deduplication.rs",
        ),
    ] {
        assert!(
            set.contains(known),
            "detector precondition: {known} must be found by the regex sweep; found {set:?}"
        );
    }
    assert!(
        !is_capacity_site("batches.push(vec![candidate]);"),
        "detector precondition: a literal `vec![x];` is not a sized allocation"
    );

    let doc = read(RECORD);
    for (files, heading) in [
        (&capacity_files, "## Capacity-from-input sites"),
        (&comparator_files, "## Float comparison sites"),
    ] {
        let rows = table_rows(section(&doc, heading)).join("\n");
        let missing: Vec<&String> = files
            .iter()
            .filter(|rel| !rows.contains(&format!("`{}::", basename(rel))))
            .collect();
        assert!(
            missing.is_empty(),
            "{heading} must cite a `file.rs::symbol` row for every production site the \
             regex sweep returns; uncited: {missing:?}"
        );
        assert!(
            section(&doc, heading).contains("**Regex reconciliation.**"),
            "{heading} must carry a `Regex reconciliation` note listing the non-site hits"
        );
    }
}

#[test]
fn no_output_competition_row_calls_a_dispatched_module_unreachable() {
    let dispatch = read(DISPATCH);
    assert!(
        dispatch.contains("output_competition::detect_output_competition"),
        "precondition: {DISPATCH} dispatches `output_competition::detect_output_competition` \
         (PR #2188); if that changes, re-verdict the ledger rows and this test"
    );

    let doc = read(RECORD);
    let rows: Vec<&str> = table_rows(&doc)
        .into_iter()
        .filter(|row| row.contains("output_competition.rs"))
        .collect();
    assert!(
        rows.len() >= 4,
        "precondition: the Files-swept, capacity, float and outcome tables each cite \
         `output_competition.rs`; found {} rows",
        rows.len()
    );
    for row in &rows {
        for phrase in UNREACHABLE_PHRASES {
            assert!(
                !row.contains(phrase),
                "an `output_competition.rs` row says `{phrase}` while {DISPATCH} dispatches \
                 the module: {row}"
            );
        }
    }

    let filed = section(&doc, "## Issues filed");
    let entry = filed
        .split("\n- ")
        .find(|entry| entry.starts_with(&format!("`{OUTPUT_COMPETITION_FINDING}`")))
        .unwrap_or_else(|| panic!("`## Issues filed` must list {OUTPUT_COMPETITION_FINDING}"));
    for label in ["`security`", "`lang:rust`", "`severity:", "`confidence:"] {
        assert!(
            entry.contains(label),
            "{OUTPUT_COMPETITION_FINDING}'s entry must record the {label} label: {entry}"
        );
    }
}

fn record_field<'a>(doc: &'a str, field: &str) -> &'a str {
    doc.split_once(&format!("**{field}:** `"))
        .and_then(|(_, rest)| rest.split_once('`'))
        .map_or_else(
            || panic!("{RECORD} must carry a `**{field}:**` field"),
            |(value, _)| value,
        )
}

#[test]
fn the_index_entry_equals_the_record_date_and_baseline() {
    let doc = read(RECORD);
    let index: Value =
        serde_json::from_str(&read(INDEX)).unwrap_or_else(|e| panic!("{INDEX} must parse: {e}"));
    let entry = index["chunks"]
        .as_array()
        .and_then(|chunks| chunks.iter().find(|c| c["id"] == "8b"))
        .unwrap_or_else(|| panic!("{INDEX} must carry a chunk `8b` entry"));
    assert_eq!(
        entry["last_swept"].as_str(),
        Some(record_field(&doc, "Sweep date")),
        "index `8b` last_swept must equal the record's Sweep date"
    );
    assert_eq!(
        entry["baseline_commit"].as_str(),
        Some(record_field(&doc, "Baseline commit")),
        "index `8b` baseline_commit must equal the record's Baseline commit"
    );
    assert_eq!(entry["record"].as_str(), Some(RECORD));
}
