//! Contract tests for the chunk 8b sweep record (Issue #2103).
//!
//! Chunk 8b is swept by seven audit sub-issues writing into **one** shared
//! record. Without a gate, that shape fails quietly in three ways, and each of
//! them looks like coverage from the outside:
//!
//! * a file in scope never gets a row, so it is never swept and nobody notices;
//! * a sub-issue finishes but leaves its rows reading `pending`, so a finished
//!   sweep is indistinguishable from an unstarted one;
//! * the per-section `<!-- section: … -->` markers drift, so two sub-issues
//!   append findings to the same region of a table and conflict on every PR.
//!
//! These tests read the record as the artefact it is and assert all three.
//! `tests/issue_2088_sweep_ledger_contract.rs` covers the ledger-wide rules
//! (index parity, baseline SHAs); this file covers only chunk 8b's own shape.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The chunk 8b prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md";
/// Machine-readable sweep index — must name the record above.
const INDEX: &str = "docs/audits/lib-sweep-coverage.json";

/// The baseline commit chunk 8b was cut against (Issue #2093).
const BASELINE: &str = "b85a551ed2521ed327469b20eb88aeda828357d2";

/// Source roots the chunk covers. Every `.rs` file beneath these needs a row.
const SCOPE: [&str; 4] = [
    "src/analysis/synapse",
    "src/analysis/scoring",
    "src/analysis/recommendation",
    "src/analysis/shared",
];

/// One `###` section per chunk 8b audit sub-issue, in record order. Each
/// sub-issue edits only its own section, so concurrent PRs do not conflict.
const SECTIONS: [&str; 7] = [
    "synapse pipeline",
    "synapse post-processing",
    "synapse scoring + target_analysis",
    "scoring",
    "recommendation core",
    "recommendation batch_successful + epistatic",
    "shared",
];

/// The files Issue #2103 itself swept — these must not read `pending`.
const SHARED_FILES: [&str; 4] = [
    "src/analysis/shared/mod.rs",
    "src/analysis/shared/gpu_info.rs",
    "src/analysis/shared/metadata.rs",
    "src/analysis/shared/timing.rs",
];

/// Total in-scope file count named by Issue #2103.
const EXPECTED_FILE_COUNT: usize = 58;

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

    // Walk candidate headings until one sits at the same or a higher level.
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

/// Every `.rs` file beneath `root`, repo-relative, with forward slashes.
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

/// A parsed row of the per-file outcome table.
#[derive(Debug)]
struct FileRow {
    path: String,
    lines: usize,
    outcome: String,
}

/// Parse the three-cell path / line-count / outcome rows out of a chunk of
/// Markdown, skipping header, separator and non-`src/` rows.
fn file_rows(body: &str) -> Vec<FileRow> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .filter_map(|line| {
            let cells: Vec<&str> = line.trim().trim_matches('|').split('|').collect();
            if cells.len() != 3 {
                return None;
            }
            let path = cells[0].trim().trim_matches('`').to_string();
            if !path.starts_with("src/") {
                return None;
            }
            let lines = cells[1]
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("row for {path} must carry a numeric line count: {e}"));
            Some(FileRow {
                path,
                lines,
                outcome: cells[2].trim().to_string(),
            })
        })
        .collect()
}

fn files_swept_body() -> String {
    section(&read(RECORD), "## Files swept").to_string()
}

#[test]
fn the_record_exists_and_pins_its_chunk_id_baseline_and_parent() {
    let doc = read(RECORD);
    for needle in [
        "`8b`",
        "**Exposure:** `internal`",
        BASELINE,
        "#2093",
        "**Sweep date:** `2026-09-23`",
    ] {
        assert!(
            doc.contains(needle),
            "{RECORD} must carry `{needle}` — a record that does not pin its chunk id, \
             exposure, baseline commit and parent issue cannot be checked by a later reader"
        );
    }
}

#[test]
fn the_index_names_the_chunk_8b_record() {
    let index = read(INDEX);
    assert!(
        index.contains(RECORD),
        "{INDEX} must name {RECORD} — a prose record with no index entry is invisible \
         to the next automated sweep"
    );
    assert!(
        index.contains(BASELINE),
        "{INDEX} must pin chunk 8b to baseline {BASELINE}"
    );
}

#[test]
fn every_in_scope_file_has_exactly_one_row() {
    let rows = file_rows(&files_swept_body());

    let mut on_disk: BTreeSet<String> = BTreeSet::new();
    for root in SCOPE {
        on_disk.extend(rust_files_under(root));
    }

    let mut tabled: BTreeSet<String> = BTreeSet::new();
    for row in &rows {
        assert!(
            tabled.insert(row.path.clone()),
            "{} appears twice in the per-file table — two sub-issues would record \
             conflicting outcomes for one file",
            row.path
        );
    }

    let missing: Vec<&String> = on_disk.difference(&tabled).collect();
    assert!(
        missing.is_empty(),
        "these in-scope files have no row in {RECORD}, so they would never be swept: {missing:?}"
    );
    let stale: Vec<&String> = tabled.difference(&on_disk).collect();
    assert!(
        stale.is_empty(),
        "{RECORD} carries rows for files that no longer exist: {stale:?}"
    );
    assert_eq!(
        rows.len(),
        EXPECTED_FILE_COUNT,
        "chunk 8b covers {EXPECTED_FILE_COUNT} files (Issue #2093)"
    );
}

#[test]
fn every_row_sits_under_one_named_sub_issue_section() {
    let body = files_swept_body();
    let mut counted = 0usize;

    for name in SECTIONS {
        let heading = format!("### {name}");
        let rows = file_rows(section(&body, &heading));
        assert!(
            !rows.is_empty(),
            "section `{heading}` must carry the rows its audit sub-issue owns"
        );
        counted += rows.len();
    }

    assert_eq!(
        counted,
        file_rows(&body).len(),
        "every per-file row must sit under exactly one `###` sub-issue section — a row \
         outside them has no owner, and two sub-issues editing one region conflict"
    );
}

#[test]
fn every_row_carries_an_outcome() {
    for row in file_rows(&files_swept_body()) {
        assert!(
            !row.outcome.is_empty(),
            "{} must carry an outcome — an empty cell is indistinguishable from an \
             unfinished sweep",
            row.path
        );
        assert!(
            row.lines > 0,
            "{} must carry its line count at the baseline commit",
            row.path
        );
    }
}

#[test]
fn the_shared_rows_are_swept_with_a_reason() {
    let rows = file_rows(section(&files_swept_body(), "### shared"));

    assert_eq!(
        rows.len(),
        SHARED_FILES.len(),
        "the `shared` section must carry one row per file in src/analysis/shared/"
    );

    for expected in SHARED_FILES {
        let row = rows
            .iter()
            .find(|row| row.path == expected)
            .unwrap_or_else(|| panic!("{RECORD} must carry a row for {expected}"));
        assert!(
            !row.outcome.contains("pending"),
            "{expected} was swept by Issue #2103, so its outcome must not read `pending`: {}",
            row.outcome
        );
        // "clean" on its own is an unfalsifiable claim: the reason is what a
        // later reader checks the sweep against.
        let reason = row.outcome.split_once('—').map(|(_, rest)| rest.trim());
        assert!(
            reason.is_some_and(|r| r.len() > 20),
            "{expected} must state a one-line reason after its outcome, got: {}",
            row.outcome
        );
    }
}

#[test]
fn the_shared_sweep_traces_the_timing_collector_race_class() {
    let doc = read(RECORD);
    let outcome = section(&doc, "### shared (Issue #2103)");
    for needle in [
        "TimingCollector",
        "record_shader",
        "finalize",
        "parking_lot::Mutex",
        "fetch_add",
        "par_iter",
        "from_env",
        "interior mutability",
    ] {
        assert!(
            outcome.contains(needle),
            "the `shared` outcome must show that `{needle}` was actually traced — \
             an outcome that names no reader, writer or dispatch site is a claim, not a sweep"
        );
    }
}

#[test]
fn both_finding_tables_carry_one_marker_per_sub_issue_in_section_order() {
    let doc = read(RECORD);

    for heading in ["## Capacity-from-input sites", "## Float comparison sites"] {
        let body = section(&doc, heading);
        let mut cursor = 0usize;
        for name in SECTIONS {
            let marker = format!("<!-- section: {name} -->");
            let offset = body[cursor..].find(&marker).unwrap_or_else(|| {
                panic!(
                    "`{heading}` must carry `{marker}` after the markers before it, so each \
                     sub-issue's rows land in a disjoint region"
                )
            });
            cursor = cursor + offset + marker.len();
        }
    }
}

#[test]
fn both_finding_tables_fix_their_columns() {
    let doc = read(RECORD);

    let capacity = section(&doc, "## Capacity-from-input sites");
    for column in ["`file:line`", "Expression", "Bound source", "Verdict"] {
        assert!(
            capacity.contains(column),
            "the capacity-from-input table must carry a `{column}` column so every \
             sub-issue records the same evidence"
        );
    }

    let floats = section(&doc, "## Float comparison sites");
    for column in [
        "`file:line`",
        "Comparator",
        "Value origin",
        "NaN handling",
        "Verdict",
    ] {
        assert!(
            floats.contains(column),
            "the float-comparison table must carry a `{column}` column"
        );
    }
}
