//! Contract tests for the chunk 13 sweep record scaffold (Issue #2211).
//!
//! Chunk 13 (`src/config`) is swept by several audit sub-issues writing into
//! **one** shared record. The scaffold lays out every region up front, with
//! its marker, header row and pending line, so each sub-issue only fills its
//! own region. Without a gate, that shape fails quietly in three ways, and
//! each of them looks like coverage from the outside:
//!
//! * a file in scope never gets a row, so it is never swept and nobody notices;
//! * a region loses its `<!-- section: … -->` marker, or a marker is duplicated,
//!   so two sub-issues append findings to the same place and conflict on every
//!   PR;
//! * a settings header drifts from the ten agreed columns, so rows filled in by
//!   different sub-issues no longer line up.
//!
//! These tests pin the scaffold's shape only. They deliberately do not assert
//! that every outcome is decided or that every region is populated: the
//! settings and drift regions start empty, and later sub-issues fill them.
//! `tests/issue_2088_sweep_ledger_contract.rs` covers the ledger-wide rules;
//! this file covers only chunk 13's own shape.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The chunk 13 prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-13-config-env.md";
/// Machine-readable sweep index — must name the record above.
const INDEX: &str = "docs/audits/lib-sweep-coverage.json";

/// The baseline commit chunk 13 was cut against (Issue #2224).
const BASELINE: &str = "b85a551ed2521ed327469b20eb88aeda828357d2";
/// The tracker issue the chunk 13 record reports into.
const PARENT_ISSUE: &str = "#2096";

/// The one file split across two regions of the per-file table, by line.
const USER_FACING: &str = "src/config/user_facing.rs";
/// `user_facing.rs` at the baseline commit. Its two row counts must sum to this.
const USER_FACING_BASELINE_LINES: usize = 1396;

/// Source root the chunk covers. Every `.rs` file beneath it needs a row.
const SCOPE: &str = "src/config";

/// The `###` regions shared by `## Files swept` and `## Settings table`, in
/// record order, each paired with the id of its section marker.
const REGIONS: [(&str, &str); 4] = [
    ("### small modules", "13a"),
    ("### user_facing L1-664", "13b"),
    ("### user_facing L665-end", "13c"),
    ("### readers outside src/config", "13d"),
];

/// The ten columns every settings header row carries, in order. Rows filled in
/// by different sub-issues only line up if every header agrees.
const SETTINGS_COLUMNS: [&str; 10] = [
    "variable",
    "reader file:line",
    "type",
    "default",
    "accepted range / validation",
    "on invalid input",
    "0 / opt-out semantics",
    "can weaken a guard? (which)",
    "cached (OnceLock) or live",
    "logged how",
];

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
            Some(FileRow { path, lines })
        })
        .collect()
}

/// The cells of every settings header row (first cell `variable`) in `body`.
fn settings_header_rows(body: &str) -> Vec<Vec<String>> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .map(|line| {
            line.trim()
                .trim_matches('|')
                .split('|')
                .map(|cell| cell.trim().to_string())
                .collect::<Vec<String>>()
        })
        .filter(|cells| cells.first().is_some_and(|c| c.as_str() == "variable"))
        .collect()
}

fn files_swept_body() -> String {
    section(&read(RECORD), "## Files swept").to_string()
}

/// The HTML comment that opens one region of a table or findings list.
fn marker_for(id: &str) -> String {
    format!("<!-- section: {id} -->")
}

fn is_iso_date(value: &str) -> bool {
    value.len() == 10
        && value.chars().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                c == '-'
            } else {
                c.is_ascii_digit()
            }
        })
}

#[test]
fn the_record_exists_and_pins_its_chunk_id_exposure_baseline_and_parent() {
    let doc = read(RECORD);
    for needle in ["`13`", "**Exposure:** `local`", BASELINE, PARENT_ISSUE] {
        assert!(
            doc.contains(needle),
            "{RECORD} must carry `{needle}` — a record that does not pin its chunk id, \
             exposure, baseline commit and tracker issue cannot be checked by a later reader"
        );
    }

    let Some((date, _)) = doc
        .split_once("**Sweep date:** `")
        .and_then(|(_, rest)| rest.split_once('`'))
    else {
        panic!("{RECORD} must carry a `**Sweep date:** \\`…\\`` field");
    };
    assert!(
        is_iso_date(date),
        "{RECORD} sweep date `{date}` must be an ISO YYYY-MM-DD date"
    );
}

#[test]
fn every_config_file_has_one_row_and_user_facing_is_split_across_two_regions() {
    let body = files_swept_body();
    let rows = file_rows(&body);

    let on_disk: BTreeSet<String> = rust_files_under(SCOPE).into_iter().collect();
    let small_expected: BTreeSet<String> = on_disk
        .iter()
        .filter(|path| path.as_str() != USER_FACING)
        .cloned()
        .collect();

    // user_facing.rs legitimately appears twice (one row per line range), so the
    // duplicate check covers every other file.
    let mut tabled: BTreeSet<String> = BTreeSet::new();
    for row in rows.iter().filter(|row| row.path != USER_FACING) {
        assert!(
            tabled.insert(row.path.clone()),
            "{} appears twice in the per-file tables — two regions would record \
             conflicting outcomes for one file",
            row.path
        );
    }

    let missing: Vec<&String> = small_expected.difference(&tabled).collect();
    assert!(
        missing.is_empty(),
        "these files in {SCOPE} have no row in {RECORD}, so they would never be swept: {missing:?}"
    );
    let stale: Vec<&String> = tabled
        .iter()
        .filter(|path| path.starts_with("src/config/") && !on_disk.contains(path.as_str()))
        .collect();
    assert!(
        stale.is_empty(),
        "{RECORD} carries rows for {SCOPE} files that no longer exist: {stale:?}"
    );

    // Every file except user_facing.rs sits under `### small modules`, and only
    // there, so each has exactly one owning region.
    let small_rows = file_rows(section(&body, "### small modules"));
    let mut small_tabled: BTreeSet<String> = BTreeSet::new();
    for row in &small_rows {
        assert!(
            small_tabled.insert(row.path.clone()),
            "{} appears twice under `### small modules`",
            row.path
        );
    }
    assert_eq!(
        small_tabled, small_expected,
        "`### small modules` must carry exactly one row per {SCOPE} file other than \
         {USER_FACING} — a row anywhere else has no owning region"
    );

    // user_facing.rs is split by line: one row in each range, and no third.
    let user_facing_total = rows.iter().filter(|row| row.path == USER_FACING).count();
    assert_eq!(
        user_facing_total, 2,
        "{USER_FACING} must have exactly two rows in {RECORD}, one per line range"
    );
    let low = file_rows(section(&body, "### user_facing L1-664"));
    assert_eq!(
        low.len(),
        1,
        "`### user_facing L1-664` must carry exactly one row"
    );
    assert_eq!(
        low[0].path, USER_FACING,
        "`### user_facing L1-664` must carry {USER_FACING}"
    );
    let high = file_rows(section(&body, "### user_facing L665-end"));
    assert_eq!(
        high.len(),
        1,
        "`### user_facing L665-end` must carry exactly one row"
    );
    assert_eq!(
        high[0].path, USER_FACING,
        "`### user_facing L665-end` must carry {USER_FACING}"
    );
    assert_eq!(
        low[0].lines + high[0].lines,
        USER_FACING_BASELINE_LINES,
        "the two {USER_FACING} rows must sum to its baseline line count \
         {USER_FACING_BASELINE_LINES}, or a line range has been dropped or double-counted"
    );

    for row in &rows {
        assert!(
            row.lines > 0,
            "{} must carry its line count at the baseline commit",
            row.path
        );
    }
}

#[test]
fn each_section_marker_appears_exactly_once_in_its_section() {
    let doc = read(RECORD);

    // Each `##` section, the marker ids it carries, and the number of times each
    // marker must appear inside that section.
    let plan: [(&str, &[&str]); 4] = [
        ("## Files swept", &["13a", "13b", "13c", "13d"]),
        ("## Settings table", &["13a", "13b", "13c", "13d"]),
        (
            "## Issues filed",
            &["13a-findings", "13a", "13b", "13c", "13d"],
        ),
        ("## Doc drift observed", &["13a", "13b", "13c", "13d"]),
    ];

    let mut expected_total = 0usize;
    for (heading, ids) in plan {
        let body = section(&doc, heading);
        for id in ids {
            let marker = marker_for(id);
            let count = body.matches(&marker).count();
            assert_eq!(
                count, 1,
                "`{heading}` must carry `{marker}` exactly once so its region is \
                 unambiguous to the sub-issue that owns it, found {count}"
            );
        }
        expected_total += ids.len();
    }

    // No marker may sit outside the four sections, or in prose.
    let all_markers = doc.matches("<!-- section: ").count();
    assert_eq!(
        all_markers, expected_total,
        "{RECORD} must carry exactly {expected_total} section markers, all inside the \
         sections that own them — a stray marker is a second region nobody coordinates"
    );
}

#[test]
fn the_settings_table_has_one_ten_column_header_per_region() {
    let doc = read(RECORD);
    let settings = section(&doc, "## Settings table");

    let all_headers = settings_header_rows(settings);
    assert_eq!(
        all_headers.len(),
        REGIONS.len(),
        "`## Settings table` must carry exactly one header row per region, found {}",
        all_headers.len()
    );

    for (heading, id) in REGIONS {
        let headers = settings_header_rows(section(settings, heading));
        assert_eq!(
            headers.len(),
            1,
            "`{heading}` in `## Settings table` (region {id}) must carry exactly one header row"
        );
        let columns: Vec<&str> = headers[0].iter().map(String::as_str).collect();
        assert_eq!(
            columns, SETTINGS_COLUMNS,
            "`{heading}` in `## Settings table` must carry the ten agreed columns in order — \
             rows filled in by different sub-issues only line up if every header agrees"
        );
    }
}

#[test]
fn the_index_names_the_chunk_13_record_and_its_baseline() {
    let index = read(INDEX);
    assert!(
        index.contains(RECORD),
        "{INDEX} must name {RECORD} — a prose record with no index entry is invisible \
         to the next automated sweep"
    );
    assert!(
        index.contains(BASELINE),
        "{INDEX} must pin chunk 13 to baseline {BASELINE}"
    );
}
