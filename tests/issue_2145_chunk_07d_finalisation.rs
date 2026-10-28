//! Finalisation contract for the chunk 7 sweep record (Issue #2145).
//!
//! Issues #2143 (read side) and #2144 (write side) each swept part of the
//! chunk 7 Parquet ledger, which #2279 had staged under
//! `docs/audits/in-progress/`. #2145 reconciled the tables, wrote the
//! `## Outcome` and `## Issues filed` sections, `git mv`d the record to the top
//! level and filled the `7` entry of `docs/audits/lib-sweep-coverage.json`.
//! This file pins the finished state so the ledger cannot quietly slide back to
//! a partial one:
//!
//! * the record is promoted out of `in-progress/`, and that directory is gone;
//! * no line of the record reads `pending`, `placeholder` or `in progress`, and
//!   the status heading says the sweep is complete;
//! * `## Files swept` has one row per `src/parquet_format/*.rs` file, the line
//!   counts sum to 2,642 and each outcome reads `<outcome> — <reason>`;
//! * `## Issues filed` lists each of the seven findings once, with its
//!   `SEC-2091-NN` id and severity, and `## Outcome` agrees with that list;
//! * the index's `7` entry equals the record's sweep date, baseline commit,
//!   path and tracker issue.

use std::path::PathBuf;

use serde_json::Value;

/// The promoted, top-level record: the state finalisation produces.
const RECORD: &str = "docs/audits/security-sweep-chunk-07-parquet.md";
/// The staged copy #2145 `git mv`s away. It must no longer exist.
const STAGED: &str = "docs/audits/in-progress/security-sweep-chunk-07-parquet.md";
/// The staging directory, which must be gone once the record is promoted.
const IN_PROGRESS_DIR: &str = "docs/audits/in-progress";
/// Machine-readable sweep index. Its `"7"` entry is filled at finalisation.
const INDEX: &str = "docs/audits/lib-sweep-coverage.json";
/// The baseline every line count and citation in the record is pinned to.
const BASELINE: &str = "b85a551ed2521ed327469b20eb88aeda828357d2";
/// The directory chunk 7 sweeps, scanned non-recursively.
const PARQUET_DIR: &str = "src/parquet_format";
/// The nine files of `PARQUET_DIR`, in record order.
const FILES: [&str; 9] = [
    "src/parquet_format/mod.rs",
    "src/parquet_format/footer.rs",
    "src/parquet_format/schema.rs",
    "src/parquet_format/decode_budget.rs",
    "src/parquet_format/reader.rs",
    "src/parquet_format/batch_columns.rs",
    "src/parquet_format/writer.rs",
    "src/parquet_format/shared_records.rs",
    "src/parquet_format/activation_summary.rs",
];
/// The line total the `## Files swept` rows must sum to at the baseline.
const TOTAL_LINES: usize = 2642;
/// The findings filed from this chunk as `(issue, severity)`, in `SEC-2091-NN`
/// order: the first is `SEC-2091-01`, the seventh `SEC-2091-07`.
const FILED: [(&str, &str); 7] = [
    ("#2446", "medium"),
    ("#2447", "low"),
    ("#2449", "medium"),
    ("#2450", "low"),
    ("#2451", "low"),
    ("#2452", "low"),
    ("#2453", "low"),
];
/// Words that mark a line of the record as unfinished, matched case-insensitively.
const UNFINISHED: [&str; 3] = ["pending", "placeholder", "in progress"];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// The text of a Markdown section, from its heading **line** to the next
/// heading line of the same or a higher level.
///
/// Line-anchored (matches a whole heading line via `line.trim_end() ==
/// heading`), so `## Outcome` cannot match inside a longer heading or a prose
/// sentence that happens to contain the same text.
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

/// The trimmed cells of one `| a | b |` Markdown row, outer pipes removed.
fn row_cells(line: &str) -> Vec<&str> {
    line.trim()
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect()
}

/// The data rows of the table in `body`: every `|` line after the header row,
/// with the `| --- |` separator dropped.
fn data_rows(body: &str) -> Vec<Vec<&str>> {
    body.lines()
        .map(str::trim)
        .filter(|line| line.starts_with('|'))
        .map(row_cells)
        .filter(|cells| !cells[0].starts_with("---"))
        .skip(1)
        .collect()
}

/// The backticked value of the record's bold `field` line, such as `Sweep date`.
fn record_field<'a>(doc: &'a str, field: &str) -> &'a str {
    doc.split_once(&format!("**{field}:** `"))
        .and_then(|(_, rest)| rest.split_once('`'))
        .map_or_else(
            || panic!("{RECORD} must carry a `**{field}:**` field"),
            |(value, _)| value,
        )
}

/// Every `finding #<digits>` in `text`, returned as `#<digits>`, in order.
fn finding_refs(text: &str) -> Vec<String> {
    const PREFIX: &str = "finding #";
    text.match_indices(PREFIX)
        .filter_map(|(at, _)| {
            let digits: String = text[at + PREFIX.len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            (!digits.is_empty()).then(|| format!("#{digits}"))
        })
        .collect()
}

/// `true` when `text` is an ISO `YYYY-MM-DD` date.
fn is_iso_date(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 10
        && bytes.iter().enumerate().all(|(i, b)| {
            if i == 4 || i == 7 {
                *b == b'-'
            } else {
                b.is_ascii_digit()
            }
        })
}

#[test]
fn the_record_is_promoted_out_of_in_progress() {
    let doc = read(RECORD);
    assert!(
        !doc.trim().is_empty(),
        "{RECORD} must exist and carry content (Issue #2145)"
    );
    assert!(
        !repo_root().join(STAGED).exists(),
        "{STAGED} must no longer exist once Issue #2145 has promoted {RECORD}"
    );
    assert!(
        !repo_root().join(IN_PROGRESS_DIR).exists(),
        "{IN_PROGRESS_DIR} must no longer exist once Issue #2145 has promoted {RECORD}"
    );
}

#[test]
fn no_unfinished_wording_survives_in_the_record() {
    let doc = read(RECORD);
    let offending: Vec<usize> = doc
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            let lower = line.to_lowercase();
            UNFINISHED.iter().any(|word| lower.contains(*word))
        })
        .map(|(i, _)| i + 1)
        .collect();
    assert!(
        offending.is_empty(),
        "{RECORD} must not read `pending`, `placeholder` or `in progress` once finalised \
         (Issue #2145); offending line numbers: {offending:?}"
    );
    assert!(
        doc.lines()
            .any(|line| line.trim_end() == "### Sweep status — COMPLETE"),
        "{RECORD} must carry the heading `### Sweep status — COMPLETE` (Issue #2145)"
    );
}

#[test]
fn files_swept_has_one_row_per_parquet_file_and_the_lines_total_2642() {
    let doc = read(RECORD);
    let (totals, file_rows): (Vec<Vec<&str>>, Vec<Vec<&str>>) =
        data_rows(section(&doc, "## Files swept"))
            .into_iter()
            .partition(|cells| cells[0] == "**Total**");

    for cells in &file_rows {
        assert_eq!(
            cells.len(),
            3,
            "a `## Files swept` row must have three cells (path, lines, outcome), got {cells:?}"
        );
    }

    let mut recorded: Vec<&str> = file_rows
        .iter()
        .map(|cells| cells[0].trim_matches('`'))
        .collect();
    recorded.sort_unstable();
    let mut wanted: Vec<&str> = FILES.to_vec();
    wanted.sort_unstable();
    assert_eq!(
        recorded, wanted,
        "`## Files swept` must carry exactly one row per {PARQUET_DIR} file (Issue #2145)"
    );

    let mut on_disk: Vec<String> = std::fs::read_dir(repo_root().join(PARQUET_DIR))
        .unwrap_or_else(|e| panic!("{PARQUET_DIR} must be readable: {e}"))
        .filter_map(|entry| {
            let path = entry.expect("directory entry must be readable").path();
            let name = path.file_name()?.to_str()?.to_string();
            (path.is_file() && path.extension().is_some_and(|e| e == "rs"))
                .then(|| format!("{PARQUET_DIR}/{name}"))
        })
        .collect();
    on_disk.sort_unstable();
    assert_eq!(
        on_disk.iter().map(String::as_str).collect::<Vec<_>>(),
        wanted,
        "`## Files swept` must match the on-disk `.rs` files of {PARQUET_DIR} (non-recursive)"
    );

    let mut sum = 0usize;
    for cells in &file_rows {
        let lines: usize = cells[1].replace(',', "").parse().unwrap_or_else(|e| {
            panic!(
                "{}'s Lines cell must parse as a count, got `{}`: {e}",
                cells[0], cells[1]
            )
        });
        sum += lines;
    }
    assert_eq!(
        sum, TOTAL_LINES,
        "the `## Files swept` line counts must sum to {TOTAL_LINES} (the baseline total)"
    );

    assert_eq!(
        totals.len(),
        1,
        "`## Files swept` must carry exactly one `**Total**` row"
    );
    let total_cell = totals[0].get(1).map_or("", |cell| cell.trim_matches('*'));
    assert_eq!(
        total_cell, "2,642",
        "the `## Files swept` **Total** row must read 2,642, got `{total_cell}`"
    );

    for cells in &file_rows {
        let outcome = cells[2];
        let Some((verdict, reason)) = outcome.split_once(" — ") else {
            panic!(
                "{} must read `<outcome> — <reason>`, got: {outcome}",
                cells[0]
            );
        };
        assert!(
            !verdict.trim().is_empty() && !reason.trim().is_empty(),
            "{} must carry both an outcome and a reason, got: {outcome}",
            cells[0]
        );
    }
}

#[test]
fn issues_filed_lists_each_finding_with_its_severity_and_sec_id() {
    let doc = read(RECORD);
    let rows = data_rows(section(&doc, "## Issues filed"));
    assert_eq!(
        rows.len(),
        FILED.len(),
        "`## Issues filed` must carry exactly {} rows, one per finding, got {}",
        FILED.len(),
        rows.len()
    );

    for (ordinal, (issue, severity)) in FILED.iter().enumerate() {
        let matching: Vec<&Vec<&str>> = rows
            .iter()
            .filter(|cells| cells[0].starts_with(*issue))
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "`## Issues filed` must list {issue} in exactly one row, found {}",
            matching.len()
        );
        let cells = matching[0];
        let sec_id = format!("(SEC-2091-{:02})", ordinal + 1);
        assert!(
            cells[0].contains(sec_id.as_str()),
            "{issue}'s `## Issues filed` row must carry `{sec_id}` in its first cell, got: {}",
            cells[0]
        );
        assert_eq!(
            cells.get(1).copied(),
            Some(*severity),
            "{issue}'s `## Issues filed` severity cell must read `{severity}`, got row: {cells:?}"
        );
    }

    let mut cited: Vec<String> = Vec::new();
    for cells in data_rows(section(&doc, "## Files swept")) {
        if let Some(outcome) = cells.get(2) {
            cited.extend(finding_refs(outcome));
        }
    }
    assert!(
        !cited.is_empty(),
        "precondition: `## Files swept` must cite at least one `finding #N` (Issue #2145)"
    );
    for reference in &cited {
        assert!(
            FILED.iter().any(|(issue, _)| *issue == reference.as_str()),
            "`## Files swept` cites `finding {reference}`, which is not among the filed issues \
             {FILED:?}"
        );
    }

    let outcome = section(&doc, "## Outcome");
    for (issue, _) in FILED {
        assert!(
            outcome.contains(issue),
            "`## Outcome` must mention {issue}, which `## Issues filed` lists"
        );
    }
    assert!(
        outcome.contains("2,642"),
        "`## Outcome` must state the 2,642-line total"
    );
    for line in outcome
        .lines()
        .filter(|line| line.contains("negative-result"))
    {
        assert!(
            line.contains("not applied"),
            "`## Outcome` may mention `negative-result` only in a sentence saying the label was \
             not applied, got: {line}"
        );
    }
}

#[test]
fn the_index_entry_equals_the_record_date_baseline_path_and_issue() {
    let doc = read(RECORD);
    let index: Value =
        serde_json::from_str(&read(INDEX)).unwrap_or_else(|e| panic!("{INDEX} must parse: {e}"));
    let entry = index["chunks"]
        .as_array()
        .and_then(|chunks| chunks.iter().find(|c| c["id"] == "7"))
        .unwrap_or_else(|| panic!("{INDEX} must carry a chunk `7` entry"));

    let sweep_date = record_field(&doc, "Sweep date");
    assert!(
        is_iso_date(sweep_date),
        "the record's Sweep date must be ISO YYYY-MM-DD, got `{sweep_date}`"
    );
    assert_eq!(
        entry["last_swept"].as_str(),
        Some(sweep_date),
        "index `7` last_swept must equal the record's Sweep date"
    );
    assert!(
        entry["last_swept"].as_str().is_some_and(is_iso_date),
        "index `7` last_swept must be ISO YYYY-MM-DD"
    );
    assert_eq!(
        entry["baseline_commit"].as_str(),
        Some(BASELINE),
        "index `7` baseline_commit must equal BASELINE"
    );
    assert_eq!(
        record_field(&doc, "Baseline commit"),
        BASELINE,
        "the record's Baseline commit field must equal BASELINE"
    );
    assert_eq!(
        entry["record"].as_str(),
        Some(RECORD),
        "index `7` record must equal RECORD"
    );
    assert_eq!(
        entry["issue"].as_u64(),
        Some(2091),
        "index `7` issue must be the chunk tracker, #2091"
    );
}
