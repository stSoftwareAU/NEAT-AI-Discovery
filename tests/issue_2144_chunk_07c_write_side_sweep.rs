//! Contract test for the write-side half of the chunk 7 sweep record,
//! `docs/audits/in-progress/security-sweep-chunk-07-parquet.md` (Issue #2144).
//!
//! It pins, for the three write-side files (`mod.rs`, `writer.rs` and
//! `shared_records.rs`):
//!
//! * each has exactly one `## Files swept` row whose outcome is settled: it
//!   starts with `clean` or `finding #<digits>` and never reads `pending`;
//! * every `finding #<digits>` those outcomes cite is also listed under
//!   `## Issues filed`, and the three files cite the issues that #2144 filed
//!   (#2449, #2452 and #2453);
//! * `## Defect classes probed` carries every write-side class heading, each
//!   followed by at least one table row.
//!
//! The last tests pin the `section` and `data_rows` helpers themselves, so a
//! parser that quietly returned an empty body could not make the others pass.

use std::any::Any;
use std::fs;
use std::panic;
use std::path::PathBuf;

/// The chunk 7 sweep record, still staged under `in-progress/`.
const RECORD: &str = "docs/audits/in-progress/security-sweep-chunk-07-parquet.md";

/// The three write-side files swept under #2144, in record order.
const WRITE_SIDE: [&str; 3] = [
    "src/parquet_format/mod.rs",
    "src/parquet_format/writer.rs",
    "src/parquet_format/shared_records.rs",
];

/// The defect classes the write-side sweep must record, each as a `###` heading.
const DEFECT_CLASSES: [&str; 5] = [
    "Write-path atomicity",
    "Merge safety",
    "Offset budget",
    "Cache-key soundness",
    "Public surface",
];

fn read(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// The body of the Markdown section whose heading line is exactly `heading`:
/// the lines after that heading, up to the next line starting with `## `.
///
/// Headings match whole lines only, so a `### Files swept` sub-heading never
/// matches `## Files swept`. Panics loudly if `heading` is absent.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let mut offset = 0;
    let mut body_start = None;
    let mut body_end = None;
    for line in doc.split_inclusive('\n') {
        let at = offset;
        offset += line.len();
        match body_start {
            None if line.trim_end() == heading => body_start = Some(offset),
            Some(_) if line.starts_with("## ") => {
                body_end = Some(at);
                break;
            }
            _ => {}
        }
    }
    let start = body_start.unwrap_or_else(|| {
        panic!("the record must carry the heading `{heading}` as a whole line (Issue #2144)")
    });
    &doc[start..body_end.unwrap_or(doc.len())]
}

/// Each table row under `section` whose first cell is a backticked path, as
/// its trimmed cells with the outer pipes removed. Cell 0 is the path (with
/// backticks) and, under `## Files swept`, cell 2 is the Outcome.
fn table_rows(section: &str) -> Vec<Vec<String>> {
    section
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("| `"))
        .map(|line| {
            line.trim_matches('|')
                .split('|')
                .map(|cell| cell.trim().to_string())
                .collect()
        })
        .collect()
}

/// The Outcome cell of the `## Files swept` row for `path`. Panics if no such
/// row, or no Outcome cell, exists.
fn outcome_of(rows: &[Vec<String>], path: &str) -> String {
    rows.iter()
        .find(|cells| cells[0].trim_matches('`') == path)
        .and_then(|cells| cells.get(2))
        .cloned()
        .unwrap_or_else(|| panic!("`## Files swept` must carry an Outcome row for {path}"))
}

/// The digits of a leading `finding #<digits>`, or `None` if `outcome` does
/// not start that way.
fn leading_finding(outcome: &str) -> Option<&str> {
    let rest = outcome.strip_prefix("finding #")?;
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    (digits > 0).then_some(&rest[..digits])
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

/// Whether `text` names the issue `reference` (`#<digits>`) as a whole number,
/// so `#244` does not match inside `#2447`.
fn names_issue(text: &str, reference: &str) -> bool {
    text.match_indices(reference)
        .any(|(at, _)| !text[at + reference.len()..].starts_with(|c: char| c.is_ascii_digit()))
}

/// The message carried by a caught panic payload, or an empty string if the
/// payload is neither a `String` nor a `&str`.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<String>() {
        return text.clone();
    }
    payload
        .downcast_ref::<&str>()
        .map_or_else(String::new, |text| (*text).to_string())
}

/// The data rows of a Markdown table in `block`: each line starting with `| `
/// that comes after the first separator line (one starting with `| ---`). The
/// header row and the separator are not counted.
fn data_rows(block: &str) -> usize {
    block
        .lines()
        .skip_while(|line| !line.starts_with("| ---"))
        .skip(1)
        .filter(|line| line.starts_with("| "))
        .count()
}

#[test]
fn each_write_side_file_has_exactly_one_settled_outcome() {
    let doc = read(RECORD);
    let rows = table_rows(section(&doc, "## Files swept"));
    for path in WRITE_SIDE {
        let matches = rows
            .iter()
            .filter(|cells| cells[0].trim_matches('`') == path)
            .count();
        assert_eq!(
            matches, 1,
            "`## Files swept` must carry exactly one row for {path} (Issue #2144), \
             found {matches}"
        );
        let outcome = outcome_of(&rows, path);
        assert!(
            outcome == "clean"
                || outcome.starts_with("clean ")
                || leading_finding(&outcome).is_some(),
            "{path}: `## Files swept` outcome must start with `clean` or \
             `finding #<digits>` (Issue #2144), found: {outcome}"
        );
        assert!(
            !outcome.contains("pending"),
            "{path}: `## Files swept` outcome must not still read pending (Issue #2144): \
             {outcome}"
        );
    }
}

#[test]
fn write_side_findings_name_issues_listed_under_issues_filed() {
    let doc = read(RECORD);
    let rows = table_rows(section(&doc, "## Files swept"));
    let issues_filed = section(&doc, "## Issues filed");

    let mut findings = Vec::new();
    for path in WRITE_SIDE {
        findings.extend(finding_refs(&outcome_of(&rows, path)));
    }
    for reference in ["#2449", "#2450", "#2452", "#2453"] {
        assert!(
            findings.iter().any(|found| found == reference),
            "the write-side outcomes must cite `finding {reference}` (Issue #2144)"
        );
    }
    for reference in &findings {
        assert!(
            names_issue(issues_filed, reference),
            "`## Issues filed` must list {reference}, which a write-side outcome cites \
             (Issue #2144)"
        );
    }
    assert!(
        names_issue(issues_filed, "#2451"),
        "`## Issues filed` must list #2451, the live-file truncation filed by #2144"
    );

    for (path, expected) in [
        ("src/parquet_format/mod.rs", "finding #2453"),
        ("src/parquet_format/writer.rs", "finding #2449"),
        ("src/parquet_format/shared_records.rs", "finding #2452"),
    ] {
        let outcome = outcome_of(&rows, path);
        assert!(
            outcome.starts_with(expected),
            "{path} outcome must start with `{expected}`, found: {outcome}"
        );
    }
}

#[test]
fn defect_classes_probed_records_every_write_side_class() {
    let doc = read(RECORD);
    let probed = section(&doc, "## Defect classes probed");
    for class in DEFECT_CLASSES {
        let heading = format!("### {class}");
        let mut lines = probed.lines();
        assert!(
            lines.by_ref().any(|line| line.trim_end() == heading),
            "`## Defect classes probed` must carry the heading `{heading}` (Issue #2144)"
        );
        let body = lines
            .take_while(|line| !line.starts_with("### "))
            .collect::<Vec<_>>()
            .join("\n");
        let rows = data_rows(&body);
        assert!(
            rows > 0,
            "`{heading}` must record at least one table row before the next `### ` \
             heading (Issue #2144)"
        );
    }
}

#[test]
fn header_only_table_has_no_data_rows() {
    assert_eq!(data_rows("| A | B |\n| --- | --- |\n"), 0);
    assert_eq!(data_rows("| A | B |\n| --- | --- |\n| x | y |\n"), 1);
}

#[test]
fn parser_rejects_a_missing_section_loudly() {
    let doc = "# Title\n\n### Files swept\nsub body\n\n## Files swept\nreal body\n\n\
               ## Outcome\nok\n";
    assert_eq!(
        section(doc, "## Files swept").trim(),
        "real body",
        "`## Files swept` must return its own body, not the `### Files swept` sub-heading"
    );
    assert_eq!(
        section(doc, "### Files swept").trim(),
        "sub body",
        "the `### Files swept` sub-heading must be found as its own whole line"
    );

    let missing = panic::catch_unwind(|| section(doc, "## Missing"));
    let payload = missing.expect_err("a missing heading must panic, not return an empty body");
    let message = panic_message(&*payload);
    assert!(
        message.contains("## Missing"),
        "the panic for a missing heading must name it, found: {message}"
    );
}
