//! Chunk 11b-1 ledger contract: the `debug + sampler` section of the chunk 11
//! record (Issue #2251).
//!
//! The scaffold (Issue #2233) cut every row as `pending`. This file pins the
//! swept state of the debug + sampler section only — the watchdog section is
//! owned by a sibling sub-issue and may still read `pending`:
//!
//! * each of the four inventory rows carries `<outcome> — <reason>`, and the
//!   `sample_capture.rs` row links the finding it filed (#2266);
//! * the section's region of the mutation table cites all five sites, and each
//!   cited symbol still exists in its source file (Issue #1942);
//! * the section's region of the re-verification table carries a #1905 and a
//!   #1904 row, and the guard tests they cite exist.

use std::path::PathBuf;

const RECORD: &str = "docs/audits/security-sweep-chunk-11-filesystem-lifecycle.md";
const SECTION: &str = "debug + sampler";

const FILES: [&str; 4] = [
    "src/debug.rs",
    "src/debug/sample_capture.rs",
    "src/debug/sample_dir.rs",
    "src/debug/process_state.rs",
];

/// The finding this sweep filed, linked from the `sample_capture.rs` row.
const FINDING: &str = "#2266";

/// `(cited site, source file, definition the symbol must still match)`.
const MUTATION_SITES: [(&str, &str, &str); 5] = [
    (
        "sample_dir.rs::create_private_dir",
        "src/debug/sample_dir.rs",
        "fn create_private_dir(",
    ),
    (
        "sample_dir.rs::SampleDir::create",
        "src/debug/sample_dir.rs",
        "fn create(pid: u32)",
    ),
    (
        "sample_dir.rs::Drop::drop",
        "src/debug/sample_dir.rs",
        "impl Drop for SampleDir",
    ),
    (
        "sample_dir.rs::read_guarded",
        "src/debug/sample_dir.rs",
        "fn read_guarded(",
    ),
    (
        "sample_capture.rs::run_external_command_with_timeout",
        "src/debug/sample_capture.rs",
        "fn run_external_command_with_timeout(",
    ),
];

/// `(issue, guard surface the row must cite)`.
const REVERIFIED: [(&str, &str); 2] = [
    ("#1905", "tests/issue_1905_sample_temp_dir.rs"),
    ("#1904", "src/analysis/utils/platform.rs"),
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
fn no_debug_sampler_row_reads_pending_and_each_has_an_outcome_and_a_reason() {
    let doc = read(RECORD);
    let files_swept = section(&doc, "## Files swept");
    let rows = file_rows(section(files_swept, &format!("### {SECTION}")));

    let paths: Vec<&str> = rows.iter().map(|(path, _)| path.as_str()).collect();
    assert_eq!(
        paths, FILES,
        "the section must carry exactly its four files"
    );

    for (path, outcome) in &rows {
        assert!(
            !outcome.contains("pending"),
            "{path} still reads `pending`: {outcome}"
        );
        let Some((verdict, reason)) = outcome.split_once(" — ") else {
            panic!("{path} must read `<outcome> — <reason>`, got: {outcome}");
        };
        assert!(
            !verdict.trim().is_empty() && !reason.trim().is_empty(),
            "{path} must carry both an outcome and a reason: {outcome}"
        );
    }

    let (_, capture_outcome) = rows
        .iter()
        .find(|(path, _)| path == "src/debug/sample_capture.rs")
        .expect("the sample_capture.rs row is present");
    assert!(
        capture_outcome.contains(FINDING),
        "the sample_capture.rs row must link the finding it filed ({FINDING}): {capture_outcome}"
    );
}

#[test]
fn the_debug_sampler_mutation_region_cites_all_five_sites() {
    let doc = read(RECORD);
    let region = marker_region(section(&doc, "## Filesystem mutation sites"));
    let rows = table_rows(region);

    for (site, file, definition) in MUTATION_SITES {
        let cited = format!("`{site}`");
        assert!(
            rows.iter().any(|row| {
                row.trim()
                    .trim_start_matches('|')
                    .trim_start()
                    .starts_with(&cited)
            }),
            "the {SECTION} mutation region must carry a row for {cited}"
        );
        // Issue #1942: a symbol citation is only worth something while the
        // symbol still exists.
        assert!(
            read(file).contains(definition),
            "{cited} is cited but `{definition}` is no longer in {file}"
        );
    }
    assert!(
        rows.iter().all(|row| !row.contains("| pending |")),
        "no {SECTION} mutation row may read `pending`"
    );
}

#[test]
fn the_debug_sampler_reverification_region_has_1905_and_1904_rows() {
    let doc = read(RECORD);
    let region = marker_region(section(&doc, "## Re-verified remediations"));
    let rows = table_rows(region);

    for (issue, guard) in REVERIFIED {
        let prefix = format!("| {issue} |");
        let row = rows
            .iter()
            .find(|row| row.trim_start().starts_with(&prefix))
            .unwrap_or_else(|| {
                panic!("the {SECTION} re-verification region must carry a {issue} row")
            });
        assert!(
            row.contains(guard),
            "the {issue} row must cite its guard `{guard}`: {row}"
        );
        assert!(
            repo_root().join(guard).is_file(),
            "the {issue} row cites `{guard}`, which must exist"
        );
    }
}
