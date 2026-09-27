//! Contract tests for the chunk 11 sweep record scaffold (Issue #2233).
//!
//! Chunk 11 is swept by several audit sub-issues writing into **one** shared
//! record. These tests pin the shape they write into: every in-scope file has
//! exactly one row under its owning `###` section, and both finding tables
//! carry one `<!-- section: … -->` marker per section, in order, so concurrent
//! PRs append to disjoint regions. They deliberately do not assert `pending` —
//! the audit sub-issues flip those outcomes.
//! `tests/issue_2088_sweep_ledger_contract.rs` covers the ledger-wide rules.

use std::path::PathBuf;

/// The chunk 11 prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-11-filesystem-lifecycle.md";
/// Machine-readable sweep index — must name the record above.
const INDEX: &str = "docs/audits/lib-sweep-coverage.json";

/// The baseline commit chunk 11 is cut against (Issue #2095).
const BASELINE: &str = "b85a551ed2521ed327469b20eb88aeda828357d2";

/// One `###` section per audit sub-issue, in record order, with the files it owns.
const SECTIONS: [(&str, &[&str]); 3] = [
    ("discovery_cleanup", &["src/discovery_cleanup.rs"]),
    (
        "debug + sampler",
        &[
            "src/debug.rs",
            "src/debug/sample_capture.rs",
            "src/debug/sample_dir.rs",
            "src/debug/process_state.rs",
        ],
    ),
    (
        "watchdog + tracking_alloc + discovery_history",
        &[
            "src/watchdog.rs",
            "src/tracking_alloc.rs",
            "src/discovery_history.rs",
        ],
    ),
];

/// The two finding tables, each split by the section markers.
const FINDING_TABLES: [&str; 2] = [
    "## Filesystem mutation sites",
    "## Re-verified remediations",
];

fn read(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
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
        .map(|(at, _)| at)
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

/// The repo-relative paths in the first cell of each table row under `body`.
fn row_paths(body: &str) -> Vec<String> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .filter_map(|line| {
            let first = line.trim().trim_matches('|').split('|').next()?;
            let path = first.trim().trim_matches('`');
            path.starts_with("src/").then(|| path.to_string())
        })
        .collect()
}

#[test]
fn the_record_exists_and_pins_the_full_baseline_sha() {
    let doc = read(RECORD);
    assert!(
        doc.contains(BASELINE),
        "{RECORD} must carry the full baseline SHA {BASELINE} — a record with no commit \
         SHA cannot be diffed against the current tree"
    );
}

#[test]
fn each_in_scope_file_has_exactly_one_row_under_its_owning_section() {
    let doc = read(RECORD);
    let files_swept = section(&doc, "## Files swept");
    let all_rows = row_paths(files_swept);

    for (name, files) in SECTIONS {
        let heading = format!("### {name}");
        let owned = row_paths(section(files_swept, &heading));
        for file in files {
            let under_heading = owned.iter().filter(|p| p == file).count();
            assert_eq!(
                under_heading, 1,
                "{file} must have exactly one row under `{heading}` — a missing row is never \
                 swept, and a duplicate invites two conflicting outcomes"
            );
            let anywhere = all_rows.iter().filter(|p| p == file).count();
            assert_eq!(
                anywhere, 1,
                "{file} must appear exactly once in `## Files swept`, not also under another section"
            );
        }
    }
}

#[test]
fn both_finding_tables_carry_the_section_markers_in_order() {
    let doc = read(RECORD);

    for heading in FINDING_TABLES {
        let body = section(&doc, heading);
        let mut cursor = 0usize;
        for (name, _) in SECTIONS {
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
fn the_chunk_11_index_entry_names_the_record() {
    let index = read(INDEX);
    let entry = index
        .lines()
        .find(|line| line.contains(r#""id": "11""#))
        .unwrap_or_else(|| panic!("{INDEX} must carry a chunk 11 entry"));
    assert!(
        entry.contains(&format!(r#""record": "{RECORD}""#)),
        "the chunk 11 entry in {INDEX} must name {RECORD}: {entry}"
    );
    assert!(
        entry.contains(&format!(r#""baseline_commit": "{BASELINE}""#)),
        "the chunk 11 entry in {INDEX} must pin baseline {BASELINE}: {entry}"
    );
}
