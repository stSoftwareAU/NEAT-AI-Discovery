//! Contract tests for the chunk 9 (GPU dispatch + WGSL) sweep record scaffold
//! (Issue #2288).
//!
//! Chunk 9 is swept by several slices writing into **one** shared record. These
//! tests pin the shape they write into: a complete `## Record` header that
//! agrees with the index entry, an inventory row per file grouped under one of
//! the five section headings, a body region per section, and both finding
//! tables split by the same five markers in order, so concurrent PRs append to
//! disjoint regions. They deliberately do not assert `pending` — the slices
//! flip those outcomes — nor inventory completeness, which #2249 pins.
//! `tests/issue_2088_sweep_ledger_contract.rs` covers the ledger-wide rules.

use std::path::PathBuf;

use serde_json::Value;

/// The chunk 9 prose record. Not zero-padded: the chunk-9 slices already cite
/// this path, and the ledger contract's `normalise_id` accepts both spellings.
const RECORD: &str = "docs/audits/security-sweep-chunk-9-gpu-wgsl.md";
/// Machine-readable sweep index — must name the record above.
const INDEX: &str = "docs/audits/lib-sweep-coverage.json";

/// The five section regions, in record order.
const SECTIONS: [&str; 5] = [
    "shaders",
    "evaluation",
    "device",
    "queue-core",
    "queue-lifecycle",
];

/// The two finding tables, each split by the section markers.
const FINDING_TABLES: [&str; 2] = ["## Ledger", "## Refuted / not findings"];

/// Heading of the body region holding each section's audit prose.
const AUDIT_SECTIONS: &str = "## Audit sections";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

fn chunk_9_entry() -> Value {
    let index: Value = serde_json::from_str(&read(INDEX))
        .unwrap_or_else(|e| panic!("{INDEX} must be valid JSON: {e}"));
    index["chunks"]
        .as_array()
        .unwrap_or_else(|| panic!("{INDEX} must carry a `chunks` array"))
        .iter()
        .find(|entry| entry["id"].as_str() == Some("9"))
        .cloned()
        .unwrap_or_else(|| panic!("{INDEX} must carry a chunk 9 entry"))
}

fn entry_str(entry: &Value, key: &str) -> String {
    entry[key]
        .as_str()
        .unwrap_or_else(|| panic!("chunk 9 entry must carry a string `{key}`: {entry}"))
        .to_string()
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

/// The value after `- **<label>:**` in the `## Record` list, backticks trimmed.
fn record_field(record: &str, label: &str) -> String {
    let prefix = format!("- **{label}:**");
    let line = record
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("`## Record` must carry a `{prefix}` field"));
    let value = line[prefix.len()..].trim();
    let value = value.strip_prefix('`').map_or(value, |rest| {
        rest.split('`').next().unwrap_or_default()
    });
    assert!(!value.is_empty(), "`{prefix}` must not be empty");
    value.to_string()
}

/// `(path, lines, outcome)` for each inventory row under `body`.
fn inventory_rows(body: &str) -> Vec<(String, String, String)> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .filter_map(|line| {
            let cells: Vec<&str> = line.trim().trim_matches('|').split('|').collect();
            let path = cells.first()?.trim().trim_matches('`');
            path.starts_with("src/").then(|| {
                (
                    path.to_string(),
                    cells.get(1).map_or("", |c| c.trim()).to_string(),
                    cells.get(2).map_or("", |c| c.trim()).to_string(),
                )
            })
        })
        .collect()
}

fn in_chunk_9_scope(path: &str) -> bool {
    (path.starts_with("src/analysis/gpu/") && path.ends_with(".rs"))
        || (path.starts_with("src/shaders/")
            && path.ends_with(".wgsl")
            && !path["src/shaders/".len()..].contains('/'))
}

/// Asserts every section marker appears in `body`, in record order.
fn assert_markers_in_order(body: &str, heading: &str) {
    let mut cursor = 0usize;
    for name in SECTIONS {
        let marker = format!("<!-- section: {name} -->");
        let offset = body[cursor..].find(&marker).unwrap_or_else(|| {
            panic!(
                "`{heading}` must carry `{marker}` after the markers before it, so each \
                 slice's content lands in a disjoint region"
            )
        });
        cursor = cursor + offset + marker.len();
    }
}

#[test]
fn the_record_header_carries_every_field_and_agrees_with_the_index() {
    let doc = read(RECORD);
    let record = section(&doc, "## Record");
    let entry = chunk_9_entry();

    assert_eq!(record_field(record, "Chunk id"), "9");
    assert_eq!(
        record_field(record, "Sweep date"),
        entry_str(&entry, "last_swept"),
        "the record's sweep date must match the index's `last_swept`"
    );

    let baseline = record_field(record, "Baseline commit");
    assert!(
        baseline.len() == 40 && baseline.chars().all(|c| c.is_ascii_hexdigit()),
        "the baseline commit must be a full 40-character SHA, found `{baseline}`"
    );
    assert_eq!(
        baseline,
        entry_str(&entry, "baseline_commit"),
        "the record and the index must pin the same baseline commit"
    );

    assert_eq!(
        record_field(record, "Exposure"),
        entry_str(&entry, "exposure"),
        "the record's exposure must match the index entry"
    );
    record_field(record, "Swept by");
    assert_eq!(record_field(record, "Tracker issue"), "#2094");
    assert_eq!(entry_str(&entry, "record"), RECORD);

    let verify = section(&doc, "## Verify this record");
    assert!(
        verify.contains(&format!(
            "git diff {baseline}..HEAD -- src/analysis/gpu src/shaders"
        )),
        "`## Verify this record` must give the falsifying diff against the baseline"
    );
}

#[test]
fn every_inventory_row_sits_under_a_section_heading_and_names_a_real_file() {
    let doc = read(RECORD);
    let files_swept = section(&doc, "## Files swept");
    let all_rows = inventory_rows(files_swept);
    assert!(
        !all_rows.is_empty(),
        "`## Files swept` must carry the chunk 9 inventory"
    );

    let mut grouped = 0usize;
    for name in SECTIONS {
        let rows = inventory_rows(section(files_swept, &format!("### {name}")));
        assert!(
            !rows.is_empty(),
            "`### {name}` in `## Files swept` must own at least one file"
        );
        grouped += rows.len();
    }
    assert_eq!(
        grouped,
        all_rows.len(),
        "every inventory row must sit under one of the five section headings, so each \
         file has exactly one owning slice"
    );

    let mut seen: Vec<&str> = Vec::new();
    for (path, lines, outcome) in &all_rows {
        assert!(
            in_chunk_9_scope(path),
            "{path} is outside chunk 9 (src/analysis/gpu/**/*.rs, src/shaders/*.wgsl)"
        );
        assert!(
            repo_root().join(path).is_file(),
            "{path} is cited in `## Files swept` but does not exist"
        );
        assert!(
            lines.parse::<usize>().is_ok_and(|n| n > 0),
            "{path} must carry a positive line count at the baseline, found `{lines}`"
        );
        assert!(!outcome.is_empty(), "{path} must carry an outcome");
        assert!(
            !seen.contains(&path.as_str()),
            "{path} must appear exactly once in `## Files swept`"
        );
        seen.push(path);
    }
}

#[test]
fn the_body_carries_one_region_per_section_in_order() {
    let doc = read(RECORD);
    assert_markers_in_order(section(&doc, AUDIT_SECTIONS), AUDIT_SECTIONS);
}

#[test]
fn both_finding_tables_carry_the_section_markers_in_order() {
    let doc = read(RECORD);
    for heading in FINDING_TABLES {
        assert_markers_in_order(section(&doc, heading), heading);
    }
    assert!(
        section(&doc, "## Ledger")
            .contains("| finding-id | file:line | CWE | severity | status |"),
        "`## Ledger` must carry the `finding-id | file:line | CWE | severity | status` header"
    );
}
