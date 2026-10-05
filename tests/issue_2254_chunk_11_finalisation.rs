//! Finalisation contract for the chunk 11 sweep record (Issue #2254).
//!
//! Issues #2234 (11a), #2251 (11b), #2252 (11c-1) and #2253 (11c-2) each swept
//! one slice of a shared record; #2254 reconciled the tables below against a
//! fresh grep inventory and wrote the `## Outcome` and `## Issues filed`
//! sections. This file pins the finished state, so the ledger cannot quietly
//! slide back to a partial one:
//!
//! * every one of the eight chunk-11 files has a `## Files swept` row, no row
//!   reads `pending`, and the status heading says the sweep is complete;
//! * every file whose production source matches the Issue #2120 filesystem/
//!   process grep is named in `## Filesystem mutation sites`'s grep
//!   reconciliation, and every production mutating line it finds has a row in
//!   the mutation table;
//! * the five 2025-era remediations (#1902–#1906) each keep exactly one
//!   `## Re-verified remediations` row, citing a real symbol and a yes/no
//!   verdict;
//! * `## Issues filed` and `## Outcome` agree on which findings exist, with
//!   the scaffold's placeholder wording gone either way;
//! * the index's `11` entry equals the record's sweep date and baseline.

use std::path::PathBuf;

use serde_json::Value;

const RECORD: &str = "docs/audits/security-sweep-chunk-11-filesystem-lifecycle.md";
const INDEX: &str = "docs/audits/lib-sweep-coverage.json";
const BASELINE: &str = "b85a551ed2521ed327469b20eb88aeda828357d2";

/// The eight files this chunk covers (Issues #2117–#2120).
const FILES: [&str; 8] = [
    "src/discovery_cleanup.rs",
    "src/debug.rs",
    "src/debug/sample_capture.rs",
    "src/debug/sample_dir.rs",
    "src/debug/process_state.rs",
    "src/watchdog.rs",
    "src/tracking_alloc.rs",
    "src/discovery_history.rs",
];

/// The 2025-era remediations this chunk re-verifies on the live path.
const REVERIFIED: [&str; 5] = ["#1902", "#1903", "#1904", "#1905", "#1906"];

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
/// heading`), per `tests/common/ledger.rs` — unlike a substring `find`, this
/// cannot match `## Outcome` inside a longer heading or a prose sentence that
/// happens to contain the same text.
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

/// The source up to the file's test module.
///
/// Cutting at the **first** `#[cfg(test)]` (as `tests/issue_2210_chunk_08b_finalisation.rs`
/// does) is wrong here: `src/debug/sample_dir.rs` carries a `#[cfg(test)]` on
/// a single accessor method well before its test module, and
/// `src/watchdog.rs` carries several `#[cfg(test)]` statics and functions
/// before its test module too — cutting at the first one would silently drop
/// real production lines (e.g. `sample_dir.rs`'s `Drop::drop` `remove_dir_all`
/// call) from the "production" half. Instead, cut at the first line that is
/// exactly `#[cfg(test)]` (trimmed, at column 0) whose next non-blank line
/// starts with `mod ` — i.e. the attribute on the test module itself. If no
/// such line exists, the whole file is production source.
fn production_source(rel: &str) -> String {
    let body = read(rel);
    let lines: Vec<&str> = body.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if *line != "#[cfg(test)]" {
            continue;
        }
        let next = lines[i + 1..].iter().find(|l| !l.trim().is_empty());
        if next.is_some_and(|l| l.trim_start().starts_with("mod ")) {
            return lines[..i].join("\n");
        }
    }
    body
}

/// The `|`-prefixed table rows within a section body.
fn table_rows(body: &str) -> Vec<&str> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .collect()
}

fn record_field<'a>(doc: &'a str, field: &str) -> &'a str {
    doc.split_once(&format!("**{field}:** `"))
        .and_then(|(_, rest)| rest.split_once('`'))
        .map_or_else(
            || panic!("{RECORD} must carry a `**{field}:**` field"),
            |(value, _)| value,
        )
}

fn basename(rel: &str) -> &str {
    rel.rsplit('/').next().unwrap_or(rel)
}

/// `remove_(file|dir_all)|create_dir|rename|symlink|canonicalize|set_permissions|\.exists\(\)|Command::new`
/// (Issue #2120), as plain substrings — the `regex` crate is not a dependency
/// of this crate (checked against `Cargo.toml`'s `[dev-dependencies]`).
const INVENTORY_NEEDLES: [&str; 9] = [
    "remove_file",
    "remove_dir_all",
    "create_dir",
    "rename",
    "symlink",
    "canonicalize",
    "set_permissions",
    ".exists()",
    "Command::new",
];

fn is_inventory_hit(line: &str) -> bool {
    INVENTORY_NEEDLES.iter().any(|needle| line.contains(needle))
}

/// A hit that mutates the filesystem or spawns a process, as opposed to a
/// read-only probe, a doc comment or an unrelated word match. `rename(` (not
/// bare `rename`) so serde's `rename_all = ` attribute never counts.
const MUTATING_NEEDLES: [&str; 8] = [
    "remove_file(",
    "remove_dir_all(",
    "create_dir(",
    "create_dir_all(",
    "rename(",
    "set_permissions(",
    "Command::new(",
    "os::unix::fs::symlink(",
];

fn is_mutating_hit(line: &str) -> bool {
    !line.trim_start().starts_with("//")
        && MUTATING_NEEDLES.iter().any(|needle| line.contains(needle))
}

#[test]
fn production_source_cut_rule_keeps_sample_dir_drop_remove_dir_all() {
    // Guards the cut rule itself: if this regressed to "first #[cfg(test)]",
    // `sample_dir.rs`'s lone `#[cfg(test)]` accessor (well before its test
    // module) would wrongly truncate away `Drop::drop`'s `remove_dir_all`.
    let production = production_source("src/debug/sample_dir.rs");
    assert!(
        production.contains("remove_dir_all"),
        "production_source(\"src/debug/sample_dir.rs\") must still carry \
         `Drop::drop`'s `remove_dir_all` call"
    );
}

#[test]
fn files_swept_lists_every_chunk_11_file_with_a_reasoned_outcome() {
    let doc = read(RECORD);
    let body = section(&doc, "## Files swept");

    let rows: Vec<(String, String)> = table_rows(body)
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
        .collect();

    let mut got: Vec<&str> = rows.iter().map(|(p, _)| p.as_str()).collect();
    got.sort_unstable();
    let mut want: Vec<&str> = FILES.to_vec();
    want.sort_unstable();
    assert_eq!(
        got, want,
        "`## Files swept` must list exactly the eight chunk-11 files, each once"
    );

    assert!(
        !body.contains("| pending"),
        "no `## Files swept` table cell may read `pending`"
    );

    for (path, outcome) in &rows {
        assert!(
            !outcome.contains("pending"),
            "{path} still reads `pending` — the chunk is not finished: {outcome}"
        );
        let Some((verdict, reason)) = outcome.split_once(" — ") else {
            panic!("{path} must read `<verdict> — <reason>`, got: {outcome}");
        };
        assert!(
            !verdict.trim().is_empty() && !reason.trim().is_empty(),
            "{path} must carry both a verdict and a reason: {outcome}"
        );
    }
}

#[test]
fn the_status_heading_reads_complete_and_the_scaffold_is_gone() {
    let doc = read(RECORD);
    assert!(
        doc.contains("### Sweep status — COMPLETE"),
        "{RECORD} must carry a `### Sweep status — COMPLETE` heading"
    );
    assert!(
        !doc.contains("IN PROGRESS"),
        "{RECORD} still carries an `IN PROGRESS` status although the sweep is complete"
    );
    assert!(
        !doc.contains("<!-- section:"),
        "{RECORD} must not carry a leftover `<!-- section: ... -->` marker"
    );
    assert!(
        !doc.contains("This record is a scaffold"),
        "{RECORD} must not carry the scaffold placeholder wording"
    );
}

#[test]
fn grep_reconciliation_names_every_matching_file_and_the_mutation_table_carries_every_hit() {
    let doc = read(RECORD);
    let section_body = section(&doc, "## Filesystem mutation sites");

    assert!(
        section_body.contains("**Grep reconciliation.**"),
        "`## Filesystem mutation sites` must carry a `Grep reconciliation` note"
    );
    assert!(
        section_body.contains("Bare-call blind spot"),
        "`## Filesystem mutation sites` must name the bare-call blind spot \
         (an imported verb called bare that the grep's own wording does not name)"
    );

    let mutation_rows = table_rows(section_body).join("\n");
    let mut any_mutating_hit = false;

    for rel in FILES {
        let production = production_source(rel);
        let lines: Vec<&str> = production.lines().collect();

        let has_inventory_hit = lines.iter().any(|l| is_inventory_hit(l));
        if has_inventory_hit {
            let full_path = rel.to_string();
            let base = basename(rel);
            assert!(
                section_body.contains(&full_path) || section_body.contains(base),
                "`## Filesystem mutation sites` must name {rel} (by path or basename) \
                 since its production source matches the Issue #2120 inventory"
            );
        }

        for line in &lines {
            if is_mutating_hit(line) {
                any_mutating_hit = true;
                let base = basename(rel);
                let needle = format!("`{base}::");
                assert!(
                    mutation_rows.contains(&needle),
                    "a production mutating line in {rel} ({line}) has no \
                     `{needle}` row in the mutation table"
                );
            }
        }
    }

    assert!(
        any_mutating_hit,
        "precondition: at least one production mutating hit must exist across the \
         chunk-11 files, or this test's loop is vacuous"
    );
}

#[test]
fn re_verified_remediations_carries_exactly_one_sound_row_per_issue() {
    let doc = read(RECORD);
    let body = section(&doc, "## Re-verified remediations");

    let data_rows: Vec<&str> = table_rows(body)
        .into_iter()
        .filter(|row| {
            let first = row.trim_matches('|').split('|').next().unwrap_or("").trim();
            first != "Issue" && !row.contains("---")
        })
        .collect();

    assert_eq!(
        data_rows.len(),
        5,
        "`## Re-verified remediations` must carry exactly 5 data rows, found: {data_rows:?}"
    );

    for issue in REVERIFIED {
        let matching: Vec<&&str> = data_rows
            .iter()
            .filter(|row| {
                let cells: Vec<&str> = row.trim().trim_matches('|').split('|').collect();
                cells.first().map(|c| c.trim()) == Some(issue)
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "`## Re-verified remediations` must carry exactly one row for {issue}, found: {matching:?}"
        );
        let row = matching[0];
        let cells: Vec<&str> = row.trim().trim_matches('|').split('|').collect();
        let citing = cells.get(2).map_or("", |c| c.trim());
        assert!(
            citing.contains(".rs::"),
            "{issue}'s citing-site cell must contain `.rs::`, got: {citing}"
        );
        let verdict = cells.last().map_or("", |c| c.trim());
        assert!(
            verdict.starts_with("yes") || verdict.starts_with("no"),
            "{issue}'s live-path verdict must start with `yes` or `no`, got: {verdict}"
        );
    }
}

#[test]
fn issues_filed_and_outcome_agree_on_which_findings_exist() {
    let doc = read(RECORD);
    let filed = section(&doc, "## Issues filed");
    let outcome = section(&doc, "## Outcome");

    let findings: Vec<String> = filed
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("- #") {
                return None;
            }
            let rest = &trimmed[3..];
            if !rest.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                return None;
            }
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            Some(format!("#{digits}"))
        })
        .collect();

    if findings.is_empty() {
        assert!(
            filed.contains("negative-result"),
            "`## Issues filed` carries no `- #N` finding, so it must read `negative-result`"
        );
        assert!(
            outcome.contains("negative-result"),
            "`## Outcome` must also read `negative-result` when no finding was filed"
        );
    } else {
        assert!(
            !filed.contains("negative-result"),
            "`## Issues filed` lists findings, so it must not also read `negative-result`"
        );
        assert!(
            !outcome.contains("negative-result"),
            "`## Outcome` must not read `negative-result` while `## Issues filed` lists findings"
        );
        for finding in &findings {
            assert!(
                outcome.contains(finding),
                "`## Outcome` must mention {finding}, which `## Issues filed` lists"
            );
        }
    }

    assert!(
        !outcome.trim_start().starts_with("Pending"),
        "`## Outcome` must not start with the scaffold's `Pending` wording"
    );
    assert!(
        !filed.contains("Placeholder"),
        "`## Issues filed` must not carry the scaffold's `Placeholder` wording"
    );
}

#[test]
fn the_index_entry_equals_the_record_date_and_baseline() {
    let doc = read(RECORD);
    let raw_index = read(INDEX);
    let index: Value =
        serde_json::from_str(&raw_index).unwrap_or_else(|e| panic!("{INDEX} must parse: {e}"));
    let entry = index["chunks"]
        .as_array()
        .and_then(|chunks| chunks.iter().find(|c| c["id"] == "11"))
        .unwrap_or_else(|| panic!("{INDEX} must carry a chunk `11` entry"));

    assert_eq!(
        entry["last_swept"].as_str(),
        Some(record_field(&doc, "Sweep date")),
        "index `11` last_swept must equal the record's Sweep date"
    );
    assert_eq!(
        entry["baseline_commit"].as_str(),
        Some(BASELINE),
        "index `11` baseline_commit must equal BASELINE"
    );
    assert_eq!(
        record_field(&doc, "Baseline commit"),
        BASELINE,
        "the record's Baseline commit field must equal BASELINE"
    );
    assert_eq!(entry["record"].as_str(), Some(RECORD));

    let on_one_line = raw_index
        .lines()
        .find(|line| line.contains("\"id\": \"11\""))
        .unwrap_or_else(|| panic!("{INDEX} must carry the chunk `11` entry on one line"));
    assert!(
        on_one_line.contains("\"last_swept\"") && on_one_line.contains("\"baseline_commit\""),
        "the chunk `11` entry must sit on one line of {INDEX}, got: {on_one_line}"
    );
}
