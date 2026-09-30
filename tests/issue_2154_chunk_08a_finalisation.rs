//! Finalisation contract for the chunk 8a security-sweep record
//! (Issue #2302, part of #2154).
//!
//! Issue #2280 staged the record's skeleton, and its six sections were then
//! filled one at a time: `shared` by #2281, `graph` by #2282/#2283,
//! `pairwise` by #2294/#2295, `per-neuron-a` by #2284/#2285, `per-neuron-b`
//! by #2298/#2299 and `neuron` by #2300/#2301, at
//! `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md`.
//! Finalisation `git mv`s that file to
//! `docs/audits/security-sweep-chunk-08a-detection-neuron.md` and fills the
//! `"8a"` entry of `docs/audits/lib-sweep-coverage.json` in the same commit.
//! This file pins the finished state so the record cannot quietly slide
//! back to a staged or partial one:
//!
//! * the record is promoted out of `in-progress/` and the staged copy is gone;
//! * no cell anywhere in the record still reads `pending`;
//! * `## Files swept` carries exactly one row per in-scope file, matching the
//!   on-disk `.rs` files of `src/analysis/detection` and `src/analysis/neuron`;
//! * the six `<!-- section: … -->` markers are present, in order;
//! * every production capacity site the capacity regex sweep
//!   (`with_capacity\(|vec!\[[^\]]*;|\.reserve\(`) finds is cited by symbol
//!   in a `capacity`-kind row of `## Capacity and traversal table`;
//! * every traversal-shaped function (`fn (...) {` whose signature mentions
//!   depth/visited/frontier/queue) is cited by a `traversal`-kind row;
//! * the tail sections (`## Outcome`, `## Issues filed`,
//!   `## Related remediations (not sweep coverage)`, `## Verify this record`)
//!   are filled in, not left as placeholders;
//! * the index's `8a` entry equals the record's sweep date, baseline commit
//!   and path.

use std::collections::BTreeSet;
use std::path::PathBuf;

use serde_json::Value;

/// The promoted, top-level record — the state finalisation produces.
const RECORD: &str = "docs/audits/security-sweep-chunk-08a-detection-neuron.md";
/// The staged copy finalisation `git mv`s away — must no longer exist.
const STAGED: &str = "docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md";
/// Machine-readable sweep index — its `"8a"` entry is filled at finalisation.
const INDEX: &str = "docs/audits/lib-sweep-coverage.json";

/// The two directories chunk 8a sweeps, scanned non-recursively.
const SCOPE_DIRS: [&str; 2] = ["src/analysis/detection", "src/analysis/neuron"];

/// Total files across both scope directories — also the row count in
/// `## Files swept`.
const EXPECTED_FILE_COUNT: usize = 52;

/// One `<!-- section: … -->` marker per audit sub-issue, in record order.
const SECTION_NAMES: [&str; 6] = [
    "shared",
    "graph",
    "pairwise",
    "per-neuron-a",
    "per-neuron-b",
    "neuron",
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
/// of the same or a higher level. Matches the heading line exactly (trimmed
/// of trailing whitespace), so e.g. `## Outcome` does not match inside a
/// longer heading.
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

fn rust_files_under(root: &str) -> Vec<String> {
    let base = repo_root();
    let dir = base.join(root);
    let entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", dir.display()));
    let mut found: Vec<String> = entries
        .filter_map(|entry| {
            let path = entry.expect("directory entry must be readable").path();
            (path.is_file() && path.extension().is_some_and(|e| e == "rs")).then(|| {
                path.strip_prefix(&base)
                    .expect("walked path sits under the repo root")
                    .to_string_lossy()
                    .replace('\\', "/")
            })
        })
        .collect();
    found.sort();
    found
}

/// Split a Markdown table row into trimmed cells, on `|` **not** preceded by
/// a backslash, unescaping `\|` to a literal `|` in the cell text. Several
/// cells in this ledger carry an escaped `||` guard expression, so a naive
/// `split('|')` would slice that expression apart.
fn split_row_cells(line: &str) -> Vec<String> {
    let inner = line.trim().trim_matches('|');
    let mut cells = Vec::new();
    let mut current = String::new();
    let mut chars = inner.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek() == Some(&'|') {
            current.push('|');
            chars.next();
        } else if c == '|' {
            cells.push(current.trim().to_string());
            current = String::new();
        } else {
            current.push(c);
        }
    }
    cells.push(current.trim().to_string());
    cells
}

/// Markdown table rows: lines starting with `|`, split into trimmed cells,
/// skipping the header row and the `| --- |` separator row.
fn table_rows(body: &str) -> Vec<Vec<String>> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .filter_map(|line| {
            let cells = split_row_cells(line);
            let first = cells.first()?;
            if first == "Path" || first == "Symbol" || first.starts_with("---") {
                return None;
            }
            Some(cells)
        })
        .collect()
}

/// The three-cell `| path | lines | outcome |` rows of `## Files swept`.
struct FileRow {
    path: String,
    outcome: String,
}

fn file_rows(body: &str) -> Vec<FileRow> {
    table_rows(body)
        .into_iter()
        .filter(|cells| cells.len() == 3)
        .map(|cells| FileRow {
            path: cells[0].trim_matches('`').to_string(),
            outcome: cells[2].clone(),
        })
        .collect()
}

/// The source up to the `#[cfg(test)]` that actually opens the real test
/// module — i.e. whose next non-blank, non-attribute, non-comment line
/// starts with `mod `. A naive cut at the *first* `#[cfg(test)]` is wrong:
/// `src/analysis/neuron/post_processing.rs` has a test-only
/// `#[cfg(test)] fn apply_per_target_cap` declared *before* the production
/// `fn apply_distinct_target_spread`, so cutting there would silently drop
/// its capacity sites. If no such `#[cfg(test)] mod ...` pair exists, return
/// the whole file.
fn production_source(rel: &str) -> String {
    let body = read(rel);
    let lines: Vec<&str> = body.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        if line.trim() != "#[cfg(test)]" {
            continue;
        }
        let next = lines[i + 1..].iter().find(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#') && !t.starts_with("//")
        });
        if next.is_some_and(|l| l.trim_start().starts_with("mod ")) {
            let mut offset = 0usize;
            for earlier in &lines[..i] {
                offset += earlier.len() + 1;
            }
            return body[..offset].to_string();
        }
    }
    body
}

/// `true` when the line allocates a collection whose size is an expression —
/// `with_capacity(…)`, `.reserve(…)`, or the `vec![value; count]` form.
/// Equivalent to the regex `with_capacity\(|vec!\[[^\]]*;|\.reserve\(`.
fn is_capacity_site(line: &str) -> bool {
    if line.contains("with_capacity(") || line.contains(".reserve(") {
        return true;
    }
    line.match_indices("vec![").any(|(at, needle)| {
        let after = &line[at + needle.len()..];
        after
            .find(']')
            .is_some_and(|close| after[..close].contains(';'))
    })
}

fn basename(rel: &str) -> &str {
    rel.rsplit('/').next().unwrap_or(rel)
}

/// Every in-scope file and its count of production capacity sites, for
/// files with at least one site.
fn scope_capacity_site_counts() -> Vec<(String, usize)> {
    SCOPE_DIRS
        .iter()
        .flat_map(|root| rust_files_under(root))
        .filter_map(|rel| {
            let count = production_source(&rel)
                .lines()
                .filter(|line| is_capacity_site(line))
                .count();
            (count > 0).then_some((rel, count))
        })
        .collect()
}

/// A `fn (...) {`-shaped signature hit: the identifiers appear in order on a
/// single line, mirroring the issue's
/// `grep -rnE 'fn .*\(.*\) .*\{' <scope> | grep -E 'depth|visited|frontier|queue'`.
struct TraversalHit {
    file: String,
    func: String,
}

fn traversal_regex_hits() -> Vec<TraversalHit> {
    let keywords = ["depth", "visited", "frontier", "queue"];
    let mut hits = Vec::new();
    for root in SCOPE_DIRS {
        for rel in rust_files_under(root) {
            let body = read(&rel);
            for line in body.lines() {
                let Some(fn_at) = line.find("fn ") else {
                    continue;
                };
                let after_fn = &line[fn_at + "fn ".len()..];
                let Some(paren_at) = after_fn.find('(') else {
                    continue;
                };
                let after_paren = &after_fn[paren_at + 1..];
                let Some(close_at) = after_paren.find(") ") else {
                    continue;
                };
                let after_close = &after_paren[close_at + 2..];
                if !after_close.contains('{') {
                    continue;
                }
                if !keywords.iter().any(|kw| line.contains(kw)) {
                    continue;
                }
                let name_end = after_fn
                    .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .unwrap_or(after_fn.len());
                hits.push(TraversalHit {
                    file: rel.clone(),
                    func: after_fn[..name_end].to_string(),
                });
            }
        }
    }
    hits
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
fn the_record_is_promoted_out_of_in_progress() {
    let doc = read(RECORD);
    assert!(
        !doc.trim().is_empty(),
        "{RECORD} must exist and carry content"
    );
    let staged_path = repo_root().join(STAGED);
    assert!(
        !staged_path.exists(),
        "{STAGED} must no longer exist once finalisation has promoted {RECORD}"
    );
}

#[test]
fn no_word_pending_survives_anywhere_in_the_record() {
    let doc = read(RECORD);
    let offending: Vec<usize> = doc
        .lines()
        .enumerate()
        .filter(|(_, line)| line.to_lowercase().contains("pending"))
        .map(|(i, _)| i + 1)
        .collect();
    assert!(
        offending.is_empty(),
        "{RECORD} must not read `pending` anywhere once every section is finalised — offending \
         line numbers: {offending:?}"
    );
}

#[test]
fn files_swept_has_52_rows_each_with_an_outcome_and_a_reason() {
    let doc = read(RECORD);
    let rows = file_rows(section(&doc, "## Files swept"));
    assert_eq!(
        rows.len(),
        EXPECTED_FILE_COUNT,
        "{RECORD} must carry exactly {EXPECTED_FILE_COUNT} rows under `## Files swept`, got {}",
        rows.len()
    );

    let mut paths: Vec<&str> = rows.iter().map(|r| r.path.as_str()).collect();
    paths.sort_unstable();
    let mut deduped = paths.clone();
    deduped.dedup();
    assert_eq!(
        paths.len(),
        deduped.len(),
        "`## Files swept` must not carry duplicate paths, got {paths:?}"
    );

    let on_disk: BTreeSet<String> = SCOPE_DIRS
        .iter()
        .flat_map(|root| rust_files_under(root))
        .collect();
    let recorded: BTreeSet<String> = paths.iter().map(ToString::to_string).collect();
    assert_eq!(
        recorded, on_disk,
        "`## Files swept`'s path set must equal the on-disk `.rs` files of {SCOPE_DIRS:?}"
    );

    for row in &rows {
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
}

#[test]
fn the_six_section_markers_are_present_in_order() {
    let doc = read(RECORD);
    let found: Vec<&str> = doc
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            line.strip_prefix("<!-- section: ")
                .and_then(|rest| rest.strip_suffix(" -->"))
        })
        .collect();
    assert_eq!(
        found, SECTION_NAMES,
        "{RECORD} must carry the six `<!-- section: … -->` markers in order, got {found:?}"
    );
}

#[test]
fn every_production_capacity_site_is_cited_in_the_capacity_table() {
    // Issue #1799: pin the detector on known production sites before trusting
    // the loop below, so an empty sweep cannot pass vacuously.
    assert!(
        is_capacity_site("let v = vec![0.0; n];"),
        "detector precondition: `let v = vec![0.0; n];` must be a sized allocation"
    );
    assert!(
        !is_capacity_site("batches.push(vec![candidate]);"),
        "detector precondition: a literal `vec![x];` is not a sized allocation"
    );
    assert!(
        is_capacity_site("vec![vec![0.0; n]; n]"),
        "detector precondition: a nested `vec![vec![0.0; n]; n]` is a sized allocation"
    );
    assert!(
        is_capacity_site("let a = vec![x]; let b = vec![0u8; n];"),
        "detector precondition: a line with an unsized `vec![x];` followed by a sized \
         `vec![0u8; n];` must still be found — the sweep matches every `vec![` occurrence on \
         the line, not just the first"
    );

    let site_counts = scope_capacity_site_counts();
    let site_map: std::collections::BTreeMap<&str, usize> =
        site_counts.iter().map(|(f, c)| (f.as_str(), *c)).collect();
    assert!(
        site_map.contains_key("src/analysis/detection/topology_cache.rs"),
        "precondition: src/analysis/detection/topology_cache.rs must carry at least one \
         production capacity site, otherwise this test checks nothing about a known #2078 site"
    );
    assert!(
        site_map.contains_key("src/analysis/neuron/preparation.rs"),
        "precondition: src/analysis/neuron/preparation.rs must carry at least one production \
         capacity site, otherwise this test checks nothing about a known neuron site"
    );
    assert!(
        !site_counts.is_empty(),
        "precondition: at least one in-scope file must carry a production capacity site"
    );

    let doc = read(RECORD);
    let capacity_heading = section(&doc, "## Capacity and traversal table");
    let rows = table_rows(capacity_heading);

    let mut mismatches = Vec::new();
    for (file, expected_count) in &site_counts {
        let base = basename(file);
        let row_count = rows
            .iter()
            .filter(|row| {
                row.first().is_some_and(|first| {
                    first.starts_with(&format!("`{base}::"))
                        || first.starts_with(&format!("`{file}::"))
                }) && row.get(1).is_some_and(|kind| kind == "capacity")
            })
            .count();
        if row_count != *expected_count {
            mismatches.push(format!(
                "{file}: {expected_count} production capacity site(s), {row_count} capacity \
                 row(s)"
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "every production capacity site must be cited by a `capacity`-kind row in `## Capacity \
         and traversal table`; mismatches: {mismatches:?}"
    );

    assert!(
        capacity_heading.contains("**Regex reconciliation.**"),
        "`## Capacity and traversal table` must carry a `Regex reconciliation` note"
    );
}

#[test]
fn every_traversal_regex_hit_is_cited_as_a_traversal_row() {
    let hits = traversal_regex_hits();
    assert!(
        hits.iter()
            .any(|h| h.file == "src/analysis/detection/skip_connection.rs"
                && h.func == "compute_depths_from_inputs"),
        "precondition: the `fn (...) {{` / depth-visited-frontier-queue regex sweep must find \
         `skip_connection.rs::compute_depths_from_inputs`, otherwise this test checks nothing"
    );
    assert!(
        !hits.is_empty(),
        "precondition: at least one traversal-shaped function must be found in scope"
    );

    let doc = read(RECORD);
    let rows = table_rows(section(&doc, "## Capacity and traversal table"));

    let mut missing = Vec::new();
    for hit in &hits {
        let base = basename(&hit.file);
        let cited = rows.iter().any(|row| {
            row.first().is_some_and(|first| {
                first.starts_with(&format!("`{base}::{}", hit.func))
                    || first.starts_with(&format!("`{}::{}", hit.file, hit.func))
            }) && row.get(1).is_some_and(|kind| kind == "traversal")
        });
        if !cited {
            missing.push(format!("{}::{}", hit.file, hit.func));
        }
    }
    assert!(
        missing.is_empty(),
        "every traversal-shaped function the regex sweep finds must be cited by a \
         `traversal`-kind row in `## Capacity and traversal table`; uncited: {missing:?}"
    );
}

#[test]
fn the_tail_sections_are_filled() {
    let doc = read(RECORD);

    let outcome = section(&doc, "## Outcome");
    assert!(
        !outcome.trim().is_empty(),
        "`## Outcome` must not be empty once the record is finalised"
    );

    let issues_filed = section(&doc, "## Issues filed");
    let has_issue_ref = issues_filed
        .as_bytes()
        .windows(2)
        .any(|w| w[0] == b'#' && w[1].is_ascii_digit());
    assert!(
        has_issue_ref || issues_filed.contains("None"),
        "`## Issues filed` must either reference at least one `#NNNN` issue or read `None`"
    );

    let related = section(&doc, "## Related remediations (not sweep coverage)");
    for needle in [
        "#2078",
        "validate_creature_input_bounds",
        "MAX_CREATURE_INPUT_NEURONS",
        "MAX_CREATURE_OUTPUT_NEURONS",
        "#1184",
        "#1906",
    ] {
        assert!(
            related.contains(needle),
            "`## Related remediations (not sweep coverage)` must mention `{needle}`, got: \
             {related}"
        );
    }

    let baseline = record_field(&doc, "Baseline commit");
    let verify = section(&doc, "## Verify this record");
    let expected_diff = format!("git diff {baseline}..HEAD");
    assert!(
        verify.contains(&expected_diff),
        "`## Verify this record` must quote `{expected_diff}`, got: {verify}"
    );
    assert!(
        verify.contains("p[e]nding"),
        "`## Verify this record` must quote the literal `p[e]nding` guard (escaped so this very \
         record does not trip `no_word_pending_survives_anywhere_in_the_record`), got: {verify}"
    );
}

#[test]
fn the_index_entry_equals_the_record_date_baseline_and_path() {
    let doc = read(RECORD);
    let index: Value =
        serde_json::from_str(&read(INDEX)).unwrap_or_else(|e| panic!("{INDEX} must parse: {e}"));
    let entry = index["chunks"]
        .as_array()
        .and_then(|chunks| chunks.iter().find(|c| c["id"] == "8a"))
        .unwrap_or_else(|| panic!("{INDEX} must carry a chunk `8a` entry"));
    assert_eq!(
        entry["last_swept"].as_str(),
        Some(record_field(&doc, "Sweep date")),
        "index `8a` last_swept must equal the record's Sweep date"
    );
    assert_eq!(
        entry["baseline_commit"].as_str(),
        Some(record_field(&doc, "Baseline commit")),
        "index `8a` baseline_commit must equal the record's Baseline commit"
    );
    assert_eq!(
        entry["record"].as_str(),
        Some(RECORD),
        "index `8a` record must equal {RECORD}"
    );
}
