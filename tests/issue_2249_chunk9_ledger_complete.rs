//! Contract test pinning inventory completeness of the chunk 9 (GPU dispatch +
//! WGSL) sweep ledger, `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`
//! (Issue #2249).
//!
//! `tests/issue_2288_chunk_09_ledger_scaffold.rs` pins the record's *shape*
//! and the per-slice sweep tests (`tests/issue_2113_chunk_09c_device_sweep.rs`
//! and its siblings) pin what each slice writes into its own region. Neither
//! checks that the `## Files swept` inventory actually covers every file in
//! chunk 9's scope. This file is the inventory-completeness gate: it fails
//! loudly when
//!
//! * a new `.rs` file lands under `src/analysis/gpu/` (recursively) or a new
//!   `.wgsl` file lands directly under `src/shaders/` without a matching
//!   `## Files swept` row *and* a matching `## Reconciliation` checklist
//!   line;
//! * a `## Files swept` row reverts to `pending`;
//! * a cited file is renamed or deleted (its row would no longer resolve to
//!   a real, in-scope file).
//!
//! Paths are always compared in their full repo-relative form, never by
//! basename: `src/analysis/gpu/mod.rs` and `src/analysis/gpu/queue/mod.rs`
//! would otherwise collide.
//!
//! This file also pins, for Issue #2250:
//!
//! * the `## Outcome` severity counts equal the `## Ledger` rows per
//!   severity;
//! * `## Ledger` rows stay sorted by severity within each
//!   `<!-- section: … -->` region;
//! * the `## Outcome` names every ledger finding-id and every issue the
//!   ledger links;
//! * the `## Outcome` Mermaid block is a `flowchart` with no `;` (AGENTS.md's
//!   Mermaid rule, deliberately stricter here: every `;` is banned rather
//!   than parsing quoted labels; other Mermaid syntax is only checked by the
//!   worker's external validator);
//! * the six named candidates each keep a row in `## Ledger` or
//!   `## Refuted / not findings`.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The chunk 9 prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-9-gpu-wgsl.md";

/// The five `## Files swept` / `## Reconciliation` sections, in record order.
const GROUPS: [&str; 5] = [
    "shaders",
    "evaluation",
    "device",
    "queue-core",
    "queue-lifecycle",
];

/// Severities in rank order, most severe first — used both to check the
/// `## Ledger` table's within-region sort order and to cross-check the
/// `## Outcome` severity-counts table (Issue #2250).
const SEVERITIES: [&str; 4] = ["critical", "high", "medium", "low"];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// The text of a Markdown section, from its heading line to the next heading
/// of the same or a higher level. `## Files swept` carries `###` subsections,
/// which must stay inside the body this returns, so a caller can re-slice it
/// by subsection heading.
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

/// Every `*.rs` file under `dir`, recursively, appended to `out` as
/// `<rel_prefix>/<name>` — built from the directory-entry names, not from
/// `Path` components, so the result always uses `/` regardless of platform.
fn walk_rs(dir: &Path, rel_prefix: &str, out: &mut Vec<String>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{} must be a readable directory: {e}", dir.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()));
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            walk_rs(&path, &format!("{rel_prefix}/{name}"), out);
        } else if name.ends_with(".rs") {
            out.push(format!("{rel_prefix}/{name}"));
        }
    }
}

/// Every file in chunk 9's scope: `src/analysis/gpu/**/*.rs` (recursive) plus
/// `src/shaders/*.wgsl` (non-recursive), as sorted full repo-relative paths.
///
/// The issue named `src/shaders/matching.wgsl` as a scope example, but #2309
/// deleted it, so a shader that still exists (`src/shaders/helpful.wgsl`) is
/// pinned below instead.
fn enumerate_in_scope() -> Vec<String> {
    let root = repo_root();
    let mut paths = Vec::new();
    walk_rs(
        &root.join("src/analysis/gpu"),
        "src/analysis/gpu",
        &mut paths,
    );
    for entry in fs::read_dir(root.join("src/shaders")).expect("src/shaders must exist") {
        let entry = entry.expect("reading src/shaders");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "wgsl") {
            let name = entry.file_name().to_string_lossy().into_owned();
            paths.push(format!("src/shaders/{name}"));
        }
    }
    paths.sort();
    paths
}

/// `(group, path, outcome)` for every inventory row under `## Files swept`,
/// read subsection by subsection so a row's owning group is known.
fn files_swept_rows(doc: &str) -> Vec<(String, String, String)> {
    let files_swept = section(doc, "## Files swept");
    let mut rows = Vec::new();
    for group in GROUPS {
        let body = section(files_swept, &format!("### {group}"));
        for line in body.lines() {
            let line = line.trim();
            if !line.starts_with('|') {
                continue;
            }
            let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
            let Some(path) = cells.first().map(|c| c.trim_matches('`')) else {
                continue;
            };
            if !path.starts_with("src/") {
                continue;
            }
            let outcome = cells.last().copied().unwrap_or("").to_string();
            rows.push((group.to_string(), path.to_string(), outcome));
        }
    }
    rows
}

/// `(path, section, verdict)` for every `## Reconciliation` checklist line,
/// parsed as ``- [x] `<path>` — <section> — <verdict>``, where ` — ` is
/// space, em dash, space. Splitting with `splitn(3, " — ")` keeps any further
/// em dashes inside the verdict (e.g. `finding filed — #2308`) intact. Panics
/// if a line is unticked (`- [ ]`) or malformed.
fn checklist_lines(doc: &str) -> Vec<(String, String, String)> {
    let reconciliation = section(doc, "## Reconciliation");
    reconciliation
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("- ["))
        .map(|line| {
            let rest = line
                .strip_prefix("- [x] ")
                .unwrap_or_else(|| panic!("Reconciliation checklist line must be ticked: {line}"));
            let parts: Vec<&str> = rest.splitn(3, " — ").collect();
            assert_eq!(
                parts.len(),
                3,
                "Reconciliation checklist line must read `path` — section — verdict: {line}"
            );
            let path = parts[0].trim().trim_matches('`').to_string();
            (
                path,
                parts[1].trim().to_string(),
                parts[2].trim().to_string(),
            )
        })
        .collect()
}

/// `(region, cells)` for every `| SEC-… |` row under `## Ledger`, in document
/// order. `region` comes from the nearest preceding
/// `<!-- section: NAME -->` marker; a `| SEC-` row appearing before any
/// marker is a malformed ledger and panics rather than silently defaulting to
/// an empty region (Issue #2250).
fn ledger_rows(doc: &str) -> Vec<(String, Vec<String>)> {
    let ledger = section(doc, "## Ledger");
    let mut region: Option<String> = None;
    let mut rows = Vec::new();
    for line in ledger.lines() {
        let trimmed = line.trim();
        if let Some(name) = trimmed
            .strip_prefix("<!-- section: ")
            .and_then(|rest| rest.strip_suffix(" -->"))
        {
            region = Some(name.to_string());
            continue;
        }
        if trimmed.starts_with("| SEC-") {
            let current_region = region.clone().unwrap_or_else(|| {
                panic!("`## Ledger` row appears before any `<!-- section: … -->` marker: {trimmed}")
            });
            let cells: Vec<String> = trimmed
                .trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect();
            rows.push((current_region, cells));
        }
    }
    rows
}

/// Every `#<digits>` substring in `text`, returned as `"#1234"` strings, in
/// order (Issue #2250). No regex crate needed — a plain character scan.
fn issue_refs(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut refs = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '#' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 {
                let digits: String = chars[i + 1..j].iter().collect();
                refs.push(format!("#{digits}"));
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    refs
}

/// The text of the ```` ```mermaid ```` fenced block inside `## Outcome`
/// (Issue #2250). Panics with a clear message if `## Outcome` carries no such
/// block, or an unterminated one.
fn outcome_mermaid(doc: &str) -> String {
    let outcome = section(doc, "## Outcome");
    let mut lines = outcome.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "```mermaid" {
            let mut body = Vec::new();
            for inner in lines.by_ref() {
                if inner.trim() == "```" {
                    return body.join("\n");
                }
                body.push(inner);
            }
            panic!("`## Outcome` opens a ```mermaid block with no closing ``` fence");
        }
    }
    panic!("`## Outcome` must carry a ```mermaid flowchart block (Issue #2250)")
}

#[test]
fn the_enumerator_is_pinned_before_trusting_it() {
    let files = enumerate_in_scope();
    assert!(
        files.len() >= 34,
        "expected at least 34 files in chunk 9's scope, found {}: {files:?}",
        files.len()
    );
    assert!(
        files.iter().any(|p| p == "src/analysis/gpu/queue/mod.rs"),
        "the enumerator must recurse into src/analysis/gpu/queue, found: {files:?}"
    );
    assert!(
        files.iter().any(|p| p == "src/shaders/helpful.wgsl"),
        "the enumerator must find src/shaders/*.wgsl, found: {files:?}"
    );
}

#[test]
fn every_on_disk_file_has_exactly_one_files_swept_row_with_a_settled_outcome() {
    let doc = read(RECORD);
    let rows = files_swept_rows(&doc);
    for path in enumerate_in_scope() {
        let matches: Vec<&(String, String, String)> =
            rows.iter().filter(|(_, p, _)| *p == path).collect();
        assert_eq!(
            matches.len(),
            1,
            "{path} must carry exactly one `## Files swept` row, found {}",
            matches.len()
        );
        let outcome = &matches[0].2;
        assert!(
            !outcome.is_empty(),
            "{path}: `## Files swept` outcome must not be empty"
        );
        assert!(
            !outcome.starts_with("pending"),
            "{path}: `## Files swept` outcome must not still read pending: {outcome}"
        );
        assert!(
            !outcome.contains("pending —"),
            "{path}: `## Files swept` outcome must not have reverted to pending: {outcome}"
        );
    }
}

#[test]
fn every_files_swept_row_cites_a_real_in_scope_file() {
    let doc = read(RECORD);
    let rows = files_swept_rows(&doc);
    let enumerated: HashSet<String> = enumerate_in_scope().into_iter().collect();
    for (group, path, _) in &rows {
        assert!(
            repo_root().join(path).is_file(),
            "`### {group}` cites {path}, which does not exist on disk"
        );
        assert!(
            enumerated.contains(path),
            "`### {group}` cites {path}, which is out of chunk 9's scope or has moved"
        );
    }
}

#[test]
fn the_reconciliation_checklist_is_fully_ticked_and_matches_the_on_disk_inventory() {
    let doc = read(RECORD);
    let checklist = checklist_lines(&doc);
    let rows = files_swept_rows(&doc);
    let enumerated: HashSet<String> = enumerate_in_scope().into_iter().collect();

    let mut seen = HashSet::new();
    for (path, section_name, verdict) in &checklist {
        assert!(
            GROUPS.contains(&section_name.as_str()),
            "{path}: Reconciliation section `{section_name}` is not one of {GROUPS:?}"
        );
        assert!(
            !verdict.is_empty(),
            "{path}: Reconciliation verdict must not be empty"
        );
        assert!(
            !verdict.starts_with("pending"),
            "{path}: Reconciliation verdict must not still read pending: {verdict}"
        );
        assert!(
            seen.insert(path.clone()),
            "{path} appears more than once in the Reconciliation checklist"
        );
        let row_group = rows
            .iter()
            .find(|(_, p, _)| p == path)
            .map(|(g, _, _)| g.as_str());
        assert_eq!(
            row_group,
            Some(section_name.as_str()),
            "{path}: Reconciliation section `{section_name}` disagrees with its `## Files swept` group"
        );
    }

    let checklist_paths: HashSet<&str> = checklist.iter().map(|(p, _, _)| p.as_str()).collect();
    let enumerated_refs: HashSet<&str> = enumerated.iter().map(String::as_str).collect();
    assert_eq!(
        checklist_paths, enumerated_refs,
        "the Reconciliation checklist must list exactly the on-disk chunk 9 files, one line each"
    );
}

#[test]
fn the_record_section_references_the_related_sweeps() {
    let doc = read(RECORD);
    let record = section(&doc, "## Record");
    for issue in ["#2088", "#2096"] {
        assert!(record.contains(issue), "`## Record` must reference {issue}");
    }
}

#[test]
fn ledger_rows_are_sorted_by_severity_within_each_region() {
    let doc = read(RECORD);
    let rows = ledger_rows(&doc);
    assert!(!rows.is_empty(), "`## Ledger` must carry at least one row");

    let mut last_rank: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut markers: Vec<String> = Vec::new();
    for (region, cells) in &rows {
        if !markers.contains(region) {
            markers.push(region.clone());
        }
        assert_eq!(
            cells.len(),
            5,
            "`## Ledger` row in region `{region}` must have exactly 5 cells, found {}: {cells:?}",
            cells.len()
        );
        let id = &cells[0];
        let severity = cells[3].as_str();
        let rank = SEVERITIES
            .iter()
            .position(|s| *s == severity)
            .unwrap_or_else(|| {
                panic!("{id}: `## Ledger` severity `{severity}` must be one of {SEVERITIES:?}")
            });
        let entry = last_rank.entry(region.clone()).or_insert(rank);
        assert!(
            rank >= *entry,
            "{id}: `## Ledger` rows in region `{region}` must stay sorted by severity \
             (critical → high → medium → low), found rank {rank} after rank {entry}"
        );
        *entry = rank;
    }

    let expected: Vec<String> = GROUPS.iter().map(ToString::to_string).collect();
    assert_eq!(
        markers, expected,
        "`## Ledger` must carry a `<!-- section: … -->` marker for each of {GROUPS:?} in order, found {markers:?}"
    );
}

#[test]
fn the_outcome_severity_counts_equal_the_ledger_rows_per_severity() {
    let doc = read(RECORD);
    let rows = ledger_rows(&doc);
    let outcome = section(&doc, "## Outcome");

    let table_lines: Vec<Vec<String>> = outcome
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('|'))
        .map(|line| {
            line.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect()
        })
        .collect();

    let mut total = 0usize;
    for severity in SEVERITIES {
        let matches: Vec<usize> = table_lines
            .iter()
            .filter(|cells| cells.first().map(String::as_str) == Some(severity))
            .filter_map(|cells| cells.get(1).and_then(|c| c.parse::<usize>().ok()))
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "`## Outcome` must carry exactly one severity-counts row for `{severity}` \
             (Issue #2250), found {}: {matches:?}",
            matches.len()
        );
        let table_count = matches[0];
        let ledger_count = rows
            .iter()
            .filter(|(_, cells)| cells[3] == severity)
            .count();
        assert_eq!(
            table_count, ledger_count,
            "`## Outcome` severity-counts row for `{severity}` reads {table_count}, \
             but `## Ledger` has {ledger_count} rows of that severity"
        );
        total += table_count;
    }

    assert_eq!(
        total,
        rows.len(),
        "the sum of `## Outcome` severity counts ({total}) must equal the total `## Ledger` row count ({})",
        rows.len()
    );
}

#[test]
fn the_outcome_names_every_ledger_finding_and_issue() {
    let doc = read(RECORD);
    let outcome = section(&doc, "## Outcome");
    let outcome_issues: HashSet<String> = issue_refs(outcome).into_iter().collect();
    for (_, cells) in ledger_rows(&doc) {
        let id = &cells[0];
        assert!(
            outcome.contains(id.as_str()),
            "`## Outcome` must name the ledger finding `{id}` (Issue #2250)"
        );
        for issue in issue_refs(&cells[4]) {
            assert!(
                outcome_issues.contains(&issue),
                "`## Outcome` must name issue `{issue}`, cited by ledger row `{id}` (Issue #2250)"
            );
        }
    }
}

#[test]
fn the_outcome_mermaid_block_is_a_flowchart_with_no_semicolon() {
    let doc = read(RECORD);
    let block = outcome_mermaid(&doc);
    assert!(
        block.contains("flowchart"),
        "the `## Outcome` mermaid block must be a `flowchart` (Issue #2250), found: {block}"
    );
    assert!(
        !block.contains(';'),
        "AGENTS.md bans a bare `;` in Mermaid note/message text — this gate is stricter \
         still and bans every `;` in the `## Outcome` mermaid block, rather than parsing \
         quoted labels like the worker's external validator does: {block}"
    );
    let block_issues: HashSet<String> = issue_refs(&block).into_iter().collect();
    for (_, cells) in ledger_rows(&doc) {
        for issue in issue_refs(&cells[4]) {
            assert!(
                block_issues.contains(&issue),
                "every ledger finding must be attached to a node in the `## Outcome` mermaid \
                 block; `{issue}` (from status cell `{}`) is missing",
                cells[4]
            );
        }
    }
}

#[test]
fn each_named_candidate_keeps_a_ledger_or_refuted_row() {
    let doc = read(RECORD);
    const CANDIDATES: [(&str, &str); 6] = [
        (
            "dispatch-limit overflow",
            "max_compute_workgroups_per_dimension",
        ),
        ("`bias_evaluation` `num_steps`", "num_steps"),
        ("uninitialised pool buffers", "pool"),
        ("SEC-fe0b268a3799", "SEC-fe0b268a3799"),
        ("CPU-fallback visibility", "fallback"),
        (
            "`is_device_lost_error` mis-classification",
            "is_device_lost_error",
        ),
    ];

    fn data_rows(body: &str) -> Vec<Vec<String>> {
        body.lines()
            .map(str::trim)
            .filter(|line| line.starts_with('|'))
            .filter(|line| !line.starts_with("| ---"))
            .map(|line| {
                line.trim_matches('|')
                    .split('|')
                    .map(|c| c.trim().to_string())
                    .collect::<Vec<String>>()
            })
            .filter(|cells| {
                cells
                    .first()
                    .is_some_and(|c| c != "finding-id" && c != "Candidate")
            })
            .collect()
    }

    let ledger_data = data_rows(section(&doc, "## Ledger"));
    let refuted_data = data_rows(section(&doc, "## Refuted / not findings"));

    for (description, keyword) in CANDIDATES {
        let ledger_hit = ledger_data
            .iter()
            .any(|cells| cells.iter().any(|c| c.contains(keyword)));
        let refuted_hit = refuted_data.iter().any(|cells| {
            cells.iter().any(|c| c.contains(keyword))
                && cells.len() >= 3
                && cells.get(1).is_some_and(|cite| {
                    cite.find(':').is_some_and(|at| {
                        cite[at + 1..]
                            .chars()
                            .next()
                            .is_some_and(|c| c.is_ascii_digit())
                    })
                })
        });
        assert!(
            ledger_hit || refuted_hit,
            "{description} (`{keyword}`) must keep a row in `## Ledger` or a \
             `## Refuted / not findings` row citing a file:line"
        );
    }
}
