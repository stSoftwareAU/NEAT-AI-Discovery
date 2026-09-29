//! Contract tests for the `shared` section of the staged chunk 8a sweep
//! record (Issue #2281).
//!
//! The record is staged under `docs/audits/in-progress/` while the chunk 8a
//! audit sub-issues fill their sections; `tests/issue_2280_chunk_08a_ledger_scaffold.rs`
//! gates the record's shape (exactly six `<!-- section: NAME -->` markers and
//! 52 `src/` rows under `## Files swept`). This file gates only the `shared`
//! section's content: the five shared-infrastructure files it owns, and the
//! shared rows of the capacity and traversal table. Finalisation (#2154)
//! `git mv`s the record to the top level in the commit that sets the chunk
//! `"8a"` index entry, so `RECORD` below changes to
//! `docs/audits/security-sweep-chunk-08a-detection-neuron.md` at that point.

use std::path::PathBuf;

/// The staged chunk 8a prose record. Finalisation (#2154) moves this to
/// `docs/audits/security-sweep-chunk-08a-detection-neuron.md`.
const RECORD: &str = "docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md";

/// The five files the `shared` section owns, in record order.
const SHARED_FILES: [&str; 5] = [
    "src/analysis/detection/mod.rs",
    "src/analysis/detection/helpers.rs",
    "src/analysis/detection/stats.rs",
    "src/analysis/detection/topology_cache.rs",
    "src/analysis/detection/activation_properties.rs",
];

/// The number of capacity-table rows the `shared` sweep owns.
const SHARED_CAPACITY_ROWS: usize = 8;

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

/// The text of the `shared` section: from its `<!-- section: shared -->`
/// marker to the next `### ` or `## ` heading.
fn shared_region(doc: &str) -> &str {
    let marker = "<!-- section: shared -->";
    let start = doc
        .find(marker)
        .unwrap_or_else(|| panic!("{RECORD} must carry the `{marker}` marker"))
        + marker.len();
    let rest = &doc[start..];
    let end_h3 = rest.find("\n### ");
    let end_h2 = rest.find("\n## ");
    let end = match (end_h3, end_h2) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => rest.len(),
    };
    &rest[..end]
}

/// Markdown table rows: lines starting with `|`, split into trimmed cells,
/// skipping the header row and the `| --- |` separator row.
fn table_rows(body: &str) -> Vec<Vec<String>> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .filter_map(|line| {
            let cells: Vec<String> = line
                .trim()
                .trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect();
            let first = cells.first()?;
            if first == "Path" || first == "Symbol" || first.starts_with("---") {
                return None;
            }
            Some(cells)
        })
        .collect()
}

/// The `shared` rows of `## Capacity and traversal table`: rows whose first
/// cell starts with `` `<basename>:: `` for one of the five shared files.
fn shared_capacity_rows(doc: &str) -> Vec<Vec<String>> {
    let basenames = [
        "mod.rs",
        "helpers.rs",
        "stats.rs",
        "topology_cache.rs",
        "activation_properties.rs",
    ];
    table_rows(section(doc, "## Capacity and traversal table"))
        .into_iter()
        .filter(|row| {
            row.first().is_some_and(|first| {
                basenames
                    .iter()
                    .any(|base| first.starts_with(&format!("`{base}::")))
            })
        })
        .collect()
}

/// Capacity sites (`with_capacity(` or `vec![…; …]`) in the production half of
/// a source file (everything before the first `#[cfg(test)]`), paired with
/// the enclosing function name and the binding the site initialises.
fn production_capacity_sites(rel: &str) -> Vec<(String, String)> {
    let body = read(rel);
    let production = match body.find("#[cfg(test)]") {
        Some(at) => &body[..at],
        None => body.as_str(),
    };
    let lines: Vec<&str> = production.lines().collect();

    let mut current_fn = String::new();
    let mut sites = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        if let Some(after) = trimmed
            .contains("fn ")
            .then(|| trimmed.split("fn ").nth(1))
            .flatten()
        {
            let end = after.find(['(', '<']).unwrap_or(after.len());
            current_fn = after[..end].trim().to_string();
        }

        let is_capacity = line.contains("with_capacity(") || is_vec_repeat(line);
        if !is_capacity {
            continue;
        }

        // Scan back from this line (inclusive) to the nearest `let `.
        let mut binding = None;
        for back in (0..=i).rev() {
            let candidate = lines[back];
            if let Some(at) = candidate.find("let ") {
                let after = &candidate[at + "let ".len()..];
                let after = after.strip_prefix("mut ").unwrap_or(after);
                let end = after.find([':', ' ', '=']).unwrap_or(after.len());
                binding = Some(after[..end].trim().to_string());
                break;
            }
        }
        if let Some(binding) = binding {
            sites.push((current_fn.clone(), binding));
        }
    }

    sites
}

/// `true` when the line contains a `vec![value; count]` repeat expression —
/// a `;` inside the brackets, not just the statement terminator after them.
fn is_vec_repeat(line: &str) -> bool {
    let Some(at) = line.find("vec![") else {
        return false;
    };
    let after = &line[at + "vec![".len()..];
    after
        .find(']')
        .is_some_and(|close| after[..close].contains(';'))
}

#[test]
fn every_shared_row_carries_a_non_pending_outcome_with_a_reason() {
    let doc = read(RECORD);
    let region = shared_region(&doc);
    let rows = table_rows(region);

    let paths: std::collections::BTreeSet<String> = rows
        .iter()
        .filter_map(|r| r.first())
        .map(|p| p.trim_matches('`').to_string())
        .collect();
    let expected: std::collections::BTreeSet<String> =
        SHARED_FILES.iter().map(ToString::to_string).collect();
    assert_eq!(
        paths, expected,
        "the `shared` section must carry exactly the five files it owns, no more and no fewer"
    );
    assert_eq!(
        rows.len(),
        5,
        "the `shared` section must carry exactly one row per file, no duplicates"
    );

    for row in &rows {
        let path = row.first().expect("row has a path cell");
        let outcome = row
            .get(2)
            .unwrap_or_else(|| panic!("row for {path} must carry a third (Outcome) cell"));
        assert!(
            !outcome.is_empty(),
            "{path}'s Outcome cell must not be empty"
        );
        assert!(
            !outcome.starts_with("pending"),
            "{path} was swept by the `shared` sweep, so its outcome must not read `pending`: \
             {outcome}"
        );
        let split = outcome.split_once(" — ");
        assert!(
            split.is_some_and(|(before, after)| !before.is_empty() && !after.is_empty()),
            "{path}'s Outcome cell must read `<outcome> — <reason>` with non-empty text on \
             both sides, got: {outcome}"
        );
    }
}

#[test]
fn every_shared_capacity_site_is_cited_by_symbol() {
    let mut all_sites: Vec<(&str, String, String)> = Vec::new();
    for file in SHARED_FILES {
        let sites = production_capacity_sites(file);
        for (func, binding) in sites {
            all_sites.push((file, func, binding));
        }
    }
    assert!(
        !all_sites.is_empty(),
        "precondition: production_capacity_sites must find at least one site in the shared \
         files (topology_cache.rs alone has seven `with_capacity` calls), otherwise this test \
         passes vacuously"
    );

    let doc = read(RECORD);
    let rows = shared_capacity_rows(&doc);

    for (file, func, binding) in &all_sites {
        let basename = file.rsplit('/').next().expect("path names a file");
        let cited = rows.iter().any(|row| {
            row.first().is_some_and(|first| {
                first.contains(&format!("{basename}::"))
                    && first.contains(&format!("::{func}"))
                    && first.contains(&format!("`{binding}`"))
            })
        });
        assert!(
            cited,
            "{file}'s capacity site in `{func}` (binding `{binding}`) must be cited by symbol \
             in a `shared` row of `## Capacity and traversal table` — an uncited site could be \
             deleted from the row without the ledger noticing"
        );
    }
}

#[test]
fn the_capacity_table_has_exactly_eight_shared_rows() {
    let doc = read(RECORD);
    let rows = shared_capacity_rows(&doc);
    assert_eq!(
        rows.len(),
        SHARED_CAPACITY_ROWS,
        "the `shared` section must own exactly {SHARED_CAPACITY_ROWS} capacity-table rows"
    );

    let topology_cache_rows = rows
        .iter()
        .filter(|row| {
            row.first()
                .is_some_and(|first| first.contains("CreatureTopologyCache::new"))
        })
        .count();
    assert_eq!(
        topology_cache_rows, 7,
        "exactly 7 of the shared capacity rows must cite `CreatureTopologyCache::new`"
    );
    let compute_ranks_rows = rows
        .iter()
        .filter(|row| {
            row.first()
                .is_some_and(|first| first.contains("compute_ranks"))
        })
        .count();
    assert_eq!(
        compute_ranks_rows, 1,
        "exactly 1 of the shared capacity rows must cite `compute_ranks`"
    );

    for row in &rows {
        let symbol = row.first().cloned().unwrap_or_default();
        let sized_from = row.get(2).cloned().unwrap_or_default();
        let bound = row
            .get(3)
            .unwrap_or_else(|| panic!("row {symbol} must carry a fourth (Bound) cell"));
        assert!(
            !bound.is_empty() && bound != "pending",
            "{symbol}'s Bound cell must be filled in and not `pending`, got: {bound}"
        );
        if sized_from.contains("neuron_count") {
            assert!(
                bound.contains("no numeric cap"),
                "{symbol} is sized from `neuron_count`, so its Bound cell must state there is \
                 no numeric cap, got: {bound}"
            );
        }
    }
}

#[test]
fn every_issue_linked_from_a_shared_row_appears_under_issues_filed() {
    let doc = read(RECORD);

    let mut issue_refs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    collect_issue_refs(shared_region(&doc), &mut issue_refs);
    let capacity_region = section(&doc, "## Capacity and traversal table");
    let shared_rows_text: String = shared_capacity_rows(&doc)
        .iter()
        .map(|row| row.join(" | "))
        .collect::<Vec<_>>()
        .join("\n");
    let _ = capacity_region; // rows already extracted structurally above
    collect_issue_refs(&shared_rows_text, &mut issue_refs);

    assert!(
        !issue_refs.is_empty(),
        "precondition: the `shared` section must reference at least one `#N` issue, otherwise \
         this test passes vacuously"
    );

    let issues_filed = section(&doc, "## Issues filed");
    for issue in &issue_refs {
        let needle = format!("#{issue}");
        let found = issues_filed.match_indices(&needle).any(|(at, _)| {
            let after = &issues_filed[at + needle.len()..];
            !after.chars().next().is_some_and(|c| c.is_ascii_digit())
        });
        assert!(
            found,
            "issue {needle} is referenced from a `shared` row, so it must also appear under \
             `## Issues filed` — an unlinked finding is invisible to the finalisation sub-issue \
             that reconciles this record"
        );
    }
}

/// Collect every `#<digits>` reference in `text` into `out`, as the bare
/// digit string (no `#`).
fn collect_issue_refs(text: &str, out: &mut std::collections::BTreeSet<String>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'#' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 {
                out.insert(text[i + 1..j].to_string());
            }
            i = j;
        } else {
            i += 1;
        }
    }
}
