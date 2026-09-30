//! Contract tests for the `neuron` section of the staged chunk 8a sweep
//! record (Issue #2300, part of #2153).
//!
//! The record is staged under `docs/audits/in-progress/` while the chunk 8a
//! audit sub-issues fill their sections; `tests/issue_2280_chunk_08a_ledger_scaffold.rs`
//! gates the record's shape. This file gates only the `neuron` section's
//! content, and only the two files Issue #2300 swept
//! (`src/analysis/neuron/mod.rs` and `src/analysis/neuron/preparation.rs`); a
//! later sub-issue widens `SWEPT` to the remaining three files
//! (`evaluation.rs`, `post_processing.rs`, `ranking_score.rs`) before the
//! `neuron` section as a whole can flip from `pending`. Finalisation (#2154)
//! `git mv`s the record to the top level in the commit that sets the chunk
//! `"8a"` index entry, so `RECORD` below changes to
//! `docs/audits/security-sweep-chunk-08a-detection-neuron.md` at that point.

use std::path::PathBuf;

/// The staged chunk 8a prose record. Finalisation (#2154) moves this to
/// `docs/audits/security-sweep-chunk-08a-detection-neuron.md`.
const RECORD: &str = "docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md";

/// The neuron files Issue #2300 swept. The next sub-issue widens this to all five.
const SWEPT: [&str; 2] = [
    "src/analysis/neuron/mod.rs",
    "src/analysis/neuron/preparation.rs",
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

/// The text of the `neuron` section: from its `<!-- section: neuron -->`
/// marker to the next `### ` or `## ` heading.
fn neuron_region(doc: &str) -> &str {
    let marker = "<!-- section: neuron -->";
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

/// Split a Markdown table row into trimmed cells, on `|` **not** preceded by
/// a backslash, unescaping `\|` to a literal `|` in the cell text. Several
/// cells in this ledger carry a GFM-escaped `||` guard expression (e.g.
/// `` analysis_timed_out \|\| saturation_aborted \|\| deadline_passed(&deadline) ``)
/// and a naive `split('|')` would slice that expression apart.
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

/// The neuron rows of `## Capacity and traversal table`: rows whose Symbol
/// cell names a `src/analysis/neuron/` path.
fn neuron_capacity_rows(doc: &str) -> Vec<Vec<String>> {
    table_rows(section(doc, "## Capacity and traversal table"))
        .into_iter()
        .filter(|row| {
            row.first()
                .is_some_and(|first| first.contains("src/analysis/neuron/"))
        })
        .collect()
}

/// The production half of a source file: everything before the `#[cfg(test)]`
/// that actually opens the test module.
///
/// A naive cut at the *first* `#[cfg(test)]` is wrong here:
/// `src/analysis/neuron/post_processing.rs` has a test-only
/// `#[cfg(test)] fn apply_per_target_cap` declared *before* the production
/// `fn apply_distinct_target_spread`, so cutting there would silently drop
/// capacity sites once `SWEPT` widens to cover that file. Instead, find the
/// `#[cfg(test)]` whose next non-blank, non-attribute, non-comment line
/// starts with `mod ` (the real `mod tests { ... }` / `mod some_tests { ... }`
/// block), and cut there. If no such pair exists, return the whole file.
fn production_source(rel: &str) -> String {
    let body = read(rel);
    let lines: Vec<&str> = body.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        if line.trim() != "#[cfg(test)]" {
            continue;
        }
        // Find the next non-blank, non-attribute, non-comment line.
        let next = lines[i + 1..].iter().find(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#') && !t.starts_with("//")
        });
        if next.is_some_and(|l| l.trim_start().starts_with("mod ")) {
            // Cut at the byte offset of this `#[cfg(test)]` line.
            let mut offset = 0usize;
            for earlier in &lines[..i] {
                offset += earlier.len() + 1;
            }
            return body[..offset].to_string();
        }
    }
    body
}

#[test]
fn production_source_skips_the_test_only_cfg_test_fn_in_post_processing() {
    let file = "src/analysis/neuron/post_processing.rs";
    let full = read(file);
    assert!(
        full.contains("fn apply_distinct_target_spread"),
        "precondition: {file} must still declare `fn apply_distinct_target_spread`, otherwise \
         this test is checking against code that has moved or gone"
    );
    let per_target_cap_at = full
        .find("#[cfg(test)]\npub(crate) fn apply_per_target_cap")
        .unwrap_or_else(|| {
            panic!(
                "precondition: {file} must carry a test-only `#[cfg(test)] fn apply_per_target_cap` \
                 ahead of `apply_distinct_target_spread` — if this test-only fn has been removed or \
                 renamed, `production_source`'s cut-at-the-real-mod-block logic is untested against \
                 its motivating case"
            )
        });
    let spread_at = full
        .find("fn apply_distinct_target_spread")
        .expect("checked above");
    assert!(
        per_target_cap_at < spread_at,
        "precondition: the test-only `#[cfg(test)] fn apply_per_target_cap` must sit before \
         `apply_distinct_target_spread` in {file} — that ordering is exactly what makes cutting \
         at the first `#[cfg(test)]` wrong"
    );

    let production = production_source(file);
    assert!(
        production.contains("fn apply_distinct_target_spread"),
        "production_source({file}) must still contain `fn apply_distinct_target_spread` — cutting \
         at the first `#[cfg(test)]` (the test-only `apply_per_target_cap`) would silently drop it"
    );
}

/// `true` when the line allocates a collection whose size is an expression —
/// `with_capacity(…)`, `.reserve(…)`, or the `vec![value; count]` form.
/// Equivalent to the regex `with_capacity\(|vec!\[[^]]*;|\.reserve\(`.
fn is_capacity_site(line: &str) -> bool {
    if line.trim_start().starts_with("//") {
        return false;
    }
    if line.contains("with_capacity(") || line.contains(".reserve(") {
        return true;
    }
    let Some(at) = line.find("vec![") else {
        return false;
    };
    let after = &line[at + "vec![".len()..];
    after
        .find(']')
        .is_some_and(|close| after[..close].contains(';'))
}

/// Capacity sites in the production half of a source file, paired with the
/// enclosing function name and the `let` binding the site initialises.
fn production_capacity_sites(rel: &str) -> Vec<(String, String)> {
    let production = production_source(rel);
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

        if !is_capacity_site(line) {
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

#[test]
fn each_swept_row_is_present_and_not_pending() {
    let doc = read(RECORD);
    let region = neuron_region(&doc);
    let rows = table_rows(region);

    assert_eq!(
        rows.len(),
        5,
        "the `neuron` section must still carry exactly 5 rows (one per file it owns) — a \
         changed row count means a file was added, removed, or duplicated"
    );

    for path in SWEPT {
        let matching: Vec<&Vec<String>> = rows
            .iter()
            .filter(|row| {
                row.first()
                    .is_some_and(|first| first.trim_matches('`') == path)
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "the `neuron` section must carry exactly one row for {path}, got {}",
            matching.len()
        );
        let row = matching[0];
        let outcome = row
            .get(2)
            .unwrap_or_else(|| panic!("row for {path} must carry a third (Outcome) cell"));
        assert!(
            !outcome.is_empty(),
            "{path}'s Outcome cell must not be empty"
        );
        assert!(
            !outcome.contains("pending"),
            "{path} was swept by Issue #2300, so its outcome must not read `pending`: {outcome}"
        );
        let split = outcome.split_once(" — ");
        assert!(
            split.is_some_and(|(before, after)| !before.is_empty() && !after.is_empty()),
            "{path}'s Outcome cell must read `<outcome> — <reason>` with non-empty text on both \
             sides, got: {outcome}"
        );
    }
}

#[test]
fn every_capacity_site_in_a_swept_file_is_cited_by_symbol() {
    // Issue #1799: an assertion that holds either way is not coverage. Pin
    // one concrete site first so the loop below cannot pass vacuously.
    let preparation_source = production_source("src/analysis/neuron/preparation.rs");
    assert!(
        preparation_source.contains(
            "HashMap::with_capacity(input.creature.input + input.creature.neurons.len())"
        ),
        "precondition: src/analysis/neuron/preparation.rs must still carry the \
         `neuron_type_map` capacity site the ledger cites, otherwise this test checks nothing"
    );

    let mut all_sites: Vec<(&str, String, String)> = Vec::new();
    for file in SWEPT {
        for (func, binding) in production_capacity_sites(file) {
            all_sites.push((file, func, binding));
        }
    }
    assert!(
        !all_sites.is_empty(),
        "precondition: production_capacity_sites must find at least one site across the swept \
         neuron files, otherwise this test passes vacuously"
    );

    let doc = read(RECORD);
    let rows = neuron_capacity_rows(&doc);

    for (file, func, binding) in &all_sites {
        let cited = rows.iter().any(|row| {
            row.first().is_some_and(|first| {
                first.starts_with(&format!("`{file}::{func}"))
                    && first.contains(&format!("`{binding}`"))
            }) && row.get(1).is_some_and(|kind| kind == "capacity")
        });
        assert!(
            cited,
            "{file}'s capacity site in `{func}` (binding `{binding}`) must be cited by symbol in \
             a neuron row of `## Capacity and traversal table` with Kind `capacity` — an uncited \
             site could be deleted from the row without the ledger noticing"
        );
    }

    let capacity_rows_for_swept_files = rows
        .iter()
        .filter(|row| {
            row.first().is_some_and(|first| {
                SWEPT
                    .iter()
                    .any(|file| first.starts_with(&format!("`{file}::")))
            }) && row.get(1).is_some_and(|kind| kind == "capacity")
        })
        .count();
    assert_eq!(
        capacity_rows_for_swept_files,
        all_sites.len(),
        "the number of neuron capacity rows citing a swept file must equal the number of \
         capacity sites actually found in those files — a stale row for deleted code must fail \
         this too"
    );

    // mod.rs has no capacity site, and the ledger must say so explicitly.
    let mod_rs_has_site = production_capacity_sites("src/analysis/neuron/mod.rs").len();
    assert_eq!(
        mod_rs_has_site, 0,
        "precondition: src/analysis/neuron/mod.rs must have zero capacity sites for this branch \
         of the test to be meaningful"
    );
    let mod_rs_outcome = table_rows(neuron_region(&doc))
        .into_iter()
        .find(|row| {
            row.first()
                .is_some_and(|first| first.trim_matches('`') == "src/analysis/neuron/mod.rs")
        })
        .and_then(|row| row.get(2).cloned())
        .expect("mod.rs row must exist (checked above)");
    assert!(
        mod_rs_outcome.contains("hit in `mod.rs`"),
        "src/analysis/neuron/mod.rs has no capacity site, so its Outcome must record the \
         negative result (\"... hit in `mod.rs`\"), got: {mod_rs_outcome}"
    );
}

#[test]
fn the_try_fold_traversal_row_is_cancellation_checked_per_focus_target() {
    // Pin the source this row claims to describe.
    let mod_rs_source = production_source("src/analysis/neuron/mod.rs");
    assert!(
        mod_rs_source.contains(".try_fold("),
        "precondition: src/analysis/neuron/mod.rs must still call `.try_fold(`, otherwise the \
         traversal row describes code that has moved or gone"
    );
    assert!(
        mod_rs_source.contains("deadline_passed(&deadline)"),
        "precondition: src/analysis/neuron/mod.rs must still call `deadline_passed(&deadline)`, \
         otherwise the cancellation-checked claim below is unfalsifiable"
    );

    let doc = read(RECORD);
    let rows = neuron_capacity_rows(&doc);
    let matching: Vec<&Vec<String>> = rows
        .iter()
        .filter(|row| {
            row.first().is_some_and(|first| {
                first.starts_with("`src/analysis/neuron/mod.rs::") && first.contains("try_fold")
            })
        })
        .collect();
    assert_eq!(
        matching.len(),
        1,
        "there must be exactly one neuron capacity-table row for the `mod.rs` try_fold \
         traversal, got {}",
        matching.len()
    );
    let row = matching[0];

    let kind = row
        .get(1)
        .unwrap_or_else(|| panic!("try_fold row must carry a Kind cell"));
    assert_eq!(
        kind, "traversal",
        "the try_fold row's Kind must be `traversal`, got: {kind}"
    );

    let cancellation = row
        .get(4)
        .unwrap_or_else(|| panic!("try_fold row must carry a Cancellation-checked cell"));
    assert!(
        cancellation.starts_with("yes (per focus target)"),
        "the try_fold row's Cancellation-checked cell must start with `yes (per focus target)`, \
         got: {cancellation}"
    );
    assert!(
        cancellation
            .contains("analysis_timed_out || saturation_aborted || deadline_passed(&deadline)"),
        "the try_fold row's Cancellation-checked cell must quote the guard expression \
         `analysis_timed_out || saturation_aborted || deadline_passed(&deadline)` (after \
         unescaping `\\|` back to `|`), got: {cancellation}"
    );
}

#[test]
fn the_mod_rs_outcome_records_the_record_cache_verdict() {
    let doc = read(RECORD);
    let mod_rs_outcome = table_rows(neuron_region(&doc))
        .into_iter()
        .find(|row| {
            row.first()
                .is_some_and(|first| first.trim_matches('`') == "src/analysis/neuron/mod.rs")
        })
        .and_then(|row| row.get(2).cloned())
        .expect("mod.rs row must exist");

    for needle in [
        "RecordCache",
        "#2295",
        "hard_sample_neuron_uuid",
        "split_neuron_uuid",
        "OnceLock",
        "shared_cache",
    ] {
        assert!(
            mod_rs_outcome.contains(needle),
            "the mod.rs Outcome must record the `RecordCache` collision verdict — missing \
             `{needle}`, got: {mod_rs_outcome}"
        );
    }
}

#[test]
fn every_issue_linked_from_a_swept_row_appears_under_issues_filed() {
    let doc = read(RECORD);
    let region = neuron_region(&doc);
    let rows = table_rows(region);

    let mut issue_refs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for path in SWEPT {
        if let Some(outcome) = rows
            .iter()
            .find(|row| {
                row.first()
                    .is_some_and(|first| first.trim_matches('`') == path)
            })
            .and_then(|row| row.get(2))
        {
            collect_issue_refs(outcome, &mut issue_refs);
        }
    }
    let capacity_rows_text: String = neuron_capacity_rows(&doc)
        .iter()
        .map(|row| row.join(" | "))
        .collect::<Vec<_>>()
        .join("\n");
    collect_issue_refs(&capacity_rows_text, &mut issue_refs);

    assert!(
        !issue_refs.is_empty(),
        "precondition: the swept neuron rows must reference at least one `#N` issue, otherwise \
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
            "issue {needle} is referenced from a swept `neuron` row, so it must also appear \
             under `## Issues filed` — an unlinked finding is invisible to the finalisation \
             sub-issue that reconciles this record"
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
