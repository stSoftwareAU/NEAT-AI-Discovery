//! Contract tests for the `pairwise` section of the chunk 8a sweep record
//! (Issues #2294 and #2295, part of #2150).
//!
//! `SWEPT` holds all nine files the `pairwise` section owns: the five
//! Issue #2294 swept, plus the four Issue #2295 swept
//! (`opposing_synapse.rs`, `output_conflict.rs`, `hard_sample_cluster.rs`,
//! `sentinel_cluster.rs`) together with the `TRAVERSAL` rows those files
//! add. Finalisation (#2302, part of #2154) `git mv`d the record to the
//! top level and retargeted `RECORD` there in the same commit.

use std::path::PathBuf;

/// The chunk 8a prose record, promoted to the top level by finalisation
/// (#2302, part of #2154).
const RECORD: &str = "docs/audits/security-sweep-chunk-08a-detection-neuron.md";

/// The nine files the `pairwise` section owns, in record order: the five
/// Issue #2294 swept, then the four Issue #2295 swept.
const SWEPT: &[&str] = &[
    "src/analysis/detection/correlated_error.rs",
    "src/analysis/detection/weight_coherence.rs",
    "src/analysis/detection/co_adaptation.rs",
    "src/analysis/detection/symmetry_breaking.rs",
    "src/analysis/detection/fanin_polarity_conflict.rs",
    "src/analysis/detection/opposing_synapse.rs",
    "src/analysis/detection/output_conflict.rs",
    "src/analysis/detection/hard_sample_cluster.rs",
    "src/analysis/detection/sentinel_cluster.rs",
];

/// One entry per traversal / pairwise-loop row this sweep added to
/// `## Capacity and traversal table`: (source path, function name, qualifier
/// text as written in the row's Symbol cell parentheses).
const TRAVERSAL: &[(&str, &str, &str)] = &[
    (
        "src/analysis/detection/correlated_error.rs",
        "detect_correlated_error_patterns",
        "i<j correlation pair loop",
    ),
    (
        "src/analysis/detection/correlated_error.rs",
        "detect_correlated_error_patterns",
        "group mean-correlation pair loop",
    ),
    (
        "src/analysis/detection/correlated_error.rs",
        "cluster_correlated_outputs",
        "i<j complete-linkage loop",
    ),
    (
        "src/analysis/detection/correlated_error.rs",
        "find_predictive_inputs",
        "inputs × shared samples, per group",
    ),
    (
        "src/analysis/detection/weight_coherence.rs",
        "detect_symmetric_cancellation",
        "per-`topo.fan_in`-target i<j loop over `incoming_synapses`",
    ),
    (
        "src/analysis/detection/co_adaptation.rs",
        "detect_co_adapted_neurons",
        "i<j loop over `eligible`",
    ),
    (
        "src/analysis/detection/symmetry_breaking.rs",
        "detect_symmetric_neurons",
        "`hidden_neurons.iter().any` per synapse",
    ),
    (
        "src/analysis/detection/symmetry_breaking.rs",
        "detect_symmetric_neurons",
        "i<j loop over `eligible_neurons`",
    ),
    (
        "src/analysis/detection/symmetry_breaking.rs",
        "build_weight_vector",
        "linear `find` per source, twice per pair",
    ),
    (
        "src/analysis/detection/fanin_polarity_conflict.rs",
        "detect_fanin_polarity_conflicts",
        "`incoming_by_target` pass",
    ),
    (
        "src/analysis/detection/fanin_polarity_conflict.rs",
        "fanin_polarity_conflicts_to_coordinated_candidates",
        "`synapses_to_move` pass",
    ),
    (
        "src/analysis/detection/fanin_polarity_conflict.rs",
        "fanin_polarity_conflicts_to_coordinated_candidates",
        "`creature.neurons.iter().find` per candidate",
    ),
    (
        "src/analysis/detection/opposing_synapse.rs",
        "detect_opposing_synapses",
        "synapses × `source_records`",
    ),
    (
        "src/analysis/detection/output_conflict.rs",
        "output_conflicts_to_coordinated_candidates",
        "`existing_synapses` filter per conflict",
    ),
    (
        "src/analysis/detection/output_conflict.rs",
        "output_conflicts_to_coordinated_candidates",
        "`existing_synapses.iter().find` per harmed output",
    ),
    (
        "src/analysis/detection/output_conflict.rs",
        "split_neuron_uuid",
        "FNV-1a over the key bytes",
    ),
    (
        "src/analysis/detection/hard_sample_cluster.rs",
        "aggregate_obs_errors",
        "outputs × records pass",
    ),
    (
        "src/analysis/detection/hard_sample_cluster.rs",
        "find_dominant_inputs",
        "inputs × records pass",
    ),
    (
        "src/analysis/detection/hard_sample_cluster.rs",
        "hard_sample_clusters_to_coordinated_candidates",
        "clusters × outputs `AddSynapse` fan-out",
    ),
    (
        "src/analysis/detection/hard_sample_cluster.rs",
        "hard_sample_neuron_uuid",
        "FNV-1a over the key bytes",
    ),
    (
        "src/analysis/detection/sentinel_cluster.rs",
        "assess_sentinel_cluster",
        "`sentinel_indices.contains` per sample",
    ),
    (
        "src/analysis/detection/sentinel_cluster.rs",
        "compute_error_variance",
        "two passes over `indices`",
    ),
];

/// The six #2092 defect classes, quoted verbatim (bold Markdown) as the
/// `pairwise` outcomes cite them.
const DEFECT_CLASSES: [&str; 6] = [
    "**allocation**",
    "**recursion**",
    "**quadratic**",
    "**panic**",
    "**integer overflow**",
    "**cache poisoning**",
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

/// The text of the `pairwise` section: from its `<!-- section: pairwise -->`
/// marker to the next `### ` or `## ` heading.
fn pairwise_region(doc: &str) -> &str {
    let marker = "<!-- section: pairwise -->";
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

/// The production half of a source file: everything before the first
/// `#[cfg(test)]`. Sweep verdicts are about code an untrusted input can
/// reach, and a fixture in a `#[cfg(test)] mod tests` is not that code.
fn production_source(rel: &str) -> String {
    let body = read(rel);
    match body.find("#[cfg(test)]") {
        Some(at) => body[..at].to_string(),
        None => body,
    }
}

/// `true` when the line allocates a collection whose size is an expression —
/// `with_capacity(…)`, `reserve(…)`, or the `vec![value; count]` form. Uses
/// `rfind(']')` so the nested `vec![vec![0.0; n]; n]` counts as one line.
fn is_capacity_site(line: &str) -> bool {
    if line.contains("with_capacity(") || line.contains(".reserve(") {
        return true;
    }
    let Some(at) = line.find("vec![") else {
        return false;
    };
    let after = &line[at + "vec![".len()..];
    after
        .rfind(']')
        .is_some_and(|close| after[..close].contains(';'))
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

fn basename(rel: &str) -> &str {
    rel.rsplit('/').next().expect("path names a file")
}

/// The capacity-table rows for one swept file's basename: rows whose Symbol
/// cell starts with `` `<basename>:: ``.
fn capacity_table_rows_for(doc: &str, base: &str) -> Vec<Vec<String>> {
    table_rows(section(doc, "## Capacity and traversal table"))
        .into_iter()
        .filter(|row| {
            row.first()
                .is_some_and(|first| first.starts_with(&format!("`{base}::")))
        })
        .collect()
}

#[test]
fn each_swept_file_has_exactly_one_filled_pairwise_row() {
    let doc = read(RECORD);
    let region = pairwise_region(&doc);
    let rows = table_rows(region);

    for path in SWEPT {
        let matching: Vec<&Vec<String>> = rows
            .iter()
            .filter(|row| {
                row.first()
                    .is_some_and(|first| first.trim_matches('`') == *path)
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "{path} must have exactly one row in the `pairwise` section — a missing or \
             duplicated row means the ledger no longer matches what was swept"
        );
        let outcome = matching[0]
            .get(2)
            .unwrap_or_else(|| panic!("{path}'s row must carry a third (Outcome) cell"));
        assert!(
            !outcome.contains("pending"),
            "{path} was swept by Issues #2294 / #2295, so its outcome must not read `pending`: \
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
fn each_swept_row_states_all_six_defect_classes_probed() {
    let doc = read(RECORD);
    let region = pairwise_region(&doc);
    let rows = table_rows(region);

    for path in SWEPT {
        let row = rows
            .iter()
            .find(|row| {
                row.first()
                    .is_some_and(|first| first.trim_matches('`') == *path)
            })
            .unwrap_or_else(|| panic!("{path} must have a row in the `pairwise` section"));
        let outcome = row
            .get(2)
            .unwrap_or_else(|| panic!("{path}'s row must carry a third (Outcome) cell"));
        for class in DEFECT_CLASSES {
            assert!(
                outcome.contains(class),
                "{path}'s outcome must state that {class} was probed, so a reader can trust \
                 the sweep covered all six #2092 defect classes and not just the ones that \
                 found something; got: {outcome}"
            );
        }
    }
}

#[test]
fn capacity_row_count_per_file_matches_the_production_capacity_sites() {
    let doc = read(RECORD);
    let mut total = 0usize;

    for path in SWEPT {
        let base = basename(path);
        let source_sites = production_source(path)
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .filter(|line| is_capacity_site(line))
            .count();

        let table_capacity_rows = capacity_table_rows_for(&doc, base)
            .into_iter()
            .filter(|row| row.get(1).is_some_and(|kind| kind == "capacity"))
            .count();

        assert_eq!(
            source_sites, table_capacity_rows,
            "{path} has {source_sites} capacity site(s) in production code but the capacity \
             table carries {table_capacity_rows} `capacity`-kind row(s) for it — an uncited \
             site could be deleted from the row without the ledger noticing, and a stale row \
             describes code that is gone"
        );
        total += source_sites;
    }

    assert!(
        total > 0,
        "precondition: the nine swept files must together carry at least one capacity site \
         (correlated_error.rs's dense correlation matrix alone is one), otherwise the equality \
         checks above pass vacuously"
    );
}

#[test]
fn every_traversal_symbol_has_a_row_marked_module_level_only() {
    let doc = read(RECORD);
    let capacity_region = section(&doc, "## Capacity and traversal table");
    let rows = table_rows(capacity_region);

    for (path, name, qualifier) in TRAVERSAL {
        assert!(
            production_source(path).contains(&format!("fn {name}")),
            "the `pairwise` sweep cites `{name}` in {path}, but no `fn {name}` is declared \
             there any more — the sweep describes code that has moved or gone"
        );

        let base = basename(path);
        let symbol_needle = format!("`{base}::{name}`");
        let qualifier_needle = format!("({qualifier})");
        let found = rows.iter().any(|row| {
            let symbol = row.first().map(String::as_str).unwrap_or_default();
            let kind = row.get(1).map(String::as_str).unwrap_or_default();
            let cancellation = row.get(4).map(String::as_str).unwrap_or_default();
            symbol.contains(&symbol_needle)
                && symbol.contains(&qualifier_needle)
                && (kind == "pairwise loop" || kind == "traversal")
                && cancellation == "no (module-level only)"
        });
        assert!(
            found,
            "`## Capacity and traversal table` must carry a `pairwise loop` or `traversal` row \
             for `{base}::{name}` ({qualifier}) marked `no (module-level only)` in its \
             Cancellation-checked? cell — an uncited loop is invisible to a reader deciding \
             whether it needs a deadline check"
        );
    }
}

#[test]
fn no_swept_file_checks_cancellation_itself() {
    for path in SWEPT {
        let source = read(path);
        assert!(
            !source.contains("deadline_passed") && !source.contains("is_cancelled"),
            "{path} now checks cancellation itself, but this sweep recorded every pairwise \
             loop in it as `no (module-level only)` — if a fix adds a per-iteration check, \
             update that file's traversal rows' Cancellation-checked? cell and this test \
             together"
        );
    }
}

#[test]
fn every_issue_linked_from_a_swept_row_appears_under_issues_filed() {
    let doc = read(RECORD);
    let pairwise = pairwise_region(&doc);
    let rows = table_rows(pairwise);

    let mut issue_refs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for path in SWEPT {
        if let Some(row) = rows.iter().find(|row| {
            row.first()
                .is_some_and(|first| first.trim_matches('`') == *path)
        }) {
            collect_issue_refs(&row.join(" | "), &mut issue_refs);
        }
        let base = basename(path);
        for row in capacity_table_rows_for(&doc, base) {
            collect_issue_refs(&row.join(" | "), &mut issue_refs);
        }
    }

    assert!(
        !issue_refs.is_empty(),
        "precondition: the swept `pairwise` rows and their capacity/traversal rows must \
         reference at least one `#N` issue, otherwise this test passes vacuously"
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
            "issue {needle} is referenced from a swept `pairwise` row, so it must also appear \
             under `## Issues filed` — an unlinked finding is invisible to the finalisation \
             sub-issue that reconciles this record"
        );
    }
}

#[test]
fn pairwise_section_holds_exactly_nine_rows_none_pending() {
    assert_eq!(
        SWEPT.len(),
        9,
        "SWEPT must list all nine files the `pairwise` section owns, or the checks below are \
         checking the wrong count"
    );

    let doc = read(RECORD);
    let region = pairwise_region(&doc);
    let rows = table_rows(region);

    assert_eq!(
        rows.len(),
        9,
        "the `pairwise` section must carry exactly nine rows now that both sub-issues have \
         landed — a missing or extra row means the ledger no longer matches the files the \
         section owns"
    );

    for row in &rows {
        let outcome = row
            .get(2)
            .unwrap_or_else(|| panic!("every `pairwise` row must carry a third (Outcome) cell"));
        assert!(
            !outcome.contains("pending"),
            "the `pairwise` section is complete (Issues #2294 and #2295), so no row may still \
             read `pending`: {outcome}"
        );
    }

    let row_paths: std::collections::BTreeSet<String> = rows
        .iter()
        .filter_map(|row| row.first())
        .map(|first| first.trim_matches('`').to_string())
        .collect();
    let swept_paths: std::collections::BTreeSet<String> =
        SWEPT.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(
        row_paths, swept_paths,
        "the set of paths in the `pairwise` section's rows must exactly match SWEPT — a \
         mismatch means either the ledger carries a file this test does not know about, or \
         SWEPT claims a file the ledger does not"
    );
}

#[test]
fn files_without_capacity_sites_state_none_explicitly() {
    let doc = read(RECORD);
    let region = pairwise_region(&doc);
    let rows = table_rows(region);

    let mut checked_any = false;
    for path in SWEPT {
        let source_sites = production_source(path)
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .filter(|line| is_capacity_site(line))
            .count();
        if source_sites != 0 {
            continue;
        }
        checked_any = true;

        let row = rows
            .iter()
            .find(|row| {
                row.first()
                    .is_some_and(|first| first.trim_matches('`') == *path)
            })
            .unwrap_or_else(|| panic!("{path} must have a row in the `pairwise` section"));
        let outcome = row
            .get(2)
            .unwrap_or_else(|| panic!("{path}'s row must carry a third (Outcome) cell"));
        assert!(
            outcome.contains("capacity sites: none"),
            "{path} has no `with_capacity` / `vec![_; n]` / `.reserve(` site in production code, \
             but its Outcome cell does not say `capacity sites: none` — a reader cannot tell an \
             audited absence from an unaudited one: {outcome}"
        );
    }

    assert!(
        checked_any,
        "precondition: at least one SWEPT file (hard_sample_cluster.rs and sentinel_cluster.rs \
         have none) must have zero capacity sites, otherwise this test passes vacuously"
    );
}

#[test]
fn synthetic_uuid_helpers_record_a_collision_verdict_cross_linked_to_2153() {
    let doc = read(RECORD);
    let capacity_region = section(&doc, "## Capacity and traversal table");
    let rows = table_rows(capacity_region);

    for (path, name) in [
        (
            "src/analysis/detection/hard_sample_cluster.rs",
            "hard_sample_neuron_uuid",
        ),
        (
            "src/analysis/detection/output_conflict.rs",
            "split_neuron_uuid",
        ),
    ] {
        assert!(
            production_source(path).contains(&format!("fn {name}")),
            "the `pairwise` sweep cites `{name}` in {path}, but no `fn {name}` is declared \
             there any more — the sweep describes code that has moved or gone"
        );

        let base = basename(path);
        let symbol_needle = format!("`{base}::{name}`");
        let found = rows.iter().any(|row| {
            let symbol = row.first().map(String::as_str).unwrap_or_default();
            let bound = row.get(3).map(String::as_str).unwrap_or_default();
            symbol.contains(&symbol_needle)
                && bound.contains("Collision verdict")
                && bound.contains("#2153")
        });
        assert!(
            found,
            "`## Capacity and traversal table` must carry a row for `{base}::{name}` whose \
             Bound cell states a `Collision verdict` cross-linked to #2153 — without it a \
             reader cannot tell whether the synthetic UUID this helper derives was ever checked \
             against `RecordCache`'s key derivation, which #2153 owns"
        );
    }
}
