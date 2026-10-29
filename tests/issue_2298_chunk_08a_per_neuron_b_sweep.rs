//! Contract tests for the whole `per-neuron-b` section of the chunk 8a sweep
//! record (part of #2152): the first six rows are swept by Issue #2298, and
//! the remaining six by Issue #2299. Finalisation (#2302, part of #2154)
//! promoted the record to the top level and retargeted `RECORD` below.

use std::path::PathBuf;

/// The chunk 8a prose record, promoted to the top level by finalisation
/// (#2302, part of #2154).
const RECORD: &str = "docs/audits/security-sweep-chunk-08a-detection-neuron.md";

/// The twelve files the `per-neuron-b` section sweeps: the first six under
/// Issue #2298, and the remaining six under Issue #2299.
const SWEPT: &[&str] = &[
    "src/analysis/detection/operating_point.rs",
    "src/analysis/detection/oscillating_neuron.rs",
    "src/analysis/detection/output_range_compression.rs",
    "src/analysis/detection/output_squash_mismatch.rs",
    "src/analysis/detection/restricted_range.rs",
    "src/analysis/detection/saturation.rs",
    "src/analysis/detection/sentinel_gating.rs",
    "src/analysis/detection/squash_weight_rescale.rs",
    "src/analysis/detection/topology_diversification.rs",
    "src/analysis/detection/unbounded_capping.rs",
    "src/analysis/detection/weight_magnitude_reset.rs",
    "src/analysis/detection/weight_polarity_flip.rs",
];

/// The traversal cited by `topology_diversification.rs`: `has_unhealthy_intermediates`
/// walks the graph via `dfs_max_hidden_depth`, so both symbols are pinned here
/// to catch a rename before the ledger row goes stale.
const TRAVERSAL: &[&str] = &[
    "topology_diversification.rs::has_unhealthy_intermediates",
    "topology_diversification.rs::dfs_max_hidden_depth",
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

/// The text of the `per-neuron-b` section: from its
/// `<!-- section: per-neuron-b -->` marker to the next `### ` or `## `
/// heading.
fn per_neuron_b_region(doc: &str) -> &str {
    let marker = "<!-- section: per-neuron-b -->";
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
/// a backslash, unescaping `\|` to a literal `|` in the cell text.
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

fn basename(rel: &str) -> &str {
    rel.rsplit('/').next().expect("path names a file")
}

/// The `## Capacity and traversal table` rows for one swept file's basename:
/// rows whose Symbol cell starts with `` `<basename>:: ``.
fn capacity_table_rows_for(doc: &str, base: &str) -> Vec<Vec<String>> {
    table_rows(section(doc, "## Capacity and traversal table"))
        .into_iter()
        .filter(|row| {
            row.first()
                .is_some_and(|first| first.starts_with(&format!("`{base}::")))
        })
        .collect()
}

/// The production half of a source file: everything before the `#[cfg(test)]`
/// that actually opens the test module. Some `SWEPT` files carry a
/// `#[cfg(test)]` module and are trimmed accordingly; for the rest, no such
/// pair exists and the whole file is returned.
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

#[test]
fn each_swept_row_is_present_and_not_pending() {
    let doc = read(RECORD);
    let region = per_neuron_b_region(&doc);
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
            "{path} must have exactly one row in the `per-neuron-b` section, got {}",
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
            "{path} was swept by Issue #2298 or #2299, so its outcome must not read `pending`: \
             {outcome}"
        );
        assert!(
            outcome.contains(" — "),
            "{path}'s Outcome cell must be shaped `<outcome> — <reason>`, got: {outcome}"
        );
    }

    assert_eq!(
        SWEPT.len(),
        12,
        "the per-neuron-b sweep (Issues #2298 and #2299 together) must cover exactly 12 files"
    );

    // Every row that has been flipped away from `pending` must be listed in
    // SWEPT — a row flipped without being accounted for here must fail.
    for row in &rows {
        let Some(first) = row.first() else { continue };
        let path = first.trim_matches('`');
        let outcome = row.get(2).map(String::as_str).unwrap_or_default();
        if !outcome.contains("pending") {
            assert!(
                SWEPT.contains(&path),
                "{path}'s row in the `per-neuron-b` section has a non-pending outcome but is not \
                 listed in SWEPT — a row flipped without being accounted for here"
            );
        }
    }

    // With Issue #2299 sweeping the remaining six rows, the whole
    // `per-neuron-b` section must now be non-pending.
    let non_pending_rows = rows
        .iter()
        .filter(|row| {
            row.get(2)
                .is_some_and(|outcome| !outcome.contains("pending"))
        })
        .count();
    assert_eq!(
        non_pending_rows,
        SWEPT.len(),
        "the `per-neuron-b` section must have exactly {} non-pending rows (Issues #2298 and \
         #2299 together sweep the whole section), got {non_pending_rows}",
        SWEPT.len()
    );
}

#[test]
fn every_capacity_site_in_a_swept_file_is_cited_by_symbol() {
    // Issue #1799: an assertion that holds either way is not coverage. Pin
    // one concrete site first so the loop below cannot pass vacuously.
    let saturation_source = production_source("src/analysis/detection/saturation.rs");
    assert!(
        saturation_source.contains("Vec::with_capacity(candidates.len() * 2)")
            && saturation_source.contains("fn saturated_neurons_to_coordinated_candidates"),
        "precondition: src/analysis/detection/saturation.rs must still carry a \
         `Vec::with_capacity(candidates.len() * 2)` site inside \
         `saturated_neurons_to_coordinated_candidates`, otherwise this test checks nothing"
    );
    assert!(
        production_capacity_sites("src/analysis/detection/saturation.rs").contains(&(
            "saturated_neurons_to_coordinated_candidates".to_string(),
            "results".to_string()
        )),
        "precondition: production_capacity_sites must find the `results` binding inside \
         `saturated_neurons_to_coordinated_candidates` in saturation.rs"
    );
    assert!(
        !is_capacity_site("batches.push(vec![candidate]);"),
        "precondition: is_capacity_site must not flag a `vec![single_element]` push as a \
         capacity site"
    );
    assert!(
        is_capacity_site("let v = vec![0.0; n];"),
        "precondition: is_capacity_site must flag the `vec![value; count]` form as a capacity \
         site"
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
         per-neuron-b files, otherwise this test passes vacuously"
    );

    let doc = read(RECORD);

    for (file, func, binding) in &all_sites {
        let base = basename(file);
        let rows = capacity_table_rows_for(&doc, base);
        let cited = rows.iter().any(|row| {
            row.first().is_some_and(|first| {
                first.starts_with(&format!("`{base}::{func}"))
                    && first.contains(&format!("`{binding}`"))
            }) && row.get(1).is_some_and(|kind| kind == "capacity")
        });
        assert!(
            cited,
            "{file}'s capacity site in `{func}` (binding `{binding}`) must be cited by symbol in \
             a `## Capacity and traversal table` row with Kind `capacity` — an uncited site \
             could be deleted from the row without the ledger noticing"
        );
    }

    let capacity_rows_for_swept_files: usize = SWEPT
        .iter()
        .map(|file| {
            capacity_table_rows_for(&doc, basename(file))
                .into_iter()
                .filter(|row| row.get(1).is_some_and(|kind| kind == "capacity"))
                .count()
        })
        .sum();
    assert_eq!(
        capacity_rows_for_swept_files,
        all_sites.len(),
        "the number of capacity-table rows citing a swept per-neuron-b file must equal the \
         number of capacity sites actually found in those files — a stale row for deleted code \
         must fail this too"
    );
    assert_eq!(
        capacity_rows_for_swept_files, 17,
        "the per-neuron-b capacity table must hold exactly 17 rows (8 from Issue #2298 + 9 from \
         Issue #2299)"
    );
    assert_eq!(
        all_sites.len(),
        17,
        "the per-neuron-b capacity table must hold exactly 17 rows (8 from Issue #2298 + 9 from \
         Issue #2299) — the production source must also carry exactly 17 capacity sites"
    );
}

#[test]
fn every_traversal_symbol_has_a_bounded_and_cancellation_checked_row() {
    // Issue #1799: pin the symbol this test actually depends on, so a rename
    // of `dfs_max_hidden_depth` cannot let the row below go stale unnoticed.
    let topology_diversification_source =
        production_source("src/analysis/detection/topology_diversification.rs");
    assert!(
        topology_diversification_source.contains("fn dfs_max_hidden_depth"),
        "precondition: src/analysis/detection/topology_diversification.rs must still declare \
         `fn dfs_max_hidden_depth`, otherwise this test checks nothing"
    );

    let doc = read(RECORD);

    for symbol in TRAVERSAL {
        let (base, name) = symbol
            .split_once("::")
            .unwrap_or_else(|| panic!("TRAVERSAL entry `{symbol}` must be `<basename>::<fn>`"));
        let file = SWEPT
            .iter()
            .find(|f| basename(f) == base)
            .unwrap_or_else(|| panic!("TRAVERSAL entry `{symbol}` names a file not in SWEPT"));
        assert!(
            production_source(file).contains(&format!("fn {name}")),
            "the per-neuron-b sweep cites `{name}` in {file}, but no `fn {name}` is declared \
             there any more — the sweep describes code that has moved or gone"
        );

        let rows = capacity_table_rows_for(&doc, base);
        let matching: Vec<&Vec<String>> = rows
            .iter()
            .filter(|row| {
                row.first()
                    .is_some_and(|first| first.starts_with(&format!("`{base}::{name}")))
                    && row.get(1).is_some_and(|kind| kind == "traversal")
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "there must be exactly one `## Capacity and traversal table` row with Kind \
             `traversal` for `{base}::{name}`, got {}",
            matching.len()
        );
        let row = matching[0];

        let bound = row
            .get(3)
            .unwrap_or_else(|| panic!("`{symbol}` traversal row must carry a Bound cell"));
        assert!(
            !bound.is_empty(),
            "`{symbol}` traversal row's Bound cell must not be empty"
        );

        let cancellation = row.get(4).unwrap_or_else(|| {
            panic!("`{symbol}` traversal row must carry a Cancellation-checked? cell")
        });
        assert!(
            !cancellation.is_empty(),
            "`{symbol}` traversal row's Cancellation-checked? cell must not be empty"
        );
    }
}

#[test]
fn every_issue_linked_from_a_swept_row_appears_under_issues_filed() {
    let doc = read(RECORD);
    let region = per_neuron_b_region(&doc);
    let rows = table_rows(region);

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
        "precondition: the swept `per-neuron-b` rows and their capacity/traversal rows must \
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
            "issue {needle} is referenced from a swept `per-neuron-b` row, so it must also \
             appear under `## Issues filed` — an unlinked finding is invisible to the \
             finalisation sub-issue that reconciles this record"
        );
    }
}
