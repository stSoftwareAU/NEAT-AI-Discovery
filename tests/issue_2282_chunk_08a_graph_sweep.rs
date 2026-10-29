//! Contract tests for the whole `graph` section of the chunk 8a sweep record
//! (Issues #2282 and #2283, part of #2217).
//!
//! `SWEPT` holds all eight files the `graph` section sweeps: `topology.rs`,
//! `skip_connection.rs`, `dead_neuron.rs`, `compound_degradation.rs`,
//! `redundant_path.rs`, `bottleneck.rs`, `low_impact_neuron.rs` and
//! `cross_detection_synthesis.rs`. Finalisation (#2302, part of #2154)
//! promoted the record to the top level and retargeted `RECORD` below.

use std::path::PathBuf;

/// The chunk 8a prose record, promoted to the top level by finalisation
/// (#2302, part of #2154).
const RECORD: &str = "docs/audits/security-sweep-chunk-08a-detection-neuron.md";

/// The eight files the `graph` section sweeps.
const SWEPT: &[&str] = &[
    "src/analysis/detection/topology.rs",
    "src/analysis/detection/skip_connection.rs",
    "src/analysis/detection/dead_neuron.rs",
    "src/analysis/detection/compound_degradation.rs",
    "src/analysis/detection/redundant_path.rs",
    "src/analysis/detection/bottleneck.rs",
    "src/analysis/detection/low_impact_neuron.rs",
    "src/analysis/detection/cross_detection_synthesis.rs",
];

/// One graph traversal or pair scan cited per file that has one, as
/// `<basename>::<fn>` — not every swept file carries a traversal.
const TRAVERSAL: &[&str] = &[
    "topology.rs::compute_shortest_paths_to_output",
    "skip_connection.rs::compute_depths_from_inputs",
    "dead_neuron.rs::find_connected_outputs_cached",
    "compound_degradation.rs::reachable_outputs",
    "redundant_path.rs::detect_redundant_paths",
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

/// The text of the `graph` section: from its `<!-- section: graph -->`
/// marker to the next `### ` or `## ` heading.
fn graph_region(doc: &str) -> &str {
    let marker = "<!-- section: graph -->";
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
/// that actually opens the test module. Some `SWEPT` files (`redundant_path.rs`
/// and `cross_detection_synthesis.rs`) carry a `#[cfg(test)]` module and are
/// trimmed accordingly; for the rest, no such pair exists and the whole file
/// is returned.
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
    let region = graph_region(&doc);
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
            "{path} must have exactly one row in the `graph` section, got {}",
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
            "{path} was swept by Issue #2282 or #2283, so its outcome must not read `pending`: \
             {outcome}"
        );
    }
}

#[test]
fn graph_section_holds_exactly_the_eight_swept_rows() {
    let doc = read(RECORD);
    let rows = table_rows(graph_region(&doc));

    let found: std::collections::BTreeSet<String> = rows
        .iter()
        .filter_map(|row| row.first())
        .map(|first| first.trim_matches('`').to_string())
        .collect();
    let expected: std::collections::BTreeSet<String> =
        SWEPT.iter().map(ToString::to_string).collect();

    assert_eq!(
        found.len(),
        SWEPT.len(),
        "the `graph` section must hold exactly {} rows (one per SWEPT file), found {}: {found:?}",
        SWEPT.len(),
        found.len()
    );

    let missing: Vec<&String> = expected.difference(&found).collect();
    let unexpected: Vec<&String> = found.difference(&expected).collect();
    assert!(
        missing.is_empty() && unexpected.is_empty(),
        "the `graph` section's rows must be exactly the SWEPT set — missing: {missing:?}, \
         unexpected: {unexpected:?}"
    );

    for row in &rows {
        let path = row.first().map_or("", |f| f.trim_matches('`'));
        let outcome = row.get(2).map_or("", String::as_str);
        assert!(
            !outcome.contains("pending"),
            "{path}'s Outcome cell must not read `pending`: {outcome}"
        );
    }
}

#[test]
fn every_capacity_site_in_a_swept_file_is_cited_by_symbol() {
    // Issue #1799: an assertion that holds either way is not coverage. Pin
    // one concrete site first so the loop below cannot pass vacuously.
    let dead_neuron_source = production_source("src/analysis/detection/dead_neuron.rs");
    assert!(
        dead_neuron_source.contains("HashSet::with_capacity(neuron_count)")
            && dead_neuron_source.contains("fn find_connected_outputs_cached"),
        "precondition: src/analysis/detection/dead_neuron.rs must still carry a \
         `with_capacity(` site inside `find_connected_outputs_cached`, otherwise this test \
         checks nothing"
    );
    let bottleneck_source = production_source("src/analysis/detection/bottleneck.rs");
    assert!(
        bottleneck_source.contains("Vec::with_capacity(candidates.len() * 2)")
            && bottleneck_source.contains("fn bottleneck_neurons_to_coordinated_candidates"),
        "precondition: src/analysis/detection/bottleneck.rs must still carry a \
         `Vec::with_capacity(candidates.len() * 2)` site inside \
         `bottleneck_neurons_to_coordinated_candidates`, otherwise this test checks nothing"
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
         graph files, otherwise this test passes vacuously"
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
        "the number of capacity-table rows citing a swept graph file must equal the number of \
         capacity sites actually found in those files — a stale row for deleted code must fail \
         this too"
    );
}

#[test]
fn every_traversal_symbol_has_a_bounded_and_cancellation_checked_row() {
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
            "the `graph` sweep cites `{name}` in {file}, but no `fn {name}` is declared there \
             any more — the sweep describes code that has moved or gone"
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
    let region = graph_region(&doc);
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
        "precondition: the swept `graph` rows and their capacity/traversal rows must reference \
         at least one `#N` issue, otherwise this test passes vacuously"
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
            "issue {needle} is referenced from a swept `graph` row, so it must also appear \
             under `## Issues filed` — an unlinked finding is invisible to the finalisation \
             sub-issue that reconciles this record"
        );
    }
}
