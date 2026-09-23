//! Contract tests for the chunk 8b `synapse scoring + target_analysis` sweep (Issue #2106).
//!
//! `tests/issue_2103_chunk_08b_ledger_scaffold.rs` gates the record's *shape* —
//! one row per in-scope file, one section per audit sub-issue, the two finding
//! tables and their per-section markers. This file gates the one section Issue
//! #2106 owns, and it does so against the source it claims to have swept:
//!
//! * the 10 rows this sub-issue owns are present and none still reads
//!   `pending`, each with a reason a later reader can check;
//! * every capacity site (`with_capacity` / `vec![_; n]` / `reserve`) and every
//!   float-comparison site in the production half of those files is cited in the
//!   matching finding table, so a new one cannot land unrecorded;
//! * the symbols the outcome traces still exist;
//! * the outcome records a negative result — no finding was filed.

use std::path::PathBuf;

/// The chunk 8b prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md";

/// The 10 files Issue #2106 swept, in record order.
const SCORING_FILES: [&str; 10] = [
    "src/analysis/synapse/scoring/mod.rs",
    "src/analysis/synapse/scoring/boost_functions.rs",
    "src/analysis/synapse/scoring/discounting.rs",
    "src/analysis/synapse/scoring/improvement.rs",
    "src/analysis/synapse/scoring/test_helpers.rs",
    "src/analysis/synapse/scoring/tests.rs",
    "src/analysis/synapse/target_analysis/mod.rs",
    "src/analysis/synapse/target_analysis/candidate_selection.rs",
    "src/analysis/synapse/target_analysis/evaluation.rs",
    "src/analysis/synapse/target_analysis/statistics.rs",
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// The text of a Markdown section, from its heading to the next heading of the
/// same or a higher level.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let start = doc
        .find(heading)
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

/// The region of a finding table owned by one sub-issue: the text between its
/// `<!-- section: … -->` marker and the next marker (or the end of the table).
fn marker_region<'a>(table: &'a str, owner: &str) -> &'a str {
    let marker = format!("<!-- section: {owner} -->");
    let start = table
        .find(&marker)
        .unwrap_or_else(|| panic!("the table must carry `{marker}`"))
        + marker.len();
    let rest = &table[start..];
    let end = rest.find("<!-- section:").unwrap_or(rest.len());
    &rest[..end]
}

/// `(path, line count, outcome)` rows of the per-file outcome table.
fn file_rows(body: &str) -> Vec<(String, String)> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .filter_map(|line| {
            let cells: Vec<&str> = line.trim().trim_matches('|').split('|').collect();
            if cells.len() != 3 {
                return None;
            }
            let path = cells[0].trim().trim_matches('`').to_string();
            if !path.starts_with("src/") {
                return None;
            }
            Some((path, cells[2].trim().to_string()))
        })
        .collect()
}

fn scoring_rows() -> Vec<(String, String)> {
    let doc = read(RECORD);
    file_rows(section(&doc, "### synapse scoring + target_analysis"))
}

/// The production half of a source file: everything before the first
/// `#[cfg(test)]`. Sweep verdicts are about code an untrusted input can reach,
/// and a fixture in a `#[cfg(test)] mod tests` is not that code.
fn production_source(rel: &str) -> String {
    let body = read(rel);
    match body.find("#[cfg(test)]") {
        Some(at) => body[..at].to_string(),
        None => body,
    }
}

/// `true` when the line allocates a collection whose size is an expression —
/// `with_capacity(…)`, `reserve(…)`, or the `vec![value; count]` form.
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

fn is_comparator_site(line: &str) -> bool {
    ["total_cmp", "partial_cmp", "sort_by", "max_by", "min_by", "<=", ">=", "<", ">"]
        .iter()
        .any(|needle| line.contains(needle))
}

/// File stem plus `.rs::`, the citation prefix the record uses for a symbol in
/// that file (CONTRIBUTING.md § Cite Code by Symbol, Never by Line Number).
fn citation_prefix(rel: &str) -> String {
    let file = rel.rsplit('/').next().expect("path names a file");
    format!("{file}::")
}

#[test]
fn the_synapse_scoring_section_owns_exactly_the_files_issue_2106_swept() {
    let rows = scoring_rows();
    let paths: Vec<&str> = rows.iter().map(|(path, _)| path.as_str()).collect();
    assert_eq!(
        paths, SCORING_FILES,
        "the `synapse scoring + target_analysis` section must carry one row per file Issue #2106 \
         swept, in record order — a row under the wrong section has no owner and is swept twice \
         or never"
    );
}

#[test]
fn every_synapse_scoring_row_is_swept_with_a_reason() {
    for (path, outcome) in scoring_rows() {
        assert!(
            !outcome.contains("pending"),
            "{path} was swept by Issue #2106, so its outcome must not read `pending`: {outcome}"
        );
        let reason = outcome.split_once('—').map(|(_, rest)| rest.trim());
        assert!(
            reason.is_some_and(|r| r.len() > 20),
            "{path} must state a one-line reason after its outcome, got: {outcome}"
        );
    }
}

#[test]
fn every_capacity_site_in_the_swept_files_has_a_table_row() {
    let doc = read(RECORD);
    let region = marker_region(
        section(&doc, "## Capacity-from-input sites"),
        "synapse scoring + target_analysis",
    );

    for file in SCORING_FILES {
        let has_site = production_source(file).lines().any(is_capacity_site);
        let cited = region.contains(&citation_prefix(file));
        assert_eq!(
            has_site, cited,
            "{file} allocates from an expression: {has_site}, but the capacity table cites it: \
             {cited} — every sized allocation in a swept file needs a row naming what bounds it, \
             and a row for a file with no such site describes code that is gone"
        );
    }
}

#[test]
fn every_float_comparator_in_the_swept_files_has_a_table_row() {
    let doc = read(RECORD);
    let region = marker_region(
        section(&doc, "## Float comparison sites"),
        "synapse scoring + target_analysis",
    );

    for file in SCORING_FILES {
        if production_source(file).lines().any(is_comparator_site) {
            assert!(
                region.contains(&citation_prefix(file)),
                "{file} ranks with a float comparator, so the float-comparison table must carry \
                 a row recording where the value comes from and what happens when it is NaN"
            );
        }
    }
}

/// Every symbol the `synapse scoring + target_analysis` outcome claims to have
/// traced, paired with the file that must still declare it. An outcome citing a
/// symbol that no longer exists is describing code that has moved or gone.
const TRACED_SYMBOLS: [(&str, &str); 5] = [
    (
        "src/analysis/synapse/scoring/improvement.rs",
        "fn compute_synapse_improvement_and_count",
    ),
    (
        "src/analysis/synapse/target_analysis/evaluation.rs",
        "fn process_harmful_batch_from_prepared",
    ),
    (
        "src/analysis/synapse/target_analysis/statistics.rs",
        "fn filter_and_load_sources",
    ),
    (
        "src/analysis/synapse/target_analysis/statistics.rs",
        "fn prepare_harmful_samples",
    ),
    (
        "src/analysis/synapse/target_analysis/evaluation.rs",
        "fn collect_and_process_helpful_results",
    ),
];

#[test]
fn the_synapse_scoring_outcome_cites_symbols_that_still_exist() {
    let doc = read(RECORD);
    let outcome = section(&doc, "### synapse scoring + target_analysis (Issue #2106)");

    for (file, declaration) in TRACED_SYMBOLS {
        let symbol = declaration
            .rsplit(' ')
            .next()
            .expect("declaration names a symbol");
        assert!(
            outcome.contains(symbol),
            "the `synapse scoring + target_analysis` outcome must name `{symbol}` — an outcome \
             that cites no guard, bound or dispatch site is a claim, not a sweep"
        );
        assert!(
            read(file).contains(declaration),
            "the outcome cites `{symbol}`, but `{declaration}` is no longer declared in {file} — \
             the sweep describes code that has moved or gone, so this section needs re-sweeping \
             whatever the ledger says"
        );
    }
}

#[test]
fn the_synapse_scoring_outcome_is_a_negative_result() {
    let doc = read(RECORD);
    let outcome = section(&doc, "### synapse scoring + target_analysis (Issue #2106)");
    assert!(
        outcome.contains("Negative result"),
        "the outcome must record whether a finding was filed — a missing verdict is an \
         incomplete sweep"
    );
    assert!(
        outcome.contains("no finding filed"),
        "Issue #2106 is a negative-result sweep, so the outcome must state explicitly that no \
         finding was filed"
    );
}

#[test]
fn the_synapse_scoring_negative_result_is_linked_in_issues_filed() {
    let doc = read(RECORD);
    let filed = section(&doc, "## Issues filed");
    // The section must explicitly mention "negative-result" somewhere in the
    // context of Issue #2106, even if as a bullet, to make the negative outcome
    // plain to the finalisation sub-issue.
    assert!(
        filed.contains("negative-result") && filed.contains("synapse scoring"),
        "`## Issues filed` must record the negative-result status of the `synapse scoring + \
         target_analysis` sweep"
    );
}
