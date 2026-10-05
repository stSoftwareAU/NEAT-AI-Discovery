//! Chunk 11c-2 ledger contract: the `src/discovery_history.rs` row of the
//! chunk 11 record (Issue #2253).
//!
//! The scaffold (Issue #2233) cut the row as `pending`. This file pins the
//! swept state of that row:
//!
//! * the row carries `audited — <reason>` and names both findings filed
//!   against the module (#2391, #2392);
//! * the section records a disposition for every probe run, including the
//!   re-verification of #1906 and #1902;
//! * the "Re-verified remediations" region carries exactly one live row each
//!   for #1906 and #1902;
//! * the "Filesystem mutation sites" region cites the #1902 `Drop` site and
//!   states `discovery_history.rs` has no site of its own;
//! * every `file.rs::symbol` the section cites still exists, and no citation
//!   uses a line number (Issue #1942);
//! * every unit test the ledger names by name still exists in its source
//!   file.

use std::path::PathBuf;

const RECORD: &str = "docs/audits/security-sweep-chunk-11-filesystem-lifecycle.md";
const SECTION: &str = "watchdog + tracking_alloc + discovery_history";

/// `(cited site, source file, definition the symbol must still match)`.
const CITED: [(&str, &str, &str); 11] = [
    (
        "discovery_history.rs::NeuronDiscoveryHistory::try_from",
        "src/discovery_history.rs",
        "impl TryFrom<NeuronDiscoveryHistoryWire> for NeuronDiscoveryHistory",
    ),
    (
        "discovery_history.rs::NeuronDiscoveryHistory::bayesian_score",
        "src/discovery_history.rs",
        "pub fn bayesian_score(&self)",
    ),
    (
        "discovery_history.rs::NeuronDiscoveryHistory::record_attempt",
        "src/discovery_history.rs",
        "pub fn record_attempt(",
    ),
    (
        "discovery_history.rs::compute_calibration_factor",
        "src/discovery_history.rs",
        "fn compute_calibration_factor(",
    ),
    (
        "discovery_history.rs::DiscoveryHistory::prune",
        "src/discovery_history.rs",
        "pub fn prune(",
    ),
    (
        "ffi/utilities.rs::get_calibration_summary",
        "src/ffi/utilities.rs",
        "fn get_calibration_summary(",
    ),
    (
        "ffi_internal/analysis.rs::get_calibration_summary_internal",
        "src/ffi_internal/analysis.rs",
        "pub fn get_calibration_summary_internal(",
    ),
    (
        "focus/ranking/mod.rs::rank_focus_neurons_with_history",
        "src/focus/ranking/mod.rs",
        "pub fn rank_focus_neurons_with_history(",
    ),
    (
        "streaming.rs::finish_session",
        "src/streaming.rs",
        "pub fn finish_session(",
    ),
    (
        "streaming.rs::Drop::drop",
        "src/streaming.rs",
        "impl Drop for RecordingSession",
    ),
    (
        "ffi/recording.rs::finish_discovery_session",
        "src/ffi/recording.rs",
        "fn finish_discovery_session(",
    ),
];

/// `(source file, test function name)` the ledger names by name.
const NAMED_TESTS: [(&str, &str); 9] = [
    (
        "src/discovery_history.rs",
        "test_deserialize_rejects_successes_exceeding_attempts",
    ),
    (
        "src/discovery_history.rs",
        "test_bayesian_score_invalid_counts_saturates",
    ),
    (
        "src/discovery_history.rs",
        "test_deserialize_accepts_valid_counts",
    ),
    (
        "src/discovery_history.rs",
        "test_record_attempt_keeps_successes_within_attempts_at_u32_max",
    ),
    (
        "src/discovery_history.rs",
        "test_calibration_summary_stays_finite_for_extreme_finite_observations",
    ),
    (
        "src/discovery_history.rs",
        "test_calibration_factor_is_never_nan_for_finite_observations",
    ),
    (
        "src/discovery_history.rs",
        "test_deserialize_rejects_non_finite_json_numbers",
    ),
    ("src/streaming.rs", "test_failed_finish_preserves_tmp"),
    ("src/streaming.rs", "test_empty_session_finish_removes_tmp"),
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

/// This section's marked region of a finding table: from its marker to the
/// next `<!-- section:` marker.
fn marker_region(table: &str) -> &str {
    let marker = format!("<!-- section: {SECTION} -->");
    let start = table
        .find(&marker)
        .unwrap_or_else(|| panic!("the table must carry `{marker}`"))
        + marker.len();
    let end = table[start..]
        .find("<!-- section:")
        .map_or(table.len(), |offset| start + offset);
    &table[start..end]
}

fn table_rows(body: &str) -> Vec<&str> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .collect()
}

/// `(path, outcome)` for each three-cell `src/` row of an inventory table.
fn file_rows(body: &str) -> Vec<(String, String)> {
    table_rows(body)
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
        .collect()
}

#[test]
fn discovery_history_row_is_audited_and_links_both_findings() {
    let doc = read(RECORD);
    let files_swept = section(&doc, "## Files swept");
    let rows = file_rows(section(files_swept, &format!("### {SECTION}")));

    let (_, outcome) = rows
        .iter()
        .find(|(path, _)| path == "src/discovery_history.rs")
        .unwrap_or_else(|| {
            panic!("the {SECTION} section must carry a src/discovery_history.rs row")
        });
    let Some((verdict, reason)) = outcome.split_once(" — ") else {
        panic!("src/discovery_history.rs must read `audited — <reason>`, got: {outcome}");
    };
    assert_eq!(verdict.trim(), "audited", "outcome: {outcome}");
    assert!(!reason.trim().is_empty(), "must give a reason: {outcome}");
    assert!(outcome.contains("#2391"), "outcome: {outcome}");
    assert!(outcome.contains("#2392"), "outcome: {outcome}");

    for (path, outcome) in &rows {
        assert_ne!(
            outcome.trim(),
            "pending",
            "{path} must not still read `pending`"
        );
    }
}

#[test]
fn the_section_records_a_verdict_for_each_probe() {
    let doc = read(RECORD);
    let body = section(section(&doc, "## Files swept"), &format!("### {SECTION}"));
    for term in [
        "Probe dispositions (Issue #2253)",
        "`record_attempt` overflow",
        "Calibration NaN/inf",
        "`record_calibration` bound",
        "`prune` — no finding",
        "#2391",
        "#2392",
        "#1906",
        "#1902",
    ] {
        assert!(
            body.contains(term),
            "the {SECTION} section must mention `{term}`"
        );
    }
}

#[test]
fn re_verified_region_has_one_live_row_each_for_1906_and_1902() {
    let doc = read(RECORD);
    let region = marker_region(section(&doc, "## Re-verified remediations"));
    for issue in ["#1906", "#1902"] {
        let matches: Vec<&str> = table_rows(region)
            .into_iter()
            .filter(|row| {
                let cells: Vec<&str> = row.trim().trim_matches('|').split('|').collect();
                cells.first().map(|c| c.trim()) == Some(issue)
            })
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one row for {issue}, found {}: {matches:?}",
            matches.len()
        );
        let row = matches[0];
        let cells: Vec<&str> = row.trim().trim_matches('|').split('|').collect();
        let last = cells.last().unwrap().trim();
        assert!(
            last.starts_with("yes"),
            "{issue} row's last cell must start with `yes`, got: {last}"
        );
    }
}

#[test]
fn the_mutation_region_cites_the_1902_drop_site_and_discovery_history_has_none() {
    let doc = read(RECORD);
    let region = marker_region(section(&doc, "## Filesystem mutation sites"));
    let rows = table_rows(region);

    let drop_row = rows
        .iter()
        .find(|row| row.trim_start().starts_with("| `streaming.rs::Drop::drop`"))
        .unwrap_or_else(|| panic!("the mutation region must carry a streaming.rs::Drop::drop row"));
    assert!(
        drop_row.contains("fs::remove_file"),
        "the streaming.rs::Drop::drop row must mention fs::remove_file: {drop_row}"
    );

    let none_row = rows
        .iter()
        .find(|row| {
            row.contains("`src/discovery_history.rs`") && row.trim_start().starts_with("| none")
        })
        .unwrap_or_else(|| {
            panic!("the mutation region must carry a `none` row for src/discovery_history.rs")
        });
    assert!(none_row.trim_start().starts_with("| none"));
}

#[test]
fn every_cited_symbol_exists_and_no_citation_uses_a_line_number() {
    let doc = read(RECORD);
    let body = section(section(&doc, "## Files swept"), &format!("### {SECTION}"));
    let mutation_region = marker_region(section(&doc, "## Filesystem mutation sites"));
    let reverified_region = marker_region(section(&doc, "## Re-verified remediations"));
    let combined = format!("{body}\n{mutation_region}\n{reverified_region}");

    for (site, file, definition) in CITED {
        assert!(
            combined.contains(site),
            "the ledger text must cite `{site}`"
        );
        // Issue #1942: a symbol citation is only worth something while the
        // symbol still exists.
        assert!(
            read(file).contains(definition),
            "`{site}` is cited but `{definition}` is no longer in {file}"
        );
    }

    let bytes = combined.as_bytes();
    let line_cited = combined
        .match_indices(".rs:")
        .any(|(at, _)| bytes.get(at + 4).is_some_and(u8::is_ascii_digit));
    assert!(
        !line_cited,
        "the ledger text must cite by symbol, not `file.rs:<line>`"
    );
}

#[test]
fn every_unit_test_the_ledger_names_exists() {
    let doc = read(RECORD);
    let body = section(section(&doc, "## Files swept"), &format!("### {SECTION}"));
    let mutation_region = marker_region(section(&doc, "## Filesystem mutation sites"));
    let reverified_region = marker_region(section(&doc, "## Re-verified remediations"));
    let combined = format!("{body}\n{mutation_region}\n{reverified_region}");

    for (file, name) in NAMED_TESTS {
        assert!(
            combined.contains(name),
            "the ledger text must mention test `{name}`"
        );
        assert!(
            read(file).contains(&format!("fn {name}()")),
            "`{name}` is named but not defined in {file}"
        );
    }
}
