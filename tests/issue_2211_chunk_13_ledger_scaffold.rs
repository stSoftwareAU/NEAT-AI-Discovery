//! Contract tests for the chunk 13 sweep record scaffold (Issue #2211).
//!
//! Chunk 13 (`src/config`) is swept by several audit sub-issues writing into
//! **one** shared record. The scaffold lays out every region up front, with
//! its marker, header row and pending line, so each sub-issue only fills its
//! own region. Without a gate, that shape fails quietly in five ways, and
//! each of them looks like coverage from the outside:
//!
//! * a file in scope never gets a row, so it is never swept and nobody notices;
//! * a region loses its `<!-- section: … -->` marker, or a marker is duplicated,
//!   so two sub-issues append findings to the same place and conflict on every
//!   PR;
//! * a settings header drifts from the ten agreed columns, so rows filled in by
//!   different sub-issues no longer line up;
//! * a sub-issue finishes but leaves rows reading `pending` while the record
//!   carries no status, so an unfinished sweep reads as a completed one;
//! * a 13a row reverts to `pending` or loses its validation or on-invalid cell,
//!   so the region reads as swept when it is not.
//!
//! These shape tests do not require every region to be populated: an empty
//! region is not a failure, and later sub-issues fill them. Region 13a is held
//! to a stricter contract once it is filled (Issue #2273): no `pending` row or
//! entry inside it, each of the twelve settings rows carries all ten cells
//! filled (including the two validation cells), and its issues and drift are
//! recorded. While any row reads `pending`, the record must say the sweep is
//! in progress. `tests/issue_2088_sweep_ledger_contract.rs` covers the
//! ledger-wide rules; this file covers only chunk 13's own shape.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The chunk 13 prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-13-config-env.md";
/// Machine-readable sweep index — must name the record above.
const INDEX: &str = "docs/audits/lib-sweep-coverage.json";

/// The baseline commit chunk 13 was cut against (Issue #2224).
const BASELINE: &str = "b85a551ed2521ed327469b20eb88aeda828357d2";
/// The tracker issue the chunk 13 record reports into.
const PARENT_ISSUE: &str = "#2096";

/// The one file split across two regions of the per-file table, by line.
const USER_FACING: &str = "src/config/user_facing.rs";
/// `user_facing.rs` at the baseline commit. Its two row counts must sum to this.
const USER_FACING_BASELINE_LINES: usize = 1396;

/// Source root the chunk covers. Every `.rs` file beneath it needs a row.
const SCOPE: &str = "src/config";

/// The `###` regions shared by `## Files swept` and `## Settings table`, in
/// record order, each paired with the id of its section marker.
const REGIONS: [(&str, &str); 4] = [
    ("### small modules", "13a"),
    ("### user_facing L1-664", "13b"),
    ("### user_facing L665-end", "13c"),
    ("### readers outside src/config", "13d"),
];

/// The ten columns every settings header row carries, in order. Rows filled in
/// by different sub-issues only line up if every header agrees.
const SETTINGS_COLUMNS: [&str; 10] = [
    "variable",
    "reader file:line",
    "type",
    "default",
    "accepted range / validation",
    "on invalid input",
    "0 / opt-out semantics",
    "can weaken a guard? (which)",
    "cached (OnceLock) or live",
    "logged how",
];

/// The twelve variables the 13a region owns, each with the `NEAT_AI_DISCOVERY_`
/// prefix.
const REGION_13A_VARIABLES: [&str; 12] = [
    "NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD",
    "NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD",
    "NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD",
    "NEAT_AI_DISCOVERY_MODULE_TIERING_HIDDEN_THRESHOLD",
    "NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES",
    "NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS",
    "NEAT_AI_DISCOVERY_BATCH_TARGET_FAILURE_LIMIT",
    "NEAT_AI_DISCOVERY_TIMING",
    "NEAT_AI_DISCOVERY_PROFILE",
    "NEAT_AI_DISCOVERY_GPU_METRICS",
    "NEAT_AI_DISCOVERY_STRICT_CANDIDATE_RECONCILIATION",
    "NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD",
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

    // Walk candidate headings until one sits at the same or a higher level.
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

/// Every `.rs` file beneath `root`, repo-relative, with forward slashes.
fn rust_files_under(root: &str) -> Vec<String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", dir.display()));
        for entry in entries {
            let path = entry.expect("directory entry must be readable").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    let base = repo_root();
    let mut found = Vec::new();
    walk(&base.join(root), &mut found);
    found
        .iter()
        .map(|p| {
            p.strip_prefix(&base)
                .expect("walked path sits under the repo root")
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect()
}

/// A parsed row of the per-file outcome table.
#[derive(Debug)]
struct FileRow {
    path: String,
    lines: usize,
    outcome: String,
}

/// Parse the three-cell path / line-count / outcome rows out of a chunk of
/// Markdown, skipping header, separator and non-`src/` rows.
fn file_rows(body: &str) -> Vec<FileRow> {
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
            let lines = cells[1]
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("row for {path} must carry a numeric line count: {e}"));
            Some(FileRow {
                path,
                lines,
                outcome: cells[2].trim().to_string(),
            })
        })
        .collect()
}

/// The cells of every settings header row (first cell `variable`) in `body`.
fn settings_header_rows(body: &str) -> Vec<Vec<String>> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .map(|line| {
            line.trim()
                .trim_matches('|')
                .split('|')
                .map(|cell| cell.trim().to_string())
                .collect::<Vec<String>>()
        })
        .filter(|cells| cells.first().is_some_and(|c| c.as_str() == "variable"))
        .collect()
}

/// The data rows of a settings table region: every `|` row except the header
/// (first cell `variable`) and the `---` separator row.
fn settings_data_rows(body: &str) -> Vec<Vec<String>> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .map(|line| {
            let trimmed = line.trim();
            let inner = trimmed.strip_prefix('|').unwrap_or(trimmed);
            let inner = inner.strip_suffix('|').unwrap_or(inner);
            inner
                .split('|')
                .map(|cell| cell.trim().to_string())
                .collect::<Vec<String>>()
        })
        .filter(|cells| !is_header_or_separator(cells))
        .collect()
}

/// True for the header row (first cell `variable`) and for the separator row,
/// whose every cell is only `-` and optional `:`.
fn is_header_or_separator(cells: &[String]) -> bool {
    let is_header = cells.first().is_some_and(|c| c.as_str() == "variable");
    let is_separator = cells
        .iter()
        .all(|c| c.contains('-') && c.chars().all(|ch| ch == '-' || ch == ':'));
    is_header || is_separator
}

fn files_swept_body() -> String {
    section(&read(RECORD), "## Files swept").to_string()
}

/// The HTML comment that opens one region of a table or findings list.
fn marker_for(id: &str) -> String {
    format!("<!-- section: {id} -->")
}

fn is_iso_date(value: &str) -> bool {
    value.len() == 10
        && value.chars().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                c == '-'
            } else {
                c.is_ascii_digit()
            }
        })
}

/// The text of one region of a shared record: from its `<!-- section: id -->`
/// marker to the earliest of the next marker, a line starting `**` or a heading.
/// Linear in the length of `body`.
fn region_text<'a>(body: &'a str, id: &str) -> &'a str {
    let marker = marker_for(id);
    let Some(at) = body.find(&marker) else {
        panic!("the section must carry `{marker}` for region {id}");
    };
    let rest = &body[at + marker.len()..];
    let end = [
        rest.find("<!-- section:"),
        rest.find("\n**"),
        rest.find("\n#"),
    ]
    .into_iter()
    .flatten()
    .min()
    .unwrap_or(rest.len());
    &rest[..end]
}

/// True when `text` carries `pending` as a whole word, so `env_setup_pending`
/// and `spending` do not count. Linear in the length of `text`.
fn has_standalone_pending(text: &str) -> bool {
    text.to_ascii_lowercase()
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|token| token == "pending")
}

#[test]
fn the_record_exists_and_pins_its_chunk_id_exposure_baseline_and_parent() {
    let doc = read(RECORD);
    for needle in ["`13`", "**Exposure:** `local`", BASELINE, PARENT_ISSUE] {
        assert!(
            doc.contains(needle),
            "{RECORD} must carry `{needle}` — a record that does not pin its chunk id, \
             exposure, baseline commit and tracker issue cannot be checked by a later reader"
        );
    }

    let Some((date, _)) = doc
        .split_once("**Sweep date:** `")
        .and_then(|(_, rest)| rest.split_once('`'))
    else {
        panic!("{RECORD} must carry a `**Sweep date:** \\`…\\`` field");
    };
    assert!(
        is_iso_date(date),
        "{RECORD} sweep date `{date}` must be an ISO YYYY-MM-DD date"
    );
}

#[test]
fn every_config_file_has_one_row_and_user_facing_is_split_across_two_regions() {
    let body = files_swept_body();
    let rows = file_rows(&body);

    let on_disk: BTreeSet<String> = rust_files_under(SCOPE).into_iter().collect();
    let small_expected: BTreeSet<String> = on_disk
        .iter()
        .filter(|path| path.as_str() != USER_FACING)
        .cloned()
        .collect();

    // user_facing.rs legitimately appears twice (one row per line range), so the
    // duplicate check covers every other file.
    let mut tabled: BTreeSet<String> = BTreeSet::new();
    for row in rows.iter().filter(|row| row.path != USER_FACING) {
        assert!(
            tabled.insert(row.path.clone()),
            "{} appears twice in the per-file tables — two regions would record \
             conflicting outcomes for one file",
            row.path
        );
    }

    let missing: Vec<&String> = small_expected.difference(&tabled).collect();
    assert!(
        missing.is_empty(),
        "these files in {SCOPE} have no row in {RECORD}, so they would never be swept: {missing:?}"
    );
    let stale: Vec<&String> = tabled
        .iter()
        .filter(|path| path.starts_with("src/config/") && !on_disk.contains(path.as_str()))
        .collect();
    assert!(
        stale.is_empty(),
        "{RECORD} carries rows for {SCOPE} files that no longer exist: {stale:?}"
    );

    // Every file except user_facing.rs sits under `### small modules`, and only
    // there, so each has exactly one owning region.
    let small_rows = file_rows(section(&body, "### small modules"));
    let mut small_tabled: BTreeSet<String> = BTreeSet::new();
    for row in &small_rows {
        assert!(
            small_tabled.insert(row.path.clone()),
            "{} appears twice under `### small modules`",
            row.path
        );
    }
    assert_eq!(
        small_tabled, small_expected,
        "`### small modules` must carry exactly one row per {SCOPE} file other than \
         {USER_FACING} — a row anywhere else has no owning region"
    );

    // user_facing.rs is split by line: one row in each range, and no third.
    let user_facing_total = rows.iter().filter(|row| row.path == USER_FACING).count();
    assert_eq!(
        user_facing_total, 2,
        "{USER_FACING} must have exactly two rows in {RECORD}, one per line range"
    );
    let low = file_rows(section(&body, "### user_facing L1-664"));
    assert_eq!(
        low.len(),
        1,
        "`### user_facing L1-664` must carry exactly one row"
    );
    assert_eq!(
        low[0].path, USER_FACING,
        "`### user_facing L1-664` must carry {USER_FACING}"
    );
    let high = file_rows(section(&body, "### user_facing L665-end"));
    assert_eq!(
        high.len(),
        1,
        "`### user_facing L665-end` must carry exactly one row"
    );
    assert_eq!(
        high[0].path, USER_FACING,
        "`### user_facing L665-end` must carry {USER_FACING}"
    );
    assert_eq!(
        low[0].lines + high[0].lines,
        USER_FACING_BASELINE_LINES,
        "the two {USER_FACING} rows must sum to its baseline line count \
         {USER_FACING_BASELINE_LINES}, or a line range has been dropped or double-counted"
    );

    for row in &rows {
        assert!(
            row.lines > 0,
            "{} must carry its line count at the baseline commit",
            row.path
        );
    }
}

#[test]
fn each_section_marker_appears_exactly_once_in_its_section() {
    let doc = read(RECORD);

    // Each `##` section, the marker ids it carries, and the number of times each
    // marker must appear inside that section.
    let plan: [(&str, &[&str]); 4] = [
        ("## Files swept", &["13a", "13b", "13c", "13d"]),
        ("## Settings table", &["13a", "13b", "13c", "13d"]),
        (
            "## Issues filed",
            &["13a-findings", "13a", "13b", "13c", "13d"],
        ),
        ("## Doc drift observed", &["13a", "13b", "13c", "13d"]),
    ];

    for (heading, ids) in plan {
        let body = section(&doc, heading);
        for id in ids {
            let marker = marker_for(id);
            let count = body.matches(&marker).count();
            assert_eq!(
                count, 1,
                "`{heading}` must carry `{marker}` exactly once so its region is \
                 unambiguous to the sub-issue that owns it, found {count}"
            );
        }
    }
}

#[test]
fn the_settings_table_has_one_ten_column_header_per_region() {
    let doc = read(RECORD);
    let settings = section(&doc, "## Settings table");

    let all_headers = settings_header_rows(settings);
    assert_eq!(
        all_headers.len(),
        REGIONS.len(),
        "`## Settings table` must carry exactly one header row per region, found {}",
        all_headers.len()
    );

    for (heading, id) in REGIONS {
        let headers = settings_header_rows(section(settings, heading));
        assert_eq!(
            headers.len(),
            1,
            "`{heading}` in `## Settings table` (region {id}) must carry exactly one header row"
        );
        let columns: Vec<&str> = headers[0].iter().map(String::as_str).collect();
        assert_eq!(
            columns, SETTINGS_COLUMNS,
            "`{heading}` in `## Settings table` must carry the ten agreed columns in order — \
             rows filled in by different sub-issues only line up if every header agrees"
        );
    }
}

#[test]
fn the_index_names_the_chunk_13_record_and_its_baseline() {
    let index = read(INDEX);
    assert!(
        index.contains(RECORD),
        "{INDEX} must name {RECORD} — a prose record with no index entry is invisible \
         to the next automated sweep"
    );
    assert!(
        index.contains(BASELINE),
        "{INDEX} must pin chunk 13 to baseline {BASELINE}"
    );
}

#[test]
fn an_unfinished_chunk_says_so_above_its_table() {
    let doc = read(RECORD);
    let pending = file_rows(&files_swept_body())
        .iter()
        .filter(|row| row.outcome.contains("pending"))
        .count();
    if pending == 0 {
        return;
    }
    // The index cannot express "partly swept": the ledger contract rejects a
    // record with a null `last_swept`. So while rows are outstanding, the
    // record itself must say so — otherwise the index's date reads as a
    // completed sweep and nothing contradicts it.
    assert!(
        doc.contains("### Sweep status — IN PROGRESS"),
        "{pending} rows still read `pending`, so {RECORD} must carry a \
         `### Sweep status — IN PROGRESS` heading — the index's `last_swept` date would \
         otherwise be the only signal, and it reads as a finished sweep"
    );
}

#[test]
fn region_13a_files_swept_rows_are_decided() {
    const SMALL_MODULES: [&str; 5] = [
        "src/config/helpers.rs",
        "src/config/detection.rs",
        "src/config/drought_mitigation.rs",
        "src/config/observability.rs",
        "src/config/mod.rs",
    ];
    let body = files_swept_body();
    let rows = file_rows(section(&body, "### small modules"));

    let paths: BTreeSet<&str> = rows.iter().map(|row| row.path.as_str()).collect();
    let expected: BTreeSet<&str> = SMALL_MODULES.into_iter().collect();
    assert_eq!(
        rows.len(),
        SMALL_MODULES.len(),
        "`### small modules` in {RECORD} must carry exactly {} rows, found {}",
        SMALL_MODULES.len(),
        rows.len()
    );
    assert_eq!(
        paths, expected,
        "`### small modules` in {RECORD} must carry the five 13a files"
    );

    for row in &rows {
        assert!(
            !has_standalone_pending(&row.outcome),
            "{} still reads `pending` in the 13a region of {RECORD}: a pending row means \
             the 13a sweep was reverted, so the file reads as swept when it is not",
            row.path
        );
        let Some((outcome, reason)) = row.outcome.split_once(" — ") else {
            panic!(
                "{} must record its outcome as `<outcome> — <reason>`, found `{}`",
                row.path, row.outcome
            );
        };
        assert!(
            !outcome.trim().is_empty() && outcome.trim() == outcome,
            "{} must carry a non-empty, trimmed outcome before ` — `",
            row.path
        );
        assert!(
            !reason.trim().is_empty() && reason.trim() == reason,
            "{} must carry a non-empty, trimmed reason after ` — `",
            row.path
        );
    }
}

#[test]
fn region_13a_settings_rows_cover_the_twelve_variables_with_filled_cells() {
    let doc = read(RECORD);
    let region = section(section(&doc, "## Settings table"), "### small modules");
    let rows = settings_data_rows(region);
    assert_eq!(
        rows.len(),
        REGION_13A_VARIABLES.len(),
        "the 13a region of `## Settings table` in {RECORD} must carry one row per \
         variable, found {}",
        rows.len()
    );

    let mut seen: BTreeSet<String> = BTreeSet::new();
    for row in &rows {
        let variable = row[0].trim_matches('`').to_string();
        assert_eq!(
            row.len(),
            SETTINGS_COLUMNS.len(),
            "`{variable}` in the 13a settings region must carry {} cells, found {}",
            SETTINGS_COLUMNS.len(),
            row.len()
        );
        for (column, cell) in SETTINGS_COLUMNS.iter().zip(row) {
            assert!(
                !cell.is_empty(),
                "`{variable}` leaves its `{column}` cell empty in 13a of {RECORD}"
            );
            assert!(
                !has_standalone_pending(cell),
                "`{variable}` still reads `pending` in its `{column}` cell in 13a of \
                 {RECORD}: a pending row means the 13a sweep was reverted"
            );
        }
        for index in [4, 5] {
            assert!(
                !row[index].is_empty(),
                "`{variable}` must fill its `{}` cell in 13a: the validation cells say what \
                 the reader accepts and what it does on invalid input",
                SETTINGS_COLUMNS[index]
            );
        }
        assert!(
            seen.insert(variable.clone()),
            "`{variable}` appears twice in the 13a settings region of {RECORD}"
        );
    }

    let expected: BTreeSet<String> = REGION_13A_VARIABLES
        .into_iter()
        .map(str::to_string)
        .collect();
    assert_eq!(
        seen, expected,
        "the 13a settings region of {RECORD} must cover exactly the twelve 13a variables"
    );
}

#[test]
fn region_13a_issues_filed_and_doc_drift_are_recorded_without_pending() {
    let doc = read(RECORD);

    let issues = region_text(section(&doc, "## Issues filed"), "13a");
    assert!(
        !has_standalone_pending(issues),
        "the 13a region of `## Issues filed` in {RECORD} still reads `pending`: a pending \
         entry means the 13a sweep was reverted"
    );
    for needle in ["#2438", "#2213", "#2096"] {
        assert!(
            issues.contains(needle),
            "the 13a region of `## Issues filed` in {RECORD} must record {needle}"
        );
    }

    let drift = region_text(section(&doc, "## Doc drift observed"), "13a");
    assert!(
        !has_standalone_pending(drift),
        "the 13a region of `## Doc drift observed` in {RECORD} still reads `pending`: a \
         pending entry means the 13a sweep was reverted"
    );
    assert!(
        drift.contains("MODULE_TIERING_HIDDEN_THRESHOLD"),
        "the 13a region of `## Doc drift observed` in {RECORD} must record the \
         `MODULE_TIERING_HIDDEN_THRESHOLD` drift"
    );
    assert!(
        drift.contains("Re-checked against the parent's list (#2121)"),
        "the 13a region of `## Doc drift observed` in {RECORD} must record the re-check \
         against the parent's list of missing `mod.rs` variables"
    );
}

#[test]
fn region_13a_probe_dispositions_record_probe_a_and_probe_b() {
    let doc = read(RECORD);
    let probes = section(&doc, "## Probe dispositions (13a)");
    for needle in [
        "#2438",
        "log_effective_drought_mitigation_config",
        "info!",
        "RUST_LOG",
    ] {
        assert!(
            probes.contains(needle),
            "`## Probe dispositions (13a)` in {RECORD} must record `{needle}` for probes (a) and (b)"
        );
    }
}

#[test]
fn pending_detection_matches_standalone_word_only() {
    for text in ["pending — x", "Pending.", "(pending)"] {
        assert!(
            has_standalone_pending(text),
            "`{text}` must count as a pending row"
        );
    }
    for text in ["env_setup_pending", "spending", "pendingx", ""] {
        assert!(
            !has_standalone_pending(text),
            "`{text}` must not count as a pending row"
        );
    }
    // Hostile inputs must scan in linear time and report no pending word.
    assert!(!has_standalone_pending(&"a".repeat(200_000)));
    assert!(!has_standalone_pending(&"_".repeat(200_000)));
    assert!(!has_standalone_pending(&" ".repeat(200_000)));
}

#[test]
fn region_text_stops_at_the_next_marker_bold_heading_or_heading() {
    let by_marker = "<!-- section: 13a -->\nalpha\n<!-- section: 13b -->\nbeta";
    assert_eq!(region_text(by_marker, "13a").trim(), "alpha");

    let by_bold = "<!-- section: 13a -->\nalpha\n\n**next**\nbeta";
    assert_eq!(region_text(by_bold, "13a").trim(), "alpha");

    let by_heading = "<!-- section: 13a -->\nalpha\n# next\nbeta";
    assert_eq!(region_text(by_heading, "13a").trim(), "alpha");

    // The `13a-findings` marker is a different region and must not be selected.
    let shared = "<!-- section: 13a-findings -->\n- pending\n\n**Small modules (13a)**\n\n\
                  <!-- section: 13a -->\n- #2438 ok\n\n**next**";
    let region = region_text(shared, "13a");
    assert!(
        region.contains("#2438"),
        "region 13a must carry its own `#2438` entry, got `{region}`"
    );
    assert!(
        !region.contains("pending"),
        "region 13a must not pick up the `13a-findings` entries, got `{region}`"
    );
}
