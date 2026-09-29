//! Contract tests for the chunk 9c device sweep (Issue #2113), first slice
//! (Issue #2240).
//!
//! `tests/issue_2288_chunk_09_ledger_scaffold.rs` gates the chunk 9 record's
//! *shape*. This file gates what the device slice writes into its own marked
//! regions, and pins that record against the source it describes:
//!
//! * the `## Ledger` device region carries a `SEC-fe0b268a3799` row whose
//!   status is a verdict — `remediated`, `still live` or `refuted`;
//! * the disposition cites all eight `setup_gpu_environment` sites;
//! * the refuted region carries the CPU-fallback cross-check row, citing
//!   `no_gpu_result`, `gpu_unavailable` and `classify_gpu_unavailable_reason`;
//! * every symbol the write-path and cross-check tables cite still exists in
//!   the file they cite, so the record fails loudly when the code moves;
//! * every finding the device region files is open and linked to its issue.
//!
//! The second slice (Issue #2241) adds the `GpuAnalyzer` `None → Err` record:
//!
//! * the entry-point table carries all eleven entry points, in order, each
//!   citing its device, queue, layout, pipeline and (where it has one) reduce
//!   check as `path.rs:line` plus the `.context("…")` string, and every string
//!   still guards that entry point's body in that order;
//! * every delegation row still delegates to the entry point it names;
//! * every empty-input `Ok` short-circuit carries a `benign` verdict;
//! * the device audit region states "Outcome (#2241): no finding".
//!
//! Only the device check is reachable without a GPU, so this string check is
//! what pins the queue, layout, pipeline and reduce checks; the device check is
//! also exercised by `src/analysis/gpu/none_field_tests.rs`.
//!
//! Line numbers are baseline-relative, so only their range is checked here;
//! the runtime guard itself stays pinned by
//! `test_may_mutate_environment_requires_single_thread` and
//! `tests/gpu/issue_1873_gpu_env_setup_thread_guard.rs`.

use std::path::PathBuf;

/// The chunk 9 prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-9-gpu-wgsl.md";

/// The re-verified finding this slice disposes of.
const DISPOSED: &str = "SEC-fe0b268a3799";

/// The verdicts the disposition may carry.
const VERDICTS: [&str; 3] = ["remediated", "still live", "refuted"];

/// Every `setup_gpu_environment` site at the baseline: imports, calls,
/// re-exports and the module doc line.
const CALL_SITES: [&str; 8] = [
    "src/analysis/gpu/analyzer.rs:23",
    "src/analysis/gpu/analyzer.rs:270",
    "src/analysis/gpu/analyzer.rs:352",
    "src/analysis/gpu/device.rs:31",
    "src/analysis/gpu/device.rs:442",
    "src/analysis/system.rs:46",
    "src/analysis/system.rs:88",
    "src/analysis/utils/mod.rs:60",
];

/// Symbols the write path and the cross-check must cite, with their home file.
const REQUIRED_SYMBOLS: [(&str, &str); 7] = [
    ("set_env_if_unset", "src/analysis/utils/platform.rs"),
    ("may_mutate_environment", "src/analysis/utils/platform.rs"),
    ("setup_gpu_environment", "src/analysis/utils/platform.rs"),
    ("no_gpu_result", "src/analysis/gpu/device.rs"),
    ("gpu_unavailable", "src/analysis/analysis_outcome.rs"),
    (
        "is_environmentally_disabled",
        "src/analysis/analysis_outcome.rs",
    ),
    ("classify_gpu_unavailable_reason", "src/ffi_internal/gpu.rs"),
];

/// Symbols the CPU-fallback cross-check row must name.
const CROSS_CHECK_SYMBOLS: [&str; 3] = [
    "no_gpu_result",
    "gpu_unavailable",
    "classify_gpu_unavailable_reason",
];

/// The findings the device slice filed; the record states "one finding, #2318".
const SURVIVING_FINDINGS: [&str; 1] = ["SEC-2b0c59cc73d5"];

const WRITE_PATH_TABLE: &str = "#### SEC-fe0b268a3799 — write path (Issue #2240)";
const CALL_SITE_TABLE: &str = "#### SEC-fe0b268a3799 — `setup_gpu_environment` sites (Issue #2240)";
const CROSS_CHECK_TABLE: &str = "#### CPU-fallback cross-check (Issue #2240)";
const ENTRY_POINT_TABLE: &str = "#### Entry-point None → Err sites (Issue #2241)";
const DELEGATION_TABLE: &str = "#### Entry-point delegations (Issue #2241)";
const SHORT_CIRCUIT_TABLE: &str = "#### Empty-input Ok short-circuits (Issue #2241)";

/// Every public `GpuAnalyzer` evaluation entry point, in record order.
const ENTRY_POINTS: [&str; 11] = [
    "evaluate_relu_gpu",
    "evaluate_relu_gpu_with_budget",
    "evaluate_activation_gpu",
    "evaluate_activation_gpu_with_budget",
    "evaluate_activations_batched_gpu",
    "evaluate_activations_batched_gpu_with_budget",
    "evaluate_bias_gpu",
    "evaluate_harmful_batch",
    "evaluate_harmful_batch_with_budget",
    "evaluate_helpful_batch",
    "evaluate_helpful_batch_with_budget",
];

/// The phrase each check column's `.context("…")` string must carry.
const CHECK_PHRASES: [&str; 4] = [
    "GPU device unavailable",
    "GPU queue not initialised",
    "layout",
    "pipeline",
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
    let mut offset = 0;
    let mut start = None;
    for line in doc.split_inclusive('\n') {
        if line.trim_end() == heading {
            start = Some(offset + line.len());
            break;
        }
        offset += line.len();
    }
    let start = start.unwrap_or_else(|| panic!("{RECORD} must carry the heading `{heading}`"));
    let level = heading.chars().take_while(|c| *c == '#').count();
    let mut end = doc.len();
    let mut cursor = start;
    for line in doc[start..].split_inclusive('\n') {
        let depth = line.chars().take_while(|c| *c == '#').count();
        if depth > 0 && depth <= level {
            end = cursor;
            break;
        }
        cursor += line.len();
    }
    &doc[start..end]
}

/// The device slice's region: between `<!-- section: device -->` and
/// `<!-- section: queue-core -->`.
fn device_region(body: &str) -> &str {
    let open = "<!-- section: device -->";
    let close = "<!-- section: queue-core -->";
    let start = body
        .find(open)
        .unwrap_or_else(|| panic!("{RECORD} must carry `{open}`"))
        + open.len();
    let end = start
        + body[start..]
            .find(close)
            .unwrap_or_else(|| panic!("{RECORD} must carry `{close}` after `{open}`"));
    &body[start..end]
}

/// Data rows of every Markdown table in `body`; header and separator skipped.
fn table_rows(body: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut in_table = false;
    for line in body.lines() {
        let line = line.trim();
        if !line.starts_with('|') {
            in_table = false;
            continue;
        }
        let cells: Vec<String> = line
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim().to_string())
            .collect();
        if !in_table {
            in_table = true;
            continue;
        }
        if cells
            .iter()
            .all(|c| c.chars().all(|ch| ch == '-' || ch == ':'))
        {
            continue;
        }
        rows.push(cells);
    }
    rows
}

/// Region rows with no header of their own, e.g. every `| SEC-` ledger row.
fn prefixed_rows(body: &str, prefix: &str) -> Vec<Vec<String>> {
    body.lines()
        .map(str::trim)
        .filter(|line| line.starts_with(prefix))
        .map(|line| {
            line.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect()
        })
        .collect()
}

/// The first `` `path.rs:line` `` citation in a cell, split into its parts.
fn first_citation(cell: &str) -> Option<(String, usize)> {
    citations(cell).into_iter().next()
}

/// Every `` `path.rs:line` `` citation in a cell, in order.
fn citations(cell: &str) -> Vec<(String, usize)> {
    cell.split('`')
        .skip(1)
        .step_by(2)
        .filter_map(|token| {
            let (path, line) = token.rsplit_once(':')?;
            if !path.ends_with(".rs") {
                return None;
            }
            Some((path.to_string(), line.parse().ok()?))
        })
        .collect()
}

/// Every `"…"` string in a cell, in order.
fn quoted(cell: &str) -> Vec<String> {
    cell.split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect()
}

/// `(path, line, string)` for every citation/string pair in a check cell.
fn checks(entry: &str, cell: &str) -> Vec<(String, usize, String)> {
    let cites = citations(cell);
    let strings = quoted(cell);
    assert_eq!(
        cites.len(),
        strings.len(),
        "{entry}: every cited check must quote its `.context` string: {cell}"
    );
    cites
        .into_iter()
        .zip(strings)
        .map(|((path, line), text)| (path, line, text))
        .collect()
}

/// The body of each `fn name(` in `source`, up to the next item at the same
/// indent or the end of the `impl` block.
fn fn_bodies<'a>(source: &'a str, name: &str) -> Vec<&'a str> {
    let needle = format!("fn {name}(");
    source
        .match_indices(&needle)
        .map(|(start, _)| {
            let rest = &source[start + needle.len()..];
            let end = [
                "\n    pub fn ",
                "\n    fn ",
                "\n    pub(super) fn ",
                "\n    pub(crate) fn ",
                "\n}",
            ]
            .iter()
            .filter_map(|t| rest.find(t))
            .min()
            .unwrap_or(rest.len());
            &rest[..end]
        })
        .collect()
}

/// The device audit region's rows under one `####` table heading.
fn audit_table(heading: &str) -> Vec<Vec<String>> {
    let doc = read(RECORD);
    let region = device_region(section(&doc, "## Audit sections")).to_string();
    table_rows(section(&region, heading))
}

fn assert_in_range(label: &str, path: &str, line: usize) -> String {
    let source = read(path);
    assert!(
        line >= 1 && line <= source.lines().count(),
        "{label} cites {path}:{line}, past the end of the file"
    );
    source
}

/// Every `#N` issue reference in a cell.
fn issue_refs(cell: &str) -> Vec<u32> {
    cell.split('#')
        .skip(1)
        .filter_map(|rest| {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            digits.parse().ok()
        })
        .collect()
}

/// Whether `source` still defines `symbol`: a `fn` for a function, the bare
/// name for a type or variant.
fn defines(source: &str, symbol: &str) -> bool {
    if symbol.starts_with(|c: char| c.is_ascii_uppercase()) {
        source.contains(symbol)
    } else {
        source.contains(&format!("fn {symbol}("))
    }
}

/// `(symbol, cited path, cited line)` for every row of a `Symbol | Cited at`
/// table in the device audit region.
fn symbol_rows(heading: &str) -> Vec<(String, String, usize)> {
    audit_table(heading)
        .into_iter()
        .map(|row| {
            let symbol = row[0]
                .trim_matches('`')
                .rsplit("::")
                .next()
                .unwrap_or_default();
            let (path, line) = first_citation(&row[1])
                .unwrap_or_else(|| panic!("{heading}: `{}` must cite `path.rs:line`", row[0]));
            (symbol.to_string(), path, line)
        })
        .collect()
}

#[test]
fn the_ledger_device_region_carries_a_verdict_for_sec_fe0b268a3799() {
    let doc = read(RECORD);
    let rows = prefixed_rows(device_region(section(&doc, "## Ledger")), "| SEC-");
    let row = rows
        .iter()
        .find(|row| row[0] == DISPOSED)
        .unwrap_or_else(|| panic!("the ledger device region must carry a {DISPOSED} row"));
    assert_eq!(
        row.len(),
        5,
        "finding-id | file:line | CWE | severity | status"
    );
    let status = &row[4];
    let verdict = VERDICTS
        .iter()
        .find(|v| status.starts_with(**v))
        .unwrap_or_else(|| {
            panic!("{DISPOSED} status must open with one of {VERDICTS:?}, got: {status}")
        });
    assert!(
        first_citation(&row[1]).is_some(),
        "{DISPOSED} must cite the write site as `path.rs:line`: {}",
        row[1]
    );
    assert!(
        row[2].starts_with("CWE-"),
        "{DISPOSED}: CWE cell: {}",
        row[2]
    );
    if *verdict == "still live" {
        assert!(
            !issue_refs(status).is_empty(),
            "a still-live {DISPOSED} must link the finding issue it filed: {status}"
        );
    }
}

#[test]
fn the_disposition_cites_every_setup_gpu_environment_site() {
    let doc = read(RECORD);
    let region = device_region(section(&doc, "## Audit sections")).to_string();
    let cited: Vec<String> = table_rows(section(&region, CALL_SITE_TABLE))
        .iter()
        .map(|row| row[0].trim_matches('`').to_string())
        .collect();
    assert_eq!(
        cited, CALL_SITES,
        "the call-site table must carry one row per `setup_gpu_environment` site, in order"
    );
    for site in CALL_SITES {
        let (path, line) = site.rsplit_once(':').expect("path:line");
        let source = read(path);
        let line: usize = line.parse().expect("line number");
        assert!(
            line <= source.lines().count(),
            "{site} is past the end of {path}"
        );
        assert!(
            source.contains("setup_gpu_environment"),
            "{path} no longer mentions `setup_gpu_environment` — the disposition is stale"
        );
    }
}

#[test]
fn every_cited_symbol_still_exists_in_its_cited_file() {
    let mut rows = symbol_rows(WRITE_PATH_TABLE);
    rows.extend(symbol_rows(CROSS_CHECK_TABLE));
    for (symbol, path, line) in &rows {
        let source = read(path);
        assert!(
            *line >= 1 && *line <= source.lines().count(),
            "`{symbol}` cites {path}:{line}, past the end of the file"
        );
        assert!(
            defines(&source, symbol),
            "`{symbol}` is no longer defined in {path} — the record describes code that moved"
        );
    }
    for (symbol, path) in REQUIRED_SYMBOLS {
        assert!(
            rows.iter().any(|(s, p, _)| s == symbol && p == path),
            "the device record must cite `{symbol}` in {path}"
        );
    }
}

#[test]
fn the_cpu_fallback_cross_check_row_names_every_hop() {
    let doc = read(RECORD);
    let refuted = device_region(section(&doc, "## Refuted / not findings"));
    let rows = prefixed_rows(refuted, "| ");
    let row = rows
        .iter()
        .find(|row| row[0].contains("CPU-fallback cross-check"))
        .expect("the refuted device region must carry the CPU-fallback cross-check row");
    assert_eq!(
        row.len(),
        3,
        "candidate | refuting file:line | why: {row:?}"
    );
    for symbol in CROSS_CHECK_SYMBOLS {
        assert!(
            row[1].contains(symbol),
            "the cross-check row must cite `{symbol}`: {}",
            row[1]
        );
    }
    assert!(
        first_citation(&row[1]).is_some(),
        "the cross-check row must cite file:line: {}",
        row[1]
    );
}

#[test]
fn every_device_finding_is_open_linked_and_named_in_the_audit_region() {
    let doc = read(RECORD);
    let audit = device_region(section(&doc, "## Audit sections"));
    let issues_filed = section(&doc, "## Issues filed");
    let rows = prefixed_rows(device_region(section(&doc, "## Ledger")), "| SEC-");
    let findings: Vec<&Vec<String>> = rows.iter().filter(|row| row[0] != DISPOSED).collect();
    let filed: Vec<&str> = findings.iter().map(|row| row[0].as_str()).collect();
    assert_eq!(
        filed, SURVIVING_FINDINGS,
        "the device ledger region must carry exactly the findings this slice filed"
    );
    for row in findings {
        let hex = row[0].strip_prefix("SEC-").expect("SEC- id");
        assert!(
            hex.len() == 12
                && hex
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "ledger id must be SEC- + 12 lowercase hex: {}",
            row[0]
        );
        let status = row.last().expect("status cell");
        let refs = issue_refs(status);
        assert!(
            status.starts_with("open") && !refs.is_empty(),
            "{} must read `open — #N`: {status}",
            row[0]
        );
        for n in refs {
            assert!(
                audit.contains(&format!("#{n}")),
                "{}: finding #{n} must be named in the device audit region",
                row[0]
            );
            assert!(
                issues_filed.contains(&format!("- #{n} — `{}`", row[0])),
                "{}: finding #{n} must be listed under `## Issues filed`",
                row[0]
            );
        }
    }
}

#[test]
fn every_entry_point_cites_its_none_to_err_checks_in_order() {
    let rows = audit_table(ENTRY_POINT_TABLE);
    let names: Vec<&str> = rows.iter().map(|row| row[0].trim_matches('`')).collect();
    assert_eq!(
        names, ENTRY_POINTS,
        "the entry-point table must carry one row per entry point, in order"
    );
    for row in &rows {
        let entry = row[0].trim_matches('`');
        assert_eq!(
            row.len(),
            7,
            "{entry}: entry point | defined at | device | queue | layout | pipeline | reduce"
        );
        let (def_path, def_line) = first_citation(&row[1])
            .unwrap_or_else(|| panic!("{entry} must cite its definition as `path.rs:line`"));
        let source = assert_in_range(entry, &def_path, def_line);
        assert!(
            defines(&source, entry),
            "`{entry}` is no longer defined in {def_path}"
        );

        let mut cited = Vec::new();
        for (column, phrase) in CHECK_PHRASES.iter().enumerate() {
            let pairs = checks(entry, &row[column + 2]);
            assert_eq!(
                pairs.len(),
                1,
                "{entry}: the {phrase} column cites exactly one check"
            );
            assert!(
                pairs[0].2.contains(phrase),
                "{entry}: the {phrase} column quotes {:?}",
                pairs[0].2
            );
            cited.extend(pairs);
        }
        let reduce = checks(entry, &row[6]);
        if !reduce.is_empty() {
            assert_eq!(reduce.len(), 2, "{entry}: reduce cites layout / pipeline");
            assert!(reduce[0].2.contains("reduce layout"), "{entry}: {reduce:?}");
            assert!(
                reduce[1].2.contains("reduce pipeline"),
                "{entry}: {reduce:?}"
            );
        } else {
            assert_eq!(
                row[6], "—",
                "{entry}: an entry point with no reduce stage reads —"
            );
        }
        cited.extend(reduce);

        // The unbudgeted wrappers delegate, so their checks live in `_with_budget`.
        let checker = if entry.ends_with("_with_budget") || entry == "evaluate_bias_gpu" {
            entry.to_string()
        } else {
            format!("{entry}_with_budget")
        };
        let bodies = fn_bodies(&source, &checker);
        assert_eq!(bodies.len(), 1, "{def_path} must define `{checker}` once");
        let mut cursor = 0;
        let mut last_line = 0;
        for (path, line, text) in &cited {
            assert_eq!(
                path, &def_path,
                "{entry}: checks live beside the definition"
            );
            assert_in_range(entry, path, *line);
            assert!(
                *line > last_line,
                "{entry}: cited lines must run device < queue < layout < pipeline < reduce"
            );
            last_line = *line;
            let context = format!(".context(\"{text}\")");
            let at = bodies[0][cursor..].find(&context).unwrap_or_else(|| {
                panic!(
                    "`{checker}` in {path} no longer guards with {context} after the previous check"
                )
            });
            cursor += at + context.len();
        }
    }
}

#[test]
fn every_delegation_row_still_delegates_to_an_entry_point() {
    let rows = audit_table(DELEGATION_TABLE);
    assert_eq!(
        rows.len(),
        13,
        "5 inherent wrappers + 3 GpuEvaluator + 5 RequestEvaluator delegations"
    );
    for row in &rows {
        assert_eq!(row.len(), 3, "caller | delegates to | cited at: {row:?}");
        let caller = row[0]
            .trim_matches('`')
            .rsplit("::")
            .next()
            .unwrap_or_default();
        let delegate = row[1].trim_matches('`');
        assert!(
            ENTRY_POINTS.contains(&delegate),
            "{caller} must delegate to an entry point, not `{delegate}`"
        );
        let (path, line) =
            first_citation(&row[2]).unwrap_or_else(|| panic!("{caller} must cite `path.rs:line`"));
        let source = assert_in_range(caller, &path, line);
        let call = format!("self.{delegate}(");
        assert!(
            fn_bodies(&source, caller)
                .iter()
                .any(|body| body.contains(&call)),
            "`{caller}` in {path} no longer calls `{call}` — the delegation record is stale"
        );
    }
    for trait_name in ["GpuEvaluator::", "RequestEvaluator::"] {
        assert!(
            rows.iter().any(|row| row[0].contains(trait_name)),
            "the delegation table must cover `{trait_name}`"
        );
    }
}

#[test]
fn every_empty_input_short_circuit_carries_a_benign_verdict() {
    let rows = audit_table(SHORT_CIRCUIT_TABLE);
    assert_eq!(rows.len(), 7, "one row per empty-input `Ok` short-circuit");
    for row in &rows {
        assert_eq!(
            row.len(),
            5,
            "entry point | guard | cited at | returns | verdict: {row:?}"
        );
        let entry = row[0].trim_matches('`');
        assert!(
            ENTRY_POINTS.contains(&entry),
            "`{entry}` is not an entry point"
        );
        let (path, line) =
            first_citation(&row[2]).unwrap_or_else(|| panic!("{entry} must cite `path.rs:line`"));
        let source = assert_in_range(entry, &path, line);
        let guard = row[1].trim_matches('`');
        assert!(
            fn_bodies(&source, entry)
                .iter()
                .any(|body| body.contains(&format!("if {guard} {{"))),
            "`{entry}` no longer short-circuits on `{guard}` in {path}"
        );
        // #2241 records no finding, so every short-circuit must be benign.
        assert!(
            row[4].starts_with("benign"),
            "{entry}: verdict must open with `benign`: {}",
            row[4]
        );
    }
}

#[test]
fn the_device_region_states_the_2241_outcome() {
    let doc = read(RECORD);
    let audit = device_region(section(&doc, "## Audit sections"));
    assert!(
        audit.contains("**Outcome (#2241): no finding.**"),
        "the device audit region must state the #2241 outcome explicitly"
    );
}
