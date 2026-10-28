//! Ledger contract for region 13d of the chunk 13 sweep record (Issue #2271,
//! chunk 13d-2 of parent #2124).
//!
//! Region 13d is the `### readers outside src/config` region of
//! `docs/audits/security-sweep-chunk-13-config-env.md`. It inventories every
//! environment reader, temp-dir call and compile-time env read under `src/`
//! outside `src/config/`, classifies each hit, gives every named reader a
//! settings row and a verdict, and files the findings and doc drift it turns up.
//! Three silent failures would make 13d look finished when it is not:
//!
//! * a 13d region left unfinished. A `pending` marker, or an empty table, reads
//!   as a completed sweep, so the chunk looks covered when a region is still open;
//! * a runtime reader with no settings row. An env var that code reads but the
//!   settings table omits is invisible to operators, so it is never tuned or
//!   documented;
//! * the `NEAT_AI_DISCOVERY_LIB_PATH` drift entry lost. Its documentation no
//!   longer matches its reader, and the note that tracks it (#2125) disappears
//!   without anyone deciding to drop it.
//!
//! These tests read the record and check its 13d shape, and fail if a 13d region
//! is empty or contains a standalone `pending`, a named reader loses its settings
//! row or verdict, a cited test no longer exists in `platform.rs`, a 13d region
//! cites code by line number, or the `LIB_PATH` drift entry is dropped. `tests/issue_2211_chunk_13_ledger_scaffold.rs`
//! covers the shape shared by every region; this file covers only 13d.

use std::path::PathBuf;

/// The chunk 13 prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-13-config-env.md";

/// The sub-heading of region 13d in both `## Files swept` and `## Settings table`.
const REGION_13D: &str = "### readers outside src/config";

/// The `rg` command that produces the 13d hit inventory. Recorded verbatim in the
/// files region so a later reader can re-run it.
const RG_COMMAND: &str =
    r"rg -n 'env::var|var_os|vars\(\)|option_env!|env!\(|temp_dir\(' src --glob '!src/config/**'";

/// The total hit count the `rg` command must produce, as recorded in the inventory.
const HIT_COUNT: usize = 28;

/// The five classes a hit may take. `unclassified` is not allowed in the table,
/// only as a `0 unclassified` count in the summary.
const CLASSES: [&str; 5] = [
    "reader",
    "doc-only",
    "compile-time",
    "write",
    "cross-referenced",
];

/// The prefix every runtime reader's variable name carries. Typed once here.
const PREFIX: &str = "NEAT_AI_DISCOVERY_";

/// Short names of the 13 readers that carry `PREFIX`.
const PREFIXED_SHORT_NAMES: [&str; 13] = [
    "MAX_ADD_NEURON_PER_TARGET",
    "MAX_COORDINATED_PER_TARGET",
    "MIN_DISTINCT_TARGETS_PER_BATCH",
    "RISKY_SQUASH_PRIOR",
    "MIN_EXPECTED_GAIN",
    "NEURON_RANKING_BANDS",
    "MIN_BYPASS_WEIGHT_FOR_COLLAPSE",
    "COORDINATED_NOISE_FLOOR_MULTIPLIER",
    "REMOVE_LOW_IMPACT_NOISE_FLOOR",
    "REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS",
    "COOLDOWN_CONSERVATIVE_DIVISOR",
    "COOLDOWN_EXTENDED_DROUGHT_DIVISOR",
    "CONSTANT_SOURCE_EFFECT_THRESHOLD",
];

/// Readers that carry no `PREFIX`. They are environment variables the platform
/// or the logger reads.
const UNPREFIXED_NAMES: [&str; 3] = ["XDG_RUNTIME_DIR", "TMPDIR", "RUST_LOG"];

/// The XDG and env-setup tests the dispositions must cite, each of which must
/// exist in `src/analysis/utils/platform.rs`.
const XDG_TESTS: [&str; 7] = [
    "xdg_runtime_dir_rejects_preexisting_world_writable",
    "xdg_runtime_dir_rejects_symlink",
    "runtime_dir_rejects_other_uid",
    "xdg_runtime_dir_created_with_mode_0700",
    "test_set_env_if_unset_preserves_existing",
    "test_setup_gpu_environment_skips_when_other_threads_live",
    "xdg_runtime_dir_apply_rejects_symlink",
];

/// Symbols the dispositions must name, and the `platform.rs` definitions they
/// must correspond to.
const XDG_SYMBOLS: [&str; 5] = [
    "symlink_metadata",
    "RUNTIME_DIR_MODE",
    "set_env_if_unset",
    "env_setup_pending",
    "runtime_dir_is_trustworthy",
];

const PLATFORM: &str = "src/analysis/utils/platform.rs";

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

/// The files region of 13d: `### readers outside src/config` under `## Files swept`.
fn files_region(doc: &str) -> &str {
    section(section(doc, "## Files swept"), REGION_13D)
}

/// The settings region of 13d: `### readers outside src/config` under `## Settings table`.
fn settings_region(doc: &str) -> &str {
    section(section(doc, "## Settings table"), REGION_13D)
}

/// The part of `body` after the 13d marker, up to the next section marker (or the
/// end of `body`). The marker is the boundary between one region and the next.
fn after_13d_marker<'a>(body: &'a str, heading: &str) -> &'a str {
    let marker = "<!-- section: 13d -->";
    let start = body
        .find(marker)
        .unwrap_or_else(|| panic!("`{heading}` in {RECORD} must carry `{marker}`"));
    let rest = &body[start + marker.len()..];
    let end = rest.find("<!-- section:").unwrap_or(rest.len());
    &rest[..end]
}

/// The 13d part of `## Issues filed`.
fn issues_text(doc: &str) -> &str {
    after_13d_marker(section(doc, "## Issues filed"), "## Issues filed")
}

/// The 13d part of `## Doc drift observed`.
fn drift_text(doc: &str) -> &str {
    after_13d_marker(
        section(doc, "## Doc drift observed"),
        "## Doc drift observed",
    )
}

/// The `## Probe dispositions (13d)` section.
fn dispositions(doc: &str) -> &str {
    section(doc, "## Probe dispositions (13d)")
}

/// Split a Markdown table row into trimmed cells. Strips one leading and one
/// trailing pipe, so an empty trailing cell is kept.
fn cells(line: &str) -> Vec<String> {
    let t = line.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    t.split('|').map(|c| c.trim().to_string()).collect()
}

/// Every table row (a line starting with `|`) in `body`, as cells. Header and
/// separator rows are included; callers filter them out by content.
fn table_rows(body: &str) -> Vec<Vec<String>> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .map(cells)
        .collect()
}

/// Every reader variable name 13d must cover, in full.
fn reader_vars() -> Vec<String> {
    PREFIXED_SHORT_NAMES
        .iter()
        .map(|short| format!("{PREFIX}{short}"))
        .chain(UNPREFIXED_NAMES.iter().copied().map(String::from))
        .collect()
}

/// A full `NEAT_AI_DISCOVERY_*` name from its short name.
fn prefixed(short: &str) -> String {
    format!("{PREFIX}{short}")
}

/// True when the lowercase word `pending` occurs with neither neighbour being an
/// ASCII alphanumeric or `_`. So `env_setup_pending` and `spending` are not hits,
/// while `pending — x` and `Pending.` are. Linear in the length of `text`.
fn has_standalone_pending(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    const WORD: &[u8] = b"pending";
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    bytes.windows(WORD.len()).enumerate().any(|(i, window)| {
        if window != WORD {
            return false;
        }
        let before = i.checked_sub(1).map(|j| bytes[j]);
        let after = bytes.get(i + WORD.len()).copied();
        !before.is_some_and(is_word) && !after.is_some_and(is_word)
    })
}

/// True when `line` cites a line number, which CONTRIBUTING.md forbids (Issue
/// #1942). Three shapes are flagged: `.rs:` or `.md:` followed by a digit, and a
/// standalone `L` followed by digits, with no word character before the `L` and
/// none after the digits. Scanned by hand over the chars in one pass, so it is
/// linear in the length of `line` and needs no regex crate.
fn cites_a_line_number(line: &str) -> bool {
    let chars: Vec<char> = line.chars().collect();
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    for (i, &c) in chars.iter().enumerate() {
        if c == ':'
            && i >= 3
            && matches!(&chars[i - 3..i], ['.', 'r', 's'] | ['.', 'm', 'd'])
            && chars.get(i + 1).is_some_and(char::is_ascii_digit)
        {
            return true;
        }
        if c == 'L' && !(i > 0 && is_word(chars[i - 1])) {
            let digits = chars[i + 1..]
                .iter()
                .take_while(|d| d.is_ascii_digit())
                .count();
            let after_digits = chars.get(i + 1 + digits).copied();
            if digits > 0 && !after_digits.is_some_and(is_word) {
                return true;
            }
        }
    }
    false
}

#[test]
fn region_13d_is_non_empty_in_every_section_and_has_no_pending() {
    let doc = read(RECORD);
    let files = files_region(&doc);
    let settings = settings_region(&doc);
    let issues = issues_text(&doc);
    let drift = drift_text(&doc);
    let dispositions_body = dispositions(&doc);

    let has_src_row = table_rows(files).iter().any(|row| {
        row.first()
            .is_some_and(|c| c.starts_with('`') && c.trim_matches('`').starts_with("src/"))
    });
    assert!(
        has_src_row,
        "{RECORD} region 13d in `## Files swept` has no data row with a backticked `src/` \
         path, so the 13d inventory is empty and no reader was swept"
    );

    let has_settings_row = table_rows(settings)
        .iter()
        .any(|row| row.first().is_some_and(|c| c.starts_with('`')));
    assert!(
        has_settings_row,
        "{RECORD} region 13d in `## Settings table` has no row whose first cell is a \
         backticked variable name, so no runtime reader has a settings row"
    );

    assert!(
        issues.lines().any(|l| l.starts_with("- ")),
        "{RECORD} 13d part of `## Issues filed` has no `- ` finding, so the 13d findings \
         are unfiled"
    );
    assert!(
        drift.lines().any(|l| l.starts_with("- ")),
        "{RECORD} 13d part of `## Doc drift observed` has no `- ` entry, so the 13d drift \
         is unrecorded"
    );
    assert!(
        dispositions_body.lines().any(|l| l.starts_with("- **")),
        "{RECORD} `## Probe dispositions (13d)` has no `- **` disposition, so the probe \
         results are unrecorded"
    );

    let texts = [
        ("files region 13d", files),
        ("settings region 13d", settings),
        ("13d issues text", issues),
        ("13d drift text", drift),
        ("probe dispositions (13d)", dispositions_body),
    ];
    for (label, text) in texts {
        assert!(
            !has_standalone_pending(text),
            "{label} in {RECORD} still carries a standalone `pending`, so 13d reads as \
             unfinished but the record does not say so; finish the row or say why it is open"
        );
    }
}

#[test]
fn the_inventory_records_the_exact_rg_command_hit_count_and_one_class_per_hit() {
    let doc = read(RECORD);
    let files = files_region(&doc);

    assert!(
        files.contains(RG_COMMAND),
        "{RECORD} region 13d must record the exact rg command `{RG_COMMAND}` so the \
         inventory can be re-run"
    );
    assert!(
        files.contains("28 hits"),
        "{RECORD} region 13d must state `28 hits` for the rg command, so a hit added or \
         lost since the sweep is noticed"
    );

    let hits: Vec<Vec<String>> = table_rows(files)
        .into_iter()
        .filter(|row| {
            row.len() == 5
                && row
                    .first()
                    .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        })
        .collect();
    assert_eq!(
        hits.len(),
        HIT_COUNT,
        "{RECORD} region 13d classification table must carry one row per rg hit, \
         {HIT_COUNT} in total, found {}; a missing row is a hit nobody classified",
        hits.len()
    );

    let mut numbers: Vec<usize> = hits
        .iter()
        .map(|row| {
            row[0]
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("hit number `{}` must parse: {e}", row[0]))
        })
        .collect();
    numbers.sort_unstable();
    let expected: Vec<usize> = (1..=HIT_COUNT).collect();
    assert_eq!(
        numbers, expected,
        "{RECORD} region 13d hit numbers must be 1..={HIT_COUNT}, consecutive with no \
         duplicates, so every rg hit has exactly one row"
    );

    for row in &hits {
        assert!(
            CLASSES.contains(&row[3].as_str()),
            "{RECORD} region 13d hit {} carries class `{}`, which is not one of {CLASSES:?}",
            row[0],
            row[3]
        );
    }

    let count = |class: &str| hits.iter().filter(|row| row[3] == class).count();
    let summary = format!(
        "{} reader, {} doc-only, {} compile-time, {} write, {} cross-referenced",
        count("reader"),
        count("doc-only"),
        count("compile-time"),
        count("write"),
        count("cross-referenced"),
    );
    assert!(
        files.contains(&summary),
        "{RECORD} region 13d must carry the summary `{summary}` built from its table, so the \
         count and the table cannot drift apart"
    );
    assert!(
        files.contains("0 unclassified"),
        "{RECORD} region 13d must state `0 unclassified`, or an unclassified hit goes unnoticed"
    );
}

#[test]
fn every_named_reader_has_one_ten_column_settings_row_and_a_verdict() {
    let doc = read(RECORD);
    let settings = settings_region(&doc);
    let rows = table_rows(settings);

    for name in reader_vars() {
        let key = format!("`{name}`");

        let ten: Vec<&Vec<String>> = rows
            .iter()
            .filter(|row| row.len() == 10 && row[0] == key)
            .collect();
        assert_eq!(
            ten.len(),
            1,
            "{RECORD} region 13d must carry exactly one ten-column settings row for {key}, \
             found {}; a runtime reader with no settings row is invisible to operators",
            ten.len()
        );
        assert!(
            ten[0].iter().all(|cell| !cell.is_empty()),
            "{RECORD} settings row for {key} has an empty cell, so one of the ten agreed \
             columns was never filled in"
        );

        let verdicts: Vec<&Vec<String>> = rows
            .iter()
            .filter(|row| row.len() == 4 && row[0] == key)
            .collect();
        assert_eq!(
            verdicts.len(),
            1,
            "{RECORD} region 13d must carry exactly one verdict row for {key}, found {}; \
             without a verdict the reader has no recorded outcome",
            verdicts.len()
        );
        let verdict = &verdicts[0][3];
        assert!(
            verdict.starts_with("matches") || verdict.starts_with("drift"),
            "{RECORD} verdict for {key} must start with `matches` or `drift`, found `{verdict}`"
        );
    }
}

#[test]
fn every_xdg_check_is_cited_with_a_test_that_exists() {
    let doc = read(RECORD);
    let dispositions_body = dispositions(&doc);
    let platform = read(PLATFORM);

    for name in XDG_TESTS {
        assert!(
            dispositions_body.contains(name),
            "`## Probe dispositions (13d)` in {RECORD} must cite the XDG test `{name}`, \
             or the probe result for it is not recorded"
        );
        let definition = format!("fn {name}(");
        assert!(
            platform.contains(&definition),
            "{PLATFORM} must define `{definition}`, which the dispositions cite; a cited \
             test that no longer exists makes the disposition unverifiable"
        );
    }

    for symbol in XDG_SYMBOLS {
        assert!(
            dispositions_body.contains(symbol),
            "`## Probe dispositions (13d)` in {RECORD} must name `{symbol}`, the XDG \
             guard it dispositions"
        );
    }

    for definition in [
        "fn runtime_dir_is_trustworthy",
        "fn set_env_if_unset",
        "fn env_setup_pending",
        "const RUNTIME_DIR_MODE",
    ] {
        assert!(
            platform.contains(definition),
            "{PLATFORM} must contain `{definition}`, the guard the dispositions refer to"
        );
    }
}

#[test]
fn the_named_dispositions_are_recorded() {
    let doc = read(RECORD);
    let dispositions_body = dispositions(&doc);

    for needle in [
        "NEURON_RANKING_BANDS",
        "samples/thresholds.rs::get_constant_source_threshold",
        "XDG_RUNTIME_DIR",
        "RUST_LOG",
        "temp_dir",
        "#2118",
        "utils/mod.rs::verbose_enabled",
        "option_env!",
        "CARGO_PKG_VERSION",
        "LIB_PATH",
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "#2463",
    ] {
        assert!(
            dispositions_body.contains(needle),
            "`## Probe dispositions (13d)` in {RECORD} must record `{needle}`; a probe \
             without a recorded disposition reads as unexamined"
        );
    }
}

#[test]
fn lib_path_is_recorded_under_doc_drift_for_2125() {
    let doc = read(RECORD);
    let drift_section = section(&doc, "## Doc drift observed");
    let lib_path = prefixed("LIB_PATH");

    assert!(
        drift_section.contains(&lib_path),
        "`## Doc drift observed` in {RECORD} must carry `{lib_path}`; its drift entry would \
         otherwise be lost"
    );

    let drift = drift_text(&doc);
    assert!(
        drift.contains(&lib_path),
        "the 13d drift text in {RECORD} must carry `{lib_path}`, the drift entry for 13d"
    );
    assert!(
        drift.contains("#2125"),
        "the 13d drift text in {RECORD} must link `{lib_path}` to #2125, the issue that \
         tracks it"
    );
}

#[test]
fn the_filed_finding_is_linked_and_has_its_probe() {
    let doc = read(RECORD);
    let issues = issues_text(&doc);

    assert!(
        issues.contains("#2463"),
        "the 13d part of `## Issues filed` in {RECORD} must file #2463, the constant-source \
         threshold finding"
    );

    let constant = prefixed("CONSTANT_SOURCE_EFFECT_THRESHOLD");
    let key = format!("`{constant}`");
    let settings = settings_region(&doc);
    let row = table_rows(settings)
        .into_iter()
        .find(|row| row.len() == 10 && row[0] == key)
        .unwrap_or_else(|| {
            panic!("{RECORD} region 13d must carry a ten-column settings row for {key}")
        });
    assert!(
        row.iter().any(|cell| cell.contains("#2463")),
        "the settings row for {key} in {RECORD} must link #2463, the finding it is affected by"
    );

    let probe_path = "tests/issue_2463_constant_source_threshold_explicit_default.rs";
    let probe = read(probe_path);
    assert!(
        probe.contains(r#"#[ignore = "finding: #2463"#),
        "{probe_path} must carry `#[ignore = \"finding: #2463\"]`, the failing-first probe \
         the finding links to"
    );

    let comment_links = issues.matches("issuecomment-").count();
    assert!(
        comment_links >= 3,
        "the 13d part of `## Issues filed` in {RECORD} must link the #2276, #2125 and #2096 \
         comments with `issuecomment-` URLs (found {comment_links}), so each finding can be \
         traced to its source"
    );
}

#[test]
fn pending_check_flags_a_standalone_word_and_ignores_identifiers() {
    for text in ["pending", "Pending.", "x pending — y", "(pending)"] {
        assert!(
            has_standalone_pending(text),
            "`{text}` carries a standalone `pending` and must be flagged"
        );
    }
    for text in ["env_setup_pending", "spending", "pendingly", ""] {
        assert!(
            !has_standalone_pending(text),
            "`{text}` does not carry a standalone `pending` and must not be flagged"
        );
    }
}

#[test]
fn the_13d_regions_cite_no_line_numbers() {
    // CONTRIBUTING.md § Cite Code by Symbol, Never by Line Number (Issue #1942),
    // applied to the five 13d texts. Modelled on the chunk 8b record check.
    let doc = read(RECORD);
    let texts = [
        ("files region 13d", files_region(&doc)),
        ("settings region 13d", settings_region(&doc)),
        ("13d issues text", issues_text(&doc)),
        ("13d drift text", drift_text(&doc)),
        ("probe dispositions (13d)", dispositions(&doc)),
    ];
    for (label, text) in texts {
        let offenders: Vec<&str> = text
            .lines()
            .filter(|line| cites_a_line_number(line))
            .collect();
        assert!(
            offenders.is_empty(),
            "{label} in {RECORD} cites code by line number, which CONTRIBUTING.md forbids \
             (Issue #1942); cite `file.rs::symbol` instead: {offenders:?}"
        );
    }
}

#[test]
fn the_line_number_scanner_flags_citations_and_ignores_identifiers() {
    for text in ["x.rs:12", "see L59", "(L1528)", "a.md:104"] {
        assert!(
            cites_a_line_number(text),
            "`{text}` cites a line number and must be flagged"
        );
    }
    for text in [
        "L",
        "LIB_PATH",
        "SOLID",
        "2L",
        "NEAT_AI_DISCOVERY_LIB_PATH",
        "0o700",
    ] {
        assert!(
            !cites_a_line_number(text),
            "`{text}` does not cite a line number and must not be flagged"
        );
    }
}
