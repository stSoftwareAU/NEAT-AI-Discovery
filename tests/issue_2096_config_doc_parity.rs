//! Issue #2125 (chunk 13e of the #2096 config sweep): the `NEAT_AI_DISCOVERY_*` parity gate.
//!
//! `docs/CONFIGURATION.md` is the one place an operator learns that a variable
//! exists. These tests keep that document and the source in step by reading real
//! text, with no hand-kept list of readers:
//!
//! * a variable read under `src/` that the document omits fails test 1, so a new
//!   knob cannot ship undocumented;
//! * a variable the document names that no source file mentions fails test 2, so
//!   a removed knob cannot linger as a documented lever that does nothing;
//! * an allowlisted test-only name that has left `src/` or entered the document
//!   fails test 3;
//! * every platform variable the library sets for itself, read from the
//!   `set_env_if_unset("...")` calls in `src/analysis/utils/platform.rs`, sits under the
//!   `## Set internally by the library` heading (test 4);
//! * `src/config/mod.rs` carries no second copy of the variable table (test 5).
//!
//! Tokens are matched exactly, never with `contains`, so a documented `FOO_BAR`
//! cannot satisfy an undocumented `FOO`. No regex crate is used.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const CONFIGURATION: &str = "docs/CONFIGURATION.md";
const CONFIG_MOD: &str = "src/config/mod.rs";
const USER_FACING: &str = "src/config/user_facing.rs";
const PREFIX: &str = "NEAT_AI_DISCOVERY_";

/// Names that exist only inside unit tests. Each stays out of
/// `docs/CONFIGURATION.md` because no operator ever sets it.
const TEST_ONLY_ALLOWLIST: &[&str] = &[
    // Set in `src/analysis/utils/platform.rs` to prove an unset value is not applied.
    "NEAT_AI_DISCOVERY_TEST_PLATFORM_UNSET",
    // Set in `src/analysis/utils/platform.rs` to prove a preset value is honoured.
    "NEAT_AI_DISCOVERY_TEST_PLATFORM_PRESET",
    // Set in `src/config/helpers.rs` as a unit-test fixture for the parse helper.
    "NEAT_AI_DISCOVERY_PARSE_ENV_HELPER_TEST",
];

/// Variables the host application's loader reads, not this library. They are
/// documented, and the doc labels them as host-loader variables.
const HOST_LOADER_VARS: &[&str] = &["NEAT_AI_DISCOVERY_LIB_PATH"];

/// The heading under which `docs/CONFIGURATION.md` lists the platform variables
/// the library sets for itself.
const INTERNAL_HEADING: &str = "## Set internally by the library";

/// The source file that writes the platform variables the library sets for itself.
const PLATFORM_RS: &str = "src/analysis/utils/platform.rs";

/// The call whose first string argument names a variable the library writes.
const SET_ENV_CALL: &str = "set_env_if_unset(\"";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_text(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// Collect every `*.rs` file under `dir`, recursively. An unreadable directory or
/// entry panics with its path, so a partial walk cannot pass silently.
fn walk_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read directory {}: {e}", dir.display()));
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|e| panic!("cannot read an entry in {}: {e}", dir.display()));
        let path = entry.path();
        let meta = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("cannot stat {}: {e}", path.display()));
        if meta.is_dir() {
            walk_rs(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// Every `src/**/*.rs` file as (repo-relative path, text), sorted by path.
fn src_files() -> Vec<(String, String)> {
    let root = repo_root();
    let src = root.join("src");
    let mut paths = Vec::new();
    walk_rs(&src, &mut paths);
    paths.sort();
    assert!(
        !paths.is_empty(),
        "walk of {} found no .rs files, so the parity gate would pass vacuously",
        src.display()
    );
    let files: Vec<(String, String)> = paths
        .iter()
        .map(|path| {
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .display()
                .to_string();
            (rel, text)
        })
        .collect();
    assert!(
        files.iter().any(|(rel, _)| rel == USER_FACING),
        "walk of {} did not reach {USER_FACING}, so the walk is incomplete",
        src.display()
    );
    files
}

/// What one text yields: the exact `NEAT_AI_DISCOVERY_*` tokens it names, and how
/// many bare prefix hits it carried (a prefix literal such as `NEAT_AI_DISCOVERY_*`
/// in a doc comment, which names no variable).
#[derive(Debug, Default, PartialEq, Eq)]
struct Scan {
    tokens: BTreeSet<String>,
    prefix_only: usize,
}

/// Scan `text` for `NEAT_AI_DISCOVERY_` and take the maximal run of `[A-Z0-9_]`
/// after it as the token. A run ending in `_` is a prefix literal, not a variable
/// name: `NEAT_AI_DISCOVERY_*` and `NEAT_AI_DISCOVERY_FOO_*` (a glob) are both
/// counted in `prefix_only` and dropped from `tokens`.
///
/// Count at the time of writing, from a grep of `src/`: one prefix-only hit, the
/// `NEAT_AI_DISCOVERY_*` glob in the doc comment of `src/config/helpers.rs`. No
/// src token ends in `_`, so the allowlist is not padded with prefix literals.
fn scan_text(text: &str) -> Scan {
    let bytes = text.as_bytes();
    let mut scan = Scan::default();
    let mut from = 0;
    while let Some(rel) = text[from..].find(PREFIX) {
        let start = from + rel;
        let mut end = start + PREFIX.len();
        while end < bytes.len()
            && (bytes[end].is_ascii_uppercase()
                || bytes[end].is_ascii_digit()
                || bytes[end] == b'_')
        {
            end += 1;
        }
        let name = &text[start..end];
        if name.ends_with('_') {
            scan.prefix_only += 1;
        } else {
            scan.tokens.insert(name.to_string());
        }
        from = end;
    }
    scan
}

/// True when `word` occurs in `text` as a whole token: neither neighbour is an
/// ASCII alphanumeric or `_`. So `XEGL_LOG_LEVEL` and `EGL_LOG_LEVEL_X` do not
/// match `EGL_LOG_LEVEL`, while a backticked `` `EGL_LOG_LEVEL` `` does.
fn has_token(text: &str, word: &str) -> bool {
    let bytes = text.as_bytes();
    let is_name = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    text.match_indices(word).any(|(i, _)| {
        let before = i.checked_sub(1).map(|j| bytes[j]);
        let after = bytes.get(i + word.len()).copied();
        !before.is_some_and(is_name) && !after.is_some_and(is_name)
    })
}

/// Every token under `src`, mapped to the first file (in walk order) it appears
/// in, plus the total bare-prefix hits across `src`.
fn index_src(files: &[(String, String)]) -> (BTreeMap<String, String>, usize) {
    let mut first = BTreeMap::new();
    let mut prefix_only = 0;
    for (rel, text) in files {
        let scan = scan_text(text);
        prefix_only += scan.prefix_only;
        for token in scan.tokens {
            first.entry(token).or_insert_with(|| rel.clone());
        }
    }
    (first, prefix_only)
}

/// Source tokens that are neither allowlisted nor documented, with the first file
/// each appears in.
fn undocumented(
    src: &BTreeMap<String, String>,
    doc: &BTreeSet<String>,
    allow: &[&str],
) -> Vec<(String, String)> {
    src.iter()
        .filter(|(token, _)| !allow.contains(&token.as_str()) && !doc.contains(*token))
        .map(|(token, file)| (token.clone(), file.clone()))
        .collect()
}

/// Documented tokens that no source file names and that are not host-loader
/// variables.
fn unread_documented(
    doc: &BTreeSet<String>,
    src: &BTreeMap<String, String>,
    host: &[&str],
) -> Vec<String> {
    doc.iter()
        .filter(|token| !src.contains_key(*token) && !host.contains(&token.as_str()))
        .cloned()
        .collect()
}

/// Allowlisted names that no longer appear in `src`, or that the doc now documents.
fn stale_allowlist(
    allow: &[&str],
    src: &BTreeMap<String, String>,
    doc: &BTreeSet<String>,
) -> Vec<String> {
    allow
        .iter()
        .filter(|name| !src.contains_key(**name) || doc.contains(**name))
        .map(ToString::to_string)
        .collect()
}

/// The lines under the exact heading line `heading`, up to the next `# ` or `## `
/// heading. `None` when the heading line is absent.
fn section_body(doc: &str, heading: &str) -> Option<String> {
    let mut lines = doc.lines();
    lines.find(|line| line.trim_end() == heading)?;
    let body: Vec<&str> = lines
        .take_while(|line| !(line.starts_with("# ") || line.starts_with("## ")))
        .collect();
    Some(body.join("\n"))
}

/// The leading run of `//!` module-doc lines, skipping leading blank lines.
fn module_doc(text: &str) -> Vec<&str> {
    text.lines()
        .skip_while(|line| line.trim().is_empty())
        .take_while(|line| line.starts_with("//!"))
        .collect()
}

/// The module-doc lines that open a Markdown table row, that is `//!` followed by
/// optional whitespace and a `|`.
fn module_doc_table_rows<'a>(lines: &[&'a str]) -> Vec<&'a str> {
    lines
        .iter()
        .copied()
        .filter(|line| line.trim_start_matches("//!").trim_start().starts_with('|'))
        .collect()
}

/// The variable names passed as the first string argument of `set_env_if_unset("..."`
/// in the non-test part of `text`, that is everything before the first `#[cfg(test)]`.
/// The function's own definition and its doc links are not followed by `("`, so they
/// name nothing.
fn set_env_names(text: &str) -> BTreeSet<String> {
    let production = text.split("#[cfg(test)]").next().unwrap_or(text);
    production
        .match_indices(SET_ENV_CALL)
        .map(|(i, _)| {
            let rest = &production[i + SET_ENV_CALL.len()..];
            let end = rest
                .find('"')
                .unwrap_or_else(|| panic!("unterminated string after `{SET_ENV_CALL}`"));
            rest[..end].to_string()
        })
        .collect()
}

fn doc_tokens() -> BTreeSet<String> {
    scan_text(&read_text(CONFIGURATION)).tokens
}

#[test]
fn every_src_variable_is_documented_or_allowlisted_as_test_only() {
    let (src, _) = index_src(&src_files());
    let doc = doc_tokens();
    let missing = undocumented(&src, &doc, TEST_ONLY_ALLOWLIST);
    let report: Vec<String> = missing
        .iter()
        .map(|(token, file)| {
            format!(
                "`{token}` is read first in {file} but has no row in {CONFIGURATION}; add a row \
                 to docs/CONFIGURATION.md for it"
            )
        })
        .collect();
    assert!(
        missing.is_empty(),
        "undocumented src variables (Issue #2096 parity gate):\n{}",
        report.join("\n")
    );
}

#[test]
fn every_documented_variable_is_named_in_src_or_is_a_host_loader_var() {
    let (src, _) = index_src(&src_files());
    let unread = unread_documented(&doc_tokens(), &src, HOST_LOADER_VARS);
    assert!(
        unread.is_empty(),
        "{CONFIGURATION} documents {unread:?}, but no src file names it and it is not in \
         HOST_LOADER_VARS; remove the row, or wire the reader, or list it as a host-loader var"
    );
}

#[test]
fn test_only_allowlist_is_not_stale() {
    let (src, _) = index_src(&src_files());
    let stale = stale_allowlist(TEST_ONLY_ALLOWLIST, &src, &doc_tokens());
    assert!(
        stale.is_empty(),
        "TEST_ONLY_ALLOWLIST carries {stale:?}, which either no longer appears in src or now \
         appears in {CONFIGURATION}; drop it from the allowlist"
    );
}

#[test]
fn platform_variables_set_internally_sit_under_their_heading() {
    let doc = read_text(CONFIGURATION);
    let body = section_body(&doc, INTERNAL_HEADING)
        .unwrap_or_else(|| panic!("{CONFIGURATION} must carry the heading `{INTERNAL_HEADING}`"));
    let written = set_env_names(&read_text(PLATFORM_RS));
    assert!(
        !written.is_empty(),
        "{PLATFORM_RS} yielded no `{SET_ENV_CALL}...` calls, so the check would pass vacuously"
    );
    let missing: Vec<&String> = written
        .iter()
        .filter(|var| !has_token(&body, var))
        .collect();
    assert!(
        missing.is_empty(),
        "{CONFIGURATION} must list {missing:?} under `{INTERNAL_HEADING}`, the section that \
         says which platform variables the library sets for itself"
    );
}

#[test]
fn config_module_doc_is_a_pointer_not_a_second_table() {
    let mod_rs = read_text(CONFIG_MOD);
    let lines = module_doc(&mod_rs);
    assert!(
        !lines.is_empty(),
        "{CONFIG_MOD} must open with `//!` module docs"
    );
    let rows = module_doc_table_rows(&lines);
    assert!(
        rows.is_empty(),
        "{CONFIG_MOD} module docs carry {} Markdown table row(s), a second copy of the \
         variable reference (Issue #2096 single source of truth); replace the table with a \
         pointer to {CONFIGURATION}; first row: {:?}",
        rows.len(),
        rows.first()
    );
    assert!(
        lines.iter().any(|line| line.contains(CONFIGURATION)),
        "{CONFIG_MOD} module docs must link to {CONFIGURATION} as the single source of truth"
    );
}

#[test]
fn scan_text_stops_a_token_at_a_backtick_pipe_quote_or_paren() {
    let text =
        "`NEAT_AI_DISCOVERY_A` | \"NEAT_AI_DISCOVERY_B\" NEAT_AI_DISCOVERY_C) NEAT_AI_DISCOVERY_D,";
    let scan = scan_text(text);
    let expected: BTreeSet<String> = ["A", "B", "C", "D"]
        .iter()
        .map(|s| format!("{PREFIX}{s}"))
        .collect();
    assert_eq!(scan.tokens, expected);
    assert_eq!(scan.prefix_only, 0);
}

#[test]
fn scan_text_keeps_a_token_whose_prefix_is_another_token_distinct() {
    let scan = scan_text("NEAT_AI_DISCOVERY_GPU and NEAT_AI_DISCOVERY_GPU_TIMING");
    let expected: BTreeSet<String> = ["NEAT_AI_DISCOVERY_GPU", "NEAT_AI_DISCOVERY_GPU_TIMING"]
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(scan.tokens, expected);
}

#[test]
fn scan_text_ignores_bare_prefix_and_glob_literals() {
    let scan = scan_text("the NEAT_AI_DISCOVERY_* glob, the NEAT_AI_DISCOVERY_FOO_* family");
    assert!(
        scan.tokens.is_empty(),
        "prefix literals name no variable, found {:?}",
        scan.tokens
    );
    assert_eq!(scan.prefix_only, 2);
}

#[test]
fn has_token_matches_whole_words_only() {
    assert!(has_token("`EGL_LOG_LEVEL` is set", "EGL_LOG_LEVEL"));
    assert!(!has_token("XEGL_LOG_LEVEL", "EGL_LOG_LEVEL"));
    assert!(!has_token("EGL_LOG_LEVEL_X", "EGL_LOG_LEVEL"));
}

#[test]
fn undocumented_flags_a_missing_token_and_respects_exact_match_and_allowlist() {
    let src: BTreeMap<String, String> = [
        ("NEAT_AI_DISCOVERY_FOO", "src/a.rs"),
        ("NEAT_AI_DISCOVERY_BAR", "src/b.rs"),
        ("NEAT_AI_DISCOVERY_TEST_ONLY", "src/c.rs"),
    ]
    .iter()
    .map(|(t, f)| (t.to_string(), f.to_string()))
    .collect();
    // `FOO_BAR` in the doc must not satisfy `FOO` in src.
    let doc: BTreeSet<String> = ["NEAT_AI_DISCOVERY_FOO_BAR", "NEAT_AI_DISCOVERY_BAR"]
        .iter()
        .map(ToString::to_string)
        .collect();
    let missing = undocumented(&src, &doc, &["NEAT_AI_DISCOVERY_TEST_ONLY"]);
    assert_eq!(
        missing,
        vec![("NEAT_AI_DISCOVERY_FOO".to_string(), "src/a.rs".to_string())]
    );
}

#[test]
fn unread_documented_flags_a_token_no_src_names_and_exempts_host_loader_vars() {
    let src: BTreeMap<String, String> = [("NEAT_AI_DISCOVERY_FOO", "src/a.rs")]
        .iter()
        .map(|(t, f)| (t.to_string(), f.to_string()))
        .collect();
    // `FOO` in src must not satisfy the documented `FOO_BAR`.
    let doc: BTreeSet<String> = [
        "NEAT_AI_DISCOVERY_FOO",
        "NEAT_AI_DISCOVERY_FOO_BAR",
        "NEAT_AI_DISCOVERY_LIB",
        "NEAT_AI_DISCOVERY_GONE",
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    let unread = unread_documented(&doc, &src, &["NEAT_AI_DISCOVERY_LIB"]);
    assert_eq!(
        unread,
        vec![
            "NEAT_AI_DISCOVERY_FOO_BAR".to_string(),
            "NEAT_AI_DISCOVERY_GONE".to_string(),
        ]
    );
}

#[test]
fn stale_allowlist_flags_names_absent_from_src_or_present_in_doc() {
    let src: BTreeMap<String, String> = [("A", "src/a.rs"), ("B", "src/b.rs")]
        .iter()
        .map(|(t, f)| (t.to_string(), f.to_string()))
        .collect();
    let doc: BTreeSet<String> = ["B".to_string()].into_iter().collect();
    let stale = stale_allowlist(&["A", "B", "C"], &src, &doc);
    assert_eq!(stale, vec!["B".to_string(), "C".to_string()]);
}

#[test]
fn section_body_stops_at_the_next_level_two_heading_and_needs_an_exact_heading() {
    let doc = "## Set internally by the library\n- `EGL_LOG_LEVEL`\n### sub\n- `MESA_DEBUG`\n## Other\n- `XDG_RUNTIME_DIR`\n";
    let body = section_body(doc, INTERNAL_HEADING).expect("heading present");
    assert!(body.contains("EGL_LOG_LEVEL"));
    assert!(
        body.contains("MESA_DEBUG"),
        "a `###` sub-heading stays in the section"
    );
    assert!(
        !body.contains("XDG_RUNTIME_DIR"),
        "the next `## ` heading ends the section"
    );

    let suffixed = "## Set internally by the library (legacy)\n- `EGL_LOG_LEVEL`\n";
    assert!(
        section_body(suffixed, INTERNAL_HEADING).is_none(),
        "a heading with a suffix is not the exact heading"
    );
}

#[test]
fn module_doc_table_detection_flags_pipe_rows_and_ignores_prose() {
    let lines = [
        "//! Central configuration.",
        "//!",
        "//! | Variable | Type |",
        "//!|----------|------|",
        "//! Pointer to docs/CONFIGURATION.md.",
    ];
    let rows = module_doc_table_rows(&lines);
    assert_eq!(
        rows.len(),
        2,
        "both `//! |` and `//!|` rows are flagged: {rows:?}"
    );
    assert!(
        module_doc_table_rows(&["//! Pointer to docs/CONFIGURATION.md."]).is_empty(),
        "prose is not a table row"
    );
}

#[test]
fn module_doc_is_the_leading_run_of_bang_slash_slash_lines() {
    let text = "\n//! one\n//! two\n\nfn main() {}\n//! not leading\n";
    assert_eq!(module_doc(text), vec!["//! one", "//! two"]);
}

#[test]
fn set_env_names_reads_production_calls_only_and_follows_additions_and_renames() {
    let text = "/// See [`set_env_if_unset`].\n\
                unsafe fn set_env_if_unset(key: &str, value: &str) -> bool { true }\n\
                set_env_if_unset(\"EGL_LOG_LEVEL\", \"fatal\");\n\
                set_env_if_unset(\"MESA_NEW_SWITCH\", \"1\");\n\
                #[cfg(test)]\n\
                set_env_if_unset(\"NEAT_AI_DISCOVERY_TEST_ONLY_FIXTURE\", \"x\");\n";
    let expected: BTreeSet<String> = ["EGL_LOG_LEVEL", "MESA_NEW_SWITCH"]
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(set_env_names(text), expected);
}

#[test]
fn platform_rs_writes_the_variables_the_doc_section_names() {
    // Positive control: the scan reaches the real calls in `apply_mesa_suppression`
    // and `apply_xdg_runtime_dir`, not an empty or partial set.
    let written = set_env_names(&read_text(PLATFORM_RS));
    for var in [
        "EGL_LOG_LEVEL",
        "MESA_GLSL_CACHE_DISABLE",
        "MESA_DEBUG",
        "XDG_RUNTIME_DIR",
    ] {
        assert!(
            written.contains(var),
            "scan of {PLATFORM_RS} missed `{var}`: {written:?}"
        );
    }
}
