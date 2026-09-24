//! Doc-parity gate for Issue #2177.
//!
//! `error_distribution.rs` carried an outlier-analysis surface
//! (`count_outliers`, `filter_outliers`, `is_likely_bimodal`,
//! `has_significant_outliers`, `detect_error_modes`, `detect_modes_histogram`,
//! `OutlierReductionInfo`) with no production caller — the FFI wire field it
//! fed was assigned `None` at every construction site. Its two env vars,
//! `NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS` and `NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE`,
//! were nonetheless documented in `docs/CONFIGURATION.md` and the
//! `src/config/mod.rs` module-doc table: a dead lever per AGENTS.md § "Dead
//! Levers" — an operator who changes either during an incident changes
//! nothing.
//!
//! This file gates the fix from two directions:
//!
//! * **doc-parity** — every `NEAT_AI_DISCOVERY_*` env var documented in
//!   `docs/CONFIGURATION.md` or the `src/config/mod.rs` table has a
//!   reachable production reader, so the next dead lever cannot be
//!   documented into existence either;
//! * **the specific regression** — the two outlier env vars are gone from
//!   both documents;
//!
//! and it checks the surface that must survive still behaves correctly —
//! `ErrorDistribution::from_errors` is live (called by
//! `synapse/post_processing.rs` and `neuron/post_processing.rs`) and must
//! keep computing real statistics after the dead code around it is removed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use neat_ai_discovery::analysis::scoring::error_distribution::ErrorDistribution;

/// Env vars already known to be dead levers, disclosed and tracked ahead of
/// this test (see the follow-up issue filed alongside #2177). Excluded here
/// so this gate stays focused on the #2177 regression; removing them is a
/// separate change.
const DISCLOSED_DEAD_LEVERS: [&str; 3] = [
    "NEAT_AI_DISCOVERY_LIB_PATH",
    "NEAT_AI_DISCOVERY_PRELOAD_ALL",
    "NEAT_AI_DISCOVERY_NOVELTY_GAIN_RELAXATION",
];

/// The two env vars #2177 removes. Must not appear in either document after
/// the fix, and (today, before the fix) must be reachable-dead.
const OUTLIER_VARS: [&str; 2] = [
    "NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS",
    "NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE",
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"))
}

/// Recursively collect every `*.rs` file under `src/`, sorted for
/// deterministic scan order.
fn source_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk(&repo_root().join("src"), &mut out);
    out.sort();
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// The production-only slice of a file's text: everything before the first
/// `#[cfg(test)]` module. Mirrors `production_source` in
/// `tests/issue_2107_chunk_08b_scoring_sweep.rs`.
fn production_text(path: &Path) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    match text.find("#[cfg(test)]") {
        Some(idx) => text[..idx].to_string(),
        None => text,
    }
}

/// Blank out `//` and `/* */` comments while leaving string literals
/// untouched, so quoted env-var literals still match. Deliberately does NOT
/// track single-quote/char-literal state: an unmatched lifetime apostrophe
/// (e.g. `'a`) would otherwise wrongly toggle "in char literal" mode for the
/// rest of the file.
fn strip_comments(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut in_string = false;
    let mut block_depth = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if block_depth > 0 {
            if c == '/' && i + 1 < bytes.len() && bytes[i + 1] as char == '*' {
                block_depth += 1;
                out.push(' ');
                out.push(' ');
                i += 2;
                continue;
            }
            if c == '*' && i + 1 < bytes.len() && bytes[i + 1] as char == '/' {
                block_depth -= 1;
                out.push(' ');
                out.push(' ');
                i += 2;
                continue;
            }
            out.push(if c == '\n' { '\n' } else { ' ' });
            i += 1;
            continue;
        }
        if in_string {
            out.push(c);
            if c == '\\' && i + 1 < bytes.len() {
                out.push(bytes[i + 1] as char);
                i += 2;
                continue;
            }
            if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            i += 1;
            continue;
        }
        if c == '/' && i + 1 < bytes.len() && bytes[i + 1] as char == '/' {
            while i < bytes.len() && bytes[i] as char != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && i + 1 < bytes.len() && bytes[i + 1] as char == '*' {
            block_depth = 1;
            out.push(' ');
            out.push(' ');
            i += 2;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// A function's name and (comment-stripped) body text.
struct FnSpan {
    name: String,
    body: String,
}

/// Good enough for this repo's style; it does not need to be a general Rust
/// parser. Scans for `fn ` at a word boundary, extracts the identifier,
/// skips a `;`-terminated trait-decl signature with no body, otherwise
/// brace-balances from the first `{` to the matching `}`.
fn extract_functions(text: &str) -> Vec<FnSpan> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(rel) = text[i..].find("fn ") {
        let start = i + rel;
        let boundary_ok = start == 0
            || !(bytes[start - 1] as char).is_alphanumeric() && bytes[start - 1] != b'_';
        if !boundary_ok {
            i = start + 3;
            continue;
        }
        let name_start = start + 3;
        let mut j = name_start;
        while j < bytes.len() && (bytes[j] as char).is_alphanumeric() || (j < bytes.len() && bytes[j] == b'_') {
            j += 1;
        }
        if j == name_start {
            i = start + 3;
            continue;
        }
        let name = text[name_start..j].to_string();

        // Find the first `{` or `;` after the signature, whichever comes first.
        let rest = &text[j..];
        let brace_pos = rest.find('{');
        let semi_pos = rest.find(';');
        let body_open = match (brace_pos, semi_pos) {
            (Some(b), Some(s)) if s < b => {
                // Trait-decl signature with no body: skip it.
                i = j + s + 1;
                continue;
            }
            (Some(b), _) => j + b,
            (None, _) => {
                i = j;
                continue;
            }
        };

        // Brace-balance from body_open to find the matching close.
        let mut depth = 0usize;
        let mut k = body_open;
        let mut end = None;
        while k < bytes.len() {
            match bytes[k] as char {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(k);
                        break;
                    }
                }
                _ => {}
            }
            k += 1;
        }
        let Some(end) = end else {
            break;
        };
        out.push(FnSpan {
            name,
            body: text[body_open..=end].to_string(),
        });
        i = end + 1;
    }
    out
}

/// Module-level `const`/`static` `&str` aliases: `const NAME: &str =
/// "literal";` / `static NAME: &str = "literal";` (the `heartbeat.rs:32` /
/// `recovery.rs:17` pattern), keyed by alias name.
fn extract_str_aliases(text: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for kw in ["const ", "static "] {
        let mut i = 0;
        while let Some(rel) = text[i..].find(kw) {
            let start = i + rel + kw.len();
            let Some(colon) = text[start..].find(':') else {
                break;
            };
            let name = text[start..start + colon].trim().to_string();
            let rest = &text[start + colon..];
            if !rest.trim_start().starts_with(':') {
                i = start;
                continue;
            }
            if let Some(str_ty) = rest.find("&str") {
                if let Some(eq) = rest.find('=') {
                    if eq < str_ty + 20 && eq > str_ty {
                        if let Some(q1) = rest[eq..].find('"') {
                            let after = &rest[eq + q1 + 1..];
                            if let Some(q2) = after.find('"') {
                                if !name.is_empty() {
                                    out.insert(name, after[..q2].to_string());
                                }
                            }
                        }
                    }
                }
            }
            i = start;
        }
    }
    out
}

/// The production database: every function body and module-level string
/// alias found across `src/**/*.rs`.
struct ProdDb {
    functions: BTreeMap<String, String>,
    aliases: BTreeMap<String, String>,
}

fn build_prod_db() -> ProdDb {
    let mut functions: BTreeMap<String, String> = BTreeMap::new();
    let mut aliases = BTreeMap::new();
    for path in source_files() {
        let stripped = strip_comments(&production_text(&path));
        for (name, value) in extract_str_aliases(&stripped) {
            aliases.entry(name).or_insert(value);
        }
        for f in extract_functions(&stripped) {
            functions
                .entry(f.name)
                .and_modify(|b: &mut String| {
                    b.push('\n');
                    b.push_str(&f.body);
                })
                .or_insert(f.body);
        }
    }
    ProdDb { functions, aliases }
}

/// Whole-word substring search, avoiding a match on `ident` as a fragment of
/// a longer identifier.
fn contains_ident(body: &str, ident: &str) -> bool {
    let bytes = body.as_bytes();
    let ib = ident.as_bytes();
    if ib.is_empty() {
        return false;
    }
    let mut start = 0;
    while let Some(rel) = body[start..].find(ident) {
        let pos = start + rel;
        let before_ok = pos == 0 || {
            let c = bytes[pos - 1] as char;
            !c.is_alphanumeric() && c != '_'
        };
        let after = pos + ib.len();
        let after_ok = after >= bytes.len() || {
            let c = bytes[after] as char;
            !c.is_alphanumeric() && c != '_'
        };
        if before_ok && after_ok {
            return true;
        }
        start = pos + 1;
    }
    false
}

/// Functions whose body reads `var` directly: either the quoted literal
/// `"VAR"`, or an alias whose value equals `var`.
fn direct_readers(db: &ProdDb, var: &str) -> BTreeSet<String> {
    let literal = format!("\"{var}\"");
    let alias_names: Vec<&String> = db
        .aliases
        .iter()
        .filter(|(_, v)| v.as_str() == var)
        .map(|(k, _)| k)
        .collect();
    let mut out = BTreeSet::new();
    for (name, body) in &db.functions {
        if body.contains(&literal) {
            out.insert(name.clone());
            continue;
        }
        if alias_names.iter().any(|alias| contains_ident(body, alias)) {
            out.insert(name.clone());
        }
    }
    out
}

/// A pure delegating wrapper (<=2 non-empty lines) does not itself count as
/// proof of liveness — only its callers do.
fn is_delegating_wrapper(body: &str) -> bool {
    body.lines().filter(|l| !l.trim().is_empty()).count() <= 2
}

/// Other functions that call `callee` (whole-word name, plus call syntax
/// `callee(`).
fn callers_of<'a>(db: &'a ProdDb, callee: &str) -> Vec<&'a String> {
    let call = format!("{callee}(");
    db.functions
        .iter()
        .filter(|(name, body)| name.as_str() != callee && body.contains(&call))
        .map(|(name, _)| name)
        .collect()
}

/// Rule D: wrapper-aware in-edge liveness. Seed from direct readers, walk
/// upward through callers; a non-wrapper caller proves liveness; a wrapper
/// caller keeps the climb going; the frontier exhausting without a
/// non-wrapper caller means dead. The seed's own wrapper-ness is never
/// treated as proof either way — only callers are examined.
fn is_live(db: &ProdDb, var: &str) -> bool {
    let seeds = direct_readers(db, var);
    if seeds.is_empty() {
        return false;
    }
    let mut visited: BTreeSet<String> = seeds.clone();
    let mut frontier: Vec<String> = seeds.into_iter().collect();
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for callee in &frontier {
            for caller in callers_of(db, callee) {
                if visited.contains(caller) {
                    continue;
                }
                let caller_body = db.functions.get(caller).map(String::as_str).unwrap_or("");
                if !is_delegating_wrapper(caller_body) {
                    return true;
                }
                visited.insert(caller.clone());
                next.push(caller.clone());
            }
        }
        frontier = next;
    }
    false
}

/// Parse `| \`NEAT_AI_DISCOVERY_...\` |` rows out of `docs/CONFIGURATION.md`.
fn configuration_md_vars() -> Vec<String> {
    let text = read("docs/CONFIGURATION.md");
    table_vars(text.lines(), "| `NEAT_AI_DISCOVERY_")
}

/// Parse the same rows out of the `src/config/mod.rs` module-doc table.
fn config_mod_vars() -> Vec<String> {
    let text = read("src/config/mod.rs");
    table_vars(text.lines(), "//! | `NEAT_AI_DISCOVERY_")
}

fn table_vars<'a>(lines: impl Iterator<Item = &'a str>, prefix: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if !trimmed.starts_with(prefix) {
            continue;
        }
        let after_first_tick = &trimmed[trimmed.find('`').unwrap() + 1..];
        if let Some(end) = after_first_tick.find('`') {
            out.push(after_first_tick[..end].to_string());
        }
    }
    out
}

#[test]
fn every_documented_env_var_in_configuration_md_has_a_reachable_reader() {
    let db = build_prod_db();
    let dead: Vec<String> = configuration_md_vars()
        .into_iter()
        .filter(|v| !DISCLOSED_DEAD_LEVERS.contains(&v.as_str()))
        .filter(|v| !is_live(&db, v))
        .collect();
    assert!(
        dead.is_empty(),
        "docs/CONFIGURATION.md documents env var(s) with no reachable production \
         reader (dead levers): {dead:?}. Either wire a reader or delete the row."
    );
}

#[test]
fn every_documented_env_var_in_config_mod_table_has_a_reachable_reader() {
    let db = build_prod_db();
    let dead: Vec<String> = config_mod_vars()
        .into_iter()
        .filter(|v| !DISCLOSED_DEAD_LEVERS.contains(&v.as_str()))
        .filter(|v| !is_live(&db, v))
        .collect();
    assert!(
        dead.is_empty(),
        "src/config/mod.rs table documents env var(s) with no reachable production \
         reader (dead levers): {dead:?}. Either wire a reader or delete the row."
    );
}

#[test]
fn outlier_analysis_and_outlier_percentile_are_no_longer_documented() {
    let md = configuration_md_vars();
    let modrs = config_mod_vars();
    for var in OUTLIER_VARS {
        assert!(
            !md.contains(&var.to_string()),
            "docs/CONFIGURATION.md still documents {var}, but it has no production \
             reader (Issue #2177)"
        );
        assert!(
            !modrs.contains(&var.to_string()),
            "src/config/mod.rs table still documents {var}, but it has no production \
             reader (Issue #2177)"
        );
    }
}

#[test]
fn error_distribution_from_errors_still_works_after_the_dead_lever_cleanup() {
    let errors: Vec<f32> = vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 100.0];
    let dist = ErrorDistribution::from_errors(&errors).expect("non-empty samples yield Some");

    assert_eq!(dist.sample_count, errors.len());
    assert!(dist.mean > 0.0);
    assert!(dist.std_dev >= 0.0);
    assert!(dist.min <= dist.max);
    // percentiles: [p10, p25, p50, p75, p90] — must be non-decreasing.
    for pair in dist.percentiles.windows(2) {
        assert!(
            pair[1] >= pair[0],
            "percentiles must be non-decreasing: {:?}",
            dist.percentiles
        );
    }
    assert!((dist.iqr - (dist.percentiles[3] - dist.percentiles[1])).abs() < 1e-6);
}
