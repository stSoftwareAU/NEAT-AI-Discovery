//! Issue #1942 — `docs/COST_FUNCTION_NOTES.md` must not rest on a premise the
//! source contradicts, and every site it catalogues must resolve to a symbol
//! that actually exists.
//!
//! The document previously asserted "zero references to any cost name in
//! `src/`" and handed the reader the grep to prove it — a grep that returns
//! over a hundred hits. Sixteen of its 66 `file:line` catalogue references had
//! also drifted onto unrelated code. Each test below proves the current
//! behaviour or structure from the real source first, then asserts the prose
//! agrees with what was just demonstrated.

use neat_ai_discovery::analysis::cost_function_hint::CostFunctionHint;
use neat_ai_discovery::analysis::task_descriptor::{TargetTopology, TaskDescriptor};
use std::path::{Path, PathBuf};

const NOTES: &str = include_str!("../docs/COST_FUNCTION_NOTES.md");

/// The seven NEAT-AI built-in cost names the document's own grep searches for.
const COST_NAMES: [&str; 7] = [
    "MSE",
    "MAE",
    "MAPE",
    "MSLE",
    "HINGE",
    "CROSS_ENTROPY",
    "CATEGORICAL_ERROR",
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.rs` file under `src/`, relative to the repo root.
fn source_files() -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![repo_root().join("src")];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("src/ must be readable") {
            let path = entry.expect("directory entry must be readable").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                found.push(path);
            }
        }
    }
    assert!(!found.is_empty(), "src/ must contain Rust sources");
    found
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// `true` for lines that are only a comment — the document distinguishes a
/// cost name *mentioned* in prose from one *branched on* in code.
fn is_comment(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("//") || trimmed.starts_with('*')
}

/// Files carrying a `"COST_NAME" =>` match arm, i.e. code that dispatches on a
/// cost name. Returns repo-relative paths.
fn files_dispatching_on_a_cost_name() -> Vec<String> {
    let mut hits: Vec<String> = Vec::new();
    for path in source_files() {
        let body = read(&path);
        let dispatches = body.lines().filter(|l| !is_comment(l)).any(|line| {
            COST_NAMES
                .iter()
                .any(|name| line.contains(&format!("\"{name}\"")) && line.contains("=>"))
        });
        if dispatches {
            let rel = path
                .strip_prefix(repo_root())
                .expect("source file must live under the repo root");
            hits.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    hits.sort();
    hits
}

/// Backticked `path/to/file.rs::symbol` references, in document order.
fn site_references(doc: &str) -> Vec<(String, String)> {
    let mut refs = Vec::new();
    for span in doc.split('`').skip(1).step_by(2) {
        if let Some((path, symbol)) = span.split_once("::")
            && path.ends_with(".rs")
            && !symbol.contains(' ')
            // `<file>.rs::<function>` is the convention note, not a site.
            && !path.starts_with('<')
        {
            refs.push((path.to_string(), symbol.to_string()));
        }
    }
    assert!(
        refs.len() > 50,
        "the catalogue must still carry its per-consumer sites, found {}",
        refs.len()
    );
    refs
}

/// Resolve a documented path suffix (`fan_in.rs`, `batch_successful/detection.rs`)
/// to exactly one file under `src/`.
fn resolve_suffix(suffix: &str) -> PathBuf {
    let matches: Vec<PathBuf> = source_files()
        .into_iter()
        .filter(|p| {
            p.to_string_lossy()
                .replace('\\', "/")
                .ends_with(&format!("/{suffix}"))
        })
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "`{suffix}` must identify exactly one file under src/, found {matches:?}"
    );
    matches.into_iter().next().expect("checked above")
}

/// The premise: cost names DO reach this crate, and the only code that
/// dispatches on them is the residual-semantics hint plumbing.
#[test]
fn cost_names_reach_the_crate_only_through_the_two_name_mappers() {
    // Behaviour first — both mappers genuinely branch on the cost name.
    assert_eq!(
        CostFunctionHint::from_name("MSE"),
        CostFunctionHint::LinearResidual,
        "MSE records a linear residual"
    );
    assert_eq!(
        CostFunctionHint::from_name("HINGE"),
        CostFunctionHint::NonLinearResidual,
        "HINGE does not, so the implied-target sites must gate off"
    );
    assert_eq!(
        TaskDescriptor::from_name("CATEGORICAL_ERROR", 3).target_topology,
        TargetTopology::OneHot,
        "the descriptor maps a cost name onto a target topology"
    );
    assert_ne!(
        TaskDescriptor::from_name("MSE", 3).target_topology,
        TaskDescriptor::from_name("CATEGORICAL_ERROR", 3).target_topology,
        "different cost names must yield different descriptors"
    );

    // Structure — dispatch is confined to the two mappers named in §1.
    assert_eq!(
        files_dispatching_on_a_cost_name(),
        vec![
            "src/analysis/cost_function_hint.rs".to_string(),
            "src/analysis/task_descriptor.rs".to_string(),
        ],
        "only the hint plumbing may branch on a cost name — a new dispatch site \
         breaks the cost-agnostic invariant §1 documents"
    );
}

/// The falsified premise must be gone, and replaced by one that names the
/// plumbing the grep actually finds.
#[test]
fn the_background_section_does_not_claim_zero_cost_name_references() {
    let hits: usize = source_files()
        .iter()
        .map(|p| {
            read(p)
                .lines()
                .filter(|l| COST_NAMES.iter().any(|n| l.contains(n)))
                .count()
        })
        .sum();
    assert!(
        hits > 0,
        "the doc's own grep must return hits — that is the premise being fixed"
    );

    assert!(
        !NOTES.contains("zero references to any cost name"),
        "§1 must not repeat the falsified zero-references claim ({hits} lines match the grep)"
    );
    for mapper in [
        "CostFunctionHint::from_name",
        "TaskDescriptor::from_name",
        "analysis::task_descriptor",
    ] {
        assert!(
            NOTES.contains(mapper),
            "§1 must name the plumbing the grep finds: {mapper}"
        );
    }
}

/// Every catalogued site must resolve to a function that exists today.
#[test]
fn every_documented_site_resolves_to_a_function_in_the_source() {
    for (path, symbol) in site_references(NOTES) {
        let resolved = resolve_suffix(&path);
        let body = read(&resolved);
        assert!(
            body.contains(&format!("fn {symbol}")),
            "`{path}::{symbol}` is cited by COST_FUNCTION_NOTES.md but {} defines no such function",
            resolved.display()
        );
    }
}

/// Line numbers rot; symbols do not. The catalogue must stay symbol-anchored.
#[test]
fn the_catalogue_cites_no_bare_line_numbers() {
    let stale: Vec<&str> = NOTES
        .split('`')
        .skip(1)
        .step_by(2)
        .filter(|span| {
            span.split_once(".rs:")
                .is_some_and(|(_, tail)| tail.starts_with(|c: char| c.is_ascii_digit()))
        })
        .collect();
    assert!(
        stale.is_empty(),
        "cite `file.rs::symbol`, not a line number that rots on the next refactor: {stale:?}"
    );
}

/// §8 provenance — the two anchors the audit was cross-checked against.
#[test]
fn the_provenance_anchors_still_carry_the_errors_field() {
    for (rel, symbol) in [
        ("src/types.rs", "DiscoverRecord::errors"),
        (
            "src/ffi_types/responses/export.rs",
            "DiscoverRecordJson::errors",
        ),
    ] {
        let body = read(&repo_root().join(rel));
        assert!(
            body.contains("pub errors: Vec<f32>"),
            "{rel} must still declare the errors field §8 cites"
        );
        assert!(
            NOTES.contains(symbol) && NOTES.contains(rel),
            "§8 must cite {symbol} in {rel} by name rather than by line number"
        );
    }
}
