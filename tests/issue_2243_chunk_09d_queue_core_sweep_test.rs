//! Contract tests for the chunk 9d-1 queue-core sweep (Issue #2243).
//!
//! `tests/issue_2288_chunk_09_ledger_scaffold.rs` gates the chunk 9 record's
//! *shape*. This file gates what the #2243 slice writes into its `queue-core`
//! regions, and pins the record against the source it describes:
//!
//! * the `submission.rs` and `execution.rs` rows under `## Files swept` carry a
//!   verdict, not `pending — #2114`;
//! * the queue-core audit region carries one sub-section per file and states
//!   "Outcome (#2243)";
//! * the `send_timeout` finding (`SEC-1124ca631044`) is an open ledger row
//!   linked to #2339 and listed under `## Issues filed`;
//! * the three empty-input short-circuits the record refutes as unreachable
//!   stay unreachable: every production caller of `evaluate_relu`,
//!   `evaluate_activation` and `evaluate_activations_batched` sits in a file
//!   that compares against `MIN_NEURON_SAMPLE_COUNT` before its first call.
//!
//! The short-circuits themselves are driven through the production work loop
//! by the in-crate `src/analysis/gpu/queue/empty_vs_zero_tests.rs`, which needs
//! private `GpuWorkQueue` fields and so cannot live here.

use std::path::{Path, PathBuf};

/// The chunk 9 prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-9-gpu-wgsl.md";

/// The two files the #2243 slice sweeps, as `## Files swept` cites them.
const SWEPT_FILES: [&str; 2] = [
    "src/analysis/gpu/queue/submission.rs",
    "src/analysis/gpu/queue/execution.rs",
];

/// The per-file sub-sections the queue-core audit region must carry.
const SWEEP_HEADINGS: [&str; 2] = [
    "#### `src/analysis/gpu/queue/submission.rs` (Issue #2243)",
    "#### `src/analysis/gpu/queue/execution.rs` (Issue #2243)",
];

/// The finding this slice files.
const FINDING: &str = "SEC-1124ca631044";
const FINDING_ISSUE: &str = "#2339";

/// The `GpuEvaluator` calls whose empty-input short-circuits return zero
/// statistics indistinguishable from an all-zero answer.
const ZERO_SHORT_CIRCUIT_CALLS: [&str; 3] = [
    ".evaluate_relu(",
    ".evaluate_activation(",
    ".evaluate_activations_batched(",
];

/// The production callers the refuted rows name; each must guard on
/// `MIN_NEURON_SAMPLE_COUNT` before submitting.
const GUARDED_CALLERS: [&str; 4] = [
    "src/analysis/synapse/activation_evaluation.rs",
    "src/analysis/synapse/activation_subset_evaluation.rs",
    "src/analysis/synapse/gpu_evaluation.rs",
    "src/analysis/synapse/relu_evaluation.rs",
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

/// The queue-core slice's region: between `<!-- section: queue-core -->` and
/// `<!-- section: queue-lifecycle -->`.
fn queue_core_region(body: &str) -> &str {
    let open = "<!-- section: queue-core -->";
    let close = "<!-- section: queue-lifecycle -->";
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

/// Cells of every Markdown table row in `body` that starts with `prefix`.
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

/// Production source: the text before the first `#[cfg(test)]`.
fn production(source: &str) -> &str {
    source.find("#[cfg(test)]").map_or(source, |i| &source[..i])
}

/// Every `.rs` file under `dir`, as a repo-relative path with `/` separators.
fn rust_files(dir: &Path, out: &mut Vec<String>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let rel = path
                .strip_prefix(repo_root())
                .expect("path under the repo root")
                .to_string_lossy()
                .replace('\\', "/");
            out.push(rel);
        }
    }
}

#[test]
fn no_queue_core_submission_or_execution_row_is_still_pending() {
    let doc = read(RECORD);
    let files = section(section(&doc, "## Files swept"), "### queue-core");
    for path in SWEPT_FILES {
        let rows = prefixed_rows(files, &format!("| `{path}` |"));
        assert_eq!(
            rows.len(),
            1,
            "{path} must have exactly one Files swept row"
        );
        let outcome = rows[0].last().expect("outcome cell");
        assert!(
            !outcome.contains("pending — #2114"),
            "{path}: the queue-core row must carry a verdict, not `pending — #2114`"
        );
        assert!(
            outcome.starts_with("audited, no finding") || outcome.starts_with("finding filed — #"),
            "{path}: outcome must open with `audited, no finding` or `finding filed — #N`: {outcome}"
        );
    }
}

#[test]
fn the_queue_core_region_carries_the_2243_subsections_and_outcome() {
    let doc = read(RECORD);
    let audit = queue_core_region(section(&doc, "## Audit sections")).to_string();
    for heading in SWEEP_HEADINGS {
        // `section` panics with the heading when it is missing.
        assert!(
            !section(&audit, heading).trim().is_empty(),
            "`{heading}` must record a verdict"
        );
    }
    let submission = section(&audit, SWEEP_HEADINGS[0]);
    for check in 1..=5 {
        assert!(
            submission.contains(&format!("**Check {check} ")),
            "the submission.rs sub-section must record check {check}"
        );
    }
    assert!(
        audit.contains(&format!(
            "**Outcome (#2243): one finding, {FINDING_ISSUE}**"
        )),
        "the queue-core audit region must state the #2243 outcome explicitly"
    );
}

#[test]
fn the_send_timeout_finding_is_open_and_linked_to_2339() {
    let doc = read(RECORD);
    let ledger = queue_core_region(section(&doc, "## Ledger"));
    let rows = prefixed_rows(ledger, &format!("| {FINDING} |"));
    assert_eq!(
        rows.len(),
        1,
        "the queue-core ledger must carry one `{FINDING}` row"
    );
    let row = &rows[0];
    assert!(
        row[1].starts_with("`src/analysis/gpu/queue/submission.rs:"),
        "{FINDING} must cite submission.rs: {}",
        row[1]
    );
    assert_eq!(row[2], "CWE-400", "{FINDING} must carry its CWE");
    assert_eq!(
        row.last().map(String::as_str),
        Some(format!("open — {FINDING_ISSUE}").as_str()),
        "{FINDING} must be open and linked to {FINDING_ISSUE}"
    );
    assert!(
        section(&doc, "## Issues filed").contains(&format!("- {FINDING_ISSUE} — `{FINDING}`")),
        "`## Issues filed` must list {FINDING_ISSUE} for {FINDING}"
    );
}

#[test]
fn every_zero_short_circuit_caller_guards_on_min_neuron_sample_count() {
    let mut files = Vec::new();
    rust_files(&repo_root().join("src"), &mut files);
    files.sort();

    let mut callers = Vec::new();
    for rel in &files {
        // The GPU module implements the calls; test files are not production.
        if rel.starts_with("src/analysis/gpu/")
            || rel.ends_with("_tests.rs")
            || rel.ends_with("_test.rs")
            || rel.ends_with("/tests.rs")
        {
            continue;
        }
        let source = read(rel);
        let body = production(&source);
        let Some(first_call) = ZERO_SHORT_CIRCUIT_CALLS
            .iter()
            .filter_map(|call| body.find(call))
            .min()
        else {
            continue;
        };
        let guard = ["< MIN_NEURON_SAMPLE_COUNT", ">= MIN_NEURON_SAMPLE_COUNT"]
            .iter()
            .filter_map(|cmp| body.find(cmp))
            .min();
        assert!(
            guard.is_some_and(|g| g < first_call),
            "{rel} submits a zero-returning GPU evaluation without first comparing \
             against MIN_NEURON_SAMPLE_COUNT, so the #2243 refuted empty-vs-zero rows \
             no longer hold"
        );
        callers.push(rel.clone());
    }
    assert_eq!(
        callers, GUARDED_CALLERS,
        "the production callers changed — re-audit the #2243 refuted empty-vs-zero rows"
    );
}
