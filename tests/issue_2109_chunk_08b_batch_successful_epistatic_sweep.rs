//! Contract tests for the chunk 8b `recommendation batch_successful + epistatic`
//! sweep (Issue #2109).
//!
//! `tests/issue_2103_chunk_08b_ledger_scaffold.rs` gates the record's *shape* —
//! one row per in-scope file, one section per audit sub-issue, the two finding
//! tables and their per-section markers. This file gates the one section Issue
//! #2109 owns, and it does so against the source it claims to have swept:
//!
//! * the 8 rows this sub-issue owns are present and none still reads `pending`,
//!   each with a reason a later reader can check;
//! * every capacity site (`with_capacity` / `vec![_; n]` / `reserve`) and every
//!   float-comparison site in the production half of those files is cited in
//!   the matching finding table, so a new one cannot land unrecorded;
//! * the symbols the outcome traces still exist;
//! * the findings the sweep filed are linked from the outcome and from
//!   `## Issues filed`.
//!
//! The second half of the file is the other kind of check the record needs. The
//! sweep's verdicts rest on *behaviours*, not on prose, and a verdict backed
//! only by a paragraph rots silently. Two tests pin the reachability the
//! findings rest on, and three pin the `clean` verdicts — in particular the
//! answer to the issue body's direct question, whether a NaN "dominant"
//! candidate can suppress real ones in the dominant-neuron deduplicator. When
//! the two findings are fixed the two reachability tests must fail, which is
//! the signal that this section's rows need re-sweeping rather than merely
//! re-reading.

use std::collections::HashSet;
use std::path::PathBuf;

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::recommendation::batch_successful::detect_individually_successful;
use neat_ai_discovery::analysis::recommendation::epistatic::{
    EpistaticPairCandidate, build_source_contribution, deduplicate_by_dominant_neuron,
    detect_epistatic_pairs, detect_synergistic_candidates,
};
use neat_ai_discovery::analysis::samples::{HelpfulSample, HelpfulStats};
use neat_ai_discovery::types::DiscoverRecord;

/// The chunk 8b prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md";

/// The 8 files Issue #2109 swept, in record order.
const SWEPT_FILES: [&str; 8] = [
    "src/analysis/recommendation/batch_successful/mod.rs",
    "src/analysis/recommendation/batch_successful/detection.rs",
    "src/analysis/recommendation/batch_successful/grouping.rs",
    "src/analysis/recommendation/epistatic/mod.rs",
    "src/analysis/recommendation/epistatic/candidate_generation.rs",
    "src/analysis/recommendation/epistatic/deduplication.rs",
    "src/analysis/recommendation/epistatic/pre_screening.rs",
    "src/analysis/recommendation/epistatic/scoring.rs",
];

/// The findings this sweep filed. `#2192` is out of class and is recorded in
/// the outcome, not here.
const FILED_FINDINGS: [&str; 2] = ["#2190", "#2191"];

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

/// `(path, outcome)` rows of the per-file outcome table.
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

fn swept_rows() -> Vec<(String, String)> {
    let doc = read(RECORD);
    file_rows(section(
        &doc,
        "### recommendation batch_successful + epistatic",
    ))
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
    ["total_cmp", "partial_cmp", "sort_by", "max_by", "min_by"]
        .iter()
        .any(|needle| line.contains(needle))
}

/// File stem plus `.rs::`, the citation prefix the record uses for a symbol in
/// that file (CONTRIBUTING.md § Cite Code by Symbol, Never by Line Number).
fn citation_prefix(rel: &str) -> String {
    let file = rel.rsplit('/').next().expect("path names a file");
    format!("{file}::")
}

// =============================================================================
// Record contract — this section describes the code it swept
// =============================================================================

#[test]
fn the_section_owns_exactly_the_files_issue_2109_swept() {
    let rows = swept_rows();
    let paths: Vec<&str> = rows.iter().map(|(path, _)| path.as_str()).collect();
    assert_eq!(
        paths, SWEPT_FILES,
        "the `recommendation batch_successful + epistatic` section must carry one row per file \
         Issue #2109 swept, in record order — a row under the wrong section has no owner and is \
         swept twice or never"
    );
}

#[test]
fn every_row_is_swept_with_a_reason() {
    for (path, outcome) in swept_rows() {
        assert!(
            !outcome.contains("pending"),
            "{path} was swept by Issue #2109, so its outcome must not read `pending`: {outcome}"
        );
        // "clean" alone is an unfalsifiable claim: the reason is what a later
        // reader checks the sweep against.
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
        "recommendation batch_successful + epistatic",
    );

    for file in SWEPT_FILES {
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
        "recommendation batch_successful + epistatic",
    );

    // Issue #1799: an assertion that holds either way is not coverage. The loop
    // below passes vacuously for a file with no comparator, so pin the
    // precondition first — `deduplication.rs` holds the four descending
    // `total_cmp` sorts this section's NaN question is about, and if the
    // detector stops matching there, every citation check below is vacuous.
    let ranking_file = "src/analysis/recommendation/epistatic/deduplication.rs";
    assert!(
        production_source(ranking_file)
            .lines()
            .any(is_comparator_site),
        "{ranking_file} holds the four descending `total_cmp` sorts the sweep answers the \
         NaN-suppression question against — if no comparator is detected there, the detector \
         stopped matching and every citation check below is vacuous"
    );

    for file in SWEPT_FILES {
        if production_source(file).lines().any(is_comparator_site) {
            assert!(
                region.contains(&citation_prefix(file)),
                "{file} ranks with a float comparator, so the float-comparison table must carry \
                 a row recording where the value comes from and what happens when it is NaN"
            );
        }
    }
}

/// Every symbol the outcome claims to have traced, paired with the file that
/// must still declare it. An outcome citing a symbol that no longer exists is
/// describing code that has moved or gone.
const TRACED_SYMBOLS: [(&str, &str); 14] = [
    (
        "src/analysis/recommendation/epistatic/candidate_generation.rs",
        "pub fn detect_epistatic_pairs",
    ),
    (
        "src/analysis/recommendation/epistatic/candidate_generation.rs",
        "fn evaluate_pair_for_epistasis",
    ),
    (
        "src/analysis/recommendation/epistatic/candidate_generation.rs",
        "fn compute_combined_improvement_on_range",
    ),
    (
        "src/analysis/recommendation/epistatic/candidate_generation.rs",
        "fn has_cross_sample_harm",
    ),
    (
        "src/analysis/recommendation/epistatic/deduplication.rs",
        "pub fn deduplicate_by_dominant_neuron",
    ),
    (
        "src/analysis/recommendation/epistatic/deduplication.rs",
        "pub fn deduplicate_synergistic_by_dominant_neuron",
    ),
    (
        "src/analysis/recommendation/epistatic/pre_screening.rs",
        "pub fn detect_synergistic_candidates",
    ),
    (
        "src/analysis/recommendation/epistatic/pre_screening.rs",
        "fn evaluate_residual_reduction",
    ),
    (
        "src/analysis/recommendation/epistatic/scoring.rs",
        "pub fn detect_interfering_pairs",
    ),
    (
        "src/analysis/recommendation/epistatic/scoring.rs",
        "fn check_saturation_risk",
    ),
    (
        "src/analysis/recommendation/batch_successful/detection.rs",
        "pub fn detect_individually_successful",
    ),
    (
        "src/analysis/recommendation/batch_successful/grouping.rs",
        "pub fn group_into_batches",
    ),
    (
        "src/analysis/detection/stats.rs",
        "pub fn pearson_correlation_samples",
    ),
    (
        "src/analysis/synapse/target_analysis/candidate_selection.rs",
        "pub(crate) fn detect_epistatic_and_synergistic",
    ),
]; // Issue #1942: symbols, not line numbers — a symbol survives the refactor.

#[test]
fn the_outcome_cites_symbols_that_still_exist() {
    let doc = read(RECORD);
    let outcome = section(
        &doc,
        "### recommendation batch_successful + epistatic (Issue #2109)",
    );

    for (file, declaration) in TRACED_SYMBOLS {
        let symbol = declaration
            .rsplit(' ')
            .next()
            .expect("declaration names a symbol");
        assert!(
            outcome.contains(symbol),
            "the `recommendation batch_successful + epistatic` outcome must name `{symbol}` — an \
             outcome that cites no guard, bound or dispatch site is a claim, not a sweep"
        );
        assert!(
            read(file).contains(declaration),
            "the outcome cites `{symbol}`, but `{declaration}` is no longer declared in {file} \
             — the sweep describes code that has moved or gone, so this section needs \
             re-sweeping whatever the ledger says"
        );
    }
}

#[test]
fn the_outcome_links_its_filed_findings() {
    let doc = read(RECORD);
    let outcome = section(
        &doc,
        "### recommendation batch_successful + epistatic (Issue #2109)",
    );
    let filed = section(&doc, "## Issues filed");

    for finding in FILED_FINDINGS {
        assert!(
            outcome.contains(finding),
            "the outcome must link every finding it filed — {finding} is missing, and an \
             unlinked finding is invisible to the finalisation sub-issue that reconciles this \
             record"
        );
        assert!(
            filed.contains(finding),
            "`## Issues filed` must list {finding} alongside the sweep's other findings"
        );
    }
}

/// The question the issue body asks this section directly: whether a NaN
/// "dominant" candidate can suppress real candidates in the deduplicator. The
/// outcome must answer it, not merely mention the function.
#[test]
fn the_outcome_answers_the_nan_suppression_question() {
    let doc = read(RECORD);
    let outcome = section(
        &doc,
        "### recommendation batch_successful + epistatic (Issue #2109)",
    );

    assert!(
        outcome.contains("NaN suppression"),
        "the outcome must carry the **NaN suppression** verdict — it is the question Issue #2109 \
         asked this section to answer, and a mention of the function is not an answer"
    );
}

// =============================================================================
// Behaviour contract — the reachability the findings rest on, and the guards
// the `clean` verdicts rest on
// =============================================================================

/// `count` sources over `2 * count` samples, each firing on exactly one sample
/// in each half and nowhere else, with a unit error everywhere.
///
/// Every pair is therefore fully complementary (disjoint firing indices), does
/// no cross-sample harm (each source's activation is `0.0` wherever the other
/// fires), is super-additive against a `0.0` individual improvement, and
/// survives cross-validation because every source contributes to both halves.
/// So every one of the `count * (count - 1) / 2` pairs is emitted, which is
/// what makes the pair scan's growth observable from the outside.
fn complementary_contributions(
    count: usize,
) -> Vec<neat_ai_discovery::analysis::recommendation::epistatic::SourceContribution> {
    let sample_count = count * 2;
    (0..count)
        .map(|source| {
            let samples: Vec<HelpfulSample> = (0..sample_count)
                .map(|i| HelpfulSample {
                    activation: if i == source || i == count + source {
                        1.0
                    } else {
                        0.0
                    },
                    avg_error: 1.0,
                    target_value: None,
                    target_activation: None,
                })
                .collect();
            build_source_contribution(
                &format!("source-{source}"),
                samples,
                HelpfulStats::default(),
                1.0,
                0.0,
            )
        })
        .collect()
}

/// #2190: `detect_epistatic_pairs` runs the full `n * (n - 1) / 2` pair scan
/// with no ceiling and no deadline or cancellation check, so the work grows
/// quadratically in a count the caller sets and cannot be interrupted once it
/// starts.
///
/// The assertion is a *ratio* between two runs of the same code at `n` and
/// `2n` (CODING-STANDARDS § Unit Tests vs Benchmarks — never an absolute
/// wall-clock threshold), expressed as the exact pair count each run emits.
/// When #2190 lands a ceiling or an early return, the larger run stops short
/// and this test fails — which is the signal the row needs re-sweeping.
#[test]
fn epistatic_pair_generation_still_scans_every_pair_with_no_ceiling() {
    const SMALL: usize = 12;
    const LARGE: usize = 24;

    let small = detect_epistatic_pairs("output-0", &complementary_contributions(SMALL), 1.0, None);
    let large = detect_epistatic_pairs("output-0", &complementary_contributions(LARGE), 1.0, None);

    assert_eq!(
        small.len(),
        SMALL * (SMALL - 1) / 2,
        "every pair of fully complementary sources must still be emitted, or the trigger proves \
         nothing about the scan's extent"
    );
    assert_eq!(
        large.len(),
        LARGE * (LARGE - 1) / 2,
        "#2190 says doubling the source count still quadruples the pair scan with no ceiling \
         and no deadline check; if the larger run is now truncated, the bound has landed and \
         the `candidate_generation.rs` row must be re-swept"
    );
}

fn pair(source_a: &str, source_b: &str, combined_improvement: f32) -> EpistaticPairCandidate {
    EpistaticPairCandidate {
        source_a_uuid: source_a.to_string(),
        source_b_uuid: source_b.to_string(),
        target_uuid: "output-0".to_string(),
        weight_a: 0.5,
        weight_b: 0.5,
        combined_improvement,
        // `source_a` dominates, so the group key is its UUID.
        individual_improvement_a: 0.2,
        individual_improvement_b: 0.1,
        complementarity_score: 0.9,
        reason: "fixture".to_string(),
    }
}

/// #2191: `deduplicate_by_dominant_neuron` assembles its result by iterating a
/// `HashMap` whose `RandomState` is seeded per instance, then stabilises it
/// with a `sort_by` that preserves the order of equal keys. Tied
/// `combined_improvement` values therefore come out in an order that differs
/// between two calls on identical input — the same defect class as #2184,
/// which was filed and fixed for `activation_recommendation.rs`.
///
/// Exact ties are not exotic here: quantised `{0, 1}` activations and errors
/// make `compute_combined_improvement_on_range` return bit-identical values
/// for structurally different pairs, and a caller who supplies the records
/// chooses them outright.
///
/// When #2191 lands a deterministic tie-break this test must fail, which is the
/// signal the `deduplication.rs` row needs re-sweeping.
#[test]
fn dominant_neuron_dedup_still_orders_tied_candidates_non_deterministically() {
    const GROUPS: usize = 8;
    const TRIALS: usize = 64;

    let build = || -> Vec<EpistaticPairCandidate> {
        (0..GROUPS)
            .map(|g| pair(&format!("dominant-{g}"), "other", 0.25))
            .collect()
    };

    let orders: HashSet<Vec<String>> = (0..TRIALS)
        .map(|_| {
            deduplicate_by_dominant_neuron(build())
                .into_iter()
                .map(|p| p.source_a_uuid)
                .collect()
        })
        .collect();

    // Sanity first (Issue #1799): the deduplicator must keep every group, or
    // the orderings below are measuring truncation rather than iteration order.
    assert!(
        orders.iter().all(|order| order.len() == GROUPS),
        "each distinct dominant neuron forms its own group, so all {GROUPS} pairs must survive \
         the per-group cap"
    );
    assert!(
        orders.len() > 1,
        "#2191 says {TRIALS} calls on identical input still produce more than one ordering; \
         seeing exactly one means the tie-break is now deterministic and the `deduplication.rs` \
         row must be re-swept"
    );
}

/// The issue body's direct question, answered: a NaN `combined_improvement`
/// cannot reach the deduplicator, so a NaN "dominant" candidate cannot take the
/// head of a group under IEEE-754 totalOrder and suppress real candidates.
///
/// Both producers gate it out before the deduplicator sees it — the epistatic
/// path with `improvement.is_finite()` inside
/// `compute_combined_improvement_on_range` followed by the
/// `combined_improvement > 0.0` super-additivity gate, the synergistic path
/// with the same `> 0.0` gate in `f64` — and the `individual_improvement >= 0.0`
/// pre-screen is fail-closed for a NaN dominance key. This test drives both
/// producers with finite-but-hostile records rather than asserting the prose.
#[test]
fn neither_producer_can_hand_the_deduplicator_a_non_finite_ranking_key() {
    const SAMPLES: usize = 64;

    // Magnitudes that overflow an `f32` accumulator downstream but are each
    // finite on the wire — the shape #2181 / #2182 turn on elsewhere.
    let hostile = |phase: usize| -> Vec<HelpfulSample> {
        (0..SAMPLES)
            .map(|i| HelpfulSample {
                activation: if (i + phase) % 2 == 0 { 3.0e38 } else { -3.0e38 },
                avg_error: if i % 3 == 0 { 1.0e38 } else { -2.0e38 },
                target_value: None,
                target_activation: None,
            })
            .collect()
    };

    let contributions = vec![
        build_source_contribution("hostile-a", hostile(0), HelpfulStats::default(), 1.0e30, 0.0),
        build_source_contribution("hostile-b", hostile(1), HelpfulStats::default(), -1.0e30, 0.0),
        build_source_contribution("hostile-c", hostile(2), HelpfulStats::default(), 1.0e30, 0.5),
    ];

    for candidate in detect_epistatic_pairs("output-0", &contributions, 1.0, None) {
        assert!(
            candidate.combined_improvement.is_finite()
                && candidate.combined_improvement > 0.0
                && candidate.individual_improvement_a.is_finite()
                && candidate.individual_improvement_b.is_finite(),
            "an epistatic pair reaching the deduplicator must carry a finite positive ranking \
             key and finite dominance keys, got {candidate:?}"
        );
    }

    for candidate in detect_synergistic_candidates("output-0", &contributions, 1.0, None) {
        assert!(
            candidate.combined_improvement.is_finite() && candidate.combined_improvement > 0.0,
            "a synergistic candidate reaching the deduplicator must carry a finite positive \
             ranking key, got {candidate:?}"
        );
    }
}

fn record(uuid: &str, obs_index: u32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![error],
    }
}

fn neuron(uuid: &str, neuron_type: &str) -> neat_ai_discovery::NeuronJson {
    neat_ai_discovery::NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

/// The `batch_successful/detection.rs` row: the explicit
/// `!improvement.is_finite()` rejection sits **before** the descending
/// `total_cmp`, so nothing non-finite can reach rank 1 — the guard ordering the
/// ranked detectors of the `recommendation core` section do not have.
#[test]
fn the_individual_candidate_detector_rejects_every_unusable_improvement_before_it_ranks() {
    const SAMPLES: u32 = 64;

    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input"),
            neuron("input-1", "input"),
            neuron("output-0", "output"),
        ],
        synapses: Vec::new(),
        input: 2,
        output: 1,
    };

    // Finite on the wire, but large enough that the f64 least-squares sums and
    // the residual SSE saturate when narrowed back to f32.
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-0".to_string(),
            (0..SAMPLES)
                .map(|i| {
                    let swing = if i % 2 == 0 { 3.0e38 } else { -3.0e38 };
                    record("input-0", i, swing, 0.0)
                })
                .collect(),
        ),
        (
            "input-1".to_string(),
            (0..SAMPLES)
                .map(|i| record("input-1", i, f32::from(u16::try_from(i).unwrap()) * 0.25, 0.0))
                .collect(),
        ),
        (
            "output-0".to_string(),
            (0..SAMPLES)
                .map(|i| {
                    let error = if i % 3 == 0 { 2.0e38 } else { -1.0e38 };
                    record("output-0", i, 0.5, error)
                })
                .collect(),
        ),
    ];

    assert!(
        records
            .iter()
            .flat_map(|(_, rs)| rs.iter())
            .all(|r| r.activation.is_finite() && r.errors.iter().all(|e| e.is_finite())),
        "every recorded value must be finite, or the trigger bypasses the FFI gate instead of \
         defeating it"
    );

    let candidates = detect_individually_successful(&creature, &records);
    let mut previous = f32::INFINITY;
    for candidate in &candidates {
        assert!(
            candidate.improvement.is_finite() && candidate.improvement > 0.0,
            "a non-finite improvement must be rejected before the sort, got {}",
            candidate.improvement
        );
        assert!(
            candidate.weight.is_finite(),
            "the emitted weight must be finite, got {}",
            candidate.weight
        );
        assert!(
            candidate.improvement <= previous,
            "the sort must leave the list in descending improvement order"
        );
        previous = candidate.improvement;
    }
}

/// The `epistatic/pre_screening.rs` row: the `individual_improvement >= 0.0`
/// pre-screen is **fail-closed** for a NaN — a NaN loses the `>=` and the
/// source is dropped — so neither the `max_by` dominance choice nor the
/// descending sort below it can see a NaN key.
#[test]
fn the_synergistic_prescreen_drops_a_nan_individual_improvement() {
    const SAMPLES: usize = 64;

    let samples = |offset: usize| -> Vec<HelpfulSample> {
        (0..SAMPLES)
            .map(|i| HelpfulSample {
                activation: if (i + offset) % 4 < 2 { 1.0 } else { 0.0 },
                avg_error: 0.3,
                target_value: None,
                target_activation: None,
            })
            .collect()
    };

    let contributions = vec![
        build_source_contribution("nan", samples(0), HelpfulStats::default(), 0.5, f32::NAN),
        build_source_contribution("good-a", samples(1), HelpfulStats::default(), 0.5, 0.05),
        build_source_contribution("good-b", samples(2), HelpfulStats::default(), 0.5, 0.02),
    ];

    for candidate in detect_synergistic_candidates("output-0", &contributions, 1.0, None) {
        assert!(
            candidate.primary_source_uuid != "nan" && candidate.complement_source_uuid != "nan",
            "a source whose individual improvement is NaN must be pre-screened out, got \
             {candidate:?}"
        );
    }

    for candidate in detect_epistatic_pairs("output-0", &contributions, 1.0, None) {
        assert!(
            candidate.source_a_uuid != "nan" && candidate.source_b_uuid != "nan",
            "the same pre-screen guards the epistatic path, got {candidate:?}"
        );
    }
}
