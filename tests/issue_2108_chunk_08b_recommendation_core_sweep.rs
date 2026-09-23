//! Contract tests for the chunk 8b `recommendation core` sweep (Issue #2108).
//!
//! `tests/issue_2103_chunk_08b_ledger_scaffold.rs` gates the record's *shape* —
//! one row per in-scope file, one section per audit sub-issue, the two finding
//! tables and their per-section markers. This file gates the one section Issue
//! #2108 owns, and it does so against the source it claims to have swept:
//!
//! * the 8 rows this sub-issue owns are present and none still reads `pending`,
//!   each with a reason a later reader can check;
//! * every capacity site (`with_capacity` / `vec![_; n]` / `reserve`) and every
//!   float-comparison site in the production half of those files is cited in
//!   the matching finding table, so a new one cannot land unrecorded;
//! * the symbols the outcome traces still exist;
//! * the three findings the sweep filed are linked from the outcome and from
//!   `## Issues filed`, and the ranking-integrity verdict is recorded per
//!   detector.
//!
//! The second half of the file is the other kind of check the record needs. The
//! sweep's verdicts rest on *behaviours*, not on prose, and a verdict backed
//! only by a paragraph rots silently. Three of these tests pin the reachability
//! the findings rest on — including one that drives the FFI boundary itself, so
//! "reachable from finite records" is asserted where the gate actually lives —
//! and three pin the `clean` verdicts. When #2181 is fixed the two `fan_in.rs`
//! reachability tests must fail, which is the signal that this section's rows
//! need re-sweeping rather than merely re-reading.

use std::path::PathBuf;

use neat_ai_discovery::analysis::detection::stats::pearson_correlation;
use neat_ai_discovery::analysis::recommendation::activation_recommendation::recommend_activation_function;
use neat_ai_discovery::analysis::recommendation::fan_in::detect_fan_in_candidates;
use neat_ai_discovery::analysis::recommendation::gradient_discovery::compute_synapse_gradient;
use neat_ai_discovery::analysis::recommendation::sample_weighted::{
    compute_sample_weights, stratify_samples,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronData, NeuronJson};

/// The chunk 8b prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md";

/// The 8 files Issue #2108 swept, in record order.
const RECOMMENDATION_CORE_FILES: [&str; 8] = [
    "src/analysis/recommendation/mod.rs",
    "src/analysis/recommendation/activation_recommendation.rs",
    "src/analysis/recommendation/fan_in.rs",
    "src/analysis/recommendation/gradient_discovery.rs",
    "src/analysis/recommendation/multi_hop.rs",
    "src/analysis/recommendation/output_bias_drift.rs",
    "src/analysis/recommendation/output_competition.rs",
    "src/analysis/recommendation/sample_weighted.rs",
];

/// The findings this sweep filed. `#2184` and `#2185` are out of class and are
/// recorded in the outcome, not here.
const FILED_FINDINGS: [&str; 3] = ["#2181", "#2182", "#2183"];

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

fn recommendation_core_rows() -> Vec<(String, String)> {
    let doc = read(RECORD);
    file_rows(section(&doc, "### recommendation core"))
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
// Record contract — the `recommendation core` section describes the code it swept
// =============================================================================

#[test]
fn the_recommendation_core_section_owns_exactly_the_files_issue_2108_swept() {
    let rows = recommendation_core_rows();
    let paths: Vec<&str> = rows.iter().map(|(path, _)| path.as_str()).collect();
    assert_eq!(
        paths, RECOMMENDATION_CORE_FILES,
        "the `recommendation core` section must carry one row per file Issue #2108 swept, in \
         record order — a row under the wrong section has no owner and is swept twice or never"
    );
}

#[test]
fn every_recommendation_core_row_is_swept_with_a_reason() {
    for (path, outcome) in recommendation_core_rows() {
        assert!(
            !outcome.contains("pending"),
            "{path} was swept by Issue #2108, so its outcome must not read `pending`: {outcome}"
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
        "recommendation core",
    );

    for file in RECOMMENDATION_CORE_FILES {
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
        "recommendation core",
    );

    // Issue #1799: an assertion that holds either way is not coverage. The loop
    // below passes vacuously for a file with no comparator, so pin the
    // precondition first — `fan_in.rs` holds the two non-total comparators this
    // sweep is about, and if the detector stops matching there, every citation
    // check below is vacuous.
    let ranking_file = "src/analysis/recommendation/fan_in.rs";
    assert!(
        production_source(ranking_file)
            .lines()
            .any(is_comparator_site),
        "{ranking_file} holds the two `partial_cmp(…).unwrap_or(Equal)` sorts the sweep filed \
         #2181 against — if no comparator is detected there, the detector stopped matching and \
         every citation check below is vacuous"
    );

    for file in RECOMMENDATION_CORE_FILES {
        if production_source(file).lines().any(is_comparator_site) {
            assert!(
                region.contains(&citation_prefix(file)),
                "{file} ranks with a float comparator, so the float-comparison table must carry \
                 a row recording where the value comes from and what happens when it is NaN"
            );
        }
    }
}

/// Every symbol the `recommendation core` outcome claims to have traced, paired
/// with the file that must still declare it. An outcome citing a symbol that no
/// longer exists is describing code that has moved or gone.
const TRACED_SYMBOLS: [(&str, &str); 22] = [
    (
        "src/analysis/recommendation/fan_in.rs",
        "pub fn detect_fan_in_candidates",
    ),
    (
        "src/analysis/recommendation/fan_in.rs",
        "fn evaluate_fan_in_pair",
    ),
    (
        "src/analysis/recommendation/fan_in.rs",
        "fn compute_least_squares_improvement",
    ),
    (
        "src/analysis/recommendation/fan_in.rs",
        "fn compute_two_input_regression",
    ),
    (
        "src/analysis/detection/stats.rs",
        "pub fn pearson_correlation",
    ),
    (
        "src/analysis/detection/stats.rs",
        "pub fn pearson_correlation_hashmaps",
    ),
    ("src/ffi_types/mod.rs", "fn deserialise_activation"),
    ("src/ffi_types/mod.rs", "fn deserialise_finite_errors"),
    (
        "src/analysis/recommendation/output_bias_drift.rs",
        "pub fn detect_output_bias_drift",
    ),
    (
        "src/analysis/recommendation/output_bias_drift.rs",
        "pub fn output_bias_drift_to_coordinated_candidates",
    ),
    (
        "src/analysis/recommendation/output_bias_drift.rs",
        "fn summarise_positive_support",
    ),
    (
        "src/analysis/recommendation/multi_hop.rs",
        "pub fn detect_multi_hop_candidates",
    ),
    (
        "src/analysis/recommendation/multi_hop.rs",
        "fn compute_mean_abs_error",
    ),
    (
        "src/analysis/recommendation/multi_hop.rs",
        "fn find_three_hop_extensions",
    ),
    (
        "src/analysis/recommendation/gradient_discovery.rs",
        "pub fn detect_gradient_candidates",
    ),
    (
        "src/analysis/recommendation/gradient_discovery.rs",
        "pub fn compute_synapse_gradient",
    ),
    (
        "src/analysis/recommendation/sample_weighted.rs",
        "pub fn compute_sample_weights",
    ),
    (
        "src/analysis/recommendation/sample_weighted.rs",
        "pub fn stratify_samples",
    ),
    (
        "src/analysis/recommendation/output_competition.rs",
        "pub fn detect_output_competition",
    ),
    (
        "src/analysis/recommendation/output_competition.rs",
        "fn co_activation",
    ),
    (
        "src/analysis/recommendation/activation_recommendation.rs",
        "pub fn recommend_activation_function",
    ),
    (
        "src/analysis/discovery_dispatch.rs",
        "pub fn detect_discovery_modules_parallel",
    ),
]; // Issue #1942: symbols, not line numbers — a symbol survives the refactor.

#[test]
fn the_recommendation_core_outcome_cites_symbols_that_still_exist() {
    let doc = read(RECORD);
    let outcome = section(&doc, "### recommendation core (Issue #2108)");

    for (file, declaration) in TRACED_SYMBOLS {
        let symbol = declaration
            .rsplit(' ')
            .next()
            .expect("declaration names a symbol");
        assert!(
            outcome.contains(symbol),
            "the `recommendation core` outcome must name `{symbol}` — an outcome that cites no \
             guard, bound or dispatch site is a claim, not a sweep"
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
fn the_recommendation_core_outcome_links_its_filed_findings() {
    let doc = read(RECORD);
    let outcome = section(&doc, "### recommendation core (Issue #2108)");
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

/// The acceptance criterion this section exists to answer: a ranking-integrity
/// verdict per detector, not a single sentence about the section as a whole.
#[test]
fn the_recommendation_core_outcome_records_a_ranking_verdict_per_detector() {
    let doc = read(RECORD);
    let outcome = section(&doc, "### recommendation core (Issue #2108)");

    assert!(
        outcome.contains("Ranking integrity"),
        "the outcome must carry the ranking-integrity verdict table — it is the primary lens \
         Issue #2108 asked this section to apply"
    );

    // Every file with a ranking comparator needs its verdict named. `mod.rs`
    // has none, so it is excluded by the same rule the float table uses.
    for file in RECOMMENDATION_CORE_FILES {
        if !production_source(file).lines().any(is_comparator_site) {
            continue;
        }
        assert!(
            outcome.contains(&citation_prefix(file)),
            "{file} ends in a ranking sort the host reads head-first, so the outcome must state \
             whether a crafted input can reach rank 1 there and by what path"
        );
    }
}

// =============================================================================
// Behaviour contract — the reachability the findings rest on, and the guards
// the `clean` verdicts rest on
// =============================================================================

fn rec(uuid: &str, obs_index: u32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![error],
    }
}

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

/// #2181, first half: a NaN correlation is reachable from records every FFI
/// finitude gate accepts, and `fan_in.rs`'s threshold filter keeps it.
///
/// `pearson_correlation`'s `denom < f32::EPSILON` guard is a skip-on-comparison
/// test a NaN loses, and `f32::clamp` propagates a NaN, so an activation swing
/// that overflows the `f32` covariance accumulator leaves the function as a
/// NaN. Issues #2134 / #2135 reject `Infinity` and `NaN` **on the wire**; they
/// do not reject a finite magnitude that overflows later.
///
/// When #2181 is fixed this test must fail — that is the signal the
/// `fan_in.rs` row needs re-sweeping rather than merely re-reading.
#[test]
fn a_finite_record_set_still_drives_the_fan_in_correlation_to_nan() {
    let activations: Vec<f32> = (0..30_u32)
        .map(|i| if i.is_multiple_of(2) { 2.0e30 } else { -2.0e30 })
        .collect();
    let errors: Vec<f32> = (0..30_u32)
        .map(|i| if i.is_multiple_of(3) { 1.0e10 } else { -5.0e9 })
        .collect();

    assert!(
        activations
            .iter()
            .chain(errors.iter())
            .all(|v| v.is_finite()),
        "the trigger must use only values the FFI boundary accepts, or it proves nothing"
    );

    let corr = pearson_correlation(&activations, &errors);
    assert!(
        corr.is_nan(),
        "the f32 covariance accumulator must still overflow to a NaN correlation, got {corr}"
    );

    // The filter `detect_fan_in_candidates` applies to this value, spelled the
    // way the production code spells it. A NaN loses the `<`, so the input is
    // kept and its NaN key reaches the non-total comparator.
    const INPUT_ERROR_CORRELATION_THRESHOLD: f32 = 0.3;
    assert!(
        corr.abs()
            .partial_cmp(&INPUT_ERROR_CORRELATION_THRESHOLD)
            .is_none(),
        "the threshold filter must still fail open on a NaN correlation — a fail-closed filter \
         would close #2181 at the source"
    );
}

/// The half of the #2181 / #2182 reachability claim that lives at the shipped
/// entry point (CONTRIBUTING.md § Guard Wiring at the Shipped Entry Point): the
/// magnitudes both findings are triggered with are **accepted** by the FFI
/// boundary, because Issues #2134 / #2135 reject `Infinity` and `NaN` on the
/// wire and nothing else. A test that only built `DiscoverRecord`s in-process
/// could not tell a reachable trigger from an unreachable one.
#[test]
fn the_ffi_boundary_still_accepts_the_magnitudes_both_findings_are_triggered_with() {
    for magnitude in ["2e30", "-2e30", "1e10", "1e38"] {
        let payload = format!(
            r#"{{"neuron_uuid": "input-0", "activation": {magnitude}, "errors": [{magnitude}]}}"#
        );
        let parsed = serde_json::from_str::<NeuronData>(&payload)
            .unwrap_or_else(|e| panic!("{magnitude} is finite and must deserialise: {e}"));
        assert!(
            parsed.activation.is_finite() && parsed.errors[0].is_finite(),
            "{magnitude} must survive the boundary as a finite value"
        );
    }

    // The boundary that *is* closed, for contrast: a magnitude that saturates
    // to infinity on the narrowing cast is refused (Issue #2134), which is why
    // the triggers above have to manufacture their infinity downstream instead.
    let error = serde_json::from_str::<NeuronData>(
        r#"{"neuron_uuid": "input-0", "activation": 1e39, "errors": [0.1]}"#,
    )
    .expect_err("a saturating activation must not deserialise");
    assert!(
        error.to_string().contains("finite"),
        "the refusal must name finitude as the fault, got: {error}"
    );
}

/// #2181, second half: the resulting order is *unspecified*, and the caller
/// picks it. The same records in two neuron orderings must not produce
/// different candidate sets — today they do.
///
/// Listing the poisoned inputs after the honest ones returns every genuine
/// candidate; listing them first returns none, because a comparator that
/// reports `Equal` for every NaN-vs-finite pair gives `sort_by` no total order
/// to preserve and `truncate(MAX_INPUTS_PER_TARGET)` then keeps whatever
/// survived.
#[test]
fn fan_in_candidates_still_depend_on_the_order_the_caller_lists_neurons_in() {
    const SAMPLES: u32 = 60;
    const HONEST_INPUTS: usize = 15;
    const POISON_INPUTS: usize = 20;

    fn target_error(j: u32) -> f32 {
        if j.is_multiple_of(3) { 1.0e10 } else { -5.0e9 }
    }

    fn build(poison_first: bool) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
        let honest = |neurons: &mut Vec<NeuronJson>| {
            for i in 0..HONEST_INPUTS {
                neurons.push(neuron(&format!("honest-{i}"), "input"));
            }
        };
        let poison = |neurons: &mut Vec<NeuronJson>| {
            for i in 0..POISON_INPUTS {
                neurons.push(neuron(&format!("poison-{i}"), "input"));
            }
        };

        let mut neurons = Vec::new();
        if poison_first {
            poison(&mut neurons);
            honest(&mut neurons);
        } else {
            honest(&mut neurons);
            poison(&mut neurons);
        }
        neurons.push(neuron("output-1", "output"));
        let input = neurons.len() - 1;

        let mut records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
            "output-1".to_string(),
            (0..SAMPLES)
                .map(|j| rec("output-1", j, 0.5, target_error(j)))
                .collect(),
        )];
        for i in 0..HONEST_INPUTS {
            let uuid = format!("honest-{i}");
            let phase = u32::try_from(i).expect("15 fits in u32");
            records.push((
                uuid.clone(),
                (0..SAMPLES)
                    .map(|j| {
                        let base = if (j + phase).is_multiple_of(3) {
                            1.0
                        } else {
                            -0.5
                        };
                        let jitter =
                            f32::from(u16::try_from(j).expect("SAMPLES fits in u16")) * 0.001;
                        rec(&uuid, j, base + jitter, 0.0)
                    })
                    .collect(),
            ));
        }
        for i in 0..POISON_INPUTS {
            let uuid = format!("poison-{i}");
            records.push((
                uuid.clone(),
                (0..SAMPLES)
                    .map(|j| {
                        let swing = if j.is_multiple_of(2) { 2.0e30 } else { -2.0e30 };
                        rec(&uuid, j, swing, 0.0)
                    })
                    .collect(),
            ));
        }

        (
            CreatureJson {
                neurons,
                synapses: Vec::new(),
                input,
                output: 1,
            },
            records,
        )
    }

    let (creature, records) = build(false);
    let honest_first = detect_fan_in_candidates(&creature, &records);
    let (creature, records) = build(true);
    let poison_first = detect_fan_in_candidates(&creature, &records);

    assert!(
        !honest_first.is_empty(),
        "the honest inputs must still produce fan-in candidates, or the trigger proves nothing"
    );
    assert_ne!(
        honest_first.len(),
        poison_first.len(),
        "#2181 says the two orderings still disagree ({} vs {}) — if they now agree, the \
         comparator has been made total and the `fan_in.rs` row must be re-swept to `clean`",
        honest_first.len(),
        poison_first.len()
    );
    assert!(
        poison_first.is_empty(),
        "listing the NaN-correlation inputs first must still empty the \
         MAX_INPUTS_PER_TARGET window, got {} candidates",
        poison_first.len()
    );
}

/// #2182, the `fan_in.rs` path the first pass of this sweep missed: the
/// *improvement* comparator can see a non-finite key after all, and a crafted
/// creature lands at **rank 0** with `estimated_improvement = +inf`.
///
/// The first pass ruled out a NaN improvement — correctly, because both
/// estimators end in `.max(0.0)` and `f32::max` returns the non-NaN operand —
/// and stopped there. It never applied the `+inf`-from-an-`f32`-accumulator
/// reasoning it applied to the four other detectors. Two sites manufacture one
/// from finite records:
///
/// * `fan_in.rs::compute_least_squares_improvement` accumulates
///   `original_sse` in `f32`; sixty errors near `1e19` overflow it while the
///   residual stays small, so `(inf - finite).max(0.0)` is `+inf`, not `0.0`.
/// * `fan_in.rs::compute_two_input_regression` accumulates in `f64` and then
///   narrows with `improvement as f32`, which saturates to `+inf` above
///   `f32::MAX`.
///
/// Both feed `evaluate_fan_in_pair`, whose four gates are all fail-open for
/// `+inf` — `inf <= 0.0` is false, and `inf < inf * 1.05` is false — so the
/// candidate is emitted and the descending sort puts it first.
///
/// Every recorded value is finite, so the FFI gates of Issues #2134 / #2135 do
/// not apply. When #2182 grows a finitude gate on `estimated_improvement` this
/// test must fail, which is the signal the `fan_in.rs` row needs re-sweeping.
#[test]
fn a_finite_record_set_still_ranks_a_fan_in_candidate_at_infinity() {
    const SAMPLES: u32 = 60;
    /// Large enough that `error²` summed over `SAMPLES` overflows `f32`, small
    /// enough that every value itself is finite and crosses the FFI boundary.
    const ACTIVATION_BASE: f32 = 3.3e8;
    /// Makes the target error exactly proportional to input `a`, so the
    /// single-input residual is ~0 while the uncentred `Σ error²` is `+inf`.
    const ERROR_PER_ACTIVATION: f32 = 3.03e10;

    // Two spreads: `a` carries the first, `b` carries the first plus a second
    // independent one, so `corr(a, b)` lands between the 0.3 filter floor and
    // the 0.8 mutual-correlation ceiling instead of outside both.
    let spread_a = |j: u32| f32::from(i16::try_from(j % 5).expect("j % 5 fits")) - 2.0;
    let spread_b = |j: u32| f32::from(i16::try_from(j % 7).expect("j % 7 fits")) - 3.0;
    let act_a = |j: u32| ACTIVATION_BASE + 1.0e6 * spread_a(j);
    let act_b = |j: u32| ACTIVATION_BASE + 1.0e6 * (spread_a(j) + spread_b(j));
    let target_error = |j: u32| act_a(j) * ERROR_PER_ACTIVATION;

    let creature = CreatureJson {
        neurons: vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron("output-1", "output"),
        ],
        synapses: Vec::new(),
        input: 2,
        output: 1,
    };
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-a".to_string(),
            (0..SAMPLES)
                .map(|j| rec("input-a", j, act_a(j), 0.0))
                .collect(),
        ),
        (
            "input-b".to_string(),
            (0..SAMPLES)
                .map(|j| rec("input-b", j, act_b(j), 0.0))
                .collect(),
        ),
        (
            "output-1".to_string(),
            (0..SAMPLES)
                .map(|j| rec("output-1", j, 0.5, target_error(j)))
                .collect(),
        ),
    ];

    // Issue #1799: the claim is "reachable from finite records". If the trigger
    // itself smuggles a non-finite value in, the assertion below proves nothing.
    assert!(
        records
            .iter()
            .flat_map(|(_, rs)| rs.iter())
            .all(|r| r.activation.is_finite() && r.errors.iter().all(|e| e.is_finite())),
        "every recorded value must be finite, or the trigger bypasses the FFI gate instead of \
         defeating it"
    );

    let candidates = detect_fan_in_candidates(&creature, &records);
    let first = candidates
        .first()
        .expect("the crafted pair must still produce a fan-in candidate");
    assert!(
        !first.estimated_improvement.is_finite(),
        "#2182 says rank 0 still carries a non-finite estimated_improvement — got {}; if it is \
         finite now, the improvement is gated and the `fan_in.rs` row must be re-swept",
        first.estimated_improvement
    );
}

/// The `sample_weighted.rs` row: every per-record error is laundered through
/// `is_finite` to `0.0` **before** anything is compared, so neither the median
/// select nor the ratio can see a NaN, and an overflowed weight total
/// renormalises to zero rather than to an infinity.
#[test]
fn sample_weighted_launders_every_unusable_error_before_it_ranks() {
    let hostile: Vec<DiscoverRecord> = (0..20)
        .map(|i| {
            let error = match i % 4 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                _ => 0.4,
            };
            rec("n", i, 0.5, error)
        })
        .collect();

    let weights = compute_sample_weights(&hostile);
    assert_eq!(weights.len(), hostile.len(), "one weight per record");
    for (index, weight) in weights.iter().enumerate() {
        assert!(
            weight.is_finite() && *weight >= 0.0,
            "weight {index} must be finite and non-negative, got {weight}"
        );
    }

    let stratified = stratify_samples(&hostile);
    for (name, value) in [
        ("easy_mean_error", stratified.easy_mean_error),
        ("hard_mean_error", stratified.hard_mean_error),
        ("hard_to_easy_ratio", stratified.hard_to_easy_ratio),
    ] {
        assert!(value.is_finite(), "{name} must be finite, got {value}");
    }

    // An f32 total that overflows must renormalise every weight to zero, not
    // produce an infinity — the reason this detector cannot be forced to the
    // head of its own list.
    let overflowing: Vec<DiscoverRecord> = (0..20).map(|i| rec("n", i, 0.5, 1.0e38)).collect();
    let weights = compute_sample_weights(&overflowing);
    for weight in &weights {
        assert!(
            weight.is_finite(),
            "an overflowed weight total must still yield finite weights, got {weight}"
        );
    }
}

/// The `gradient_discovery.rs` row, clean half: the explicit
/// `!mean_gradient.is_finite()` rejection closes the NaN path that
/// `output_bias_drift.rs` leaves open. (The `+inf` product past this gate is
/// the open half, filed as #2182.)
#[test]
fn the_synapse_gradient_rejects_every_unusable_mean_before_returning_it() {
    let target: Vec<DiscoverRecord> = (0..20).map(|i| rec("t", i, 0.5, 0.2)).collect();

    let nan_source: Vec<DiscoverRecord> = (0..20).map(|i| rec("s", i, f32::NAN, 0.0)).collect();
    assert_eq!(
        compute_synapse_gradient(&nan_source, &target),
        None,
        "every sample is skipped by the activation finitude test, so no gradient survives"
    );

    let overflowing: Vec<DiscoverRecord> = (0..20).map(|i| rec("s", i, 1.0e38, 0.0)).collect();
    let big_errors: Vec<DiscoverRecord> = (0..20).map(|i| rec("t", i, 0.5, 1.0e38)).collect();
    assert_eq!(
        compute_synapse_gradient(&overflowing, &big_errors),
        None,
        "a mean gradient that overflows to +inf must be rejected, not returned"
    );

    let usable: Vec<DiscoverRecord> = (0..20).map(|i| rec("s", i, 0.8, 0.0)).collect();
    let gradient = compute_synapse_gradient(&usable, &target)
        .expect("a well-conditioned pair must still yield a gradient");
    assert!(
        gradient.is_finite(),
        "the finite path must still return a usable gradient, got {gradient}"
    );
}

/// The `activation_recommendation.rs` row: the whole score space is
/// compile-time literals, so no record — however hostile, as long as it is
/// finite — can move the ranked value off that space.
#[test]
fn the_activation_recommender_ranks_over_a_constant_score_space() {
    // Issue #1799: the recommender legitimately returns `None` when the best
    // activation beats the current one by less than `MIN_IMPROVEMENT_THRESHOLD`,
    // so the loop below skips such a magnitude — which means it would pass with
    // zero assertions executed if every magnitude were skipped. Count the ones
    // that actually ranked and pin that count after the loop.
    let mut ranked = 0_usize;
    for magnitude in [1.0e-30_f32, 1.0, 1.0e30, 3.0e38] {
        let records: Vec<DiscoverRecord> = (0..60_u32)
            .map(|i| {
                let activation = if i.is_multiple_of(2) {
                    magnitude
                } else {
                    -magnitude
                };
                rec("hidden-1", i, activation, 0.1)
            })
            .collect();

        let Some(recommendation) = recommend_activation_function(&records, "IDENTITY") else {
            continue;
        };
        ranked += 1;
        assert!(
            recommendation.expected_improvement.is_finite()
                && recommendation.expected_improvement > 0.0,
            "magnitude {magnitude}: expected_improvement must stay on the constant score space, \
             got {}",
            recommendation.expected_improvement
        );
        assert!(
            (0.0..=1.0).contains(&recommendation.confidence),
            "magnitude {magnitude}: confidence must stay inside its clamp, got {}",
            recommendation.confidence
        );
    }
    assert!(
        ranked > 0,
        "no magnitude produced a recommendation, so every assertion above was skipped — the \
         `clean` verdict for activation_recommendation.rs rests on a ranked value being seen"
    );
}
