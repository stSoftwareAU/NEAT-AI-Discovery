//! Contract tests for the chunk 8b `scoring` sweep (Issue #2107).
//!
//! `tests/issue_2103_chunk_08b_ledger_scaffold.rs` gates the record's *shape* —
//! one row per in-scope file, one section per audit sub-issue, the two finding
//! tables and their per-section markers. This file gates the one section Issue
//! #2107 owns, and it does so against the source it claims to have swept:
//!
//! * the 10 rows this sub-issue owns are present and none still reads
//!   `pending`, each with a reason a later reader can check;
//! * every capacity site (`with_capacity` / `vec![_; n]` / `reserve`) and every
//!   float-comparison site in the production half of those files is cited in
//!   the matching finding table, so a new one cannot land unrecorded;
//! * the symbols the outcome traces still exist;
//! * the outcome records a negative result — no finding was filed.
//!
//! The second half of the file is the other kind of check the record needs: the
//! sweep's `clean` verdicts rest on a handful of *behaviours*, not on prose, and
//! those are exercised here against the real functions. A verdict backed only by
//! a paragraph rots silently; one backed by a test fails loudly.

use std::path::PathBuf;

use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::scoring::calibration_correction::{
    CHANGE_TYPE_ADD_NEURONS, CalibrationCorrection, FailureCacheEntry, MIN_CALIBRATION_CORRECTION,
    NEUTRAL_CORRECTION,
};
use neat_ai_discovery::analysis::scoring::cross_validation::{
    CrossValidationConfig, compute_cross_validation_score,
};
use neat_ai_discovery::analysis::scoring::error_distribution::ErrorDistribution;
use neat_ai_discovery::analysis::scoring::sample_creature_disconnect::detect_disconnect;
use neat_ai_discovery::analysis::scoring::weights::{
    calculate_optimal_outgoing_weight, clamp_weight_update_delta,
    coordinated_structural_activation_delta,
};
use neat_ai_discovery::ffi_types::CreatureJson;

/// The chunk 8b prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md";

/// The 10 files Issue #2107 swept, in record order.
const SCORING_FILES: [&str; 10] = [
    "src/analysis/scoring/mod.rs",
    "src/analysis/scoring/calibration_correction.rs",
    "src/analysis/scoring/confidence.rs",
    "src/analysis/scoring/cross_validation.rs",
    "src/analysis/scoring/error_distribution.rs",
    "src/analysis/scoring/sample_creature_disconnect.rs",
    "src/analysis/scoring/weights/mod.rs",
    "src/analysis/scoring/weights/adjustment.rs",
    "src/analysis/scoring/weights/calculation.rs",
    "src/analysis/scoring/weights/normalisation.rs",
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

fn scoring_rows() -> Vec<(String, String)> {
    let doc = read(RECORD);
    file_rows(section(&doc, "### scoring"))
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
// Record contract — the `scoring` section describes the code it swept
// =============================================================================

#[test]
fn the_scoring_section_owns_exactly_the_files_issue_2107_swept() {
    let rows = scoring_rows();
    let paths: Vec<&str> = rows.iter().map(|(path, _)| path.as_str()).collect();
    assert_eq!(
        paths, SCORING_FILES,
        "the `scoring` section must carry one row per file Issue #2107 swept, in record order \
         — a row under the wrong section has no owner and is swept twice or never"
    );
}

#[test]
fn every_scoring_row_is_swept_with_a_reason() {
    for (path, outcome) in scoring_rows() {
        assert!(
            !outcome.contains("pending"),
            "{path} was swept by Issue #2107, so its outcome must not read `pending`: {outcome}"
        );
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
    let region = marker_region(section(&doc, "## Capacity-from-input sites"), "scoring");

    for file in SCORING_FILES {
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
    let region = marker_region(section(&doc, "## Float comparison sites"), "scoring");

    for file in SCORING_FILES {
        if production_source(file).lines().any(is_comparator_site) {
            assert!(
                region.contains(&citation_prefix(file)),
                "{file} ranks with a float comparator, so the float-comparison table must carry \
                 a row recording where the value comes from and what happens when it is NaN"
            );
        }
    }
}

/// The two swept files that compare no floats at all, because they contain no
/// executable code: `scoring/mod.rs` is six `pub mod` lines, and the production
/// half of `weights/mod.rs` is constants and re-exports. Every other swept file
/// must appear in the float table.
const DECLARATION_ONLY_FILES: [&str; 2] = [
    "src/analysis/scoring/mod.rs",
    "src/analysis/scoring/weights/mod.rs",
];

/// `is_comparator_site` above only catches *ranking* comparators, which is a
/// narrower thing than "compares a float" — the table is mostly `<` / `>` /
/// `==` guards, which no lexical rule separates from integer comparisons
/// reliably. So the coverage of the acceptance criterion is asserted the other
/// way round: every swept file is cited unless it demonstrably has no
/// executable code to compare anything in.
#[test]
fn every_scoring_file_that_executes_anything_is_cited_in_the_float_table() {
    let doc = read(RECORD);
    let region = marker_region(section(&doc, "## Float comparison sites"), "scoring");

    for file in SCORING_FILES {
        let declaration_only = DECLARATION_ONLY_FILES.contains(&file);
        if declaration_only {
            let source = production_source(file);
            // Strip each line's comment before looking for a control-flow
            // keyword — prose says "responsible for computing…" and would
            // otherwise read as a loop.
            let executable = source
                .lines()
                .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
                .any(|code| {
                    ["if ", "for ", "while ", "match "]
                        .iter()
                        .any(|keyword| code.contains(keyword))
                });
            assert!(
                !executable,
                "{file} is listed as declaration-only, but its production half now carries a \
                 branch or a loop — it can compare floats, so it needs a float-table row"
            );
            continue;
        }
        assert!(
            region.contains(&citation_prefix(file)),
            "{file} carries executable code, so the float-comparison table must record what its \
             guards do when the value they test is NaN — a swept file with no row leaves that \
             unanswered"
        );
    }
}

/// Every symbol the `scoring` outcome claims to have traced, paired with the
/// file that must still declare it. An outcome citing a symbol that no longer
/// exists is describing code that has moved or gone.
const TRACED_SYMBOLS: [(&str, &str); 11] = [
    (
        "src/analysis/scoring/cross_validation.rs",
        "pub fn compute_cross_validation_score",
    ),
    (
        "src/analysis/neuron/evaluation.rs",
        "fn apply_cross_validation_penalty",
    ),
    (
        "src/analysis/scoring/sample_creature_disconnect.rs",
        "pub fn detect_disconnect",
    ),
    (
        "src/analysis/scoring/calibration_correction.rs",
        "pub fn from_failure_cache",
    ),
    (
        "src/analysis/scoring/error_distribution.rs",
        "fn detect_modes_histogram",
    ),
    (
        "src/analysis/scoring/error_distribution.rs",
        "fn compute_percentiles",
    ),
    ("src/ffi_types/mod.rs", "fn deserialise_synapse_weight"),
    (
        "src/analysis/scoring/weights/calculation.rs",
        "fn compute_outgoing_weight",
    ),
    (
        "src/analysis/scoring/weights/adjustment.rs",
        "pub fn clamp_weight_update_delta",
    ),
    (
        "src/analysis/scoring/weights/adjustment.rs",
        "pub fn coordinated_structural_activation_delta",
    ),
    ("src/analysis/scoring/confidence.rs", "fn t_critical_95"),
]; // Issue #1942: symbols, not line numbers — a symbol survives the refactor.

#[test]
fn the_scoring_outcome_cites_symbols_that_still_exist() {
    let doc = read(RECORD);
    let outcome = section(&doc, "### scoring (Issue #2107)");

    for (file, declaration) in TRACED_SYMBOLS {
        let symbol = declaration
            .rsplit(' ')
            .next()
            .expect("declaration names a symbol");
        assert!(
            outcome.contains(symbol),
            "the `scoring` outcome must name `{symbol}` — an outcome that cites no guard, bound \
             or dispatch site is a claim, not a sweep"
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
fn the_scoring_outcome_is_a_negative_result() {
    let doc = read(RECORD);
    let outcome = section(&doc, "### scoring (Issue #2107)");
    assert!(
        outcome.contains("Negative result"),
        "the outcome must record whether a finding was filed — a missing verdict is an \
         incomplete sweep"
    );
    assert!(
        outcome.contains("no finding filed"),
        "Issue #2107 is a negative-result sweep, so the outcome must state explicitly that no \
         finding was filed"
    );
}

#[test]
fn the_scoring_negative_result_is_linked_in_issues_filed() {
    let doc = read(RECORD);
    let filed = section(&doc, "## Issues filed");
    assert!(
        filed.contains("negative-result") && filed.contains("`scoring` sweep"),
        "`## Issues filed` must record the negative-result status of the `scoring` sweep"
    );
}

// =============================================================================
// Behaviour contract — the guards the `clean` verdicts rest on
// =============================================================================

fn sample(activation: f32, avg_error: f32) -> HelpfulSample {
    HelpfulSample {
        activation,
        avg_error,
        target_value: None,
        target_activation: None,
    }
}

fn failure(change_type: &str, expected: f32, actual: f32) -> FailureCacheEntry {
    FailureCacheEntry {
        change_type: change_type.to_string(),
        expected_error_reduction: expected,
        actual_error_reduction: actual,
        target_squash: None,
        variant_key: None,
        target_uuid: None,
        improved_count: None,
        total_count: None,
        age_epochs: None,
    }
}

/// The `sample_creature_disconnect.rs` row: the divisor, the numerator and the
/// finitude are all checked **before** the division, so no corrupt failure-cache
/// counter reaches the ratio.
#[test]
fn the_disconnect_detector_rejects_every_unusable_counter_before_dividing() {
    assert!(
        !detect_disconnect(0, 0, -1.0),
        "a zero total must be rejected before the division, not divided by"
    );
    assert!(
        !detect_disconnect(7, 3, -1.0),
        "improved > total is a corrupt counter pair and must not produce a ratio above 1.0"
    );
    assert!(
        !detect_disconnect(100, 100, f32::NAN),
        "a non-finite actual reduction cannot be compared and must return false"
    );
    assert!(
        !detect_disconnect(100, 100, f32::INFINITY),
        "an infinite actual reduction must be rejected on the same finitude test"
    );
    assert!(
        detect_disconnect(100, 100, -0.5),
        "the detector must still fire for the pattern it exists to catch"
    );
}

/// The `calibration_correction.rs` row: a zero (or negative-zero) divisor and a
/// non-finite ratio are both skipped, and every surviving correction is inside
/// the documented clamp — so an untrusted `failureCache` cannot move the
/// correction outside `[MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION]`.
#[test]
fn a_hostile_failure_cache_cannot_move_the_correction_outside_its_clamp() {
    let cache = vec![
        // Undefined ratio — skipped by the `expected == 0.0` test.
        failure(CHANGE_TYPE_ADD_NEURONS, 0.0, 1.0),
        // Negative zero compares equal to zero, so it takes the same path.
        failure(CHANGE_TYPE_ADD_NEURONS, -0.0, 1.0),
        // Enormous-but-finite operands: the quotient overflows to `+inf` and is
        // dropped by the explicit `is_finite` test.
        failure(CHANGE_TYPE_ADD_NEURONS, f32::MIN_POSITIVE, f32::MAX),
        // A genuine ~1000x over-prediction, which must survive and be floored.
        failure(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.000_000_1),
    ];

    let correction = CalibrationCorrection::from_failure_cache(&cache);
    let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);

    assert!(
        value.is_finite(),
        "a correction built from a hostile cache must still be finite, got {value}"
    );
    assert!(
        (MIN_CALIBRATION_CORRECTION..=NEUTRAL_CORRECTION).contains(&value),
        "the correction must stay inside [{MIN_CALIBRATION_CORRECTION}, {NEUTRAL_CORRECTION}], \
         got {value}"
    );
    assert!(
        (value - MIN_CALIBRATION_CORRECTION).abs() < 1e-9,
        "the one usable entry is a ~1000x over-prediction, so the EWMA must clamp to the floor \
         rather than to a value inflated by the skipped entries, got {value}"
    );
    assert_eq!(
        correction.get_correction("never-seen"),
        NEUTRAL_CORRECTION,
        "an unseen change type must resolve to the neutral correction"
    );
}

/// The `error_distribution.rs` row: `from_samples` filters non-finite errors
/// before the `total_cmp` sort, so no NaN can reach a percentile — which is
/// what stops a NaN sorting to the head or the tail of the order.
#[test]
fn the_error_distribution_never_reports_a_non_finite_percentile() {
    let samples = vec![
        sample(1.0, f32::NAN),
        sample(1.0, f32::INFINITY),
        sample(1.0, f32::NEG_INFINITY),
        sample(1.0, 0.1),
        sample(1.0, 0.2),
        sample(1.0, 0.3),
        sample(1.0, 9.0),
    ];

    let distribution = ErrorDistribution::from_samples(&samples)
        .expect("four finite errors are enough to build a distribution");

    assert_eq!(
        distribution.sample_count, 4,
        "the three non-finite errors must be filtered out before any statistic is computed"
    );
    for (index, value) in distribution.percentiles.iter().enumerate() {
        assert!(
            value.is_finite(),
            "percentile {index} must be finite, got {value}"
        );
    }
    for (name, value) in [
        ("mean", distribution.mean),
        ("std_dev", distribution.std_dev),
        ("variance", distribution.variance),
        ("skewness", distribution.skewness),
        ("kurtosis", distribution.kurtosis),
        ("min", distribution.min),
        ("max", distribution.max),
        ("iqr", distribution.iqr),
    ] {
        assert!(value.is_finite(), "{name} must be finite, got {value}");
    }

    assert!(
        ErrorDistribution::from_samples(&[sample(1.0, f32::NAN)]).is_none(),
        "a sample set with no finite error must yield no distribution at all"
    );
}

/// The `error_distribution.rs` degenerate row: an all-identical error set drives
/// `std_dev` below the `1e-10` guard, which must take the documented default
/// arm rather than divide by it.
#[test]
fn a_zero_variance_error_set_takes_the_documented_degenerate_defaults() {
    let samples: Vec<HelpfulSample> = (0..20).map(|_| sample(1.0, 0.25)).collect();
    let distribution = ErrorDistribution::from_samples(&samples)
        .expect("20 identical errors build a distribution");

    assert_eq!(
        distribution.skewness, 0.0,
        "the std_dev guard must emit a zero skewness, not a division by ~0"
    );
    assert_eq!(
        distribution.kurtosis, 3.0,
        "the std_dev guard must emit the Gaussian kurtosis default"
    );
}

/// The `weights/calculation.rs` row: the finitude test sits **between** the
/// divisor guard and the clamp, so a non-finite least-squares sum is rejected
/// rather than clamped into a plausible-looking weight.
#[test]
fn a_non_finite_least_squares_sum_is_rejected_before_the_clamp() {
    for (error_activation, activation_sq) in [
        (f32::NAN, 1.0),
        (1.0, f32::NAN),
        (f32::INFINITY, f32::INFINITY),
        (1.0, 0.0),
    ] {
        assert_eq!(
            calculate_optimal_outgoing_weight(error_activation, activation_sq, 1.0),
            None,
            "sums ({error_activation}, {activation_sq}) must yield no weight — a clamp would \
             turn a non-finite quotient into a weight that looks valid"
        );
    }

    let weight = calculate_optimal_outgoing_weight(1.0, 1.0, 1.0)
        .expect("a well-conditioned pair of sums must still yield a weight");
    assert!(
        weight.is_finite() && weight.abs() > 0.0,
        "the finite path must still return a usable weight, got {weight}"
    );
}

/// The composition the `weights/adjustment.rs` verdict actually rests on: the
/// helpers would happily return `Some(non-finite)`, and the only reason they
/// never see one is the gate *upstream* of them. A unit test of either half
/// would stay green with that gate gone, so this drives the wire type a
/// creature is deserialised from and asserts the payload is refused —
/// CONTRIBUTING.md § Guard Wiring at the Shipped Entry Point.
///
/// `1e39` is the payload that matters: it is an ordinary `f64` in JSON, so
/// serde raises nothing, and it saturates to `f32::INFINITY` on the cast down.
#[test]
fn a_creature_carrying_an_overflowing_synapse_weight_is_refused_at_the_boundary() {
    let creature = r#"{
        "neurons": [
            {"uuid": "n-0", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "n-1", "type": "output", "squash": "IDENTITY", "bias": 0.0}
        ],
        "synapses": [{"fromUUID": "n-0", "toUUID": "n-1", "weight": 1e39}],
        "input": 1,
        "output": 1
    }"#;

    let error = serde_json::from_str::<CreatureJson>(creature)
        .expect_err("a synapse weight that saturates to infinity must not deserialise");
    let message = error.to_string();
    assert!(
        message.contains("synapse weight must be finite"),
        "the refusal must name the weight as the fault so the caller can correct it, got: \
         {message}"
    );

    let finite = creature.replace("1e39", "0.5");
    let parsed = serde_json::from_str::<CreatureJson>(&finite)
        .expect("the same creature with a finite weight must still deserialise");
    assert!(
        parsed.synapses[0].weight.is_finite(),
        "every weight that survives the boundary is finite, which is what lets \
         clamp_weight_update_delta skip a finitude test of its own"
    );
}

/// The `weights/adjustment.rs` row: both helpers are only safe because their
/// operands are finite — the FFI gate of Issue #2132 upstream, and
/// `compute_outgoing_weight`'s finitude test. This test locks in the half that
/// lives here: given finite operands, neither helper emits a non-finite value.
#[test]
fn the_weight_adjustment_helpers_emit_only_finite_values_for_finite_operands() {
    for old_weight in [0.0_f32, 0.005, -0.005, 1.0, -1.0, f32::MAX, f32::MIN] {
        for proposed in [0.004_f32, -0.004, 1.0, -1.0, f32::MAX, f32::MIN] {
            if let Some((new_weight, delta)) = clamp_weight_update_delta(old_weight, proposed) {
                assert!(
                    new_weight.is_finite() && delta.is_finite(),
                    "clamp_weight_update_delta({old_weight}, {proposed}) returned a non-finite \
                     pair ({new_weight}, {delta}) from finite operands"
                );
            }
        }
    }

    for noisy_weight in [0.5_f32, -0.5, f32::MAX, f32::MIN] {
        let delta = coordinated_structural_activation_delta(0.25, 0.75, noisy_weight, 0.5);
        if let Some(delta) = delta {
            assert!(
                delta.is_finite(),
                "coordinated_structural_activation_delta with a finite noisy weight \
                 {noisy_weight} returned {delta}"
            );
        }
    }

    assert_eq!(
        coordinated_structural_activation_delta(0.25, 0.75, 0.0, 0.5),
        None,
        "a zero noisy weight has no scale to divide by and must be refused"
    );
}

/// The `cross_validation.rs` capacity row: `fold_count` is the compile-time `5`
/// of the default config, and the two preconditions bound the reservation for
/// any other caller.
#[test]
fn cross_validation_reserves_only_what_its_preconditions_allow() {
    let config = CrossValidationConfig::default();
    assert_eq!(
        config.fold_count, 5,
        "the only production constructor must keep supplying the compile-time fold count"
    );

    // `u8 -> f32` is the lossless `From` impl, so no cast lint is needed here.
    let samples: Vec<HelpfulSample> = (0..100u8)
        .map(|i| sample(f32::from(i) * 0.01, f32::from(i) * 0.02))
        .collect();
    let result = compute_cross_validation_score(&samples, &config)
        .expect("100 samples over 5 folds clears the 15-sample minimum");
    assert_eq!(
        result.fold_results.len(),
        config.fold_count,
        "the reservation and the fold loop must agree on the fold count"
    );
    assert!(
        result.brittleness_penalty.is_finite(),
        "the penalty is a clamped ratio and must be finite, got {}",
        result.brittleness_penalty
    );

    let huge = CrossValidationConfig {
        fold_count: usize::MAX,
        ..CrossValidationConfig::default()
    };
    assert!(
        compute_cross_validation_score(&samples, &huge).is_none(),
        "a fold count above samples.len() / min_samples_per_fold must be refused before the \
         allocation, not reserved for"
    );
    let too_few = CrossValidationConfig {
        fold_count: 1,
        ..CrossValidationConfig::default()
    };
    assert!(
        compute_cross_validation_score(&samples, &too_few).is_none(),
        "fewer than two folds is not cross-validation and must return None"
    );
}
