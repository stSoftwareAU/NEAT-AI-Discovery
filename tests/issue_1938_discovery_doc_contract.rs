//! Issue #1938 — `docs/DISCOVERY_TYPES.md` and `docs/ANALYSIS_DEEP_DIVE.md` must
//! only state statuses, thresholds, and parameter ranges the shipped code honours.
//!
//! `DISCOVERY_TYPES.md` is the designated ground truth (CONTRIBUTING.md and
//! AGENTS.md both point at it), so a stale threshold there propagates everywhere.
//! Each test below pins one documented claim to observable behaviour of the real
//! code, then asserts the prose agrees with what that behaviour just proved.

mod common;

use common::{make_creature, neuron, record, synapse};
use neat_ai_discovery::CandidateNeuronJson;
use neat_ai_discovery::analysis::constants::{
    ADAPTIVE_PROPOSAL_CANDIDATE_COUNT, ADAPTIVE_PROPOSAL_MIN_HISTORY,
};
use neat_ai_discovery::analysis::detection::low_impact_neuron::detect_low_impact_neurons;
use neat_ai_discovery::analysis::synapse::adaptive_proposal::{
    AcceptanceTracker, generate_fixed_grid, generate_weight_candidates,
};
use neat_ai_discovery::analysis::utils::filter_candidates_to_sensible_ranges;
use neat_ai_discovery::config::batch_successful_enabled;
use neat_ai_discovery::focus::triage_removal_candidates;
use neat_ai_discovery::types::DiscoverRecord;

const DISCOVERY_TYPES: &str = include_str!("../docs/DISCOVERY_TYPES.md");
const DEEP_DIVE: &str = include_str!("../docs/ANALYSIS_DEEP_DIVE.md");

/// Text of the markdown section introduced by `heading`, up to the next heading
/// of the same or a higher level.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let start = doc
        .find(heading)
        .unwrap_or_else(|| panic!("doc must contain the heading {heading:?}"));
    let level = heading.chars().filter(|c| *c == '#').count();
    let body = &doc[start + heading.len()..];
    body.match_indices("\n#")
        .find(|(idx, _)| body[idx + 1..].chars().take_while(|c| *c == '#').count() <= level)
        .map_or(body, |(idx, _)| &body[..idx])
}

/// The summary-table row whose first cell links to `anchor`.
fn summary_row<'a>(doc: &'a str, anchor: &str) -> &'a str {
    doc.lines()
        .find(|line| line.starts_with("| [") && line.contains(anchor))
        .unwrap_or_else(|| panic!("summary table must contain a row linking to {anchor}"))
}

/// An add-neuron candidate with the three range-filtered parameters supplied.
fn candidate(incoming: f32, outgoing: f32, bias: f32, squash: &str) -> CandidateNeuronJson {
    CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: incoming,
        outgoing_weight: outgoing,
        squash: squash.to_string(),
        bias,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 1.0e-5,
        expected_creature_score_gain: 1.0e-5,
        improved_count: 10,
        total_count: 20,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [0.0, 0.0],
        target_saturation_factor: None,
        variant_key: None,
    }
}

/// Item 1 — batch-successful grouping is gated off in the shipped defaults
/// (Issue #1059), so the reference must not advertise it as active.
#[test]
fn batch_successful_is_off_by_default_and_the_reference_says_so() {
    if std::env::var("NEAT_AI_DISCOVERY_BATCH_SUCCESSFUL").is_err() {
        assert!(
            !batch_successful_enabled(),
            "batch-successful must stay disabled unless the env gate is set"
        );
    }

    let row = summary_row(DISCOVERY_TYPES, "#batch-successful-grouping");
    assert!(
        !row.contains("🟢 Active"),
        "batch-successful is disabled by default; summary row must not claim 🟢 Active: {row}"
    );
    assert!(
        row.contains('⛔'),
        "batch-successful summary row must carry the disabled marker: {row}"
    );

    let detail = section(DISCOVERY_TYPES, "\n### Batch-Successful Grouping");
    assert!(
        detail.contains("NEAT_AI_DISCOVERY_BATCH_SUCCESSFUL"),
        "the Batch-Successful detail section must document the opt-in env gate"
    );
}

/// Item 2 — the low-impact ceiling was widened 1e-3 → 0.04 (Issue #892), so a
/// neuron the doc's stale ceiling excluded is in fact detected.
#[test]
fn low_impact_detection_admits_activations_above_the_stale_1e_3_ceiling() {
    let creature = make_creature(
        vec![
            neuron("hidden-quiet", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-quiet", "output-1", 0.5)],
    );

    // 0.02 sits above the documented 1e-3 ceiling but below the shipped 0.04.
    let admitted: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-quiet", i, 0.02, Some(0.02)))
        .collect();
    let candidates =
        detect_low_impact_neurons(&creature, &[("hidden-quiet".to_string(), admitted)], None);
    assert_eq!(
        candidates.len(),
        1,
        "mean activation 0.02 is below the shipped 0.04 ceiling and must be detected"
    );

    // 0.05 is above the shipped ceiling and must still be rejected.
    let rejected: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-quiet", i, 0.05, Some(0.05)))
        .collect();
    assert!(
        detect_low_impact_neurons(&creature, &[("hidden-quiet".to_string(), rejected)], None)
            .is_empty(),
        "mean activation 0.05 is above the shipped 0.04 ceiling and must be rejected"
    );

    let detail = section(DISCOVERY_TYPES, "\n### Low-Impact Neuron Detection");
    assert!(
        detail.contains("0.04"),
        "the low-impact section must quote the shipped 0.04 ceiling"
    );
    assert!(
        !detail.contains("1e-3"),
        "the low-impact section must not quote the pre-#892 1e-3 ceiling"
    );
}

/// Item 4 — the worked add-neuron example must survive the sensible-range
/// filter, otherwise the reference shows a candidate that can never be emitted.
#[test]
fn documented_add_neuron_example_survives_the_sensible_range_filter() {
    let example = section(DISCOVERY_TYPES, "\n### Add Neurons");
    let block = example
        .split("```json")
        .nth(1)
        .and_then(|rest| rest.split("```").next())
        .expect("the Add Neurons section must show a JSON example");
    let parsed: serde_json::Value =
        serde_json::from_str(block).expect("the documented example must be valid JSON");
    let shown = &parsed["rustRequest"]["neuronCandidate"];

    let read = |key: &str| {
        #[allow(clippy::cast_possible_truncation)]
        let value = shown[key]
            .as_f64()
            .unwrap_or_else(|| panic!("example must show a numeric {key}"))
            as f32;
        value
    };
    let documented = candidate(
        read("incomingWeight"),
        read("outgoingWeight"),
        read("bias"),
        shown["squash"]
            .as_str()
            .expect("example must show a squash"),
    );

    assert_eq!(
        filter_candidates_to_sensible_ranges(vec![documented]).len(),
        1,
        "the documented add-neuron example must pass filter_candidates_to_sensible_ranges"
    );
}

/// Item 6 — the deep dive's sensible ranges must be the post-#888 ones: the old
/// bounds it quoted are rejected outright by the shipped filter.
#[test]
fn deep_dive_sensible_ranges_match_the_shipped_filter() {
    assert_eq!(
        filter_candidates_to_sensible_ranges(vec![candidate(5.0, 0.01, 2.0, "TANH")]).len(),
        1,
        "the post-#888 bounds (5.0 / 0.01 / 2.0) must pass the filter"
    );
    for stale in [
        candidate(20.0, 0.01, 2.0, "TANH"),
        candidate(5.0, 0.1, 2.0, "TANH"),
        candidate(5.0, 0.01, 10.0, "TANH"),
    ] {
        assert!(
            filter_candidates_to_sensible_ranges(vec![stale]).is_empty(),
            "the pre-#888 bounds documented in ANALYSIS_DEEP_DIVE.md are rejected by the filter"
        );
    }

    let ranges = section(
        DEEP_DIVE,
        "\n### 📏 Sensible parameter ranges (add-neurons)",
    );
    assert!(
        ranges.contains("|w| ≤ 5") && ranges.contains("|b| ≤ 2") && ranges.contains("|w| ≤ 0.01"),
        "the deep dive must quote the shipped 5 / 2 / 0.01 bounds: {ranges}"
    );
    assert!(
        !ranges.contains("≤ 20") && !ranges.contains("≤ 10") && !ranges.contains("≤ 0.1\n"),
        "the deep dive must not quote the pre-#888 20 / 10 / 0.1 bounds: {ranges}"
    );
}

/// Item 5 — the live add-synapse path proposes `ADAPTIVE_PROPOSAL_CANDIDATE_COUNT`
/// weights; the 9-variant grid is only the no-history fallback.
#[test]
fn add_synapse_weight_search_documents_the_live_candidate_count() {
    let cold = AcceptanceTracker::new();
    assert_eq!(
        generate_weight_candidates(1.0, "src", "tgt", &cold, "output").len(),
        generate_fixed_grid(1.0).len(),
        "without history the add-synapse path falls back to the fixed grid"
    );

    let mut warm = AcceptanceTracker::new();
    warm.record_batch(
        "output",
        5,
        u32::try_from(ADAPTIVE_PROPOSAL_MIN_HISTORY).expect("min history fits in u32"),
    );
    assert_eq!(
        generate_weight_candidates(1.0, "src", "tgt", &warm, "output").len(),
        ADAPTIVE_PROPOSAL_CANDIDATE_COUNT,
        "with history the live path proposes ADAPTIVE_PROPOSAL_CANDIDATE_COUNT weights"
    );

    let detail = section(DISCOVERY_TYPES, "\n### Add Synapses");
    assert!(
        detail.contains(&ADAPTIVE_PROPOSAL_CANDIDATE_COUNT.to_string()),
        "the Add Synapses section must quote the live candidate count"
    );
    assert!(
        detail.contains("fallback"),
        "the Add Synapses section must mark the 9-variant grid as the fallback"
    );
}

/// Item 3 — the shipped removal criterion is boosted savings vs contribution,
/// not `impact < costOfGrowth`; the reference must defer to `FOCUS_SELECTION.md`.
#[test]
fn removal_criterion_is_boosted_savings_and_lives_in_focus_selection() {
    let creature = make_creature(
        vec![
            neuron("h-high", "hidden", "IDENTITY"),
            neuron("h-low", "hidden", "IDENTITY"),
            neuron("out", "output", "IDENTITY"),
        ],
        vec![synapse("h-high", "out", 1.0), synapse("h-low", "out", 1e-8)],
    );

    let triage = triage_removal_candidates(&creature, Some(1e-4));
    let reason = &triage
        .candidates
        .iter()
        .find(|c| c.neuron_uuid == "h-low")
        .expect("the near-zero-contribution hidden neuron must survive triage")
        .reason;
    assert!(
        reason.contains("boosted") && reason.contains("impact"),
        "the shipped reason states boosted savings against contribution: {reason}"
    );
    assert!(
        !reason.contains("< costOfGrowth"),
        "the shipped reason is not the stale `impact < costOfGrowth` criterion: {reason}"
    );

    let detail = section(DISCOVERY_TYPES, "\n### Remove Low-Impact Neurons");
    assert!(
        !detail.contains("impact < `costOfGrowth`"),
        "the removal criterion must not be restated in its pre-boost form"
    );
    assert!(
        detail.contains("FOCUS_SELECTION.md"),
        "the removal criterion must live in one place — link to FOCUS_SELECTION.md"
    );
}

/// Item 5 (minor) — one status marker per module: the summary row and the detail
/// section for remove-neuron-high-error must agree.
#[test]
fn remove_neuron_high_error_uses_one_status_marker() {
    let row = summary_row(DISCOVERY_TYPES, "#remove-neuron-high-error");
    let detail = section(DISCOVERY_TYPES, "\n### Remove Neuron (High Error)");
    assert!(
        row.contains('⛔'),
        "summary row must use the disabled marker: {row}"
    );
    assert!(
        detail.contains('⛔') && !detail.contains('🔴'),
        "the detail section must use the same ⛔ marker as the summary row"
    );
}
