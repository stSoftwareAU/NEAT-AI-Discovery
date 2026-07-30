//! Issue #1812 — Gate 1: the sole-op `RemoveNeuron` acceptance rule.
//!
//! Every sole-op `RemoveNeuron` candidate used to be dropped by construction:
//! `apply_honest_remove_neuron_gain` wrote the estimator's **unitless,
//! non-positive** influence into `expectedCreatureScoreGain`, and
//! `apply_final_coordinated_gain_floor` screened it against the strictly
//! positive `coordinated_post_discount_noise_floor(1)`. No non-positive value
//! clears a positive floor, so the survivor set was empty for every neuron that
//! was not promoted (#1785, characterised by #1810).
//!
//! The rule decided by #1811 (`docs/analysis/remove-neuron-gain-scale-1785.md`)
//! nets the exact complexity saving against the calibrated influence loss and
//! screens the result against `removal_net_gain_floor(costOfGrowth)`. These
//! tests pin the four boundaries that rule must not cross:
//!
//! 1. A harmless removal is reachable.
//! 2. A harmful removal is still rejected — and counted under a named reason.
//! 3. Multi-op coordinated candidates keep the shared floor, unrescoped.
//! 4. The #1622 promotion to `CONSTANT_NEURON_PRIORITY_GAIN` still passes.

use neat_ai_discovery::analysis::candidate_aggregation::apply_final_coordinated_gain_floor;
use neat_ai_discovery::analysis::constants::{
    COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP, REMOVE_INFLUENCE_CALIBRATION, removal_net_gain_floor,
};
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    ALL_REJECTION_REASONS, REJECTION_BELOW_EXPECTED_GAIN_FLOOR,
    REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING, RejectionBreakdown, top_level_summary,
};
use neat_ai_discovery::analysis::discovery_mode::DiscoveryMode;
use neat_ai_discovery::analysis::remove_neuron_constant_promotion::CONSTANT_NEURON_PRIORITY_GAIN;
use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
use neat_ai_discovery::analysis::{analysis_cost_of_growth, removal_net_gain};
use neat_ai_discovery::focus::calculate_removal_savings;
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

fn remove_neuron_candidate(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: uuid.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(uuid.to_string()),
    }
}

/// A two-op group ending in a `RemoveNeuron` — an atomic coordinated change, not
/// a bare neuron removal.
fn multi_op_remove_neuron_candidate(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "src".to_string(),
                to_neuron_uuid: uuid.to_string(),
            },
            CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: uuid.to_string(),
            },
        ],
        expected_creature_score_gain: gain,
        comment: Some(format!("{uuid}-multi-op")),
    }
}

/// Drive the FFI-facing final floor and return the surviving candidate labels
/// plus the two drop counts that matter here.
fn run_final_floor(candidates: Vec<CoordinatedStructuralCandidateJson>) -> (Vec<String>, u32, u32) {
    let mut synapse = AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: candidates,
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: SynapseAnalysisMetadata::default(),
    };
    apply_final_coordinated_gain_floor(&mut synapse, DiscoveryMode::Normal, 1.0);
    let counts = synapse.metadata.rejection_breakdown.counts();
    let removal_drops = counts
        .get(REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING)
        .copied()
        .unwrap_or(0);
    let shared_drops = counts
        .get(REJECTION_BELOW_EXPECTED_GAIN_FLOOR)
        .copied()
        .unwrap_or(0);
    let survivors = synapse
        .coordinated_structural_candidates
        .iter()
        .map(|c| c.comment.clone().unwrap_or_default())
        .collect();
    (survivors, removal_drops, shared_drops)
}

/// A degree-2 zero-influence orphan nets `1.2 × costOfGrowth` and reaches the
/// FFI response — the case the old floor made unreachable by construction. It
/// sits far below the shared floor, so this only passes because the removal is
/// routed to its own screen.
#[test]
fn harmless_removal_reaches_the_response() {
    let cost_of_growth = analysis_cost_of_growth();
    let orphan_gain = removal_net_gain(calculate_removal_savings(2, 0, cost_of_growth), 0.0);
    assert!(
        orphan_gain < COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP,
        "the orphan's gain {orphan_gain:e} must sit below the shared floor, or this test would \
         pass without the routing under test"
    );

    let (survivors, removal_drops, shared_drops) =
        run_final_floor(vec![remove_neuron_candidate("orphan", orphan_gain)]);

    assert_eq!(survivors, vec!["orphan".to_string()]);
    assert_eq!(removal_drops, 0);
    assert_eq!(shared_drops, 0);
}

/// A neuron carrying real downstream influence still loses: the calibrated loss
/// outweighs the saving whatever its degree, and the drop is counted under the
/// removal-specific reason rather than vanishing.
#[test]
fn harmful_removal_is_rejected_and_counted() {
    let cost_of_growth = analysis_cost_of_growth();
    // A modest influence for a deep neuron — three orders above the break-even.
    let influence = -1e-1_f64;
    let candidates: Vec<_> = [2_usize, 5, 12, 600]
        .iter()
        .map(|degree| {
            let gain = removal_net_gain(
                calculate_removal_savings(*degree, 0, cost_of_growth),
                influence,
            );
            assert!(
                gain < 0.0,
                "a degree-{degree} neuron at influence {influence} must net a loss, got {gain:e}"
            );
            remove_neuron_candidate(&format!("hot-{degree}"), gain)
        })
        .collect();
    let expected_drops = u32::try_from(candidates.len()).expect("small count");

    let (survivors, removal_drops, shared_drops) = run_final_floor(candidates);

    assert!(
        survivors.is_empty(),
        "no influence-carrying removal may reach the response, got {survivors:?}"
    );
    assert_eq!(
        removal_drops, expected_drops,
        "every rejection must be counted under {REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING}"
    );
    assert_eq!(
        shared_drops, 0,
        "a sole-op removal must never be counted under the shared add-path floor reason"
    );
}

/// A removal sitting just under its own floor is rejected, and one just over it
/// is accepted — the screen is a real boundary, not a rubber stamp.
#[test]
fn the_removal_floor_is_a_real_boundary() {
    let floor = removal_net_gain_floor(analysis_cost_of_growth());
    let (survivors, removal_drops, _) = run_final_floor(vec![
        remove_neuron_candidate("just-under", floor * 0.9),
        remove_neuron_candidate("at-floor", floor),
        remove_neuron_candidate("just-over", floor * 1.1),
    ]);

    assert_eq!(
        survivors,
        vec!["at-floor".to_string(), "just-over".to_string()]
    );
    assert_eq!(removal_drops, 1);
}

/// Multi-op coordinated candidates keep `coordinated_post_discount_noise_floor`
/// — their gain reflects the whole atomic group, not a bare neuron removal — and
/// their drops keep the shared reason.
#[test]
fn multi_op_candidates_keep_the_shared_floor() {
    let orphan_gain = removal_net_gain(
        calculate_removal_savings(2, 0, analysis_cost_of_growth()),
        0.0,
    );
    // The same gain that a *sole-op* removal survives on.
    let (survivors, removal_drops, shared_drops) = run_final_floor(vec![
        multi_op_remove_neuron_candidate("grouped", orphan_gain),
        remove_neuron_candidate("bare", orphan_gain),
    ]);

    assert_eq!(
        survivors,
        vec!["bare".to_string()],
        "the multi-op group must still be screened against the shared floor"
    );
    assert_eq!(shared_drops, 1, "its drop keeps the shared reason");
    assert_eq!(removal_drops, 0);
}

/// The #1622 promotion marker clears the new floor unchanged, so a measured
/// constant neuron still reaches the response by that route.
#[test]
fn promoted_constant_neuron_still_passes() {
    let (survivors, removal_drops, shared_drops) = run_final_floor(vec![remove_neuron_candidate(
        "promoted",
        CONSTANT_NEURON_PRIORITY_GAIN,
    )]);

    assert_eq!(survivors, vec!["promoted".to_string()]);
    assert_eq!(removal_drops, 0);
    assert_eq!(shared_drops, 0);
}

/// The new reason is registered and reaches an operator as prose, not as a bare
/// key — the `describe` arm is exercised through the public summary.
#[test]
fn the_rejection_reason_is_registered_and_described() {
    assert!(
        ALL_REJECTION_REASONS.contains(&REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING),
        "REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING missing from ALL_REJECTION_REASONS"
    );
    let mut breakdown = RejectionBreakdown::new();
    breakdown.record_many_u32(REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING, 33);
    let summary = top_level_summary(&breakdown, Some(33)).expect("summary for a dominant reason");
    assert!(
        summary.contains("cost more score than it saves"),
        "the summary must explain the removal verdict, got: {summary}"
    );
}

/// The calibration converts the estimator's unitless influence onto the
/// creature-score scale, and is applied **bare** — no per-creature correction,
/// which would only ever discount a cost and so make removals easier to accept.
#[test]
fn influence_is_converted_by_the_bare_calibration() {
    let cost_of_growth = analysis_cost_of_growth();
    let saving = calculate_removal_savings(4, 0, cost_of_growth);
    let influence = -1e-4_f64;
    let expected = saving - 1e-4_f32 * REMOVE_INFLUENCE_CALIBRATION;
    let actual = removal_net_gain(saving, influence);
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected:e}, got {actual:e}"
    );
    // The #1811 worked flip: bare it is rejected; at the correction's 0.001
    // clamp it would be accepted. The rule must land on the rejecting side.
    assert!(
        actual < removal_net_gain_floor(cost_of_growth),
        "a degree-4 neuron at 1e-4 influence must be rejected, net {actual:e}"
    );
}
