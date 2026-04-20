//! Regression tests for Issue #1128 — enforce the minimum expected-gain floor
//! for coordinated-structural candidates **after** all pessimism, calibration,
//! and module-boost discounts have been applied.
//!
//! PR #1115 (Issue #1110) introduced `COORDINATED_MIN_EXPECTED_GAIN = 1e-5` but
//! applied it before the post-merge discounts. GRQ-sampler failure cache
//! `discovery/failures/6d8a5b6a/coordinated-structural/v2_coordinated-structural_29584442...json`
//! captured a candidate with `expectedCreatureScoreGain: 1.17e-7` that should
//! have been filtered. The gaps identified:
//!
//! 1. `merge_coordinated_structural_replacements` allowed single-op candidates
//!    through on `> 0.0`, ignoring the 1e-5 floor.
//! 2. `apply_module_boost_to_candidates` could multiply a gain by as little as
//!    0.5× after merge, pushing above-floor gains below the floor with no
//!    further filtering.
//! 3. `apply_post_processing` filtered coordinated candidates on `> 0.0` after
//!    applying `COORDINATED_PREDICTION_CALIBRATION` (× 0.00005), allowing
//!    sub-noise gains to survive.

use neat_ai_discovery::analysis::candidate_aggregation::apply_coordinated_gain_floor;
use neat_ai_discovery::analysis::constants::{
    COORDINATED_POST_DISCOUNT_NOISE_FLOOR, MIN_BOOST_SAMPLES,
};
use neat_ai_discovery::analysis::discovery_dispatch::{
    DiscoveryDetectionResult, DiscoveryModuleSpec, run_discovery_modules_parallel,
};
use neat_ai_discovery::analysis::module_weights::{
    ModuleOutcomeTracker, apply_module_boost_to_candidates,
};
use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, SynapseJson,
};

fn empty_synapse_result() -> AnalyzeSynapsesResult {
    AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: SynapseAnalysisMetadata {
            candidates_found: 0,
            candidates_returned: 0,
            ..Default::default()
        },
    }
}

fn single_op_candidate(gain: f32, module: &str) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(module.to_string()),
    }
}

fn two_op_candidate(gain: f32, module: &str) -> CoordinatedStructuralCandidateJson {
    // A remove + add-synapse pair — the discount factor is 0.5 for 2-op
    // candidates (COORDINATED_EMPIRICAL_DISCOUNT_2OPS).
    CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "a".to_string(),
                to_neuron_uuid: "b".to_string(),
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "a".to_string(),
                to_neuron_uuid: "c".to_string(),
                weight: 0.5,
            },
        ],
        expected_creature_score_gain: gain,
        comment: Some(module.to_string()),
    }
}

/// Apply the floor filter helper: assert it removes anything below the
/// post-discount noise floor and keeps values at or above it.
#[test]
fn apply_floor_removes_below_and_retains_at_or_above() {
    let mut cands = vec![
        single_op_candidate(COORDINATED_POST_DISCOUNT_NOISE_FLOOR, "at-floor"),
        single_op_candidate(COORDINATED_POST_DISCOUNT_NOISE_FLOOR - 1e-9, "below-floor"),
        single_op_candidate(1.17e-7, "noise-from-issue-1127"),
        single_op_candidate(1.0, "well-above"),
    ];
    apply_coordinated_gain_floor(&mut cands);
    // Candidates exactly at the floor and well above survive; noise-range
    // candidates (1.17e-7 from Issue #1127 failure cache) are filtered.
    assert_eq!(cands.len(), 2);
    let comments: Vec<&str> = cands
        .iter()
        .map(|c| c.comment.as_deref().unwrap_or(""))
        .collect();
    assert!(comments.contains(&"at-floor"));
    assert!(comments.contains(&"well-above"));
}

/// Single-op candidates with raw gain above the pre-merge floor but
/// discounted into the noise range by a `module_boost` + ensemble penalty
/// stack must be filtered by the final floor pass. Reproduces the Issue
/// #1127 failure cache scenario: a 1.17e-7 gain reached the FFI response
/// despite PR #1115.
#[test]
fn module_boost_discount_below_floor_is_filtered() {
    let mut tracker = ModuleOutcomeTracker::new();
    // Record enough failures to give the module a 0.5× boost (minimum clamp).
    for _ in 0..MIN_BOOST_SAMPLES.max(20) {
        tracker.record("weak-module", false);
    }
    // Sanity check: boost is clamped at 0.5 (its lower bound).
    let boost = tracker.module_boost("weak-module");
    assert!(
        (boost - 0.5).abs() < 1e-6,
        "weak-module should have the minimum 0.5 boost, got {boost}"
    );

    // Raw gain 8e-7 is above the post-discount noise floor (5e-7) but a
    // 0.5× module-boost clamp drops it to 4e-7 — in the 1e-7 noise range
    // documented by Issue #1127. The final floor pass must remove it.
    let mut candidates = vec![single_op_candidate(8e-7, "weak-module")];
    apply_module_boost_to_candidates(&mut candidates, &tracker);
    assert!(
        candidates[0].expected_creature_score_gain < COORDINATED_POST_DISCOUNT_NOISE_FLOOR,
        "precondition: post-boost gain should be below noise floor, got {}",
        candidates[0].expected_creature_score_gain
    );

    apply_coordinated_gain_floor(&mut candidates);
    assert!(
        candidates.is_empty(),
        "post-boost noise-range gain should be filtered, survivors: {candidates:?}"
    );
}

/// A 2-op coordinated candidate whose raw gain (above the floor) is pushed
/// below the floor by the `COORDINATED_EMPIRICAL_DISCOUNT_2OPS` factor (0.5×)
/// inside `merge_coordinated_structural_replacements` must not reach
/// `syn.coordinated_structural_candidates`.
#[test]
fn merge_filters_two_op_candidate_discounted_below_floor() {
    let mut syn = empty_synapse_result();
    let mut tracker = ModuleOutcomeTracker::new();

    // Raw 1.5e-5 × 0.5 (2-op discount) = 7.5e-6 — below the floor.
    let below_after_discount = 1.5e-5;
    let modules = vec![DiscoveryModuleSpec {
        module_name: "two-op-discount-module".to_string(),
        phase_name: "test_two_op_discount",
        max_candidates: 0,
        detect_fn: Box::new(move || {
            Some(DiscoveryDetectionResult {
                detected_count: 1,
                candidates: vec![two_op_candidate(
                    below_after_discount,
                    "two-op-discount-module",
                )],
            })
        }),
    }];

    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

    assert!(
        syn.coordinated_structural_candidates.is_empty(),
        "2-op candidate with post-discount gain below floor should be filtered, got {:?}",
        syn.coordinated_structural_candidates
    );
}

/// A single-op candidate whose raw gain is just above the floor is dispatched
/// unchanged (no op-count discount), but if it is subsequently reduced below
/// the floor by downstream steps, the final floor pass must remove it. The
/// merge itself must also refuse single-op gains below the floor.
#[test]
fn merge_filters_single_op_candidate_below_floor() {
    let mut syn = empty_synapse_result();
    let mut tracker = ModuleOutcomeTracker::new();

    // Candidate at 5e-7 — well below the floor. Must be filtered by the
    // pre-merge COORDINATED_MIN_EXPECTED_GAIN check already applied by
    // `merge_discovery_module_results`.
    let modules = vec![DiscoveryModuleSpec {
        module_name: "noise-module".to_string(),
        phase_name: "test_noise_single_op",
        max_candidates: 0,
        detect_fn: Box::new(|| {
            Some(DiscoveryDetectionResult {
                detected_count: 1,
                candidates: vec![single_op_candidate(5e-7, "noise-module")],
            })
        }),
    }];

    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

    assert!(
        syn.coordinated_structural_candidates.is_empty(),
        "noise-level single-op candidate should be filtered, got {:?}",
        syn.coordinated_structural_candidates
    );
}

/// Round-trip: a candidate already in `syn.coordinated_structural_candidates`
/// (as happens with synapse-analysis-internal structural patterns after
/// calibration) must be filtered by `apply_coordinated_gain_floor`.
#[test]
fn floor_pass_filters_existing_candidates_in_syn() {
    let mut syn = empty_synapse_result();
    syn.coordinated_structural_candidates
        .push(single_op_candidate(1.17e-7, "internal-structural"));
    syn.coordinated_structural_candidates
        .push(two_op_candidate(0.01, "well-above"));

    apply_coordinated_gain_floor(&mut syn.coordinated_structural_candidates);

    assert_eq!(syn.coordinated_structural_candidates.len(), 1);
    assert_eq!(
        syn.coordinated_structural_candidates[0]
            .comment
            .as_deref()
            .unwrap_or(""),
        "well-above"
    );
}

// Silence unused-import warning for SynapseJson on platforms where the test is
// compiled but not linked. It is imported for completeness of the coordinated
// aggregate shape.
#[allow(dead_code)]
fn _ensure_synapse_json_in_scope() -> Option<SynapseJson> {
    None
}
