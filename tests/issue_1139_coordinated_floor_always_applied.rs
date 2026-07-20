//! Regression tests for Issue #1139 — apply the coordinated-structural gain
//! floor unconditionally after variant generation.
//!
//! ## Bug evidence
//!
//! GRQ-sampler commit `744ac60d` (`2026-04-22T22:31:14.839Z`,
//! discoveryVersion `0.74.16`) captured two `Gentle Nudge` variants with
//! `expectedCreatureScoreGain` below `COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP`
//! (5e-7) reaching the FFI response and damaging the creature when tested.
//!
//! ## Root cause
//!
//! `pair_coordinated_structural_with_weight_variants` multiplies the base
//! candidate's `expected_creature_score_gain` by 0.75 / 0.5 / 0.25 / 0.1 for
//! the paired variants, so a base candidate just above the floor produces
//! sub-floor variants. The final cleanup filter
//! (`apply_coordinated_gain_floor_with_multiplier`) only ran inside the
//! `!memory_budget_exceeded && !post_processing_deadline_passed` guard, so
//! whenever either condition tripped the sub-floor variants leaked out.
//!
//! ## Fix
//!
//! The final floor sweep is now driven through
//! `apply_final_coordinated_gain_floor` and called unconditionally after the
//! guarded post-processing block. This test exercises the helper against a
//! `AnalyzeSynapsesResult` populated with freshly generated variants to
//! verify:
//!
//! 1. No coordinated-structural candidate below the floor remains.
//! 2. The rejection breakdown records the drop.
//! 3. `candidates_returned` is refreshed.

use neat_ai_discovery::analysis::candidate_aggregation::apply_final_coordinated_gain_floor;
use neat_ai_discovery::analysis::constants::COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::REJECTION_BELOW_EXPECTED_GAIN_FLOOR;
use neat_ai_discovery::analysis::discovery_mode::DiscoveryMode;
use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
use neat_ai_discovery::analysis::utils::pair_coordinated_structural_with_weight_variants;
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

fn empty_synapse_result() -> AnalyzeSynapsesResult {
    AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: SynapseAnalysisMetadata::default(),
    }
}

/// Build an `AddSynapse`-only coordinated candidate whose variant generation
/// will produce sub-floor expected-gain values.
fn add_synapse_candidate(weight: f32, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: "src".to_string(),
            to_neuron_uuid: "dst".to_string(),
            weight,
        }],
        expected_creature_score_gain: gain,
        comment: Some("base".to_string()),
    }
}

/// Verify that variant generation can indeed push a base candidate below the
/// floor. This reproduces the precondition for the Issue #1139 leak.
#[test]
fn variant_generation_can_produce_subfloor_gains() {
    // Base gain 1.9e-6 × gentle-nudge multiplier (0.25) = 4.75e-7, below the
    // 5e-7 floor. The GRQ-sampler failure cache captured ~1.3e-7 `Gentle
    // Nudge` variants — same mechanism, lower base gain.
    let base_gain = 1.9e-6_f32;
    let base = add_synapse_candidate(0.1, base_gain);
    let variants = pair_coordinated_structural_with_weight_variants(vec![base], None);

    let any_subfloor = variants
        .iter()
        .any(|c| c.expected_creature_score_gain < COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP);
    assert!(
        any_subfloor,
        "precondition: at least one variant should fall below the floor, \
         got gains {:?}",
        variants
            .iter()
            .map(|c| c.expected_creature_score_gain)
            .collect::<Vec<_>>()
    );
}

/// The helper must strip every sub-floor coordinated candidate and refresh
/// the rejection breakdown plus `candidates_returned`. This is the core
/// guarantee the orchestration relies on in the fast-path and the
/// skipped-post-processing fallback alike.
#[test]
fn final_floor_removes_subfloor_variants_and_updates_metadata() {
    let mut syn = empty_synapse_result();

    // Simulate what `apply_post_processing` produces just before the guard:
    // the base plus its paired variants including sub-floor `Gentle Nudge`.
    let base = add_synapse_candidate(0.1, 1.9e-6);
    syn.coordinated_structural_candidates =
        pair_coordinated_structural_with_weight_variants(vec![base], None);

    let total_before = syn.coordinated_structural_candidates.len();
    let subfloor_before = syn
        .coordinated_structural_candidates
        .iter()
        .filter(|c| c.expected_creature_score_gain < COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP)
        .count();
    assert!(
        subfloor_before > 0,
        "precondition: at least one sub-floor variant must be present"
    );

    let removed = apply_final_coordinated_gain_floor(&mut syn, DiscoveryMode::Normal, 1.0);

    assert_eq!(
        removed as usize, subfloor_before,
        "removed count must equal the number of sub-floor variants"
    );
    assert!(
        syn.coordinated_structural_candidates
            .iter()
            .all(|c| c.expected_creature_score_gain >= COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP),
        "no sub-floor coordinated candidate should survive, got gains {:?}",
        syn.coordinated_structural_candidates
            .iter()
            .map(|c| c.expected_creature_score_gain)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        total_before - subfloor_before,
        "kept count must equal total minus sub-floor"
    );

    // Rejection breakdown must reflect the drop.
    let recorded = syn
        .metadata
        .rejection_breakdown
        .counts()
        .get(REJECTION_BELOW_EXPECTED_GAIN_FLOOR)
        .copied()
        .unwrap_or(0);
    assert_eq!(
        recorded, removed,
        "rejection breakdown must record the removed count, got {recorded}"
    );

    // `candidates_returned` must be refreshed to reflect the post-filter total.
    let expected_returned = syn.helpful_synapses.len()
        + syn.harmful_synapses.len()
        + syn.coordinated_structural_candidates.len();
    assert_eq!(
        syn.metadata.candidates_returned, expected_returned,
        "candidates_returned must be refreshed after filtering"
    );
}

/// In conservative mode the configured multiplier must tighten the floor.
/// A candidate at exactly the normal-mode floor survives Normal mode but is
/// dropped by Conservative mode with a 10× multiplier.
#[test]
fn final_floor_honours_conservative_multiplier() {
    let mut syn = empty_synapse_result();
    syn.coordinated_structural_candidates
        .push(add_synapse_candidate(
            0.1,
            COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP,
        ));

    let removed_normal = apply_final_coordinated_gain_floor(&mut syn, DiscoveryMode::Normal, 10.0);
    assert_eq!(removed_normal, 0);
    assert_eq!(syn.coordinated_structural_candidates.len(), 1);

    // Same candidate, Conservative mode with 10× multiplier — now dropped.
    let removed_conservative =
        apply_final_coordinated_gain_floor(&mut syn, DiscoveryMode::Conservative, 10.0);
    assert_eq!(removed_conservative, 1);
    assert!(syn.coordinated_structural_candidates.is_empty());
}

/// When no sub-floor candidates exist the helper must be a no-op on the
/// candidate list while still refreshing `candidates_returned` (important
/// when the skipped-post-processing path never wrote it earlier).
#[test]
fn final_floor_noop_when_all_above_floor() {
    let mut syn = empty_synapse_result();
    syn.coordinated_structural_candidates
        .push(add_synapse_candidate(0.1, 1.0));
    syn.coordinated_structural_candidates
        .push(add_synapse_candidate(0.1, 0.5));
    syn.metadata.candidates_returned = 0;

    let removed = apply_final_coordinated_gain_floor(&mut syn, DiscoveryMode::Normal, 1.0);

    assert_eq!(removed, 0);
    assert_eq!(syn.coordinated_structural_candidates.len(), 2);
    assert_eq!(syn.metadata.candidates_returned, 2);
}
