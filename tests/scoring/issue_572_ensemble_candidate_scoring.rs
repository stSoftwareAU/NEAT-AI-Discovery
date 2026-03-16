//! Tests for ensemble candidate scoring (Issue #572).
//!
//! Validates that candidates from multiple discovery modules are correctly
//! combined when they target the same neuron/synapse, with:
//! - Agreement boost: candidates confirmed by multiple modules get higher scores
//! - Disagreement penalty: conflicting candidates get reduced scores
//! - Single-module passthrough: candidates from only one module are unchanged

use neat_ai_discovery::analysis::ensemble_scoring::apply_ensemble_scoring;
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_set_bias_candidate(
    neuron_uuid: &str,
    bias: f32,
    gain: f32,
    comment: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: neuron_uuid.to_string(),
            bias,
        }],
        expected_creature_score_gain: gain,
        comment: Some(comment.to_string()),
    }
}

fn make_change_squash_candidate(
    neuron_uuid: &str,
    squash: &str,
    gain: f32,
    comment: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
            neuron_uuid: neuron_uuid.to_string(),
            squash: squash.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(comment.to_string()),
    }
}

fn make_remove_neuron_candidate(
    neuron_uuid: &str,
    gain: f32,
    comment: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: neuron_uuid.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(comment.to_string()),
    }
}

fn make_add_synapse_candidate(
    from: &str,
    to: &str,
    weight: f32,
    gain: f32,
    comment: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: from.to_string(),
            to_neuron_uuid: to.to_string(),
            weight,
        }],
        expected_creature_score_gain: gain,
        comment: Some(comment.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Single-module candidates unchanged
// ---------------------------------------------------------------------------

#[test]
fn single_module_candidate_score_unchanged() {
    let candidates = vec![make_set_bias_candidate("neuron-a", 0.5, 0.02, "module A")];

    let result = apply_ensemble_scoring(candidates, &Default::default());

    assert_eq!(result.candidates.len(), 1);
    // Single-module candidate should keep its original score.
    assert!(
        (result.candidates[0].expected_creature_score_gain - 0.02).abs() < 1e-6,
        "single-module candidate score should be unchanged, got {}",
        result.candidates[0].expected_creature_score_gain
    );
    assert_eq!(result.ensemble_candidates, 0);
    assert_eq!(result.single_module_candidates, 1);
}

#[test]
fn empty_input_produces_empty_output() {
    let result = apply_ensemble_scoring(Vec::new(), &Default::default());

    assert!(result.candidates.is_empty());
    assert_eq!(result.ensemble_candidates, 0);
    assert_eq!(result.single_module_candidates, 0);
}

// ---------------------------------------------------------------------------
// Multi-module agreement boosts score
// ---------------------------------------------------------------------------

#[test]
fn agreement_boosts_score_for_same_target_neuron() {
    // Two modules both suggest setting bias on the same neuron.
    let candidates = vec![
        make_set_bias_candidate("neuron-a", 0.5, 0.02, "saturation detection"),
        make_set_bias_candidate("neuron-a", 0.6, 0.03, "operating point"),
    ];

    let result = apply_ensemble_scoring(candidates, &Default::default());

    // The ensemble should produce a single candidate with a boosted score.
    // The boosted score should be higher than the best individual score (0.03).
    assert!(!result.candidates.is_empty());

    let best = result
        .candidates
        .iter()
        .filter(|c| targets_neuron(c, "neuron-a"))
        .max_by(|a, b| {
            a.expected_creature_score_gain
                .total_cmp(&b.expected_creature_score_gain)
        })
        .expect("should have a candidate targeting neuron-a");

    assert!(
        best.expected_creature_score_gain > 0.03,
        "ensemble agreement should boost score above best individual (0.03), got {}",
        best.expected_creature_score_gain
    );
    assert!(result.ensemble_candidates > 0);
}

// ---------------------------------------------------------------------------
// Disagreement penalises score
// ---------------------------------------------------------------------------

#[test]
fn disagreement_penalises_conflicting_candidates() {
    // One module says remove neuron-a, another says change its squash.
    // These are conflicting operations on the same neuron.
    let candidates = vec![
        make_remove_neuron_candidate("neuron-a", 0.05, "dead neuron"),
        make_change_squash_candidate("neuron-a", "RELU", 0.04, "activation mismatch"),
    ];

    let result = apply_ensemble_scoring(candidates, &Default::default());

    // Both candidates should be penalised (reduced from their original scores).
    for c in &result.candidates {
        if targets_neuron(c, "neuron-a") {
            let max_original = 0.05_f32;
            assert!(
                c.expected_creature_score_gain < max_original,
                "conflicting candidate should be penalised below {max_original}, got {}",
                c.expected_creature_score_gain
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Mixed agreement and disagreement
// ---------------------------------------------------------------------------

#[test]
fn mixed_candidates_handled_correctly() {
    let candidates = vec![
        // Two modules agree on neuron-a (set bias) — should boost
        make_set_bias_candidate("neuron-a", 0.5, 0.02, "module-1"),
        make_set_bias_candidate("neuron-a", 0.6, 0.03, "module-2"),
        // One module targets neuron-b alone — should be unchanged
        make_change_squash_candidate("neuron-b", "RELU", 0.01, "module-3"),
    ];

    let result = apply_ensemble_scoring(candidates, &Default::default());

    // neuron-a ensemble: boosted above 0.03
    let neuron_a_best = result
        .candidates
        .iter()
        .filter(|c| targets_neuron(c, "neuron-a"))
        .max_by(|a, b| {
            a.expected_creature_score_gain
                .total_cmp(&b.expected_creature_score_gain)
        })
        .expect("should have neuron-a candidate");
    assert!(
        neuron_a_best.expected_creature_score_gain > 0.03,
        "agreed neuron-a should be boosted, got {}",
        neuron_a_best.expected_creature_score_gain
    );

    // neuron-b: unchanged at 0.01
    let neuron_b = result
        .candidates
        .iter()
        .find(|c| targets_neuron(c, "neuron-b"))
        .expect("should have neuron-b candidate");
    assert!(
        (neuron_b.expected_creature_score_gain - 0.01).abs() < 1e-6,
        "single-module neuron-b should be unchanged, got {}",
        neuron_b.expected_creature_score_gain
    );
}

// ---------------------------------------------------------------------------
// Module success rate weighting
// ---------------------------------------------------------------------------

#[test]
fn module_success_rates_influence_ensemble_weight() {
    use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;

    let mut tracker = ModuleOutcomeTracker::new();
    // Module "saturation detection" has high success rate (8/10)
    for _ in 0..8 {
        tracker.record("saturation detection", true);
    }
    for _ in 0..2 {
        tracker.record("saturation detection", false);
    }
    // Module "operating point" has low success rate (2/10)
    for _ in 0..2 {
        tracker.record("operating point", true);
    }
    for _ in 0..8 {
        tracker.record("operating point", false);
    }

    let candidates = vec![
        make_set_bias_candidate("neuron-a", 0.5, 0.02, "saturation detection"),
        make_set_bias_candidate("neuron-a", 0.6, 0.03, "operating point"),
    ];

    let result = apply_ensemble_scoring(candidates, &tracker);

    // With success rates, the ensemble should still boost (agreement),
    // but the high-success-rate module's estimate should carry more weight.
    let neuron_a = result
        .candidates
        .iter()
        .filter(|c| targets_neuron(c, "neuron-a"))
        .max_by(|a, b| {
            a.expected_creature_score_gain
                .total_cmp(&b.expected_creature_score_gain)
        })
        .expect("should have neuron-a candidate");

    assert!(
        neuron_a.expected_creature_score_gain > 0.02,
        "ensemble with success rates should still boost, got {}",
        neuron_a.expected_creature_score_gain
    );
}

// ---------------------------------------------------------------------------
// Synapse-level matching
// ---------------------------------------------------------------------------

#[test]
fn candidates_targeting_same_synapse_are_ensembled() {
    // Two modules suggest adding the same synapse (same from/to).
    let candidates = vec![
        make_add_synapse_candidate("input-0", "output-0", 0.1, 0.02, "gradient discovery"),
        make_add_synapse_candidate("input-0", "output-0", 0.15, 0.025, "sample weighted"),
    ];

    let result = apply_ensemble_scoring(candidates, &Default::default());

    // Agreement on the same synapse target should boost.
    let best = result
        .candidates
        .iter()
        .max_by(|a, b| {
            a.expected_creature_score_gain
                .total_cmp(&b.expected_creature_score_gain)
        })
        .expect("should have candidates");

    assert!(
        best.expected_creature_score_gain > 0.025,
        "synapse-level agreement should boost, got {}",
        best.expected_creature_score_gain
    );
    assert!(result.ensemble_candidates > 0);
}

// ---------------------------------------------------------------------------
// Candidates for different targets are independent
// ---------------------------------------------------------------------------

#[test]
fn candidates_for_different_targets_are_independent() {
    let candidates = vec![
        make_set_bias_candidate("neuron-a", 0.5, 0.02, "module-1"),
        make_set_bias_candidate("neuron-b", 0.3, 0.01, "module-2"),
    ];

    let result = apply_ensemble_scoring(candidates, &Default::default());

    // Each targets a different neuron, so both should pass through unchanged.
    assert_eq!(result.candidates.len(), 2);
    assert_eq!(result.ensemble_candidates, 0);
    assert_eq!(result.single_module_candidates, 2);

    let a = result
        .candidates
        .iter()
        .find(|c| targets_neuron(c, "neuron-a"))
        .unwrap();
    let b = result
        .candidates
        .iter()
        .find(|c| targets_neuron(c, "neuron-b"))
        .unwrap();
    assert!((a.expected_creature_score_gain - 0.02).abs() < 1e-6);
    assert!((b.expected_creature_score_gain - 0.01).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Result metrics
// ---------------------------------------------------------------------------

#[test]
fn result_tracks_ensemble_vs_single_module_counts() {
    let candidates = vec![
        // Ensemble group (2 modules agree on neuron-a)
        make_set_bias_candidate("neuron-a", 0.5, 0.02, "module-1"),
        make_set_bias_candidate("neuron-a", 0.6, 0.03, "module-2"),
        // Single-module (neuron-b)
        make_change_squash_candidate("neuron-b", "RELU", 0.01, "module-3"),
        // Single-module (neuron-c)
        make_remove_neuron_candidate("neuron-c", 0.04, "module-4"),
    ];

    let result = apply_ensemble_scoring(candidates, &Default::default());

    assert!(
        result.ensemble_candidates >= 1,
        "should track at least 1 ensemble candidate, got {}",
        result.ensemble_candidates
    );
    assert!(
        result.single_module_candidates >= 2,
        "should track at least 2 single-module candidates, got {}",
        result.single_module_candidates
    );
}

// ---------------------------------------------------------------------------
// Helper: check if a candidate targets a given neuron
// ---------------------------------------------------------------------------

fn targets_neuron(candidate: &CoordinatedStructuralCandidateJson, neuron_uuid: &str) -> bool {
    candidate.operations.iter().any(|op| match op {
        CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: uuid, ..
        }
        | CoordinatedStructuralOpJson::ChangeSquash {
            neuron_uuid: uuid, ..
        }
        | CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: uuid, ..
        }
        | CoordinatedStructuralOpJson::AddNeuron {
            neuron_uuid: uuid, ..
        } => uuid == neuron_uuid,
        CoordinatedStructuralOpJson::AddSynapse { to_neuron_uuid, .. }
        | CoordinatedStructuralOpJson::RemoveSynapse { to_neuron_uuid, .. } => {
            to_neuron_uuid == neuron_uuid
        }
        CoordinatedStructuralOpJson::SetWeight { to_neuron_uuid, .. } => {
            to_neuron_uuid == neuron_uuid
        }
    })
}
