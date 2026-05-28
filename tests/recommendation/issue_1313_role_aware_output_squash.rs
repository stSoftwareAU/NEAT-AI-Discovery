//! Tests for Issue #1313: Role-aware activation recommendation.
//!
//! When the [`TaskDescriptor`] reports a `OneHot` or `Simplex` target
//! topology, **output-neuron** activation recommendations must be drawn
//! from the descriptor's `output_squash_family` (a bounded family such
//! as LOGISTIC / sigmoid). Hidden-neuron recommendations and the
//! neutral / `OTHER` descriptor path are unchanged.
//!
//! Acceptance criteria from the issue:
//!
//! 1. With a `CATEGORICAL_ERROR` (or other OneHot/Simplex) descriptor,
//!    output-neuron activation recommendations come from the bounded
//!    family.
//! 2. Hidden-neuron recommendations are unchanged.
//! 3. `OTHER` / `Unknown` / absent descriptor ⇒ current behaviour
//!    (regression guard).
//! 4. Focused: small net with an unbounded output squash + one-hot
//!    descriptor ⇒ recommends a bounded unipolar squash for the
//!    output neuron.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::recommendation::activation_recommendation::{
    recommend_activation_function, recommend_activation_function_for_role,
};
use neat_ai_discovery::analysis::task_descriptor::{
    OutputSquashFamily, TargetTopology, TaskDescriptor,
};
use neat_ai_discovery::types::DiscoverRecord;

fn record(uuid: &str, idx: u32, activation: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![0.01],
    }
}

/// 80 records spread across an unbounded range — the classifier sees
/// these as a Uniform distribution, and the unconstrained recommender
/// would prefer something like TANH / IDENTITY / RELU. Used as the raw
/// material for "output neuron currently has an unbounded squash" tests.
fn unbounded_records(uuid: &str) -> Vec<DiscoverRecord> {
    (0..80)
        .map(|i| record(uuid, i, -8.0 + (i as f32) * 0.2))
        .collect()
}

/// Set of squash names that constitute the bounded-unipolar output family.
/// Recommendations under a OneHot/Simplex descriptor must land in this set.
fn bounded_unipolar_family() -> &'static [&'static str] {
    &["LOGISTIC", "STEP"]
}

// =============================================================================
// 1. OneHot + unbounded output squash ⇒ recommends a bounded unipolar squash
// =============================================================================

#[test]
fn one_hot_descriptor_recommends_bounded_unipolar_for_output_neuron() {
    let records = unbounded_records("output-0");
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 3);
    assert_eq!(descriptor.target_topology, TargetTopology::OneHot);
    assert_eq!(
        descriptor.output_squash_family,
        OutputSquashFamily::BoundedUnipolar
    );

    let recommendation =
        recommend_activation_function_for_role(&records, "IDENTITY", true, &descriptor)
            .expect("OneHot + unbounded output squash must produce a recommendation");

    assert_eq!(recommendation.current_squash, "IDENTITY");
    assert!(
        bounded_unipolar_family().contains(&recommendation.recommended_squash.as_str()),
        "Expected recommendation in bounded-unipolar family, got {}",
        recommendation.recommended_squash
    );
}

// =============================================================================
// 2. Simplex (CROSS_ENTROPY) + unbounded output squash ⇒ same bounded family
// =============================================================================

#[test]
fn simplex_descriptor_recommends_bounded_unipolar_for_output_neuron() {
    let records = unbounded_records("output-0");
    let descriptor = TaskDescriptor::from_name("CROSS_ENTROPY", 5);
    assert_eq!(descriptor.target_topology, TargetTopology::Simplex);

    let recommendation =
        recommend_activation_function_for_role(&records, "RELU", true, &descriptor)
            .expect("Simplex + RELU output must produce a recommendation");

    assert!(
        bounded_unipolar_family().contains(&recommendation.recommended_squash.as_str()),
        "Expected bounded-unipolar recommendation under Simplex, got {}",
        recommendation.recommended_squash
    );
}

// =============================================================================
// 3. Neutral / OTHER descriptor ⇒ falls back to existing recommender
// =============================================================================

#[test]
fn neutral_descriptor_falls_back_to_existing_behaviour() {
    let records = unbounded_records("output-0");
    let neutral = TaskDescriptor::neutral();

    let role_aware = recommend_activation_function_for_role(&records, "IDENTITY", true, &neutral);
    let legacy = recommend_activation_function(&records, "IDENTITY");

    match (role_aware, legacy) {
        (Some(a), Some(b)) => {
            assert_eq!(
                a.recommended_squash, b.recommended_squash,
                "Neutral descriptor must mirror legacy recommendation"
            );
        }
        (None, None) => {}
        (a, b) => panic!(
            "Neutral descriptor must mirror legacy recommender; got role_aware={:?} legacy={:?}",
            a.map(|r| r.recommended_squash),
            b.map(|r| r.recommended_squash),
        ),
    }
}

// =============================================================================
// 4. OTHER cost name ⇒ neutral descriptor ⇒ existing behaviour
// =============================================================================

#[test]
fn other_cost_descriptor_falls_back_to_existing_behaviour() {
    let records = unbounded_records("output-0");
    let other = TaskDescriptor::from_name("OTHER", 3);
    assert_eq!(other, TaskDescriptor::neutral());

    let role_aware = recommend_activation_function_for_role(&records, "IDENTITY", true, &other);
    let legacy = recommend_activation_function(&records, "IDENTITY");

    assert_eq!(
        role_aware.map(|r| r.recommended_squash),
        legacy.map(|r| r.recommended_squash),
        "OTHER cost must produce the legacy recommendation",
    );
}

// =============================================================================
// 5. Hidden neurons are NOT biased to the descriptor's family
// =============================================================================

#[test]
fn one_hot_descriptor_does_not_bias_hidden_neuron_recommendation() {
    let records = unbounded_records("hidden-1");
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 3);

    let hidden_rec =
        recommend_activation_function_for_role(&records, "IDENTITY", false, &descriptor);
    let legacy = recommend_activation_function(&records, "IDENTITY");

    assert_eq!(
        hidden_rec.map(|r| r.recommended_squash),
        legacy.map(|r| r.recommended_squash),
        "Hidden-neuron path must mirror legacy recommender even under OneHot",
    );
}

// =============================================================================
// 6. Output neuron already in the bounded family ⇒ no recommendation
// =============================================================================

#[test]
fn output_neuron_already_in_family_yields_no_recommendation() {
    // Build records that are themselves already in [0, 1] — typical for a
    // sigmoid output. The role-aware recommender should not propose a
    // change when the current squash already matches the descriptor
    // family and the data fits.
    let records: Vec<DiscoverRecord> = (0..80)
        .map(|i| record("output-0", i, (i as f32) / 80.0))
        .collect();
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 3);

    let recommendation =
        recommend_activation_function_for_role(&records, "LOGISTIC", true, &descriptor);

    assert!(
        recommendation.is_none(),
        "Output neuron already on LOGISTIC under OneHot should not be re-recommended (got {:?})",
        recommendation.map(|r| r.recommended_squash),
    );
}

// =============================================================================
// 7. Insufficient samples ⇒ None (regression guard)
// =============================================================================

#[test]
fn insufficient_samples_returns_none_for_output_role() {
    let records: Vec<DiscoverRecord> = (0..5).map(|i| record("output-0", i, 0.5)).collect();
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 3);

    let recommendation =
        recommend_activation_function_for_role(&records, "IDENTITY", true, &descriptor);

    assert!(
        recommendation.is_none(),
        "Should not recommend with insufficient samples"
    );
}
