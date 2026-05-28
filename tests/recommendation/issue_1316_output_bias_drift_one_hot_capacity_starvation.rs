//! Tests for Issue #1316: Weight output bias-drift detection by class prior.
//!
//! Under `OneHot` / `Simplex` target topologies, an output neuron whose class
//! has positive support but whose activation never crosses a saturating
//! threshold for that class is suffering capacity starvation. The role-aware
//! detector flags / weights up those neurons for growth. For every other
//! descriptor (`Independent`, `Margin`, `Unknown`, `OTHER`) the behaviour
//! must be identical to the legacy detector — regression guard.
//!
//! Acceptance criteria from the issue:
//!
//! 1. Under a `OneHot` descriptor, an output neuron that stays unsaturated for
//!    a well-supported class is flagged / weighted up for growth.
//! 2. `OTHER` / `Unknown` / absent ⇒ current behaviour (regression guard).

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::recommendation::output_bias_drift::{
    detect_output_bias_drift, detect_output_bias_drift_with_descriptor,
};
use neat_ai_discovery::analysis::task_descriptor::TaskDescriptor;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn make_record(uuid: &str, idx: u32, value: f32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(value),
        activation,
        errors: vec![error],
    }
}

fn neuron(uuid: &str, neuron_type: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "LOGISTIC".to_string(),
        bias,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

fn creature_with_one_output() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        synapses: vec![synapse("input-1", "output-1", 0.5)],
        input: 1,
        output: 1,
    }
}

/// 60 records — 30 with target=1.0 (positive support) and 30 with target=0.0.
/// On the positive-support samples the activation never crosses the
/// saturating threshold (peaks at ~0.55), so the neuron is capacity-starved.
fn capacity_starved_records(uuid: &str) -> Vec<DiscoverRecord> {
    (0..60)
        .map(|i| {
            if i < 30 {
                // Positive-support class: target=1, activation stuck near 0.5,
                // error = target - activation ≈ +0.45 (positive bias drift).
                make_record(uuid, i, 1.0, 0.55, 0.45)
            } else {
                make_record(uuid, i, 0.0, 0.05, -0.05)
            }
        })
        .collect()
}

/// 60 records that are *both* a capacity-starvation case under `OneHot` AND a
/// legacy bias-drift case (mean error positive, >70% same-sign errors). Used
/// to verify the role-aware path boosts the existing candidate.
fn capacity_starved_and_biased_records(uuid: &str) -> Vec<DiscoverRecord> {
    (0..60)
        .map(|i| {
            if i < 30 {
                make_record(uuid, i, 1.0, 0.55, 0.45)
            } else {
                // Off-class records still record a positive error (e.g. global
                // upward shift) so the legacy detector sees a majority-positive
                // error signal and emits a candidate.
                make_record(uuid, i, 0.0, 0.05, 0.10)
            }
        })
        .collect()
}

/// 60 records — output neuron *does* reach the saturating region on its
/// positive-support class (activation peaks at ~0.95 for target=1).
fn well_saturated_records(uuid: &str) -> Vec<DiscoverRecord> {
    (0..60)
        .map(|i| {
            if i < 30 {
                make_record(uuid, i, 1.0, 0.95, 0.05)
            } else {
                make_record(uuid, i, 0.0, 0.05, -0.05)
            }
        })
        .collect()
}

// =============================================================================
// 1. OneHot descriptor flags a capacity-starved output neuron
// =============================================================================
#[test]
fn one_hot_descriptor_flags_capacity_starved_output_neuron() {
    let creature = creature_with_one_output();
    let records = capacity_starved_records("output-1");
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 1);

    let candidates = detect_output_bias_drift_with_descriptor(
        &creature,
        &[("output-1".to_string(), records)],
        &descriptor,
    );

    assert!(
        !candidates.is_empty(),
        "OneHot + capacity-starved output must produce a candidate",
    );
    let starved = candidates
        .iter()
        .find(|c| c.neuron_uuid == "output-1")
        .expect("capacity-starved output-1 must be present");
    assert!(
        starved.capacity_starved,
        "candidate must be marked capacity_starved under OneHot",
    );
}

// =============================================================================
// 2. Simplex descriptor (CROSS_ENTROPY) — same flag fires
// =============================================================================
#[test]
fn simplex_descriptor_flags_capacity_starved_output_neuron() {
    let creature = creature_with_one_output();
    let records = capacity_starved_records("output-1");
    let descriptor = TaskDescriptor::from_name("CROSS_ENTROPY", 3);

    let candidates = detect_output_bias_drift_with_descriptor(
        &creature,
        &[("output-1".to_string(), records)],
        &descriptor,
    );

    let starved = candidates
        .iter()
        .find(|c| c.neuron_uuid == "output-1")
        .expect("Simplex + capacity-starved output must produce a candidate");
    assert!(
        starved.capacity_starved,
        "candidate must be marked capacity_starved under Simplex",
    );
}

// =============================================================================
// 3. Saturated output under OneHot — capacity_starved is false
// =============================================================================
#[test]
fn one_hot_does_not_flag_well_saturated_output() {
    let creature = creature_with_one_output();
    let records = well_saturated_records("output-1");
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 1);

    let candidates = detect_output_bias_drift_with_descriptor(
        &creature,
        &[("output-1".to_string(), records)],
        &descriptor,
    );

    // The neuron may or may not emit a bias-drift candidate from its
    // residual errors; what must NOT happen is the capacity-starved flag.
    for c in &candidates {
        assert!(
            !c.capacity_starved,
            "well-saturated output must not be flagged capacity_starved (got {})",
            c.neuron_uuid,
        );
    }
}

// =============================================================================
// 4. Neutral descriptor ⇒ legacy behaviour (regression guard)
// =============================================================================
#[test]
fn neutral_descriptor_matches_legacy_detector() {
    let creature = creature_with_one_output();
    let records = capacity_starved_records("output-1");
    let neutral = TaskDescriptor::neutral();

    let legacy = detect_output_bias_drift(&creature, &[("output-1".to_string(), records.clone())]);
    let role_aware = detect_output_bias_drift_with_descriptor(
        &creature,
        &[("output-1".to_string(), records)],
        &neutral,
    );

    assert_eq!(
        legacy.len(),
        role_aware.len(),
        "Neutral descriptor must return the same number of candidates as legacy detector",
    );
    for (a, b) in legacy.iter().zip(role_aware.iter()) {
        assert_eq!(a.neuron_uuid, b.neuron_uuid);
        assert!(
            !b.capacity_starved,
            "Neutral descriptor must never set capacity_starved",
        );
        assert!(
            (a.estimated_improvement - b.estimated_improvement).abs() < 1e-6,
            "Neutral descriptor must not boost estimated_improvement",
        );
    }
}

// =============================================================================
// 5. OTHER cost ⇒ legacy behaviour (regression guard)
// =============================================================================
#[test]
fn other_cost_matches_legacy_detector() {
    let creature = creature_with_one_output();
    let records = capacity_starved_records("output-1");
    let other = TaskDescriptor::from_name("OTHER", 1);

    let legacy = detect_output_bias_drift(&creature, &[("output-1".to_string(), records.clone())]);
    let role_aware = detect_output_bias_drift_with_descriptor(
        &creature,
        &[("output-1".to_string(), records)],
        &other,
    );

    assert_eq!(legacy.len(), role_aware.len());
    for c in &role_aware {
        assert!(
            !c.capacity_starved,
            "OTHER descriptor must never set capacity_starved",
        );
    }
}

// =============================================================================
// 6. Independent descriptor (MSE) ⇒ legacy behaviour (regression guard)
// =============================================================================
#[test]
fn mse_descriptor_matches_legacy_detector() {
    let creature = creature_with_one_output();
    let records = capacity_starved_records("output-1");
    let mse = TaskDescriptor::from_name("MSE", 1);

    let legacy = detect_output_bias_drift(&creature, &[("output-1".to_string(), records.clone())]);
    let role_aware = detect_output_bias_drift_with_descriptor(
        &creature,
        &[("output-1".to_string(), records)],
        &mse,
    );

    assert_eq!(legacy.len(), role_aware.len());
    for c in &role_aware {
        assert!(
            !c.capacity_starved,
            "MSE descriptor must never set capacity_starved",
        );
    }
}

// =============================================================================
// 7. Class with no positive support is not flagged even when unsaturated.
// =============================================================================
#[test]
fn class_without_positive_support_is_not_flagged() {
    let creature = creature_with_one_output();
    // All records have target=0.0 — no positive support for this class.
    let records: Vec<DiscoverRecord> = (0..60)
        .map(|i| make_record("output-1", i, 0.0, 0.05, -0.05))
        .collect();
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 1);

    let candidates = detect_output_bias_drift_with_descriptor(
        &creature,
        &[("output-1".to_string(), records)],
        &descriptor,
    );

    for c in &candidates {
        assert!(
            !c.capacity_starved,
            "Class with no positive support must not be flagged capacity_starved",
        );
    }
}

// =============================================================================
// 8. Weight-up: starved candidate has higher estimated_improvement than the
//    same neuron under the legacy detector.
// =============================================================================
#[test]
fn capacity_starved_candidate_is_weighted_up_vs_legacy() {
    let creature = creature_with_one_output();
    let records = capacity_starved_and_biased_records("output-1");
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 1);

    let legacy = detect_output_bias_drift(&creature, &[("output-1".to_string(), records.clone())]);
    let role_aware = detect_output_bias_drift_with_descriptor(
        &creature,
        &[("output-1".to_string(), records)],
        &descriptor,
    );

    let legacy_gain = legacy
        .iter()
        .find(|c| c.neuron_uuid == "output-1")
        .map(|c| c.estimated_improvement)
        .expect("Legacy detector must emit a candidate for the biased output neuron");
    let role_gain = role_aware
        .iter()
        .find(|c| c.neuron_uuid == "output-1")
        .map(|c| c.estimated_improvement)
        .expect("Role-aware detector must emit a candidate for the biased output neuron");

    assert!(
        role_gain > legacy_gain,
        "capacity-starved candidate must be weighted up: role_gain={role_gain}, legacy_gain={legacy_gain}",
    );
}

// =============================================================================
// 9. Hidden neurons are excluded from capacity-starvation flagging.
// =============================================================================
#[test]
fn hidden_neuron_is_not_flagged_capacity_starved() {
    // A creature with a hidden neuron; output bias drift only looks at outputs
    // anyway, so this is mostly a sanity check that the role-aware path
    // doesn't accidentally pull hidden neurons into the candidate set.
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("hidden-1", "hidden", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        synapses: vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.5),
        ],
        input: 1,
        output: 1,
    };
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 1);

    // Hidden neuron records also have positive support and low activation,
    // but the output detector must ignore them.
    let hidden_records = capacity_starved_records("hidden-1");
    let candidates = detect_output_bias_drift_with_descriptor(
        &creature,
        &[("hidden-1".to_string(), hidden_records)],
        &descriptor,
    );

    assert!(
        candidates.iter().all(|c| c.neuron_uuid != "hidden-1"),
        "Hidden neurons must never appear in output_bias_drift candidates",
    );
}
