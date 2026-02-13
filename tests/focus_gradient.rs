//! Issue #523: Targeted tests for the focus `gradient` sub-module.
//!
//! Tests gradient flow analysis with known activation functions and weights:
//! - RELU dead neuron detection
//! - TANH saturation detection
//! - LOGISTIC saturation detection
//! - IDENTITY (always gradient = 1.0)
//! - Mixed activation creatures
//!
//! Since `compute_activation_gradient` and `compute_gradient_flow_factor` are
//! not public, we test them indirectly via `compute_gradient_flow_stats` which
//! reads from a parquet file.

mod common;

use neat_ai_discovery::focus::{GradientFlowStats, compute_gradient_flow_stats};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Build a creature with the given hidden neurons (each with a specified squash)
/// plus one output neuron.
fn make_creature_with_squashes(neurons: &[(&str, &str)]) -> CreatureJson {
    let mut all_neurons: Vec<NeuronJson> = neurons
        .iter()
        .map(|(uuid, squash)| NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: "hidden".to_string(),
            squash: squash.to_string(),
            bias: 0.0,
        })
        .collect();

    all_neurons.push(NeuronJson {
        uuid: "out".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    let mut synapses: Vec<SynapseJson> = neurons
        .iter()
        .map(|(uuid, _)| SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: uuid.to_string(),
            weight: 1.0,
            synapse_type: None,
        })
        .collect();

    for (uuid, _) in neurons {
        synapses.push(SynapseJson {
            from_uuid: uuid.to_string(),
            to_uuid: "out".to_string(),
            weight: 1.0,
            synapse_type: None,
        });
    }

    CreatureJson {
        input: 1,
        output: 1,
        neurons: all_neurons,
        synapses,
    }
}

/// Write records to a temporary parquet file and return the path.
fn write_temp_parquet(records: &[DiscoverRecord]) -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("create temp file");
    write_records_to_parquet(tmp.path().to_str().unwrap(), records).expect("write parquet");
    tmp
}

/// Create records for a neuron with given pre-activation values and activations.
fn make_records(uuid: &str, values_and_activations: &[(f32, f32)]) -> Vec<DiscoverRecord> {
    values_and_activations
        .iter()
        .enumerate()
        .map(|(i, &(value, activation))| DiscoverRecord {
            obs_index: i as u32,
            neuron_uuid: uuid.to_string(),
            value: Some(value),
            activation,
            errors: vec![0.1],
        })
        .collect()
}

// =============================================================================
// RELU — dead neuron detection
// =============================================================================

#[test]
fn relu_all_negative_values_fully_dead() {
    let creature = make_creature_with_squashes(&[("relu-1", "RELU")]);

    // All pre-activation values negative → gradient = 0, dead = true
    let mut records = make_records("relu-1", &[(-5.0, 0.0), (-2.0, 0.0), (-0.1, 0.0)]);
    // Output neuron needs records too
    records.extend(make_records("out", &[(1.0, 1.0)]));

    let tmp = write_temp_parquet(&records);
    let stats = compute_gradient_flow_stats(tmp.path().to_str().unwrap(), &creature)
        .expect("compute gradient stats");

    let relu_stats = stats.get("relu-1").expect("relu-1 stats present");
    assert!(
        relu_stats.dead_ratio > 0.99,
        "All-negative RELU should be fully dead, got dead_ratio={}",
        relu_stats.dead_ratio
    );
    assert!(
        relu_stats.avg_gradient_magnitude < 0.01,
        "All-negative RELU should have near-zero gradient, got {}",
        relu_stats.avg_gradient_magnitude
    );
}

#[test]
fn relu_all_positive_values_not_dead() {
    let creature = make_creature_with_squashes(&[("relu-1", "RELU")]);

    let mut records = make_records("relu-1", &[(1.0, 1.0), (5.0, 5.0), (0.5, 0.5)]);
    records.extend(make_records("out", &[(1.0, 1.0)]));

    let tmp = write_temp_parquet(&records);
    let stats = compute_gradient_flow_stats(tmp.path().to_str().unwrap(), &creature)
        .expect("compute gradient stats");

    let relu_stats = stats.get("relu-1").expect("relu-1 stats present");
    assert!(
        relu_stats.dead_ratio < 0.01,
        "All-positive RELU should have zero dead ratio, got {}",
        relu_stats.dead_ratio
    );
    assert!(
        (relu_stats.avg_gradient_magnitude - 1.0).abs() < 0.01,
        "Positive RELU gradient should be 1.0, got {}",
        relu_stats.avg_gradient_magnitude
    );
}

#[test]
fn relu_mixed_values_partial_dead() {
    let creature = make_creature_with_squashes(&[("relu-1", "RELU")]);

    // 2 positive, 2 negative → 50% dead
    let mut records = make_records(
        "relu-1",
        &[(1.0, 1.0), (-1.0, 0.0), (2.0, 2.0), (-3.0, 0.0)],
    );
    records.extend(make_records("out", &[(1.0, 1.0)]));

    let tmp = write_temp_parquet(&records);
    let stats = compute_gradient_flow_stats(tmp.path().to_str().unwrap(), &creature)
        .expect("compute gradient stats");

    let relu_stats = stats.get("relu-1").expect("relu-1 stats present");
    assert!(
        (relu_stats.dead_ratio - 0.5).abs() < 0.01,
        "Half-dead RELU should have dead_ratio ≈ 0.5, got {}",
        relu_stats.dead_ratio
    );
}

// =============================================================================
// TANH — saturation detection
// =============================================================================

#[test]
fn tanh_extreme_values_saturated() {
    let creature = make_creature_with_squashes(&[("tanh-1", "TANH")]);

    // |value| > 3.0 → saturated for TANH
    let mut records = make_records(
        "tanh-1",
        &[(10.0, 1.0), (-10.0, -1.0), (5.0, 1.0), (-5.0, -1.0)],
    );
    records.extend(make_records("out", &[(1.0, 1.0)]));

    let tmp = write_temp_parquet(&records);
    let stats = compute_gradient_flow_stats(tmp.path().to_str().unwrap(), &creature)
        .expect("compute gradient stats");

    let tanh_stats = stats.get("tanh-1").expect("tanh-1 stats present");
    assert!(
        tanh_stats.saturation_ratio > 0.99,
        "Extreme TANH values should be fully saturated, got {}",
        tanh_stats.saturation_ratio
    );
}

#[test]
fn tanh_moderate_values_not_saturated() {
    let creature = make_creature_with_squashes(&[("tanh-1", "TANH")]);

    // |value| < 1.0 → good gradient region for TANH
    let mut records = make_records(
        "tanh-1",
        &[(0.1, 0.1), (-0.2, -0.2), (0.5, 0.46), (-0.5, -0.46)],
    );
    records.extend(make_records("out", &[(1.0, 1.0)]));

    let tmp = write_temp_parquet(&records);
    let stats = compute_gradient_flow_stats(tmp.path().to_str().unwrap(), &creature)
        .expect("compute gradient stats");

    let tanh_stats = stats.get("tanh-1").expect("tanh-1 stats present");
    assert!(
        tanh_stats.saturation_ratio < 0.01,
        "Moderate TANH values should not be saturated, got {}",
        tanh_stats.saturation_ratio
    );
    assert!(
        tanh_stats.avg_gradient_magnitude > 0.5,
        "Moderate TANH should have good gradient flow, got {}",
        tanh_stats.avg_gradient_magnitude
    );
}

// =============================================================================
// LOGISTIC — saturation detection
// =============================================================================

#[test]
fn logistic_extreme_values_saturated() {
    let creature = make_creature_with_squashes(&[("sig-1", "LOGISTIC")]);

    // |value| > 5.0 → saturated for LOGISTIC
    let mut records = make_records("sig-1", &[(20.0, 1.0), (-20.0, 0.0), (10.0, 1.0)]);
    records.extend(make_records("out", &[(1.0, 1.0)]));

    let tmp = write_temp_parquet(&records);
    let stats = compute_gradient_flow_stats(tmp.path().to_str().unwrap(), &creature)
        .expect("compute gradient stats");

    let sig_stats = stats.get("sig-1").expect("sig-1 stats present");
    assert!(
        sig_stats.saturation_ratio > 0.99,
        "Extreme LOGISTIC values should be fully saturated, got {}",
        sig_stats.saturation_ratio
    );
}

// =============================================================================
// IDENTITY — constant gradient
// =============================================================================

#[test]
fn identity_always_has_gradient_one() {
    let creature = make_creature_with_squashes(&[("id-1", "IDENTITY")]);

    let mut records = make_records("id-1", &[(100.0, 100.0), (-50.0, -50.0), (0.0, 0.0)]);
    records.extend(make_records("out", &[(1.0, 1.0)]));

    let tmp = write_temp_parquet(&records);
    let stats = compute_gradient_flow_stats(tmp.path().to_str().unwrap(), &creature)
        .expect("compute gradient stats");

    let id_stats = stats.get("id-1").expect("id-1 stats present");
    assert!(
        (id_stats.avg_gradient_magnitude - 1.0).abs() < 0.01,
        "IDENTITY gradient should be 1.0, got {}",
        id_stats.avg_gradient_magnitude
    );
    assert!(
        id_stats.dead_ratio < 0.01,
        "IDENTITY should never be dead, got {}",
        id_stats.dead_ratio
    );
}

// =============================================================================
// Mixed activations
// =============================================================================

#[test]
fn mixed_activations_produce_independent_stats() {
    let creature = make_creature_with_squashes(&[
        ("relu-1", "RELU"),
        ("tanh-1", "TANH"),
        ("id-1", "IDENTITY"),
    ]);

    let mut records = Vec::new();
    // RELU with all-negative → fully dead
    records.extend(make_records("relu-1", &[(-1.0, 0.0), (-5.0, 0.0)]));
    // TANH with moderate values → good gradient
    records.extend(make_records("tanh-1", &[(0.1, 0.1), (0.2, 0.2)]));
    // IDENTITY → always gradient 1.0
    records.extend(make_records("id-1", &[(3.0, 3.0), (7.0, 7.0)]));
    // Output
    records.extend(make_records("out", &[(1.0, 1.0)]));

    let tmp = write_temp_parquet(&records);
    let stats = compute_gradient_flow_stats(tmp.path().to_str().unwrap(), &creature)
        .expect("compute gradient stats");

    // RELU should be fully dead
    let relu = stats.get("relu-1").expect("relu-1");
    assert!(relu.dead_ratio > 0.99, "RELU should be dead");

    // TANH should have good gradient and no dead
    let tanh = stats.get("tanh-1").expect("tanh-1");
    assert!(tanh.dead_ratio < 0.01, "TANH should not be dead");
    assert!(
        tanh.avg_gradient_magnitude > 0.5,
        "TANH should have decent gradient"
    );

    // IDENTITY should have gradient 1.0
    let id = stats.get("id-1").expect("id-1");
    assert!(
        (id.avg_gradient_magnitude - 1.0).abs() < 0.01,
        "IDENTITY gradient should be 1.0"
    );
}

// =============================================================================
// GradientFlowStats default values
// =============================================================================

#[test]
fn gradient_flow_stats_default_assumes_full_flow() {
    let default = GradientFlowStats::default();

    assert!(
        (default.avg_gradient_magnitude - 1.0).abs() < f32::EPSILON,
        "Default gradient magnitude should be 1.0 (assume full gradient)"
    );
    assert!(
        default.saturation_ratio.abs() < f32::EPSILON,
        "Default saturation should be 0.0 (assume not saturated)"
    );
    assert!(
        default.dead_ratio.abs() < f32::EPSILON,
        "Default dead ratio should be 0.0 (assume not dead)"
    );
}

// =============================================================================
// Edge: neuron with no finite values
// =============================================================================

#[test]
fn neuron_with_no_finite_values_gets_default_stats() {
    let creature = make_creature_with_squashes(&[("nan-1", "RELU")]);

    let mut records: Vec<DiscoverRecord> = (0..3)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "nan-1".to_string(),
            value: Some(f32::NAN), // Non-finite values
            activation: 0.0,
            errors: vec![0.1],
        })
        .collect();
    records.extend(make_records("out", &[(1.0, 1.0)]));

    let tmp = write_temp_parquet(&records);
    let stats = compute_gradient_flow_stats(tmp.path().to_str().unwrap(), &creature)
        .expect("compute gradient stats");

    let nan_stats = stats.get("nan-1").expect("nan-1 should be present");
    // With no valid samples, should get default stats
    assert!(
        (nan_stats.avg_gradient_magnitude - 1.0).abs() < f32::EPSILON,
        "No valid samples should use default gradient (1.0), got {}",
        nan_stats.avg_gradient_magnitude
    );
}

// =============================================================================
// ELU — non-zero gradient for negative values
// =============================================================================

#[test]
fn elu_negative_values_have_non_zero_gradient() {
    let creature = make_creature_with_squashes(&[("elu-1", "ELU")]);

    // ELU has exp(x) gradient for negative values → never truly dead
    let mut records = make_records("elu-1", &[(-0.5, -0.39), (-1.0, -0.63), (-2.0, -0.86)]);
    records.extend(make_records("out", &[(1.0, 1.0)]));

    let tmp = write_temp_parquet(&records);
    let stats = compute_gradient_flow_stats(tmp.path().to_str().unwrap(), &creature)
        .expect("compute gradient stats");

    let elu_stats = stats.get("elu-1").expect("elu-1 present");
    assert!(
        elu_stats.dead_ratio < 0.01,
        "ELU should never be dead (has non-zero gradient for negative), got {}",
        elu_stats.dead_ratio
    );
    assert!(
        elu_stats.avg_gradient_magnitude > 0.0,
        "ELU should have positive gradient for negative values, got {}",
        elu_stats.avg_gradient_magnitude
    );
}
