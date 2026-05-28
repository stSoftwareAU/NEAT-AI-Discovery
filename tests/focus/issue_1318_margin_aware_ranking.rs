//! Issue #1318: Margin-aware candidate ranking for argmax/margin costs.
//!
//! Under a `OneHot` / `Margin` task descriptor, focus ranking should reflect
//! the **decision margin** (top-1 vs top-2 output activation) rather than the
//! raw residual. Observations whose decision is on the edge — where the margin
//! is small — contribute more weight to a candidate's score than observations
//! whose decision is already dominant.
//!
//! These tests verify:
//! 1. With a `OneHot` / `Margin` descriptor, the hidden neuron whose error
//!    lands on close-margin observations ranks above one whose error lands on
//!    wide-margin observations even when their unweighted mean errors match.
//! 2. With `Independent` / `Simplex` / `Unknown` / `OTHER` / absent descriptors
//!    the legacy unweighted ranking is preserved (regression guard).
//! 3. [`compute_per_obs_margins`] correctly extracts the top-1 minus top-2 gap
//!    from recorded output activations.

#![allow(clippy::cast_precision_loss)] // Test arithmetic only.
use std::sync::Arc;

use neat_ai_discovery::analysis::task_descriptor::{
    OutputSquashFamily, TargetRange, TargetTopology, TaskDescriptor,
};
use neat_ai_discovery::focus::{
    MARGIN_WEIGHT_EPS, compute_per_obs_margins, margin_weights_from_margins, rank_focus_neurons,
    rank_focus_neurons_with_descriptor,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Build a creature with three outputs and two hidden neurons. Each hidden
/// neuron feeds all three outputs so structural impacts are roughly balanced.
fn make_three_class_creature() -> CreatureJson {
    let neurons = vec![
        NeuronJson {
            uuid: "h1".into(),
            neuron_type: "hidden".into(),
            squash: "IDENTITY".into(),
            bias: 0.0,
        },
        NeuronJson {
            uuid: "h2".into(),
            neuron_type: "hidden".into(),
            squash: "IDENTITY".into(),
            bias: 0.0,
        },
        NeuronJson {
            uuid: "out-0".into(),
            neuron_type: "output".into(),
            squash: "LOGISTIC".into(),
            bias: 0.0,
        },
        NeuronJson {
            uuid: "out-1".into(),
            neuron_type: "output".into(),
            squash: "LOGISTIC".into(),
            bias: 0.0,
        },
        NeuronJson {
            uuid: "out-2".into(),
            neuron_type: "output".into(),
            squash: "LOGISTIC".into(),
            bias: 0.0,
        },
    ];
    let synapses = vec![
        SynapseJson {
            from_uuid: "input-0".into(),
            to_uuid: "h1".into(),
            weight: 1.0,
            synapse_type: None,
        },
        SynapseJson {
            from_uuid: "input-0".into(),
            to_uuid: "h2".into(),
            weight: 1.0,
            synapse_type: None,
        },
        SynapseJson {
            from_uuid: "h1".into(),
            to_uuid: "out-0".into(),
            weight: 0.5,
            synapse_type: None,
        },
        SynapseJson {
            from_uuid: "h1".into(),
            to_uuid: "out-1".into(),
            weight: 0.5,
            synapse_type: None,
        },
        SynapseJson {
            from_uuid: "h1".into(),
            to_uuid: "out-2".into(),
            weight: 0.5,
            synapse_type: None,
        },
        SynapseJson {
            from_uuid: "h2".into(),
            to_uuid: "out-0".into(),
            weight: 0.5,
            synapse_type: None,
        },
        SynapseJson {
            from_uuid: "h2".into(),
            to_uuid: "out-1".into(),
            weight: 0.5,
            synapse_type: None,
        },
        SynapseJson {
            from_uuid: "h2".into(),
            to_uuid: "out-2".into(),
            weight: 0.5,
            synapse_type: None,
        },
    ];
    CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 3,
    }
}

fn record(obs_index: u32, uuid: &str, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.into(),
        value: Some(activation),
        activation,
        errors: vec![error],
    }
}

fn write_temp_parquet(records: &[DiscoverRecord]) -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("create temp file");
    write_records_to_parquet(tmp.path().to_str().unwrap(), records).expect("write parquet");
    tmp
}

/// Build the standard fixture used by the ranking tests: two observations,
/// one with a close margin and one with a wide margin. h1 has its error
/// concentrated on the close-margin observation, h2 on the wide-margin one.
fn build_margin_fixture() -> (CreatureJson, NamedTempFile) {
    let creature = make_three_class_creature();

    // Observation 0 — close margin: top1=0.51 (out-0), top2=0.50 (out-1).
    // Observation 1 — wide margin:  top1=0.95 (out-0), top2=0.10 (out-1).
    let records = vec![
        // out-0
        record(0, "out-0", 0.51, 0.0),
        record(1, "out-0", 0.95, 0.0),
        // out-1
        record(0, "out-1", 0.50, 0.0),
        record(1, "out-1", 0.10, 0.0),
        // out-2
        record(0, "out-2", 0.05, 0.0),
        record(1, "out-2", 0.05, 0.0),
        // h1 has high error on close-margin obs, low error on wide obs.
        record(0, "h1", 0.5, 0.9),
        record(1, "h1", 0.5, 0.1),
        // h2 has the mirror: low error on close-margin obs, high error on wide.
        record(0, "h2", 0.5, 0.1),
        record(1, "h2", 0.5, 0.9),
    ];

    let tmp = write_temp_parquet(&records);
    (creature, tmp)
}

// =============================================================================
// compute_per_obs_margins
// =============================================================================

#[test]
fn margins_reflect_top1_minus_top2_per_observation() {
    let (creature, tmp) = build_margin_fixture();

    // Re-load the parquet via an eager provider through the public ranking API
    // — we go through the public path so we know we are exercising the same
    // record-provider used in production.
    use neat_ai_discovery::parquet_format::read_all_records_grouped_by_neuron;
    let grouped =
        read_all_records_grouped_by_neuron(tmp.path().to_str().unwrap()).expect("read records");

    struct InMemoryProvider {
        records: std::collections::HashMap<String, Arc<Vec<DiscoverRecord>>>,
    }

    impl neat_ai_discovery::focus::RecordProvider for InMemoryProvider {
        fn get(&self, neuron_uuid: &str) -> anyhow::Result<Option<Arc<Vec<DiscoverRecord>>>> {
            Ok(self.records.get(neuron_uuid).cloned())
        }
        fn len(&self) -> usize {
            self.records.len()
        }
    }

    let provider = InMemoryProvider {
        records: grouped.into_iter().map(|(k, v)| (k, Arc::new(v))).collect(),
    };

    let margins = compute_per_obs_margins(&creature, &provider).expect("margins");
    assert!(
        (margins[&0] - 0.01).abs() < 1e-4,
        "obs 0 margin: {}",
        margins[&0]
    );
    assert!(
        (margins[&1] - 0.85).abs() < 1e-4,
        "obs 1 margin: {}",
        margins[&1]
    );
}

#[test]
fn margin_weights_upweight_small_margins() {
    let mut margins = std::collections::HashMap::new();
    margins.insert(0u32, 0.01_f32);
    margins.insert(1u32, 0.85_f32);

    let weights = margin_weights_from_margins(&margins);

    let w_close = weights[&0];
    let w_wide = weights[&1];

    assert!(
        w_close > w_wide,
        "close-margin weight {w_close} must exceed wide-margin weight {w_wide}"
    );
    // Sanity check the exact formula: w = 1 / (margin + eps).
    assert!(
        (w_close - 1.0 / (0.01 + MARGIN_WEIGHT_EPS)).abs() < 1e-4,
        "close weight does not match formula: {w_close}"
    );
}

// =============================================================================
// Ranking under OneHot / Margin descriptors
// =============================================================================

fn one_hot_descriptor() -> TaskDescriptor {
    TaskDescriptor {
        target_topology: TargetTopology::OneHot,
        target_range: TargetRange::Unit,
        output_squash_family: OutputSquashFamily::BoundedUnipolar,
        num_outputs: 3,
    }
}

fn margin_descriptor() -> TaskDescriptor {
    TaskDescriptor {
        target_topology: TargetTopology::Margin,
        target_range: TargetRange::SignedUnit,
        output_squash_family: OutputSquashFamily::BoundedBipolar,
        num_outputs: 3,
    }
}

#[test]
fn one_hot_descriptor_promotes_close_margin_error_neuron() {
    let (creature, tmp) = build_margin_fixture();
    let descriptor = one_hot_descriptor();

    let result = rank_focus_neurons_with_descriptor(
        tmp.path().to_str().unwrap(),
        &creature,
        None,
        None,
        Some(&descriptor),
    )
    .expect("rank focus neurons");

    let h1_pos = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "h1")
        .expect("h1 ranked");
    let h2_pos = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "h2")
        .expect("h2 ranked");

    assert!(
        h1_pos < h2_pos,
        "Under OneHot, h1 (error on close margin) at {h1_pos} must rank above h2 \
         (error on wide margin) at {h2_pos}"
    );
}

#[test]
fn margin_descriptor_promotes_close_margin_error_neuron() {
    let (creature, tmp) = build_margin_fixture();
    let descriptor = margin_descriptor();

    let result = rank_focus_neurons_with_descriptor(
        tmp.path().to_str().unwrap(),
        &creature,
        None,
        None,
        Some(&descriptor),
    )
    .expect("rank focus neurons");

    let h1_pos = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "h1")
        .expect("h1 ranked");
    let h2_pos = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "h2")
        .expect("h2 ranked");

    assert!(
        h1_pos < h2_pos,
        "Under Margin, h1 (error on close margin) at {h1_pos} must rank above h2 \
         (error on wide margin) at {h2_pos}"
    );
}

#[test]
fn one_hot_descriptor_changes_total_error_for_close_margin_neuron() {
    // Direct numeric guard: under OneHot, h1's reweighted total_error must be
    // strictly larger than h2's. Under no descriptor the two are equal.
    let (creature, tmp) = build_margin_fixture();

    let baseline =
        rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None).expect("baseline");
    let h1_baseline = baseline
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "h1")
        .expect("h1");
    let h2_baseline = baseline
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "h2")
        .expect("h2");
    assert!(
        (h1_baseline.total_error - h2_baseline.total_error).abs() < 1e-6,
        "baseline: equal mean errors expected, got h1={} h2={}",
        h1_baseline.total_error,
        h2_baseline.total_error
    );

    let descriptor = one_hot_descriptor();
    let onehot = rank_focus_neurons_with_descriptor(
        tmp.path().to_str().unwrap(),
        &creature,
        None,
        None,
        Some(&descriptor),
    )
    .expect("onehot");
    let h1_onehot = onehot
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "h1")
        .expect("h1");
    let h2_onehot = onehot
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "h2")
        .expect("h2");
    assert!(
        h1_onehot.total_error > h2_onehot.total_error,
        "under OneHot: h1 total_error {} must exceed h2 {}",
        h1_onehot.total_error,
        h2_onehot.total_error,
    );
}

// =============================================================================
// Regression guard — non-margin topologies preserve legacy behaviour
// =============================================================================

#[test]
fn no_descriptor_matches_legacy_ranking() {
    let (creature, tmp) = build_margin_fixture();

    let legacy =
        rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None).expect("legacy");
    let with_none = rank_focus_neurons_with_descriptor(
        tmp.path().to_str().unwrap(),
        &creature,
        None,
        None,
        None,
    )
    .expect("with_none");

    assert_eq!(legacy.neurons.len(), with_none.neurons.len());
    for (a, b) in legacy.neurons.iter().zip(with_none.neurons.iter()) {
        assert_eq!(a.neuron_uuid, b.neuron_uuid, "ordering must match");
        assert!(
            (a.total_error - b.total_error).abs() < 1e-6,
            "total_error must match for {}",
            a.neuron_uuid
        );
    }
}

#[test]
fn unknown_descriptor_matches_legacy_ranking() {
    let (creature, tmp) = build_margin_fixture();

    let legacy =
        rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None).expect("legacy");

    let descriptor = TaskDescriptor::neutral(); // Unknown / OTHER / absent.
    let with_neutral = rank_focus_neurons_with_descriptor(
        tmp.path().to_str().unwrap(),
        &creature,
        None,
        None,
        Some(&descriptor),
    )
    .expect("with neutral descriptor");

    for (a, b) in legacy.neurons.iter().zip(with_neutral.neurons.iter()) {
        assert_eq!(a.neuron_uuid, b.neuron_uuid, "ordering must match");
        assert!(
            (a.total_error - b.total_error).abs() < 1e-6,
            "total_error must match for {}",
            a.neuron_uuid
        );
    }
}

#[test]
fn independent_descriptor_matches_legacy_ranking() {
    // MSE / MAE / BCE produce Independent topology; the margin path must NOT
    // engage here.
    let (creature, tmp) = build_margin_fixture();

    let legacy =
        rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None).expect("legacy");

    let descriptor = TaskDescriptor::from_name("MSE", 3);
    assert_eq!(descriptor.target_topology, TargetTopology::Independent);
    let with_indep = rank_focus_neurons_with_descriptor(
        tmp.path().to_str().unwrap(),
        &creature,
        None,
        None,
        Some(&descriptor),
    )
    .expect("with independent descriptor");

    for (a, b) in legacy.neurons.iter().zip(with_indep.neurons.iter()) {
        assert_eq!(a.neuron_uuid, b.neuron_uuid, "ordering must match");
        assert!(
            (a.total_error - b.total_error).abs() < 1e-6,
            "total_error must match for {}",
            a.neuron_uuid
        );
    }
}

#[test]
fn simplex_descriptor_matches_legacy_ranking() {
    // CROSS_ENTROPY yields Simplex topology — also not in the margin-aware
    // set (top-1 vs top-2 attribution does not match the softmax structure).
    let (creature, tmp) = build_margin_fixture();

    let legacy =
        rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None).expect("legacy");

    let descriptor = TaskDescriptor::from_name("CROSS_ENTROPY", 3);
    assert_eq!(descriptor.target_topology, TargetTopology::Simplex);
    let with_simplex = rank_focus_neurons_with_descriptor(
        tmp.path().to_str().unwrap(),
        &creature,
        None,
        None,
        Some(&descriptor),
    )
    .expect("with simplex descriptor");

    for (a, b) in legacy.neurons.iter().zip(with_simplex.neurons.iter()) {
        assert_eq!(a.neuron_uuid, b.neuron_uuid, "ordering must match");
        assert!(
            (a.total_error - b.total_error).abs() < 1e-6,
            "total_error must match for {}",
            a.neuron_uuid
        );
    }
}
