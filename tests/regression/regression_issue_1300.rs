//! Regression tests for Issue #1300: squash-bounded influence and min-gate awareness.
//!
//! Mirrors the fix from NEAT-AI-Explore#266. The impact attribution previously had
//! three correctness gaps that this fix addresses:
//!
//! 1. **Squash-bounded contribution** — the sum of inbound contributions to a neuron
//!    must respect the downstream squash's emit magnitude (e.g. TANH caps at ±1).
//!    Previously, threshold squashes (STEP/BIPOLAR) returned the full `child_impact`
//!    for every inbound synapse, so the sum was `N × child_impact`. The new code
//!    normalises by inbound weights and caps by the squash's emit magnitude.
//!
//! 2. **Min-gate awareness via consumer contract** — outputs gated by an external
//!    `min(output, constant)` only drive the downstream value when the network
//!    output is the smaller operand. The new `ConsumerContract` API lets callers
//!    declare such gates so impact attribution scales by the gate's pass-through
//!    probability.
//!
//! 3. **Auto-derived regime thresholds** — `derive_regime_threshold_from_records`
//!    picks a percentile from the recorded distribution, so callers do not need
//!    to hard-code a "low volume" constant.

#![allow(clippy::cast_precision_loss)] // Intentional u32→f32 casts for test data construction

use neat_ai_discovery::activations::squash_emit_magnitude;
use neat_ai_discovery::focus::{
    ConsumerContract, OutputGate, compute_impacts_public, compute_impacts_with_contract,
    derive_regime_threshold_from_records,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn hidden(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "hidden".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

fn input_n(uuid: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "input".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn output(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "output".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
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

// ---------------------------------------------------------------------------
// 1. Squash emit magnitude lookup
// ---------------------------------------------------------------------------

#[test]
fn squash_emit_magnitude_known_bounds() {
    // Bounded ±1 squashes
    assert_eq!(squash_emit_magnitude("TANH"), Some(1.0));
    assert_eq!(squash_emit_magnitude("LOGISTIC"), Some(1.0));
    assert_eq!(squash_emit_magnitude("HARD_TANH"), Some(1.0));
    assert_eq!(squash_emit_magnitude("CLIPPED"), Some(1.0));
    assert_eq!(squash_emit_magnitude("SOFTSIGN"), Some(1.0));
    assert_eq!(squash_emit_magnitude("STEP"), Some(1.0));
    assert_eq!(squash_emit_magnitude("BIPOLAR"), Some(1.0));
    assert_eq!(squash_emit_magnitude("GAUSSIAN"), Some(1.0));

    // RELU6 clamps to [0, 6]
    assert_eq!(squash_emit_magnitude("RELU6"), Some(6.0));

    // Unbounded squashes
    assert_eq!(squash_emit_magnitude("IDENTITY"), None);
    assert_eq!(squash_emit_magnitude("RELU"), None);
    assert_eq!(squash_emit_magnitude("ELU"), None);
    assert_eq!(squash_emit_magnitude("LEAKYRELU"), None);
    assert_eq!(squash_emit_magnitude("CUBE"), None);
    assert_eq!(squash_emit_magnitude("SQUARE"), None);

    // Aggregate squashes — selection stats handle these.
    assert_eq!(squash_emit_magnitude("MINIMUM"), None);
    assert_eq!(squash_emit_magnitude("MAXIMUM"), None);
    assert_eq!(squash_emit_magnitude("IF"), None);

    // Case-insensitive lookup.
    assert_eq!(squash_emit_magnitude("tanh"), Some(1.0));
    assert_eq!(squash_emit_magnitude("Tanh"), Some(1.0));
}

// ---------------------------------------------------------------------------
// 2. Squash-bounded contribution: TANH feeding many outputs
// ---------------------------------------------------------------------------

/// A saturated TANH hidden neuron that feeds many outputs would previously
/// accumulate `child_impact = N × 1.0 = N` from N outputs, and the sum of
/// inbound contributions could exceed ±1 — but TANH cannot emit more than ±1.
///
/// Acceptance criterion (Issue #1300): the sum of inbound contributions to a
/// bounded-squash neuron stays inside `[-M, M]` where `M` is the squash's emit
/// magnitude.
#[test]
fn tanh_inbound_contributions_stay_inside_emit_range() {
    // Network: many inputs each feed a single TANH hidden with weight `10` (so
    // it would saturate). The TANH then feeds 5 outputs with weight `1` each.
    // child_impact(TANH) = 5 outputs × 1.0 = 5.0 (multi-output hub).
    // Without squash bounding, sum_inbound = child_impact = 5.0 > 1.0.
    // With the fix, sum_inbound is capped at M = 1.0.
    let num_inputs = 10;
    let num_outputs = 5;

    let mut neurons = vec![hidden("tanh-hub", "TANH")];
    let mut synapses = Vec::new();
    for i in 0..num_inputs {
        neurons.push(input_n(&format!("input-{i}")));
        synapses.push(synapse(&format!("input-{i}"), "tanh-hub", 10.0));
    }
    for o in 0..num_outputs {
        let uuid = format!("output-{o}");
        neurons.push(output(&uuid, "IDENTITY"));
        synapses.push(synapse("tanh-hub", &uuid, 1.0));
    }
    let creature = CreatureJson {
        input: num_inputs,
        output: num_outputs,
        neurons,
        synapses,
    };

    let impacts = compute_impacts_public(&creature);

    // Sum of inbound impacts to the TANH hub = sum of input neurons' impacts
    // (each input has only one path, through the TANH).
    let sum_inbound: f32 = (0..num_inputs)
        .map(|i| impacts.get(&format!("input-{i}")).copied().unwrap_or(0.0))
        .sum();

    let m = squash_emit_magnitude("TANH").unwrap();
    assert!(
        sum_inbound <= m + 1e-5,
        "sum of inbound contributions to TANH ({sum_inbound}) must not exceed emit magnitude {m}"
    );
}

/// STEP/BIPOLAR previously returned `child_impact` for every inbound synapse —
/// so 10 inbound synapses to a STEP feeding one output would each get
/// `child_impact = 1.0`, summing to `10.0`. With Issue #1300, the sum is
/// bounded by `STEP`'s emit magnitude (`1.0`).
#[test]
fn step_inbound_contributions_stay_bounded() {
    let num_inputs = 10;
    let mut neurons = vec![hidden("step-hub", "STEP"), output("output-0", "IDENTITY")];
    let mut synapses = Vec::new();
    for i in 0..num_inputs {
        neurons.push(input_n(&format!("input-{i}")));
        synapses.push(synapse(&format!("input-{i}"), "step-hub", 1.0));
    }
    synapses.push(synapse("step-hub", "output-0", 1.0));

    let creature = CreatureJson {
        input: num_inputs,
        output: 1,
        neurons,
        synapses,
    };

    let impacts = compute_impacts_public(&creature);

    let sum_inbound: f32 = (0..num_inputs)
        .map(|i| impacts.get(&format!("input-{i}")).copied().unwrap_or(0.0))
        .sum();

    let m = squash_emit_magnitude("STEP").unwrap();
    assert!(
        sum_inbound <= m + 1e-5,
        "sum of inbound contributions to STEP ({sum_inbound}) must not exceed emit magnitude {m}"
    );
}

/// Sole-input case: a single synapse to a bounded squash should still get its
/// full impact (the cap is per-aggregate, not per-synapse).
#[test]
fn sole_inbound_synapse_unaffected_by_squash_bounding() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![hidden("tanh-1", "TANH"), output("output-0", "IDENTITY")],
        synapses: vec![
            synapse("input-0", "tanh-1", 1.0),
            synapse("tanh-1", "output-0", 1.0),
        ],
    };
    let impacts = compute_impacts_public(&creature);
    let tanh_impact = impacts.get("tanh-1").copied().unwrap_or(0.0);

    // Sole synapse feeding sole output: impact = 1.0 (unaffected by capping).
    assert!(
        (tanh_impact - 1.0).abs() < 1e-5,
        "sole-input TANH should retain full impact 1.0, got {tanh_impact}"
    );
}

// ---------------------------------------------------------------------------
// 3. Min-gate (consumer contract) awareness
// ---------------------------------------------------------------------------

/// In-memory `RecordProvider` for testing.
struct MapRecordProvider {
    inner: HashMap<String, Arc<Vec<DiscoverRecord>>>,
}

impl MapRecordProvider {
    fn new(map: HashMap<String, Vec<DiscoverRecord>>) -> Self {
        Self {
            inner: map.into_iter().map(|(k, v)| (k, Arc::new(v))).collect(),
        }
    }
}

impl neat_ai_discovery::focus::RecordProvider for MapRecordProvider {
    fn get(&self, neuron_uuid: &str) -> anyhow::Result<Option<Arc<Vec<DiscoverRecord>>>> {
        Ok(self.inner.get(neuron_uuid).cloned())
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[test]
fn min_gate_against_constant_masks_impact_outside_regime() {
    // Network: input -> hidden -> output. Without a contract, hidden has
    // impact 1.0. With a `MinAgainstConstant(threshold)` gate that lets
    // 20% of observations through, hidden's impact must scale to 0.2.
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("hidden-1", "IDENTITY"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "hidden-1", 1.0),
            synapse("hidden-1", "output-0", 1.0),
        ],
    };

    // 100 observations: 20 below the gate threshold (gate lets them through),
    // 80 above (gate masks the network output).
    let mut output_records = Vec::new();
    for i in 0..100u32 {
        let activation = if i < 20 { 0.1 } else { 1.0 };
        output_records.push(DiscoverRecord::new(
            i,
            "output-0".to_string(),
            Some(activation),
            activation,
            vec![0.0],
        ));
    }
    let mut hidden_records = Vec::new();
    for i in 0..100u32 {
        hidden_records.push(DiscoverRecord::new(
            i,
            "hidden-1".to_string(),
            Some(0.5),
            0.5,
            vec![0.0],
        ));
    }
    let mut input_records = Vec::new();
    for i in 0..100u32 {
        input_records.push(DiscoverRecord::new(
            i,
            "input-0".to_string(),
            Some(0.5),
            0.5,
            vec![],
        ));
    }
    let mut map: HashMap<String, Vec<DiscoverRecord>> = HashMap::new();
    map.insert("output-0".to_string(), output_records);
    map.insert("hidden-1".to_string(), hidden_records);
    map.insert("input-0".to_string(), input_records);
    let provider = MapRecordProvider::new(map);

    // Baseline: no contract → hidden impact = 1.0.
    let baseline = compute_impacts_with_contract(&creature, Some(&provider), None).unwrap();
    let baseline_hidden = baseline.get("hidden-1").copied().unwrap_or(0.0);
    assert!(
        (baseline_hidden - 1.0).abs() < 1e-5,
        "baseline hidden impact should be 1.0, got {baseline_hidden}"
    );

    // With gate: only 20/100 = 0.2 of observations let the output drive.
    let contract =
        ConsumerContract::new().with_gate("output-0", OutputGate::MinAgainstConstant(0.5));
    let gated = compute_impacts_with_contract(&creature, Some(&provider), Some(&contract)).unwrap();
    let gated_hidden = gated.get("hidden-1").copied().unwrap_or(0.0);
    assert!(
        (gated_hidden - 0.2).abs() < 1e-3,
        "gated hidden impact should be 0.2 (gate pass = 0.2), got {gated_hidden}"
    );
}

#[test]
fn identity_gate_does_not_change_impact() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("hidden-1", "IDENTITY"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "hidden-1", 1.0),
            synapse("hidden-1", "output-0", 1.0),
        ],
    };
    let contract = ConsumerContract::new().with_gate("output-0", OutputGate::Identity);
    let with_identity = compute_impacts_with_contract(&creature, None, Some(&contract)).unwrap();
    let without = compute_impacts_with_contract(&creature, None, None).unwrap();
    for uuid in ["hidden-1", "output-0"] {
        let a = with_identity.get(uuid).copied().unwrap_or(0.0);
        let b = without.get(uuid).copied().unwrap_or(0.0);
        assert!((a - b).abs() < 1e-5, "{uuid}: identity gate changed impact");
    }
}

// ---------------------------------------------------------------------------
// 4. Auto-derived regime threshold
// ---------------------------------------------------------------------------

#[test]
fn derive_regime_threshold_percentile_basic() {
    let records: Vec<DiscoverRecord> = (0..100u32)
        .map(|i| DiscoverRecord::new(i, "n".to_string(), Some(i as f32), i as f32, vec![]))
        .collect();

    // 0.0 → minimum
    assert_eq!(
        derive_regime_threshold_from_records(&records, 0.0),
        Some(0.0)
    );
    // 1.0 → maximum
    assert_eq!(
        derive_regime_threshold_from_records(&records, 1.0),
        Some(99.0)
    );
    // 0.25 → ~25
    let p25 = derive_regime_threshold_from_records(&records, 0.25).unwrap();
    assert!((p25 - 25.0).abs() <= 1.0, "p25 should be ~25, got {p25}");
}

#[test]
fn derive_regime_threshold_empty_returns_none() {
    assert_eq!(derive_regime_threshold_from_records(&[], 0.5), None);
}

#[test]
fn derive_regime_threshold_filters_non_finite() {
    let records = vec![
        DiscoverRecord::new(0, "n".into(), None, f32::NAN, vec![]),
        DiscoverRecord::new(1, "n".into(), None, f32::INFINITY, vec![]),
        DiscoverRecord::new(2, "n".into(), None, 5.0, vec![]),
    ];
    assert_eq!(
        derive_regime_threshold_from_records(&records, 0.0),
        Some(5.0)
    );
}
