//! Integration tests: analytical dominated-branch collapse for MAX/MIN selection
//! aggregates (Issue #1711, gap **G1** of
//! `docs/DOMINATED_BRANCH_COLLAPSE_EXTENT.md`).
//!
//! The characterisation suite `tests/issue_1706_dominated_branch_characterisation.rs`
//! pins the *current-vs-target* gap: on the committed fixtures the dominated
//! branch is provably losing yet the engine collapses nothing. These tests drive
//! the **target** side of that gap: the new analytical detector flags the
//! dominated branch and the collapse transform reaches the full-collapse end
//! state, behind the #1623-style evaluate-before-accept gate.
//!
//! They load the *same* committed fixtures as the characterisation suite, so the
//! two stay anchored to one shared worked example.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::{
    AggregateKind, COLLAPSE_GATE_TOLERANCE, analytically_dominated_branch_uuids,
    collapse_dominated_branch, detect_dominated_branches,
};
use neat_ai_discovery::types::DiscoverRecord;
use std::path::{Path, PathBuf};

const ABS_UUID: &str = "neuron-abs";
const RELU_UUID: &str = "neuron-relu";

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dominated_branch_collapse")
}

/// Load a committed aggregate network fixture. Panics with a clear message if the
/// fixture is missing or malformed — the same drift guard as the characterisation
/// suite.
fn load_network(file: &str) -> CreatureJson {
    let path = fixture_root().join("networks").join(file);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing/unreadable fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("malformed fixture {}: {e}", path.display()))
}

fn record(obs: u32, uuid: &str, activation: f32) -> DiscoverRecord {
    DiscoverRecord::new(obs, uuid.to_string(), None, activation, vec![])
}

/// A recorded window in which the ABSOLUTE×(−1) branch (≤ 0) never wins the
/// MAXIMUM and the RELU branch (≥ 0) never wins the MINIMUM — the empirical
/// backing the evaluate-before-accept gate requires.
fn dominance_records(aggregate_uuid: &str, kind: AggregateKind) -> Vec<DiscoverRecord> {
    let mut recs = Vec::new();
    for obs in 0..16u32 {
        // ABSOLUTE branch source activation (≥ 0), contribution = weight(−1)×a ≤ 0.
        let abs_act = ((obs as f32) - 7.5).abs();
        // RELU branch source activation (≥ 0), contribution = weight(+1)×a ≥ 0.
        let relu_act = (obs as f32 - 5.0).max(0.0);
        recs.push(record(obs, ABS_UUID, abs_act));
        recs.push(record(obs, RELU_UUID, relu_act));
        let abs_contrib = -abs_act; // weight −1
        let relu_contrib = relu_act; // weight +1
        let agg = match kind {
            AggregateKind::Maximum => abs_contrib.max(relu_contrib),
            AggregateKind::Minimum => abs_contrib.min(relu_contrib),
        };
        recs.push(record(obs, aggregate_uuid, agg));
    }
    recs
}

#[test]
fn maximum_fixture_flags_dominated_absolute_branch() {
    let creature = load_network("maximum_aggregate.json");
    let dominated = detect_dominated_branches(&creature);
    assert_eq!(
        dominated.len(),
        1,
        "exactly the ABSOLUTE branch must be flagged in the MAXIMUM fixture"
    );
    assert_eq!(dominated[0].branch_uuid, ABS_UUID);
    assert_eq!(dominated[0].dominator_uuid, RELU_UUID);
    assert_eq!(dominated[0].aggregate_kind, AggregateKind::Maximum);

    let uuids = analytically_dominated_branch_uuids(&creature);
    assert!(uuids.contains(ABS_UUID));
    assert!(
        !uuids.contains(RELU_UUID),
        "the surviving RELU branch must never be flagged"
    );
}

#[test]
fn maximum_fixture_collapses_to_passthrough() {
    let mut creature = load_network("maximum_aggregate.json");
    let records = dominance_records("neuron-max", AggregateKind::Maximum);

    let outcome = collapse_dominated_branch(
        &mut creature,
        &records,
        "neuron-max",
        COLLAPSE_GATE_TOLERANCE,
    )
    .expect("dominated branch present in the MAXIMUM fixture");
    assert!(outcome.accepted, "{:?}", outcome.rejection_reason);
    assert!(outcome.folded_to_passthrough);
    assert!(outcome.max_residual <= COLLAPSE_GATE_TOLERANCE);

    // Full-collapse target: input-1 → neuron-relu → output-0 (2 neurons, 2 synapses).
    assert_eq!(
        creature.neurons.len(),
        2,
        "neuron-abs and neuron-max removed"
    );
    assert_eq!(creature.synapses.len(), 2);
    assert!(!creature.neurons.iter().any(|n| n.uuid == ABS_UUID));
    assert!(!creature.neurons.iter().any(|n| n.uuid == "neuron-max"));
    assert!(creature.neurons.iter().any(|n| n.uuid == RELU_UUID));
    let edge = creature
        .synapses
        .iter()
        .find(|s| s.from_uuid == RELU_UUID && s.to_uuid == "output-0")
        .expect("survivor rewired straight to output-0");
    assert!(
        (edge.weight - 1.0).abs() < f32::EPSILON,
        "folded weight = w_in(1) × w_out(1) = 1"
    );
}

#[test]
fn minimum_fixture_flags_and_collapses_dominated_relu_branch() {
    let mut creature = load_network("minimum_aggregate.json");
    let dominated = detect_dominated_branches(&creature);
    assert_eq!(dominated.len(), 1);
    assert_eq!(dominated[0].branch_uuid, RELU_UUID);
    assert_eq!(dominated[0].dominator_uuid, ABS_UUID);
    assert_eq!(dominated[0].aggregate_kind, AggregateKind::Minimum);

    let records = dominance_records("neuron-min", AggregateKind::Minimum);
    let outcome = collapse_dominated_branch(
        &mut creature,
        &records,
        "neuron-min",
        COLLAPSE_GATE_TOLERANCE,
    )
    .expect("dominated branch present in the MINIMUM fixture");
    assert!(outcome.accepted, "{:?}", outcome.rejection_reason);
    assert!(outcome.folded_to_passthrough);

    // Mirror target: input-0 → neuron-abs → output-0.
    assert_eq!(creature.neurons.len(), 2);
    assert_eq!(creature.synapses.len(), 2);
    assert!(!creature.neurons.iter().any(|n| n.uuid == RELU_UUID));
    assert!(!creature.neurons.iter().any(|n| n.uuid == "neuron-min"));
    assert!(creature.neurons.iter().any(|n| n.uuid == ABS_UUID));
    assert!(
        creature
            .synapses
            .iter()
            .any(|s| s.from_uuid == ABS_UUID && s.to_uuid == "output-0"),
        "surviving ABSOLUTE branch rewired straight to output-0"
    );
}

#[test]
fn if_fixture_is_not_collapsed_out_of_scope() {
    // IF dominance is conditional, not a magnitude property — explicitly out of
    // scope for this issue. The analytical detector must flag nothing.
    let creature = load_network("if_aggregate.json");
    assert!(
        detect_dominated_branches(&creature).is_empty(),
        "IF aggregates are out of scope and must not be flagged for analytical collapse"
    );
}

#[test]
fn empty_records_reject_the_collapse_no_blind_delete() {
    // The evaluate-before-accept gate must refuse to delete without empirical
    // backing (Issue #3234 — never fail silently / never delete blind).
    let mut creature = load_network("maximum_aggregate.json");
    let before = creature.clone();
    let outcome =
        collapse_dominated_branch(&mut creature, &[], "neuron-max", COLLAPSE_GATE_TOLERANCE)
            .expect("dominated branch flagged");
    assert!(!outcome.accepted);
    assert!(outcome.rejection_reason.is_some());
    assert_eq!(creature.neurons.len(), before.neurons.len());
    assert_eq!(creature.synapses.len(), before.synapses.len());
}
