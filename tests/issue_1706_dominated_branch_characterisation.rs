//! Characterisation tests: dominated-branch collapse through MAX / MIN / IF
//! (Issue #1706, parent #1704).
//!
//! These are **characterisation** tests, not an implementation. They pin the
//! *current* engine behaviour for a dominated branch feeding a selection
//! aggregate and, in each `current vs target` assertion, name the full-collapse
//! **target** the engine does not yet reach. Two facts are held together:
//!
//! 1. **Dominance is real** — for the committed fixtures under
//!    `tests/fixtures/dominated_branch_collapse/` the dominated branch is
//!    provably losing, on both bases the issue asks us to characterise:
//!    - *Analytical*: activation range × weight sign. `ABSOLUTE(x) ≥ 0` scaled
//!      by `−1` is always `≤ 0`; `RELU(x) ≥ 0` scaled by `+1` is always `≥ 0`.
//!      A MAXIMUM can therefore never select the ABSOLUTE branch, a MINIMUM can
//!      never select the RELU branch, and an IF never routes to a branch its
//!      condition does not choose.
//!    - *Empirical*: over a sampled observation window the dominated branch
//!      never wins the aggregate's selection.
//!
//! 2. **The engine does not collapse it** — the nearest transform in the engine
//!    today is the constant-neuron bias-fold removal of #1620/#1623, whose
//!    detector seam [`functionally_constant_neuron_uuids`] flags nothing *for
//!    these fixtures*: since Issue #1813 that seam is wired, but it only flags a
//!    neuron whose output cannot vary at all, and every branch here is driven by
//!    a live input. There is **no analytical dominance proof** in the engine, so
//!    a dominated (but variance-carrying) branch is never flagged and never
//!    removed.
//!    The **target** end state — dominated branch removed *and* the surviving
//!    single-branch aggregate folded to a pass-through (`InputB → RELU →
//!    output`) — is not reached. Each such test asserts today's non-collapse and
//!    labels the divergence.
//!
//! The worked example from the issue (`InputA → ABSOLUTE × (−1)` vs
//! `InputB → RELU` into MAX) has a dedicated test,
//! [`worked_example_max_current_non_collapse`], which is the earliest detection
//! point: any engine change that starts collapsing this network — or regresses
//! its current shape — trips that single case in CI.
//!
//! Fixture drift is caught at load: the fixture loaders panic on a missing or
//! malformed file, so renaming or deleting a committed fixture fails these tests
//! with a clear error rather than passing silently.
//!
//! Partially-dominated ("not so clean") cases are **not handled here**; they are
//! recorded as findings for the extent-report sub-issue (#1708). See
//! `docs/DOMINATED_BRANCH_COLLAPSE_EXTENT.md`.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::activations::{apply_scalar_squash, is_aggregate_squash};
use neat_ai_discovery::analysis::functionally_constant_neuron_uuids;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Fixture loading (offline, never fetched at runtime — Issue #1705).
// ---------------------------------------------------------------------------

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dominated_branch_collapse")
}

/// Load a committed aggregate network fixture into `CreatureJson`. Panics with a
/// clear message if the fixture is missing or malformed — the drift guard named
/// in the issue's failure-detection section.
fn load_network(file: &str) -> CreatureJson {
    let path = fixture_root().join("networks").join(file);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing/unreadable dominated-branch fixture {}: {e}",
            path.display()
        )
    });
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("malformed fixture {}: {e}", path.display()))
}

/// The single outgoing weight from `from_uuid` into the aggregate. Panics if the
/// fixture does not carry exactly one such synapse — the branch shape the
/// characterisation depends on.
fn outgoing_weight(creature: &CreatureJson, from_uuid: &str) -> f32 {
    let outs: Vec<f32> = creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid == from_uuid)
        .map(|s| s.weight)
        .collect();
    assert_eq!(
        outs.len(),
        1,
        "{from_uuid}: expected exactly one outgoing synapse into the aggregate, found {}",
        outs.len()
    );
    outs[0]
}

fn squash_of(creature: &CreatureJson, uuid: &str) -> String {
    creature
        .neurons
        .iter()
        .find(|n| n.uuid == uuid)
        .unwrap_or_else(|| panic!("neuron {uuid} missing from fixture"))
        .squash
        .clone()
}

/// A wide, deterministic pre-activation sample grid standing in for an
/// observation window. Spans negatives, zero, and positives so a one-signed
/// squash range is fully exercised.
fn input_window() -> Vec<f32> {
    let mut xs = Vec::new();
    let mut x = -50.0_f32;
    while x <= 50.0 {
        xs.push(x);
        x += 0.25;
    }
    xs
}

/// Contribution a scalar-squash branch feeds into the aggregate for one input:
/// `weight × squash(x)`.
fn branch_contribution(squash: &str, weight: f32, x: f32) -> f32 {
    let activation =
        apply_scalar_squash(squash, x).unwrap_or_else(|| panic!("{squash} is not a scalar squash"));
    weight * activation
}

/// Inclusive `[min, max]` range of a branch's contribution across the window.
fn branch_range(squash: &str, weight: f32, window: &[f32]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &x in window {
        let c = branch_contribution(squash, weight, x);
        lo = lo.min(c);
        hi = hi.max(c);
    }
    (lo, hi)
}

// ---------------------------------------------------------------------------
// MAXIMUM — the worked example. ABSOLUTE × (−1) (≤ 0) is dominated; RELU (≥ 0)
// wins every time.
// ---------------------------------------------------------------------------

const ABS_UUID: &str = "neuron-abs";
const RELU_UUID: &str = "neuron-relu";

#[test]
fn max_dominated_branch_analytically_provable() {
    let creature = load_network("maximum_aggregate.json");
    assert!(
        is_aggregate_squash(&squash_of(&creature, "neuron-max")),
        "neuron-max must be a selection aggregate"
    );

    let window = input_window();
    let (abs_lo, abs_hi) = branch_range(
        &squash_of(&creature, ABS_UUID),
        outgoing_weight(&creature, ABS_UUID),
        &window,
    );
    let (relu_lo, relu_hi) = branch_range(
        &squash_of(&creature, RELU_UUID),
        outgoing_weight(&creature, RELU_UUID),
        &window,
    );

    // Analytical dominance: the ABSOLUTE×(−1) branch is one-signed ≤ 0, the RELU
    // branch one-signed ≥ 0. Their ranges only meet at 0, so the ABSOLUTE branch
    // can never strictly exceed the RELU branch — it can never win the MAXIMUM.
    assert!(
        abs_hi <= 0.0,
        "ABSOLUTE×(−1) branch max {abs_hi} must be ≤ 0"
    );
    assert!(relu_lo >= 0.0, "RELU branch min {relu_lo} must be ≥ 0");
    assert!(
        abs_hi <= relu_lo,
        "ABSOLUTE branch (≤ {abs_hi}) must never exceed RELU branch (≥ {relu_lo})"
    );
    // Both branches are genuinely non-trivial (real spread), not degenerate.
    assert!(
        abs_lo < abs_hi && relu_lo < relu_hi,
        "branches must vary across the window"
    );
}

#[test]
fn max_dominated_branch_empirically_never_wins() {
    let creature = load_network("maximum_aggregate.json");
    let abs_sq = squash_of(&creature, ABS_UUID);
    let abs_w = outgoing_weight(&creature, ABS_UUID);
    let relu_sq = squash_of(&creature, RELU_UUID);
    let relu_w = outgoing_weight(&creature, RELU_UUID);

    let mut abs_wins = 0_usize;
    let window = input_window();
    for &x in &window {
        for &y in &window {
            let abs_c = branch_contribution(&abs_sq, abs_w, x);
            let relu_c = branch_contribution(&relu_sq, relu_w, y);
            // MAXIMUM selects the strictly larger branch.
            if abs_c > relu_c {
                abs_wins += 1;
            }
        }
    }
    assert_eq!(
        abs_wins, 0,
        "empirical dominance: the ABSOLUTE×(−1) branch won the MAXIMUM {abs_wins} times — it must never win"
    );
}

#[test]
fn worked_example_max_current_non_collapse() {
    // The issue's worked example, and the earliest detection point in CI.
    let creature = load_network("maximum_aggregate.json");
    let neurons_before = creature.neurons.len();
    let synapses_before = creature.synapses.len();

    // CURRENT engine behaviour: the nearest transform (constant-neuron bias-fold
    // removal, #1620/#1623) flags nothing — its structural detector (#1813) only
    // flags neurons that cannot vary, and every branch here is input-driven.
    // There is no analytical dominance proof in the engine, so the dominated
    // ABSOLUTE branch is not detected.
    let flagged = functionally_constant_neuron_uuids(&creature);
    assert!(
        flagged.is_empty(),
        "current vs target: the engine flags {} neuron(s) for collapse; today it flags none \
         (no analytical dominance proof exists — TARGET: flag + remove the dominated ABSOLUTE branch)",
        flagged.len()
    );
    assert!(
        !flagged.contains(ABS_UUID),
        "current vs target: dominated ABSOLUTE branch is not flagged today (TARGET: it is flagged and removed)"
    );

    // CURRENT: the network is unchanged — every neuron and synapse survives.
    // TARGET: full collapse to `input-1 → neuron-relu → output-0`, i.e.
    // neuron-abs and neuron-max removed and their weights/biases folded, leaving
    // 2 neurons and 2 synapses.
    assert_eq!(
        creature.neurons.len(),
        neurons_before,
        "current vs target: no neurons removed today (TARGET: drop neuron-abs and neuron-max)"
    );
    assert_eq!(
        creature.synapses.len(),
        synapses_before,
        "current vs target: no synapses folded today (TARGET: fold to input-1 → neuron-relu → output-0)"
    );
    assert!(
        creature.neurons.iter().any(|n| n.uuid == "neuron-max"),
        "current: the MAXIMUM aggregate still exists (TARGET: folded to pass-through)"
    );
}

// ---------------------------------------------------------------------------
// MINIMUM — mirror of MAX with the sign flipped. RELU (≥ 0) is the dominated
// branch; ABSOLUTE × (−1) (≤ 0) wins every time.
// ---------------------------------------------------------------------------

#[test]
fn min_dominated_branch_analytically_provable() {
    let creature = load_network("minimum_aggregate.json");
    assert!(
        is_aggregate_squash(&squash_of(&creature, "neuron-min")),
        "neuron-min must be a selection aggregate"
    );

    let window = input_window();
    let (abs_lo, abs_hi) = branch_range(
        &squash_of(&creature, ABS_UUID),
        outgoing_weight(&creature, ABS_UUID),
        &window,
    );
    let (relu_lo, relu_hi) = branch_range(
        &squash_of(&creature, RELU_UUID),
        outgoing_weight(&creature, RELU_UUID),
        &window,
    );

    // For a MINIMUM the RELU branch (≥ 0) is dominated: it can never be strictly
    // smaller than the ABSOLUTE×(−1) branch (≤ 0).
    assert!(relu_lo >= 0.0, "RELU branch min {relu_lo} must be ≥ 0");
    assert!(
        abs_hi <= 0.0,
        "ABSOLUTE×(−1) branch max {abs_hi} must be ≤ 0"
    );
    assert!(
        abs_hi <= relu_lo,
        "ABSOLUTE branch (≤ {abs_hi}) must never exceed RELU branch (≥ {relu_lo}) — RELU never wins the MINIMUM"
    );
    assert!(
        abs_lo < abs_hi && relu_lo < relu_hi,
        "branches must vary across the window"
    );
}

#[test]
fn min_dominated_branch_empirically_never_wins() {
    let creature = load_network("minimum_aggregate.json");
    let abs_sq = squash_of(&creature, ABS_UUID);
    let abs_w = outgoing_weight(&creature, ABS_UUID);
    let relu_sq = squash_of(&creature, RELU_UUID);
    let relu_w = outgoing_weight(&creature, RELU_UUID);

    let mut relu_wins = 0_usize;
    let window = input_window();
    for &x in &window {
        for &y in &window {
            let abs_c = branch_contribution(&abs_sq, abs_w, x);
            let relu_c = branch_contribution(&relu_sq, relu_w, y);
            // MINIMUM selects the strictly smaller branch.
            if relu_c < abs_c {
                relu_wins += 1;
            }
        }
    }
    assert_eq!(
        relu_wins, 0,
        "empirical dominance: the RELU branch won the MINIMUM {relu_wins} times — it must never win"
    );
}

#[test]
fn min_current_non_collapse() {
    let creature = load_network("minimum_aggregate.json");
    let flagged = functionally_constant_neuron_uuids(&creature);
    assert!(
        flagged.is_empty(),
        "current vs target: MINIMUM fixture flags nothing today (TARGET: remove the dominated RELU branch and fold to input-0 → neuron-abs → output-0)"
    );
    assert!(
        creature.neurons.iter().any(|n| n.uuid == "neuron-min"),
        "current: the MINIMUM aggregate still exists (TARGET: folded to pass-through)"
    );
    assert!(
        creature.neurons.iter().any(|n| n.uuid == RELU_UUID),
        "current: the dominated RELU branch still exists (TARGET: removed)"
    );
}

// ---------------------------------------------------------------------------
// IF — condition-synapse driven. Dominance here is *conditional*: the negative
// (ABSOLUTE × (−1)) branch is dominated on the window where the condition
// selects positive. This is a materially weaker dominance than MAX/MIN and is
// flagged as such for the extent report (#1708).
// ---------------------------------------------------------------------------

const COND_UUID: &str = "neuron-cond";

/// The IF fixture must carry the explicit condition synapse plus positive and
/// negative branches — the selection contract the characterisation exercises.
#[test]
fn if_condition_synapse_present_and_typed() {
    let creature = load_network("if_aggregate.json");
    assert!(
        is_aggregate_squash(&squash_of(&creature, "neuron-if")),
        "neuron-if must be a selection aggregate"
    );

    let typed = |t: &str| {
        creature
            .synapses
            .iter()
            .find(|s| s.synapse_type.as_deref() == Some(t))
            .unwrap_or_else(|| panic!("IF fixture missing a `{t}` synapse"))
    };
    // The condition synapse is driven by the TANH condition neuron.
    let cond = typed("condition");
    assert_eq!(
        cond.from_uuid, COND_UUID,
        "condition synapse must come from neuron-cond"
    );
    assert_eq!(
        squash_of(&creature, COND_UUID),
        "TANH",
        "condition neuron is TANH"
    );
    // Positive = RELU branch, negative = ABSOLUTE branch.
    assert_eq!(
        typed("positive").from_uuid,
        RELU_UUID,
        "positive branch is the RELU branch"
    );
    assert_eq!(
        typed("negative").from_uuid,
        ABS_UUID,
        "negative branch is the ABSOLUTE branch"
    );
}

#[test]
fn if_negative_branch_empirically_dominated_when_condition_positive() {
    // Engine IF semantics (src/focus/impact.rs): the positive branch is used
    // when the summed condition contribution > 0, the negative branch when ≤ 0.
    // Here condition = tanh(input-2) > 0 ⟺ input-2 > 0. On a window whose
    // condition input is strictly positive, the negative (ABSOLUTE) branch is
    // never selected — it is empirically dominated.
    let creature = load_network("if_aggregate.json");
    let cond_w = outgoing_weight_typed(&creature, "condition");
    let cond_sq = squash_of(&creature, COND_UUID);

    let mut negative_selected = 0_usize;
    // Condition input strictly positive across the whole window.
    for i in 1..=200 {
        let cond_input = i as f32 * 0.25;
        let condition_sum = cond_w * apply_scalar_squash(&cond_sq, cond_input).unwrap();
        // Positive branch iff condition_sum > 0, else negative.
        if condition_sum <= 0.0 {
            negative_selected += 1;
        }
    }
    assert_eq!(
        negative_selected, 0,
        "empirical dominance (IF): the negative ABSOLUTE branch was selected {negative_selected} times on a condition>0 window — it must never be selected there"
    );
}

#[test]
fn if_dominance_is_conditional_not_global() {
    // Finding for the extent report (#1708): unlike MAX/MIN, IF dominance is not
    // a magnitude property — it is driven entirely by the condition. Flip the
    // condition sign and the "dominated" negative branch becomes the *only*
    // selected branch, so it is NOT globally dominated. This is a
    // partially-dominated case, characterised here and left for #1708.
    let creature = load_network("if_aggregate.json");
    let cond_w = outgoing_weight_typed(&creature, "condition");
    let cond_sq = squash_of(&creature, COND_UUID);

    let mut negative_selected = 0_usize;
    // Condition input strictly negative across the whole window.
    for i in 1..=200 {
        let cond_input = -(i as f32) * 0.25;
        let condition_sum = cond_w * apply_scalar_squash(&cond_sq, cond_input).unwrap();
        if condition_sum <= 0.0 {
            negative_selected += 1;
        }
    }
    assert_eq!(
        negative_selected, 200,
        "IF dominance is condition-driven: on a condition<0 window the negative branch is always selected, proving it is not globally dominated (extent-report finding for #1708)"
    );
}

#[test]
fn if_current_non_collapse() {
    let creature = load_network("if_aggregate.json");
    let flagged = functionally_constant_neuron_uuids(&creature);
    assert!(
        flagged.is_empty(),
        "current vs target: IF fixture flags nothing today (TARGET on a condition-always-positive window: remove the dominated negative branch and its condition machinery)"
    );
    assert!(
        creature.neurons.iter().any(|n| n.uuid == "neuron-if"),
        "current: the IF aggregate still exists (TARGET: folded when its condition is degenerate)"
    );
    assert!(
        creature.neurons.iter().any(|n| n.uuid == COND_UUID),
        "current: the condition neuron still exists"
    );
}

/// Outgoing weight of the single synapse carrying the given `synapse_type`.
fn outgoing_weight_typed(creature: &CreatureJson, synapse_type: &str) -> f32 {
    let matches: Vec<f32> = creature
        .synapses
        .iter()
        .filter(|s| s.synapse_type.as_deref() == Some(synapse_type))
        .map(|s| s.weight)
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one `{synapse_type}` synapse, found {}",
        matches.len()
    );
    matches[0]
}

// ---------------------------------------------------------------------------
// Cross-aggregate: none of the three fixtures collapses today, and every
// fixture loads offline (fixture-drift guard).
// ---------------------------------------------------------------------------

#[test]
fn no_aggregate_fixture_collapses_today() {
    // One consolidated `current vs target` pin across all three aggregates: the
    // engine has no analytical-dominance transform, so every dominated branch
    // survives. TARGET for all three: full collapse to the surviving branch.
    for file in [
        "maximum_aggregate.json",
        "minimum_aggregate.json",
        "if_aggregate.json",
    ] {
        let creature = load_network(file);
        assert!(
            functionally_constant_neuron_uuids(&creature).is_empty(),
            "current vs target: {file} must show no collapse today (TARGET: dominated branch removed and aggregate folded)"
        );
        // Both dominated-branch candidates (ABSOLUTE and RELU) still present.
        assert!(
            creature.neurons.iter().any(|n| n.uuid == ABS_UUID)
                && creature.neurons.iter().any(|n| n.uuid == RELU_UUID),
            "{file}: both branch neurons must survive under current behaviour"
        );
    }
}
