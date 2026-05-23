//! Tests for Issue #1249: SSE-based "expected improvement" calculations
//! collapse under `CATEGORICAL_ERROR` (quantised `{0, 1}` flags).
//!
//! The fix gates the affected SSE-improvement code paths off at runtime
//! when the target's recorded errors are in the quantised `{0, 1}`
//! regime, using `analysis::quantised_error::is_quantised_zero_one` —
//! the same helper that Issue #1247 introduced for the distribution-
//! sensitive detectors.
//!
//! Two affected entry points are covered here directly via their public
//! API:
//!
//! - `analysis::recommendation::fan_in::detect_fan_in_candidates` — the
//!   private `compute_least_squares_improvement` helper returns `0.0`
//!   for quantised target errors, which causes the
//!   `best_individual <= 0.0` guard to drop every fan-in pair targeting
//!   that neuron.
//! - `analysis::recommendation::batch_successful::detect_individually_successful`
//!   — the `evaluate_individual` helper returns `None` for quantised
//!   target errors, so no individually-successful candidate is emitted.
//!
//! See `docs/COST_FUNCTION_NOTES.md` §4.7 for the per-cost catalogue.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use neat_ai_discovery::analysis::recommendation::batch_successful::detect_individually_successful;
use neat_ai_discovery::analysis::recommendation::fan_in::detect_fan_in_candidates;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
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

fn record(uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
    }
}

// ---------------------------------------------------------------------------
// fan_in.rs — least-squares improvement must collapse to 0 for the quantised
// regime. The downstream `best_individual <= 0.0` guard then drops every pair
// targeting that neuron.
// ---------------------------------------------------------------------------

/// Build a creature with two correlated inputs and a single output. The
/// inputs each correlate strongly with a continuous target error, so the
/// fan-in detector should normally produce a candidate.
fn make_two_input_fan_in_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("input-b", "input", "IDENTITY"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-a", "output-0", 0.05),
            synapse("input-b", "output-0", 0.05),
        ],
        input: 2,
        output: 1,
    }
}

/// Build the neuron record set used by both the "continuous" and
/// "quantised" sanity checks. `make_err` decides the recorded error
/// shape on the output target, and `make_act` decides the input
/// activations. Inputs must correlate with the target error for a
/// fan-in candidate to survive the `INPUT_ERROR_CORRELATION_THRESHOLD`
/// gate.
fn build_fan_in_records<E, A>(make_err: E, make_act: A) -> Vec<(String, Vec<DiscoverRecord>)>
where
    E: Fn(usize) -> f32,
    A: Fn(&str, usize) -> f32,
{
    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for uuid in ["input-a", "input-b", "output-0"] {
        let recs: Vec<DiscoverRecord> = (0..60)
            .map(|i| {
                let act = make_act(uuid, i);
                let errors = if uuid == "output-0" {
                    vec![make_err(i)]
                } else {
                    vec![]
                };
                record(uuid, i as u32, act, errors)
            })
            .collect();
        records.push((uuid.to_string(), recs));
    }
    records
}

/// Sanity check: with continuous (non-quantised) target errors that are
/// strongly correlated with the inputs, the fan-in detector should emit
/// at least one candidate. This anchors the "quantised" tests below —
/// without this baseline, an absence of candidates would not prove the
/// gate is doing any work.
#[test]
fn fan_in_emits_candidates_for_continuous_errors() {
    let creature = make_two_input_fan_in_creature();

    let records = build_fan_in_records(
        |i| {
            let phase = (i as f32) * 0.21;
            // Continuous, correlated, signed residual.
            0.5 * phase.sin() + 0.3 * (phase * 1.7).cos()
        },
        |uuid, i| {
            let phase = (i as f32) * 0.21;
            match uuid {
                "input-a" => phase.sin(),
                "input-b" => (phase * 1.7).cos(),
                _ => 0.1 * phase.sin(),
            }
        },
    );

    let candidates = detect_fan_in_candidates(&creature, &records);
    assert!(
        !candidates.is_empty(),
        "continuous fan-in baseline must emit at least one candidate (got 0); without this the gate test below is vacuous",
    );
}

/// Under the quantised `{0, 1}` regime the SSE-based "expected
/// improvement" is bounded by the misclassification count rather than a
/// loss reduction (Issue #1249). The fan-in detector must therefore
/// emit **zero** candidates against quantised target errors, even when
/// the inputs would otherwise look like a strong fan-in pair.
#[test]
fn fan_in_emits_no_candidates_for_quantised_errors() {
    let creature = make_two_input_fan_in_creature();

    // Inputs deliberately correlate strongly with the misclassification
    // flag while staying mutually decorrelated. Without the gate the
    // SSE-improvement formula reports a sizeable "improvement" because
    // `Σe² = Σe = error_count` and the residual nearly collapses
    // against the flag-tracking inputs. The independent sinusoidal
    // jitter keeps mutual correlation below the fan-in detector's
    // redundancy threshold.
    let records = build_fan_in_records(
        |i| if i % 2 == 0 { 0.0 } else { 1.0 },
        |uuid, i| {
            let flag = if i % 2 == 0 { 0.0 } else { 1.0 };
            match uuid {
                "input-a" => flag + 0.6 * (i as f32 * 0.7).sin(),
                "input-b" => flag + 0.6 * (i as f32 * 1.3).cos(),
                _ => 0.0,
            }
        },
    );

    let candidates = detect_fan_in_candidates(&creature, &records);
    assert!(
        candidates.is_empty(),
        "quantised {{0,1}} errors must gate off fan-in SSE improvement, got {} candidate(s)",
        candidates.len(),
    );
}

// ---------------------------------------------------------------------------
// batch_successful::detection — `improvement = 1 - residual_sse/original_sse`
// must not be returned when the target errors are quantised flags, since
// the ratio no longer corresponds to NEAT-AI's loss reduction.
// ---------------------------------------------------------------------------

/// Build a creature with one input and one output (no existing synapse).
/// The batch-successful detector would normally propose an `input → output`
/// synapse if the input's activation correlates with the output error.
fn make_individual_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        synapses: vec![],
        input: 1,
        output: 1,
    }
}

fn build_individual_records<E, A>(make_err: E, make_act: A) -> Vec<(String, Vec<DiscoverRecord>)>
where
    E: Fn(usize) -> f32,
    A: Fn(usize) -> f32,
{
    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for uuid in ["input-a", "output-0"] {
        let recs: Vec<DiscoverRecord> = (0..60)
            .map(|i| {
                let act = if uuid == "input-a" { make_act(i) } else { 0.0 };
                let errors = if uuid == "output-0" {
                    vec![make_err(i)]
                } else {
                    vec![]
                };
                record(uuid, i as u32, act, errors)
            })
            .collect();
        records.push((uuid.to_string(), recs));
    }
    records
}

/// Sanity check: continuous, correlated errors produce at least one
/// individually-successful candidate. Without this anchor the gating
/// test below would be vacuous.
#[test]
fn batch_successful_emits_candidates_for_continuous_errors() {
    let creature = make_individual_creature();

    let records = build_individual_records(
        |i| {
            let phase = (i as f32) * 0.21;
            // Strongly correlated continuous residual.
            0.8 * phase.sin()
        },
        |i| (i as f32 * 0.21).sin(),
    );

    let candidates = detect_individually_successful(&creature, &records);
    assert!(
        !candidates.is_empty(),
        "continuous batch-successful baseline must emit at least one candidate (got 0); without this the gate test is vacuous",
    );
}

/// Quantised `{0, 1}` target errors must gate the SSE-improvement
/// ratio off. The `improvement = 1 − residual_sse / original_sse`
/// number is invalid under the regime (Σe² = Σe = misclassification
/// count), so no individually-successful candidate may be emitted for
/// that target.
#[test]
fn batch_successful_emits_no_candidates_for_quantised_errors() {
    let creature = make_individual_creature();

    // Input activation strongly tracks the misclassification flag —
    // without the gate the SSE-improvement ratio would approach 1.0
    // (`residual_sse → 0`) and produce a "highly successful" candidate
    // that does not correspond to any real loss reduction.
    let records = build_individual_records(
        |i| if i % 2 == 0 { 0.0 } else { 1.0 },
        |i| {
            let flag = if i % 2 == 0 { 0.0 } else { 1.0 };
            flag + 0.1 * (i as f32 * 0.13).sin()
        },
    );

    let candidates = detect_individually_successful(&creature, &records);
    assert!(
        candidates.is_empty(),
        "quantised {{0,1}} errors must gate off batch-successful SSE ratio, got {} candidate(s)",
        candidates.len(),
    );
}
