//! Issue #1923: `remove-low-impact` must rank on a **live** signal.
//!
//! The structural removal path constructed every candidate with
//! `mean_activation: 0.0` / `activation_weighted_impact: 0.0` and a reason
//! string admitting the activation-weighted gate was "deferred to analysis".
//! Nothing downstream ever resolved that deferral, so the strategy that
//! produces 53% of every cached candidate and 79% of all realised gain ranked
//! on a dead field — `impact` correlated with realised gain at r = −0.036 and
//! `meanActivation` had zero variance across all 67 cached records (#1920,
//! finding A).
//!
//! These tests drive `rank_focus_neurons_internal` — the shipped FFI path — and
//! assert on the wire contract: the fields carry measurements when records
//! exist, an actively-firing neuron is gated out, unmeasured candidates rank
//! last, and an unreadable parquet fails **loud** rather than passing an
//! unmeasured `0.0` off as a measurement.

use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::rank_focus_neurons_internal;
use neat_ai_discovery::types::DiscoverRecord;
use serde_json::{Value, json};
use tempfile::NamedTempFile;

/// One high-impact hidden neuron driving the output plus a dead hidden chain
/// that never reaches it. The dead chain carries zero structural impact, so its
/// neurons are the low-contribution removal candidates.
fn creature_with_dead_branch() -> Value {
    json!({
        "neurons": [
            {"uuid": "i0", "type": "input", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "h-main", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "o0", "type": "output", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "d0", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "d1", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "d2", "type": "hidden", "squash": "IDENTITY", "bias": 0.0}
        ],
        "synapses": [
            {"from_uuid": "i0", "to_uuid": "h-main", "weight": 1.0},
            {"from_uuid": "h-main", "to_uuid": "o0", "weight": 1.0},
            {"from_uuid": "i0", "to_uuid": "d0", "weight": 0.5},
            {"from_uuid": "d0", "to_uuid": "d1", "weight": 0.5},
            {"from_uuid": "d1", "to_uuid": "d2", "weight": 0.5}
        ],
        "input": 1,
        "output": 1
    })
}

/// A creature whose single hidden neuron has a small but non-zero structural
/// impact, so the activation-weighted gate has something to weight.
fn creature_with_weak_hidden(weight: f64) -> Value {
    json!({
        "neurons": [
            {"uuid": "i0", "type": "input", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "h-weak", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "h-main", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "o0", "type": "output", "squash": "IDENTITY", "bias": 0.0}
        ],
        "synapses": [
            {"from_uuid": "i0", "to_uuid": "h-main", "weight": 1.0},
            {"from_uuid": "h-main", "to_uuid": "o0", "weight": 1.0},
            {"from_uuid": "i0", "to_uuid": "h-weak", "weight": 1.0},
            {"from_uuid": "h-weak", "to_uuid": "o0", "weight": weight}
        ],
        "input": 1,
        "output": 1
    })
}

fn record(obs: u32, uuid: &str, activation: f32) -> DiscoverRecord {
    DiscoverRecord::new(obs, uuid.to_string(), Some(0.5), activation, vec![0.1])
}

fn parquet_with(records: &[DiscoverRecord]) -> (NamedTempFile, String) {
    let file = NamedTempFile::new().expect("temp file");
    let path = file.path().to_str().expect("utf-8 path").to_string();
    write_records_to_parquet(&path, records).expect("write parquet");
    (file, path)
}

fn rank(creature: &Value, parquet_file: &str) -> Value {
    // costOfGrowth 1e-4 so the savings clear the default noise floor.
    let input = json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "maxResults": 32,
        "focusSetSize": 4,
        "focusSelectionCursor": 0,
        "costOfGrowth": 1e-4
    })
    .to_string();
    let result_json = rank_focus_neurons_internal(&input).expect("rank_focus_neurons_internal");
    serde_json::from_str(&result_json).expect("valid JSON")
}

fn removal_candidates(result: &Value) -> Vec<Value> {
    result["removalCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

fn candidate<'a>(candidates: &'a [Value], uuid: &str) -> Option<&'a Value> {
    candidates
        .iter()
        .find(|c| c["neuronUuid"].as_str() == Some(uuid))
}

/// The headline fix: with records available, `meanActivation` carries a real
/// measurement and `activationWeightedImpact` is `impact × meanActivation`.
#[test]
fn mean_activation_is_measured_from_recorded_samples() {
    let creature = creature_with_weak_hidden(1e-6);
    let (_file, path) = parquet_with(&[
        record(0, "h-weak", 0.01),
        record(1, "h-weak", -0.03),
        record(2, "h-weak", 0.02),
        record(0, "h-main", 1.0),
    ]);

    let result = rank(&creature, &path);
    let candidates = removal_candidates(&result);
    let weak = candidate(&candidates, "h-weak")
        .unwrap_or_else(|| panic!("h-weak must be a removal candidate: {result}"));

    let mean = weak["meanActivation"].as_f64().expect("meanActivation");
    assert!(
        (mean - 0.02).abs() < 1e-6,
        "mean |activation| of 0.01/-0.03/0.02 is 0.02, got {mean}"
    );

    let impact = weak["impact"].as_f64().expect("impact");
    let awi = weak["activationWeightedImpact"]
        .as_f64()
        .expect("activationWeightedImpact");
    assert!(
        (awi - impact * mean).abs() <= 1e-12_f64.max(impact * mean * 1e-5),
        "activationWeightedImpact must equal impact × meanActivation: \
         {awi} vs {impact} × {mean}"
    );
    assert!(
        awi > 0.0,
        "the ranking signal must be live, not the hard-coded 0.0 this issue fixes"
    );
    assert!(
        (weak["expectedErrorReduction"].as_f64().expect("expected") - awi).abs() < 1e-12,
        "expectedErrorReduction is the activation-weighted contribution given up"
    );
    let reason = weak["reason"].as_str().expect("reason");
    assert!(
        reason.contains("activation-weighted gate resolved"),
        "the reason must record that the gate ran: {reason}"
    );
    assert!(
        !reason.contains("pending"),
        "a resolved gate must not still claim to be pending: {reason}"
    );
}

/// Issue #892's active-neuron gate, previously unreachable on this path: a
/// neuron that is still firing hard is contributing and must not be proposed
/// for removal, however small its structural impact.
#[test]
fn an_actively_firing_neuron_is_gated_out_and_counted() {
    let creature = creature_with_weak_hidden(1e-6);
    // Mean |activation| 5.0 — far above REMOVAL_MEAN_ACTIVATION_THRESHOLD (0.04).
    let (_file, path) = parquet_with(&[
        record(0, "h-weak", 5.0),
        record(1, "h-weak", -5.0),
        record(0, "h-main", 1.0),
    ]);

    let result = rank(&creature, &path);
    let candidates = removal_candidates(&result);

    assert!(
        candidate(&candidates, "h-weak").is_none(),
        "an actively-firing neuron must not be offered for removal: {result}"
    );
    let breakdown = &result["rejectionBreakdown"];
    let active = breakdown
        .as_object()
        .and_then(|m| {
            m.iter()
                .find(|(k, _)| k.contains("active"))
                .and_then(|(_, v)| v.as_u64())
        })
        .unwrap_or_default();
    assert!(
        active >= 1,
        "the rejection must be counted, not silently dropped: {breakdown}"
    );
}

/// A neuron with no recorded activations stays a candidate but must rank
/// **below** every measured one — an unmeasured `activationWeightedImpact` of
/// `0.0` would otherwise flatter it to the top of the list.
#[test]
fn unmeasured_candidates_rank_below_measured_ones() {
    let creature = creature_with_dead_branch();
    // Only d1 is recorded; d0 and d2 have no rows at all.
    let (_file, path) = parquet_with(&[record(0, "d1", 0.001), record(1, "d1", 0.001)]);

    let result = rank(&creature, &path);
    let candidates = removal_candidates(&result);
    assert!(
        candidates.len() >= 2,
        "the dead branch must yield several candidates: {result}"
    );

    let first = candidates[0]["neuronUuid"].as_str().expect("uuid");
    assert_eq!(
        first, "d1",
        "the only measured candidate must rank first: {candidates:?}"
    );

    for c in candidates.iter().skip(1) {
        let reason = c["reason"].as_str().expect("reason");
        assert!(
            reason.contains("no recorded activation samples"),
            "an unmeasured candidate must say so on the wire: {reason}"
        );
        assert_eq!(
            c["meanActivation"].as_f64().expect("meanActivation"),
            0.0,
            "an unmeasured candidate keeps an unmeasured meanActivation"
        );
    }
}

/// Fail loud, not silent: when the discovery records cannot be read at all, the
/// candidates must say the gate did not run instead of shipping an unmeasured
/// `0.0` that reads as "measured and inactive".
#[test]
fn an_unreadable_parquet_marks_the_gate_unresolved() {
    let creature = creature_with_dead_branch();
    let result = rank(&creature, "/nonexistent/never-written.parquet");
    let candidates = removal_candidates(&result);

    assert!(
        !candidates.is_empty(),
        "structural triage must still produce candidates without records: {result}"
    );
    for c in &candidates {
        let reason = c["reason"].as_str().expect("reason");
        assert!(
            reason.contains("unresolved"),
            "an unresolved gate must be declared on the wire: {reason}"
        );
        assert_eq!(c["meanActivation"].as_f64().expect("meanActivation"), 0.0);
        assert_eq!(
            c["activationWeightedImpact"]
                .as_f64()
                .expect("activationWeightedImpact"),
            0.0
        );
    }
}
