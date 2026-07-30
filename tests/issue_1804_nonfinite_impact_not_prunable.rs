//! Issue #1804: the shipped removal path must never treat a non-finite
//! structural impact as maximally prunable.
//!
//! `identify_structural_removal_candidates` (the live FFI path behind
//! `rank_focus_neurons`) compares boosted complexity savings against a neuron's
//! structural contribution. Mapping a NaN/infinite impact to `0.0` made such a
//! neuron the *strongest* removal candidate — bad numbers licensed the most
//! destructive edit. The policy is now the defensive one: non-finite →
//! `f32::INFINITY` (never pruned), genuine zero → still prunable.
//!
//! These tests drive the real FFI entry point (`rank_focus_neurons_internal`)
//! end to end. A **NaN** weight cannot be expressed in JSON at all, and an
//! overflowing weight is rejected by the deserialiser (pinned below), so the
//! NaN-impact half of the policy is only reachable in-process — it is covered
//! by `structural_removal_tests::nan_impact_neuron_absent_from_candidates` in
//! `src/focus/ranking/removal_candidates.rs`. What this file guards on the
//! shipped surface is the pair that *is* reachable: the zero-impact
//! non-regression, and the "no candidate carries a non-finite impact"
//! invariant.

use neat_ai_discovery::rank_focus_neurons_internal;
use serde_json::Value;

/// Cost of growth large enough that the complexity savings clear the
/// `REMOVE_LOW_IMPACT_NOISE_FLOOR` (1e-5) for a zero-impact neuron.
const TEST_COST_OF_GROWTH: &str = "1e-4";

/// A creature pinning the two non-regression cases on the shipped FFI path:
///
/// * `h-zero` — weight `0.0` into `out-0` gives a genuine `0.0` impact, so it
///   must remain a removal candidate.
/// * `h-keep` — weight `1.0` into `out-0` gives impact `1.0`, far above the
///   savings, so it is never a candidate.
fn creature_input_json(keep_path_weight: &str) -> String {
    format!(
        r#"{{
            "parquetFile": "/nonexistent/issue-1804/discovery.parquet",
            "costOfGrowth": {TEST_COST_OF_GROWTH},
            "creature": {{
                "input": 1,
                "output": 1,
                "neurons": [
                    {{"uuid": "in-0",   "type": "input",  "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "h-zero", "type": "hidden", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "h-keep", "type": "hidden", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "out-0",  "type": "output", "squash": "IDENTITY", "bias": 0.0}}
                ],
                "synapses": [
                    {{"fromUUID": "in-0",   "toUUID": "h-zero", "weight": 0.5}},
                    {{"fromUUID": "in-0",   "toUUID": "h-keep", "weight": {keep_path_weight}}},
                    {{"fromUUID": "h-zero", "toUUID": "out-0",  "weight": 0.0}},
                    {{"fromUUID": "h-keep", "toUUID": "out-0",  "weight": 1.0}}
                ]
            }}
        }}"#
    )
}

/// Run the shipped focus/removal path and return the parsed output.
fn rank(input: &str) -> Value {
    let raw = rank_focus_neurons_internal(input).expect("rank_focus_neurons_internal must not Err");
    serde_json::from_str(&raw).expect("output must be valid JSON")
}

/// Run the shipped path, asserting it succeeded.
fn rank_ok(input: &str) -> Value {
    let parsed = rank(input);
    assert_eq!(
        parsed["success"],
        true,
        "ranking failed: {}",
        parsed["error"].as_str().unwrap_or("unknown error")
    );
    parsed
}

/// The removal candidates emitted for the fixture creature.
fn removal_candidates(parsed: &Value) -> Vec<Value> {
    parsed["removalCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

fn candidate_uuids(candidates: &[Value]) -> Vec<String> {
    candidates
        .iter()
        .map(|c| c["neuronUuid"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// Acceptance: the fix must not over-correct — a genuinely zero-contribution
/// neuron is still prunable, while a high-impact one is still safe.
#[test]
fn zero_impact_neuron_still_candidate() {
    let parsed = rank_ok(&creature_input_json("0.5"));
    let uuids = candidate_uuids(&removal_candidates(&parsed));

    assert!(
        uuids.iter().any(|u| u == "h-zero"),
        "a genuine 0.0-impact neuron must remain a removal candidate, got {uuids:?}"
    );
    assert!(
        !uuids.iter().any(|u| u == "h-keep"),
        "a high-impact neuron must never be a removal candidate, got {uuids:?}"
    );
}

/// Acceptance: the invariant asserted directly over the returned list — no
/// emitted candidate carries a non-finite impact.
#[test]
fn no_candidate_has_nonfinite_raw_impact() {
    let parsed = rank_ok(&creature_input_json("0.5"));

    for candidate in removal_candidates(&parsed) {
        let uuid = candidate["neuronUuid"].as_str().unwrap_or_default();
        // serde_json serialises non-finite floats as `null`, so a non-numeric
        // value here is itself proof of a non-finite impact escaping.
        let impact = candidate["impact"]
            .as_f64()
            .unwrap_or_else(|| panic!("candidate {uuid} has a non-numeric impact"));
        assert!(
            impact.is_finite(),
            "candidate {uuid} has a non-finite impact {impact}"
        );
    }
}

/// Defence in depth: an overflowing weight — the closest JSON can get to a
/// non-finite one — is rejected at the FFI boundary rather than deserialising
/// to infinity, so no non-finite impact can be injected through this surface.
/// The failure is loud: `success: false` with a parse error, not a silent
/// clamp.
#[test]
fn overflowing_weight_is_rejected_at_the_ffi_boundary() {
    let parsed = rank(&creature_input_json("1e400"));

    assert_eq!(
        parsed["success"], false,
        "an out-of-range weight must be rejected, not silently coerced"
    );
    assert!(
        parsed["error"].as_str().is_some_and(|e| !e.is_empty()),
        "a rejected input must carry a descriptive error"
    );
}
