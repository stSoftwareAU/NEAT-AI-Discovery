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
//!
//! ## Issue #1872 — the record-derived twin
//!
//! `rank_focus_neurons` (still shipped public API) reaches the *record-derived*
//! `identify_removal_candidates`, which missed both guards. It takes a
//! `CreatureJson` **struct**, not JSON, so a NaN synapse weight — inexpressible
//! over the FFI boundary — is reachable here, and `cost_of_growth` is an
//! `Option<f32>` the host can set to a non-finite or negative value. The
//! `record_derived_path` module below pins both on that surface.

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

// =============================================================================
// Issue #1872 — the record-derived removal path (`rank_focus_neurons`)
// =============================================================================

mod record_derived_path {
    //! Issue #1872: `identify_removal_candidates` — the record-derived twin of
    //! the structural triage above — accepted non-finite host-shaped values that
    //! #1804 and #1783 hardened the structural path against.
    //!
    //! Every gate on that path is NaN-false (`boosted_savings <= impact`, the
    //! active-neuron gate behind `impact > EPSILON`, and
    //! `net_improvement < noise_floor`), and the descending `total_cmp` sort
    //! orders a positive NaN *above* `+inf`. So a NaN contribution did not merely
    //! survive triage — it became the **top-ranked** removal candidate, and a
    //! non-finite `costOfGrowth` made *every* ranked neuron one.

    use neat_ai_discovery::focus::rank_focus_neurons;
    use neat_ai_discovery::parquet_format::write_records_to_parquet;
    use neat_ai_discovery::types::DiscoverRecord;
    use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
    use tempfile::NamedTempFile;

    /// Large enough that a zero-contribution neuron's boosted savings clear the
    /// `costOfGrowth`-denominated noise floor (Issue #1814).
    const TEST_COST_OF_GROWTH: f32 = 1e-4;

    fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: neuron_type.to_string(),
            squash: "IDENTITY".to_string(),
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

    /// Two observations per neuron, at the given mean absolute activation.
    fn records(neuron_activations: &[(&str, f32)]) -> Vec<DiscoverRecord> {
        neuron_activations
            .iter()
            .flat_map(|(uuid, activation)| {
                (0..2).map(move |obs| {
                    DiscoverRecord::new(obs, (*uuid).to_string(), Some(0.5), *activation, vec![0.1])
                })
            })
            .collect()
    }

    /// Rank the creature against a temporary parquet holding `activations`, and
    /// return the removal-candidate UUIDs in emitted (rank) order.
    fn removal_candidate_uuids(
        creature: &CreatureJson,
        activations: &[(&str, f32)],
        cost_of_growth: Option<f32>,
    ) -> Vec<String> {
        let temp_file = NamedTempFile::new().expect("temp parquet");
        let path = temp_file.path().to_str().expect("utf-8 temp path");
        write_records_to_parquet(path, &records(activations)).expect("write parquet");

        rank_focus_neurons(path, creature, None, cost_of_growth)
            .expect("ranking must succeed")
            .removal_candidates
            .into_iter()
            .map(|c| c.neuron_uuid)
            .collect()
    }

    /// A creature carrying a NaN synapse weight — the propagation path the issue
    /// describes, where `total_inbound_weight += synapse.weight.abs()` carries
    /// NaN into `activation_weighted_impact`.
    ///
    /// `h-nan` is isolated on its own output so the NaN cannot contaminate the
    /// other two neurons' impacts:
    ///
    /// * `h-nan`  — NaN weight into `out-0` ⇒ non-finite contribution.
    /// * `h-zero` — `0.0` weight into `out-1` ⇒ a genuine `0.0` contribution.
    /// * `h-keep` — `1.0` weight into `out-1` ⇒ contribution `1.0`.
    fn creature_with_nan_weight() -> CreatureJson {
        CreatureJson {
            neurons: vec![
                neuron("h-nan", "hidden"),
                neuron("h-zero", "hidden"),
                neuron("h-keep", "hidden"),
                neuron("out-0", "output"),
                neuron("out-1", "output"),
            ],
            synapses: vec![
                synapse("in-0", "h-nan", 0.5),
                synapse("in-0", "h-zero", 0.5),
                synapse("in-0", "h-keep", 0.5),
                synapse("h-nan", "out-0", f32::NAN),
                synapse("h-zero", "out-1", 0.0),
                synapse("h-keep", "out-1", 1.0),
            ],
            input: 1,
            output: 2,
        }
    }

    /// Activations low enough that the `REMOVAL_MEAN_ACTIVATION_THRESHOLD`
    /// (0.04) gate never fires, so each verdict is decided by the guards under
    /// test rather than by the activity screen.
    const LOW_ACTIVATIONS: [(&str, f32); 5] = [
        ("h-nan", 0.01),
        ("h-zero", 0.01),
        ("h-keep", 0.01),
        ("out-0", 0.01),
        ("out-1", 0.01),
    ];

    /// Acceptance 1: a neuron whose contribution is non-finite is never offered
    /// for removal on the record-derived path either — bad numbers must not
    /// license a destructive edit.
    #[test]
    fn nan_contribution_is_never_a_removal_candidate() {
        let uuids = removal_candidate_uuids(
            &creature_with_nan_weight(),
            &LOW_ACTIVATIONS,
            Some(TEST_COST_OF_GROWTH),
        );

        assert!(
            !uuids.iter().any(|u| u == "h-nan"),
            "a NaN contribution must never license removal, got {uuids:?}"
        );
    }

    /// Acceptance 2: the guard must not over-correct — a genuinely
    /// zero-contribution neuron is still prunable, and a high-contribution one
    /// is still safe.
    #[test]
    fn zero_contribution_still_prunable_and_high_contribution_still_safe() {
        let uuids = removal_candidate_uuids(
            &creature_with_nan_weight(),
            &LOW_ACTIVATIONS,
            Some(TEST_COST_OF_GROWTH),
        );

        assert!(
            uuids.iter().any(|u| u == "h-zero"),
            "a genuine 0.0-contribution neuron must remain a removal candidate, got {uuids:?}"
        );
        assert!(
            !uuids.iter().any(|u| u == "h-keep"),
            "a high-contribution neuron must never be a removal candidate, got {uuids:?}"
        );
    }

    /// Acceptance 3: a non-finite or non-positive host `costOfGrowth` must not
    /// flood the result with removal candidates. `savings` would be NaN, and the
    /// NaN-false gates would then pass every ranked neuron through.
    ///
    /// `h-keep` is the witness: its contribution (`1.0` × mean activation
    /// `0.01`) dwarfs the savings at any sane `costOfGrowth`, and its mean
    /// activation sits below the 0.04 activity gate, so only the savings-vs-
    /// contribution comparison can reject it.
    #[test]
    fn nonsense_cost_of_growth_does_not_flood_removal_candidates() {
        let creature = creature_with_nan_weight();

        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1e-4] {
            let uuids = removal_candidate_uuids(&creature, &LOW_ACTIVATIONS, Some(invalid));
            assert!(
                !uuids.iter().any(|u| u == "h-keep"),
                "costOfGrowth {invalid} must fall back to the default, not make every \
                 neuron prunable; got {uuids:?}"
            );
            assert!(
                !uuids.iter().any(|u| u == "h-nan"),
                "costOfGrowth {invalid} must not resurrect the NaN-contribution neuron; \
                 got {uuids:?}"
            );
        }
    }
}
