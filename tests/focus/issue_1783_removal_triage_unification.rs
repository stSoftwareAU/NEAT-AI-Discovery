//! Issue #1805 (sub-issue of #1783): the structural removal criterion must exist
//! in exactly one place, so the two surviving entry points cannot drift apart.
//!
//! `focus::triage_removal_candidates` is now a thin adapter over the shipped
//! path (`identify_structural_removal_candidates`, reached here through the FFI
//! entry point `rank_focus_neurons_internal`). These tests assert the drift
//! detector the issue asks for: **identical candidate sets, identical ordering
//! and identical per-candidate values** for the same creature and cost of
//! growth, observed across the crate boundary.
//!
//! The non-finite-impact case — where the two copies previously disagreed
//! (`f32::INFINITY` vs `0.0`) — needs a NaN synapse weight, which serde refuses
//! to carry across the JSON FFI boundary, so it is pinned by the in-crate parity
//! tests in `src/focus/ranking/removal_triage.rs` instead.

// f32 values round-trip through JSON as f64; narrowing back to f32 is exact and
// is the only way to compare the two surfaces bit-for-bit.
#![allow(clippy::cast_possible_truncation)]

use neat_ai_discovery::focus::triage_removal_candidates;
use neat_ai_discovery::rank_focus_neurons_internal;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serde_json::{Value, json};
use serial_test::serial;

/// The noise-floor env var, unset in tests that depend on the default.
const NOISE_FLOOR_ENV: &str = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR";

/// Cost-of-growth large enough that the boosted savings clear the default 1e-5
/// noise floor for the low-impact neurons.
const TEST_COST_OF_GROWTH: f32 = 1e-4;

/// RAII guard that unsets an env var for a single test and restores it after.
struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn unset(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::remove_var(key) };
        Self { key, previous }
    }

    /// Pin an env var for a single test and restore it after (Issue #1814).
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            Some(v) => unsafe { std::env::set_var(self.key, v) },
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

fn neuron(uuid: &str, ntype: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: ntype.to_string(),
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

/// Hidden neurons with alternating impact levels and deliberately different
/// synapse counts, so the candidate set, the ordering and the noise-floor
/// rejections all carry real signal.
///
/// Laid out forward-only (all inputs, then the hidden neurons, then the output
/// last) so the FFI's `validate_forward_only_synapses` gate accepts it.
const HIDDEN_COUNT: usize = 6;

fn mixed_creature() -> CreatureJson {
    let mut inputs = vec![neuron("in-0", "input")];
    let mut hidden = vec![];
    let mut synapses = vec![];

    for i in 0..HIDDEN_COUNT {
        let uuid = format!("h-{i}");
        hidden.push(neuron(&uuid, "hidden"));
        synapses.push(synapse("in-0", &uuid, 0.5));
        // Even neurons dominate the output; odd ones are negligible.
        let weight = if i % 2 == 0 { 1.0 } else { 1e-8 };
        synapses.push(synapse(&uuid, "out", weight));
        // Vary the synapse count so savings — and therefore sort order — differ.
        for j in 0..i {
            let extra = format!("in-{i}-{j}");
            inputs.push(neuron(&extra, "input"));
            synapses.push(synapse(&extra, &uuid, 0.5));
        }
    }

    let input = inputs.len();
    let mut neurons = inputs;
    neurons.extend(hidden);
    neurons.push(neuron("out", "output"));
    CreatureJson {
        neurons,
        synapses,
        input,
        output: 1,
    }
}

/// Run the shipped FFI focus path and return its `removalCandidates` array.
///
/// A deliberately non-existent parquet path keeps the run structure-only
/// (Issue #1766) — the same conditions the adapter runs under.
fn ffi_removal_candidates(creature: &CreatureJson, cost_of_growth: f32) -> Vec<Value> {
    let input = json!({
        "parquetFile": "/nonexistent/issue-1783/discovery.parquet",
        "creature": creature,
        "maxResults": 64,
        "focusSetSize": 4,
        "focusSelectionCursor": 0,
        "costOfGrowth": cost_of_growth,
    })
    .to_string();
    let result: Value =
        serde_json::from_str(&rank_focus_neurons_internal(&input).expect("FFI focus path"))
            .expect("FFI response JSON");
    assert_eq!(
        result["success"], true,
        "the structure-only FFI focus path must succeed: {result:?}"
    );
    result["removalCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

/// Acceptance: both entry points produce identical candidate sets and identical
/// ordering for the same creature and cost of growth.
#[test]
#[serial]
fn both_entry_points_agree_on_candidates_and_ordering() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = mixed_creature();

    let shipped = ffi_removal_candidates(&creature, TEST_COST_OF_GROWTH);
    let adapted = triage_removal_candidates(&creature, Some(TEST_COST_OF_GROWTH));

    assert!(
        adapted.candidates.len() >= 2,
        "the fixture must yield several candidates or the ordering assertion is vacuous"
    );

    let shipped_ids: Vec<&str> = shipped
        .iter()
        .map(|c| c["neuronUuid"].as_str().expect("neuronUuid"))
        .collect();
    let adapted_ids: Vec<&str> = adapted
        .candidates
        .iter()
        .map(|c| c.neuron_uuid.as_str())
        .collect();

    assert_eq!(
        adapted_ids, shipped_ids,
        "the adapter and the shipped FFI path must return the same candidates in the same order"
    );
}

/// The per-candidate numbers — structural impact and the post-boost savings —
/// must match too, so the boost application point cannot move on one path only
/// (Issue #892 contract preserved through the merge).
#[test]
#[serial]
fn both_entry_points_agree_on_impact_and_boosted_savings() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = mixed_creature();

    let shipped = ffi_removal_candidates(&creature, TEST_COST_OF_GROWTH);
    let adapted = triage_removal_candidates(&creature, Some(TEST_COST_OF_GROWTH));

    for (a, s) in adapted.candidates.iter().zip(&shipped) {
        assert_eq!(
            a.impact,
            s["impact"].as_f64().expect("impact") as f32,
            "structural impact must match for {}",
            a.neuron_uuid
        );
        assert_eq!(
            a.removal_savings,
            s["removalSavings"].as_f64().expect("removalSavings") as f32,
            "post-boost savings must match for {}",
            a.neuron_uuid
        );
        assert_eq!(
            a.net_improvement,
            a.removal_savings - a.impact,
            "net improvement must stay savings − impact for {}",
            a.neuron_uuid
        );
    }
}

/// Sub-noise-floor candidates are dropped on both paths, and the shipped path
/// reports the drop under the stable rejection key while the adapter reports the
/// same count (Issue #1142 contract preserved through the merge).
///
/// **Issue #1814 changed this test's setup, not its contract.** The default
/// floor is now `REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS × costOfGrowth`, so at the
/// production cost-of-growth these ~2e-7 net improvements correctly survive.
/// The historical absolute floor is pinned so both entry points are still
/// checked for agreement on a *rejecting* floor.
#[test]
#[serial]
fn both_entry_points_agree_on_noise_floor_rejections() {
    let _floor = EnvVarGuard::set(NOISE_FLOOR_ENV, "1e-5");
    let creature = mixed_creature();

    // Production default growth cost against the pinned 1e-5 absolute floor:
    // every net improvement lands far below it.
    let shipped = ffi_removal_candidates(&creature, 1e-7);
    let adapted = triage_removal_candidates(&creature, Some(1e-7));

    assert!(
        shipped.is_empty(),
        "1.8e-7 net improvements are noise and must be dropped on the FFI path: {shipped:?}"
    );
    assert!(
        adapted.candidates.is_empty(),
        "…and on the adapter path: {:?}",
        adapted.candidates
    );
    // Only the three negligible-path neurons reach the noise-floor gate; the
    // three dominant ones fail the savings-vs-impact test outright and are never
    // noise-floor rejections.
    assert_eq!(
        adapted.noise_floor_rejections,
        (HIDDEN_COUNT / 2) as u32,
        "every dropped hidden neuron must be counted, not silently discarded"
    );
}

/// A non-positive `costOfGrowth` is a caller bug on **both** paths: it falls back
/// to the crate default instead of producing nonsense savings. Before the merge
/// only the adapter validated it, so the shipped FFI path silently emitted
/// garbage (or nothing) for the same input.
#[test]
#[serial]
fn invalid_cost_of_growth_falls_back_to_the_default_on_both_paths() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = mixed_creature();

    let default_shipped = ffi_removal_candidates(&creature, 1e-7);
    for invalid in [-1.0_f32, 0.0] {
        let shipped = ffi_removal_candidates(&creature, invalid);
        let adapted = triage_removal_candidates(&creature, Some(invalid));

        assert_eq!(
            shipped.len(),
            default_shipped.len(),
            "costOfGrowth {invalid} must behave as the 1e-7 default on the FFI path"
        );
        assert_eq!(
            adapted.candidates.len(),
            shipped.len(),
            "costOfGrowth {invalid} must agree across both entry points"
        );
    }
}
