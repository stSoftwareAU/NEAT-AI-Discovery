//! Issue #1767: removal selection must use near-opposite criteria to focus and
//! must not require focus-time parquet.
//!
//! Focus prefers **high** structural impact (Issue #1766). Removal is the
//! near-opposite axis: **low** contribution versus the complexity savings from
//! pruning. Critically, triaging removal candidates must not force a discovery
//! parquet warm during focus selection — that coupling burned ~2 h on a
//! production deployment before any useful discovery work started.
//!
//! # Why these tests drive the FFI entry point (Issue #1806)
//!
//! This suite used to call `focus::triage_removal_candidates` exclusively, which
//! at the time was a *second* implementation of the criterion with no caller in
//! `src/` — every behaviour below was pinned on code that did not ship. The
//! shipped path is
//! `focus::identify_structural_removal_candidates`, reached from
//! `rank_focus_neurons_internal` in `src/ffi_internal/analysis.rs`. It is
//! `pub(crate)`, so these tests reach it the way production does: through the
//! FFI entry point, which also covers the JSON mapping onto
//! `removalCandidates` / `rejectionBreakdown`.
//!
//! One case cannot be expressed on this surface: a **non-finite**
//! `costOfGrowth`. `NaN` and `Infinity` are not representable in JSON, so they
//! can never reach the FFI at all — that half of the cost-of-growth fallback is
//! pinned directly against the shipped function by
//! `unification_parity_tests::entry_points_agree_on_an_invalid_cost_of_growth`
//! in `src/focus/ranking/removal_triage.rs`. The reachable half (non-positive
//! values) is asserted here.

use neat_ai_discovery::analysis::diagnostics::rejection_reasons::REJECTION_REMOVAL_BELOW_NOISE_FLOOR;
use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::rank_focus_neurons_internal;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serde_json::{Value, json};
use serial_test::serial;
use std::time::Instant;

/// Cost-of-growth large enough that the complexity savings clear the
/// `REMOVE_LOW_IMPACT_NOISE_FLOOR` (1e-5) for the low-impact neurons. The
/// production default (1e-7) is exercised separately by
/// [`noise_floor_rejections_are_reported_not_silently_dropped`].
const TEST_COST_OF_GROWTH: f32 = 1e-4;

/// The noise-floor env var, unset in tests that depend on the default.
const NOISE_FLOOR_ENV: &str = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR";

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
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(v) = &self.previous {
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            unsafe { std::env::set_var(self.key, v) };
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

/// One input feeds a dominant hidden neuron (weight 1.0 into the output) and
/// three near-zero-contribution hidden neurons (weight 1e-8 into the output).
/// Structural impact is the normalised weight fraction, so `h-high` sits at the
/// top of the focus axis while `h-low-*` sit at the bottom.
///
/// Neurons are ordered inputs → hidden → output so every synapse points forward,
/// which the FFI's `validate_forward_only_synapses` gate requires (Issue #1184).
fn make_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("in-0", "input"),
            neuron("h-high", "hidden"),
            neuron("h-low-a", "hidden"),
            neuron("h-low-b", "hidden"),
            neuron("h-low-c", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![
            synapse("in-0", "h-high", 0.5),
            synapse("in-0", "h-low-a", 0.5),
            synapse("in-0", "h-low-b", 0.5),
            synapse("in-0", "h-low-c", 0.5),
            synapse("h-high", "out", 1.0),
            synapse("h-low-a", "out", 1e-8),
            synapse("h-low-b", "out", 1e-8),
            synapse("h-low-c", "out", 1e-8),
        ],
        input: 1,
        output: 1,
    }
}

/// Production shape: 16 hidden neurons + 1 output (17 selectable), one input.
///
/// The output is placed **last** so every hidden → output synapse points forward
/// (Issue #1184 gate on the FFI path).
fn make_production_shaped_creature() -> CreatureJson {
    let mut neurons = vec![neuron("in-0", "input")];
    let mut synapses = Vec::new();
    for i in 0..16 {
        let uuid = format!("h-{i}");
        neurons.push(neuron(&uuid, "hidden"));
        synapses.push(synapse("in-0", &uuid, 0.5));
        // Alternate a dominant and a negligible path to the output so both
        // axes have real mass.
        let weight = if i % 2 == 0 { 1.0 } else { 1e-8 };
        synapses.push(synapse(&uuid, "out", weight));
    }
    neurons.push(neuron("out", "output"));
    CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    }
}

/// Drive the shipped focus path — `rank_focus_neurons_internal`, which calls
/// `focus::identify_structural_removal_candidates` — and return the parsed
/// response.
///
/// The parquet path deliberately does not exist: focus selection is
/// structure-only (Issue #1766) and the removal triage must not reintroduce the
/// dependency, so a successful response *is* the no-parquet guarantee.
fn ffi_focus_response(creature: &CreatureJson, cost_of_growth: Option<f32>) -> Value {
    let input = json!({
        "parquetFile": MISSING_PARQUET,
        "creature": creature,
        "maxResults": 64,
        "focusSetSize": 4,
        "focusSelectionCursor": 0,
        "costOfGrowth": cost_of_growth,
    })
    .to_string();
    let response: Value =
        serde_json::from_str(&rank_focus_neurons_internal(&input).expect("FFI focus path"))
            .expect("FFI response JSON");
    assert_eq!(
        response["success"], true,
        "the structure-only FFI focus path must succeed: {response:?}"
    );
    response
}

/// A discovery parquet that cannot be opened — proof the shipped triage never
/// reads one.
const MISSING_PARQUET: &str = "/nonexistent/issue-1767/discovery.parquet";

/// The `removalCandidates` array from an FFI focus response (omitted when empty).
fn removal_candidates(response: &Value) -> Vec<Value> {
    response["removalCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

fn uuids(candidates: &[Value]) -> Vec<String> {
    candidates
        .iter()
        .map(|c| {
            c["neuronUuid"]
                .as_str()
                .expect("every candidate carries neuronUuid")
                .to_string()
        })
        .collect()
}

fn field(candidate: &Value, key: &str) -> f64 {
    candidate[key]
        .as_f64()
        .unwrap_or_else(|| panic!("candidate must carry a numeric {key}: {candidate:?}"))
}

/// `removalSavings − impact` — the ordering key the shipped sort uses.
fn net_improvement(candidate: &Value) -> f64 {
    field(candidate, "removalSavings") - field(candidate, "impact")
}

/// Noise-floor rejections reported under the stable FFI reason key.
fn noise_floor_rejections(response: &Value) -> u64 {
    response["rejectionBreakdown"][REJECTION_REMOVAL_BELOW_NOISE_FLOOR]
        .as_u64()
        .unwrap_or(0)
}

/// Acceptance: triaging removal candidates on the shipped path does not require
/// opening or decoding discovery parquet.
///
/// The parquet-backed ranker cannot even start without a readable discovery
/// file, while the shipped FFI focus path returns removal candidates for the
/// same creature and the same missing file.
#[test]
#[serial]
fn triage_needs_no_parquet_while_the_ranker_does() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    let ranked = rank_focus_neurons(MISSING_PARQUET, &creature, None, Some(TEST_COST_OF_GROWTH));
    assert!(
        ranked.is_err(),
        "the parquet-backed ranker must fail without a discovery file — otherwise this test proves nothing"
    );

    let response = ffi_focus_response(&creature, Some(TEST_COST_OF_GROWTH));
    assert!(
        !removal_candidates(&response).is_empty(),
        "the shipped removal triage must succeed with no discovery parquet present: {response:?}"
    );
    assert_eq!(
        response["loadingMode"],
        Value::Null,
        "no record-loading mode may be reported — a decode that never happened must not be claimed"
    );
}

/// Acceptance: removal uses near-**opposite** criteria to focus.
///
/// Over the same structural impact map the shipped path reports — the ranked
/// pool on the FFI response — the highest-impact neuron (focus's first pick) is
/// never offered for removal, and every removal candidate sits strictly below it
/// on the impact axis.
#[test]
#[serial]
fn removal_axis_is_opposite_to_the_focus_axis() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    let response = ffi_focus_response(&creature, Some(TEST_COST_OF_GROWTH));

    let ranked = response["neurons"]
        .as_array()
        .expect("the shipped focus path must report its ranked pool");
    let focus_pick = ranked
        .iter()
        .map(|n| {
            (
                n["neuronUuid"].as_str().expect("neuronUuid").to_string(),
                field(n, "impact").abs(),
            )
        })
        .filter(|(uuid, _)| uuid.starts_with("h-"))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("the ranked pool must contain the hidden neurons");
    assert_eq!(
        focus_pick.0, "h-high",
        "focus's high-impact axis must prefer the dominant-path neuron"
    );

    let candidates = removal_candidates(&response);
    let mut low = uuids(&candidates);
    assert!(
        !low.contains(&focus_pick.0),
        "the highest-impact neuron must never be a removal candidate; got {low:?}"
    );
    low.sort();
    assert_eq!(
        low,
        vec!["h-low-a", "h-low-b", "h-low-c"],
        "the low-contribution neurons are exactly the removal candidates"
    );

    for candidate in &candidates {
        assert!(
            field(candidate, "impact") < focus_pick.1,
            "removal candidate {:?} (impact {}) must sit below the focus pick (impact {})",
            candidate["neuronUuid"],
            field(candidate, "impact"),
            focus_pick.1
        );
    }
}

/// Only hidden neurons are prunable: outputs seed the impact map at 1.0 and are
/// the add-neuron targets; inputs are not selectable at all.
#[test]
#[serial]
fn inputs_and_outputs_are_never_triaged_for_removal() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    // A cost-of-growth this large would otherwise dwarf every impact value.
    let response = ffi_focus_response(&creature, Some(1.0e3));
    let candidates = uuids(&removal_candidates(&response));

    assert!(
        !candidates.is_empty(),
        "a huge cost-of-growth must still yield hidden-neuron candidates, or the gate assertion is vacuous"
    );
    assert!(
        !candidates.iter().any(|u| u == "out" || u == "in-0"),
        "inputs and outputs must never appear as removal candidates; got {candidates:?}"
    );
}

/// Sub-noise-floor candidates are dropped, but the drop is **reported** on the
/// shipped path's `rejectionBreakdown`, never silently swallowed (Issue #1142
/// contract carried onto the structural path).
#[test]
#[serial]
fn noise_floor_rejections_are_reported_not_silently_dropped() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    // Production default cost-of-growth: savings ≈ 1.8e-7, far below the 1e-5
    // noise floor, so all three low-impact neurons are rejected.
    let response = ffi_focus_response(&creature, None);

    assert!(
        removal_candidates(&response).is_empty(),
        "net improvements around 1.8e-7 are indistinguishable from noise; got {:?}",
        removal_candidates(&response)
    );
    assert_eq!(
        noise_floor_rejections(&response),
        3,
        "each dropped candidate must be counted under {REJECTION_REMOVAL_BELOW_NOISE_FLOOR}, not silently discarded: {response:?}"
    );
}

/// Candidates are ordered best-net-improvement first, so a caller taking the
/// head of the list prunes the biggest win. More synapses ⇒ larger savings.
#[test]
#[serial]
fn candidates_are_sorted_by_net_improvement_descending() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let mut creature = make_creature();
    // Give `h-low-b` two extra inbound synapses so pruning it saves more. The
    // new inputs are inserted ahead of the hidden neurons to keep the creature
    // forward-only for the FFI gate.
    creature.neurons.insert(1, neuron("in-1", "input"));
    creature.neurons.insert(2, neuron("in-2", "input"));
    creature.input = 3;
    creature.synapses.push(synapse("in-1", "h-low-b", 0.5));
    creature.synapses.push(synapse("in-2", "h-low-b", 0.5));

    let response = ffi_focus_response(&creature, Some(TEST_COST_OF_GROWTH));
    let candidates = removal_candidates(&response);
    let ids = uuids(&candidates);

    assert_eq!(
        ids.first().map(String::as_str),
        Some("h-low-b"),
        "the highest-savings neuron must lead the list; got {ids:?}"
    );
    for pair in candidates.windows(2) {
        assert!(
            net_improvement(&pair[0]) >= net_improvement(&pair[1]),
            "net improvement must be non-increasing: {} then {}",
            net_improvement(&pair[0]),
            net_improvement(&pair[1])
        );
    }
}

/// A non-positive cost-of-growth is a caller bug: the shipped path falls back to
/// the crate default rather than producing nonsense savings.
///
/// `NaN` / `Infinity` cannot be encoded in JSON, so they are unreachable through
/// the FFI; that half of the guard is pinned against the shipped function by
/// `entry_points_agree_on_an_invalid_cost_of_growth` in
/// `src/focus/ranking/removal_triage.rs` (Issue #1806).
#[test]
#[serial]
fn invalid_cost_of_growth_falls_back_to_the_default() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    let expected = ffi_focus_response(&creature, None);
    for invalid in [-1.0_f32, 0.0] {
        let actual = ffi_focus_response(&creature, Some(invalid));
        assert_eq!(
            removal_candidates(&actual),
            removal_candidates(&expected),
            "cost-of-growth {invalid} must fall back to the default candidate list"
        );
        assert_eq!(
            noise_floor_rejections(&actual),
            noise_floor_rejections(&expected),
            "cost-of-growth {invalid} must fall back to the default rejection count"
        );
    }
}

/// Acceptance: the seconds bar holds on the shipped (rayon) implementation. On a
/// production-shaped creature (17 selectable) the triage is topology-only, so it
/// completes in milliseconds regardless of how much discovery data exists on
/// disk.
#[test]
#[serial]
fn triage_meets_the_seconds_bar_on_a_production_shaped_creature() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_production_shaped_creature();

    let started = Instant::now();
    let response = ffi_focus_response(&creature, Some(TEST_COST_OF_GROWTH));
    let elapsed = started.elapsed();

    assert!(
        elapsed.as_secs_f64() < 1.0,
        "the shipped structural removal triage must finish well inside the seconds bar; took {elapsed:?}"
    );
    // The path reports its own wall clock, so a regression is visible even when
    // the harness is noisy.
    assert!(
        response["durationMs"].as_u64().expect("durationMs") < 1_000,
        "the shipped path must report a sub-second focus pass: {response:?}"
    );
    let candidates = removal_candidates(&response);
    assert_eq!(
        candidates.len(),
        8,
        "the eight negligible-path hidden neurons are the removal candidates; got {:?}",
        uuids(&candidates)
    );
}
