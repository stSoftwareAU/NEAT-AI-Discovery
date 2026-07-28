//! Issue #1767: removal selection must use near-opposite criteria to focus and
//! must not require focus-time parquet.
//!
//! Focus prefers **high** structural impact (Issue #1766). Removal is the
//! near-opposite axis: **low** contribution versus the complexity savings from
//! pruning. Critically, triaging removal candidates must not force a discovery
//! parquet warm during focus selection — that coupling burned ~2 h on a
//! production deployment before any useful discovery work started.
//!
//! These tests exercise `triage_removal_candidates` directly: it takes only a
//! creature, so a run succeeds where the parquet-backed ranker cannot even
//! start, the axes are provably opposite over the same impact map, and the
//! triage finishes far inside the seconds bar.

use neat_ai_discovery::focus::{
    compute_impacts_public, rank_focus_neurons, triage_removal_candidates,
};
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
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
fn make_production_shaped_creature() -> CreatureJson {
    let mut neurons = vec![neuron("in-0", "input"), neuron("out", "output")];
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
    CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    }
}

fn uuids(triage: &neat_ai_discovery::focus::StructuralRemovalTriage) -> Vec<String> {
    triage
        .candidates
        .iter()
        .map(|c| c.neuron_uuid.clone())
        .collect()
}

/// Acceptance: triaging removal candidates does not require opening or decoding
/// discovery parquet.
///
/// The parquet-backed ranker cannot even start without a readable discovery
/// file, while the structural triage returns candidates for the same creature.
#[test]
#[serial]
fn triage_needs_no_parquet_while_the_ranker_does() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();
    let missing_parquet = "/nonexistent/issue-1767/discovery.parquet";

    let ranked = rank_focus_neurons(missing_parquet, &creature, None, Some(TEST_COST_OF_GROWTH));
    assert!(
        ranked.is_err(),
        "the parquet-backed ranker must fail without a discovery file — otherwise this test proves nothing"
    );

    let triage = triage_removal_candidates(&creature, Some(TEST_COST_OF_GROWTH));
    assert!(
        !triage.candidates.is_empty(),
        "structural removal triage must succeed with no discovery parquet present"
    );
}

/// Acceptance: removal uses near-**opposite** criteria to focus.
///
/// Over the same structural impact map, the highest-impact neuron (focus's
/// first pick) is never offered for removal, and every removal candidate sits
/// strictly below it on the impact axis.
#[test]
#[serial]
fn removal_axis_is_opposite_to_the_focus_axis() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    let impacts = compute_impacts_public(&creature);
    let focus_pick = impacts
        .iter()
        .filter(|(uuid, _)| uuid.as_str().starts_with("h-"))
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map(|(uuid, _)| uuid.clone())
        .expect("impact map must contain the hidden neurons");
    assert_eq!(
        focus_pick, "h-high",
        "focus's high-impact axis must prefer the dominant-path neuron"
    );

    let triage = triage_removal_candidates(&creature, Some(TEST_COST_OF_GROWTH));
    let candidates = uuids(&triage);

    assert!(
        !candidates.contains(&focus_pick),
        "the highest-impact neuron must never be a removal candidate; got {candidates:?}"
    );
    let mut low: Vec<String> = candidates;
    low.sort();
    assert_eq!(
        low,
        vec!["h-low-a", "h-low-b", "h-low-c"],
        "the low-contribution neurons are exactly the removal candidates"
    );

    let focus_impact = impacts[&focus_pick].abs();
    for candidate in &triage.candidates {
        assert!(
            candidate.impact < focus_impact,
            "removal candidate {} (impact {}) must sit below the focus pick (impact {focus_impact})",
            candidate.neuron_uuid,
            candidate.impact
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
    let triage = triage_removal_candidates(&creature, Some(1.0e3));
    let candidates = uuids(&triage);

    assert!(
        !candidates.iter().any(|u| u == "out" || u == "in-0"),
        "inputs and outputs must never appear as removal candidates; got {candidates:?}"
    );
}

/// Sub-noise-floor candidates are dropped, but the drop is **reported**, never
/// silently swallowed (Issue #1142 contract carried onto the structural path).
#[test]
#[serial]
fn noise_floor_rejections_are_reported_not_silently_dropped() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    // Production default cost-of-growth: savings ≈ 1.8e-7, far below the 1e-5
    // noise floor, so all three low-impact neurons are rejected.
    let triage = triage_removal_candidates(&creature, None);

    assert!(
        triage.candidates.is_empty(),
        "net improvements around 1.8e-7 are indistinguishable from noise; got {:?}",
        uuids(&triage)
    );
    assert_eq!(
        triage.noise_floor_rejections, 3,
        "each dropped candidate must be counted, not silently discarded"
    );
}

/// Candidates are ordered best-net-improvement first, so a caller taking the
/// head of the list prunes the biggest win. More synapses ⇒ larger savings.
#[test]
#[serial]
fn candidates_are_sorted_by_net_improvement_descending() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let mut creature = make_creature();
    // Give `h-low-b` two extra inbound synapses so pruning it saves more.
    creature.neurons.push(neuron("in-1", "input"));
    creature.neurons.push(neuron("in-2", "input"));
    creature.input = 3;
    creature.synapses.push(synapse("in-1", "h-low-b", 0.5));
    creature.synapses.push(synapse("in-2", "h-low-b", 0.5));

    let triage = triage_removal_candidates(&creature, Some(TEST_COST_OF_GROWTH));
    let candidates = uuids(&triage);

    assert_eq!(
        candidates.first().map(String::as_str),
        Some("h-low-b"),
        "the highest-savings neuron must lead the list; got {candidates:?}"
    );
    for pair in triage.candidates.windows(2) {
        assert!(
            pair[0].net_improvement >= pair[1].net_improvement,
            "net improvement must be non-increasing: {} then {}",
            pair[0].net_improvement,
            pair[1].net_improvement
        );
    }
}

/// A non-finite or non-positive cost-of-growth is a caller bug: it falls back to
/// the crate default rather than producing nonsense savings.
#[test]
#[serial]
fn invalid_cost_of_growth_falls_back_to_the_default() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    let expected = triage_removal_candidates(&creature, None);
    for invalid in [f32::NAN, f32::INFINITY, -1.0, 0.0] {
        let actual = triage_removal_candidates(&creature, Some(invalid));
        assert_eq!(
            actual, expected,
            "cost-of-growth {invalid} must fall back to the default triage result"
        );
    }
}

/// Acceptance: the seconds bar holds. On a production-shaped creature (17
/// selectable) the triage is topology-only, so it completes in milliseconds
/// regardless of how much discovery data exists on disk.
#[test]
#[serial]
fn triage_meets_the_seconds_bar_on_a_production_shaped_creature() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_production_shaped_creature();

    let started = Instant::now();
    let triage = triage_removal_candidates(&creature, Some(TEST_COST_OF_GROWTH));
    let elapsed = started.elapsed();

    assert!(
        elapsed.as_secs_f64() < 1.0,
        "structural removal triage must finish well inside the seconds bar; took {elapsed:?}"
    );
    assert_eq!(
        triage.candidates.len(),
        8,
        "the eight negligible-path hidden neurons are the removal candidates; got {:?}",
        uuids(&triage)
    );
}
