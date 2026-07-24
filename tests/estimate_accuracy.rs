//! Estimate-accuracy harness (Issue #1533, re-based by Issue #1722).
//!
//! Deliverable evidence for the #1529 milestone: **prove the error-reduction
//! estimates are accurate at depth.** The harness drives the honest,
//! propagation-aware estimators end-to-end from the committed deep-chain
//! snapshot and grades every candidate against the analytic reference effect —
//! the closed-form propagated effect derived by hand from the topology, so the
//! grading is a genuine oracle rather than a recording of whatever the
//! implementation happened to emit.
//!
//! Every estimate is graded against the three #1529 pass criteria, each asserted
//! **across the full candidate set** (both change types), never a single neuron:
//! - [`estimate_sign_matches_reference`] — estimate and analytic reference agree
//!   in direction, for every candidate.
//! - [`estimate_within_10x_of_reference`] — estimate is within one order of
//!   magnitude of the analytic reference, for every candidate.
//! - [`estimate_ranking_orders_candidates`] — the estimator orders the candidates
//!   by effect magnitude the same way the analytic references do, so a single
//!   lucky per-candidate match cannot pass the suite.
//!
//! The candidate set is assembled from the committed candidate-record fixtures
//! and spans both estimator paths wired for the milestone:
//! - **remove-neuron** — [`estimate_remove_neuron_gain`] (Issue #1518/#1530),
//! - **change-squash** — [`estimate_change_squash_gain`] (Issue #1532).
//!
//! Why this fails on the pre-fix estimators (the harness is the acceptance gate):
//! the retired placeholders fabricated a large **positive** remove-neuron gain
//! (`+0.18`) and a near-zero **positive** change-squash gain (`+5e-10`). Both
//! flip the sign case and blow the 10× case, and — because the remove-neuron
//! placeholder (`0.18`) dwarfs the change-squash placeholder (`5e-10`) while the
//! analytic references rank the other way — they also **invert** the ranking
//! case. Only the honest propagation-aware estimators pass all three.
//!
//! Every fixture here is hand-authored and synthetic, so this public repository
//! stays fully self-contained (Issue #1722).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::{estimate_change_squash_gain, estimate_remove_neuron_gain};
use std::path::{Path, PathBuf};

/// The deep-chain snapshot dimensions. If the committed topology ever
/// deserialises to different dimensions the whole suite fails fast at setup
/// (rather than silently passing on a changed shape whose analytic references no
/// longer hold).
const EXPECTED_NEURONS: usize = 27;
const EXPECTED_SYNAPSES: usize = 40;

fn remove_neuron_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/remove_neuron_propagation")
}

fn change_squash_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/change_squash_propagation")
}

/// Load the deep-chain creature topology from the committed fixture (shared by
/// both change types — both candidates sit on the same creature). A missing /
/// corrupt fixture fails here with the path, so a hermeticity breakage is caught
/// in the same CI run rather than downstream.
fn load_network() -> CreatureJson {
    let path = remove_neuron_fixture_dir().join("network.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read network fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse network fixture {}: {e}", path.display()))
}

/// Read a JSON fixture into a `serde_json::Value`, failing with the path on any
/// read / parse error.
fn load_json(path: &Path) -> serde_json::Value {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse fixture {}: {e}", path.display()))
}

/// One candidate: the honest estimate produced by the wired estimator and the
/// analytic reference effect it is graded against.
struct Candidate {
    /// Human-readable label used in ranking / failure messages.
    label: &'static str,
    /// The change-type path this candidate exercises (`remove-neuron` /
    /// `change-squash`) — the set must span both so the harness proves accuracy
    /// across every wired estimator, not just one.
    change_type: &'static str,
    /// The honest, propagation-aware estimate.
    estimate: f64,
    /// The closed-form propagated effect derived from the committed topology.
    reference: f64,
}

/// Assemble the full candidate set: one candidate record per wired change type,
/// each with its honest estimate and analytic reference, driven by the committed
/// deep-chain snapshot.
///
/// This is the reusable core of the harness — extending it with another
/// candidate fixture automatically subjects the new candidate to all three
/// criteria below.
fn load_candidates(creature: &CreatureJson) -> Vec<Candidate> {
    let mut candidates = Vec::new();

    // --- remove-neuron candidate (spine-0, 13 halving hops) ----------------
    {
        let json = load_json(&remove_neuron_fixture_dir().join("v2_remove-neuron_spine-0.json"));
        let candidate = &json["rustRequest"]["harmfulNeuronCandidate"];
        let uuid = candidate["neuronUuid"]
            .as_str()
            .expect("remove-neuron fixture missing neuronUuid");
        let reference = json["analyticErrorReduction"]
            .as_f64()
            .expect("remove-neuron fixture missing analyticErrorReduction");
        let estimate = estimate_remove_neuron_gain(creature, uuid)
            .expect("remove-neuron estimator must return a gain for the candidate hidden neuron");
        candidates.push(Candidate {
            label: "remove-neuron:spine-0",
            change_type: "remove-neuron",
            estimate,
            reference,
        });
    }

    // --- change-squash candidate (spine-1, 12 halving hops) ----------------
    {
        let json = load_json(&change_squash_fixture_dir().join("v2_change-squash_spine-1.json"));
        let candidate = &json["rustRequest"]["squashCandidate"];
        let uuid = candidate["neuronUuid"]
            .as_str()
            .expect("change-squash fixture missing neuronUuid");
        let current_local_error = candidate["currentError"]
            .as_f64()
            .expect("change-squash fixture missing currentError");
        let proposed_local_error = candidate["improvedError"]
            .as_f64()
            .expect("change-squash fixture missing improvedError");
        let reference = json["analyticErrorReduction"]
            .as_f64()
            .expect("change-squash fixture missing analyticErrorReduction");
        let estimate =
            estimate_change_squash_gain(creature, uuid, current_local_error, proposed_local_error)
                .expect(
                    "change-squash estimator must return a gain for the candidate hidden neuron",
                );
        candidates.push(Candidate {
            label: "change-squash:spine-1",
            change_type: "change-squash",
            estimate,
            reference,
        });
    }

    candidates
}

/// Fail-fast setup shared by every criterion: load the snapshot, assert its
/// dimensions, assemble the candidate set, and assert the set is non-empty and
/// spans both wired change types. A broken fixture or an empty candidate set
/// turns the suite red at setup rather than passing vacuously.
fn setup() -> Vec<Candidate> {
    let creature = load_network();
    assert_eq!(
        creature.neurons.len(),
        EXPECTED_NEURONS,
        "fixture creature is not the expected {EXPECTED_NEURONS}-neuron deep-chain snapshot"
    );
    assert_eq!(
        creature.synapses.len(),
        EXPECTED_SYNAPSES,
        "fixture creature is not the expected {EXPECTED_SYNAPSES}-synapse deep-chain snapshot"
    );

    let candidates = load_candidates(&creature);

    // Harness-integrity guard: never grade an empty / single-candidate set — the
    // ranking check is meaningless without at least two candidates, and a silent
    // empty set would make every criterion pass vacuously.
    assert!(
        candidates.len() >= 2,
        "candidate set must contain at least two candidates, got {}",
        candidates.len()
    );
    assert!(
        candidates.iter().any(|c| c.change_type == "remove-neuron"),
        "candidate set must cover the remove-neuron estimator path"
    );
    assert!(
        candidates.iter().any(|c| c.change_type == "change-squash"),
        "candidate set must cover the change-squash estimator path"
    );

    candidates
}

/// Ratio of two magnitudes, guarding against a zero denominator.
fn magnitude_ratio(a: f64, b: f64) -> f64 {
    let denom = b.abs();
    assert!(denom > 0.0, "magnitude_ratio: zero denominator");
    a.abs() / denom
}

/// True when `a` and `b` share magnitude to within one order of magnitude (10×).
fn within_one_order(a: f64, b: f64) -> bool {
    (0.1..=10.0).contains(&magnitude_ratio(a, b))
}

/// Criterion 1 — correct sign. Every candidate's estimate must agree in
/// direction with its analytic reference, across the full candidate set.
#[test]
fn estimate_sign_matches_reference() {
    let candidates = setup();
    for c in &candidates {
        assert_eq!(
            c.estimate.signum(),
            c.reference.signum(),
            "{}: estimate {:e} and analytic reference {:e} must agree in sign",
            c.label,
            c.estimate,
            c.reference
        );
    }
}

/// Criterion 2 — within 10×. Every candidate's estimate must match its analytic
/// reference within one order of magnitude, across the full candidate set.
#[test]
fn estimate_within_10x_of_reference() {
    let candidates = setup();
    for c in &candidates {
        assert!(
            within_one_order(c.estimate, c.reference),
            "{}: estimate {:e} must be within 10× of the analytic reference {:e} \
             (ratio {:e})",
            c.label,
            c.estimate,
            c.reference,
            magnitude_ratio(c.estimate, c.reference)
        );
    }
}

/// Criterion 3 — ranking holds. The estimator must order the candidates by
/// effect magnitude the same way the analytic references do. Asserted as a
/// pairwise concordance over *every* pair in the set (a Kendall-style check that
/// generalises to any number of candidates), so no single lucky per-candidate
/// match can mask a broken estimator that mis-ranks the set.
///
/// This is where the retired placeholders are caught even if they somehow slid
/// past the per-candidate checks: the remove-neuron placeholder (`0.18`) dwarfs
/// the change-squash placeholder (`5e-10`), yet the analytic references rank the
/// change-squash effect (`4.9e-4`) above the remove-neuron effect (`1.2e-4`) —
/// an inversion this concordance check fails on.
#[test]
fn estimate_ranking_orders_candidates() {
    let candidates = setup();

    let mut compared = 0usize;
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let a = &candidates[i];
            let b = &candidates[j];

            // Order by effect magnitude: the estimator claims which candidate has
            // the larger-magnitude effect, and the analytic references decide the
            // truth. Both orderings must agree.
            let estimate_order = a.estimate.abs().partial_cmp(&b.estimate.abs()).unwrap();
            let reference_order = a.reference.abs().partial_cmp(&b.reference.abs()).unwrap();

            assert_eq!(
                estimate_order, reference_order,
                "ranking inversion between {} (estimate {:e}, reference {:e}) and {} \
                 (estimate {:e}, reference {:e}): the estimator must order candidates \
                 by effect magnitude the same way the analytic references do",
                a.label, a.estimate, a.reference, b.label, b.estimate, b.reference
            );
            compared += 1;
        }
    }

    // Guard against a degenerate single-candidate set silently skipping the loop.
    assert!(
        compared >= 1,
        "ranking check compared no candidate pairs; the candidate set is too small"
    );
}
