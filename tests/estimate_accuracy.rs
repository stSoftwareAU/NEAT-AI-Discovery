//! Production-scale estimate-accuracy harness (Issue #1533).
//!
//! Deliverable evidence for the #1529 milestone: **prove the error-reduction
//! estimates are accurate for a production-scale creature — not a toy example.**
//! The harness drives the honest, propagation-aware estimators end-to-end from
//! the committed 1,666-neuron / 21,532-synapse GRQ-cluster snapshot and grades
//! every recorded change against the empirically measured actual error change.
//!
//! Every estimate is graded against the three #1529 pass criteria, each asserted
//! **across the full candidate set** (both change types), never a single neuron:
//! - [`estimate_sign_matches_actual`] — estimate and established actual agree in
//!   direction, for every candidate.
//! - [`estimate_within_10x_of_actual`] — estimate is within one order of
//!   magnitude of the established actual, for every candidate.
//! - [`estimate_ranking_orders_candidates`] — the estimator orders the
//!   candidates by effect magnitude the same way the established actuals do, so a
//!   single lucky per-candidate match cannot pass the suite.
//!
//! The candidate set is assembled from the committed production failure fixtures
//! and spans both estimator paths wired for the milestone:
//! - **remove-neuron** — [`estimate_remove_neuron_gain`] (Issue #1518/#1530),
//! - **change-squash** — [`estimate_change_squash_gain`] (Issue #1532).
//!
//! Why this fails on the pre-fix estimators (the harness is the acceptance gate):
//! the retired placeholders fabricated a large **positive** remove-neuron gain
//! (`+0.17882921`) and a near-zero **positive** change-squash gain (`+8.6e-10`).
//! Both flip the sign case and blow the 10× case, and — because the remove-neuron
//! placeholder (`0.18`) dwarfs the change-squash placeholder (`8.6e-10`) while the
//! measured actuals rank the other way — they also **invert** the ranking case.
//! Only the honest propagation-aware estimators pass all three.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::{estimate_change_squash_gain, estimate_remove_neuron_gain};
use std::path::{Path, PathBuf};

/// The production GRQ-cluster snapshot dimensions. If the committed topology ever
/// deserialises to different dimensions the whole suite fails fast at setup
/// (rather than silently passing on an empty / wrong candidate set).
const EXPECTED_NEURONS: usize = 1666;
const EXPECTED_SYNAPSES: usize = 21_532;

fn remove_neuron_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/remove_neuron_propagation")
}

fn change_squash_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/change_squash_propagation")
}

/// Load the production creature topology from the committed fixture (shared by
/// both change types — the recorded failures are on the same creature). A
/// missing / corrupt fixture fails here with the path, so a hermeticity breakage
/// is caught in the same CI run rather than at runtime in GRQ-cluster.
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

/// One production candidate: the honest estimate produced by the wired estimator
/// and the empirically measured actual error change it is graded against.
struct Candidate {
    /// Human-readable label used in ranking / failure messages.
    label: &'static str,
    /// The change-type path this candidate exercises (`remove-neuron` /
    /// `change-squash`) — the set must span both so the harness proves accuracy
    /// across every wired estimator, not just one.
    change_type: &'static str,
    /// The honest, propagation-aware estimate.
    estimate: f64,
    /// The established (measured) actual error change from the production run.
    actual: f64,
}

/// Assemble the full production candidate set: one recorded failure per wired
/// change type, each with its honest estimate and established actual, driven by
/// the committed GRQ-cluster snapshot.
///
/// This is the reusable core of the harness — extending it with another recorded
/// failure fixture automatically subjects the new candidate to all three
/// criteria below.
fn load_candidates(creature: &CreatureJson) -> Vec<Candidate> {
    let mut candidates = Vec::new();

    // --- remove-neuron candidate (neuron-1802938338) -----------------------
    {
        let json =
            load_json(&remove_neuron_fixture_dir().join("v2_remove-neuron_neuron-1802938338.json"));
        let candidate = &json["rustRequest"]["harmfulNeuronCandidate"];
        let uuid = candidate["neuronUuid"]
            .as_str()
            .expect("remove-neuron fixture missing neuronUuid");
        let actual = json["actualErrorReduction"]
            .as_f64()
            .expect("remove-neuron fixture missing actualErrorReduction");
        let estimate = estimate_remove_neuron_gain(creature, uuid)
            .expect("remove-neuron estimator must return a gain for the recorded hidden neuron");
        candidates.push(Candidate {
            label: "remove-neuron:neuron-1802938338",
            change_type: "remove-neuron",
            estimate,
            actual,
        });
    }

    // --- change-squash candidate (neuron-1481550544) -----------------------
    {
        let json =
            load_json(&change_squash_fixture_dir().join("v2_change-squash_neuron-1481550544.json"));
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
        let actual = json["actualErrorReduction"]
            .as_f64()
            .expect("change-squash fixture missing actualErrorReduction");
        let estimate =
            estimate_change_squash_gain(creature, uuid, current_local_error, proposed_local_error)
                .expect(
                    "change-squash estimator must return a gain for the recorded hidden neuron",
                );
        candidates.push(Candidate {
            label: "change-squash:neuron-1481550544",
            change_type: "change-squash",
            estimate,
            actual,
        });
    }

    candidates
}

/// Fail-fast setup shared by every criterion: load the production snapshot,
/// assert its dimensions, assemble the candidate set, and assert the set is
/// non-empty and spans both wired change types. A broken fixture or an empty
/// candidate set turns the suite red at setup rather than passing vacuously.
fn setup() -> Vec<Candidate> {
    let creature = load_network();
    assert_eq!(
        creature.neurons.len(),
        EXPECTED_NEURONS,
        "fixture creature is not the expected {EXPECTED_NEURONS}-neuron production snapshot"
    );
    assert_eq!(
        creature.synapses.len(),
        EXPECTED_SYNAPSES,
        "fixture creature is not the expected {EXPECTED_SYNAPSES}-synapse production snapshot"
    );

    let candidates = load_candidates(&creature);

    // Harness-integrity guard: never grade an empty / single-candidate set — the
    // ranking check is meaningless without at least two candidates, and a silent
    // empty set would make every criterion pass vacuously.
    assert!(
        candidates.len() >= 2,
        "candidate set must contain at least two production candidates, got {}",
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
/// direction with its established actual, across the full candidate set.
#[test]
fn estimate_sign_matches_actual() {
    let candidates = setup();
    for c in &candidates {
        assert_eq!(
            c.estimate.signum(),
            c.actual.signum(),
            "{}: estimate {:e} and established actual {:e} must agree in sign",
            c.label,
            c.estimate,
            c.actual
        );
    }
}

/// Criterion 2 — within 10×. Every candidate's estimate must match its
/// established actual within one order of magnitude, across the full candidate
/// set.
#[test]
fn estimate_within_10x_of_actual() {
    let candidates = setup();
    for c in &candidates {
        assert!(
            within_one_order(c.estimate, c.actual),
            "{}: estimate {:e} must be within 10× of the established actual {:e} \
             (ratio {:e})",
            c.label,
            c.estimate,
            c.actual,
            magnitude_ratio(c.estimate, c.actual)
        );
    }
}

/// Criterion 3 — ranking holds. The estimator must order the candidates by
/// effect magnitude the same way the established actuals do. Asserted as a
/// pairwise concordance over *every* pair in the set (a Kendall-style check that
/// generalises to any number of candidates), so no single lucky per-candidate
/// match can mask a broken estimator that mis-ranks the set.
///
/// This is where the retired placeholders are caught even if they somehow slid
/// past the per-candidate checks: the remove-neuron placeholder (`0.18`) dwarfs
/// the change-squash placeholder (`8.6e-10`), yet the measured actuals rank the
/// change-squash effect (`3.4e-4`) above the remove-neuron effect (`1.9e-4`) —
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
            // the larger-magnitude effect, and the established actuals decide the
            // truth. Both orderings must agree.
            let estimate_order = a.estimate.abs().partial_cmp(&b.estimate.abs()).unwrap();
            let actual_order = a.actual.abs().partial_cmp(&b.actual.abs()).unwrap();

            assert_eq!(
                estimate_order, actual_order,
                "ranking inversion between {} (estimate {:e}, actual {:e}) and {} \
                 (estimate {:e}, actual {:e}): the estimator must order candidates \
                 by effect magnitude the same way the established actuals do",
                a.label, a.estimate, a.actual, b.label, b.estimate, b.actual
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
