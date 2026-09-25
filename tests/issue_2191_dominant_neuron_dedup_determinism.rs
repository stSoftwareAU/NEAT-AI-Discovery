//! Issue #2191: dominant-neuron deduplication must order tied candidates
//! deterministically.
//!
//! Both deduplicators group candidates in a `HashMap` (per-instance random
//! seed) and then stable-sort on `combined_improvement` alone, so equally
//! scored candidates came out in hash-seed order and a discovery run was not
//! reproducible. The fix breaks ties on the candidate UUIDs, so the output is
//! a pure function of the input set: identical across runs and independent of
//! input order, including which members a tied group keeps when truncated.

use neat_ai_discovery::analysis::recommendation::epistatic::{
    EpistaticPairCandidate, SynergisticCandidate, deduplicate_by_dominant_neuron,
    deduplicate_synergistic_by_dominant_neuron,
};

/// Runs per deduplicator; each run builds a fresh `HashMap` with a new seed.
const RUNS: usize = 64;
/// Tied score shared by every fixture candidate.
const TIED: f32 = 0.5;

fn pair(source_a: &str, source_b: &str) -> EpistaticPairCandidate {
    EpistaticPairCandidate {
        source_a_uuid: source_a.to_string(),
        source_b_uuid: source_b.to_string(),
        target_uuid: "output-0".to_string(),
        weight_a: 0.1,
        weight_b: 0.1,
        combined_improvement: TIED,
        // `a` is always dominant, so the group key is `source_a`.
        individual_improvement_a: 0.3,
        individual_improvement_b: 0.1,
        complementarity_score: 0.5,
        reason: "Issue #2191 fixture".to_string(),
    }
}

fn synergistic(primary: &str, complement: &str) -> SynergisticCandidate {
    SynergisticCandidate {
        primary_source_uuid: primary.to_string(),
        complement_source_uuid: complement.to_string(),
        target_uuid: "output-0".to_string(),
        primary_weight: 0.1,
        complement_weight: 0.1,
        combined_improvement: TIED,
        primary_improvement: 0.3,
        complement_improvement: 0.1,
        residual_reduction: 0.5,
        synergy_ratio: 1.5,
        reason: "Issue #2191 fixture".to_string(),
    }
}

/// Eight singleton groups with distinct dominant neurons, plus one group of
/// five whose partners arrive out of UUID order so truncation to three is
/// exercised on a tie.
fn fixture_ids() -> Vec<(String, String)> {
    let mut ids: Vec<(String, String)> = (0..8)
        .map(|i| (format!("dom-{i}"), format!("partner-{i}")))
        .collect();
    for partner in ["p-e", "p-b", "p-d", "p-a", "p-c"] {
        ids.push(("dom-big".to_string(), partner.to_string()));
    }
    ids
}

/// Expected output: all nine groups' survivors, ordered by UUID tuple, with
/// the big group keeping its three lowest-ordered partners.
fn expected_ids() -> Vec<(String, String)> {
    let mut ids: Vec<(String, String)> = (0..8)
        .map(|i| (format!("dom-{i}"), format!("partner-{i}")))
        .collect();
    for partner in ["p-a", "p-b", "p-c"] {
        ids.push(("dom-big".to_string(), partner.to_string()));
    }
    ids.sort();
    ids
}

fn epistatic_ids(pairs: &[EpistaticPairCandidate]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|p| (p.source_a_uuid.clone(), p.source_b_uuid.clone()))
        .collect()
}

fn synergistic_ids(candidates: &[SynergisticCandidate]) -> Vec<(String, String)> {
    candidates
        .iter()
        .map(|c| {
            (
                c.primary_source_uuid.clone(),
                c.complement_source_uuid.clone(),
            )
        })
        .collect()
}

#[test]
fn epistatic_dedup_orders_tied_candidates_identically_on_every_run() {
    let input: Vec<EpistaticPairCandidate> =
        fixture_ids().iter().map(|(a, b)| pair(a, b)).collect();
    let first = epistatic_ids(&deduplicate_by_dominant_neuron(input.clone()));

    for run in 1..RUNS {
        let got = epistatic_ids(&deduplicate_by_dominant_neuron(input.clone()));
        assert_eq!(
            got, first,
            "run {run} ordered tied candidates differently from run 0"
        );
    }
    assert_eq!(first, expected_ids());
}

#[test]
fn epistatic_dedup_is_independent_of_input_order() {
    let mut reversed: Vec<EpistaticPairCandidate> =
        fixture_ids().iter().map(|(a, b)| pair(a, b)).collect();
    reversed.reverse();

    let got = epistatic_ids(&deduplicate_by_dominant_neuron(reversed));
    assert_eq!(got, expected_ids());
}

#[test]
fn synergistic_dedup_orders_tied_candidates_identically_on_every_run() {
    let input: Vec<SynergisticCandidate> = fixture_ids()
        .iter()
        .map(|(p, c)| synergistic(p, c))
        .collect();
    let first = synergistic_ids(&deduplicate_synergistic_by_dominant_neuron(input.clone()));

    for run in 1..RUNS {
        let got = synergistic_ids(&deduplicate_synergistic_by_dominant_neuron(input.clone()));
        assert_eq!(
            got, first,
            "run {run} ordered tied candidates differently from run 0"
        );
    }
    assert_eq!(first, expected_ids());
}

#[test]
fn synergistic_dedup_is_independent_of_input_order() {
    let mut reversed: Vec<SynergisticCandidate> = fixture_ids()
        .iter()
        .map(|(p, c)| synergistic(p, c))
        .collect();
    reversed.reverse();

    let got = synergistic_ids(&deduplicate_synergistic_by_dominant_neuron(reversed));
    assert_eq!(got, expected_ids());
}

#[test]
fn higher_combined_improvement_still_ranks_first() {
    let mut low = pair("dom-a", "partner-a");
    low.combined_improvement = 0.1;
    let mut high = pair("dom-z", "partner-z");
    high.combined_improvement = 0.9;
    let input = vec![low, pair("dom-m", "partner-m"), high, pair("dom-c", "p")];

    let got = epistatic_ids(&deduplicate_by_dominant_neuron(input));
    let order: Vec<&str> = got.iter().map(|(a, _)| a.as_str()).collect();
    assert_eq!(order, ["dom-z", "dom-c", "dom-m", "dom-a"]);
}
