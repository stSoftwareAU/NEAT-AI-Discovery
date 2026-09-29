//! Issue #2181 / #2303: a non-finite input–error correlation must never enter
//! the fan-in `MAX_INPUTS_PER_TARGET` window, and the fan-in candidate set
//! must not depend on the order the caller lists neurons in.
//!
//! `pearson_correlation`'s `f32` accumulators overflow on finite `±2e30`
//! activations and return a NaN. Before the fix the `corr.abs() < 0.3` filter
//! kept it (a NaN loses `<`), and the `partial_cmp(..).unwrap_or(Equal)` sort
//! gave `truncate` no total order — so listing the poisoned inputs first
//! emptied the window of honest inputs.

use neat_ai_discovery::analysis::recommendation::fan_in::{
    MAX_INPUTS_PER_TARGET, detect_fan_in_candidates, rank_input_scores,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson};

// =============================================================================
// Helper contract — `rank_input_scores` fed the score vector directly
// =============================================================================

/// UUIDs and correlations of the ranked window, in rank order.
fn ranked(scores: &[(&'static str, f32)]) -> Vec<(String, f32)> {
    let input = scores
        .iter()
        .map(|&(uuid, corr)| (uuid, corr, ()))
        .collect();
    rank_input_scores(input)
        .into_iter()
        .map(|(uuid, corr, ())| (uuid.to_string(), corr))
        .collect()
}

fn mixed_scores() -> Vec<(&'static str, f32)> {
    vec![
        ("nan-a", f32::NAN),
        ("honest-0.4", 0.4),
        ("pos-inf", f32::INFINITY),
        ("honest-neg-0.9", -0.9),
        ("weak-0.1", 0.1),
        ("nan-b", -f32::NAN),
        ("honest-0.6", 0.6),
        ("neg-inf", f32::NEG_INFINITY),
        ("honest-0.3", 0.3),
    ]
}

#[test]
fn ranking_drops_non_finite_and_weak_correlations_and_orders_by_strength() {
    let window = ranked(&mixed_scores());

    assert!(
        window.iter().all(|(_, corr)| corr.is_finite()),
        "no NaN or ±inf correlation may enter the ranked window, got {window:?}"
    );
    let uuids: Vec<&str> = window.iter().map(|(u, _)| u.as_str()).collect();
    assert_eq!(
        uuids,
        vec!["honest-neg-0.9", "honest-0.6", "honest-0.4", "honest-0.3"],
        "only honest inputs at or above the threshold survive, strongest |corr| first"
    );
}

#[test]
fn ranking_does_not_depend_on_the_input_order() {
    let forward = mixed_scores();
    let mut reversed = forward.clone();
    reversed.reverse();
    // A fixed interleave, so the NaNs land at different relative positions.
    let shuffled: Vec<_> = [5, 2, 8, 0, 7, 3, 1, 6, 4]
        .iter()
        .map(|&i| forward[i])
        .collect();

    // Two honest inputs share a |corr| so the UUID tie-break is exercised.
    let mut tied = forward.clone();
    tied.push(("honest-tie-b", 0.6));
    tied.push(("honest-tie-a", -0.6));
    let mut tied_reversed = tied.clone();
    tied_reversed.reverse();

    let render = |w: Vec<(String, f32)>| w.into_iter().map(|(u, _)| u).collect::<Vec<_>>();
    let baseline = render(ranked(&forward));
    assert_eq!(render(ranked(&reversed)), baseline, "reversed order");
    assert_eq!(render(ranked(&shuffled)), baseline, "shuffled order");
    assert_eq!(
        render(ranked(&tied)),
        render(ranked(&tied_reversed)),
        "equal |corr| must be broken on the input UUID, not on caller order"
    );
    assert_eq!(
        render(ranked(&tied))[1..4],
        ["honest-0.6", "honest-tie-a", "honest-tie-b"],
        "ties rank by ascending UUID"
    );
}

#[test]
fn the_window_holds_only_the_top_honest_scores_when_nans_crowd_it() {
    const HONEST: [&str; 20] = [
        "h00", "h01", "h02", "h03", "h04", "h05", "h06", "h07", "h08", "h09", "h10", "h11", "h12",
        "h13", "h14", "h15", "h16", "h17", "h18", "h19",
    ];
    const NANS: [&str; 20] = [
        "n00", "n01", "n02", "n03", "n04", "n05", "n06", "n07", "n08", "n09", "n10", "n11", "n12",
        "n13", "n14", "n15", "n16", "n17", "n18", "n19",
    ];
    assert!(
        HONEST.len() > MAX_INPUTS_PER_TARGET,
        "the honest scores alone must overflow the window, or truncation is untested"
    );

    // NaNs first: the ordering that emptied the window before the fix.
    let mut scores: Vec<(&'static str, f32)> = NANS.iter().map(|&u| (u, f32::NAN)).collect();
    // h00 = 0.30 … h19 = 0.49: strictly increasing |corr|.
    scores.extend(
        HONEST
            .iter()
            .zip(0_u8..)
            .map(|(&u, i)| (u, 0.30 + f32::from(i) * 0.01)),
    );

    let window: Vec<String> = ranked(&scores).into_iter().map(|(u, _)| u).collect();
    let expected: Vec<String> = HONEST
        .iter()
        .rev()
        .take(MAX_INPUTS_PER_TARGET)
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        window, expected,
        "the window must hold exactly the top {MAX_INPUTS_PER_TARGET} honest scores"
    );
}

// =============================================================================
// End-to-end — `detect_fan_in_candidates` on finite `DiscoverRecord`s only
// =============================================================================

const SAMPLES: u32 = 60;
const HONEST_INPUTS: u32 = 15;
const POISON_INPUTS: u32 = 20;

#[derive(Clone, Copy)]
enum Poison {
    Absent,
    First,
    Last,
}

fn rec(uuid: &str, obs_index: u32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![error],
    }
}

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn target_error(j: u32) -> f32 {
    if j.is_multiple_of(3) { 1.0e10 } else { -5.0e9 }
}

fn honest_activation(i: u32, j: u32) -> f32 {
    let base = if (j + i).is_multiple_of(3) { 1.0 } else { -0.5 };
    let jitter = f32::from(u16::try_from(j).expect("SAMPLES fits in u16")) * 0.001;
    base + jitter
}

fn build(poison: Poison) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let honest: Vec<String> = (0..HONEST_INPUTS).map(|i| format!("honest-{i}")).collect();
    let poisoned: Vec<String> = match poison {
        Poison::Absent => Vec::new(),
        Poison::First | Poison::Last => (0..POISON_INPUTS).map(|i| format!("poison-{i}")).collect(),
    };
    let ordered: Vec<&String> = match poison {
        Poison::First => poisoned.iter().chain(honest.iter()).collect(),
        Poison::Absent | Poison::Last => honest.iter().chain(poisoned.iter()).collect(),
    };

    let mut neurons: Vec<NeuronJson> = ordered.iter().map(|u| neuron(u, "input")).collect();
    let input = neurons.len();
    neurons.push(neuron("output-1", "output"));

    let mut records = vec![(
        "output-1".to_string(),
        (0..SAMPLES)
            .map(|j| rec("output-1", j, 0.5, target_error(j)))
            .collect(),
    )];
    for (i, uuid) in (0_u32..).zip(&honest) {
        records.push((
            uuid.clone(),
            (0..SAMPLES)
                .map(|j| rec(uuid, j, honest_activation(i, j), 0.0))
                .collect(),
        ));
    }
    for uuid in &poisoned {
        records.push((
            uuid.clone(),
            (0..SAMPLES)
                .map(|j| {
                    let swing = if j.is_multiple_of(2) { 2.0e30 } else { -2.0e30 };
                    rec(uuid, j, swing, 0.0)
                })
                .collect(),
        ));
    }

    (
        CreatureJson {
            neurons,
            synapses: Vec::new(),
            input,
            output: 1,
        },
        records,
    )
}

/// Sorted `(target_uuid, input_uuids)` keys. The pair is sorted too: which
/// input ranks first hangs on ULP-level `|corr|` differences that follow the
/// `HashMap` iteration order, but the pair itself is the candidate's identity.
fn keys(poison: Poison) -> Vec<(String, Vec<String>)> {
    let (creature, records) = build(poison);

    // Issue #1799: the claim is "reachable from finite records".
    assert!(
        records
            .iter()
            .flat_map(|(_, recs)| recs)
            .all(|r| r.activation.is_finite() && r.errors.iter().all(|e| e.is_finite())),
        "every recorded value must be finite, or the trigger proves nothing"
    );

    let mut keys: Vec<(String, Vec<String>)> = detect_fan_in_candidates(&creature, &records, &None)
        .into_iter()
        .map(|c| {
            let mut inputs = c.input_uuids;
            inputs.sort();
            (c.target_uuid, inputs)
        })
        .collect();
    keys.sort();
    keys
}

#[test]
fn fan_in_candidates_agree_whichever_order_the_caller_lists_neurons_in() {
    let baseline = keys(Poison::Absent);
    assert!(
        !baseline.is_empty(),
        "the honest inputs must produce fan-in candidates, or the comparison is vacuous"
    );

    let poison_first = keys(Poison::First);
    let poison_last = keys(Poison::Last);

    assert_eq!(
        poison_first, poison_last,
        "the candidate set must not depend on the order the caller lists neurons in"
    );
    assert_eq!(
        poison_first, baseline,
        "NaN-correlation inputs must be dropped, leaving exactly the honest-only candidates"
    );
    assert!(
        poison_first
            .iter()
            .chain(&poison_last)
            .all(|(_, inputs)| inputs.iter().all(|u| !u.starts_with("poison-"))),
        "no candidate may name an input whose correlation was NaN"
    );
}
