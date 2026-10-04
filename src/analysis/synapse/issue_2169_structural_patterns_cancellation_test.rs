//! Issue #2169 regression tests for the structural-pattern detectors.
//!
//! `detect_noisy_vs_trusted` scans every pair of `input-*` synapses feeding a
//! target — O(n²) in the incoming-input count — and
//! `detect_collapsible_hidden_neurons` rebuilt the shared `a`/`b` activation
//! maps once per hidden neuron — O(hidden × records). Neither carried a
//! cancellation point, so `analysis_deadline_ms` and a host cancellation
//! request could not interrupt them once started. These tests pin the
//! deadline exits, the incoming-input ceiling, the work counts of both scans,
//! and that the detectors' output is unchanged below the ceiling.
//!
//! Every test is `#[serial]`: the scans honour the process-global cancellation
//! flag, which the `#[serial]` tests in `cancellation.rs` set and reset, so a
//! concurrent run would cut a scan short and fail the full-output assertions.

#![allow(clippy::cast_precision_loss)] // Synthetic fixtures map small u32 indices onto f32 activations.

use super::{
    MAX_INCOMING_INPUTS_FOR_NOISY_SCAN, detect_collapsible_hidden_neurons,
    detect_collapsible_hidden_neurons_observed, detect_noisy_vs_trusted,
    detect_noisy_vs_trusted_observed,
};
use crate::analysis::cache::RecordCache;
use crate::analysis::diagnostics::TargetMap;
use crate::analysis::scoring::weights::calculate_optimal_outgoing_weight;
use crate::ffi_types::{CreatureJson, NeuronJson, SynapseJson};
use crate::types::DiscoverRecord;
use crate::{
    AnalyzeSynapsesInput, CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson,
};
use serial_test::serial;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

const TARGET: &str = "target-0";

/// Observations per input in the noisy-vs-trusted fixtures.
const NOISY_OBS: u32 = 16;

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

fn map_loader(records: HashMap<String, Vec<DiscoverRecord>>) -> RecordCache {
    RecordCache::with_loader(
        "<in-memory>",
        Arc::new(move |_path: &str, uuid: &str| {
            records
                .get(uuid)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("unexpected uuid {uuid}"))
        }),
    )
}

/// Operation identities: kind plus endpoints, ignoring computed weights.
fn op_identities(candidate: &CoordinatedStructuralCandidateJson) -> Vec<String> {
    candidate
        .operations
        .iter()
        .map(|op| match op {
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
            } => format!("remove-synapse {from_neuron_uuid}->{to_neuron_uuid}"),
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                ..
            } => format!("add-synapse {from_neuron_uuid}->{to_neuron_uuid}"),
            CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } => {
                format!("remove-neuron {neuron_uuid}")
            }
            other => format!("{other:?}"),
        })
        .collect()
}

// -----------------------------------------------------------------------------
// Noisy vs trusted
// -----------------------------------------------------------------------------

/// `input-0` (trusted, low variance) and `input-1` (noisy, 100× the variance)
/// share a mean of zero and a weight of 0.5. The target's error is exactly
/// what moving the noisy weight onto the trusted input removes, so the pair is
/// beneficial.
struct NoisyFixture {
    synapses: Vec<SynapseJson>,
    cache: RecordCache,
    target_map: TargetMap,
}

fn build_noisy_fixture() -> NoisyFixture {
    build_padded_noisy_fixture(0)
}

/// The noisy/trusted fixture plus `padding` constant `input-pad-*` inputs.
/// Their weight (0.25) differs from the fixture pair's, so they only raise the
/// incoming-input count and never form a pair with `input-0`/`input-1`.
fn build_padded_noisy_fixture(padding: usize) -> NoisyFixture {
    let trusted = |k: u32| if k.is_multiple_of(2) { 0.1f32 } else { -0.1 };
    let noisy = |k: u32| {
        if (k / 2).is_multiple_of(2) {
            1.0f32
        } else {
            -1.0
        }
    };

    let rows = |uuid: &str, act: &dyn Fn(u32) -> f32| -> Vec<DiscoverRecord> {
        (0..NOISY_OBS)
            .map(|k| DiscoverRecord::new(k, uuid.to_string(), Some(act(k)), act(k), Vec::new()))
            .collect()
    };
    let mut records: HashMap<String, Vec<DiscoverRecord>> = HashMap::from([
        ("input-0".to_string(), rows("input-0", &trusted)),
        ("input-1".to_string(), rows("input-1", &noisy)),
    ]);
    let mut synapses = vec![
        synapse("input-0", TARGET, 0.5),
        synapse("input-1", TARGET, 0.5),
    ];
    for i in 0..padding {
        let uuid = format!("input-pad-{i}");
        records.insert(uuid.clone(), rows(&uuid, &|_| 0.0));
        synapses.push(synapse(&uuid, TARGET, 0.25));
    }
    let target_records: Vec<DiscoverRecord> = (0..NOISY_OBS)
        .map(|k| {
            let error = 0.5 * (trusted(k) - noisy(k));
            DiscoverRecord::new(k, TARGET.to_string(), Some(0.0), 0.0, vec![error])
        })
        .collect();

    NoisyFixture {
        synapses,
        cache: map_loader(records),
        target_map: TargetMap::from_records(&target_records),
    }
}

fn run_noisy(
    synapses: &[SynapseJson],
    cache: &RecordCache,
    target_map: &TargetMap,
    deadline: &Option<SystemTime>,
) -> Option<CoordinatedStructuralCandidateJson> {
    let refs: Vec<&SynapseJson> = synapses.iter().collect();
    detect_noisy_vs_trusted(TARGET, &refs, cache, target_map, &HashMap::new(), deadline)
}

fn expired_deadline() -> Option<SystemTime> {
    Some(SystemTime::now() - Duration::from_secs(60))
}

#[test]
#[serial]
fn noisy_vs_trusted_output_is_unchanged_without_deadline() {
    let fixture = build_noisy_fixture();
    let candidate = run_noisy(
        &fixture.synapses,
        &fixture.cache,
        &fixture.target_map,
        &None,
    )
    .expect("the noisy/trusted fixture must yield a candidate");

    assert_eq!(
        op_identities(&candidate),
        vec![
            format!("remove-synapse input-1->{TARGET}"),
            format!("remove-synapse input-0->{TARGET}"),
            format!("add-synapse input-0->{TARGET}"),
        ]
    );
    assert!(candidate.expected_creature_score_gain > 0.0);
}

#[test]
#[serial]
fn expired_deadline_stops_noisy_vs_trusted_scan() {
    let fixture = build_noisy_fixture();

    // Issue #1799: establish the positive precondition first — without a
    // deadline this fixture really does yield a candidate, so the assertion
    // below is observing cancellation and not an inert fixture.
    assert!(
        run_noisy(
            &fixture.synapses,
            &fixture.cache,
            &fixture.target_map,
            &None
        )
        .is_some(),
        "the noisy/trusted fixture must yield a candidate without a deadline"
    );

    let cancelled = run_noisy(
        &fixture.synapses,
        &fixture.cache,
        &fixture.target_map,
        &expired_deadline(),
    );
    assert!(
        cancelled.is_none(),
        "an expired deadline must stop the pairwise scan before any pair is scored"
    );
}

#[test]
#[serial]
fn noisy_vs_trusted_is_skipped_above_the_incoming_input_cap() {
    // Issue #1799: at exactly the cap the padded fixture still yields its
    // candidate, so the `None` one input past the cap is the cap at work and
    // not a fixture that never pairs.
    let at_cap = build_padded_noisy_fixture(MAX_INCOMING_INPUTS_FOR_NOISY_SCAN - 2);
    assert!(
        run_noisy(&at_cap.synapses, &at_cap.cache, &at_cap.target_map, &None).is_some(),
        "a fixture at exactly the cap must still be scanned and yield its candidate"
    );

    let above_cap = build_padded_noisy_fixture(MAX_INCOMING_INPUTS_FOR_NOISY_SCAN - 1);
    assert!(
        run_noisy(
            &above_cap.synapses,
            &above_cap.cache,
            &above_cap.target_map,
            &None
        )
        .is_none(),
        "a target with more than MAX_INCOMING_INPUTS_FOR_NOISY_SCAN inputs must skip the scan"
    );
}

/// `count` inputs with identical records: every pair passes the weight and
/// mean filters and fails the variance-ratio filter, the cheapest path through
/// the pairwise scan — so any growth counted is the scan's shape alone.
fn build_uniform_inputs(count: usize) -> (Vec<SynapseJson>, RecordCache, TargetMap) {
    const OBS: u32 = 4;
    let mut records: HashMap<String, Vec<DiscoverRecord>> = HashMap::with_capacity(count);
    let mut synapses = Vec::with_capacity(count);
    for i in 0..count {
        let uuid = format!("input-{i}");
        let rows = (0..OBS)
            .map(|k| {
                let act = if k % 2 == 0 { 1.0 } else { -1.0 };
                DiscoverRecord::new(k, uuid.clone(), Some(act), act, Vec::new())
            })
            .collect();
        synapses.push(synapse(&uuid, TARGET, 0.5));
        records.insert(uuid, rows);
    }
    let target_records: Vec<DiscoverRecord> = (0..OBS)
        .map(|k| DiscoverRecord::new(k, TARGET.to_string(), Some(0.0), 0.0, vec![0.1]))
        .collect();
    (
        synapses,
        map_loader(records),
        TargetMap::from_records(&target_records),
    )
}

/// Count the pairs of incoming inputs one noisy-vs-trusted scan considers
/// (Issue #2320).
///
/// The count is the scan's unit of work, so asserting on it is deterministic
/// where a wall-clock reading flakes under a loaded parallel test run.
fn count_noisy_pairs(
    synapses: &[SynapseJson],
    cache: &RecordCache,
    target_map: &TargetMap,
) -> usize {
    let refs: Vec<&SynapseJson> = synapses.iter().collect();
    let mut pairs = 0usize;
    let candidate = detect_noisy_vs_trusted_observed(
        TARGET,
        &refs,
        cache,
        target_map,
        &HashMap::new(),
        &None,
        || pairs += 1,
    );
    assert!(
        candidate.is_none(),
        "identical inputs never form a noisy/trusted pair"
    );
    pairs
}

#[test]
#[serial]
fn noisy_vs_trusted_cost_does_not_grow_quadratically() {
    // Issue #1799: positive precondition — below the ceiling every pair of a
    // uniform fixture is considered, so a zero count further down means the
    // scan was skipped and not that the counter is inert.
    const SCANNED: usize = 16;
    let (synapses, cache, target_map) = build_uniform_inputs(SCANNED);
    assert_eq!(
        count_noisy_pairs(&synapses, &cache, &target_map),
        SCANNED * (SCANNED - 1) / 2,
        "a uniform fixture below the ceiling must compare every pair once"
    );

    // Issue #2320: count the scan's work instead of timing it. Above the
    // ceiling the quadratic scan must never be entered, so the pair count
    // stays at zero however far the input count grows — a load spike on a
    // parallel test run cannot move it.
    let small = MAX_INCOMING_INPUTS_FOR_NOISY_SCAN + 1;
    let large = small * 2;
    let (synapses, cache, target_map) = build_uniform_inputs(large);

    let small_pairs = count_noisy_pairs(&synapses[..small], &cache, &target_map);
    let large_pairs = count_noisy_pairs(&synapses, &cache, &target_map);
    assert_eq!(
        (small_pairs, large_pairs),
        (0, 0),
        "noisy-vs-trusted scan entered above the {MAX_INCOMING_INPUTS_FOR_NOISY_SCAN}-input ceiling: \
         {small_pairs} pairs at {small} inputs, {large_pairs} at {large}"
    );
}

// -----------------------------------------------------------------------------
// Collapse 1-in/1-out hidden neurons
// -----------------------------------------------------------------------------

/// Computed bypass weight for the collapse fixtures — above the 0.01 default
/// floor and above every value the in-file tests set the floor to.
const BYPASS_WEIGHT: f32 = 0.05;

/// Weight of each existing `hidden-i → output-0` synapse.
const H_TO_B_WEIGHT: f32 = 0.5;

/// `hidden` passthrough neurons, each wired `input-0 → hidden-i → output-0`.
/// `input-0` and `output-0` carry `shared_obs` records; each hidden neuron
/// carries the first `hidden_obs` of them, with `h_act = a_act`, so every
/// chain collapses to a bypass weight of [`BYPASS_WEIGHT`].
fn build_collapse_fixture(
    hidden: usize,
    shared_obs: u32,
    hidden_obs: u32,
) -> (AnalyzeSynapsesInput, RecordCache) {
    let mut neurons = Vec::with_capacity(hidden + 1);
    let mut synapses = Vec::with_capacity(hidden * 2);
    for i in 0..hidden {
        let uuid = format!("hidden-{i}");
        neurons.push(NeuronJson {
            uuid: uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
        synapses.push(synapse("input-0", &uuid, 1.0));
        synapses.push(synapse(&uuid, "output-0", H_TO_B_WEIGHT));
    }
    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    let denom = shared_obs.saturating_sub(1).max(1) as f32;
    let a_act = |k: u32| -1.0 + 2.0 * (k as f32) / denom;

    let input0: Vec<DiscoverRecord> = (0..shared_obs)
        .map(|k| {
            DiscoverRecord::new(
                k,
                "input-0".to_string(),
                Some(a_act(k)),
                a_act(k),
                Vec::new(),
            )
        })
        .collect();
    let output0: Vec<DiscoverRecord> = (0..shared_obs)
        .map(|k| {
            let act = H_TO_B_WEIGHT * a_act(k);
            let error = (BYPASS_WEIGHT - H_TO_B_WEIGHT) * a_act(k);
            DiscoverRecord::new(k, "output-0".to_string(), Some(act), act, vec![error])
        })
        .collect();

    let mut records: HashMap<String, Vec<DiscoverRecord>> = HashMap::with_capacity(hidden + 2);
    for neuron in neurons.iter().filter(|n| n.neuron_type == "hidden") {
        let rows = (0..hidden_obs)
            .map(|k| {
                DiscoverRecord::new(k, neuron.uuid.clone(), Some(a_act(k)), a_act(k), Vec::new())
            })
            .collect();
        records.insert(neuron.uuid.clone(), rows);
    }
    records.insert("input-0".to_string(), input0);
    records.insert("output-0".to_string(), output0);

    let input = AnalyzeSynapsesInput {
        parquet_file: "<in-memory>".to_string(),
        creature: CreatureJson {
            input: 1,
            output: 1,
            neurons,
            synapses,
        },
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };
    (input, map_loader(records))
}

/// Hidden neuron each collapse candidate removes, in emission order.
fn collapsed_neurons(candidates: &[CoordinatedStructuralCandidateJson]) -> Vec<String> {
    candidates
        .iter()
        .flat_map(|c| &c.operations)
        .filter_map(|op| match op {
            CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } => Some(neuron_uuid.clone()),
            _ => None,
        })
        .collect()
}

#[test]
#[serial]
fn collapse_output_is_unchanged_without_deadline() {
    let (input, cache) = build_collapse_fixture(3, 16, 16);
    let outcome = detect_collapsible_hidden_neurons(&input, &cache, &None);

    assert_eq!(outcome.bypass_weight_below_floor_drops, 0);
    assert_eq!(
        collapsed_neurons(&outcome.candidates),
        vec!["hidden-0", "hidden-1", "hidden-2"]
    );
    for (i, candidate) in outcome.candidates.iter().enumerate() {
        assert_eq!(
            op_identities(candidate),
            vec![
                format!("remove-synapse input-0->hidden-{i}"),
                format!("remove-synapse hidden-{i}->output-0"),
                format!("remove-neuron hidden-{i}"),
                "add-synapse input-0->output-0".to_string(),
            ]
        );
        let Some(CoordinatedStructuralOpJson::AddSynapse { weight, .. }) =
            candidate.operations.last()
        else {
            panic!("the last collapse op must add the bypass synapse");
        };
        // The least-squares fit is BYPASS_WEIGHT; the production clamp then
        // applies, so derive the expectation through the same function.
        let expected = calculate_optimal_outgoing_weight(BYPASS_WEIGHT, 1.0, 1.0)
            .expect("the fixture's bypass fit is non-degenerate");
        assert!(
            (weight - expected).abs() < 1e-6,
            "bypass weight {weight} drifted from {expected}"
        );
    }
}

#[test]
#[serial]
fn expired_deadline_stops_collapse_scan() {
    let (input, cache) = build_collapse_fixture(3, 16, 16);

    // Issue #1799: the positive precondition — without a deadline the fixture
    // really does yield collapse candidates.
    let baseline = detect_collapsible_hidden_neurons(&input, &cache, &None);
    assert!(
        !baseline.candidates.is_empty(),
        "the collapse fixture must yield candidates without a deadline"
    );

    let cancelled = detect_collapsible_hidden_neurons(&input, &cache, &expired_deadline());
    assert!(
        cancelled.candidates.is_empty(),
        "an expired deadline must stop the scan before any neuron is examined"
    );
    assert_eq!(cancelled.bypass_weight_below_floor_drops, 0);
}

/// Count the shared `a`/`b` activation maps one collapse scan actually builds
/// (Issue #2320).
///
/// The count is the scan's unit of work — what the memoisation in
/// `detect_collapsible_hidden_neurons` is meant to bound — so asserting on it
/// is deterministic where a wall-clock reading flakes under a loaded parallel
/// test run.
fn count_shared_map_builds(
    input: &AnalyzeSynapsesInput,
    cache: &RecordCache,
    expected_candidates: usize,
) -> usize {
    let mut builds = 0usize;
    let outcome = detect_collapsible_hidden_neurons_observed(input, cache, &None, || builds += 1);
    assert_eq!(
        outcome.candidates.len(),
        expected_candidates,
        "every passthrough chain must collapse"
    );
    builds
}

#[test]
#[serial]
fn collapse_cost_does_not_grow_with_hidden_times_records() {
    // Hidden neurons share one `a` (input-0) and one `b` (output-0), so
    // exactly two shared maps are built however many hidden neurons and
    // records there are (Issue #2320) — rebuilding per hidden neuron would
    // give 2 × hidden instead (Issue #1799: the positive precondition this
    // test's `every passthrough chain must collapse` assertion protects).
    const HIDDEN_OBS: u32 = 12;
    let (small_hidden, small_obs) = (8usize, 64u32);
    let (large_hidden, large_obs) = (small_hidden * 2, small_obs * 2);

    let (small_input, small_cache) = build_collapse_fixture(small_hidden, small_obs, HIDDEN_OBS);
    let (large_input, large_cache) = build_collapse_fixture(large_hidden, large_obs, HIDDEN_OBS);

    let small_builds = count_shared_map_builds(&small_input, &small_cache, small_hidden);
    let large_builds = count_shared_map_builds(&large_input, &large_cache, large_hidden);

    assert_eq!(
        (small_builds, large_builds),
        (2, 2),
        "shared map builds must stay constant regardless of hidden count or record count: \
         {small_builds} builds at {small_hidden}×{small_obs} against {large_builds} at {large_hidden}×{large_obs}"
    );
}
