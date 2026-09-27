//! Issue #2169 regression tests for the structural-pattern detectors.
//!
//! `detect_noisy_vs_trusted` scans every pair of `input-*` synapses feeding a
//! target — O(n²) in the incoming-input count — and
//! `detect_collapsible_hidden_neurons` rebuilt the shared `a`/`b` activation
//! maps once per hidden neuron — O(hidden × records). Neither carried a
//! cancellation point, so `analysis_deadline_ms` and a host cancellation
//! request could not interrupt them once started. These tests pin the
//! deadline exits, the incoming-input ceiling, the linear cost of both scans,
//! and that the detectors' output is unchanged below the ceiling.
//!
//! Every test is `#[serial]`: the scans honour the process-global cancellation
//! flag, which the `#[serial]` tests in `cancellation.rs` set and reset, so a
//! concurrent run would cut a scan short and fail the full-output assertions.

#![allow(clippy::cast_precision_loss)] // Synthetic fixtures map small u32 indices onto f32 activations.

use super::{
    MAX_INCOMING_INPUTS_FOR_NOISY_SCAN, detect_collapsible_hidden_neurons, detect_noisy_vs_trusted,
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
use std::time::{Duration, Instant, SystemTime};

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
    let records: HashMap<String, Vec<DiscoverRecord>> = HashMap::from([
        ("input-0".to_string(), rows("input-0", &trusted)),
        ("input-1".to_string(), rows("input-1", &noisy)),
    ]);
    let target_records: Vec<DiscoverRecord> = (0..NOISY_OBS)
        .map(|k| {
            let error = 0.5 * (trusted(k) - noisy(k));
            DiscoverRecord::new(k, TARGET.to_string(), Some(0.0), 0.0, vec![error])
        })
        .collect();

    NoisyFixture {
        synapses: vec![
            synapse("input-0", TARGET, 0.5),
            synapse("input-1", TARGET, 0.5),
        ],
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

/// `count` inputs with identical records: every pair passes the weight and
/// mean filters and fails the variance-ratio filter, the cheapest path through
/// the pairwise scan — so any growth measured is the scan's shape alone.
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

fn min_noisy_time(
    synapses: &[SynapseJson],
    cache: &RecordCache,
    target_map: &TargetMap,
) -> Duration {
    const RUNS: usize = 5;

    (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            let candidate = run_noisy(synapses, cache, target_map, &None);
            let elapsed = start.elapsed();
            assert!(
                candidate.is_none(),
                "identical inputs never form a noisy/trusted pair"
            );
            elapsed
        })
        .min()
        .expect("RUNS is non-zero")
}

#[test]
#[serial]
fn noisy_vs_trusted_cost_does_not_grow_quadratically() {
    let small = MAX_INCOMING_INPUTS_FOR_NOISY_SCAN + 1;
    let large = small * 2;

    // Fixtures are built once, outside the timed region, and the smaller run
    // reuses a prefix of the larger so both time exactly the same kind of work.
    let (synapses, cache, target_map) = build_uniform_inputs(large);

    let t_small =
        min_noisy_time(&synapses[..small], &cache, &target_map).max(Duration::from_nanos(1));
    let t_large = min_noisy_time(&synapses, &cache, &target_map);

    // Two readings of the same work, never a reading against a wall-clock
    // constant: doubling the input may double the cost (linear) but must not
    // quadruple it (quadratic). The bound sits midway between the two.
    assert!(
        t_large <= t_small * 3,
        "noisy-vs-trusted cost grew faster than linearly: {t_small:?} at {small} inputs against {t_large:?} at {large} inputs"
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

fn min_collapse_time(
    input: &AnalyzeSynapsesInput,
    cache: &RecordCache,
    expected: usize,
) -> Duration {
    const RUNS: usize = 5;

    (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            let outcome = detect_collapsible_hidden_neurons(input, cache, &None);
            let elapsed = start.elapsed();
            assert_eq!(
                outcome.candidates.len(),
                expected,
                "every passthrough chain must collapse"
            );
            elapsed
        })
        .min()
        .expect("RUNS is non-zero")
}

#[test]
#[serial]
fn collapse_cost_does_not_grow_with_hidden_times_records() {
    // Hidden neurons share one `a` and one `b`. Doubling both the hidden count
    // and the shared record count quadruples the work when the shared maps are
    // rebuilt per hidden neuron, but only doubles it once they are memoised.
    const HIDDEN_OBS: u32 = 12;
    let (small_hidden, small_obs) = (256usize, 8_192u32);
    let (large_hidden, large_obs) = (small_hidden * 2, small_obs * 2);

    let (small_input, small_cache) = build_collapse_fixture(small_hidden, small_obs, HIDDEN_OBS);
    let (large_input, large_cache) = build_collapse_fixture(large_hidden, large_obs, HIDDEN_OBS);

    let t_small =
        min_collapse_time(&small_input, &small_cache, small_hidden).max(Duration::from_nanos(1));
    let t_large = min_collapse_time(&large_input, &large_cache, large_hidden);

    // Two readings of the same work, never a reading against a wall-clock
    // constant: doubling both dimensions may double the cost (linear in their
    // sum) but must not quadruple it (their product). The bound sits midway.
    assert!(
        t_large <= t_small * 3,
        "collapse cost grew with hidden × records: {t_small:?} at {small_hidden}×{small_obs} against {t_large:?} at {large_hidden}×{large_obs}"
    );
}
