//! Unit tests for the focus module's internal components.
//!
//! These tests exercise internal record provider behaviour that cannot be
//! tested through the public API alone.

use super::ranking::*;
use crate::ffi_types::{CreatureJson, DiscoveryError, DiscoveryErrorKind, NeuronJson, SynapseJson};
use crate::types::DiscoverRecord;
use anyhow::anyhow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Issue #1375 — wall-clock budget on focus ranking with graceful fallback.
// ---------------------------------------------------------------------------

use crate::ffi_types::{DiscoveryErrorKind, classify_anyhow_error};
use crate::{CreatureJson, NeuronJson, SynapseJson};
use std::time::{Duration, Instant};

/// A [`RecordProvider`] that sleeps on every `get`, simulating a pathologically
/// slow record loader (the #1373 incident: focus selection ran for 1h 11m).
struct SleepyProvider {
    per_call: Duration,
    neurons: usize,
    calls: AtomicUsize,
}

impl SleepyProvider {
    fn new(per_call: Duration, neurons: usize) -> Self {
        Self {
            per_call,
            neurons,
            calls: AtomicUsize::new(0),
        }
    }
}

impl RecordProvider for SleepyProvider {
    fn get(&self, neuron_uuid: &str) -> anyhow::Result<Option<Arc<Vec<DiscoverRecord>>>> {
        std::thread::sleep(self.per_call);
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Some(Arc::new(vec![
            DiscoverRecord {
                obs_index: 0,
                neuron_uuid: neuron_uuid.to_string(),
                value: None,
                activation: 0.5,
                errors: vec![0.1],
            },
            DiscoverRecord {
                obs_index: 1,
                neuron_uuid: neuron_uuid.to_string(),
                value: None,
                activation: 0.2,
                errors: vec![0.2],
            },
        ])))
    }

    fn len(&self) -> usize {
        self.neurons
    }
}

/// Build a creature with `hidden` selectable hidden neurons plus one output.
fn make_creature(hidden: usize) -> CreatureJson {
    let mut neurons: Vec<NeuronJson> = (0..hidden)
        .map(|i| NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        })
        .collect();
    neurons.push(NeuronJson {
        uuid: "out".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    let mut synapses: Vec<SynapseJson> = (0..hidden)
        .map(|i| SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: format!("hidden-{i}"),
            weight: 1.0,
            synapse_type: None,
        })
        .collect();
    for i in 0..hidden {
        synapses.push(SynapseJson {
            from_uuid: format!("hidden-{i}"),
            to_uuid: "out".to_string(),
            weight: 1.0,
            synapse_type: None,
        });
    }

    CreatureJson {
        input: 1,
        output: 1,
        neurons,
        synapses,
    }
}

/// Issue #1375: a focus-ranking run that exceeds its wall-clock budget must
/// abort within `budget + grace` with a structured `Timeout` error, instead of
/// grinding through every slow record load. Regression guard for the 1h 11m
/// unbounded focus selection in the #1373 incident.
#[test]
fn focus_ranking_aborts_when_budget_exceeded() {
    const HIDDEN: usize = 80;
    let per_call = Duration::from_millis(25);
    let budget_ms = 50;

    // Unbounded cost would be ~(HIDDEN + outputs) × per_call. The budget plus
    // the fixed 1s grace must cut this off well before completion.
    let unbounded = per_call * u32::try_from(HIDDEN + 1).expect("neuron count fits u32");

    let creature = make_creature(HIDDEN);
    let provider = Arc::new(SleepyProvider::new(per_call, HIDDEN));

    let start = Instant::now();
    let result = rank_with_provider_for_tests(&creature, provider, budget_ms);
    let elapsed = start.elapsed();

    let err = result.expect_err("ranking should abort once the budget is exceeded");
    assert_eq!(
        classify_anyhow_error(&err),
        DiscoveryErrorKind::Timeout,
        "budget abort must be classified as a retryable Timeout: {err:#}"
    );

    // Aborted within budget + grace (+ one in-flight sleep of slop), and clearly
    // faster than running the full slow loop to completion.
    let max_allowed = Duration::from_millis(budget_ms + 1000) + per_call * 3;
    assert!(
        elapsed < max_allowed,
        "expected abort within {max_allowed:?}, took {elapsed:?}"
    );
    assert!(
        elapsed < unbounded,
        "abort ({elapsed:?}) must be faster than the unbounded run ({unbounded:?})"
    );
}

/// Issue #1375: a generous budget must not abort a legitimate (merely slow)
/// run — the default fast path stays correct and complete.
#[test]
fn focus_ranking_completes_within_generous_budget() {
    const HIDDEN: usize = 3;
    let creature = make_creature(HIDDEN);
    let provider = Arc::new(SleepyProvider::new(Duration::from_millis(1), HIDDEN));

    let stats = rank_with_provider_for_tests(&creature, provider, 60_000)
        .expect("a fast run within a generous budget must not abort");

    // Output neurons are selectable too (only inputs/constants are excluded),
    // so the ranked set is the hidden neurons plus the single output.
    assert_eq!(stats.processed_neurons, HIDDEN + 1);
    assert_eq!(stats.neurons.len(), HIDDEN + 1);
}

#[test]
fn lazy_provider_defers_loading_and_bounds_cache() -> anyhow::Result<()> {
    let loads = Arc::new(AtomicUsize::new(0));
    let provider = LazyRecordProvider::with_loader_for_tests("unused.parquet", 2, {
        let loads = Arc::clone(&loads);
        Arc::new(move |_file, neuron_uuid| {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: neuron_uuid.to_string(),
                value: None,
                activation: 0.0,
                errors: vec![0.0],
            }])
        })
    });

    // No eager loads during initialisation
    assert_eq!(0, loads.load(Ordering::SeqCst));

    // First load hits the loader, subsequent load for same neuron is cached
    provider.get("a")?.expect("records should be present");
    assert_eq!(1, loads.load(Ordering::SeqCst));
    provider.get("a")?.expect("records should be cached");
    assert_eq!(1, loads.load(Ordering::SeqCst));

    // Loading a second neuron increments once and cache remains bounded
    provider.get("b")?.expect("records should be present");
    assert_eq!(2, loads.load(Ordering::SeqCst));
    assert!(provider.len() <= 2);
    Ok(())
}

/// Issue #1374: A cache sized to the ranking working set must materialise each
/// neuron **at most once** across the ranking pipeline's multiple passes.
///
/// The pathological lazy-mode cost was `passes × neurons × full-file-decode`
/// because the default 8-entry cache thrashed when the working set was larger.
/// With the cache sized to the working set, the per-neuron loader is invoked
/// `O(neurons)` times, not `O(passes × neurons)`.
#[test]
fn sized_cache_loads_each_neuron_at_most_once_across_passes() -> anyhow::Result<()> {
    const NEURONS: usize = 20;
    const PASSES: usize = 5;

    let loads = Arc::new(AtomicUsize::new(0));
    let provider = LazyRecordProvider::with_loader_for_tests("unused.parquet", NEURONS, {
        let loads = Arc::clone(&loads);
        Arc::new(move |_file, neuron_uuid| {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: neuron_uuid.to_string(),
                value: None,
                activation: 0.0,
                errors: vec![0.0],
            }])
        })
    });

    // Simulate the ranking pipeline's repeated sweeps over every neuron.
    for _pass in 0..PASSES {
        for n in 0..NEURONS {
            provider
                .get(&format!("n{n}"))?
                .expect("records should be present");
        }
    }

    assert_eq!(
        NEURONS,
        loads.load(Ordering::SeqCst),
        "each neuron must be loaded exactly once across {PASSES} passes when the \
         cache is sized to the working set (O(neurons), not O(passes × neurons))"
    );
    Ok(())
}

/// Issue #1374: Seeding the cache from a single grouped parquet pass must let
/// every seeded neuron be served from cache, so the per-neuron full-file loader
/// is never invoked during ranking.
#[test]
fn seeded_cache_serves_neurons_without_invoking_loader() -> anyhow::Result<()> {
    const NEURONS: usize = 12;

    let loads = Arc::new(AtomicUsize::new(0));
    let provider = LazyRecordProvider::with_loader_for_tests("unused.parquet", NEURONS, {
        let loads = Arc::clone(&loads);
        Arc::new(move |_file, neuron_uuid| {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: neuron_uuid.to_string(),
                value: None,
                activation: 0.0,
                errors: vec![0.0],
            }])
        })
    });

    // Warm the cache with one grouped batch (records intentionally unsorted to
    // verify seed sorts by obs_index).
    let mut grouped = std::collections::HashMap::new();
    for n in 0..NEURONS {
        let uuid = format!("n{n}");
        grouped.insert(
            uuid.clone(),
            vec![
                DiscoverRecord {
                    obs_index: 2,
                    neuron_uuid: uuid.clone(),
                    value: None,
                    activation: 0.0,
                    errors: vec![0.0],
                },
                DiscoverRecord {
                    obs_index: 0,
                    neuron_uuid: uuid.clone(),
                    value: None,
                    activation: 0.0,
                    errors: vec![0.0],
                },
            ],
        );
    }
    provider.seed(grouped)?;

    // Multiple passes over the seeded neurons must all be cache hits.
    for _pass in 0..5 {
        for n in 0..NEURONS {
            let records = provider
                .get(&format!("n{n}"))?
                .expect("seeded records should be present");
            assert_eq!(records[0].obs_index, 0, "seed must sort by obs_index");
            assert_eq!(records[1].obs_index, 2);
        }
    }

    assert_eq!(
        0,
        loads.load(Ordering::SeqCst),
        "seeded neurons must be served from cache without any per-neuron loads"
    );
    Ok(())
}

// ===========================================================================
// Issue #1375: focus-ranking wall-clock budget
// ===========================================================================

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn record_for(uuid: &str) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: 0,
        neuron_uuid: uuid.to_string(),
        value: None,
        activation: 0.5,
        errors: vec![0.1],
    }
}

/// Build a small forward-only creature with `hidden` hidden neurons feeding a
/// single output, fed by two inputs.
fn small_creature(hidden: usize) -> CreatureJson {
    let mut neurons = vec![neuron("input-0", "input"), neuron("input-1", "input")];
    let mut synapses = Vec::new();
    for h in 0..hidden {
        let uuid = format!("h{h}");
        neurons.push(neuron(&uuid, "hidden"));
        synapses.push(SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: uuid.clone(),
            weight: 0.5,
            synapse_type: None,
        });
        synapses.push(SynapseJson {
            from_uuid: uuid,
            to_uuid: "out-0".to_string(),
            weight: 0.5,
            synapse_type: None,
        });
    }
    neurons.push(neuron("out-0", "output"));
    CreatureJson {
        neurons,
        synapses,
        input: 2,
        output: 1,
    }
}

/// Issue #1375: a focus-ranking run whose record loading is pathologically slow
/// must abort within `budget + grace` with a structured timeout error, rather
/// than running unbounded (the GRQ-13 incident ran for 1h 11m).
#[test]
fn ranking_aborts_when_wall_clock_budget_exceeded() {
    const HIDDEN: usize = 40;
    const BUDGET_MS: u64 = 200;
    const PER_LOAD: Duration = Duration::from_millis(40);
    // Grace generous enough to absorb one in-flight slow load plus scheduling.
    const GRACE_MS: u64 = 3_000;

    let creature = small_creature(HIDDEN);

    // Deliberately slow per-neuron loader simulating a thrashing parquet read.
    let provider: Arc<dyn RecordProvider> = Arc::new(LazyRecordProvider::with_loader_for_tests(
        "slow.parquet",
        HIDDEN + 8,
        Arc::new(move |_file, neuron_uuid| {
            std::thread::sleep(PER_LOAD);
            Ok(vec![record_for(neuron_uuid)])
        }),
    ));

    let start = Instant::now();
    let result = rank_focus_neurons_with_provider_for_tests(
        &creature,
        provider,
        None,
        None,
        Some(BUDGET_MS),
    );
    let elapsed = start.elapsed();

    let err = result.expect_err("a budget-exceeding run must abort with an error");
    let typed = err
        .downcast_ref::<DiscoveryError>()
        .expect("abort must be a typed DiscoveryError so the FFI classifies it");
    assert_eq!(
        typed.error_kind(),
        DiscoveryErrorKind::Timeout,
        "budget abort must classify as a (retryable) timeout for graceful fallback"
    );

    assert!(
        elapsed < Duration::from_millis(BUDGET_MS + GRACE_MS),
        "ranking must abort within budget + grace; took {elapsed:?} \
         (budget {BUDGET_MS}ms + grace {GRACE_MS}ms)"
    );
}

/// Issue #1375: a fast run within budget must complete normally and return
/// ranked neurons — the budget must not abort healthy runs.
#[test]
fn ranking_completes_within_budget_for_fast_loader() -> anyhow::Result<()> {
    const HIDDEN: usize = 6;

    let creature = small_creature(HIDDEN);

    let provider: Arc<dyn RecordProvider> = Arc::new(LazyRecordProvider::with_loader_for_tests(
        "fast.parquet",
        HIDDEN + 8,
        Arc::new(|_file, neuron_uuid| Ok(vec![record_for(neuron_uuid)])),
    ));

    let stats =
        rank_focus_neurons_with_provider_for_tests(&creature, provider, None, None, Some(60_000))?;

    // Selectable = hidden + output (inputs excluded).
    assert_eq!(stats.total_neurons, HIDDEN + 1);
    assert_eq!(stats.neurons.len(), HIDDEN + 1);
    Ok(())
}

/// Issue #1375: a disabled budget (`None`) must never abort, preserving the
/// previous unbounded behaviour for callers that explicitly opt out.
#[test]
fn ranking_with_disabled_budget_does_not_abort() -> anyhow::Result<()> {
    const HIDDEN: usize = 4;

    let creature = small_creature(HIDDEN);
    let provider: Arc<dyn RecordProvider> = Arc::new(LazyRecordProvider::with_loader_for_tests(
        "fast.parquet",
        HIDDEN + 8,
        Arc::new(|_file, neuron_uuid| Ok(vec![record_for(neuron_uuid)])),
    ));

    let stats = rank_focus_neurons_with_provider_for_tests(&creature, provider, None, None, None)?;
    assert_eq!(stats.neurons.len(), HIDDEN + 1);
    Ok(())
}

#[test]
fn lazy_provider_returns_loader_errors_with_context() {
    let provider = LazyRecordProvider::with_loader_for_tests("failing.parquet", 2, {
        Arc::new(|file, neuron_uuid| {
            Err(anyhow!(
                "Simulated parquet read failure for {neuron_uuid} in {file}"
            ))
        })
    });

    let err = provider
        .get("hidden-1")
        .expect_err("loader error should surface");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Simulated parquet read failure"),
        "expected loader error, got: {msg}"
    );
    assert!(
        msg.contains("hidden-1"),
        "neuron context should be present: {msg}"
    );
    assert!(
        msg.contains("failing.parquet"),
        "file context should be present: {msg}"
    );
}
