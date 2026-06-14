//! Issue #1375: Wall-clock budget on focus ranking with graceful fallback.
//!
//! Focus ranking previously had no wall-clock bound: in the #1373 incident it
//! ran for over an hour and blew the entire 3h discovery budget. These tests
//! exercise the new budget:
//!
//! - A run backed by a deliberately slow record loader aborts inside
//!   `budget + grace` with a structured [`DiscoveryError::Timeout`], so the
//!   caller routes it into the existing local-ranking fallback (instead of
//!   running unbounded).
//! - A generous budget (and a disabled budget) lets a fast run complete
//!   normally with no abort — the default fast-path behaviour is unchanged.
//! - The `NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS` env parser clamps and
//!   disables as documented.

#![allow(clippy::cast_precision_loss)] // Numeric casts in test setup.

use std::sync::Arc;
use std::time::{Duration, Instant};

use neat_ai_discovery::DiscoveryError;
use neat_ai_discovery::config::{
    DEFAULT_FOCUS_RANKING_BUDGET_MS, MAX_FOCUS_RANKING_BUDGET_MS, MIN_FOCUS_RANKING_BUDGET_MS,
    focus_ranking_budget_ms,
};
use neat_ai_discovery::focus::{RecordProvider, rank_focus_neurons_with_provider_and_budget};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;

// ---------------------------------------------------------------------------
// Test record provider with an optional per-`get` delay.
// ---------------------------------------------------------------------------

/// Record provider that returns a fixed record set for every requested neuron,
/// optionally sleeping on each `get` to emulate a pathologically slow loader
/// (the failure mode the #1373 incident hit under lazy loading).
struct SlowProvider {
    records: Arc<Vec<DiscoverRecord>>,
    delay: Duration,
    neuron_count: usize,
}

impl SlowProvider {
    fn new(delay: Duration, neuron_count: usize) -> Self {
        let records = (0..8u32)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "shared".to_string(),
                value: Some(0.5),
                activation: 0.3,
                errors: vec![0.1],
            })
            .collect();
        Self {
            records: Arc::new(records),
            delay,
            neuron_count,
        }
    }
}

impl RecordProvider for SlowProvider {
    fn get(&self, _neuron_uuid: &str) -> anyhow::Result<Option<Arc<Vec<DiscoverRecord>>>> {
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        Ok(Some(Arc::clone(&self.records)))
    }

    fn len(&self) -> usize {
        self.neuron_count
    }
}

/// Build a creature with `hidden_count` hidden neurons feeding a single output.
fn make_creature(hidden_count: usize) -> CreatureJson {
    let mut neurons = vec![NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "LOGISTIC".to_string(),
        bias: 0.0,
    }];
    let mut synapses = Vec::new();
    for i in 0..hidden_count {
        let uuid = format!("hidden-{i}");
        synapses.push(SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: uuid.clone(),
            weight: 1.0,
            synapse_type: None,
        });
        synapses.push(SynapseJson {
            from_uuid: uuid.clone(),
            to_uuid: "output-0".to_string(),
            weight: 1.0,
            synapse_type: None,
        });
        neurons.push(NeuronJson {
            uuid,
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        });
    }
    CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    }
}

// ---------------------------------------------------------------------------
// Core acceptance criterion: a slow run aborts inside budget + grace.
// ---------------------------------------------------------------------------

#[test]
fn slow_loader_aborts_within_budget_plus_grace() {
    // 40 selectable neurons, each loading taking 50ms => unbounded ranking
    // would spend > 2s just verifying records. With a 200ms budget the run must
    // abort an order of magnitude sooner.
    let hidden_count = 40;
    let per_get = Duration::from_millis(50);
    let budget_ms = 200u64;
    let unbounded_estimate = per_get * u32::try_from(hidden_count + 1).expect("fits in u32");

    let provider: Arc<dyn RecordProvider> = Arc::new(SlowProvider::new(per_get, hidden_count + 1));
    let creature = make_creature(hidden_count);

    let start = Instant::now();
    let result = rank_focus_neurons_with_provider_and_budget(
        provider,
        &creature,
        None,
        None,
        None,
        None,
        Some(budget_ms),
    );
    let elapsed = start.elapsed();

    // Must be a structured timeout error so the FFI layer marks it retryable and
    // the controller falls back to local ranking.
    let err = result.expect_err("slow run must abort once the budget is exceeded");
    let typed = err.downcast_ref::<DiscoveryError>();
    assert!(
        matches!(typed, Some(DiscoveryError::Timeout { .. })),
        "abort must surface DiscoveryError::Timeout, got: {err:?}"
    );
    let kind = typed.unwrap().error_kind();
    assert!(
        kind.is_retryable(),
        "timeout abort must classify as retryable so the controller retries via fallback"
    );

    // Returned inside budget + a generous grace, and far below the unbounded run.
    let grace = Duration::from_millis(1_000);
    assert!(
        elapsed <= Duration::from_millis(budget_ms) + grace,
        "ranking took {elapsed:?}, expected <= budget ({budget_ms}ms) + grace (1000ms)"
    );
    assert!(
        elapsed < unbounded_estimate,
        "ranking took {elapsed:?}, must be well below the unbounded estimate {unbounded_estimate:?}"
    );
}

// ---------------------------------------------------------------------------
// A generous budget does not abort a fast run (no false positives / overhead).
// ---------------------------------------------------------------------------

#[test]
fn fast_run_completes_with_generous_budget() {
    let provider: Arc<dyn RecordProvider> = Arc::new(SlowProvider::new(Duration::ZERO, 4));
    let creature = make_creature(3);

    let stats = rank_focus_neurons_with_provider_and_budget(
        provider,
        &creature,
        None,
        None,
        None,
        None,
        Some(60_000),
    )
    .expect("fast run within a generous budget must succeed");

    // 3 hidden + 1 output are all selectable.
    assert_eq!(stats.total_neurons, 4, "all selectable neurons ranked");
    assert_eq!(stats.neurons.len(), 4, "every selectable neuron returned");
}

// ---------------------------------------------------------------------------
// A disabled budget (None) never aborts, even for a slow loader.
// ---------------------------------------------------------------------------

#[test]
fn disabled_budget_does_not_abort() {
    // Small, slow run: with the budget disabled it must complete despite the
    // per-get delay rather than aborting.
    let provider: Arc<dyn RecordProvider> =
        Arc::new(SlowProvider::new(Duration::from_millis(10), 4));
    let creature = make_creature(3);

    let stats = rank_focus_neurons_with_provider_and_budget(
        provider, &creature, None, None, None, None, None,
    )
    .expect("disabled budget must let the run finish");

    assert_eq!(stats.neurons.len(), 4);
}

// ---------------------------------------------------------------------------
// Env-var parsing for the budget (Issue #1375).
// ---------------------------------------------------------------------------

const BUDGET_ENV: &str = "NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS";

/// RAII guard that sets/restores `BUDGET_ENV` for a single test.
struct EnvVarGuard {
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(value: &str) -> Self {
        let previous = std::env::var(BUDGET_ENV).ok();
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe { std::env::set_var(BUDGET_ENV, value) };
        Self { previous }
    }

    fn unset() -> Self {
        let previous = std::env::var(BUDGET_ENV).ok();
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe { std::env::remove_var(BUDGET_ENV) };
        Self { previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            // SAFETY: serialised via #[serial] — no concurrent env access.
            Some(v) => unsafe { std::env::set_var(BUDGET_ENV, v) },
            // SAFETY: serialised via #[serial] — no concurrent env access.
            None => unsafe { std::env::remove_var(BUDGET_ENV) },
        }
    }
}

#[test]
#[serial]
fn budget_env_unset_uses_default() {
    let _guard = EnvVarGuard::unset();
    assert_eq!(
        focus_ranking_budget_ms(),
        Some(DEFAULT_FOCUS_RANKING_BUDGET_MS)
    );
}

#[test]
#[serial]
fn budget_env_zero_disables() {
    let _guard = EnvVarGuard::set("0");
    assert_eq!(focus_ranking_budget_ms(), None);
}

#[test]
#[serial]
fn budget_env_valid_value_is_used() {
    let _guard = EnvVarGuard::set("5000");
    assert_eq!(focus_ranking_budget_ms(), Some(5_000));
}

#[test]
#[serial]
fn budget_env_clamps_below_minimum() {
    let _guard = EnvVarGuard::set("1");
    assert_eq!(focus_ranking_budget_ms(), Some(MIN_FOCUS_RANKING_BUDGET_MS));
}

#[test]
#[serial]
fn budget_env_clamps_above_maximum() {
    let _guard = EnvVarGuard::set("99999999");
    assert_eq!(focus_ranking_budget_ms(), Some(MAX_FOCUS_RANKING_BUDGET_MS));
}

#[test]
#[serial]
fn budget_env_invalid_falls_back_to_default() {
    let _guard = EnvVarGuard::set("not-a-number");
    assert_eq!(
        focus_ranking_budget_ms(),
        Some(DEFAULT_FOCUS_RANKING_BUDGET_MS)
    );
}
