//! Issue #1935: the wedged-GPU sequence, end to end through the public API.
//!
//! The crate-internal half of this harness lives in
//! `src/analysis/gpu/queue/{fake_evaluator,wedge_tests}.rs`, where a fake
//! `GpuEvaluator` drives the production work loop without a device. What that
//! half cannot reach is what happens *outside* the queue once the GPU has been
//! declared wedged, so this binary pins the rest of the chain:
//!
//! ```text
//! silent GPU ──▶ stall verdict ──▶ breaker trips ──▶ no second GPU thread
//!                (Issue #1933)     (Issue #1930)     (Issue #1930)
//!                                        │
//!                                        └──▶ analyze_all → Ok, gpu_wedged
//!                                             + CPU accounting (Issue #1931)
//! ```
//!
//! No GPU is involved: the wedge is a response channel nobody answers and a
//! heartbeat nobody advances, and every skip happens before a device is
//! touched. The tests are `#[serial]` because the breaker is process-wide;
//! this is its own test binary, so tripping it here cannot reach another one.

use std::time::{Duration, Instant};

use crossbeam_channel::bounded;
use neat_ai_discovery::analysis::gpu::breaker::{
    GpuTripReason, global_gpu_breaker, gpu_breaker_trip_reason, is_gpu_breaker_tripped,
    reset_gpu_breaker,
};
use neat_ai_discovery::analysis::gpu::heartbeat::GpuHeartbeat;
use neat_ai_discovery::analysis::gpu::queue::GpuWorkQueue;
use neat_ai_discovery::analysis::gpu::queue::submission::{
    GpuWaitOutcome, heartbeat_stall_error, wait_for_gpu_response,
};
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
use neat_ai_discovery::analysis::{AnalysisOutcome, EnvironmentalDisableReason, analyze_all};
use neat_ai_discovery::ffi_types::{DiscoveryErrorKind, classify_anyhow_error};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use tempfile::NamedTempFile;

const HIDDEN: &str = "hidden-0";
const OUTPUT: &str = "output-0";

/// The stall window the wedge is detected within — short enough to keep this
/// binary fast, long enough not to misread scheduling jitter on a CI runner.
const STALL_WINDOW: Duration = Duration::from_millis(200);

/// The batch timeout the caller would otherwise have sat out in full. The whole
/// point of the #1926 breakdown is that detection costs the window, not this.
const BATCH_TIMEOUT: Duration = Duration::from_secs(300);

/// Stand-in for a run's whole GPU budget. Before these fixes a wedged GPU cost
/// each pass a fresh 60–300s wait plus an abandoned thread, walking the process
/// towards its external 3-hour kill.
const SIMULATED_RUN_BUDGET: Duration = Duration::from_secs(10);

// =============================================================================
// Fixtures
// =============================================================================

fn creature() -> CreatureJson {
    let neuron = |uuid: &str, neuron_type: &str, squash: &str| NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    };
    let synapse = |from: &str, to: &str| SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight: 0.5,
        synapse_type: None,
    };
    CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron(HIDDEN, "hidden", "TANH"),
            neuron(OUTPUT, "output", "LOGISTIC"),
        ],
        synapses: vec![synapse("input-0", HIDDEN), synapse(HIDDEN, OUTPUT)],
        input: 1,
        output: 1,
    }
}

/// History the host has already accumulated — it must survive the wedged run.
fn seeded_tracker() -> ModuleOutcomeTracker {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record("add-synapses", true);
    tracker.record("add-neurons", false);
    tracker
}

/// A small parquet so the record cache builds. The wedged skip happens before
/// any record is read; the file only has to exist and parse.
fn temp_parquet() -> NamedTempFile {
    let mut records = Vec::new();
    for obs in 0..8u32 {
        let activation = f32::from(u16::try_from(obs).unwrap_or(0)) / 8.0;
        for uuid in [HIDDEN, OUTPUT] {
            records.push(DiscoverRecord::new(
                obs,
                uuid.to_string(),
                Some(activation),
                activation,
                vec![0.1],
            ));
        }
    }
    let tmp = NamedTempFile::new().expect("create temp parquet");
    write_records_to_parquet(tmp.path().to_str().expect("utf-8 path"), &records)
        .expect("write parquet");
    tmp
}

fn all_input(parquet_file: String) -> AnalyzeAllInput {
    AnalyzeAllInput {
        parquet_file,
        creature: creature(),
        focus_neurons: vec![HIDDEN.to_string(), OUTPUT.to_string()],
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: Some(1935),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: Some(seeded_tracker()),
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        cost_name: Some("MSE".to_string()),
    }
}

/// Reset the process-wide breaker on the way in and out — even on panic — so a
/// failure here cannot wedge the rest of this binary.
struct BreakerReset;

impl BreakerReset {
    fn armed() -> Self {
        reset_gpu_breaker();
        Self
    }
}

impl Drop for BreakerReset {
    fn drop(&mut self) {
        reset_gpu_breaker();
    }
}

/// Wedge a GPU: a response channel whose sender is alive but silent, and a
/// heartbeat that never advances. Returns the elapsed detection time.
///
/// This is the production wait, not a simulation of one — only the device
/// behind it is missing.
fn detect_a_wedged_gpu() -> Duration {
    // The sender stays alive, so only the stall guard can end this wait.
    let (_response_tx, response_rx) = bounded::<anyhow::Result<()>>(1);
    let heartbeat = GpuHeartbeat::new();

    let started = Instant::now();
    let outcome = wait_for_gpu_response(&response_rx, BATCH_TIMEOUT, &heartbeat, STALL_WINDOW);
    let elapsed = started.elapsed();

    let GpuWaitOutcome::Stalled { idle, window } = outcome else {
        panic!("a silent GPU must be reported as stalled, got {outcome:?}");
    };
    assert!(idle >= window, "the reported idle time covers the window");

    // The trip site the submitter uses (Issue #1933), on the process-wide
    // breaker every GPU entry point consults (Issue #1930).
    let err = heartbeat_stall_error(
        global_gpu_breaker(),
        "helpful batch evaluation",
        idle,
        window,
    );
    assert_eq!(
        classify_anyhow_error(&err),
        DiscoveryErrorKind::GpuWedged,
        "Issue #1932: a wedged GPU must not be reported as a retryable timeout"
    );
    assert!(!DiscoveryErrorKind::GpuWedged.is_retryable());

    elapsed
}

// =============================================================================
// The sequence
// =============================================================================

/// Issues #1933 and #1930: a silent GPU is declared wedged within the stall
/// window, and that verdict is remembered process-wide.
#[test]
#[serial]
fn a_silent_gpu_trips_the_process_wide_breaker_within_the_stall_window() {
    let _reset = BreakerReset::armed();
    assert!(!is_gpu_breaker_tripped(), "the run starts healthy");

    let elapsed = detect_a_wedged_gpu();

    assert!(
        elapsed >= STALL_WINDOW,
        "the guard must not fire before the window elapses (fired after {elapsed:?})"
    );
    assert!(
        elapsed < Duration::from_secs(2) && elapsed < BATCH_TIMEOUT,
        "detection must cost the stall window, not the {BATCH_TIMEOUT:?} batch timeout \
         (took {elapsed:?})"
    );
    assert_eq!(
        gpu_breaker_trip_reason(),
        Some(GpuTripReason::HeartbeatStall)
    );
}

/// Issue #1930: the first wedge must stop the next GPU thread being spawned.
///
/// `GpuWorkQueue::new()` normally spawns a thread and waits up to
/// `GPU_INIT_TIMEOUT_SECS` for it to initialise, so returning immediately with
/// the breaker error is only possible if neither happened.
#[test]
#[serial]
fn no_second_gpu_thread_is_spawned_after_the_first_wedge() {
    let _reset = BreakerReset::armed();
    detect_a_wedged_gpu();

    let started = Instant::now();
    for attempt in 1..=3 {
        let err = GpuWorkQueue::new()
            .err()
            .unwrap_or_else(|| panic!("attempt {attempt} must be refused by the breaker"));
        assert!(
            format!("{err:#}").contains("GPU circuit breaker tripped"),
            "attempt {attempt} must fail with the breaker error, got: {err:#}"
        );
    }
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(2),
        "three refusals must not spawn or wait on a GPU thread, took {elapsed:?}"
    );
}

/// Issue #1931: with the GPU wedged, `analyze_all` returns a signalled partial
/// result rather than an error, and the accounting that does not depend on the
/// GPU is still populated.
#[test]
#[serial]
fn analyze_all_returns_a_signalled_partial_result_after_a_wedge() {
    let _reset = BreakerReset::armed();
    let parquet = temp_parquet();
    let path = parquet.path().to_str().expect("utf-8 path").to_string();

    detect_a_wedged_gpu();
    let result = analyze_all(&all_input(path)).expect("a wedged GPU must not surface as an error");

    assert!(result.gpu_wedged, "the wedged signal must be set");
    assert!(result.synapse.is_none(), "no GPU-derived synapse results");
    assert!(result.neuron.is_none(), "no GPU-derived neuron results");
    assert!(!result.cancelled, "the host did not cancel this run");

    let fingerprints = result
        .neuron_fingerprints
        .as_ref()
        .expect("fingerprint bookkeeping must still be returned");
    assert!(
        fingerprints.contains_key(HIDDEN) && fingerprints.contains_key(OUTPUT),
        "every focus neuron must still be fingerprinted for the next run"
    );
    assert_eq!(
        result.fingerprint_cache_hits + result.fingerprint_cache_misses,
        2,
        "every focus neuron must be accounted for as a hit or a miss"
    );
    assert_eq!(
        result.module_outcome_tracker.all_stats(),
        seeded_tracker().all_stats(),
        "the module outcome tracker must be returned, not dropped"
    );

    let outcome = AnalysisOutcome::from_result(&result);
    assert!(
        !outcome.is_genuinely_empty(),
        "a wedged pass must never be counted as search exhaustion"
    );
    assert_eq!(
        outcome.disable_reason(),
        Some(EnvironmentalDisableReason::GpuWedged)
    );
}

/// The umbrella criterion of Issue #1926: the whole sequence — wedge, detect,
/// trip, then keep the run going — fits inside a run budget measured in
/// seconds, so a wedged GPU can no longer walk the process towards the external
/// 3-hour kill.
#[test]
#[serial]
fn the_whole_wedge_sequence_fits_inside_a_simulated_run_budget() {
    let _reset = BreakerReset::armed();
    let parquet = temp_parquet();
    let path = parquet.path().to_str().expect("utf-8 path").to_string();

    let started = Instant::now();
    detect_a_wedged_gpu();
    // Every later analysis pass of the run, each of which used to cost a fresh
    // GPU thread and a 60–300s wait.
    for pass in 1..=5 {
        let result = analyze_all(&all_input(path.clone()))
            .unwrap_or_else(|e| panic!("pass {pass} must skip, not error: {e:#}"));
        assert!(
            result.gpu_wedged,
            "pass {pass} must stay signalled as wedged"
        );
    }
    let elapsed = started.elapsed();

    assert!(
        elapsed < SIMULATED_RUN_BUDGET,
        "a wedged GPU must cost the run seconds, not minutes (took {elapsed:?})"
    );
}
