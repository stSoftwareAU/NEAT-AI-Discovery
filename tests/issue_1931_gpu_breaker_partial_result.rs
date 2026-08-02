//! Issue #1931: a tripped GPU breaker must yield a signalled partial result,
//! never an error.
//!
//! The circuit breaker (Issue #1930) stops GPU work; these tests pin what the
//! analyses do with the rest of the run's budget. All three GPU-queue
//! construction sites used to treat "no GPU queue" as a hard error, so a wedged
//! GPU propagated a raw error out of `analyze_all`, threw away the CPU-side
//! accounting and read to the host as one more failed attempt.
//!
//! None of this needs a real device: the breaker is plain atomics and every
//! skip happens *before* the GPU is touched, so the tests run identically on a
//! CI box with no adapter. They are `#[serial]` because the breaker is
//! process-wide; each integration binary is its own process, so tripping it
//! here cannot affect any other test binary.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use neat_ai_discovery::analysis::cache::RecordCache;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    REJECTION_FINGERPRINT_UNCHANGED, REJECTION_GPU_WEDGED,
};
use neat_ai_discovery::analysis::gpu::breaker::{
    GpuTripReason, reset_gpu_breaker, trip_gpu_breaker,
};
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
use neat_ai_discovery::analysis::shared::AnalyzeAllResult;
use neat_ai_discovery::analysis::{
    AnalysisOutcome, EnvironmentalDisableReason, analyze_all, analyze_neurons_with_cache,
    analyze_synapses_with_cache,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    AnalyzeAllInput, AnalyzeNeuronsInput, AnalyzeSynapsesInput, CreatureJson, NeuronJson,
    SynapseJson,
};
use serial_test::serial;
use tempfile::NamedTempFile;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

const HIDDEN: &str = "hidden-0";
const OUTPUT: &str = "output-0";

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

/// A tracker the host has already accumulated — it must survive the wedged run
/// so the next run does not start from zero history.
fn seeded_tracker() -> ModuleOutcomeTracker {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record("add-synapses", true);
    tracker.record("add-neurons", false);
    tracker
}

fn focus_neurons() -> Vec<String> {
    vec![HIDDEN.to_string(), OUTPUT.to_string()]
}

/// A small parquet so the shared record cache builds. The wedged skip happens
/// before any record is read; the file only has to exist and parse.
fn temp_parquet() -> NamedTempFile {
    let mut records = Vec::new();
    for obs in 0..8u32 {
        let activation = f32::from(u16::try_from(obs).unwrap_or(0)) / 8.0;
        records.push(DiscoverRecord::new(
            obs,
            HIDDEN.to_string(),
            Some(activation),
            activation,
            vec![0.1],
        ));
        records.push(DiscoverRecord::new(
            obs,
            OUTPUT.to_string(),
            Some(activation),
            activation,
            vec![0.2],
        ));
    }
    let tmp = NamedTempFile::new().expect("create temp parquet");
    write_records_to_parquet(tmp.path().to_str().expect("utf-8 path"), &records)
        .expect("write parquet");
    tmp
}

fn record_cache(path: &str) -> Arc<RecordCache> {
    Arc::new(RecordCache::new_adaptive(path).expect("record cache builds"))
}

fn all_input(parquet_file: String, include_analyses: bool) -> AnalyzeAllInput {
    AnalyzeAllInput {
        parquet_file,
        creature: creature(),
        focus_neurons: focus_neurons(),
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(include_analyses),
        include_neuron_analysis: Some(include_analyses),
        random_seed: Some(1931),
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

fn neuron_input(parquet_file: String) -> AnalyzeNeuronsInput {
    AnalyzeNeuronsInput {
        parquet_file,
        creature: creature(),
        focus_neurons: focus_neurons(),
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: Some(1931),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        task_descriptor: None,
    }
}

fn synapse_input(parquet_file: String) -> AnalyzeSynapsesInput {
    AnalyzeSynapsesInput {
        parquet_file,
        creature: creature(),
        focus_neurons: focus_neurons(),
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: Some(1931),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    }
}

/// Trip the breaker for the duration of `body`, always resetting afterwards —
/// even on panic — so a failure here cannot wedge the rest of the suite.
fn with_tripped<R>(reason: GpuTripReason, body: impl FnOnce() -> R) -> R {
    struct ResetOnDrop;
    impl Drop for ResetOnDrop {
        fn drop(&mut self) {
            reset_gpu_breaker();
        }
    }
    reset_gpu_breaker();
    let _guard = ResetOnDrop;
    trip_gpu_breaker(reason);
    body()
}

fn wedged_count(breakdown: &neat_ai_discovery::analysis::diagnostics::RejectionBreakdown) -> u32 {
    breakdown
        .counts()
        .get(REJECTION_GPU_WEDGED)
        .copied()
        .unwrap_or(0)
}

// =============================================================================
// (a) Every entry point returns Ok with empty GPU-derived fields
// =============================================================================

/// `analyze_all` must mirror the cancellation path — `Ok` with empty
/// `synapse`/`neuron` and the wedged signal set — not propagate the breaker
/// error out of the GPU-queue construction site.
#[test]
#[serial]
fn analyze_all_returns_a_signalled_partial_result() {
    let parquet = temp_parquet();
    let path = parquet.path().to_str().expect("utf-8 path").to_string();

    let result = with_tripped(GpuTripReason::AbandonedThread, || {
        analyze_all(&all_input(path, true)).expect("a wedged GPU must not surface as an error")
    });

    assert!(result.gpu_wedged, "the wedged signal must be set");
    assert!(result.synapse.is_none(), "no GPU-derived synapse results");
    assert!(result.neuron.is_none(), "no GPU-derived neuron results");
    assert!(!result.cancelled, "the host did not cancel this run");
    assert!(!result.memory_budget_exceeded);
    assert!(!result.memory_pressure_cancelled);
}

#[test]
#[serial]
fn analyze_neurons_with_cache_returns_a_signalled_partial_result() {
    let parquet = temp_parquet();
    let path = parquet.path().to_str().expect("utf-8 path").to_string();
    let cache = record_cache(&path);

    let result = with_tripped(GpuTripReason::BatchTimeout, || {
        analyze_neurons_with_cache(&neuron_input(path), cache)
            .expect("a wedged GPU must not surface as an error")
    });

    assert!(result.metadata.gpu_wedged, "the wedged signal must be set");
    assert!(result.helpful_neurons.is_empty(), "no candidates");
    assert!(!result.gpu_used, "no GPU work was attempted");
    assert_eq!(result.metadata.candidates_returned, 0);
    assert_eq!(
        result.metadata.completed_focus_neurons, 0,
        "no focus neuron was evaluated"
    );
    assert_eq!(result.metadata.total_focus_neurons, 2);
    assert_eq!(
        wedged_count(&result.metadata.rejection_breakdown),
        2,
        "one gpu_wedged count per focus neuron that was never evaluated"
    );
}

#[test]
#[serial]
fn analyze_synapses_with_cache_returns_a_signalled_partial_result() {
    let parquet = temp_parquet();
    let path = parquet.path().to_str().expect("utf-8 path").to_string();
    let cache = record_cache(&path);

    let result = with_tripped(GpuTripReason::InitTimeout, || {
        analyze_synapses_with_cache(&synapse_input(path), cache)
            .expect("a wedged GPU must not surface as an error")
    });

    assert!(result.metadata.gpu_wedged, "the wedged signal must be set");
    assert!(result.helpful_synapses.is_empty(), "no candidates");
    assert!(result.harmful_synapses.is_empty());
    assert!(result.coordinated_structural_candidates.is_empty());
    assert!(!result.gpu_used, "no GPU work was attempted");
    assert_eq!(result.metadata.candidates_returned, 0);
    assert_eq!(result.metadata.completed_focus_neurons, 0);
    assert_eq!(wedged_count(&result.metadata.rejection_breakdown), 2);
}

/// The skip must not depend on which failure tripped the breaker.
#[test]
#[serial]
fn every_trip_reason_skips_rather_than_errors() {
    let parquet = temp_parquet();
    let path = parquet.path().to_str().expect("utf-8 path").to_string();

    for reason in [
        GpuTripReason::AbandonedThread,
        GpuTripReason::BatchTimeout,
        GpuTripReason::InitTimeout,
    ] {
        let result = with_tripped(reason, || {
            analyze_all(&all_input(path.clone(), true))
                .unwrap_or_else(|e| panic!("{reason} must skip, not error: {e:#}"))
        });
        assert!(result.gpu_wedged, "{reason} must set the wedged signal");
    }
}

// =============================================================================
// (b) The wedged result is distinguishable from a genuine zero-candidate pass
// =============================================================================

/// A run with both analyses switched off returns the *same* surface — `Ok`,
/// `synapse: None`, `neuron: None`, zero candidates — as a wedged run. Only the
/// signal tells them apart, which is the whole point of Issue #1931: a wedged
/// pass must never be counted as search exhaustion.
#[test]
#[serial]
fn a_wedged_pass_is_not_a_genuine_zero_candidate_pass() {
    let parquet = temp_parquet();
    let path = parquet.path().to_str().expect("utf-8 path").to_string();

    reset_gpu_breaker();
    let genuine = analyze_all(&all_input(path.clone(), false)).expect("a healthy empty pass is Ok");
    let wedged = with_tripped(GpuTripReason::BatchTimeout, || {
        analyze_all(&all_input(path, true)).expect("a wedged pass is Ok")
    });

    // Identical on the surface...
    assert!(genuine.synapse.is_none() && wedged.synapse.is_none());
    assert!(genuine.neuron.is_none() && wedged.neuron.is_none());

    // ...and unambiguously different in the signal.
    assert!(!genuine.gpu_wedged, "a healthy empty pass is not wedged");
    assert!(wedged.gpu_wedged);

    let genuine_outcome = AnalysisOutcome::from_result(&genuine);
    let wedged_outcome = AnalysisOutcome::from_result(&wedged);

    assert!(
        genuine_outcome.is_genuinely_empty(),
        "a healthy empty pass is search exhaustion: {genuine_outcome:?}"
    );
    assert_eq!(genuine_outcome.disable_reason(), None);

    assert!(
        wedged_outcome.is_environmentally_disabled(),
        "a wedged pass carries no exhaustion signal: {wedged_outcome:?}"
    );
    assert!(!wedged_outcome.is_genuinely_empty());
    assert_eq!(
        wedged_outcome.disable_reason(),
        Some(EnvironmentalDisableReason::GpuWedged),
        "a wedged GPU is its own disable reason, distinct from gpu_unavailable"
    );
    assert_eq!(EnvironmentalDisableReason::GpuWedged.as_str(), "gpu_wedged");
    assert_ne!(
        EnvironmentalDisableReason::GpuWedged,
        EnvironmentalDisableReason::GpuUnavailable,
        "a wedged GPU and a missing GPU need different operator remedies"
    );
}

// =============================================================================
// (c) The CPU-side accounting still runs
// =============================================================================

/// Skipping GPU work must not skip the bookkeeping that keeps the run's
/// accounting correct — otherwise the host loses the fingerprints it must pass
/// back and the tracker it must persist.
#[test]
#[serial]
fn cpu_side_accounting_survives_the_skip() {
    let parquet = temp_parquet();
    let path = parquet.path().to_str().expect("utf-8 path").to_string();

    let result: AnalyzeAllResult = with_tripped(GpuTripReason::AbandonedThread, || {
        analyze_all(&all_input(path, true)).expect("a wedged pass is Ok")
    });

    let fingerprints = result
        .neuron_fingerprints
        .as_ref()
        .expect("fingerprint bookkeeping must still be returned");
    assert!(
        fingerprints.contains_key(HIDDEN) && fingerprints.contains_key(OUTPUT),
        "every neuron must be fingerprinted so the next run can reuse the cache"
    );
    assert_eq!(
        result.fingerprint_cache_hits + result.fingerprint_cache_misses,
        2,
        "every focus neuron must be accounted for as a hit or a miss"
    );
    assert_eq!(
        result.fingerprint_cache_misses, 2,
        "no previous run to skip"
    );

    // The tracker round-trips so the host can persist it across the wedged run
    // rather than losing the history it had accumulated.
    assert_eq!(
        result.module_outcome_tracker.all_stats(),
        seeded_tracker().all_stats(),
        "the module outcome tracker must be returned, not dropped"
    );

    assert_eq!(
        wedged_count(&result.pass_rejection_breakdown),
        2,
        "the pass breakdown must attribute both unevaluated focus neurons to the wedged GPU"
    );
    assert_eq!(
        result
            .pass_rejection_breakdown
            .counts()
            .get(REJECTION_FINGERPRINT_UNCHANGED)
            .copied()
            .unwrap_or(0),
        0,
        "nothing was skipped by the fingerprint cache on this run"
    );
}

// =============================================================================
// (d) Exactly one warning per run, not one per analysis attempt
// =============================================================================

/// One captured tracing event.
#[derive(Debug, Clone)]
struct CapturedEvent {
    level: tracing::Level,
    message: String,
    fields: HashMap<String, String>,
}

#[derive(Default)]
struct FieldVisitor {
    message: String,
    fields: HashMap<String, String>,
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.store(field.name(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.store(field.name(), value.to_string());
    }
}

impl FieldVisitor {
    fn store(&mut self, name: &str, value: String) {
        if name == "message" {
            self.message = value;
        } else {
            self.fields.insert(name.to_string(), value);
        }
    }
}

/// Collects every event emitted while the layer is the active subscriber.
struct CaptureLayer {
    events: Arc<Mutex<Vec<CapturedEvent>>>,
}

impl<S: tracing::Subscriber> Layer<S> for CaptureLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        self.events
            .lock()
            .expect("capture buffer poisoned")
            .push(CapturedEvent {
                level: *event.metadata().level(),
                message: visitor.message,
                fields: visitor.fields,
            });
    }
}

fn capture_events<F: FnOnce()>(body: F) -> Vec<CapturedEvent> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let subscriber = Registry::default().with(CaptureLayer {
        events: Arc::clone(&events),
    });
    tracing::subscriber::with_default(subscriber, body);
    events.lock().expect("capture buffer poisoned").clone()
}

/// A wedged run re-attempts analysis pass after pass. The skip must be
/// announced once, or production logs flood with an identical warning per
/// attempt — the exact noise the runtime backstop watches for.
#[test]
#[serial]
fn the_skip_warns_exactly_once_per_run() {
    let parquet = temp_parquet();
    let path = parquet.path().to_str().expect("utf-8 path").to_string();
    let cache = record_cache(&path);

    reset_gpu_breaker();
    // Trip outside the capture: the breaker's own trip warning (Issue #1930) is
    // a separate, already-tested event.
    trip_gpu_breaker(GpuTripReason::BatchTimeout);

    let events = capture_events(|| {
        for _ in 0..3 {
            assert!(analyze_all(&all_input(path.clone(), true)).is_ok());
            assert!(
                analyze_neurons_with_cache(&neuron_input(path.clone()), Arc::clone(&cache)).is_ok()
            );
            assert!(
                analyze_synapses_with_cache(&synapse_input(path.clone()), Arc::clone(&cache))
                    .is_ok()
            );
        }
    });
    reset_gpu_breaker();

    let skip_warns: Vec<&CapturedEvent> = events
        .iter()
        .filter(|e| e.level <= tracing::Level::WARN && e.message.contains("GPU analyses skipped"))
        .collect();

    assert_eq!(
        skip_warns.len(),
        1,
        "nine analysis attempts must produce exactly one skip warning, got: {skip_warns:#?}"
    );
    let warn = skip_warns[0];
    assert!(
        warn.message.contains("restart"),
        "the warning must tell the operator the process needs an external restart: {}",
        warn.message
    );
    assert!(
        warn.message.contains("no CPU fallback"),
        "the warning must say the empty result is genuine, not a fallback: {}",
        warn.message
    );
    assert_eq!(
        warn.fields.get("reason").map(String::as_str),
        Some(GpuTripReason::BatchTimeout.as_str()),
        "the warning must carry the original trip reason"
    );
}

/// The latch is re-armed by `reset_gpu_breaker`, so a later run warns again
/// rather than going silent for the life of the test binary.
#[test]
#[serial]
fn a_later_run_warns_again_after_a_reset() {
    let parquet = temp_parquet();
    let path = parquet.path().to_str().expect("utf-8 path").to_string();

    let warns_in_one_run = || {
        capture_events(|| {
            let _ = analyze_all(&all_input(path.clone(), true));
        })
        .into_iter()
        .filter(|e| e.level <= tracing::Level::WARN && e.message.contains("GPU analyses skipped"))
        .count()
    };

    let first = with_tripped(GpuTripReason::InitTimeout, warns_in_one_run);
    let second = with_tripped(GpuTripReason::InitTimeout, warns_in_one_run);

    assert_eq!(first, 1, "the first run announces the skip");
    assert_eq!(second, 1, "a reset re-arms the latch for the next run");
}
