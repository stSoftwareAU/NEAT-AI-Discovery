//! Issue #1930: the process-wide GPU circuit breaker.
//!
//! Once the GPU has wedged, nothing in this process may spawn another GPU
//! thread or start another 60–300s wait against it. The breaker is plain
//! atomics, so none of this needs a real device — the tests trip it through the
//! public API and assert the observable outcome.
//!
//! The per-entry-point suppression of `submit_*`/`evaluate_*` is covered by the
//! crate-internal tests in `src/analysis/gpu/queue/submission.rs`, since most of
//! those methods and the `GpuWorkQueue` fixture they need are `pub(crate)`.
//! What is observable from outside the crate — queue creation, the error
//! contents, the single warn, the reset hook and the metrics counter — is
//! covered here.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use neat_ai_discovery::analysis::gpu::breaker::{
    GpuTripReason, abandoned_gpu_thread_count, check_gpu_breaker, gpu_breaker_trip_reason,
    is_gpu_breaker_tripped, record_abandoned_gpu_thread, reset_gpu_breaker, trip_gpu_breaker,
};
use neat_ai_discovery::analysis::gpu::queue::GpuWorkQueue;
use neat_ai_discovery::observability::global_gpu_metrics;
use serial_test::serial;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

/// Well under `GPU_INIT_TIMEOUT_SECS` and the 60s minimum batch wait, and far
/// above the microseconds a suppressed call actually takes.
const IMMEDIATE: Duration = Duration::from_secs(2);

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

// =============================================================================
// No further GPU threads after the first trip
// =============================================================================

/// `GpuWorkQueue::new()` must fail immediately once the breaker has tripped.
///
/// Creating a queue normally spawns a GPU thread and then waits up to
/// `GPU_INIT_TIMEOUT_SECS` for it to initialise; returning inside `IMMEDIATE`
/// is only possible if neither happened.
#[test]
#[serial]
fn queue_creation_is_refused_after_a_trip() {
    with_tripped(GpuTripReason::AbandonedThread, || {
        let started = Instant::now();
        let err = GpuWorkQueue::new()
            .err()
            .expect("queue creation must be refused while the breaker is tripped");
        let elapsed = started.elapsed();

        assert!(
            elapsed < IMMEDIATE,
            "refusal must not spawn or wait on a GPU thread, took {elapsed:?}"
        );
        assert!(
            format!("{err:#}").contains("GPU circuit breaker tripped"),
            "got: {err:#}"
        );
    });
}

/// Repeated attempts stay refused — the breaker is one-way for the life of the
/// process, so a later analysis cannot start a fresh wedged run.
#[test]
#[serial]
fn queue_creation_stays_refused_on_every_later_attempt() {
    with_tripped(GpuTripReason::BatchTimeout, || {
        for attempt in 1..=3 {
            assert!(
                GpuWorkQueue::new().is_err(),
                "attempt {attempt} must still be refused"
            );
        }
    });
}

// =============================================================================
// The error carries the trip reason and the abandoned-thread count
// =============================================================================

#[test]
#[serial]
fn the_breaker_error_carries_the_reason_and_the_abandoned_count() {
    reset_gpu_breaker();
    record_abandoned_gpu_thread();

    let err = check_gpu_breaker().expect_err("the breaker refuses work once tripped");
    let msg = format!("{err:#}");

    assert!(
        msg.contains(GpuTripReason::AbandonedThread.as_str()),
        "the error must name the original trip reason, got: {msg}"
    );
    assert!(
        msg.contains("abandoned GPU threads: 1"),
        "the error must carry the abandoned-thread count, got: {msg}"
    );

    reset_gpu_breaker();
}

/// The first trip owns the diagnosis: a later trip for a different reason must
/// not overwrite what actually went wrong first.
#[test]
#[serial]
fn the_original_trip_reason_survives_later_trips() {
    with_tripped(GpuTripReason::InitTimeout, || {
        trip_gpu_breaker(GpuTripReason::BatchTimeout);
        assert_eq!(gpu_breaker_trip_reason(), Some(GpuTripReason::InitTimeout));
        assert!(
            format!("{:#}", check_gpu_breaker().unwrap_err())
                .contains(GpuTripReason::InitTimeout.as_str())
        );
    });
}

// =============================================================================
// Abandoned-thread count via GPU metrics
// =============================================================================

/// The count the breaker reports is the same one GPU metrics publish, so a run
/// log carries the number without extra plumbing. It must never exceed 1 per
/// process now that the first abandoned thread stops all further GPU work.
#[test]
#[serial]
fn the_abandoned_thread_count_is_exposed_via_gpu_metrics() {
    reset_gpu_breaker();
    assert_eq!(global_gpu_metrics().abandoned_threads(), 0);

    record_abandoned_gpu_thread();

    assert_eq!(global_gpu_metrics().abandoned_threads(), 1);
    assert_eq!(
        abandoned_gpu_thread_count(),
        global_gpu_metrics().abandoned_threads(),
        "the breaker and GPU metrics must report the same counter"
    );

    reset_gpu_breaker();
}

// =============================================================================
// Log-flood guard: exactly one warn on trip, debug-only afterwards
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

/// A wedged GPU produces failure after failure; the trip must be announced
/// exactly once at `warn`, and everything after it must stay at debug or below,
/// or production logs flood.
#[test]
#[serial]
fn the_trip_logs_exactly_one_warn_and_nothing_louder_afterwards() {
    reset_gpu_breaker();

    let events = capture_events(|| {
        trip_gpu_breaker(GpuTripReason::BatchTimeout);
        // Everything after the first trip is suppressed noise.
        trip_gpu_breaker(GpuTripReason::BatchTimeout);
        trip_gpu_breaker(GpuTripReason::AbandonedThread);
        let _ = check_gpu_breaker();
        let _ = check_gpu_breaker();
    });

    let warns: Vec<&CapturedEvent> = events
        .iter()
        .filter(|e| e.level <= tracing::Level::WARN)
        .collect();
    assert_eq!(
        warns.len(),
        1,
        "the breaker must announce itself exactly once, got: {warns:#?}"
    );
    let warn = warns[0];
    assert!(
        warn.message.contains("GPU circuit breaker tripped"),
        "got: {}",
        warn.message
    );
    assert_eq!(
        warn.fields.get("reason").map(String::as_str),
        Some(GpuTripReason::BatchTimeout.as_str()),
        "the single warn must carry the trip reason"
    );
    assert!(
        warn.fields.contains_key("abandoned_gpu_threads"),
        "the single warn must carry the abandoned-thread count"
    );

    assert!(
        events.len() > 1,
        "the suppressed calls must still leave a debug trail for diagnosis"
    );

    reset_gpu_breaker();
}

// =============================================================================
// Test-only reset hook
// =============================================================================

/// The reset hook restores the untripped state, so tests are not
/// order-dependent (mirroring `cancellation::reset_cancellation()`).
#[test]
#[serial]
fn the_reset_hook_restores_the_untripped_state() {
    reset_gpu_breaker();
    assert!(!is_gpu_breaker_tripped(), "starts closed");

    record_abandoned_gpu_thread();
    assert!(is_gpu_breaker_tripped());
    assert_eq!(abandoned_gpu_thread_count(), 1);

    reset_gpu_breaker();

    assert!(!is_gpu_breaker_tripped(), "reset must close the breaker");
    assert!(gpu_breaker_trip_reason().is_none());
    assert!(check_gpu_breaker().is_ok(), "GPU work is allowed again");
    assert_eq!(
        abandoned_gpu_thread_count(),
        0,
        "reset must clear the abandoned-thread count too"
    );
}
