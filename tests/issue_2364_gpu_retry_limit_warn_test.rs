//! Issue #2364 regression test — `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` used to be
//! parsed with no log at all, so `""`, `"abc"` and `"999999"` silently fell back
//! to the default of 3, and `"0"` silently disabled the #647 device-lost
//! recovery guard. These tests capture the WARN output emitted for each of
//! those cases.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use serial_test::serial;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

use neat_ai_discovery::config::{gpu_retry_limit, resolve_gpu_retry_limit};

/// One captured tracing event.
#[derive(Debug, Clone)]
struct CapturedEvent {
    level: String,
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
                level: event.metadata().level().to_string(),
                message: visitor.message,
                fields: visitor.fields,
            });
    }
}

/// Run `body` with a capturing subscriber installed on this thread and return
/// the events it emitted.
fn capture_events<F: FnOnce()>(body: F) -> Vec<CapturedEvent> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let subscriber = Registry::default().with(CaptureLayer {
        events: Arc::clone(&events),
    });
    tracing::subscriber::with_default(subscriber, body);
    events.lock().expect("capture buffer poisoned").clone()
}

/// Assert that `events` contains exactly one WARN event whose message
/// contains every substring in `expected_substrings`.
fn assert_single_warn_containing(events: &[CapturedEvent], expected_substrings: &[&str]) {
    let warnings: Vec<&CapturedEvent> = events.iter().filter(|e| e.level == "WARN").collect();
    assert_eq!(
        warnings.len(),
        1,
        "expected exactly one WARN event, got {}: {:?}",
        warnings.len(),
        events
    );
    let warning = warnings[0];
    for substring in expected_substrings {
        assert!(
            warning.message.contains(substring),
            "expected WARN message to contain {substring:?}: {}",
            warning.message
        );
    }
}

#[test]
fn empty_value_warns_and_uses_default() {
    let events = capture_events(|| {
        let limit = resolve_gpu_retry_limit(Some(""));
        assert_eq!(limit, 3);
    });
    assert_single_warn_containing(
        &events,
        &[
            "NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT",
            "using the default of 3",
        ],
    );
}

#[test]
fn unparsable_value_warns_and_uses_default() {
    let events = capture_events(|| {
        let limit = resolve_gpu_retry_limit(Some("abc"));
        assert_eq!(limit, 3);
    });
    assert_single_warn_containing(
        &events,
        &[
            "NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT",
            "using the default of 3",
        ],
    );
}

#[test]
fn unparsable_value_field_escapes_control_characters() {
    let events = capture_events(|| {
        let limit = resolve_gpu_retry_limit(Some("5\nFORGED: injected"));
        assert_eq!(limit, 3);
    });
    let warnings: Vec<&CapturedEvent> = events.iter().filter(|e| e.level == "WARN").collect();
    assert_eq!(
        warnings.len(),
        1,
        "expected exactly one WARN event, got {}: {:?}",
        warnings.len(),
        events
    );
    let value_field = warnings[0]
        .fields
        .get("value")
        .expect("expected a `value` field on the WARN event");
    assert!(
        !value_field.contains('\n'),
        "value field must not contain a raw newline: {value_field:?}"
    );
    assert!(
        value_field.contains("\\n"),
        "value field must contain the escaped newline sequence: {value_field:?}"
    );
}

#[test]
fn zero_warns_that_recovery_is_disabled() {
    let events = capture_events(|| {
        let limit = resolve_gpu_retry_limit(Some("0"));
        assert_eq!(limit, 0);
    });
    assert_single_warn_containing(
        &events,
        &[
            "NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT",
            "device-lost recovery is disabled",
        ],
    );
}

#[test]
fn above_maximum_warns_and_uses_default() {
    let events = capture_events(|| {
        let limit = resolve_gpu_retry_limit(Some("999999"));
        assert_eq!(limit, 3);
    });
    assert_single_warn_containing(
        &events,
        &[
            "NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT",
            "above the maximum of 10",
            "using the default of 3",
        ],
    );
}

#[test]
fn valid_values_do_not_warn() {
    for (raw, expected) in [("1", 1), ("5", 5), ("10", 10), (" 7 ", 7)] {
        let events = capture_events(|| {
            let limit = resolve_gpu_retry_limit(Some(raw));
            assert_eq!(limit, expected, "unexpected limit for input {raw:?}");
        });
        assert!(
            events.is_empty(),
            "valid input {raw:?} must not log anything, got {events:?}"
        );
    }
}

#[test]
fn unset_does_not_warn() {
    let events = capture_events(|| {
        let limit = resolve_gpu_retry_limit(None);
        assert_eq!(limit, 3);
    });
    assert!(
        events.is_empty(),
        "an unset variable must not log anything, got {events:?}"
    );
}

/// This is the only test in this binary that touches `gpu_retry_limit()`'s
/// backing `OnceLock`, so the first-touch value is the one set here. Each
/// `tests/*.rs` file runs as its own process, so this is safe in isolation.
#[test]
#[serial]
fn accessor_routes_env_through_resolver() {
    // SAFETY: `#[serial]` prevents other tests in this binary from racing on
    // the environment while this test runs.
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT", "abc");
    }

    let events = capture_events(|| {
        let limit = gpu_retry_limit();
        assert_eq!(limit, 3);
    });

    assert_single_warn_containing(&events, &["NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT"]);

    // SAFETY: see above.
    unsafe {
        std::env::remove_var("NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT");
    }
}
