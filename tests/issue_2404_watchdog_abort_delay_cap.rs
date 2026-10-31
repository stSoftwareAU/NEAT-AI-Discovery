//! Issue #2404 (CWE-1284) — `NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS` must
//! be capped.
//!
//! `watchdog_abort_delay()` used to accept any `u64`, so a huge value turned the
//! watchdog's sleep-then-abort into an indefinite hang. The accessor now clamps
//! to `0..=MAX_WATCHDOG_ABORT_DELAY_SECS` and logs a `WARN` whenever it clamps.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use neat_ai_discovery::config::{MAX_WATCHDOG_ABORT_DELAY_SECS, watchdog_abort_delay};
use serial_test::serial;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

const ENV: &str = "NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS";

/// Set `name` to `value`, run `body`, then restore the previous value.
///
/// Every caller is `#[serial]`, so the process-wide mutation is safe.
fn with_env<T>(name: &str, value: &str, body: impl FnOnce() -> T) -> T {
    let previous = std::env::var(name).ok();
    // SAFETY: Serialised via #[serial].
    unsafe { std::env::set_var(name, value) };
    let result = body();
    restore(name, previous);
    result
}

/// Remove `name`, run `body`, then restore the previous value.
fn without_env<T>(name: &str, body: impl FnOnce() -> T) -> T {
    let previous = std::env::var(name).ok();
    // SAFETY: Serialised via #[serial].
    unsafe { std::env::remove_var(name) };
    let result = body();
    restore(name, previous);
    result
}

fn restore(name: &str, previous: Option<String>) {
    match previous {
        // SAFETY: Serialised via #[serial].
        Some(v) => unsafe { std::env::set_var(name, v) },
        // SAFETY: Serialised via #[serial].
        None => unsafe { std::env::remove_var(name) },
    }
}

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

fn cap() -> Duration {
    Duration::from_secs(MAX_WATCHDOG_ABORT_DELAY_SECS)
}

#[test]
fn cap_is_sixty_seconds() {
    assert_eq!(MAX_WATCHDOG_ABORT_DELAY_SECS, 60);
}

#[test]
#[serial]
fn huge_values_are_clamped_to_the_cap() {
    for huge in ["18446744073709551615", "2000000000"] {
        let delay = with_env(ENV, huge, watchdog_abort_delay);
        assert!(delay <= cap(), "{ENV}={huge} gave {delay:?}, above the cap");
        assert_eq!(delay, cap(), "{ENV}={huge}");
    }
}

#[test]
#[serial]
fn cap_boundary_is_accepted_and_one_above_clamps() {
    assert_eq!(with_env(ENV, "60", watchdog_abort_delay), cap());
    assert_eq!(with_env(ENV, "61", watchdog_abort_delay), cap());
    assert_eq!(
        with_env(ENV, "59", watchdog_abort_delay),
        Duration::from_secs(59)
    );
}

#[test]
#[serial]
fn in_range_values_pass_through() {
    assert_eq!(
        with_env(ENV, "5", watchdog_abort_delay),
        Duration::from_secs(5)
    );
    assert_eq!(with_env(ENV, "0", watchdog_abort_delay), Duration::ZERO);
}

#[test]
#[serial]
fn unset_and_unparsable_default_to_two_seconds() {
    assert_eq!(
        without_env(ENV, watchdog_abort_delay),
        Duration::from_secs(2)
    );
    assert_eq!(
        with_env(ENV, "abc", watchdog_abort_delay),
        Duration::from_secs(2)
    );
}

#[test]
#[serial]
fn clamping_logs_a_warning_naming_variable_value_and_cap() {
    for given in ["18446744073709551615", "2000000000", "61"] {
        let events = capture_events(|| {
            let delay = with_env(ENV, given, watchdog_abort_delay);
            assert_eq!(delay, cap());
        });
        let warnings: Vec<&CapturedEvent> = events.iter().filter(|e| e.level == "WARN").collect();
        assert_eq!(warnings.len(), 1, "{given}: expected one WARN: {events:?}");
        let warning = warnings[0];
        for needle in [ENV, given, &MAX_WATCHDOG_ABORT_DELAY_SECS.to_string()] {
            assert!(
                warning.message.contains(needle),
                "{given}: WARN message missing {needle:?}: {}",
                warning.message
            );
        }
        assert_eq!(warning.fields.get("env").map(String::as_str), Some(ENV));
        assert_eq!(warning.fields.get("value").map(String::as_str), Some(given));
        assert_eq!(warning.fields.get("max").map(String::as_str), Some("60"));
    }
}

#[test]
#[serial]
fn in_range_unset_and_unparsable_values_do_not_warn() {
    for given in [Some("0"), Some("5"), Some("60"), Some("abc"), None] {
        let events = capture_events(|| {
            let _ = match given {
                Some(v) => with_env(ENV, v, watchdog_abort_delay),
                None => without_env(ENV, watchdog_abort_delay),
            };
        });
        let warnings: Vec<&CapturedEvent> = events.iter().filter(|e| e.level == "WARN").collect();
        assert!(
            warnings.is_empty(),
            "{given:?}: unexpected WARN events: {warnings:?}"
        );
    }
}
