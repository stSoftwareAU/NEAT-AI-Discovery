//! Issue #2405: oversized watchdog stall timeouts must not truncate to a
//! shorter effective threshold in `watchdog_loop`.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use neat_ai_discovery::config::watchdog_stall_timeout;
use serial_test::serial;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

const ENV: &str = "NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS";

struct EnvGuard {
    name: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
    fn set(name: &'static str, value: &str) -> Self {
        let previous = std::env::var(name).ok();
        // SAFETY: Every test using this process-wide environment variable is serialised.
        unsafe { std::env::set_var(name, value) };
        Self { name, previous }
    }

    fn unset(name: &'static str) -> Self {
        let previous = std::env::var(name).ok();
        // SAFETY: Every test using this process-wide environment variable is serialised.
        unsafe { std::env::remove_var(name) };
        Self { name, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            // SAFETY: Every test using this process-wide environment variable is serialised.
            Some(value) => unsafe { std::env::set_var(self.name, value) },
            // SAFETY: Every test using this process-wide environment variable is serialised.
            None => unsafe { std::env::remove_var(self.name) },
        }
    }
}

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

fn capture_events<F: FnOnce()>(body: F) -> Vec<CapturedEvent> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let subscriber = Registry::default().with(CaptureLayer {
        events: Arc::clone(&events),
    });
    tracing::subscriber::with_default(subscriber, body);
    events.lock().expect("capture buffer poisoned").clone()
}

fn expected_cap() -> Duration {
    Duration::from_secs(24 * 60 * 60)
}

#[test]
#[serial]
fn oversized_values_keep_the_effective_millisecond_threshold_within_u64() {
    let mut violations = Vec::new();
    for given in [
        "18446744073709552",
        "2305843009213693952",
        "18446744073709551615",
    ] {
        let _env = EnvGuard::set(ENV, given);
        let timeout = watchdog_stall_timeout().expect("positive timeout enables watchdog");
        let milliseconds = timeout.as_millis();
        if milliseconds > u64::MAX as u128
            || timeout != expected_cap()
            || milliseconds as u64 != expected_cap().as_millis() as u64
        {
            violations.push(format!(
                "{ENV}={given} gave {timeout:?} ({milliseconds} ms), expected {:?}",
                expected_cap()
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "unsafe watchdog thresholds: {violations:?}"
    );
}

#[test]
#[serial]
fn ordinary_values_and_cap_boundary_are_preserved() {
    for (given, expected) in [
        ("1", Duration::from_secs(1)),
        ("300", Duration::from_secs(300)),
        ("86399", Duration::from_secs(86_399)),
        ("86400", expected_cap()),
        ("86401", expected_cap()),
    ] {
        let _env = EnvGuard::set(ENV, given);
        assert_eq!(watchdog_stall_timeout(), Some(expected), "{ENV}={given}");
    }
}

#[test]
#[serial]
fn clamping_logs_a_warning_naming_variable_value_and_cap() {
    for given in ["86401", "18446744073709552"] {
        let events = capture_events(|| {
            let _env = EnvGuard::set(ENV, given);
            assert_eq!(watchdog_stall_timeout(), Some(expected_cap()));
        });
        let warnings: Vec<&CapturedEvent> = events.iter().filter(|e| e.level == "WARN").collect();
        assert_eq!(warnings.len(), 1, "{given}: expected one WARN: {events:?}");
        let warning = warnings[0];
        for needle in [ENV, given, &expected_cap().as_secs().to_string()] {
            assert!(
                warning.message.contains(needle),
                "{given}: WARN message missing {needle:?}: {}",
                warning.message
            );
        }
        assert_eq!(warning.fields.get("env").map(String::as_str), Some(ENV));
        assert_eq!(warning.fields.get("value").map(String::as_str), Some(given));
        assert_eq!(warning.fields.get("max").map(String::as_str), Some("86400"));
    }
}

#[test]
#[serial]
fn whitespace_unset_zero_and_invalid_values_keep_existing_behaviour() {
    {
        let _env = EnvGuard::set(ENV, " 45 ");
        assert_eq!(watchdog_stall_timeout(), Some(Duration::from_secs(45)));
    }
    {
        let _env = EnvGuard::set(ENV, "0");
        assert_eq!(watchdog_stall_timeout(), None);
    }
    {
        let _env = EnvGuard::set(ENV, "soon");
        assert_eq!(watchdog_stall_timeout(), None);
    }
    {
        let _env = EnvGuard::unset(ENV);
        assert_eq!(watchdog_stall_timeout(), None);
    }
}
