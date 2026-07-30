//! Issue #1794 — the drought escape hatch used to log a success line even when
//! it cleared nothing, and it tombstoned the streak either way.
//!
//! These tests are the regression guard for the "fail loud on a no-op reset"
//! behaviour:
//!
//! 1. A reset that clears nothing does **not** stamp `tombstone_reset_epoch`,
//!    so the lever stays armed for the rest of the streak.
//! 2. A no-op reset logs at `ERROR` with wording that names it a no-op and
//!    never claims anything was cleared, carrying structured fields that
//!    distinguish an unwired input (`None`) from a wired-but-empty one.
//! 3. An effective reset keeps the existing success wording and the one-shot
//!    tombstone semantics.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use neat_ai_discovery::analysis::drought_reset::{
    DroughtResetInputState, maybe_perform_drought_reset,
};
use neat_ai_discovery::analysis::target_failure_tracker::TargetFailureTracker;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

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

/// Two targets in cooldown plus one below-threshold target.
fn populated_tracker() -> TargetFailureTracker {
    let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
    tracker.record_failure("A", 0);
    tracker.record_failure("A", 1);
    tracker.record_failure("B", 0);
    tracker.record_failure("B", 1);
    tracker.record_failure("C", 0);
    tracker
}

/// A no-op reset must not consume the one-shot for the streak: neither the
/// unwired (`None`) case nor the wired-but-empty case may stamp a tombstone,
/// and a later pass in the same streak must still be able to do real work.
#[test]
fn noop_reset_does_not_set_tombstone() {
    // Case 1: input not wired at all.
    let unwired = maybe_perform_drought_reset(None, 10, 10, 5).expect("fires with no tracker");
    assert_eq!(unwired.target_cooldown_cleared, 0);
    assert!(unwired.is_noop(), "no tracker means nothing was reset");
    assert_eq!(
        unwired.target_tracker_input,
        DroughtResetInputState::Unwired
    );

    // Case 2: wired, but there is nothing in cooldown to clear.
    let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
    tracker.record_failure("C", 0); // below threshold — not in cooldown
    let empty = maybe_perform_drought_reset(Some(&mut tracker), 10, 10, 5)
        .expect("fires with an empty tracker");
    assert_eq!(empty.target_cooldown_cleared, 0);
    assert!(empty.is_noop());
    assert_eq!(
        empty.target_tracker_input,
        DroughtResetInputState::WiredEmpty
    );
    assert!(
        tracker.drought_reset_tombstone().is_none(),
        "a no-op reset must not tombstone the streak"
    );

    // The streak continues; once cooldowns exist the lever may still fire.
    tracker.record_failure("A", 6);
    tracker.record_failure("A", 7);
    let effective = maybe_perform_drought_reset(Some(&mut tracker), 11, 10, 8)
        .expect("un-tombstoned lever fires again within the same streak");
    assert_eq!(effective.target_cooldown_cleared, 1);
    assert_eq!(tracker.drought_reset_tombstone(), Some(8));
}

/// The no-op log line must name itself a no-op, must not claim it cleared
/// anything, and must distinguish unwired from wired-but-empty inputs.
#[test]
fn noop_reset_log_states_nothing_cleared() {
    let unwired_events = capture_events(|| {
        let _ = maybe_perform_drought_reset(None, 10, 10, 5);
    });
    let unwired = unwired_events
        .iter()
        .find(|e| e.fields.get("noop").is_some_and(|v| v == "true"))
        .expect("a no-op reset emits a distinct no-op event");

    assert_eq!(unwired.level, "ERROR", "a no-op reset must fail loud");
    assert!(
        !unwired.message.contains("cleared"),
        "no-op message must not claim it cleared anything: {}",
        unwired.message
    );
    assert!(
        unwired.message.to_lowercase().contains("no-op"),
        "no-op message must name itself a no-op: {}",
        unwired.message
    );
    assert_eq!(
        unwired
            .fields
            .get("target_tracker_input")
            .map(String::as_str),
        Some("unwired"),
        "an absent tracker must be reported as unwired: {:?}",
        unwired.fields
    );
    assert_eq!(
        unwired.fields.get("tombstone_stamped").map(String::as_str),
        Some("false")
    );

    // Wired but empty is a different diagnosis and must read differently.
    let empty_events = capture_events(|| {
        let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
        tracker.record_failure("C", 0);
        let _ = maybe_perform_drought_reset(Some(&mut tracker), 10, 10, 5);
    });
    let empty = empty_events
        .iter()
        .find(|e| e.fields.get("noop").is_some_and(|v| v == "true"))
        .expect("an empty wired tracker also emits the no-op event");
    assert_eq!(
        empty.fields.get("target_tracker_input").map(String::as_str),
        Some("wired_empty"),
        "a wired-but-empty tracker must be distinguishable from an unwired one: {:?}",
        empty.fields
    );
}

/// A reset that clears something keeps the existing success wording and the
/// one-shot tombstone semantics.
#[test]
fn effective_reset_keeps_success_wording_and_one_shot() {
    let mut tracker = populated_tracker();

    let events = capture_events(|| {
        let outcome =
            maybe_perform_drought_reset(Some(&mut tracker), 10, 10, 5).expect("fires at threshold");
        assert_eq!(outcome.target_cooldown_cleared, 2);
        assert!(!outcome.is_noop());
        assert_eq!(
            outcome.target_tracker_input,
            DroughtResetInputState::Cleared
        );
    });

    let success = events
        .iter()
        .find(|e| {
            e.fields
                .get("reset_name")
                .is_some_and(|v| v == "drought_escape_hatch")
        })
        .expect("an effective reset emits the success event");
    assert_eq!(success.level, "WARN");
    assert!(
        success
            .message
            .contains("drought escape hatch fired — cleared 2 active target cooldowns"),
        "success wording changed: {}",
        success.message
    );
    assert!(
        events.iter().all(|e| !e.fields.contains_key("noop")),
        "an effective reset must not emit the no-op event"
    );

    // One-shot: the tombstone is stamped exactly once and blocks a re-fire.
    assert_eq!(tracker.drought_reset_tombstone(), Some(5));
    tracker.record_failure("D", 6);
    tracker.record_failure("D", 7);
    assert!(
        maybe_perform_drought_reset(Some(&mut tracker), 11, 10, 8).is_none(),
        "the one-shot must not re-fire within the same streak"
    );
    assert_eq!(
        tracker.drought_reset_tombstone(),
        Some(5),
        "the tombstone stays stamped at the firing epoch"
    );
    assert!(
        tracker.state("D").is_some(),
        "the blocked second call must leave the tracker untouched"
    );
}
