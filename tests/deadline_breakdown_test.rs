//! Tests for the consolidated per-cycle deadline-consumption breakdown
//! (Issue #1409).
//!
//! These verify the observable behaviour of the breakdown: the greppable
//! summary line, the explicit `STARVED` warning emitted when synapse/neuron
//! analysis is curtailed by the deadline, and the starvation predicate that the
//! structured analysis result surfaces to the calling layer.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

use neat_ai_discovery::analysis::deadline_breakdown::{
    DEADLINE_BREAKDOWN_MARKER, DeadlineConsumptionBreakdown, PhaseCompletion,
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum CapturedValue {
    String(String),
    U64(u64),
    Debug(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedEvent {
    level: String,
    message: String,
    fields: HashMap<String, CapturedValue>,
}

#[derive(Default)]
struct EventVisitor {
    message: String,
    fields: HashMap<String, CapturedValue>,
}

impl EventVisitor {
    fn store(&mut self, field: &Field, value: CapturedValue) {
        if field.name() == "message" {
            self.message = match value {
                CapturedValue::String(message) | CapturedValue::Debug(message) => message,
                CapturedValue::U64(value) => value.to_string(),
            };
        } else {
            self.fields.insert(field.name().to_string(), value);
        }
    }
}

impl Visit for EventVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.store(field, CapturedValue::Debug(format!("{value:?}")));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.store(field, CapturedValue::String(value.to_string()));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.store(field, CapturedValue::U64(value));
    }
}

#[derive(Clone)]
struct EventCapture(Arc<Mutex<Vec<CapturedEvent>>>);

impl<S: tracing::Subscriber> Layer<S> for EventCapture {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);
        self.0
            .lock()
            .expect("event capture buffer must not be poisoned")
            .push(CapturedEvent {
                level: event.metadata().level().to_string(),
                message: visitor.message,
                fields: visitor.fields,
            });
    }
}

fn capture_events(body: impl FnOnce()) -> Vec<CapturedEvent> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let subscriber = Registry::default().with(EventCapture(Arc::clone(&events)));
    tracing::subscriber::with_default(subscriber, body);
    events
        .lock()
        .expect("event capture buffer must not be poisoned")
        .clone()
}

/// A phase that finished all its focus neurons is never starved, even if the
/// `timed_out` flag was set after the last neuron completed.
#[test]
fn phase_completion_not_starved_when_all_completed() {
    let phase = PhaseCompletion {
        timed_out: true,
        completed_focus_neurons: 8,
        total_focus_neurons: 8,
    };
    assert!(!phase.is_starved());
    assert_eq!(phase.skipped(), 0);
}

/// A phase that timed out before reaching every focus neuron is starved.
#[test]
fn phase_completion_starved_when_timed_out_with_remaining() {
    let phase = PhaseCompletion {
        timed_out: true,
        completed_focus_neurons: 3,
        total_focus_neurons: 10,
    };
    assert!(phase.is_starved());
    assert_eq!(phase.skipped(), 7);
}

/// Partial completion without a timeout is not starvation — the phase simply
/// found nothing more to do.
#[test]
fn phase_completion_not_starved_without_timeout() {
    let phase = PhaseCompletion {
        timed_out: false,
        completed_focus_neurons: 3,
        total_focus_neurons: 10,
    };
    assert!(!phase.is_starved());
}

/// On a simulated timeout the breakdown reports starvation and the warning
/// names the skipped/total counts for each curtailed phase.
#[test]
fn simulated_timeout_emits_starved_warning_with_counts() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 4200,
        synapse_analysis_ms: Some(1200),
        neuron_analysis_ms: Some(900),
        total_analysis_ms: 6300,
        synapse: Some(PhaseCompletion {
            timed_out: true,
            completed_focus_neurons: 2,
            total_focus_neurons: 12,
        }),
        neuron: Some(PhaseCompletion {
            timed_out: true,
            completed_focus_neurons: 1,
            total_focus_neurons: 12,
        }),
    };

    assert!(breakdown.is_starved());

    let warning = breakdown
        .starvation_warning()
        .expect("a curtailed run must produce a STARVED warning");
    assert!(warning.contains("STARVED"), "warning: {warning}");
    assert!(
        warning.contains("synapse analysis skipped 10/12 targets"),
        "warning: {warning}"
    );
    assert!(
        warning.contains("neuron analysis skipped 11/12 targets"),
        "warning: {warning}"
    );

    // Emission must not panic.
    breakdown.emit();
}

/// Only the curtailed phase is named when the other phase completed in time.
#[test]
fn starved_warning_only_names_curtailed_phase() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 100,
        synapse_analysis_ms: Some(50),
        neuron_analysis_ms: Some(50),
        total_analysis_ms: 200,
        synapse: Some(PhaseCompletion {
            timed_out: true,
            completed_focus_neurons: 4,
            total_focus_neurons: 9,
        }),
        neuron: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 9,
            total_focus_neurons: 9,
        }),
    };

    let warning = breakdown
        .starvation_warning()
        .expect("synapse was curtailed");
    assert!(warning.contains("synapse analysis skipped 5/9 targets"));
    assert!(
        !warning.contains("neuron analysis skipped"),
        "neuron completed in time: {warning}"
    );
}

/// A run that completes within the deadline produces no warning.
#[test]
fn completed_run_has_no_starvation_warning() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 300,
        synapse_analysis_ms: Some(800),
        neuron_analysis_ms: Some(700),
        total_analysis_ms: 1800,
        synapse: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 16,
            total_focus_neurons: 16,
        }),
        neuron: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 16,
            total_focus_neurons: 16,
        }),
    };

    assert!(!breakdown.is_starved());
    assert!(breakdown.starvation_warning().is_none());
    breakdown.emit();
}

/// The consolidated summary line spans every phase the analysis call owns and
/// stays greppable via the stable `DEADLINE-BREAKDOWN` marker.
///
/// The literal is asserted here on purpose: the marker is a published log
/// contract that operator greps depend on, so a silent rename must fail.
#[test]
fn summary_line_attributes_all_phases() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 4200,
        synapse_analysis_ms: Some(1200),
        neuron_analysis_ms: Some(900),
        total_analysis_ms: 6300,
        synapse: Some(PhaseCompletion {
            timed_out: true,
            completed_focus_neurons: 2,
            total_focus_neurons: 12,
        }),
        neuron: Some(PhaseCompletion {
            timed_out: true,
            completed_focus_neurons: 1,
            total_focus_neurons: 12,
        }),
    };

    let line = breakdown.summary_line();
    assert!(line.contains("DEADLINE-BREAKDOWN"), "line: {line}");
    assert!(line.contains("parquet_reload=4200"), "line: {line}");
    assert!(line.contains("synapse_analysis=1200"), "line: {line}");
    assert!(line.contains("neuron_analysis=900"), "line: {line}");
    assert!(line.contains("total_analysis=6300"), "line: {line}");
    assert!(line.contains("synapse=2/12"), "line: {line}");
    assert!(line.contains("neuron=1/12"), "line: {line}");
}

/// Disabled phases render as `n/a` rather than fabricated zeroes, so the
/// breakdown does not imply a phase ran when it was switched off.
#[test]
fn summary_line_marks_disabled_phases_as_not_applicable() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 500,
        synapse_analysis_ms: Some(1000),
        neuron_analysis_ms: None,
        total_analysis_ms: 1600,
        synapse: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 5,
            total_focus_neurons: 5,
        }),
        neuron: None,
    };

    let line = breakdown.summary_line();
    assert!(line.contains("neuron_analysis=n/a"), "line: {line}");
    assert!(line.contains("neuron=n/a"), "line: {line}");
    assert!(!breakdown.is_starved());
    assert!(breakdown.starvation_warning().is_none());
}

#[test]
fn emit_logs_completed_breakdown_with_numeric_fields_and_summary() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 4200,
        synapse_analysis_ms: Some(1200),
        neuron_analysis_ms: Some(900),
        total_analysis_ms: 6300,
        synapse: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 2,
            total_focus_neurons: 12,
        }),
        neuron: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 1,
            total_focus_neurons: 7,
        }),
    };

    let events = capture_events(|| breakdown.emit());

    assert_eq!(events.len(), 1, "completed work emits one INFO event");
    assert_eq!(events[0].level, "INFO");
    assert_eq!(
        events[0].message,
        "DEADLINE-BREAKDOWN deadline consumption (ms): parquet_reload=4200 \
         synapse_analysis=1200 neuron_analysis=900 total_analysis=6300; \
         focus completed synapse=2/12 neuron=1/7"
    );
    assert_eq!(
        events[0].fields,
        HashMap::from([
            (
                "marker".to_string(),
                CapturedValue::String(DEADLINE_BREAKDOWN_MARKER.to_string()),
            ),
            ("parquet_reload_ms".to_string(), CapturedValue::U64(4200)),
            ("synapse_analysis_ms".to_string(), CapturedValue::U64(1200)),
            ("neuron_analysis_ms".to_string(), CapturedValue::U64(900)),
            ("total_analysis_ms".to_string(), CapturedValue::U64(6300)),
            ("synapse_completed".to_string(), CapturedValue::U64(2)),
            ("synapse_total".to_string(), CapturedValue::U64(12)),
            ("neuron_completed".to_string(), CapturedValue::U64(1)),
            ("neuron_total".to_string(), CapturedValue::U64(7)),
        ])
    );
}

#[test]
fn emit_logs_starved_breakdown_and_warning() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 100,
        synapse_analysis_ms: Some(50),
        neuron_analysis_ms: Some(50),
        total_analysis_ms: 200,
        synapse: Some(PhaseCompletion {
            timed_out: true,
            completed_focus_neurons: 4,
            total_focus_neurons: 9,
        }),
        neuron: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 6,
            total_focus_neurons: 6,
        }),
    };

    let events = capture_events(|| breakdown.emit());

    assert_eq!(events.len(), 2, "starvation emits INFO and WARN events");
    assert_eq!(events[0].level, "INFO");
    assert_eq!(
        events[0].message,
        "DEADLINE-BREAKDOWN deadline consumption (ms): parquet_reload=100 \
         synapse_analysis=50 neuron_analysis=50 total_analysis=200; \
         focus completed synapse=4/9 neuron=6/6"
    );
    assert_eq!(
        events[0].fields["marker"],
        CapturedValue::String(DEADLINE_BREAKDOWN_MARKER.to_string())
    );
    assert_eq!(events[0].fields["synapse_completed"], CapturedValue::U64(4));
    assert_eq!(events[0].fields["synapse_total"], CapturedValue::U64(9));
    assert_eq!(events[0].fields["neuron_completed"], CapturedValue::U64(6));
    assert_eq!(events[0].fields["neuron_total"], CapturedValue::U64(6));
    assert_eq!(
        events[0].fields.len(),
        9,
        "INFO fields stay on the numeric contract"
    );

    assert_eq!(events[1].level, "WARN");
    assert_eq!(
        events[1].message,
        "STARVED: synapse analysis skipped 5/9 targets — deadline exhausted by focus+parquet before analysis could finish"
    );
    assert_eq!(
        events[1].fields,
        HashMap::from([(
            "marker".to_string(),
            CapturedValue::String(DEADLINE_BREAKDOWN_MARKER.to_string()),
        )])
    );
}

#[test]
fn emit_logs_disabled_phase_as_not_applicable_without_warning() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 500,
        synapse_analysis_ms: Some(1000),
        neuron_analysis_ms: None,
        total_analysis_ms: 1600,
        synapse: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 5,
            total_focus_neurons: 5,
        }),
        neuron: None,
    };

    let events = capture_events(|| breakdown.emit());

    assert_eq!(
        events.len(),
        1,
        "disabled analysis emits no starvation warning"
    );
    assert_eq!(events[0].level, "INFO");
    assert_eq!(
        events[0].message,
        "DEADLINE-BREAKDOWN deadline consumption (ms): parquet_reload=500 \
         synapse_analysis=1000 neuron_analysis=n/a total_analysis=1600; \
         focus completed synapse=5/5 neuron=n/a"
    );
    assert_eq!(
        events[0].fields["marker"],
        CapturedValue::String(DEADLINE_BREAKDOWN_MARKER.to_string())
    );
    assert_eq!(
        events[0].fields["synapse_analysis_ms"],
        CapturedValue::U64(1000)
    );
    assert_eq!(events[0].fields["synapse_completed"], CapturedValue::U64(5));
    assert_eq!(events[0].fields["synapse_total"], CapturedValue::U64(5));
    assert_eq!(
        events[0].fields["neuron_analysis_ms"],
        CapturedValue::U64(0)
    );
    assert_eq!(events[0].fields["neuron_completed"], CapturedValue::U64(0));
    assert_eq!(events[0].fields["neuron_total"], CapturedValue::U64(0));
    assert_eq!(
        events[0].fields.len(),
        9,
        "INFO fields stay on the numeric contract"
    );
}
