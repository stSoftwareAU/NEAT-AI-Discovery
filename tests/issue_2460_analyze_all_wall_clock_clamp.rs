//! Issue #2460: exercise the clamp through the shipped analysis entry point.

mod common;

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::config::effective_wall_clock_minutes;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson};
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

#[derive(Debug, Clone)]
struct Event {
    level: tracing::Level,
    fields: HashMap<String, String>,
}

#[derive(Default)]
struct Visitor(HashMap<String, String>);

impl Visit for Visitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.0.insert(field.name().into(), format!("{value:?}"));
    }
}

struct Capture(Arc<Mutex<Vec<Event>>>);

impl<S: tracing::Subscriber> Layer<S> for Capture {
    fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
        let mut visitor = Visitor::default();
        event.record(&mut visitor);
        self.0.lock().unwrap().push(Event {
            level: *event.metadata().level(),
            fields: visitor.0,
        });
    }
}

fn capture(body: impl FnOnce()) -> Vec<Event> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let subscriber = Registry::default().with(Capture(Arc::clone(&events)));
    tracing::subscriber::with_default(subscriber, body);
    events.lock().unwrap().clone()
}

#[test]
fn clamp_warns_only_when_value_changes() {
    for (requested, expected) in [(Some(u64::MAX), 120), (Some(0), 1)] {
        let events = capture(|| assert_eq!(effective_wall_clock_minutes(requested), expected));
        let warnings: Vec<_> = events
            .iter()
            .filter(|e| e.level == tracing::Level::WARN)
            .collect();
        assert_eq!(warnings.len(), 1, "{events:?}");
        assert_eq!(
            warnings[0].fields.get("requested_minutes"),
            Some(&requested.unwrap().to_string())
        );
        assert_eq!(
            warnings[0].fields.get("clamped_minutes"),
            Some(&expected.to_string())
        );
    }
    for requested in [Some(45), None] {
        let events = capture(|| {
            effective_wall_clock_minutes(requested);
        });
        assert!(
            events.iter().all(|e| e.level != tracing::Level::WARN),
            "{events:?}"
        );
    }
}

#[test]
fn analyze_all_reports_clamped_cap_for_oversized_values() {
    skip_without_gpu!();
    for requested in [307_445_734_561_825_861, u64::MAX] {
        let input = AnalyzeAllInput {
            parquet_file: "nonexistent.parquet".into(),
            creature: CreatureJson {
                neurons: vec![],
                synapses: vec![],
                input: 1,
                output: 1,
            },
            focus_neurons: vec!["output-0".into()],
            max_synapse_candidates: None,
            max_neuron_candidates: None,
            analysis_deadline_ms: None,
            include_synapse_analysis: Some(true),
            include_neuron_analysis: Some(false),
            random_seed: None,
            previous_neuron_fingerprints: None,
            module_outcome_tracker: None,
            max_analysis_memory_mb: None,
            max_discovery_wall_clock_minutes: Some(requested),
            temperature: 1.0,
            failure_cache: None,
            discovery_outcome_log: None,
            cost_name: None,
        };
        let events = capture(|| {
            let _ = analyze_all(&input);
        });
        assert!(
            events.iter().any(|e| e
                .fields
                .get("wall_clock_cap_minutes")
                .is_some_and(|v| v == "120")),
            "requested {requested}: {events:?}"
        );
    }
}
