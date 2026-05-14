//! Integration tests for Issue #1202: drought diagnostic surfaced on FFI
//! metadata when consecutive empty discovery passes cross the configured
//! threshold.
//!
//! Verifies that:
//!
//! 1. After 6 forced-empty passes (above the default threshold of 5), the
//!    `synapseMetadata.droughtDiagnostic` and `neuronMetadata.droughtDiagnostic`
//!    payloads are populated with the expected counts.
//! 2. Below the threshold (4 trailing failures), the payload is `None`.
//! 3. The `tracing::warn!` event for the drought fires once per `analyze_all`
//!    invocation (verified through subscriber capture).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::analysis::discovery_mode::DiscoveryOutcomeLog;
use neat_ai_discovery::analysis::gpu::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};
use tracing::Level;
use tracing::field::{Field, Visit};
use tracing::subscriber::with_default;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::Registry;

macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

fn build_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 1,
    }
}

fn build_records(creature: &CreatureJson, count: usize) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for obs in 0..count as u32 {
        let t = obs as f32 / count as f32;
        for neuron in &creature.neurons {
            let (activation, value, errors) = match neuron.neuron_type.as_str() {
                "input" => ((t * std::f32::consts::TAU).sin(), None, Vec::new()),
                "hidden" => {
                    let act = 0.3 + 0.4 * (t * std::f32::consts::TAU).sin();
                    (act, Some(act), vec![0.05 * (t * 3.0).cos()])
                }
                "output" => {
                    let error = 0.1 * (t * std::f32::consts::PI).cos();
                    (0.5 + 0.2 * t, Some(0.5 + 0.2 * t), vec![error])
                }
                _ => continue,
            };
            records.push(DiscoverRecord {
                obs_index: obs,
                neuron_uuid: neuron.uuid.clone(),
                value,
                activation,
                errors,
            });
        }
    }
    records
}

fn make_input(parquet_file: String, outcomes: Vec<bool>) -> AnalyzeAllInput {
    let creature = build_creature();
    let focus_neurons: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output" || n.neuron_type == "hidden")
        .map(|n| n.uuid.clone())
        .collect();

    AnalyzeAllInput {
        parquet_file,
        creature,
        focus_neurons,
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: Some(42),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: Some(DiscoveryOutcomeLog::from_outcomes(outcomes)),
    }
}

// =============================================================================
// Tracing capture layer — counts warn events whose message mentions the
// drought diagnostic so we can assert the log fires exactly once per
// invocation.
// =============================================================================

#[derive(Default)]
struct DroughtMessageCapture {
    matches: Mutex<Vec<String>>,
}

impl DroughtMessageCapture {
    fn match_count(&self) -> usize {
        self.matches.lock().unwrap().len()
    }
}

struct DroughtMessageVisitor<'a> {
    matched: &'a mut bool,
    message: &'a mut String,
}

impl<'a> Visit for DroughtMessageVisitor<'a> {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            *self.message = value.to_string();
            if value.contains("Issue #1202") {
                *self.matched = true;
            }
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let formatted = format!("{value:?}");
            *self.message = formatted.clone();
            if formatted.contains("Issue #1202") {
                *self.matched = true;
            }
        }
    }
}

struct DroughtCaptureLayer {
    capture: Arc<DroughtMessageCapture>,
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for DroughtCaptureLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        if *event.metadata().level() != Level::WARN {
            return;
        }
        let mut matched = false;
        let mut message = String::new();
        let mut visitor = DroughtMessageVisitor {
            matched: &mut matched,
            message: &mut message,
        };
        event.record(&mut visitor);
        if matched {
            self.capture.matches.lock().unwrap().push(message);
        }
    }
}

fn make_subscriber(capture: Arc<DroughtMessageCapture>) -> impl tracing::Subscriber {
    Registry::default().with(DroughtCaptureLayer { capture })
}

// =============================================================================
// Tests
// =============================================================================

/// Six consecutive forced-empty passes (above the default threshold of 5)
/// causes both metadata surfaces to carry a populated `droughtDiagnostic`.
#[test]
fn six_empty_passes_populates_drought_diagnostic_on_metadata() {
    skip_without_gpu!();

    let creature = build_creature();
    let records = build_records(&creature, 60);
    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("drought.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let outcomes = vec![false; 6];
    let input = make_input(parquet_file, outcomes);

    let capture = Arc::new(DroughtMessageCapture::default());
    let subscriber = make_subscriber(Arc::clone(&capture));
    let result = with_default(subscriber, || analyze_all(&input).expect("analyze_all"));

    let synapse = result.synapse.expect("synapse analysis runs");
    let neuron = result.neuron.expect("neuron analysis runs");

    let syn_diag = synapse
        .metadata
        .drought_diagnostic
        .as_ref()
        .expect("synapse metadata carries droughtDiagnostic");
    let neu_diag = neuron
        .metadata
        .drought_diagnostic
        .as_ref()
        .expect("neuron metadata carries droughtDiagnostic");

    assert_eq!(syn_diag.consecutive_failures, 6);
    assert_eq!(neu_diag.consecutive_failures, 6);
    assert!(
        (syn_diag.rolling_success_rate).abs() < 1e-6,
        "rolling rate over 6 failures should be 0.0"
    );
    // Both surfaces share the same payload contents.
    assert_eq!(syn_diag, neu_diag);

    // The structured warn must have fired exactly once for this invocation.
    assert_eq!(
        capture.match_count(),
        1,
        "expected exactly one drought warn log per analyze_all call, got {}",
        capture.match_count()
    );
}

/// Below the threshold (4 trailing failures), no diagnostic is attached and
/// no warn log is emitted.
#[test]
fn four_empty_passes_does_not_trigger_drought_diagnostic() {
    skip_without_gpu!();

    let creature = build_creature();
    let records = build_records(&creature, 60);
    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("no-drought.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let outcomes = vec![false; 4];
    let input = make_input(parquet_file, outcomes);

    let capture = Arc::new(DroughtMessageCapture::default());
    let subscriber = make_subscriber(Arc::clone(&capture));
    let result = with_default(subscriber, || analyze_all(&input).expect("analyze_all"));

    let synapse = result.synapse.expect("synapse analysis runs");
    let neuron = result.neuron.expect("neuron analysis runs");

    assert!(
        synapse.metadata.drought_diagnostic.is_none(),
        "synapse droughtDiagnostic must be None below threshold"
    );
    assert!(
        neuron.metadata.drought_diagnostic.is_none(),
        "neuron droughtDiagnostic must be None below threshold"
    );
    assert_eq!(
        capture.match_count(),
        0,
        "no drought warn should fire below threshold"
    );
}
