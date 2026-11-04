//! Issue #2489: dispatch-level check for the `topology_diversification_detection` spec.
//!
//! Invokes the registered `detect_fn` closure directly with a scripted deadline, following the
//! Issue #2368 skip-connection test pattern.

use std::sync::Arc;
use std::time::SystemTime;

use super::append_structural_specs;
use crate::analysis::cache::RecordCache;
use crate::analysis::detection::topology_cache::CreatureTopologyCache;
use crate::analysis::discovery_dispatch::{DiscoveryDetectionResult, DiscoveryModuleSpec};
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;

const PHASE_NAME: &str = "topology_diversification_detection";

/// Samples per neuron, above `MIN_DISCOVERY_SAMPLE_COUNT` (20).
const SAMPLES: u32 = 30;

/// Matches the private `RecordCacheLoader` alias in `cache::mod`.
type TestRecordLoader = dyn Fn(&str, &str) -> anyhow::Result<Vec<DiscoverRecord>> + Send + Sync;

/// Input `in` with a direct synapse to each of two outputs, `out-0` and `out-1`.
fn fan_in_creature() -> CreatureJson {
    let neuron = |uuid: &str, neuron_type: &str| NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "TANH".to_string(),
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
            neuron("in", "input"),
            neuron("out-0", "output"),
            neuron("out-1", "output"),
        ],
        synapses: vec![synapse("in", "out-0"), synapse("in", "out-1")],
        input: 1,
        output: 2,
    }
}

/// In-memory loader: outputs carry mean error 0.25 (above the 0.05 threshold), everything
/// else carries zero error.
fn fan_in_loader() -> Arc<TestRecordLoader> {
    Arc::new(
        |_file: &str, neuron_uuid: &str| -> anyhow::Result<Vec<DiscoverRecord>> {
            let error = if neuron_uuid.starts_with("out-") {
                0.25
            } else {
                0.0
            };
            Ok((0..SAMPLES)
                .map(|i| DiscoverRecord {
                    obs_index: i,
                    neuron_uuid: neuron_uuid.to_string(),
                    value: None,
                    activation: 0.5,
                    errors: vec![error],
                })
                .collect())
        },
    )
}

/// Build the `topology_diversification_detection` spec for `deadline`, then invoke its
/// `detect_fn`. `sequence`, when given, scripts `deadline_passed` for the detect call.
fn run_topology_spec(
    deadline: Option<SystemTime>,
    sequence: Option<Vec<bool>>,
) -> Option<DiscoveryDetectionResult> {
    let creature = Arc::new(fan_in_creature());
    let hidden: Arc<Vec<(String, String, f32)>> = Arc::new(Vec::new());
    let shared_cache = Arc::new(RecordCache::with_loader("test.parquet", fan_in_loader()));
    let topo = Arc::new(CreatureTopologyCache::new(&creature));

    let mut modules: Vec<DiscoveryModuleSpec> = Vec::new();
    append_structural_specs(
        &mut modules,
        &creature,
        &hidden,
        &shared_cache,
        &topo,
        deadline,
    );

    let spec = modules
        .into_iter()
        .find(|s| s.phase_name == PHASE_NAME)
        .expect("topology_diversification_detection spec must be registered");

    let _guard = sequence.map(DeadlineOverrideGuard::with_sequence);
    (spec.detect_fn)()
}

/// Issue #2489: the `deadline` passed into `append_structural_specs` must reach the
/// topology-diversification scan, and a stopped scan must return its partial candidates.
///
/// The loader makes one `deadline_passed` check per neuron while records load, so the
/// scripted sequence starts with one `false` per neuron, then the scan's own checks: `false`
/// before `out-0` (processed) and `true` before `out-1` (stop).
#[test]
#[serial]
fn topology_spec_returns_partial_candidates_when_scan_stops() {
    let control =
        run_topology_spec(None, None).expect("two outputs must be detected without a deadline");
    assert_eq!(control.detected_count, 2);
    assert_eq!(control.candidates.len(), 2);

    let loads = fan_in_creature().neurons.len();
    let mut sequence: Vec<bool> = std::iter::repeat_n(false, loads).collect();
    sequence.extend([false, true]);
    let partial = run_topology_spec(None, Some(sequence))
        .expect("the candidate gathered before the stop must be returned");
    assert_eq!(partial.detected_count, 1);
    assert_eq!(partial.candidates.len(), 1);
}

/// An already-elapsed deadline stops the scan before any output, so the spec returns `None`.
#[test]
#[serial]
fn topology_spec_returns_none_at_elapsed_deadline() {
    assert!(run_topology_spec(Some(SystemTime::UNIX_EPOCH), None).is_none());
}
