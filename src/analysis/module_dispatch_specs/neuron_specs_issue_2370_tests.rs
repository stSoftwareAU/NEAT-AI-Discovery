//! Regression test for Issue #2370: `append_neuron_specs` must forward its
//! `deadline` argument into the dead-neuron scan.
//!
//! The `dead_neuron_detection` closure built in `neuron_specs.rs` calls
//! `dead_neuron::detect_dead_neurons_with_deadline(&creature, &records,
//! Some(&topo), &deadline)`. This test invokes the registered closure directly,
//! bypassing the dispatch-level pre-check, with an already-elapsed deadline, so
//! only the in-scan deadline check can stop it. If the call is reverted to the
//! deadline-less `detect_dead_neurons`, or passed `&None`, this test goes red.

use std::sync::Arc;
use std::time::SystemTime;

use super::append_neuron_specs;
use crate::analysis::cache::RecordCache;
use crate::analysis::cost_function_hint::CostFunctionHint;
use crate::analysis::detection::topology_cache::CreatureTopologyCache;
use crate::analysis::discovery_dispatch::{DiscoveryDetectionResult, DiscoveryModuleSpec};
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};

const DEAD_NEURON_SPEC_NAME: &str = "dead_neuron_detection";

/// Matches the private `RecordCacheLoader` alias in `cache::mod` (not exported,
/// so the test redeclares it locally).
type TestRecordLoader = dyn Fn(&str, &str) -> anyhow::Result<Vec<DiscoverRecord>> + Send + Sync;

/// `input-0 -> hidden-0 -> output-0`, with `hidden-0` dead.
fn dead_hidden_creature() -> CreatureJson {
    let neuron = |uuid: &str, neuron_type: &str| NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "LOGISTIC".to_string(),
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
            neuron("input-0", "input"),
            neuron("hidden-0", "hidden"),
            neuron("output-0", "output"),
        ],
        synapses: vec![
            synapse("input-0", "hidden-0"),
            synapse("hidden-0", "output-0"),
        ],
        input: 1,
        output: 1,
    }
}

/// An in-memory loader returning 30 zero-activation records for any neuron,
/// enough samples for the dead-neuron detector.
fn zero_activation_loader() -> Arc<TestRecordLoader> {
    Arc::new(
        |_file: &str, neuron_uuid: &str| -> anyhow::Result<Vec<DiscoverRecord>> {
            Ok((0..30u32)
                .map(|i| DiscoverRecord::new(i, neuron_uuid.to_string(), Some(0.0), 0.0, vec![0.0]))
                .collect())
        },
    )
}

/// Build the `dead_neuron_detection` spec for the given deadline and invoke
/// its `detect_fn` directly, bypassing the dispatch-level pre-check so only the
/// in-scan deadline check can stop it.
fn run_dead_neuron_spec(deadline: Option<SystemTime>) -> Option<DiscoveryDetectionResult> {
    let creature = Arc::new(dead_hidden_creature());
    let hidden: Arc<Vec<(String, String, f32)>> =
        Arc::new(vec![("hidden-0".to_string(), "LOGISTIC".to_string(), 0.0)]);
    let shared_cache = Arc::new(RecordCache::with_loader(
        "test.parquet",
        zero_activation_loader(),
    ));
    let topo = Arc::new(CreatureTopologyCache::new(&creature));

    let mut modules: Vec<DiscoveryModuleSpec> = Vec::new();
    append_neuron_specs(
        &mut modules,
        &creature,
        &hidden,
        &shared_cache,
        &topo,
        CostFunctionHint::Unknown,
        deadline,
    );

    let spec = modules
        .into_iter()
        .find(|s| s.phase_name == DEAD_NEURON_SPEC_NAME)
        .expect("dead_neuron_detection spec must be registered");

    (spec.detect_fn)()
}

/// Issue #2370: the `deadline` passed into `append_neuron_specs` must reach the
/// dead-neuron scan. With no deadline the dead hidden neuron is detected
/// (positive precondition). With an already-elapsed deadline the in-scan check
/// stops the scan before any candidate is produced.
#[test]
fn dead_neuron_spec_stops_at_elapsed_deadline() {
    let result = run_dead_neuron_spec(None)
        .expect("a dead hidden neuron must be detected without a deadline");
    assert!(
        result.detected_count >= 1,
        "expected at least one dead neuron, got {}",
        result.detected_count
    );

    assert!(
        run_dead_neuron_spec(Some(SystemTime::UNIX_EPOCH)).is_none(),
        "an already-elapsed deadline must stop the dead-neuron scan before any candidate is produced"
    );
}
