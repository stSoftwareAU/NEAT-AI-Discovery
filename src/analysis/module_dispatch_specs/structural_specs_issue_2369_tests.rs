//! Regression test for Issue #2369: `append_structural_specs` must forward its
//! `deadline` argument into the topology structure scan.
//!
//! Before this issue, the `topology_structure_analysis` closure built in
//! `structural_specs.rs` called `topology::detect_topology_issues` (no deadline).
//! It now calls `topology::detect_topology_issues_with_deadline(&creature,
//! &records, Some(&topo), &deadline)`. This test invokes the registered closure
//! directly (bypassing the dispatch-level pre-check) with an already-elapsed
//! deadline so only the in-scan deadline check can stop it. If the call is ever
//! reverted to the deadline-less wrapper, this test goes red.

use std::sync::Arc;
use std::time::SystemTime;

use super::append_structural_specs;
use crate::analysis::cache::RecordCache;
use crate::analysis::detection::topology_cache::CreatureTopologyCache;
use crate::analysis::discovery_dispatch::DiscoveryModuleSpec;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};

const TOPOLOGY_SPEC_NAME: &str = "topology_structure_analysis";

/// Matches the private `RecordCacheLoader` alias in `cache::mod` (not exported,
/// so the test redeclares it locally).
type TestRecordLoader = dyn Fn(&str, &str) -> anyhow::Result<Vec<DiscoverRecord>> + Send + Sync;

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight: 0.5,
        synapse_type: None,
    }
}

/// `in-0 -> hidden-0 -> tail-1 -> tail-2 -> tail-3 -> out-0`: `hidden-0` is four
/// hops from the output, beyond the three-hop limit, so it is a long-path
/// candidate once it has enough recorded samples.
fn long_path_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("in-0", "input"),
            neuron("hidden-0", "hidden"),
            neuron("tail-1", "hidden"),
            neuron("tail-2", "hidden"),
            neuron("tail-3", "hidden"),
            neuron("out-0", "output"),
        ],
        synapses: vec![
            synapse("in-0", "hidden-0"),
            synapse("hidden-0", "tail-1"),
            synapse("tail-1", "tail-2"),
            synapse("tail-2", "tail-3"),
            synapse("tail-3", "out-0"),
        ],
        input: 1,
        output: 1,
    }
}

/// Loader that gives `hidden-0` 20 records with non-zero errors (the minimum
/// that qualifies for topology detection) and every other neuron no records.
fn topology_loader() -> Arc<TestRecordLoader> {
    Arc::new(
        |_file: &str, neuron_uuid: &str| -> anyhow::Result<Vec<DiscoverRecord>> {
            if neuron_uuid != "hidden-0" {
                return Ok(Vec::new());
            }
            Ok((0..20u32)
                .map(|i| DiscoverRecord::new(i, neuron_uuid.to_string(), Some(0.0), 0.2, vec![0.2]))
                .collect())
        },
    )
}

/// Build the `topology_structure_analysis` spec for the given deadline and invoke
/// its `detect_fn` directly, bypassing the dispatch-level pre-check so only the
/// in-scan deadline check can stop it.
fn run_topology_spec(
    deadline: Option<SystemTime>,
) -> Option<crate::analysis::discovery_dispatch::DiscoveryDetectionResult> {
    let creature = Arc::new(long_path_creature());
    // The spec's guard requires a non-empty hidden list.
    let hidden: Arc<Vec<(String, String, f32)>> =
        Arc::new(vec![("hidden-0".to_string(), "hidden".to_string(), 0.0)]);
    let shared_cache = Arc::new(RecordCache::with_loader("test.parquet", topology_loader()));
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
        .find(|s| s.phase_name == TOPOLOGY_SPEC_NAME)
        .expect("topology_structure_analysis spec must be registered");

    (spec.detect_fn)()
}

/// Issue #2369: the `deadline` passed into `append_structural_specs` must reach
/// the topology scan. With no deadline, the long-path fixture must detect at
/// least one candidate (positive precondition); with an already-elapsed deadline,
/// the in-scan check must stop the scan before any candidate is produced.
#[test]
fn topology_spec_stops_at_elapsed_deadline() {
    let without_deadline = run_topology_spec(None);
    let result =
        without_deadline.expect("the long-path neuron must be detected without a deadline");
    assert!(
        result.detected_count >= 1,
        "expected at least one topology candidate, got {}",
        result.detected_count
    );

    let with_elapsed_deadline = run_topology_spec(Some(SystemTime::UNIX_EPOCH));
    assert!(
        with_elapsed_deadline.is_none(),
        "an already-elapsed deadline must stop the topology scan before any candidate is produced"
    );
}
