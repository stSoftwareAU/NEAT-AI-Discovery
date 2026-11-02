//! Regression test for Issue #2369 deadline forwarding through the topology spec.

use std::sync::Arc;
use std::time::SystemTime;

use serial_test::serial;

use super::append_structural_specs;
use crate::analysis::cache::RecordCache;
use crate::analysis::detection::topology_cache::CreatureTopologyCache;
use crate::analysis::discovery_dispatch::DiscoveryModuleSpec;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};

const TOPOLOGY_SPEC_NAME: &str = "topology_structure_analysis";

type TestRecordLoader = dyn Fn(&str, &str) -> anyhow::Result<Vec<DiscoverRecord>> + Send + Sync;

fn topology_fixture() -> CreatureJson {
    let neurons = (0..5)
        .map(|i| NeuronJson {
            uuid: format!("h{i}"),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        })
        .chain(std::iter::once(NeuronJson {
            uuid: "out".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }))
        .collect();
    let synapses = (0..4)
        .map(|i| SynapseJson {
            from_uuid: format!("h{i}"),
            to_uuid: format!("h{}", i + 1),
            weight: 0.5,
            synapse_type: None,
        })
        .chain(std::iter::once(SynapseJson {
            from_uuid: "h4".to_string(),
            to_uuid: "out".to_string(),
            weight: 0.5,
            synapse_type: None,
        }))
        .collect();
    CreatureJson {
        neurons,
        synapses,
        input: 0,
        output: 1,
    }
}

fn record_loader() -> Arc<TestRecordLoader> {
    Arc::new(|_file, uuid| {
        Ok((0..20)
            .map(|i| DiscoverRecord::new(i, uuid.to_string(), Some(0.5), 0.5, vec![0.2]))
            .collect())
    })
}

fn run_topology_spec(
    deadline: Option<SystemTime>,
) -> Option<crate::analysis::discovery_dispatch::DiscoveryDetectionResult> {
    let creature = Arc::new(topology_fixture());
    let hidden = Arc::new(vec![("h0".to_string(), "IDENTITY".to_string(), 0.0)]);
    let cache = Arc::new(RecordCache::with_loader("test.parquet", record_loader()));
    let topology = Arc::new(CreatureTopologyCache::new(&creature));
    let mut modules: Vec<DiscoveryModuleSpec> = Vec::new();
    append_structural_specs(
        &mut modules,
        &creature,
        &hidden,
        &cache,
        &topology,
        deadline,
    );
    let spec = modules
        .into_iter()
        .find(|spec| spec.phase_name == TOPOLOGY_SPEC_NAME)
        .expect("topology structure spec must be registered");
    (spec.detect_fn)()
}

#[test]
#[serial]
fn topology_spec_forwards_elapsed_deadline_to_detector() {
    let without_deadline = run_topology_spec(None)
        .expect("the long-path fixture must produce candidates without a deadline");
    assert!(without_deadline.detected_count > 0);

    assert!(
        run_topology_spec(Some(SystemTime::UNIX_EPOCH)).is_none(),
        "the production topology spec must stop when its analysis deadline has elapsed"
    );
}
