//! Regression test for Issue #2368: `append_structural_specs` must forward its
//! `deadline` argument into the skip-connection scan.
//!
//! The `skip_connection_discovery` closure built in `structural_specs.rs` calls
//! `skip_connection::detect_skip_connection_candidates_with_deadline(&creature,
//! &records, &deadline)`. This test invokes the registered closure directly,
//! bypassing the dispatch-level pre-check, with an already-elapsed deadline. Only the
//! in-scan deadline check can then stop it. If the call is ever reverted to the
//! deadline-less wrapper, this test goes red.

#![allow(clippy::cast_precision_loss)] // Intentional numeric cast for fixture noise (Issue #873)

use std::sync::Arc;
use std::time::SystemTime;

use super::append_structural_specs;
use crate::analysis::cache::RecordCache;
use crate::analysis::detection::topology_cache::CreatureTopologyCache;
use crate::analysis::discovery_dispatch::DiscoveryModuleSpec;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};

const SKIP_CONNECTION_SPEC_NAME: &str = "skip_connection_discovery";

/// Matches the private `RecordCacheLoader` alias in `cache::mod` (not exported,
/// so the test redeclares it locally).
type TestRecordLoader = dyn Fn(&str, &str) -> anyhow::Result<Vec<DiscoverRecord>> + Send + Sync;

/// Deep chain `input-0 -> h0 -> h1 -> h2 -> h3 -> h4 -> output-0`, mirroring
/// `deep_network_creature` in `tests/recommendation/issue_570_skip_connection_discovery.rs`.
fn deep_chain_creature() -> CreatureJson {
    let hidden = ["h0", "h1", "h2", "h3", "h4"];
    let mut neurons = vec![NeuronJson {
        uuid: "input-0".to_string(),
        neuron_type: "input".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }];
    neurons.extend(hidden.iter().map(|uuid| NeuronJson {
        uuid: (*uuid).to_string(),
        neuron_type: "hidden".to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
    }));
    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
    });

    let chain = ["input-0", "h0", "h1", "h2", "h3", "h4", "output-0"];
    let synapses = chain
        .windows(2)
        .map(|pair| SynapseJson {
            from_uuid: pair[0].to_string(),
            to_uuid: pair[1].to_string(),
            weight: 0.5,
            synapse_type: None,
        })
        .collect();

    CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    }
}

/// An in-memory loader. Hidden `h0`/`h1` (shallow) carry high error, `h2`-`h4` (deep)
/// carry low error, so the deep neurons are gradient-attenuated. Non-hidden neurons get
/// neutral records. Each neuron has 30 samples, above `MIN_DISCOVERY_SAMPLE_COUNT` (20).
fn deep_chain_loader() -> Arc<TestRecordLoader> {
    Arc::new(
        |_file: &str, neuron_uuid: &str| -> anyhow::Result<Vec<DiscoverRecord>> {
            let error = match neuron_uuid {
                "h0" | "h1" => 0.5,
                "h2" | "h3" | "h4" => 0.05,
                _ => 0.0,
            };
            let records: Vec<DiscoverRecord> = (0..30u32)
                .map(|i| {
                    DiscoverRecord::new(i, neuron_uuid.to_string(), Some(0.5), 0.5, vec![error])
                })
                .collect();
            Ok(records)
        },
    )
}

/// Build the `skip_connection_discovery` spec for the given deadline and invoke its
/// `detect_fn` directly, bypassing the dispatch-level pre-check so only the in-scan
/// deadline check can stop it.
fn run_skip_connection_spec(
    deadline: Option<SystemTime>,
) -> Option<crate::analysis::discovery_dispatch::DiscoveryDetectionResult> {
    let creature = Arc::new(deep_chain_creature());
    let hidden: Arc<Vec<(String, String, f32)>> = Arc::new(vec![
        ("h0".to_string(), "TANH".to_string(), 0.0),
        ("h1".to_string(), "TANH".to_string(), 0.0),
        ("h2".to_string(), "TANH".to_string(), 0.0),
        ("h3".to_string(), "TANH".to_string(), 0.0),
        ("h4".to_string(), "TANH".to_string(), 0.0),
    ]);
    let shared_cache = Arc::new(RecordCache::with_loader(
        "test.parquet",
        deep_chain_loader(),
    ));
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
        .find(|s| s.phase_name == SKIP_CONNECTION_SPEC_NAME)
        .expect("skip_connection_discovery spec must be registered");

    (spec.detect_fn)()
}

/// Issue #2368: the `deadline` passed into `append_structural_specs` must reach the
/// skip-connection scan. With no deadline, the deep chain must detect at least one
/// candidate (positive precondition). With an already-elapsed deadline, the in-scan
/// check must stop the scan before any candidate is produced.
#[test]
fn skip_connection_spec_stops_at_elapsed_deadline() {
    let without_deadline = run_skip_connection_spec(None);
    let result = without_deadline
        .expect("deep chain with attenuated neurons must be detected without a deadline");
    assert!(
        result.detected_count >= 1,
        "expected at least one skip-connection candidate, got {}",
        result.detected_count
    );

    let with_elapsed_deadline = run_skip_connection_spec(Some(SystemTime::UNIX_EPOCH));
    assert!(
        with_elapsed_deadline.is_none(),
        "an already-elapsed deadline must stop the skip-connection scan before any candidate is produced"
    );
}
