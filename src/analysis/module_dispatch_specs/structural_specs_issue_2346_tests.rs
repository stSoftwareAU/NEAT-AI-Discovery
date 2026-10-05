//! Regression test for Issue #2346: `append_structural_specs` must forward its
//! `deadline` argument into the correlated-error scan.
//!
//! Before this issue, the `correlated_error_detection` closure built in
//! `structural_specs.rs` called `correlated_error::detect_correlated_error_patterns`
//! (no deadline). It now calls
//! `correlated_error::detect_correlated_error_patterns_with_deadline(&creature,
//! &records, &deadline)`. This test invokes the registered closure directly
//! (bypassing the dispatch-level pre-check) with an already-elapsed deadline so
//! only the in-scan deadline check can stop it — if the call is ever reverted to
//! the legacy, deadline-less wrapper, this test goes red.

#![allow(clippy::cast_precision_loss)] // Intentional numeric cast for fixture noise (Issue #873)

use std::sync::Arc;
use std::time::SystemTime;

use super::append_structural_specs;
use crate::analysis::cache::RecordCache;
use crate::analysis::detection::topology_cache::CreatureTopologyCache;
use crate::analysis::discovery_dispatch::DiscoveryModuleSpec;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};

const CORRELATED_ERROR_SPEC_NAME: &str = "correlated_error_detection";

/// Matches the private `RecordCacheLoader` alias in `cache::mod` (not exported,
/// so the test re-declares it locally) to keep `correlated_loader`'s return
/// type readable.
type TestRecordLoader = dyn Fn(&str, &str) -> anyhow::Result<Vec<DiscoverRecord>> + Send + Sync;

/// Build a creature with two strongly-correlated outputs and one input, each
/// backed by enough records to clear `MIN_DISCOVERY_SAMPLE_COUNT` (20).
fn correlated_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-1".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-2".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-2".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 2,
    }
}

/// An in-memory loader producing, for each output neuron, 40 records with
/// strongly correlated errors (shared positive/negative error on alternating
/// samples), and a handful of neutral input records.
fn correlated_loader() -> Arc<TestRecordLoader> {
    Arc::new(
        |_file: &str, neuron_uuid: &str| -> anyhow::Result<Vec<DiscoverRecord>> {
            let records: Vec<DiscoverRecord> = (0..40u32)
                .map(|i| {
                    let base_error = if i % 2 == 0 { 0.5 } else { -0.3 };
                    let noise = (i as f32 * 0.001) % 0.01;
                    let errors = match neuron_uuid {
                        "output-1" | "output-2" => vec![base_error + noise],
                        _ => vec![0.0],
                    };
                    DiscoverRecord::new(i, neuron_uuid.to_string(), Some(0.5), 0.5, errors)
                })
                .collect();
            Ok(records)
        },
    )
}

/// Build the `correlated_error_detection` spec for the given deadline and
/// invoke its `detect_fn` directly, bypassing the dispatch-level pre-check so
/// only the in-scan deadline check can stop it.
fn run_correlated_error_spec(
    deadline: Option<SystemTime>,
) -> Option<crate::analysis::discovery_dispatch::DiscoveryDetectionResult> {
    let creature = Arc::new(correlated_creature());
    let hidden: Arc<Vec<(String, String, f32)>> = Arc::new(vec![]);
    let shared_cache = Arc::new(RecordCache::with_loader(
        "test.parquet",
        correlated_loader(),
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
        .find(|s| s.phase_name == CORRELATED_ERROR_SPEC_NAME)
        .expect("correlated_error_detection spec must be registered");

    (spec.detect_fn)()
}

/// Issue #2346: the `deadline` passed into `append_structural_specs` must reach
/// the correlated-error scan. With no deadline, the strongly-correlated fixture
/// above must detect at least one group (positive precondition); with an
/// already-elapsed deadline, the in-scan check must stop the scan before any
/// group is produced.
#[test]
fn correlated_error_spec_stops_at_elapsed_deadline() {
    let without_deadline = run_correlated_error_spec(None);
    let result =
        without_deadline.expect("strongly correlated outputs must be detected without a deadline");
    assert!(
        result.detected_count >= 1,
        "expected at least one correlated error group, got {}",
        result.detected_count
    );

    let with_elapsed_deadline = run_correlated_error_spec(Some(SystemTime::UNIX_EPOCH));
    assert!(
        with_elapsed_deadline.is_none(),
        "an already-elapsed deadline must stop the correlated-error scan before any group is produced"
    );
}
