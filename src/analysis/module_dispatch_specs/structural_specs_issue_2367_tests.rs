//! Regression test for Issue #2367: `append_structural_specs` must forward its
//! `deadline` argument into the compound bias/weight degradation scan.
//!
//! The `compound_bias_weight_degradation_detection` closure calls
//! `compound_degradation::detect_compound_bias_weight_degradations_with_deadline(
//! &creature, &records, &deadline)`. This test invokes the registered closure
//! directly (bypassing the dispatch-level pre-check) so only the in-scan
//! deadline check can stop it. If the closure is reverted to the deadline-less
//! `detect_compound_bias_weight_degradations`, or passed `&None`, the
//! elapsed-deadline assertion goes red.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for fixture records (Issue #873)

use std::sync::Arc;
use std::time::SystemTime;

use super::append_structural_specs;
use crate::analysis::cache::RecordCache;
use crate::analysis::detection::topology_cache::CreatureTopologyCache;
use crate::analysis::discovery_dispatch::DiscoveryModuleSpec;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};

const COMPOUND_SPEC_NAME: &str = "compound_bias_weight_degradation_detection";

const SAMPLES: u32 = 40;

const HIDDEN: usize = 4;

/// Matches the private `RecordCacheLoader` alias in `cache::mod`.
type TestRecordLoader = dyn Fn(&str, &str) -> anyhow::Result<Vec<DiscoverRecord>> + Send + Sync;

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// `in-0 -> h{i} -> out-0` for `HIDDEN` hidden neurons: the same shape as the
/// `compound_degradation_tests.rs` fixture, where every (bias, weight) pair is
/// on a connected path.
fn compound_creature() -> CreatureJson {
    let mut neurons = vec![neuron("in-0", "input")];
    let mut synapses = Vec::new();
    for h in 0..HIDDEN {
        let uuid = format!("h{h}");
        neurons.push(neuron(&uuid, "hidden"));
        synapses.push(synapse("in-0", &uuid, 1.0));
        synapses.push(synapse(&uuid, "out-0", 0.5));
    }
    neurons.push(neuron("out-0", "output"));
    CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    }
}

/// Hidden neurons get a consistent bias offset (a bias correction each), and
/// `out-0` gets a continuous error (a weight correction per hidden -> out-0
/// synapse). The input has no records.
fn compound_loader() -> Arc<TestRecordLoader> {
    Arc::new(
        |_file: &str, uuid: &str| -> anyhow::Result<Vec<DiscoverRecord>> {
            if uuid == "out-0" {
                return Ok((0..SAMPLES)
                    .map(|o| {
                        let error = 0.2 + 0.1 * (o as f32 * 0.9 + 0.5).sin();
                        DiscoverRecord::new(o, uuid.to_string(), None, 0.0, vec![error])
                    })
                    .collect());
            }
            let Some(h) = uuid.strip_prefix('h').and_then(|n| n.parse::<u32>().ok()) else {
                return Ok(Vec::new());
            };
            let bias_error = 0.1 + 0.01 * h as f32;
            Ok((0..SAMPLES)
                .map(|o| {
                    let activation = 1.0 + 0.5 * (o as f32 * 0.7).sin();
                    DiscoverRecord::new(
                        o,
                        uuid.to_string(),
                        Some(0.0),
                        activation,
                        vec![bias_error],
                    )
                })
                .collect())
        },
    )
}

/// Build the compound-degradation spec for `deadline` and invoke its
/// `detect_fn` directly.
fn run_compound_spec(
    deadline: Option<SystemTime>,
) -> Option<crate::analysis::discovery_dispatch::DiscoveryDetectionResult> {
    let creature = Arc::new(compound_creature());
    let hidden: Arc<Vec<(String, String, f32)>> = Arc::new(
        (0..HIDDEN)
            .map(|h| (format!("h{h}"), "IDENTITY".to_string(), 0.0))
            .collect(),
    );
    let shared_cache = Arc::new(RecordCache::with_loader("test.parquet", compound_loader()));
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
        .find(|s| s.phase_name == COMPOUND_SPEC_NAME)
        .expect("compound_bias_weight_degradation_detection spec must be registered");

    (spec.detect_fn)()
}

/// Issue #2367: with no deadline the fixture yields compound candidates
/// (positive precondition); with an already-elapsed deadline the in-scan check
/// must stop the scan before any candidate is produced.
#[test]
fn compound_degradation_spec_stops_at_elapsed_deadline() {
    let result = run_compound_spec(None)
        .expect("the compound fixture must be detected when nothing bounds the scan");
    assert!(
        result.detected_count > 0,
        "expected compound candidates without a deadline, got {}",
        result.detected_count
    );

    assert!(
        run_compound_spec(Some(SystemTime::UNIX_EPOCH)).is_none(),
        "an already-elapsed deadline must stop the compound scan before any candidate is \
         produced; the spec is dropping the deadline it was given"
    );
}
