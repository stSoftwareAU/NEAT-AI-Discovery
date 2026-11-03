//! Structural discovery module dispatch specs.
//!
//! Covers: multi-hop candidate analysis, topology structure analysis,
//! topology diversification, skip-connection discovery, correlated
//! error detection, and fan-in candidate generation.

use std::sync::Arc;
use std::time::SystemTime;

use super::super::detection::{
    compound_degradation, correlated_error, hard_sample_cluster, output_conflict, skip_connection,
    topology, topology_cache::CreatureTopologyCache, topology_diversification,
};
use super::super::recommendation::{fan_in, multi_hop};
use super::super::{cache, discovery_dispatch};

/// Append structural discovery module specs to the provided vector.
///
/// `deadline` is forwarded into the multi-hop and fan-in scans (Issue #2183),
/// the correlated-error scan (Issue #2346), topology analysis (Issue #2369),
/// the topology diversification scan (Issue #2489), the skip-connection scan (Issue #2368), the output-conflict conversion
/// (Issue #2358), and the compound bias+weight degradation scan (Issue #2367)
/// so they stop at the discovery deadline or on global cancellation.
pub(crate) fn append_structural_specs(
    modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
    creature: &Arc<crate::CreatureJson>,
    hidden_neurons: &Arc<Vec<(String, String, f32)>>,
    shared_cache: &Arc<cache::RecordCache>,
    topo: &Arc<CreatureTopologyCache>,
    deadline: Option<SystemTime>,
) {
    // Issue #344: Correlated error detection (custom: output count pre-check)
    discovery_spec!(modules, "correlated error detection", "correlated_error_detection",
        cache = shared_cache, creature = creature =>
        custom: move || {
            let output_count = creature
                .neurons
                .iter()
                .filter(|n| n.neuron_type == "output")
                .count();
            if output_count < 2 {
                return None;
            }
            let records = cache.load_records_for_neuron_types(&creature, &["output", "input"]);
            let scan = correlated_error::detect_correlated_error_patterns_with_deadline(
                &creature, &records, &deadline,
            );
            // Issue #2346: report (and return) partial results rather than failing silently.
            if let Some(reason) = scan.skip {
                tracing::warn!(reason = ?reason, returned = scan.groups.len(), "Correlated error scan skipped work or stopped early; returning partial groups.");
            }
            let detected = scan.groups;
            if detected.is_empty() {
                return None;
            }
            let candidates = correlated_error::correlated_errors_to_coordinated_candidates(
                &detected, &creature,
            );
            Some(discovery_dispatch::DiscoveryDetectionResult {
                detected_count: detected.len(),
                candidates,
            })
        },
    );

    // Issue #230: Multi-hop candidate analysis
    discovery_spec!(modules, "multi-hop analysis", "multi_hop_analysis",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard: hidden,
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| multi_hop::detect_multi_hop_candidates(&creature, &records, &deadline),
        convert: |detected| multi_hop::multi_hop_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #422: Topology-aware network structure analysis
    discovery_spec!(modules, "topology structure analysis", "topology_structure_analysis",
        cache = shared_cache, hidden = hidden_neurons, creature = creature, topo = topo =>
        custom: move || {
            if hidden.is_empty() {
                return None;
            }
            let records = cache.load_records_for_all_neurons(&creature);
            let scan = topology::detect_topology_issues_with_deadline(
                &creature, &records, Some(&topo), &deadline,
            );
            if let Some(reason) = scan.truncation {
                tracing::warn!(reason = ?reason, returned = scan.candidates.len(), "Topology scan stopped early; returning partial candidates.");
            }
            if scan.candidates.is_empty() {
                return None;
            }
            let detected = scan.candidates;
            let candidates = topology::topology_issues_to_coordinated_candidates(
                &detected, &creature, Some(&topo),
            );
            Some(discovery_dispatch::DiscoveryDetectionResult {
                detected_count: detected.len(),
                candidates,
            })
        },
    );

    // Issue #549: Topology diversification for structural jumps (custom: deadline-bounded scan, Issue #2489)
    discovery_spec!(modules, "topology diversification detection", "topology_diversification_detection",
        cache = shared_cache, creature = creature =>
        custom: move || {
            let records = cache.load_records_for_all_neurons(&creature);
            let scan = topology_diversification::detect_topology_diversification_candidates_with_deadline(
                &creature, &records, &deadline,
            );
            // Issue #2489: report (and return) partial results rather than failing silently.
            if let Some(reason) = scan.stop {
                tracing::warn!(reason = ?reason, returned = scan.candidates.len(), "Topology diversification scan stopped early; returning partial candidates.");
            }
            let detected = scan.candidates;
            if detected.is_empty() {
                return None;
            }
            let candidates = topology_diversification::topology_diversification_to_coordinated_candidates(
                &detected, &creature,
            );
            Some(discovery_dispatch::DiscoveryDetectionResult {
                detected_count: detected.len(),
                candidates,
            })
        },
    );

    // Issue #570: Skip-connection discovery for beneficial residual connections (custom: deadline-bounded scan, Issue #2368)
    discovery_spec!(modules, "skip connection discovery", "skip_connection_discovery",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        custom: move || {
            if hidden.is_empty() {
                return None;
            }
            let records = cache.load_records_for_all_neurons(&creature);
            let scan = skip_connection::detect_skip_connection_candidates_with_deadline(
                &creature, &records, &deadline,
            );
            // Issue #2368: report (and return) partial results rather than failing silently.
            if let Some(reason) = scan.stop {
                tracing::warn!(reason = ?reason, returned = scan.candidates.len(), "Skip-connection scan stopped early; returning partial candidates.");
            }
            let detected = scan.candidates;
            if detected.is_empty() {
                return None;
            }
            let candidates = skip_connection::skip_connections_to_coordinated_candidates(
                &detected, &creature,
            );
            Some(discovery_dispatch::DiscoveryDetectionResult {
                detected_count: detected.len(),
                candidates,
            })
        },
    );

    // Issue #639: Per-output error disaggregation for hidden neurons (custom: output count check)
    discovery_spec!(modules, "output conflict detection", "output_conflict_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        custom: move || {
            if hidden.is_empty() {
                return None;
            }
            let output_count = creature
                .neurons
                .iter()
                .filter(|n| n.neuron_type == "output")
                .count();
            if output_count < 2 {
                return None;
            }
            let records = cache.load_records_for_neuron_types(&creature, &["hidden"]);
            let detected = output_conflict::detect_output_conflict_neurons(&creature, &records);
            if detected.is_empty() {
                return None;
            }
            let candidates = output_conflict::output_conflicts_to_coordinated_candidates(
                &detected, &creature, &deadline,
            );
            Some(discovery_dispatch::DiscoveryDetectionResult {
                detected_count: detected.len(),
                candidates,
            })
        },
    );

    // Issue #908: Fan-in candidate generation (multiple inputs converging to one hidden neuron)
    discovery_spec!(modules, "fan-in candidate generation", "fan_in_candidate_generation",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| fan_in::detect_fan_in_candidates(&creature, &records, &deadline),
        convert: |detected| fan_in::fan_in_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #642: Hard sample cluster detection (cross-network high-error observations)
    discovery_spec!(modules, "hard sample cluster detection", "hard_sample_cluster_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| {
            let config = hard_sample_cluster::HardSampleClusterConfig::default();
            hard_sample_cluster::detect_hard_sample_clusters(&creature, &records, &config)
        },
        convert: |detected| hard_sample_cluster::hard_sample_clusters_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #929: Compound bias+weight degradation detection
    discovery_spec!(modules, "compound bias weight degradation detection", "compound_bias_weight_degradation_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard: hidden,
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| {
            // Issue #2367: deadline-aware scan, capped on pairs and candidates.
            let scan = compound_degradation::detect_compound_bias_weight_degradations_with_deadline(
                &creature, &records, &deadline,
            );
            if let Some(reason) = scan.truncation {
                tracing::warn!(reason = ?reason, returned = scan.candidates.len(), "Compound degradation scan stopped early or was capped; returning partial candidates.");
            }
            scan.candidates
        },
        convert: |detected| compound_degradation::compound_degradations_to_coordinated_candidates(&detected),
    );
}

#[cfg(test)]
#[path = "structural_specs_issue_2346_tests.rs"]
mod structural_specs_issue_2346_tests;

#[cfg(test)]
#[path = "structural_specs_issue_2368_tests.rs"]
mod structural_specs_issue_2368_tests;
#[cfg(test)]
#[path = "structural_specs_issue_2369_tests.rs"]
mod structural_specs_issue_2369_tests;

#[cfg(test)]
#[path = "structural_specs_issue_2367_tests.rs"]
mod structural_specs_issue_2367_tests;

/// Issue #2489: the dispatch-level check for the `topology_diversification_detection` spec.
///
/// Declared inline rather than as a sibling `structural_specs_issue_*_tests.rs` file, which is
/// outside this change's file set. It follows the same pattern as the Issue #2368 skip-connection
/// test: invoke the registered closure directly with a scripted deadline.
#[cfg(test)]
mod topology_diversification_dispatch_tests {
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
}
