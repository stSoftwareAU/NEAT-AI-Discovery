//! Structural discovery module dispatch specs.
//!
//! Covers: multi-hop candidate analysis, topology structure analysis,
//! topology diversification, skip-connection discovery, and correlated
//! error detection.

use std::sync::Arc;

use super::super::detection::{
    correlated_error, hard_sample_cluster, output_conflict, skip_connection, topology,
    topology_diversification,
};
use super::super::recommendation::multi_hop;
use super::super::{cache, discovery_dispatch};

/// Append structural discovery module specs to the provided vector.
pub(crate) fn append_structural_specs(
    modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
    creature: &Arc<crate::CreatureJson>,
    hidden_neurons: &Arc<Vec<(String, String, f32)>>,
    shared_cache: &Arc<cache::RecordCache>,
) {
    // Issue #344: Correlated error detection
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "correlated error detection".to_string(),
            phase_name: "correlated_error_detection",
            detect_fn: Box::new(move || {
                let output_count = creature
                    .neurons
                    .iter()
                    .filter(|n| n.neuron_type == "output")
                    .count();
                if output_count < 2 {
                    return None;
                }
                let records = cache.load_records_for_neuron_types(&creature, &["output", "input"]);
                let detected =
                    correlated_error::detect_correlated_error_patterns(&creature, &records);
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
            }),
        });
    }

    // Issue #230: Multi-hop candidate analysis
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "multi-hop analysis".to_string(),
            phase_name: "multi_hop_analysis",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_all_neurons(&creature);
                let detected = multi_hop::detect_multi_hop_candidates(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    multi_hop::multi_hop_to_coordinated_candidates(&detected, &creature);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #422: Topology-aware network structure analysis
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "topology structure analysis".to_string(),
            phase_name: "topology_structure_analysis",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_all_neurons(&creature);
                let detected = topology::detect_topology_issues(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    topology::topology_issues_to_coordinated_candidates(&detected, &creature);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #549: Topology diversification for structural jumps
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "topology diversification detection".to_string(),
            phase_name: "topology_diversification_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_all_neurons(&creature);
                let detected = topology_diversification::detect_topology_diversification_candidates(
                    &creature, &records,
                );
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    topology_diversification::topology_diversification_to_coordinated_candidates(
                        &detected, &creature,
                    );
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #570: Skip-connection discovery for beneficial residual connections
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "skip connection discovery".to_string(),
            phase_name: "skip_connection_discovery",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_all_neurons(&creature);
                let detected =
                    skip_connection::detect_skip_connection_candidates(&creature, &records);
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
            }),
        });
    }

    // Issue #639: Per-output error disaggregation for hidden neurons
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "output conflict detection".to_string(),
            phase_name: "output_conflict_detection",
            detect_fn: Box::new(move || {
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
                    &detected, &creature,
                );
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #642: Hard sample cluster detection (cross-network high-error observations)
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "hard sample cluster detection".to_string(),
            phase_name: "hard_sample_cluster_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_all_neurons(&creature);
                let config = hard_sample_cluster::HardSampleClusterConfig::default();
                let detected =
                    hard_sample_cluster::detect_hard_sample_clusters(&creature, &records, &config);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    hard_sample_cluster::hard_sample_clusters_to_coordinated_candidates(
                        &detected, &creature,
                    );
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }
}
