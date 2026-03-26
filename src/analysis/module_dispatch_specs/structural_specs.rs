//! Structural discovery module dispatch specs.
//!
//! Covers: multi-hop candidate analysis, topology structure analysis,
//! topology diversification, skip-connection discovery, correlated
//! error detection, and fan-in candidate generation.

use std::sync::Arc;

use super::super::detection::{
    compound_degradation, correlated_error, hard_sample_cluster, output_conflict, skip_connection,
    topology, topology_cache::CreatureTopologyCache, topology_diversification,
};
use super::super::recommendation::{fan_in, multi_hop};
use super::super::{cache, discovery_dispatch};

/// Append structural discovery module specs to the provided vector.
pub(crate) fn append_structural_specs(
    modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
    creature: &Arc<crate::CreatureJson>,
    hidden_neurons: &Arc<Vec<(String, String, f32)>>,
    shared_cache: &Arc<cache::RecordCache>,
    topo: &Arc<CreatureTopologyCache>,
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
        },
    );

    // Issue #230: Multi-hop candidate analysis
    discovery_spec!(modules, "multi-hop analysis", "multi_hop_analysis",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard: hidden,
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| multi_hop::detect_multi_hop_candidates(&creature, &records),
        convert: |detected| multi_hop::multi_hop_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #422: Topology-aware network structure analysis
    discovery_spec!(modules, "topology structure analysis", "topology_structure_analysis",
        cache = shared_cache, hidden = hidden_neurons, creature = creature, topo = topo =>
        guard: hidden,
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| topology::detect_topology_issues(&creature, &records, Some(&topo)),
        convert: |detected| topology::topology_issues_to_coordinated_candidates(&detected, &creature, Some(&topo)),
    );

    // Issue #549: Topology diversification for structural jumps
    discovery_spec!(modules, "topology diversification detection", "topology_diversification_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| topology_diversification::detect_topology_diversification_candidates(&creature, &records),
        convert: |detected| topology_diversification::topology_diversification_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #570: Skip-connection discovery for beneficial residual connections
    discovery_spec!(modules, "skip connection discovery", "skip_connection_discovery",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard: hidden,
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| skip_connection::detect_skip_connection_candidates(&creature, &records),
        convert: |detected| skip_connection::skip_connections_to_coordinated_candidates(&detected, &creature),
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
                &detected, &creature,
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
        detect: |records| fan_in::detect_fan_in_candidates(&creature, &records),
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
        detect: |records| compound_degradation::detect_compound_bias_weight_degradations(&creature, &records),
        convert: |detected| compound_degradation::compound_degradations_to_coordinated_candidates(&detected),
    );
}
