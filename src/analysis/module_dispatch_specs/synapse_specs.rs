//! Synapse-focused discovery module dispatch specs.
//!
//! Covers: dormant synapse, opposing synapse, weight coherence (ratio,
//! near-constant path, symmetric cancellation), noisy synapses, and
//! weight magnitude reset detection.

use std::sync::Arc;

use super::super::detection::{
    dormant_synapse, fanin_polarity_conflict, noise_signal, opposing_synapse,
    topology_cache::CreatureTopologyCache, weight_coherence, weight_magnitude_reset,
    weight_polarity_flip,
};
use super::super::{cache, discovery_dispatch};

/// Append synapse-focused discovery module specs to the provided vector.
pub(crate) fn append_synapse_specs(
    modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
    creature: &Arc<crate::CreatureJson>,
    hidden_neurons: &Arc<Vec<(String, String, f32)>>,
    shared_cache: &Arc<cache::RecordCache>,
    topo: &Arc<CreatureTopologyCache>,
) {
    // Issue #359: Dormant synapse detection
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "dormant synapse detection".to_string(),
            phase_name: "dormant_synapse_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_synapse_sources(&creature);
                let detected = dormant_synapse::detect_dormant_synapses(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    dormant_synapse::dormant_synapses_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #360: Opposing synapse detection
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "opposing synapse detection".to_string(),
            phase_name: "opposing_synapse_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_all_neurons(&creature);
                let detected = opposing_synapse::detect_opposing_synapses(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    opposing_synapse::opposing_synapses_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #437: Weight coherence validation - incoherent weight ratios
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        let topo = Arc::clone(topo);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "weight coherence ratio detection".to_string(),
            phase_name: "weight_coherence_ratio_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let config = weight_coherence::WeightCoherenceConfig::default();
                let detected = weight_coherence::detect_incoherent_weight_ratios(
                    &creature,
                    &records,
                    &config,
                    Some(&topo),
                );
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    weight_coherence::incoherent_ratios_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #437: Weight coherence validation - near-constant output paths
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        let topo = Arc::clone(topo);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "near-constant path detection".to_string(),
            phase_name: "near_constant_path_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let config = weight_coherence::WeightCoherenceConfig::default();
                let detected = weight_coherence::detect_near_constant_paths(
                    &creature,
                    &records,
                    &config,
                    Some(&topo),
                );
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    weight_coherence::near_constant_paths_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #437: Weight coherence validation - symmetric weight cancellation
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        let topo = Arc::clone(topo);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "symmetric cancellation detection".to_string(),
            phase_name: "symmetric_cancellation_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_all_neurons(&creature);
                let config = weight_coherence::WeightCoherenceConfig::default();
                let detected = weight_coherence::detect_symmetric_cancellation(
                    &creature,
                    &records,
                    &config,
                    Some(&topo),
                );
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    weight_coherence::symmetric_cancellation_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #434: Noise-to-signal ratio detection for synapses
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "noisy synapse detection".to_string(),
            phase_name: "noisy_synapse_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_all_neurons(&creature);
                let detected = noise_signal::detect_noisy_synapses(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates = noise_signal::noisy_synapses_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #550: Weight magnitude reset for stuck synapses (local minimum escape)
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "weight magnitude reset detection".to_string(),
            phase_name: "weight_magnitude_reset_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_all_neurons(&creature);
                if records.is_empty() {
                    return None;
                }
                let detected =
                    weight_magnitude_reset::detect_stuck_synapse_weight_resets(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    weight_magnitude_reset::stuck_synapses_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #641: Fan-in weight polarity conflict detection
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "fan-in polarity conflict detection".to_string(),
            phase_name: "fanin_polarity_conflict_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_all_neurons(&creature);
                let detected =
                    fanin_polarity_conflict::detect_fanin_polarity_conflicts(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    fanin_polarity_conflict::fanin_polarity_conflicts_to_coordinated_candidates(
                        &detected, &creature,
                    );
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #644: Weight polarity flip detection (gradient-weight sign disagreement)
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "weight polarity flip detection".to_string(),
            phase_name: "weight_polarity_flip_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_all_neurons(&creature);
                if records.is_empty() {
                    return None;
                }
                let detected = weight_polarity_flip::detect_weight_polarity_flip_candidates(
                    &creature, &records,
                );
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    weight_polarity_flip::polarity_flip_candidates_to_coordinated(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }
}
