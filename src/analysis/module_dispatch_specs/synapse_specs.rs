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
    discovery_spec!(modules, "dormant synapse detection", "dormant_synapse_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_synapse_sources(&creature),
        detect: |records| dormant_synapse::detect_dormant_synapses(&creature, &records),
        convert: |detected| dormant_synapse::dormant_synapses_to_coordinated_candidates(&detected),
    );

    // Issue #360: Opposing synapse detection
    discovery_spec!(modules, "opposing synapse detection", "opposing_synapse_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| opposing_synapse::detect_opposing_synapses(&creature, &records),
        convert: |detected| opposing_synapse::opposing_synapses_to_coordinated_candidates(&detected),
    );

    // Issue #437: Weight coherence validation - incoherent weight ratios
    discovery_spec!(modules, "weight coherence ratio detection", "weight_coherence_ratio_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature, topo = topo =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| {
            let config = weight_coherence::WeightCoherenceConfig::default();
            weight_coherence::detect_incoherent_weight_ratios(&creature, &records, &config, Some(&topo))
        },
        convert: |detected| weight_coherence::incoherent_ratios_to_coordinated_candidates(&detected),
    );

    // Issue #437: Weight coherence validation - near-constant output paths
    discovery_spec!(modules, "near-constant path detection", "near_constant_path_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature, topo = topo =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| {
            let config = weight_coherence::WeightCoherenceConfig::default();
            weight_coherence::detect_near_constant_paths(&creature, &records, &config, Some(&topo))
        },
        convert: |detected| weight_coherence::near_constant_paths_to_coordinated_candidates(&detected),
    );

    // Issue #437: Weight coherence validation - symmetric weight cancellation
    discovery_spec!(modules, "symmetric cancellation detection", "symmetric_cancellation_detection",
        cache = shared_cache, creature = creature, topo = topo =>
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| {
            let config = weight_coherence::WeightCoherenceConfig::default();
            weight_coherence::detect_symmetric_cancellation(&creature, &records, &config, Some(&topo))
        },
        convert: |detected| weight_coherence::symmetric_cancellation_to_coordinated_candidates(&detected),
    );

    // Issue #434: Noise-to-signal ratio detection for synapses
    discovery_spec!(modules, "noisy synapse detection", "noisy_synapse_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| noise_signal::detect_noisy_synapses(&creature, &records),
        convert: |detected| noise_signal::noisy_synapses_to_coordinated_candidates(&detected),
    );

    // Issue #550: Weight magnitude reset for stuck synapses (local minimum escape)
    discovery_spec!(modules, "weight magnitude reset detection", "weight_magnitude_reset_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        guard_records,
        detect: |records| weight_magnitude_reset::detect_stuck_synapse_weight_resets(&creature, &records),
        convert: |detected| weight_magnitude_reset::stuck_synapses_to_coordinated_candidates(&detected),
    );

    // Issue #641: Fan-in weight polarity conflict detection
    discovery_spec!(modules, "fan-in polarity conflict detection", "fanin_polarity_conflict_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard: hidden,
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| fanin_polarity_conflict::detect_fanin_polarity_conflicts(&creature, &records),
        convert: |detected| fanin_polarity_conflict::fanin_polarity_conflicts_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #644: Weight polarity flip detection (gradient-weight sign disagreement)
    discovery_spec!(modules, "weight polarity flip detection", "weight_polarity_flip_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        guard_records,
        detect: |records| weight_polarity_flip::detect_weight_polarity_flip_candidates(&creature, &records),
        convert: |detected| weight_polarity_flip::polarity_flip_candidates_to_coordinated(&detected),
    );
}
