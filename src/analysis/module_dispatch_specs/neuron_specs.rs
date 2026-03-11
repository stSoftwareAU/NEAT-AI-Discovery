//! Neuron-focused discovery module dispatch specs.
//!
//! Covers: saturated, bottleneck, dead, oscillating, restricted range,
//! operating point, unbounded capping, noisy neurons, activation
//! recommendation, activation mismatch, bimodal neuron, bias perturbation,
//! symmetry breaking, and co-adaptation detection.

use std::sync::Arc;

use super::super::detection::{
    activation_mismatch, bias_perturbation, bimodal_neuron, bottleneck, co_adaptation, dead_neuron,
    high_error_squash_exploration, monotonicity, noise_signal, operating_point, oscillating_neuron,
    restricted_range, saturation, squash_weight_rescale, symmetry_breaking,
    topology_cache::CreatureTopologyCache, unbounded_capping,
};
use super::super::recommendation::activation_recommendation;
use super::super::{cache, discovery_dispatch};

/// Append neuron-focused discovery module specs to the provided vector.
pub(crate) fn append_neuron_specs(
    modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
    creature: &Arc<crate::CreatureJson>,
    hidden_neurons: &Arc<Vec<(String, String, f32)>>,
    shared_cache: &Arc<cache::RecordCache>,
    topo: &Arc<CreatureTopologyCache>,
) {
    // Issue #342: Saturated neuron detection
    discovery_spec!(modules, "saturation detection", "saturation_detection",
        cache = shared_cache, hidden = hidden_neurons =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| saturation::detect_saturated_neurons(&hidden, &records),
        convert: |detected| saturation::saturated_neurons_to_coordinated_candidates(&detected),
    );

    // Issue #343: Bottleneck neuron detection
    discovery_spec!(modules, "bottleneck detection", "bottleneck_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature, topo = topo =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| bottleneck::detect_bottleneck_neurons(&creature, &records, Some(&topo)),
        convert: |detected| bottleneck::bottleneck_neurons_to_coordinated_candidates(&detected, &creature, Some(&topo)),
    );

    // Issue #341: Dead neuron detection
    discovery_spec!(modules, "dead neuron detection", "dead_neuron_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature, topo = topo =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| dead_neuron::detect_dead_neurons(&creature, &records, Some(&topo)),
        convert: |detected| dead_neuron::dead_neurons_to_coordinated_candidates(&detected),
    );

    // Issue #358: Oscillating neuron detection
    discovery_spec!(modules, "oscillating neuron detection", "oscillating_neuron_detection",
        cache = shared_cache, hidden = hidden_neurons =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| oscillating_neuron::detect_oscillating_neurons(&hidden, &records),
        convert: |detected| oscillating_neuron::oscillating_neurons_to_coordinated_candidates(&detected),
    );

    // Issue #399: Restricted activation range detection
    discovery_spec!(modules, "restricted range detection", "restricted_range_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| {
            let config = restricted_range::RestrictedRangeConfig::default();
            restricted_range::detect_restricted_range_neurons(&creature, &records, &config)
        },
        convert: |detected| restricted_range::restricted_range_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #401: Hidden neuron operating-point analysis
    discovery_spec!(modules, "operating point analysis", "operating_point_analysis",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| {
            let config = operating_point::OperatingPointConfig::default();
            operating_point::detect_operating_point_issues(&creature, &records, &config)
        },
        convert: |detected| operating_point::operating_point_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #441: Unbounded activation capping detection
    discovery_spec!(modules, "unbounded capping detection", "unbounded_capping_detection",
        cache = shared_cache, hidden = hidden_neurons =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| unbounded_capping::detect_unbounded_capping_candidates(&hidden, &records),
        convert: |detected| unbounded_capping::unbounded_capping_to_coordinated_candidates(&detected),
    );

    // Issue #434: Noise-to-signal ratio detection for neurons
    discovery_spec!(modules, "noisy neuron detection", "noisy_neuron_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| noise_signal::detect_noisy_neurons(&creature, &records),
        convert: |detected| noise_signal::noisy_neurons_to_coordinated_candidates(&detected),
    );

    // Issue #417: Proactive activation function recommendation (custom logic)
    discovery_spec!(modules, "activation recommendation", "activation_recommendation",
        cache = shared_cache, hidden = hidden_neurons =>
        custom: move || {
            if hidden.is_empty() {
                return None;
            }
            let records = cache.load_records_for_hidden(&hidden);
            let mut recommendations = Vec::new();
            for (uuid, squash, _bias) in hidden.iter() {
                if let Some(neuron_records) = records.iter().find(|(u, _)| u == uuid)
                    && let Some(rec) = activation_recommendation::recommend_activation_function(
                        &neuron_records.1,
                        squash,
                    )
                {
                    recommendations.push(rec);
                }
            }
            if recommendations.is_empty() {
                return None;
            }
            let candidates =
                activation_recommendation::recommendations_to_coordinated_candidates(
                    &recommendations,
                );
            Some(discovery_dispatch::DiscoveryDetectionResult {
                detected_count: recommendations.len(),
                candidates,
            })
        },
    );

    // Issue #543: Activation mismatch detection
    discovery_spec!(modules, "activation mismatch detection", "activation_mismatch_detection",
        cache = shared_cache, hidden = hidden_neurons =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| activation_mismatch::detect_activation_mismatches(&hidden, &records),
        convert: |detected| activation_mismatch::activation_mismatch_to_coordinated_candidates(&detected),
    );

    // Issue #788: High-error squash exploration (proactive change-squash volume)
    discovery_spec!(modules, "high error squash exploration", "high_error_squash_exploration",
        cache = shared_cache, hidden = hidden_neurons =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| high_error_squash_exploration::detect_high_error_squash_candidates(&hidden, &records),
        convert: |detected| high_error_squash_exploration::high_error_squash_to_coordinated_candidates(&detected),
    );

    // Issue #551: Bias perturbation for activation regime shifts (local minimum escape)
    discovery_spec!(modules, "bias perturbation regime shift detection", "bias_perturbation_regime_shift_detection",
        cache = shared_cache, hidden = hidden_neurons =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| bias_perturbation::detect_bias_perturbation_candidates(&hidden, &records),
        convert: |detected| bias_perturbation::bias_perturbation_to_coordinated_candidates(&detected),
    );

    // Issue #569: Symmetry-breaking detection for converged duplicate neurons
    discovery_spec!(modules, "symmetry breaking detection", "symmetry_breaking_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard_min: hidden 2,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| symmetry_breaking::detect_symmetric_neurons(&creature, &records),
        convert: |detected| symmetry_breaking::symmetric_neurons_to_coordinated_candidates(&detected),
    );

    // Issue #571: Activation co-adaptation detection for redundant neuron pairs
    discovery_spec!(modules, "co-adaptation detection", "co_adaptation_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard_min: hidden 2,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| co_adaptation::detect_co_adapted_neurons(&creature, &records),
        convert: |detected| co_adaptation::co_adapted_pairs_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #548: Squash + weight rescale detection
    discovery_spec!(modules, "squash weight rescale detection", "squash_weight_rescale_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| squash_weight_rescale::detect_squash_weight_rescale_candidates(&creature, &hidden, &records),
        convert: |detected| squash_weight_rescale::squash_weight_rescale_to_coordinated_candidates(&detected),
    );

    // Issue #643: Activation-error monotonicity detection
    discovery_spec!(modules, "monotonicity detection", "monotonicity_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| monotonicity::detect_non_monotonic_neurons(&creature, &records),
        convert: |detected| monotonicity::non_monotonic_neurons_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #640: Bimodal neuron detection (pre-activation distribution shape)
    discovery_spec!(modules, "bimodal neuron detection", "bimodal_neuron_detection",
        cache = shared_cache, hidden = hidden_neurons =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| bimodal_neuron::detect_bimodal_neurons(&hidden, &records),
        convert: |detected| bimodal_neuron::bimodal_neurons_to_coordinated_candidates(&detected),
    );
}
