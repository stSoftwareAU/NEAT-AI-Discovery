//! Neuron-focused discovery module dispatch specs.
//!
//! Covers: saturated, bottleneck, dead, oscillating, restricted range,
//! operating point, unbounded capping, noisy neurons, activation
//! recommendation, activation mismatch, bias perturbation, symmetry
//! breaking, and co-adaptation detection.

use std::sync::Arc;

use super::super::{
    activation_mismatch, activation_recommendation, bias_perturbation, bottleneck, cache,
    co_adaptation, dead_neuron, discovery_dispatch, noise_signal, operating_point,
    oscillating_neuron, restricted_range, saturation, squash_weight_rescale, symmetry_breaking,
    unbounded_capping,
};

/// Append neuron-focused discovery module specs to the provided vector.
pub(crate) fn append_neuron_specs(
    modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
    creature: &Arc<crate::CreatureJson>,
    hidden_neurons: &Arc<Vec<(String, String, f32)>>,
    shared_cache: &Arc<cache::RecordCache>,
) {
    // Issue #342: Saturated neuron detection
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "saturation detection".to_string(),
            phase_name: "saturation_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let detected = saturation::detect_saturated_neurons(&hidden, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates = saturation::saturated_neurons_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #343: Bottleneck neuron detection
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "bottleneck detection".to_string(),
            phase_name: "bottleneck_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let detected = bottleneck::detect_bottleneck_neurons(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    bottleneck::bottleneck_neurons_to_coordinated_candidates(&detected, &creature);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #341: Dead neuron detection
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "dead neuron detection".to_string(),
            phase_name: "dead_neuron_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let detected = dead_neuron::detect_dead_neurons(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates = dead_neuron::dead_neurons_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #358: Oscillating neuron detection
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "oscillating neuron detection".to_string(),
            phase_name: "oscillating_neuron_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let detected = oscillating_neuron::detect_oscillating_neurons(&hidden, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    oscillating_neuron::oscillating_neurons_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #399: Restricted activation range detection
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "restricted range detection".to_string(),
            phase_name: "restricted_range_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let config = restricted_range::RestrictedRangeConfig::default();
                let detected =
                    restricted_range::detect_restricted_range_neurons(&creature, &records, &config);
                if detected.is_empty() {
                    return None;
                }
                let candidates = restricted_range::restricted_range_to_coordinated_candidates(
                    &detected, &creature,
                );
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #401: Hidden neuron operating-point analysis
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "operating point analysis".to_string(),
            phase_name: "operating_point_analysis",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let config = operating_point::OperatingPointConfig::default();
                let detected =
                    operating_point::detect_operating_point_issues(&creature, &records, &config);
                if detected.is_empty() {
                    return None;
                }
                let candidates = operating_point::operating_point_to_coordinated_candidates(
                    &detected, &creature,
                );
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #441: Unbounded activation capping detection
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "unbounded capping detection".to_string(),
            phase_name: "unbounded_capping_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let detected =
                    unbounded_capping::detect_unbounded_capping_candidates(&hidden, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    unbounded_capping::unbounded_capping_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #434: Noise-to-signal ratio detection for neurons
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "noisy neuron detection".to_string(),
            phase_name: "noisy_neuron_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let detected = noise_signal::detect_noisy_neurons(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates = noise_signal::noisy_neurons_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #417: Proactive activation function recommendation
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "activation recommendation".to_string(),
            phase_name: "activation_recommendation",
            detect_fn: Box::new(move || {
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
            }),
        });
    }

    // Issue #543: Activation mismatch detection
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "activation mismatch detection".to_string(),
            phase_name: "activation_mismatch_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let detected = activation_mismatch::detect_activation_mismatches(&hidden, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    activation_mismatch::activation_mismatch_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #551: Bias perturbation for activation regime shifts (local minimum escape)
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "bias perturbation regime shift detection".to_string(),
            phase_name: "bias_perturbation_regime_shift_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let detected =
                    bias_perturbation::detect_bias_perturbation_candidates(&hidden, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    bias_perturbation::bias_perturbation_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #569: Symmetry-breaking detection for converged duplicate neurons
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "symmetry breaking detection".to_string(),
            phase_name: "symmetry_breaking_detection",
            detect_fn: Box::new(move || {
                if hidden.len() < 2 {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let detected = symmetry_breaking::detect_symmetric_neurons(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    symmetry_breaking::symmetric_neurons_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #571: Activation co-adaptation detection for redundant neuron pairs
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "co-adaptation detection".to_string(),
            phase_name: "co_adaptation_detection",
            detect_fn: Box::new(move || {
                if hidden.len() < 2 {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let detected = co_adaptation::detect_co_adapted_neurons(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    co_adaptation::co_adapted_pairs_to_coordinated_candidates(&detected, &creature);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #548: Squash + weight rescale detection (coordinated multi-neuron squash exploration)
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "squash weight rescale detection".to_string(),
            phase_name: "squash_weight_rescale_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let detected = squash_weight_rescale::detect_squash_weight_rescale_candidates(
                    &creature, &hidden, &records,
                );
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    squash_weight_rescale::squash_weight_rescale_to_coordinated_candidates(
                        &detected,
                    );
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }
}
