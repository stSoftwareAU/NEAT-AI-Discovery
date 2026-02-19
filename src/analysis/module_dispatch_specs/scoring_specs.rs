//! Scoring and recommendation discovery module dispatch specs.
//!
//! Covers: output bias drift, bounded range, sentinel gating, observation
//! utilisation, input sensitivity (dominant inputs + threshold effects),
//! sample-weighted discovery, gradient-based discovery, output squash
//! mismatch, and error plateau detection.

use std::sync::Arc;

use super::super::{
    bounded_range, cache, discovery_dispatch, error_plateau, gradient_discovery, input_sensitivity,
    observation_utilisation, output_bias_drift, output_range_compression, output_squash_mismatch,
    sample_weighted, sentinel_gating,
};

/// Append scoring and recommendation discovery module specs to the provided vector.
pub(crate) fn append_scoring_specs(
    modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
    creature: &Arc<crate::CreatureJson>,
    shared_cache: &Arc<cache::RecordCache>,
) {
    // Issue #361: Output bias drift detection
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "output bias drift detection".to_string(),
            phase_name: "output_bias_drift_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_neuron_types(&creature, &["output"]);
                let detected = output_bias_drift::detect_output_bias_drift(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    output_bias_drift::output_bias_drift_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #395: Bounded range detection
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "bounded range detection".to_string(),
            phase_name: "bounded_range_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_neuron_types(&creature, &["input", "hidden"]);
                if records.is_empty() {
                    return None;
                }
                let detected = bounded_range::detect_bounded_range_neurons(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates = bounded_range::bounded_range_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #400: Sentinel value gating
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "sentinel value gating".to_string(),
            phase_name: "sentinel_value_gating",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_neuron_types(&creature, &["input"]);
                if records.is_empty() {
                    return None;
                }
                let detected =
                    sentinel_gating::detect_sentinel_gating_candidates(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates = sentinel_gating::sentinel_gating_to_coordinated_candidates(
                    &detected, &creature,
                );
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #543: Observation utilisation detection
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "observation utilisation detection".to_string(),
            phase_name: "observation_utilisation_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_neuron_types(&creature, &["input"]);
                if records.is_empty() {
                    return None;
                }
                let detected =
                    observation_utilisation::detect_underutilised_observations(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    observation_utilisation::observation_utilisation_to_coordinated_candidates(
                        &detected, &creature,
                    );
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #435: Input sensitivity analysis for dominant inputs
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "dominant input detection".to_string(),
            phase_name: "dominant_input_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_neuron_types(&creature, &["input", "output"]);
                if records.is_empty() {
                    return None;
                }
                let config = input_sensitivity::InputSensitivityConfig::default();
                let detected =
                    input_sensitivity::detect_dominant_inputs(&creature, &records, &config);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    input_sensitivity::dominant_inputs_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #435: Input sensitivity analysis for threshold effects
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "threshold effect detection".to_string(),
            phase_name: "threshold_effect_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_all_neurons(&creature);
                let config = input_sensitivity::InputSensitivityConfig::default();
                let detected =
                    input_sensitivity::detect_threshold_effects(&creature, &records, &config);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    input_sensitivity::threshold_effects_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #423: Sample-weighted discovery — prioritise high-error samples
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "sample-weighted discovery".to_string(),
            phase_name: "sample_weighted_discovery",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_all_neurons(&creature);
                if records.is_empty() {
                    return None;
                }
                let config = sample_weighted::SampleWeightedConfig::default();
                let detected = sample_weighted::detect_high_error_neurons(&records, &config);
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    sample_weighted::high_error_neurons_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #421: Gradient-based synapse adjustment — directional improvement hints
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "gradient-based discovery".to_string(),
            phase_name: "gradient_based_discovery",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_all_neurons(&creature);
                if records.is_empty() {
                    return None;
                }
                let detected = gradient_discovery::detect_gradient_candidates(&creature, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates = gradient_discovery::gradient_candidates_to_coordinated(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #645: Output range compression detection
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "output range compression detection".to_string(),
            phase_name: "output_range_compression_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_neuron_types(&creature, &["output"]);
                if records.is_empty() {
                    return None;
                }
                let config = output_range_compression::OutputRangeCompressionConfig::default();
                let detected = output_range_compression::detect_output_range_compression(
                    &creature, &records, &config,
                );
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    output_range_compression::output_range_compression_to_coordinated_candidates(
                        &detected, &creature,
                    );
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #545: Output squash mismatch detection (local minimum escape)
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "output squash mismatch detection".to_string(),
            phase_name: "output_squash_mismatch_detection",
            detect_fn: Box::new(move || {
                let output_neurons: Vec<(String, String, f32)> = creature
                    .neurons
                    .iter()
                    .filter(|n| n.neuron_type == "output")
                    .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
                    .collect();
                if output_neurons.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_neuron_types(&creature, &["output"]);
                let detected = output_squash_mismatch::detect_output_squash_mismatches(
                    &output_neurons,
                    &records,
                );
                if detected.is_empty() {
                    return None;
                }
                let candidates =
                    output_squash_mismatch::output_squash_mismatch_to_coordinated_candidates(
                        &detected,
                    );
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }

    // Issue #545: Error stagnation plateau detection (local minimum escape)
    {
        let cache = Arc::clone(shared_cache);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "error plateau detection".to_string(),
            phase_name: "error_plateau_detection",
            detect_fn: Box::new(move || {
                let output_neurons: Vec<(String, String, f32)> = creature
                    .neurons
                    .iter()
                    .filter(|n| n.neuron_type == "output")
                    .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
                    .collect();
                if output_neurons.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_neuron_types(&creature, &["output"]);
                let detected = error_plateau::detect_error_plateaus(&output_neurons, &records);
                if detected.is_empty() {
                    return None;
                }
                let candidates = error_plateau::error_plateaus_to_coordinated_candidates(&detected);
                Some(discovery_dispatch::DiscoveryDetectionResult {
                    detected_count: detected.len(),
                    candidates,
                })
            }),
        });
    }
}
