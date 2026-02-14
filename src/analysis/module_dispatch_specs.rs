//! Discovery module specification builders (Issue #562).
//!
//! This module constructs the vector of `DiscoveryModuleSpec` entries that are
//! dispatched in parallel by `run_discovery_modules_parallel`. Each spec wraps a
//! detection module's load-records → detect → convert-to-candidates pipeline.

use std::sync::Arc;

use super::{
    activation_mismatch, activation_recommendation, bias_perturbation, bottleneck, bounded_range,
    cache, correlated_error, dead_neuron, discovery_dispatch, dormant_synapse, error_plateau,
    gradient_discovery, input_sensitivity, multi_hop, noise_signal, observation_utilisation,
    operating_point, opposing_synapse, oscillating_neuron, output_bias_drift,
    output_squash_mismatch, restricted_range, sample_weighted, saturation, sentinel_gating, shared,
    squash_weight_rescale, topology, topology_diversification, unbounded_capping, weight_coherence,
    weight_magnitude_reset,
};

/// Build all discovery module specs for parallel dispatch.
///
/// Each module follows the detect → convert-to-candidates pipeline. The returned
/// specs are consumed by `run_discovery_modules_parallel` which runs the detection
/// closures concurrently and merges results sequentially.
pub(crate) fn build_discovery_module_specs(
    creature: &Arc<crate::CreatureJson>,
    hidden_neurons: &Arc<Vec<(String, String, f32)>>,
    shared_cache: &Arc<cache::RecordCache>,
) -> Vec<discovery_dispatch::DiscoveryModuleSpec> {
    let mut modules: Vec<discovery_dispatch::DiscoveryModuleSpec> = Vec::with_capacity(32);

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

    // Issue #437: Weight coherence validation - incoherent weight ratios
    {
        let cache = Arc::clone(shared_cache);
        let hidden = Arc::clone(hidden_neurons);
        let creature = Arc::clone(creature);
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "weight coherence ratio detection".to_string(),
            phase_name: "weight_coherence_ratio_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let config = weight_coherence::WeightCoherenceConfig::default();
                let detected =
                    weight_coherence::detect_incoherent_weight_ratios(&creature, &records, &config);
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
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "near-constant path detection".to_string(),
            phase_name: "near_constant_path_detection",
            detect_fn: Box::new(move || {
                if hidden.is_empty() {
                    return None;
                }
                let records = cache.load_records_for_hidden(&hidden);
                let config = weight_coherence::WeightCoherenceConfig::default();
                let detected =
                    weight_coherence::detect_near_constant_paths(&creature, &records, &config);
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
        modules.push(discovery_dispatch::DiscoveryModuleSpec {
            module_name: "symmetric cancellation detection".to_string(),
            phase_name: "symmetric_cancellation_detection",
            detect_fn: Box::new(move || {
                let records = cache.load_records_for_all_neurons(&creature);
                let config = weight_coherence::WeightCoherenceConfig::default();
                let detected =
                    weight_coherence::detect_symmetric_cancellation(&creature, &records, &config);
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

    modules
}

/// Dispatch all discovery modules and merge results into the synapse result.
pub(crate) fn dispatch_and_merge_discovery_modules(
    syn: &mut shared::AnalyzeSynapsesResult,
    creature: &Arc<crate::CreatureJson>,
    hidden_neurons: &Arc<Vec<(String, String, f32)>>,
    shared_cache: &Arc<cache::RecordCache>,
    max_candidates: Option<usize>,
    diversify: bool,
) {
    let modules = build_discovery_module_specs(creature, hidden_neurons, shared_cache);
    discovery_dispatch::run_discovery_modules_parallel(syn, modules, max_candidates, diversify);
}

/// Perform cross-module deduplication on coordinated structural candidates.
pub(crate) fn deduplicate_cross_module_candidates(syn: &mut shared::AnalyzeSynapsesResult) {
    use super::{candidate_clustering, utils};
    use crate::observability::PhaseTimer;

    if syn.coordinated_structural_candidates.is_empty() {
        return;
    }

    crate::watchdog::beat("analysis::analyze_all → cross-module deduplication starting");
    let _dedup_timer = PhaseTimer::new("cross_module_deduplication");

    let before_count = syn.coordinated_structural_candidates.len();
    let dedup_result = candidate_clustering::deduplicate_cross_module_candidates(std::mem::take(
        &mut syn.coordinated_structural_candidates,
    ));
    syn.coordinated_structural_candidates = dedup_result.candidates;

    if dedup_result.duplicates_removed > 0 && utils::verbose_enabled() {
        eprintln!(
            "[NEAT-AI-Discovery][verbose] Cross-module deduplication: removed {} duplicate(s) from {} coordinated candidate(s) → {} remaining",
            dedup_result.duplicates_removed,
            before_count,
            syn.coordinated_structural_candidates.len()
        );
    }

    if dedup_result.conflicts_detected > 0 && utils::verbose_enabled() {
        eprintln!(
            "[NEAT-AI-Discovery][verbose] Cross-module deduplication: {} neuron conflict(s) detected (remove vs modify)",
            dedup_result.conflicts_detected
        );
    }

    // Update metadata to reflect the deduplicated count.
    syn.metadata.candidates_returned = syn.helpful_synapses.len()
        + syn.harmful_synapses.len()
        + syn.coordinated_structural_candidates.len();

    crate::watchdog::beat("analysis::analyze_all → cross-module deduplication finished");
}

/// Perform candidate clustering to reduce redundant ablation tests.
pub(crate) fn cluster_synapse_candidates(
    syn: &mut shared::AnalyzeSynapsesResult,
    creature: &crate::CreatureJson,
) {
    use super::{candidate_clustering, utils};
    use crate::observability::PhaseTimer;

    crate::watchdog::beat("analysis::analyze_all → candidate clustering starting");
    let _clustering_timer = PhaseTimer::new("candidate_clustering");

    let mut clusterable: Vec<candidate_clustering::ClusterableCandidate> = Vec::new();

    for c in &syn.helpful_synapses {
        clusterable.push(candidate_clustering::ClusterableCandidate {
            from_neuron_uuid: c.from_neuron_uuid.clone(),
            to_neuron_uuid: c.to_neuron_uuid.clone(),
            expected_improvement: c.expected_creature_score_gain,
            neuron_type: creature
                .neurons
                .iter()
                .find(|n| n.uuid == c.from_neuron_uuid)
                .map(|n| n.neuron_type.clone())
                .unwrap_or_else(|| "input".to_string()),
        });
    }

    for c in &syn.harmful_synapses {
        clusterable.push(candidate_clustering::ClusterableCandidate {
            from_neuron_uuid: c.from_neuron_uuid.clone(),
            to_neuron_uuid: c.to_neuron_uuid.clone(),
            expected_improvement: c.expected_creature_score_gain,
            neuron_type: creature
                .neurons
                .iter()
                .find(|n| n.uuid == c.from_neuron_uuid)
                .map(|n| n.neuron_type.clone())
                .unwrap_or_else(|| "input".to_string()),
        });
    }

    let clusters = candidate_clustering::cluster_candidates(&clusterable);

    if !clusters.is_empty() && utils::verbose_enabled() {
        let total_clustered: usize = clusters.iter().map(|c| c.member_count).sum();
        eprintln!(
            "[NEAT-AI-Discovery][verbose] Candidate clustering: {} cluster(s) covering {} candidate(s) of {} total",
            clusters.len(),
            total_clustered,
            clusterable.len()
        );
    }

    syn.candidate_clusters = clusters;

    crate::watchdog::beat("analysis::analyze_all → candidate clustering finished");
}
