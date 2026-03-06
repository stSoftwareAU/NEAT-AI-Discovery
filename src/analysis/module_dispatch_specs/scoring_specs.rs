//! Scoring and recommendation discovery module dispatch specs.
//!
//! Covers: output bias drift, bounded range, sentinel gating, observation
//! utilisation, input sensitivity (dominant inputs + threshold effects),
//! sample-weighted discovery, gradient-based discovery, output squash
//! mismatch, and error plateau detection.

use std::sync::Arc;

use super::super::detection::{
    bounded_range, error_plateau, input_sensitivity, observation_utilisation,
    output_range_compression, output_squash_mismatch, sentinel_gating,
};
use super::super::recommendation::{gradient_discovery, output_bias_drift, sample_weighted};
use super::super::{cache, discovery_dispatch};

/// Append scoring and recommendation discovery module specs to the provided vector.
pub(crate) fn append_scoring_specs(
    modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
    creature: &Arc<crate::CreatureJson>,
    shared_cache: &Arc<cache::RecordCache>,
) {
    // Issue #361: Output bias drift detection
    discovery_spec!(modules, "output bias drift detection", "output_bias_drift_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_neuron_types(&creature, &["output"]),
        detect: |records| output_bias_drift::detect_output_bias_drift(&creature, &records),
        convert: |detected| output_bias_drift::output_bias_drift_to_coordinated_candidates(&detected),
    );

    // Issue #395: Bounded range detection
    discovery_spec!(modules, "bounded range detection", "bounded_range_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_neuron_types(&creature, &["input", "hidden"]),
        guard_records,
        detect: |records| bounded_range::detect_bounded_range_neurons(&creature, &records),
        convert: |detected| bounded_range::bounded_range_to_coordinated_candidates(&detected),
    );

    // Issue #400: Sentinel value gating
    discovery_spec!(modules, "sentinel value gating", "sentinel_value_gating",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_neuron_types(&creature, &["input"]),
        guard_records,
        detect: |records| sentinel_gating::detect_sentinel_gating_candidates(&creature, &records),
        convert: |detected| sentinel_gating::sentinel_gating_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #543: Observation utilisation detection
    discovery_spec!(modules, "observation utilisation detection", "observation_utilisation_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_neuron_types(&creature, &["input"]),
        guard_records,
        detect: |records| observation_utilisation::detect_underutilised_observations(&creature, &records),
        convert: |detected| observation_utilisation::observation_utilisation_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #435: Input sensitivity analysis for dominant inputs
    discovery_spec!(modules, "dominant input detection", "dominant_input_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_neuron_types(&creature, &["input", "output"]),
        guard_records,
        detect: |records| {
            let config = input_sensitivity::InputSensitivityConfig::default();
            input_sensitivity::detect_dominant_inputs(&creature, &records, &config)
        },
        convert: |detected| input_sensitivity::dominant_inputs_to_coordinated_candidates(&detected),
    );

    // Issue #435: Input sensitivity analysis for threshold effects
    discovery_spec!(modules, "threshold effect detection", "threshold_effect_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| {
            let config = input_sensitivity::InputSensitivityConfig::default();
            input_sensitivity::detect_threshold_effects(&creature, &records, &config)
        },
        convert: |detected| input_sensitivity::threshold_effects_to_coordinated_candidates(&detected),
    );

    // Issue #423: Sample-weighted discovery — prioritise high-error samples
    discovery_spec!(modules, "sample-weighted discovery", "sample_weighted_discovery",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        guard_records,
        detect: |records| {
            let config = sample_weighted::SampleWeightedConfig::default();
            sample_weighted::detect_high_error_neurons(&records, &config)
        },
        convert: |detected| sample_weighted::high_error_neurons_to_coordinated_candidates(&detected),
    );

    // Issue #421: Gradient-based synapse adjustment — directional improvement hints
    discovery_spec!(modules, "gradient-based discovery", "gradient_based_discovery",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        guard_records,
        detect: |records| gradient_discovery::detect_gradient_candidates(&creature, &records),
        convert: |detected| gradient_discovery::gradient_candidates_to_coordinated(&detected),
    );

    // Issue #645: Output range compression detection
    discovery_spec!(modules, "output range compression detection", "output_range_compression_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_neuron_types(&creature, &["output"]),
        guard_records,
        detect: |records| {
            let config = output_range_compression::OutputRangeCompressionConfig::default();
            output_range_compression::detect_output_range_compression(&creature, &records, &config)
        },
        convert: |detected| output_range_compression::output_range_compression_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #545: Output squash mismatch detection (local minimum escape, custom logic)
    discovery_spec!(modules, "output squash mismatch detection", "output_squash_mismatch_detection",
        cache = shared_cache, creature = creature =>
        custom: move || {
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
        },
    );

    // Issue #545: Error stagnation plateau detection (local minimum escape, custom logic)
    discovery_spec!(modules, "error plateau detection", "error_plateau_detection",
        cache = shared_cache, creature = creature =>
        custom: move || {
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
        },
    );
}
