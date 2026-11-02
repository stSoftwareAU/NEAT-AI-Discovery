//! Scoring and recommendation discovery module dispatch specs.
//!
//! Covers: output bias drift, bounded range, sentinel gating, observation
//! utilisation, input sensitivity (dominant inputs + threshold effects),
//! sample-weighted discovery, gradient-based discovery, output squash
//! mismatch, error plateau detection, and output competition (lateral
//! inhibition).

use std::sync::Arc;
use std::time::SystemTime;

use super::super::cost_function_hint::CostFunctionHint;
use super::super::detection::{
    bounded_range, error_plateau, input_sensitivity, observation_utilisation,
    output_range_compression, output_squash_mismatch, sentinel_gating,
};
use super::super::recommendation::{
    batch_successful, gradient_discovery, output_bias_drift, output_competition, sample_weighted,
};
use super::super::task_descriptor::TaskDescriptor;
use super::super::{cache, discovery_dispatch};

/// Append scoring and recommendation discovery module specs to the provided vector.
///
/// `cost_hint` (Issue #1317) is forwarded into the
/// `output_squash_mismatch` detector so Strategy 4 (the pre-activation
/// squash comparison that reconstructs the implied target via
/// `activation − error`) is skipped when the recorded error is not a
/// linear residual (or is unknown).
///
/// `deadline` (Issue #2183) is forwarded into the gradient-based discovery
/// scan, and (Issue #2236) into the output-competition pair scan, so both
/// stop at the discovery deadline or on global cancellation.
pub(crate) fn append_scoring_specs(
    modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
    creature: &Arc<crate::CreatureJson>,
    shared_cache: &Arc<cache::RecordCache>,
    cost_hint: CostFunctionHint,
    task_descriptor: TaskDescriptor,
    deadline: Option<SystemTime>,
) {
    // Issue #361 / #1316: Output bias drift detection. Under OneHot / Simplex
    // descriptors the role-aware path additionally weights up output neurons
    // that never cross a saturating threshold for a class with positive
    // support (capacity starvation); other descriptors are unchanged.
    discovery_spec!(modules, "output bias drift detection", "output_bias_drift_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_neuron_types(&creature, &["output"]),
        detect: |records| output_bias_drift::detect_output_bias_drift_with_descriptor(
            &creature,
            &records,
            &task_descriptor,
        ),
        convert: |detected| output_bias_drift::output_bias_drift_to_coordinated_candidates(&detected),
    );

    // Issue #395: Bounded range detection
    discovery_spec!(modules, "bounded range detection", "bounded_range_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_neuron_types(&creature, &["input", "hidden"]),
        guard_records,
        detect: |records| bounded_range::detect_bounded_range_neurons(&creature, &records),
        convert: |detected| bounded_range::bounded_range_to_coordinated_candidates(&detected, &creature),
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
        convert: |detected| {
            let scan = observation_utilisation::observation_utilisation_to_coordinated_candidates_with_deadline(
                &detected,
                &creature,
                &deadline,
            );
            if let Some(reason) = scan.stop {
                tracing::warn!(
                    reason = ?reason,
                    returned = scan.candidates.len(),
                    "Observation utilisation conversion stopped early; returning partial candidates."
                );
            }
            scan.candidates
        },
    );

    // Issue #435: Input sensitivity analysis for dominant inputs
    discovery_spec!(modules, "dominant input detection", "dominant_input_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_neuron_types(&creature, &["input", "output"]),
        guard_records,
        detect: |records| {
            let config = input_sensitivity::InputSensitivityConfig::default();
            let scan = input_sensitivity::detect_dominant_inputs_with_deadline(
                &creature,
                &records,
                &config,
                &deadline,
            );
            if let Some(reason) = scan.stop {
                tracing::warn!(
                    reason = ?reason,
                    returned = scan.candidates.len(),
                    "Dominant input detection stopped early; returning partial candidates."
                );
            }
            scan.candidates
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
        convert: |detected| input_sensitivity::threshold_effects_to_coordinated_candidates(&detected, &creature),
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
        detect: |records| gradient_discovery::detect_gradient_candidates(&creature, &records, &deadline),
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
            // Issue #1317: forward the cost-function hint so Strategy 4's
            // implied-target reconstruction is skipped for non-linear /
            // unknown costs.
            let detected = output_squash_mismatch::detect_output_squash_mismatches_with_cost_hint(
                &output_neurons,
                &records,
                cost_hint,
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

    // Issue #1321 / #2185: Output competition (lateral inhibition). The
    // detector was built and tested but never dispatched — no spec existed, so
    // a one-of-N creature with competing outputs could never receive the
    // inhibitory synapse. Gated on the module's own topology predicate so the
    // record load is skipped for descriptors that can emit nothing.
    discovery_spec!(modules, "output competition detection", "output_competition_detection",
        cache = shared_cache, creature = creature =>
        custom: move || {
            if !output_competition::role_aware_topology(&task_descriptor) {
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
            let records = cache.load_records_for_neuron_types(&creature, &["output"]);
            // Issue #2236: the O(outputs²) pair scan stops at the deadline
            // or on cancellation; a partial scan is logged, not hidden.
            let scan = output_competition::detect_output_competition_with_deadline(
                &creature,
                &records,
                &task_descriptor,
                &deadline,
            );
            if let Some(reason) = scan.truncation {
                tracing::warn!(
                    reason = ?reason,
                    returned = scan.candidates.len(),
                    "Output competition scan stopped early; returning partial candidates."
                );
            }
            let detected = scan.candidates;
            if detected.is_empty() {
                return None;
            }
            let candidates =
                output_competition::output_competition_to_coordinated_candidates(&detected);
            Some(discovery_dispatch::DiscoveryDetectionResult {
                detected_count: detected.len(),
                candidates,
            })
        },
    );

    // Issue #965: Batch-successful candidate grouping — batch proven winners
    // Issue #1059: Disabled by default — zero production successes across all
    // 43 creatures in the production discovery cache. Set
    // NEAT_AI_DISCOVERY_BATCH_SUCCESSFUL=1
    // to re-enable for experimentation.
    if crate::config::batch_successful_enabled() {
        discovery_spec!(modules, "batch-successful grouping", "batch_successful_grouping",
            cache = shared_cache, creature = creature =>
            records: cache.load_records_for_all_neurons(&creature),
            guard_records,
            detect: |records| batch_successful::detect_batch_successful_groups(&creature, &records),
            convert: |detected| batch_successful::batch_successful_to_coordinated_candidates(&detected),
        );
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod issue_2379_tests {
    use super::*;
    use crate::types::DiscoverRecord;
    use crate::{NeuronJson, SynapseJson};

    fn make_records(uuid: &str) -> Vec<DiscoverRecord> {
        (0..50)
            .map(|obs_index| {
                if uuid == "input" && obs_index < 20 {
                    DiscoverRecord::new(obs_index, uuid.to_string(), Some(-1.0), -1.0, vec![0.01])
                } else {
                    let value = if uuid == "input" {
                        0.2 + (obs_index as f32 - 20.0) / 50.0
                    } else {
                        obs_index as f32
                    };
                    DiscoverRecord::new(
                        obs_index,
                        uuid.to_string(),
                        Some(value),
                        value,
                        vec![value],
                    )
                }
            })
            .collect()
    }

    fn scoring_specs(deadline: Option<SystemTime>) -> Vec<discovery_dispatch::DiscoveryModuleSpec> {
        let creature = Arc::new(crate::CreatureJson {
            neurons: vec![
                NeuronJson {
                    uuid: "input".to_string(),
                    neuron_type: "input".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![SynapseJson {
                from_uuid: "input".to_string(),
                to_uuid: "output".to_string(),
                weight: 20.0,
                synapse_type: None,
            }],
            input: 1,
            output: 1,
        });
        let cache = Arc::new(crate::analysis::cache::RecordCache::with_loader(
            "test-records",
            Arc::new(|_, uuid| Ok(make_records(uuid))),
        ));
        let mut modules = Vec::new();
        append_scoring_specs(
            &mut modules,
            &creature,
            &cache,
            CostFunctionHint::default(),
            TaskDescriptor::default(),
            deadline,
        );
        modules
    }

    fn run_named(
        modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
        name: &str,
    ) -> Option<discovery_dispatch::DiscoveryDetectionResult> {
        let index = modules
            .iter()
            .position(|module| module.module_name == name)
            .expect("requested scoring spec is present");
        let spec = modules.remove(index);
        (spec.detect_fn)()
    }

    #[test]
    fn issue_2379_observation_dispatch_passes_its_deadline_to_conversion() {
        let past_deadline = Some(SystemTime::UNIX_EPOCH);
        let mut modules = scoring_specs(past_deadline);
        let result = run_named(&mut modules, "observation utilisation detection")
            .expect("the observation detector still produced a detection");
        assert!(result.detected_count > 0);
        assert!(
            result.candidates.is_empty(),
            "deadline stops conversion before topology work"
        );
    }

    #[test]
    fn issue_2379_dominant_input_dispatch_passes_its_deadline_to_detection() {
        let mut modules = scoring_specs(Some(SystemTime::UNIX_EPOCH));
        let result = run_named(&mut modules, "dominant input detection");
        assert!(
            result.is_none(),
            "expired deadline prevents a dominant-input result"
        );

        let mut modules = scoring_specs(None);
        let result = run_named(&mut modules, "dominant input detection")
            .expect("the direct input-to-output path is a positive control");
        assert!(!result.candidates.is_empty());
    }

    #[test]
    fn issue_2412_threshold_dispatch_passes_creature_topology_to_conversion() {
        let creature = Arc::new(crate::CreatureJson {
            neurons: vec![
                NeuronJson {
                    uuid: "input".to_string(),
                    neuron_type: "input".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "steep".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "TANH".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "dampening-input-output".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                SynapseJson {
                    from_uuid: "input".to_string(),
                    to_uuid: "steep".to_string(),
                    weight: 20.0,
                    synapse_type: None,
                },
                SynapseJson {
                    from_uuid: "steep".to_string(),
                    to_uuid: "output".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                },
            ],
            input: 1,
            output: 1,
        });
        let cache = Arc::new(cache::RecordCache::with_loader(
            "issue-2412-threshold",
            Arc::new(|_, uuid| {
                Ok((0..50)
                    .map(|obs_index| {
                        let value = (obs_index as f32 - 25.0) / 250.0;
                        let activation = if uuid == "steep" { value * 20.0 } else { value };
                        DiscoverRecord::new(
                            obs_index,
                            uuid.to_string(),
                            Some(value),
                            activation,
                            vec![0.1],
                        )
                    })
                    .collect())
            }),
        ));
        let mut modules = Vec::new();
        append_scoring_specs(
            &mut modules,
            &creature,
            &cache,
            CostFunctionHint::default(),
            TaskDescriptor::default(),
            None,
        );

        let result = run_named(&mut modules, "threshold effect detection")
            .expect("fixture produces a threshold-effect detection");
        let added_uuid = result
            .candidates
            .iter()
            .flat_map(|candidate| &candidate.operations)
            .find_map(|operation| match operation {
                crate::CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. } => {
                    Some(neuron_uuid.as_str())
                }
                _ => None,
            })
            .expect("threshold dispatch emits a dampening neuron");
        assert_eq!(added_uuid, "dampening-input-output-1");
    }
}
