//! Preparation helpers for synapse analysis
//!
//! This module extracts the creature lookup map building and constant-source
//! threshold computation from `analyze_synapses_with_cache_impl`, keeping
//! the main orchestration function concise.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::intern::NeuronIndex;
use crate::{AnalyzeSynapsesInput, SynapseJson};
use std::collections::{HashMap, HashSet};

use crate::analysis::cache::RecordCache;
use crate::analysis::samples::{compute_source_std_dev, get_constant_source_threshold};
use crate::analysis::utils::verbose_enabled;

use super::candidate_generation::build_ordered_neurons;
use crate::analysis::utils::OrderedNeuron;

/// All pre-computed lookup structures derived from a creature's topology.
///
/// Built once at the start of synapse analysis and shared (via `Arc`)
/// across the per-target parallel loop.
///
/// Maps that only reference `NeuronJson` or `SynapseJson` fields use `&str`
/// to avoid cloning UUID and squash strings from the input (Issue #808).
/// Maps that require `format!("input-{i}")` keys retain owned `String`s.
pub(crate) struct CreatureLookups<'a> {
    pub ordered_neurons: Vec<OrderedNeuron>,
    pub order_map: HashMap<String, usize>,
    pub neuron_index: NeuronIndex,
    pub existing_synapses: HashSet<(u32, u32)>,
    pub existing_synapse_weights: HashMap<(u32, u32), f32>,
    pub synapses_by_target: HashMap<u32, Vec<&'a SynapseJson>>,
    pub neuron_squash_map: HashMap<&'a str, &'a str>,
    pub neuron_type_map: HashMap<String, &'a str>,
    pub input_neuron_uuids: HashSet<String>,
    pub used_inputs: HashSet<&'a str>,
    pub neuron_bias_map: HashMap<&'a str, f32>,
}

/// Build all creature lookup maps from the analysis input.
///
/// This includes ordered neurons, interned neuron indices, existing synapse
/// sets, squash/type/bias maps, and input neuron tracking structures.
pub(crate) fn build_creature_lookups<'a>(input: &'a AnalyzeSynapsesInput) -> CreatureLookups<'a> {
    let ordered_neurons = build_ordered_neurons(&input.creature);
    let order_map: HashMap<String, usize> = ordered_neurons
        .iter()
        .map(|neuron| (neuron.uuid.clone(), neuron.index))
        .collect();

    let mut neuron_index = NeuronIndex::with_capacity(
        input.creature.neurons.len() + input.creature.input + input.creature.synapses.len() / 10,
    );

    // Pre-intern all neuron UUIDs
    for i in 0..input.creature.input {
        neuron_index.intern(&format!("input-{i}"));
    }
    for neuron in &input.creature.neurons {
        neuron_index.intern(&neuron.uuid);
    }

    // Build existing_synapses using interned indices
    let existing_synapses: HashSet<(u32, u32)> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| {
            (
                neuron_index.intern(&synapse.from_uuid),
                neuron_index.intern(&synapse.to_uuid),
            )
        })
        .collect();

    let existing_synapse_weights: HashMap<(u32, u32), f32> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| {
            (
                (
                    neuron_index.intern(&synapse.from_uuid),
                    neuron_index.intern(&synapse.to_uuid),
                ),
                synapse.weight,
            )
        })
        .collect();

    let synapses_by_target: HashMap<u32, Vec<&SynapseJson>> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| (neuron_index.intern(&synapse.to_uuid), synapse))
        .fold(HashMap::new(), |mut acc, (key, val)| {
            acc.entry(key).or_default().push(val);
            acc
        });

    let neuron_squash_map: HashMap<&str, &str> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.squash.as_str()))
        .collect();

    // Build comprehensive neuron type map (Issue #808: borrow type values from input)
    let mut neuron_type_map: HashMap<String, &str> = HashMap::new();
    for input_index in 0..input.creature.input {
        neuron_type_map.insert(format!("input-{input_index}"), "input");
    }
    for neuron in &input.creature.neurons {
        neuron_type_map.insert(neuron.uuid.clone(), neuron.neuron_type.as_str());
    }

    let input_neuron_uuids: HashSet<String> = (0..input.creature.input)
        .map(|i| format!("input-{i}"))
        .collect();

    // Issue #182: Build used inputs set (Issue #808: borrow from input)
    let used_inputs: HashSet<&str> = input
        .creature
        .synapses
        .iter()
        .filter(|s| crate::analysis::utils::parse_input_index(&s.from_uuid).is_some())
        .map(|s| s.from_uuid.as_str())
        .collect();

    // Neuron bias map for constant-source folding (Issue #808: borrow from input)
    let neuron_bias_map: HashMap<&str, f32> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.bias))
        .collect();

    CreatureLookups {
        ordered_neurons,
        order_map,
        neuron_index,
        existing_synapses,
        existing_synapse_weights,
        synapses_by_target,
        neuron_squash_map,
        neuron_type_map,
        input_neuron_uuids,
        used_inputs,
        neuron_bias_map,
    }
}

/// Compute the dynamic constant-source effect threshold from cached records.
///
/// Samples up to 50 input neurons to compute the average standard deviation,
/// then uses it to derive the threshold via `get_constant_source_threshold`.
pub(crate) fn compute_constant_source_threshold_from_cache(
    input: &AnalyzeSynapsesInput,
    cache: &RecordCache,
) -> Option<f32> {
    let mut std_dev_sum = 0.0f64;
    let mut std_dev_count = 0u32;
    let max_samples = input.creature.input.min(50);

    for input_idx in 0..max_samples {
        let input_uuid = format!("input-{input_idx}");
        if let Ok(records) = cache.get(&input_uuid)
            && records.len() >= 2
        {
            let std_dev = compute_source_std_dev(&records);
            if std_dev.is_finite() {
                std_dev_sum += std_dev as f64;
                std_dev_count += 1;
            }
        }
    }

    let source_std_dev_avg = if std_dev_count > 0 {
        let avg = (std_dev_sum / std_dev_count as f64) as f32;
        if verbose_enabled() {
            tracing::debug!(
                avg_std_dev = %format!("{avg:.4}"),
                sampled_sources = std_dev_count,
                "Source variance profile"
            );
        }
        Some(avg)
    } else {
        None
    };

    get_constant_source_threshold(source_std_dev_avg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CreatureJson, NeuronJson, SynapseJson};

    fn make_test_input() -> AnalyzeSynapsesInput {
        AnalyzeSynapsesInput {
            parquet_file: String::new(),
            creature: CreatureJson {
                input: 2,
                output: 1,
                neurons: vec![
                    NeuronJson {
                        uuid: "hidden-1".to_string(),
                        squash: "TANH".to_string(),
                        bias: 0.5,
                        neuron_type: "hidden".to_string(),
                    },
                    NeuronJson {
                        uuid: "output-1".to_string(),
                        squash: "LOGISTIC".to_string(),
                        bias: -0.1,
                        neuron_type: "output".to_string(),
                    },
                ],
                synapses: vec![
                    SynapseJson {
                        from_uuid: "input-0".to_string(),
                        to_uuid: "hidden-1".to_string(),
                        weight: 0.5,
                        synapse_type: None,
                    },
                    SynapseJson {
                        from_uuid: "hidden-1".to_string(),
                        to_uuid: "output-1".to_string(),
                        weight: -0.3,
                        synapse_type: None,
                    },
                ],
            },
            focus_neurons: vec!["output-1".to_string()],
            max_candidates: None,
            random_seed: Some(42),
            analysis_deadline_ms: None,
        }
    }

    #[test]
    fn test_build_creature_lookups_ordered_neurons() {
        let input = make_test_input();
        let lookups = build_creature_lookups(&input);

        // Should have input neurons + hidden + output
        assert_eq!(lookups.ordered_neurons.len(), 4); // 2 inputs + 2 neurons
        assert_eq!(lookups.order_map.len(), 4);
    }

    #[test]
    fn test_build_creature_lookups_existing_synapses() {
        let input = make_test_input();
        let lookups = build_creature_lookups(&input);

        assert_eq!(lookups.existing_synapses.len(), 2);
        assert_eq!(lookups.existing_synapse_weights.len(), 2);
    }

    #[test]
    fn test_build_creature_lookups_neuron_maps() {
        let input = make_test_input();
        let lookups = build_creature_lookups(&input);

        // Squash map should have hidden + output neurons (Issue #808: borrowed keys/values)
        assert_eq!(lookups.neuron_squash_map.len(), 2);
        assert_eq!(lookups.neuron_squash_map.get("hidden-1"), Some(&"TANH"));

        // Type map should have inputs + neurons
        assert_eq!(lookups.neuron_type_map.len(), 4);
        assert_eq!(lookups.neuron_type_map.get("input-0"), Some(&"input"));
        assert_eq!(lookups.neuron_type_map.get("hidden-1"), Some(&"hidden"));

        // Input neuron UUIDs
        assert_eq!(lookups.input_neuron_uuids.len(), 2);
        assert!(lookups.input_neuron_uuids.contains("input-0"));
        assert!(lookups.input_neuron_uuids.contains("input-1"));

        // Used inputs (only input-0 is used in a synapse)
        assert_eq!(lookups.used_inputs.len(), 1);
        assert!(lookups.used_inputs.contains("input-0"));

        // Bias map
        assert_eq!(lookups.neuron_bias_map.len(), 2);
        assert_eq!(lookups.neuron_bias_map.get("hidden-1"), Some(&0.5));
    }

    #[test]
    fn test_build_creature_lookups_synapses_by_target() {
        let input = make_test_input();
        let lookups = build_creature_lookups(&input);

        // Two targets: hidden-1 and output-1
        assert_eq!(lookups.synapses_by_target.len(), 2);
    }

    #[test]
    fn test_build_creature_lookups_empty_creature() {
        let input = AnalyzeSynapsesInput {
            parquet_file: String::new(),
            creature: CreatureJson {
                input: 0,
                output: 0,
                neurons: vec![],
                synapses: vec![],
            },
            focus_neurons: vec![],
            max_candidates: None,
            random_seed: None,
            analysis_deadline_ms: None,
        };
        let lookups = build_creature_lookups(&input);

        assert!(lookups.ordered_neurons.is_empty());
        assert!(lookups.existing_synapses.is_empty());
        assert!(lookups.neuron_type_map.is_empty());
    }
}
