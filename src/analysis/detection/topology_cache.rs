//! Pre-computed creature topology cache (Issue #754).
//!
//! Builds common topology data structures once per `analyze_all` call so that
//! detection modules can share them instead of each rebuilding independently.

use std::collections::{HashMap, HashSet};

use crate::CreatureJson;

/// Pre-computed topology data derived from a `CreatureJson`.
///
/// Computed once per `analyze_all` call and shared (via `Arc`) across all
/// detection modules in the parallel dispatch phase, eliminating redundant
/// `HashMap` / `HashSet` construction.
#[derive(Debug)]
pub struct CreatureTopologyCache {
    /// Maps `to_uuid → [from_uuid, …]` (incoming synapses per neuron).
    pub fan_in: HashMap<String, Vec<String>>,
    /// Maps `from_uuid → [to_uuid, …]` (outgoing synapses per neuron).
    pub fan_out: HashMap<String, Vec<String>>,
    /// UUIDs of hidden neurons.
    pub hidden_uuids: HashSet<String>,
    /// UUIDs of output neurons.
    pub output_uuids: HashSet<String>,
    /// UUIDs of input neurons.
    pub input_uuids: HashSet<String>,
    /// Nested map for zero-copy synapse existence checks: `from_uuid → {to_uuid, …}`.
    existing_synapse_set: HashMap<String, HashSet<String>>,
    /// Nested map for zero-copy synapse weight lookups: `from_uuid → (to_uuid → weight)`.
    synapse_weight_map: HashMap<String, HashMap<String, f32>>,
    /// Total number of synapses (for diagnostics).
    synapse_count: usize,
}

impl CreatureTopologyCache {
    /// Build the topology cache from a creature in a single pass over neurons
    /// and synapses.
    pub fn new(creature: &CreatureJson) -> Self {
        let neuron_count = creature.neurons.len();
        let synapse_count = creature.synapses.len();

        // Classify neurons by type in a single pass.
        let mut hidden_uuids = HashSet::with_capacity(neuron_count);
        let mut output_uuids = HashSet::with_capacity(creature.output);
        let mut input_uuids = HashSet::with_capacity(creature.input);

        for n in &creature.neurons {
            match n.neuron_type.as_str() {
                "hidden" => {
                    hidden_uuids.insert(n.uuid.clone());
                }
                "output" => {
                    output_uuids.insert(n.uuid.clone());
                }
                "input" => {
                    input_uuids.insert(n.uuid.clone());
                }
                _ => {}
            }
        }

        // Build synapse maps in a single pass.
        let mut fan_in: HashMap<String, Vec<String>> = HashMap::with_capacity(neuron_count);
        let mut fan_out: HashMap<String, Vec<String>> = HashMap::with_capacity(neuron_count);
        let mut existing_synapse_set: HashMap<String, HashSet<String>> =
            HashMap::with_capacity(neuron_count);
        let mut synapse_weight_map: HashMap<String, HashMap<String, f32>> =
            HashMap::with_capacity(neuron_count);

        for s in &creature.synapses {
            fan_in
                .entry(s.to_uuid.clone())
                .or_default()
                .push(s.from_uuid.clone());
            fan_out
                .entry(s.from_uuid.clone())
                .or_default()
                .push(s.to_uuid.clone());
            existing_synapse_set
                .entry(s.from_uuid.clone())
                .or_default()
                .insert(s.to_uuid.clone());
            synapse_weight_map
                .entry(s.from_uuid.clone())
                .or_default()
                .insert(s.to_uuid.clone(), s.weight);
        }

        Self {
            fan_in,
            fan_out,
            hidden_uuids,
            output_uuids,
            input_uuids,
            existing_synapse_set,
            synapse_weight_map,
            synapse_count,
        }
    }

    /// Return the fan-in list for a neuron (empty slice if none).
    pub fn fan_in_for(&self, uuid: &str) -> &[String] {
        self.fan_in.get(uuid).map_or(&[], |v| v.as_slice())
    }

    /// Return the fan-out list for a neuron (empty slice if none).
    pub fn fan_out_for(&self, uuid: &str) -> &[String] {
        self.fan_out.get(uuid).map_or(&[], |v| v.as_slice())
    }

    /// Check whether a synapse already exists (zero heap allocations).
    pub fn synapse_exists(&self, from_uuid: &str, to_uuid: &str) -> bool {
        self.existing_synapse_set
            .get(from_uuid)
            .is_some_and(|targets| targets.contains(to_uuid))
    }

    /// Get a synapse weight (zero heap allocations, returns `None` if absent).
    pub fn synapse_weight(&self, from_uuid: &str, to_uuid: &str) -> Option<f32> {
        self.synapse_weight_map
            .get(from_uuid)
            .and_then(|targets| targets.get(to_uuid))
            .copied()
    }

    /// Return the total number of synapses in the cache.
    pub fn synapse_count(&self) -> usize {
        self.synapse_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_creature() -> CreatureJson {
        CreatureJson {
            neurons: vec![
                crate::NeuronJson {
                    uuid: "i1".to_string(),
                    neuron_type: "input".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                crate::NeuronJson {
                    uuid: "h1".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "TANH".to_string(),
                    bias: 0.1,
                },
                crate::NeuronJson {
                    uuid: "h2".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "RELU".to_string(),
                    bias: 0.0,
                },
                crate::NeuronJson {
                    uuid: "o1".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "LOGISTIC".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                crate::SynapseJson {
                    from_uuid: "i1".to_string(),
                    to_uuid: "h1".to_string(),
                    weight: 0.5,
                    ..Default::default()
                },
                crate::SynapseJson {
                    from_uuid: "i1".to_string(),
                    to_uuid: "h2".to_string(),
                    weight: -0.3,
                    ..Default::default()
                },
                crate::SynapseJson {
                    from_uuid: "h1".to_string(),
                    to_uuid: "o1".to_string(),
                    weight: 0.8,
                    ..Default::default()
                },
                crate::SynapseJson {
                    from_uuid: "h2".to_string(),
                    to_uuid: "o1".to_string(),
                    weight: 0.4,
                    ..Default::default()
                },
            ],
            input: 1,
            output: 1,
        }
    }

    #[test]
    fn test_neuron_classification() {
        let creature = make_test_creature();
        let cache = CreatureTopologyCache::new(&creature);

        assert!(cache.hidden_uuids.contains("h1"));
        assert!(cache.hidden_uuids.contains("h2"));
        assert_eq!(cache.hidden_uuids.len(), 2);

        assert!(cache.output_uuids.contains("o1"));
        assert_eq!(cache.output_uuids.len(), 1);

        assert!(cache.input_uuids.contains("i1"));
        assert_eq!(cache.input_uuids.len(), 1);
    }

    #[test]
    fn test_fan_in_fan_out() {
        let creature = make_test_creature();
        let cache = CreatureTopologyCache::new(&creature);

        // i1 fans out to h1 and h2
        let i1_out = cache.fan_out_for("i1");
        assert_eq!(i1_out.len(), 2);
        assert!(i1_out.contains(&"h1".to_string()));
        assert!(i1_out.contains(&"h2".to_string()));

        // o1 has fan-in from h1 and h2
        let o1_in = cache.fan_in_for("o1");
        assert_eq!(o1_in.len(), 2);
        assert!(o1_in.contains(&"h1".to_string()));
        assert!(o1_in.contains(&"h2".to_string()));

        // h1 has fan-in from i1 only
        assert_eq!(cache.fan_in_for("h1").len(), 1);

        // non-existent neuron returns empty
        assert!(cache.fan_in_for("nope").is_empty());
        assert!(cache.fan_out_for("nope").is_empty());
    }

    #[test]
    fn test_synapse_lookups() {
        let creature = make_test_creature();
        let cache = CreatureTopologyCache::new(&creature);

        assert!(cache.synapse_exists("i1", "h1"));
        assert!(!cache.synapse_exists("h1", "i1"));
        assert!(!cache.synapse_exists("h1", "h2"));

        assert_eq!(cache.synapse_weight("i1", "h1"), Some(0.5));
        assert_eq!(cache.synapse_weight("h2", "o1"), Some(0.4));
        assert_eq!(cache.synapse_weight("o1", "i1"), None);
    }

    #[test]
    fn test_existing_synapses_count() {
        let creature = make_test_creature();
        let cache = CreatureTopologyCache::new(&creature);
        assert_eq!(cache.synapse_count(), 4);
    }
}
