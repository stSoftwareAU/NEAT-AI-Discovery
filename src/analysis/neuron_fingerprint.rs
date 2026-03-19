//! Neuron fingerprinting for incremental analysis (Issue #490).
//!
//! Each neuron's "fingerprint" captures the structural properties that affect
//! its analysis results: incoming/outgoing synapse weights and sources,
//! activation function, and bias. When the fingerprint is unchanged between
//! discovery runs, we can skip GPU analysis for that neuron.
//!
//! The fingerprint is cheap to compute — O(s) where s is the synapse count —
//! and conservative: any structural change invalidates the cache entry.

use crate::CreatureJson;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Fingerprint capturing the structural state of a single neuron.
///
/// Two fingerprints are equal if and only if the neuron has the same
/// activation function, bias, and identical incoming/outgoing synapses
/// (same sources/targets and weights).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronFingerprint {
    /// Hash of the neuron's structural properties.
    hash: u64,
}

impl PartialEq for NeuronFingerprint {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl Eq for NeuronFingerprint {}

impl Hash for NeuronFingerprint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

/// Result of filtering focus neurons by fingerprint comparison.
pub struct FilterResult {
    /// UUIDs of neurons that have changed and need re-analysis.
    pub changed: Vec<String>,
    /// UUIDs of neurons that were skipped (unchanged fingerprints).
    pub skipped_uuids: Vec<String>,
    /// Number of focus neurons with matching (unchanged) fingerprints.
    pub cache_hits: usize,
    /// Number of focus neurons with changed or missing fingerprints.
    pub cache_misses: usize,
    /// Total focus neurons evaluated.
    pub total_focus_neurons: usize,
}

/// Compute fingerprints for all neurons in the creature.
///
/// The fingerprint for each neuron includes:
/// - Activation function (squash)
/// - Bias value
/// - Incoming synapses: sorted list of (`source_uuid`, weight)
/// - Outgoing synapses: sorted list of (`target_uuid`, weight)
///
/// This is O(n + s) where n is neuron count and s is synapse count.
pub fn compute_neuron_fingerprints(creature: &CreatureJson) -> HashMap<String, NeuronFingerprint> {
    // Pre-compute incoming and outgoing synapses per neuron.
    let mut incoming: HashMap<&str, Vec<(&str, u32)>> = HashMap::new();
    let mut outgoing: HashMap<&str, Vec<(&str, u32)>> = HashMap::new();

    for syn in &creature.synapses {
        incoming
            .entry(syn.to_uuid.as_str())
            .or_default()
            .push((syn.from_uuid.as_str(), syn.weight.to_bits()));
        outgoing
            .entry(syn.from_uuid.as_str())
            .or_default()
            .push((syn.to_uuid.as_str(), syn.weight.to_bits()));
    }

    let mut fingerprints = HashMap::with_capacity(creature.neurons.len());

    for neuron in &creature.neurons {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash activation function
        neuron.squash.hash(&mut hasher);

        // Hash bias (as bits for exact comparison)
        neuron.bias.to_bits().hash(&mut hasher);

        // Hash incoming synapses in sorted order for determinism
        let mut incoming_syns: Vec<(&str, u32)> = incoming
            .get(neuron.uuid.as_str())
            .cloned()
            .unwrap_or_default();
        incoming_syns.sort_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(&b.1)));
        incoming_syns.len().hash(&mut hasher);
        for (source, weight_bits) in &incoming_syns {
            source.hash(&mut hasher);
            weight_bits.hash(&mut hasher);
        }

        // Hash outgoing synapses in sorted order for determinism
        let mut outgoing_syns: Vec<(&str, u32)> = outgoing
            .get(neuron.uuid.as_str())
            .cloned()
            .unwrap_or_default();
        outgoing_syns.sort_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(&b.1)));
        outgoing_syns.len().hash(&mut hasher);
        for (target, weight_bits) in &outgoing_syns {
            target.hash(&mut hasher);
            weight_bits.hash(&mut hasher);
        }

        fingerprints.insert(
            neuron.uuid.clone(),
            NeuronFingerprint {
                hash: hasher.finish(),
            },
        );
    }

    fingerprints
}

/// Filter focus neurons to only those whose fingerprints have changed.
///
/// Compares current fingerprints against `previous_fingerprints` from the
/// last discovery run. Neurons with no previous fingerprint (new neurons)
/// are always included.
///
/// Returns a `FilterResult` with changed/skipped lists and hit/miss counts.
pub fn filter_changed_neurons(
    focus_neurons: &[String],
    creature: &CreatureJson,
    previous_fingerprints: &HashMap<String, NeuronFingerprint>,
) -> FilterResult {
    let current_fingerprints = compute_neuron_fingerprints(creature);

    let mut changed = Vec::new();
    let mut skipped_uuids = Vec::new();
    let mut cache_hits = 0usize;
    let mut cache_misses = 0usize;

    for uuid in focus_neurons {
        let current = current_fingerprints.get(uuid);
        let previous = previous_fingerprints.get(uuid);

        match (current, previous) {
            (Some(cur), Some(prev)) if cur == prev => {
                // Unchanged — skip this neuron
                cache_hits += 1;
                skipped_uuids.push(uuid.clone());
            }
            _ => {
                // Changed, new, or removed — re-analyse
                cache_misses += 1;
                changed.push(uuid.clone());
            }
        }
    }

    FilterResult {
        changed,
        skipped_uuids,
        cache_hits,
        cache_misses,
        total_focus_neurons: focus_neurons.len(),
    }
}
