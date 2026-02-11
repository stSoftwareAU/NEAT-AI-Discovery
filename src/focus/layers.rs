//! Network layer computation via BFS.
//!
//! Organises neurons by their depth from inputs, measured in synapse hops.
//! Used by hierarchical focus selection to guarantee coverage across all
//! network depths.

use crate::CreatureJson;
use crate::intern::NeuronIndex;

use std::collections::{HashMap, HashSet, VecDeque};

/// Represents a layer of neurons at a specific depth in the network.
///
/// Neurons are grouped by their distance from inputs (measured in synapse hops).
/// This enables hierarchical focus selection that guarantees coverage across
/// all network depths.
#[derive(Debug, Clone)]
pub struct NeuronLayer {
    /// Distance from inputs (0 = directly connected to inputs)
    pub depth: usize,
    /// Neurons at this depth (references to creature's neuron list)
    pub neurons: Vec<NeuronInfo>,
}

/// Basic neuron information for layer assignment.
#[derive(Debug, Clone)]
pub struct NeuronInfo {
    pub uuid: String,
    pub neuron_type: String,
}

/// Compute network layers by organising neurons based on their depth from inputs.
///
/// Depth is computed using BFS from input neurons:
/// - Depth 0: Input neurons (not included in returned layers as they're not selectable)
/// - Depth 1: Neurons directly connected to inputs
/// - Depth N: Neurons at N hops from any input
///
/// # Algorithm
///
/// Uses BFS to compute the MAXIMUM depth for each neuron. When a neuron has
/// multiple incoming paths, we use the longest path to determine its layer.
/// This ensures output-adjacent neurons are in deeper layers even if they
/// have short paths from some inputs.
///
/// # Handling edge cases
///
/// - **Cycles**: Detected and handled by tracking visited nodes. A node in a cycle
///   gets its depth from the first path that reaches it.
/// - **Disconnected components**: Neurons not reachable from inputs are assigned
///   to a special "unreachable" layer (depth = usize::MAX, sorted last).
///
/// # Arguments
/// * `creature` - The creature to analyse
///
/// # Returns
/// Vector of NeuronLayers sorted by depth (shallowest first)
pub fn compute_network_layers(creature: &CreatureJson) -> Vec<NeuronLayer> {
    // Issue #210: Use NeuronIndex for memory-efficient BFS traversal.
    // Instead of cloning UUID strings for the adjacency map and queue,
    // we use u32 indices (4 bytes vs ~36+ bytes per String).
    let mut neuron_index = NeuronIndex::with_capacity(creature.neurons.len() + creature.input);

    // Pre-intern all neuron UUIDs
    for neuron in &creature.neurons {
        neuron_index.intern(&neuron.uuid);
    }

    // Build neuron UUID set using indices for efficient lookup
    let neuron_indices: HashSet<u32> = creature
        .neurons
        .iter()
        .map(|n| neuron_index.get_index(&n.uuid).unwrap())
        .collect();

    // Build forward adjacency using interned indices
    let mut forward_adjacency: HashMap<u32, Vec<u32>> = HashMap::new();
    for synapse in &creature.synapses {
        let from_idx = neuron_index.intern(&synapse.from_uuid);
        let to_idx = neuron_index.intern(&synapse.to_uuid);
        forward_adjacency.entry(from_idx).or_default().push(to_idx);
    }

    // Identify input neuron indices (those not in the neuron list)
    let input_indices: HashSet<u32> = creature
        .synapses
        .iter()
        .map(|s| neuron_index.get_index(&s.from_uuid).unwrap())
        .filter(|idx| !neuron_indices.contains(idx))
        .collect();

    // BFS to compute depth from inputs using indices
    let mut depths: HashMap<u32, usize> = HashMap::new();
    let mut queue: VecDeque<u32> = VecDeque::new();

    // Maximum depth to prevent infinite loops with cycles
    let max_depth = creature.neurons.len() + creature.input + 1;

    // Initialise: all input indices have depth 0
    for &idx in &input_indices {
        depths.insert(idx, 0);
        queue.push_back(idx);
    }

    // BFS traversal with cycle protection
    while let Some(idx) = queue.pop_front() {
        let current_depth = depths[&idx];

        // Stop propagating if we've exceeded max depth (cycle detection)
        if current_depth >= max_depth {
            continue;
        }

        if let Some(targets) = forward_adjacency.get(&idx) {
            for &to_idx in targets {
                let new_depth = current_depth + 1;

                // Only update if we haven't seen this node or found a longer path
                // BUT cap at max_depth to handle cycles
                if new_depth <= max_depth {
                    let should_update = match depths.get(&to_idx) {
                        Some(&existing) => new_depth > existing && new_depth <= max_depth,
                        None => true,
                    };

                    if should_update {
                        depths.insert(to_idx, new_depth);
                        queue.push_back(to_idx);
                    }
                }
            }
        }
    }

    // Handle disconnected neurons (not reachable from any input)
    for neuron in &creature.neurons {
        let idx = neuron_index.get_index(&neuron.uuid).unwrap();
        depths.entry(idx).or_insert(usize::MAX);
    }

    // Group neurons by depth
    let mut layers_map: HashMap<usize, Vec<NeuronInfo>> = HashMap::new();
    for neuron in &creature.neurons {
        // Skip input and constant neurons (not selectable)
        if neuron.neuron_type == "input" || neuron.neuron_type == "constant" {
            continue;
        }

        let idx = neuron_index.get_index(&neuron.uuid).unwrap();
        let depth = depths.get(&idx).copied().unwrap_or(usize::MAX);
        layers_map.entry(depth).or_default().push(NeuronInfo {
            uuid: neuron.uuid.clone(),
            neuron_type: neuron.neuron_type.clone(),
        });
    }

    // Convert to sorted vector
    let mut layers: Vec<NeuronLayer> = layers_map
        .into_iter()
        .map(|(depth, neurons)| NeuronLayer { depth, neurons })
        .collect();

    // Sort by depth (shallowest first, unreachable last)
    layers.sort_by_key(|l| l.depth);

    layers
}
