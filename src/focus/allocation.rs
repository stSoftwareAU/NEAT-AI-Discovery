//! Budget allocation strategies for hierarchical focus selection.
//!
//! Allocates a total focus budget across network layers using one of three
//! strategies: Equal, Proportional, or `OutputFirst`.

use super::layers::{NeuronInfo, NeuronLayer};
use crate::CreatureJson;

use std::collections::{HashMap, HashSet};

/// Strategy for allocating focus budget across layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Equal allocation per layer (total / `num_layers`)
    Equal,
    /// Proportional to layer size (larger layers get more slots)
    Proportional,
    /// Prioritise output layers (allocate from deepest to shallowest)
    OutputFirst,
}

/// Threshold for using hierarchical selection (number of selectable neurons).
/// Below this threshold, flat selection is used for simplicity.
/// Above this threshold, hierarchical selection provides better coverage.
pub const HIERARCHICAL_SELECTION_THRESHOLD: usize = 100;

/// Allocate focus budget across layers based on the specified strategy.
///
/// # Arguments
/// * `total` - Total focus budget to allocate
/// * `layers` - Network layers (from `compute_network_layers`)
/// * `strategy` - Allocation strategy to use
///
/// # Returns
/// Vector of allocations, one per layer, in the same order as `layers`
fn allocate_focus_budget(
    total: usize,
    layers: &[NeuronLayer],
    strategy: AllocationStrategy,
) -> Vec<usize> {
    if layers.is_empty() {
        return Vec::new();
    }

    match strategy {
        AllocationStrategy::Equal => {
            // Divide equally among layers
            let per_layer = total / layers.len();
            let remainder = total % layers.len();
            let mut alloc = vec![per_layer; layers.len()];

            // Distribute remainder to last layers (closer to output)
            for i in 0..remainder {
                let idx = layers.len() - 1 - i;
                alloc[idx] += 1;
            }
            alloc
        }

        AllocationStrategy::Proportional => {
            // Allocate proportionally to layer size
            let total_neurons: usize = layers.iter().map(|l| l.neurons.len()).sum();
            if total_neurons == 0 {
                return vec![0; layers.len()];
            }

            let mut alloc: Vec<usize> = layers
                .iter()
                .map(|l| (total * l.neurons.len()) / total_neurons)
                .collect();

            // Distribute any remainder
            let allocated: usize = alloc.iter().sum();
            let remainder = total.saturating_sub(allocated);
            for i in 0..remainder {
                let idx = layers.len() - 1 - (i % layers.len());
                alloc[idx] += 1;
            }
            alloc
        }

        AllocationStrategy::OutputFirst => {
            // Prioritise layers closest to output (deepest first)
            let mut alloc = vec![0usize; layers.len()];
            let mut remaining = total;

            // Process from deepest to shallowest
            for i in (0..layers.len()).rev() {
                if remaining == 0 {
                    break;
                }
                // Take at most half of remaining budget for this layer,
                // but cap at layer size
                let take = (remaining / 2).max(1).min(layers[i].neurons.len());
                alloc[i] = take;
                remaining = remaining.saturating_sub(take);
            }

            // If we still have budget, distribute to layers that can take more
            for i in (0..layers.len()).rev() {
                if remaining == 0 {
                    break;
                }
                let can_take = layers[i].neurons.len().saturating_sub(alloc[i]);
                let take = can_take.min(remaining);
                alloc[i] += take;
                remaining = remaining.saturating_sub(take);
            }

            alloc
        }
    }
}

/// Select focus neurons hierarchically, ensuring coverage across all network layers.
///
/// This function implements hierarchical focus selection for large creatures,
/// addressing the issue where flat selection tends to over-represent certain
/// layers (typically those with the highest raw error scores).
///
/// # Algorithm
///
/// 1. Allocate focus budget across layers using the specified strategy
/// 2. Within each layer, select the top-scoring neurons up to the allocation
/// 3. If a layer doesn't have enough neurons, redistribute its unused slots
///
/// # Arguments
/// * `creature` - The creature (used for validation)
/// * `layers` - Pre-computed network layers from `compute_network_layers`
/// * `max_focus` - Maximum number of neurons to select
/// * `strategy` - How to allocate budget across layers
/// * `scores` - Map from neuron UUID to score (higher = more likely to select)
///
/// # Returns
/// Vector of selected neuron UUIDs
pub fn hierarchical_focus_selection(
    _creature: &CreatureJson,
    layers: &[NeuronLayer],
    max_focus: usize,
    strategy: AllocationStrategy,
    scores: &HashMap<String, f32>,
) -> Vec<String> {
    if layers.is_empty() || max_focus == 0 {
        return Vec::new();
    }

    // Allocate budget across layers
    let allocations = allocate_focus_budget(max_focus, layers, strategy);

    let mut selected: Vec<String> = Vec::with_capacity(max_focus);
    let mut unused_slots = 0usize;

    // Select from each layer
    for (i, layer) in layers.iter().enumerate() {
        let allocation = allocations[i] + unused_slots;
        unused_slots = 0;

        if allocation == 0 {
            continue;
        }

        // Sort neurons in this layer by score (descending)
        let mut layer_neurons: Vec<(&NeuronInfo, f32)> = layer
            .neurons
            .iter()
            .map(|n| (n, scores.get(&n.uuid).copied().unwrap_or(0.0)))
            .collect();

        layer_neurons.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.uuid.cmp(&b.0.uuid)));

        // Select top neurons from this layer
        let to_select = allocation.min(layer_neurons.len());
        for (neuron, _score) in layer_neurons.into_iter().take(to_select) {
            selected.push(neuron.uuid.clone());
        }

        // Track unused slots for redistribution
        unused_slots = allocation.saturating_sub(to_select);
    }

    // If we still have unused slots and haven't reached max_focus,
    // go back and select more from layers that have neurons left
    if selected.len() < max_focus && unused_slots > 0 {
        let already_selected: HashSet<_> = selected.iter().cloned().collect();

        for layer in layers.iter().rev() {
            // Output layers first (reversed order)
            if unused_slots == 0 {
                break;
            }

            let mut remaining: Vec<(&NeuronInfo, f32)> = layer
                .neurons
                .iter()
                .filter(|n| !already_selected.contains(&n.uuid))
                .map(|n| (n, scores.get(&n.uuid).copied().unwrap_or(0.0)))
                .collect();

            remaining.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.uuid.cmp(&b.0.uuid)));

            for (neuron, _score) in remaining {
                if unused_slots == 0 {
                    break;
                }
                selected.push(neuron.uuid.clone());
                unused_slots -= 1;
            }
        }
    }

    // Ensure we don't exceed max_focus
    selected.truncate(max_focus);

    selected
}
