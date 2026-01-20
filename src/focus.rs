use crate::analysis::{check_memory_for_parquet, verbose_enabled};
use crate::parquet_format::{read_all_records_grouped_by_neuron, read_records_from_parquet};
use crate::types::DiscoverRecord;
use crate::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson,
    SynapseJson,
};
use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Statistics for selection-based neurons (MINIMUM, MAXIMUM, IF).
/// Maps (from_uuid, to_uuid) -> win probability (0.0 to 1.0).
/// For MIN/MAX: probability this synapse provides the min/max value.
/// For IF: probability this synapse's branch is taken (condition always 1.0).
pub type SelectionStats = HashMap<(String, String), f32>;

/// A synapse key paired with its weighted activation contribution.
/// Used internally for tracking which synapse wins in MIN/MAX calculations.
type SynapseContribution = ((String, String), f32);

/// Map from observation index to list of synapse contributions for that observation.
/// Used to determine which synapse wins (has min/max value) for each observation.
type ObservationContributions = HashMap<u32, Vec<SynapseContribution>>;

// =============================================================================
// Hierarchical Focus Selection (Issue #222)
// =============================================================================

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

/// Strategy for allocating focus budget across layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Equal allocation per layer (total / num_layers)
    Equal,
    /// Proportional to layer size (larger layers get more slots)
    Proportional,
    /// Prioritise output layers (allocate from deepest to shallowest)
    OutputFirst,
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
    // Build reverse adjacency: to_uuid -> list of from_uuids
    // This lets us trace back from outputs to inputs
    let mut forward_adjacency: HashMap<String, Vec<String>> = HashMap::new();
    for synapse in &creature.synapses {
        forward_adjacency
            .entry(synapse.from_uuid.clone())
            .or_default()
            .push(synapse.to_uuid.clone());
    }

    // Identify input neuron UUIDs (inputs are represented by creature.input count, not in neurons list)
    // Input UUIDs follow the pattern "input-{index}" or similar based on observation slot
    // We need to find all source UUIDs that don't correspond to neurons in the creature
    let neuron_uuids: HashSet<String> = creature.neurons.iter().map(|n| n.uuid.clone()).collect();

    // Find all unique "from" UUIDs that are not in the neuron list - these are inputs/observations
    let input_uuids: HashSet<String> = creature
        .synapses
        .iter()
        .filter(|s| !neuron_uuids.contains(&s.from_uuid))
        .map(|s| s.from_uuid.clone())
        .collect();

    // BFS to compute depth from inputs
    // We use a modified BFS that tracks the maximum depth for each node.
    // To handle cycles, we limit depth updates to prevent infinite loops.
    let mut depths: HashMap<String, usize> = HashMap::new();
    let mut queue: VecDeque<String> = VecDeque::new();

    // Maximum depth to prevent infinite loops with cycles
    // A creature with N neurons can have at most N-1 layers (linear chain)
    let max_depth = creature.neurons.len() + creature.input + 1;

    // Initialise: all input UUIDs have depth 0
    for uuid in &input_uuids {
        depths.insert(uuid.clone(), 0);
        queue.push_back(uuid.clone());
    }

    // BFS traversal with cycle protection
    while let Some(uuid) = queue.pop_front() {
        let current_depth = depths[&uuid];

        // Stop propagating if we've exceeded max depth (cycle detection)
        if current_depth >= max_depth {
            continue;
        }

        if let Some(targets) = forward_adjacency.get(&uuid) {
            for to_uuid in targets {
                let new_depth = current_depth + 1;

                // Only update if we haven't seen this node or found a longer path
                // BUT cap at max_depth to handle cycles
                if new_depth <= max_depth {
                    let should_update = match depths.get(to_uuid) {
                        Some(&existing) => new_depth > existing && new_depth <= max_depth,
                        None => true,
                    };

                    if should_update {
                        depths.insert(to_uuid.clone(), new_depth);
                        queue.push_back(to_uuid.clone());
                    }
                }
            }
        }
    }

    // Handle disconnected neurons (not reachable from any input)
    for neuron in &creature.neurons {
        if !depths.contains_key(&neuron.uuid) {
            depths.insert(neuron.uuid.clone(), usize::MAX);
        }
    }

    // Group neurons by depth
    let mut layers_map: HashMap<usize, Vec<NeuronInfo>> = HashMap::new();
    for neuron in &creature.neurons {
        // Skip input and constant neurons (not selectable)
        if neuron.neuron_type == "input" || neuron.neuron_type == "constant" {
            continue;
        }

        let depth = depths.get(&neuron.uuid).copied().unwrap_or(usize::MAX);
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

/// Allocate focus budget across layers based on the specified strategy.
///
/// # Arguments
/// * `total` - Total focus budget to allocate
/// * `layers` - Network layers (from compute_network_layers)
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

        layer_neurons.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.uuid.cmp(&b.0.uuid))
        });

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

            remaining.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.uuid.cmp(&b.0.uuid))
            });

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

/// Threshold for using hierarchical selection (number of selectable neurons).
/// Below this threshold, flat selection is used for simplicity.
/// Above this threshold, hierarchical selection provides better coverage.
pub const HIERARCHICAL_SELECTION_THRESHOLD: usize = 100;

/// Provides access to recorded discovery data without assuming an in-memory HashMap.
/// Implementations may pre-load all records or stream them on demand with bounded caching.
pub trait RecordProvider: Send + Sync {
    fn get(&self, neuron_uuid: &str) -> Result<Option<Arc<Vec<DiscoverRecord>>>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

type RecordLoader = dyn Fn(&str, &str) -> Result<Vec<DiscoverRecord>> + Send + Sync + 'static;

struct EagerRecordProvider {
    records: HashMap<String, Arc<Vec<DiscoverRecord>>>,
}

impl EagerRecordProvider {
    fn new(records: HashMap<String, Vec<DiscoverRecord>>) -> Self {
        let records = records
            .into_iter()
            .map(|(uuid, mut recs)| {
                recs.sort_by_key(|r| r.obs_index);
                (uuid, Arc::new(recs))
            })
            .collect();
        Self { records }
    }
}

impl RecordProvider for EagerRecordProvider {
    fn get(&self, neuron_uuid: &str) -> Result<Option<Arc<Vec<DiscoverRecord>>>> {
        Ok(self.records.get(neuron_uuid).map(Arc::clone))
    }

    fn len(&self) -> usize {
        self.records.len()
    }
}

struct LazyRecordProvider {
    parquet_file: String,
    cache: Mutex<LazyCache>,
    loader: Arc<RecordLoader>,
}

struct LazyCache {
    entries: HashMap<String, Arc<Vec<DiscoverRecord>>>,
    order: VecDeque<String>,
    capacity: usize,
}

impl LazyCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    fn insert(&mut self, key: String, value: Arc<Vec<DiscoverRecord>>) {
        if self.entries.contains_key(&key) {
            self.order.retain(|k| k != &key);
        }

        self.entries.insert(key.clone(), value);
        self.order.push_back(key);
        self.evict();
    }

    fn evict(&mut self) {
        while self.entries.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }
}

impl LazyRecordProvider {
    const DEFAULT_CACHE_CAPACITY: usize = 8;

    fn new(parquet_file: &str) -> Self {
        Self {
            parquet_file: parquet_file.to_string(),
            cache: Mutex::new(LazyCache::new(Self::DEFAULT_CACHE_CAPACITY)),
            loader: Arc::new(read_records_from_parquet),
        }
    }

    #[cfg(test)]
    fn with_loader_for_tests(
        parquet_file: &str,
        capacity: usize,
        loader: Arc<RecordLoader>,
    ) -> Self {
        Self {
            parquet_file: parquet_file.to_string(),
            cache: Mutex::new(LazyCache::new(capacity)),
            loader,
        }
    }
}

impl RecordProvider for LazyRecordProvider {
    fn get(&self, neuron_uuid: &str) -> Result<Option<Arc<Vec<DiscoverRecord>>>> {
        {
            let cache = self.cache.lock().expect("lazy record cache poisoned");
            if let Some(records) = cache.entries.get(neuron_uuid) {
                return Ok(Some(Arc::clone(records)));
            }
        }

        let mut records = (self.loader)(&self.parquet_file, neuron_uuid).with_context(|| {
            format!(
                "Failed to load discovery records for {neuron_uuid} from {}",
                self.parquet_file
            )
        })?;
        if records.is_empty() {
            return Ok(None);
        }
        records.sort_by_key(|r| r.obs_index);
        let arc_records = Arc::new(records);

        let mut cache = self.cache.lock().expect("lazy record cache poisoned");
        cache.insert(neuron_uuid.to_string(), Arc::clone(&arc_records));
        Ok(Some(arc_records))
    }

    fn len(&self) -> usize {
        let cache = self.cache.lock().expect("lazy record cache poisoned");
        cache.entries.len()
    }
}

fn get_records_or_error(
    provider: &dyn RecordProvider,
    neuron_uuid: &str,
) -> Result<Arc<Vec<DiscoverRecord>>> {
    provider
        .get(neuron_uuid)?
        .ok_or_else(|| anyhow!("Missing discovery records for selectable neuron: {neuron_uuid}"))
}

#[derive(Debug)]
pub struct RankedNeuron {
    pub neuron_uuid: String,
    pub total_error: f32,
    /// Unclamped mean absolute error from recorded samples.
    ///
    /// We clamp `total_error` for focus ranking so hidden neurons with extreme raw errors do not
    /// dominate selection purely due to scale differences. However, when proposing exploratory
    /// ablation candidates we still want access to the true magnitude.
    pub raw_error: f32,
    /// Structural impact based on weight paths to output
    pub impact: f32,
    /// Mean absolute activation value from recorded samples
    pub mean_activation: f32,
    /// Activation-weighted impact = structural_impact × mean_activation
    /// This reflects the actual contribution the neuron makes during inference
    pub activation_weighted_impact: f32,
}

/// A neuron with activation-weighted impact below removal savings threshold - candidate for removal.
/// Removing such neurons improves score because complexity reduction outweighs contribution.
#[derive(Debug)]
pub struct RemovalCandidate {
    pub neuron_uuid: String,
    pub total_error: f32,
    /// Structural impact based on weight paths to output
    pub impact: f32,
    /// Mean absolute activation value from recorded samples
    pub mean_activation: f32,
    /// Activation-weighted impact = structural_impact × mean_activation
    /// This reflects the actual contribution the neuron makes during inference
    pub activation_weighted_impact: f32,
    /// Number of synapses pointing TO this neuron
    pub incoming_synapses: usize,
    /// Number of synapses pointing FROM this neuron
    pub outgoing_synapses: usize,
    /// The complexity savings from removing this neuron (based on NEAT-AI Score.ts formula)
    pub removal_savings: f32,
    /// Expected creature-level error reduction from removing this neuron.
    /// This is based on activation_weighted_impact, NOT the neuron's error.
    /// Issue #117: Previously, total_error was incorrectly used as expected error reduction,
    /// leading to predictions like 27% when actual reduction was ~0%.
    pub expected_error_reduction: f32,
    pub reason: String,
}

#[derive(Debug)]
pub struct RankFocusStats {
    pub neurons: Vec<RankedNeuron>,
    /// Neurons with impact below costOfGrowth - candidates for removal
    pub removal_candidates: Vec<RemovalCandidate>,
    /// Issue #306: Coordinated structural candidates for removing constant-value neurons.
    /// When a hidden neuron has near-zero activation variance (constant output), it can be
    /// removed and its effect folded into bias adjustments for downstream neurons.
    /// Each candidate contains:
    /// - A RemoveNeuron operation for the constant neuron
    /// - SetBias operations for all downstream neurons with adjusted biases
    pub constant_neuron_removals: Vec<CoordinatedStructuralCandidateJson>,
    pub max_output_error: f32,
    pub processed_neurons: usize,
    pub total_neurons: usize,
    pub duration_ms: u128,
}

fn is_selectable_type(neuron_type: &str) -> bool {
    neuron_type != "input" && neuron_type != "constant"
}

/// Calculate the complexity savings from removing a neuron.
///
/// Based on NEAT-AI's Score.ts formula:
/// ```typescript
/// const complexityPenalty = hiddenNeuronCount * growthCost +
///     creature.synapses.length * growthCost / 10 + penalty * growthCost / 100;
/// ```
///
/// So removing a neuron with N incoming and M outgoing synapses saves:
/// - `growth_cost` for the neuron itself
/// - `(N + M) × growth_cost / 10` for the synapses
///
/// Total: `growth_cost × (1 + (N + M) / 10)`
///
/// # Arguments
/// * `incoming_synapses` - Number of synapses pointing TO this neuron
/// * `outgoing_synapses` - Number of synapses pointing FROM this neuron
/// * `growth_cost` - The cost per hidden neuron (typically 1e-7)
///
/// # Returns
/// The total complexity savings from removing this neuron and its synapses
pub fn calculate_removal_savings(
    incoming_synapses: usize,
    outgoing_synapses: usize,
    growth_cost: f32,
) -> f32 {
    let total_synapses = incoming_synapses + outgoing_synapses;
    growth_cost * (1.0 + total_synapses as f32 / 10.0)
}

/// Count the incoming and outgoing synapses for a neuron.
///
/// # Arguments
/// * `neuron_uuid` - The UUID of the neuron to count synapses for
/// * `creature` - The creature containing the synapses
///
/// # Returns
/// A tuple of (incoming_count, outgoing_count)
fn count_synapses_for_neuron(neuron_uuid: &str, creature: &CreatureJson) -> (usize, usize) {
    let incoming = creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == neuron_uuid)
        .count();
    let outgoing = creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid == neuron_uuid)
        .count();
    (incoming, outgoing)
}

fn average_absolute_error_from_records(records: &[DiscoverRecord]) -> f32 {
    let mut sum = 0.0f32;
    let mut count: u32 = 0;

    for record in records {
        for err in &record.errors {
            if err.is_finite() {
                sum += err.abs();
                count += 1;
            }
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

/// Compute mean absolute activation from discovery records.
/// Sum of |activation| divided by number of finite records.
///
/// Non-finite values (NaN, Infinity) are filtered out to prevent
/// corruption of activation_weighted_impact calculations and sorting.
fn mean_absolute_activation_from_records(records: &[DiscoverRecord]) -> f32 {
    if records.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0f32;
    let mut count: u32 = 0;

    for record in records {
        if record.activation.is_finite() {
            sum += record.activation.abs();
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

/// Compute activation variance and mean from discovery records.
///
/// Issue #306: Used to detect constant-value neurons for removal with bias adjustments.
/// A neuron with near-zero variance has constant activation and can be removed,
/// with its effect folded into bias adjustments for downstream neurons.
///
/// # Returns
/// A tuple of (mean_activation, variance) where:
/// - `mean_activation` is the arithmetic mean (NOT absolute value)
/// - `variance` is the statistical variance of activations
///
/// Returns (0.0, 0.0) if there are insufficient records.
fn activation_mean_and_variance_from_records(records: &[DiscoverRecord]) -> (f32, f32) {
    if records.len() < 2 {
        return (0.0, 0.0);
    }

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0u32;

    for record in records {
        if record.activation.is_finite() {
            let a = record.activation as f64;
            sum += a;
            sum_sq += a * a;
            count += 1;
        }
    }

    if count < 2 {
        return (0.0, 0.0);
    }

    let n = count as f64;
    let mean = sum / n;
    let variance = (sum_sq / n) - (mean * mean);

    (mean as f32, variance.max(0.0) as f32)
}

/// Threshold for considering a neuron as "constant" (near-zero variance).
/// Issue #217/306: Neurons with variance below this threshold are treated as constant
/// and can be removed with bias adjustments for downstream neurons.
///
/// Value 1e-10 is from Issue #217's proposal for DEAD_VARIANCE_THRESHOLD.
const CONSTANT_VARIANCE_THRESHOLD: f32 = 1e-10;

fn build_adjacency(creature: &CreatureJson) -> HashMap<String, Vec<(String, f32)>> {
    let mut adjacency: HashMap<String, Vec<(String, f32)>> = HashMap::new();
    for synapse in &creature.synapses {
        adjacency
            .entry(synapse.from_uuid.clone())
            .or_default()
            .push((synapse.to_uuid.clone(), synapse.weight));
    }
    adjacency
}

/// Build a map from neuron UUID to squash function name.
fn build_squash_map(creature: &CreatureJson) -> HashMap<String, String> {
    creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.squash.to_uppercase()))
        .collect()
}

/// Categorise squash functions for impact calculation.
/// See docs/IMPACT_CALCULATION.md for detailed explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SquashCategory {
    /// Linear or approximately linear (IDENTITY, TANH, etc.)
    /// Impact = normalised weight fraction
    Linear,
    /// Threshold functions (STEP, BIPOLAR)
    /// Any input could flip the output - don't normalise
    Threshold,
    /// Selection functions (MINIMUM, MAXIMUM)
    /// Only one synapse "wins" - conservative: don't normalise
    Selection,
}

impl SquashCategory {
    fn from_squash(squash: &str) -> Self {
        match squash.to_uppercase().as_str() {
            "STEP" | "BIPOLAR" => Self::Threshold,
            "MINIMUM" | "MAXIMUM" | "IF" => Self::Selection,
            _ => Self::Linear,
        }
    }
}

/// Compute selection statistics for MINIMUM, MAXIMUM, and IF neurons using activation records.
///
/// For each selection-based neuron, this function analyses the recorded activations to determine
/// which synapse "wins" (provides the min/max value) for each observation. The result is a map
/// from (from_uuid, to_uuid) to the probability (0.0 to 1.0) that synapse wins.
///
/// For IF neurons with synapse types:
/// - "condition" synapses: Always contribute, so probability = 1.0
/// - "positive" synapses: Probability = fraction of observations where condition sum > 0
/// - "negative" synapses: Probability = fraction of observations where condition sum <= 0
///
/// # Arguments
/// * `creature` - The creature containing neurons and synapses
/// * `grouped_records` - Activation records grouped by neuron UUID
///
/// # Returns
/// Map from (from_uuid, to_uuid) to win probability for selection-based synapses
pub fn compute_selection_stats(
    creature: &CreatureJson,
    grouped_records: &dyn RecordProvider,
) -> Result<SelectionStats> {
    let squash_map = build_squash_map(creature);

    // Find all selection-based neurons (MINIMUM, MAXIMUM, IF)
    let selection_neurons: Vec<&NeuronJson> = creature
        .neurons
        .iter()
        .filter(|n| {
            let squash = squash_map.get(&n.uuid).map(|s| s.as_str()).unwrap_or("");
            matches!(squash, "MINIMUM" | "MAXIMUM" | "IF")
        })
        .collect();

    // Process each selection neuron in parallel and collect local stats
    let partial_stats: Vec<SelectionStats> = selection_neurons
        .par_iter()
        .map(|target_neuron| -> Result<Option<SelectionStats>> {
            let squash = squash_map
                .get(&target_neuron.uuid)
                .map(|s| s.as_str())
                .unwrap_or("");

            // Get incoming synapses to this neuron
            let incoming_synapses: Vec<&SynapseJson> = creature
                .synapses
                .iter()
                .filter(|s| s.to_uuid == target_neuron.uuid)
                .collect();

            if incoming_synapses.is_empty() {
                return Ok(None);
            }

            let mut local_stats = SelectionStats::new();
            match squash {
                "MINIMUM" => {
                    compute_min_stats(&incoming_synapses, grouped_records, &mut local_stats)?
                }
                "MAXIMUM" => {
                    compute_max_stats(&incoming_synapses, grouped_records, &mut local_stats)?
                }
                "IF" => compute_if_stats(&incoming_synapses, grouped_records, &mut local_stats)?,
                _ => {}
            }
            Ok(Some(local_stats))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    // Merge all partial stats into final result
    let mut stats = SelectionStats::new();
    for partial in partial_stats {
        stats.extend(partial);
    }
    Ok(stats)
}

/// Compute selection statistics for a MINIMUM neuron.
/// Counts how often each synapse provides the minimum weighted activation.
fn compute_min_stats(
    synapses: &[&SynapseJson],
    grouped_records: &dyn RecordProvider,
    stats: &mut SelectionStats,
) -> Result<()> {
    if synapses.is_empty() {
        return Ok(());
    }

    // Build a map of obs_index -> Vec<(synapse_key, weighted_activation)>
    let mut obs_contributions: ObservationContributions = HashMap::new();

    for synapse in synapses {
        match grouped_records.get(&synapse.from_uuid)? {
            None => continue,
            Some(records) => {
                for record in records.iter() {
                    if record.activation.is_finite() {
                        let weighted = synapse.weight * record.activation;
                        let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
                        obs_contributions
                            .entry(record.obs_index)
                            .or_default()
                            .push((key, weighted));
                    }
                }
            }
        }
    }

    // Count wins for each synapse
    let mut win_counts: HashMap<(String, String), u32> = HashMap::new();
    let mut total_obs = 0u32;

    for contributions in obs_contributions.values() {
        if contributions.is_empty() {
            continue;
        }
        total_obs += 1;

        // Find the minimum weighted activation
        let min_val = contributions
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::INFINITY, f32::min);

        // Count all synapses that achieved the minimum (handles ties)
        let winners: Vec<_> = contributions
            .iter()
            .filter(|(_, v)| (*v - min_val).abs() < 1e-10)
            .collect();

        for (key, _) in winners {
            *win_counts.entry(key.clone()).or_insert(0) += 1;
        }
    }

    // Convert counts to probabilities
    if total_obs > 0 {
        for synapse in synapses {
            let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
            let wins = win_counts.get(&key).copied().unwrap_or(0);
            let probability = wins as f32 / total_obs as f32;
            stats.insert(key, probability);
        }
    }
    Ok(())
}

/// Compute selection statistics for a MAXIMUM neuron.
/// Counts how often each synapse provides the maximum weighted activation.
fn compute_max_stats(
    synapses: &[&SynapseJson],
    grouped_records: &dyn RecordProvider,
    stats: &mut SelectionStats,
) -> Result<()> {
    if synapses.is_empty() {
        return Ok(());
    }

    // Build a map of obs_index -> Vec<(synapse_key, weighted_activation)>
    let mut obs_contributions: ObservationContributions = HashMap::new();

    for synapse in synapses {
        match grouped_records.get(&synapse.from_uuid)? {
            None => continue,
            Some(records) => {
                for record in records.iter() {
                    if record.activation.is_finite() {
                        let weighted = synapse.weight * record.activation;
                        let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
                        obs_contributions
                            .entry(record.obs_index)
                            .or_default()
                            .push((key, weighted));
                    }
                }
            }
        }
    }

    // Count wins for each synapse
    let mut win_counts: HashMap<(String, String), u32> = HashMap::new();
    let mut total_obs = 0u32;

    for contributions in obs_contributions.values() {
        if contributions.is_empty() {
            continue;
        }
        total_obs += 1;

        // Find the maximum weighted activation
        let max_val = contributions
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);

        // Count all synapses that achieved the maximum (handles ties)
        let winners: Vec<_> = contributions
            .iter()
            .filter(|(_, v)| (*v - max_val).abs() < 1e-10)
            .collect();

        for (key, _) in winners {
            *win_counts.entry(key.clone()).or_insert(0) += 1;
        }
    }

    // Convert counts to probabilities
    if total_obs > 0 {
        for synapse in synapses {
            let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
            let wins = win_counts.get(&key).copied().unwrap_or(0);
            let probability = wins as f32 / total_obs as f32;
            stats.insert(key, probability);
        }
    }
    Ok(())
}

/// Compute selection statistics for an IF neuron.
///
/// IF neurons have three synapse types:
/// - "condition": Always evaluated to determine which branch to take
/// - "positive": Used when sum of condition synapses > 0
/// - "negative": Used when sum of condition synapses <= 0
///
/// Impact distribution:
/// - Condition synapses: probability = 1.0 (always active)
/// - Positive synapses: probability = fraction of observations where condition > 0
/// - Negative synapses: probability = fraction of observations where condition <= 0
fn compute_if_stats(
    synapses: &[&SynapseJson],
    grouped_records: &dyn RecordProvider,
    stats: &mut SelectionStats,
) -> Result<()> {
    // Separate synapses by type
    let condition_synapses: Vec<_> = synapses
        .iter()
        .filter(|s| s.synapse_type.as_deref() == Some("condition"))
        .collect();
    let positive_synapses: Vec<_> = synapses
        .iter()
        .filter(|s| s.synapse_type.as_deref() == Some("positive"))
        .collect();
    let negative_synapses: Vec<_> = synapses
        .iter()
        .filter(|s| s.synapse_type.as_deref() == Some("negative"))
        .collect();

    // If no synapse types are set, fall back to equal probability
    if condition_synapses.is_empty() && positive_synapses.is_empty() && negative_synapses.is_empty()
    {
        // No type information - use equal probability fallback
        let n = synapses.len() as f32;
        for synapse in synapses {
            let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
            stats.insert(key, 1.0 / n);
        }
        return Ok(());
    }

    // Condition synapses are always active
    for synapse in &condition_synapses {
        let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
        stats.insert(key, 1.0);
    }

    // Compute condition sum for each observation to determine positive/negative branch usage
    let mut obs_condition_sums: HashMap<u32, f32> = HashMap::new();

    for synapse in &condition_synapses {
        match grouped_records.get(&synapse.from_uuid)? {
            None => continue,
            Some(records) => {
                for record in records.iter() {
                    if record.activation.is_finite() {
                        let contribution = synapse.weight * record.activation;
                        *obs_condition_sums.entry(record.obs_index).or_insert(0.0) += contribution;
                    }
                }
            }
        }
    }

    // Count positive vs negative branch usage
    let total_obs = obs_condition_sums.len() as f32;
    if total_obs == 0.0 {
        // No observations - use equal probability for positive/negative
        let pos_count = positive_synapses.len().max(1) as f32;
        let neg_count = negative_synapses.len().max(1) as f32;

        for synapse in &positive_synapses {
            let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
            stats.insert(key, 0.5 / pos_count);
        }
        for synapse in &negative_synapses {
            let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
            stats.insert(key, 0.5 / neg_count);
        }
        return Ok(());
    }

    let positive_obs = obs_condition_sums
        .values()
        .filter(|&&sum| sum > 0.0)
        .count() as f32;
    let negative_obs = total_obs - positive_obs;

    let positive_prob = positive_obs / total_obs;
    let negative_prob = negative_obs / total_obs;

    // Distribute probability among synapses in each branch
    // Each synapse in a branch shares that branch's probability equally
    let pos_count = positive_synapses.len().max(1) as f32;
    let neg_count = negative_synapses.len().max(1) as f32;

    for synapse in &positive_synapses {
        let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
        stats.insert(key, positive_prob / pos_count);
    }

    for synapse in &negative_synapses {
        let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
        stats.insert(key, negative_prob / neg_count);
    }
    Ok(())
}

// NOTE: build_inbound_weights was removed in v0.1.126 as part of the impact
// calculation fix. The normalisation it supported was causing massive
// underestimation of neuron impact (see regression_v0_1_126.rs tests).

/// Public version of compute_impacts for use in add-neuron analysis.
/// Computes the structural impact of each neuron on outputs (path weight products).
/// Output neurons have impact = 1.0, hidden neurons have impact in [0, 1] based on
/// their weighted paths to outputs.
///
/// NOTE: This function is squash-aware. For neurons feeding into STEP/BIPOLAR/MINIMUM/MAXIMUM
/// targets, the impact calculation uses special handling to avoid underestimation.
/// See docs/IMPACT_CALCULATION.md for detailed explanation.
pub fn compute_impacts_public(creature: &CreatureJson) -> HashMap<String, f32> {
    compute_impacts_internal(creature)
}

/// Context for impact calculation, containing pre-computed lookup tables.
/// This struct groups related parameters to avoid clippy::too_many_arguments.
struct ImpactContext {
    adjacency: HashMap<String, Vec<(String, f32)>>,
    inbound_count: HashMap<String, usize>,
    /// Sum of |weight| for all synapses INTO each target neuron.
    /// Used for normalising Linear squash impact: |w| / total_inbound_weight × child_impact
    /// This ensures hidden neurons always have impact < 1.0 (Issue #130).
    total_inbound_weight: HashMap<String, f32>,
    squash_map: HashMap<String, String>,
    outputs: HashSet<String>,
    /// Selection statistics from activation records.
    /// When available, provides actual win probabilities for MIN/MAX/IF synapses
    /// instead of the conservative 1/N equal probability fallback.
    selection_stats: Option<SelectionStats>,
}

fn compute_impacts_internal(creature: &CreatureJson) -> HashMap<String, f32> {
    compute_impacts_internal_with_stats(creature, None)
        .expect("impact computation without records should not fail")
}

/// Compute impacts with optional activation-based selection statistics.
///
/// When `grouped_records` is provided, computes actual selection probabilities for
/// MIN/MAX/IF neurons based on recorded activations. This gives more accurate
/// impact estimates than the conservative 1/N equal probability fallback.
///
/// # Arguments
/// * `creature` - The creature to compute impacts for
/// * `grouped_records` - Optional activation records grouped by neuron UUID
///
/// # Returns
/// Map from neuron UUID to impact score
fn compute_impacts_internal_with_stats(
    creature: &CreatureJson,
    grouped_records: Option<&dyn RecordProvider>,
) -> Result<HashMap<String, f32>> {
    let adjacency = build_adjacency(creature);
    let squash_map = build_squash_map(creature);

    // Build inbound synapse count for selection squashes (MIN/MAX/IF neurons)
    let inbound_count: HashMap<String, usize> = {
        let mut map: HashMap<String, usize> = HashMap::new();
        for synapse in &creature.synapses {
            *map.entry(synapse.to_uuid.clone()).or_insert(0) += 1;
        }
        map
    };

    // Build total inbound weight for Linear squash normalisation (Issue #130).
    // Sum of |weight| for all synapses INTO each target neuron.
    // This ensures hidden neurons always have impact < 1.0:
    //   contribution = |weight| / total_inbound_weight × child_impact
    let total_inbound_weight: HashMap<String, f32> = {
        let mut map: HashMap<String, f32> = HashMap::new();
        for synapse in &creature.synapses {
            *map.entry(synapse.to_uuid.clone()).or_insert(0.0) += synapse.weight.abs();
        }
        map
    };

    let outputs: HashSet<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone())
        .collect();

    // Compute selection statistics if activation records are available
    let selection_stats = grouped_records
        .map(|records| compute_selection_stats(creature, records))
        .transpose()?;

    let ctx = ImpactContext {
        adjacency,
        inbound_count,
        total_inbound_weight,
        squash_map,
        outputs,
        selection_stats,
    };

    // Parallel impact computation using thread-local caches
    // Each thread computes impacts for a subset of neurons, then we merge results.
    // This trades some redundant computation for better CPU utilization.
    //
    // Note: We include ALL neurons (including inputs) to compute their impacts.
    // Input neurons have outgoing synapses and their impact measures their
    // contribution to the final output. This is useful for visualisation/debugging.
    let all_neurons: Vec<&NeuronJson> = creature.neurons.iter().collect();

    // Use a shared cache protected by a mutex for thread-safe updates
    let shared_cache: Mutex<HashMap<String, f32>> = Mutex::new(HashMap::new());

    all_neurons.par_iter().for_each(|neuron| {
        // Check if already computed (another thread might have done it)
        {
            let cache = shared_cache.lock().unwrap();
            if cache.contains_key(&neuron.uuid) {
                return;
            }
        }

        // Compute with a local visiting set (cycle detection is per-path)
        let mut visiting = HashSet::new();

        // We need to compute recursively, but with shared cache access
        let impact =
            compute_impact_with_shared_cache(&neuron.uuid, &ctx, &shared_cache, &mut visiting);

        // Store result
        let mut cache = shared_cache.lock().unwrap();
        cache.insert(neuron.uuid.clone(), impact);
    });

    Ok(shared_cache.into_inner().unwrap())
}

/// Compute impact with a shared cache for parallel execution.
fn compute_impact_with_shared_cache(
    uuid: &str,
    ctx: &ImpactContext,
    shared_cache: &Mutex<HashMap<String, f32>>,
    visiting: &mut HashSet<String>,
) -> f32 {
    // Check cache first
    {
        let cache = shared_cache.lock().unwrap();
        if let Some(&value) = cache.get(uuid) {
            return value;
        }
    }

    if !visiting.insert(uuid.to_string()) {
        // Cycle detected; treat as zero contribution
        return 0.0;
    }

    let impact = if ctx.outputs.contains(uuid) {
        1.0
    } else if let Some(edges) = ctx.adjacency.get(uuid) {
        // Sum across all outgoing edges
        let mut total_impact = 0.0;
        for (to_uuid, weight) in edges {
            let child_impact =
                compute_impact_with_shared_cache(to_uuid, ctx, shared_cache, visiting);
            if child_impact <= 0.0 {
                continue;
            }

            let squash = ctx
                .squash_map
                .get(to_uuid)
                .map(|s| s.as_str())
                .unwrap_or("IDENTITY");
            let category = SquashCategory::from_squash(squash);

            let contribution = match category {
                SquashCategory::Linear => {
                    // Issue #130: Normalise by total inbound weight to ensure hidden neurons
                    // always have impact < 1.0. This matches the documented formula:
                    //   contribution = |weight| / total_inbound_weight × child_impact
                    //
                    // Without normalisation, a hidden neuron with weight 3.0 to an output
                    // would get impact = 3.0, which is mathematically incorrect for the
                    // PURPOSE of prediction discounting (measuring fraction of influence).
                    //
                    // Edge case: If all inbound weights are 0.0, total is 0.0, and we'd get
                    // 0.0 / 0.0 = NaN. Handle this by returning 0.0 (zero weight = zero contribution).
                    //
                    // Near-zero protection: The `.max(weight.abs())` ensures total >= weight.abs(),
                    // so the ratio weight.abs() / total is ALWAYS in [0, 1] and cannot explode.
                    // Example: weight=1e-10, total_inbound=1e-10 → total=1e-10 → ratio=1.0 ✓
                    // The only problematic case is weight=0.0 AND total=0.0 → 0/0=NaN, handled below.
                    let total = ctx
                        .total_inbound_weight
                        .get(to_uuid)
                        .copied()
                        .unwrap_or(1.0)
                        .max(weight.abs()); // Bounds ratio to [0,1]: total >= weight.abs() always

                    if total <= 0.0 {
                        // All weights are zero (including this one) → zero contribution
                        0.0
                    } else {
                        (weight.abs() / total) * child_impact
                    }
                }
                SquashCategory::Threshold => child_impact,
                SquashCategory::Selection => {
                    if let Some(ref stats) = ctx.selection_stats {
                        let key = (uuid.to_string(), to_uuid.clone());
                        let win_prob = stats.get(&key).copied().unwrap_or_else(|| {
                            let n = ctx.inbound_count.get(to_uuid).copied().unwrap_or(1).max(1);
                            1.0 / n as f32
                        });
                        win_prob * child_impact
                    } else {
                        let n = ctx.inbound_count.get(to_uuid).copied().unwrap_or(1).max(1);
                        (1.0 / n as f32) * child_impact
                    }
                }
            };
            total_impact += contribution;
        }
        total_impact
    } else {
        0.0
    };

    visiting.remove(uuid);

    // Cache the result
    {
        let mut cache = shared_cache.lock().unwrap();
        cache.insert(uuid.to_string(), impact);
    }

    impact
}

/// Public version that computes impacts with activation-based selection statistics.
///
/// When activation records are provided, this function computes actual win probabilities
/// for MIN/MAX/IF neurons instead of using the conservative 1/N equal probability.
/// This results in more accurate impact estimates.
///
/// # Arguments
/// * `creature` - The creature to compute impacts for
/// * `grouped_records` - Activation records grouped by neuron UUID
///
/// # Returns
/// Map from neuron UUID to impact score
pub fn compute_impacts_with_activations(
    creature: &CreatureJson,
    grouped_records: &dyn RecordProvider,
) -> Result<HashMap<String, f32>> {
    compute_impacts_internal_with_stats(creature, Some(grouped_records))
}

pub fn rank_focus_neurons(
    parquet_file: &str,
    creature: &CreatureJson,
    max_results: Option<usize>,
    cost_of_growth: Option<f32>,
) -> Result<RankFocusStats> {
    let start = Instant::now();
    let selectable: Vec<&NeuronJson> = creature
        .neurons
        .iter()
        .filter(|neuron| is_selectable_type(&neuron.neuron_type))
        .collect();
    let total_neurons = selectable.len();

    if total_neurons == 0 {
        return Ok(RankFocusStats {
            neurons: Vec::new(),
            removal_candidates: Vec::new(),
            constant_neuron_removals: Vec::new(),
            max_output_error: 0.0,
            processed_neurons: 0,
            total_neurons: 0,
            duration_ms: start.elapsed().as_millis(),
        });
    }

    // Try to load all records if we have enough memory.
    // If not enough memory, fall back to lazy loading with a bounded cache.
    let (records_provider, is_lazy_mode): (Arc<dyn RecordProvider>, bool) =
        match check_memory_for_parquet(parquet_file) {
            Ok(()) => {
                let records = read_all_records_grouped_by_neuron(parquet_file)
                    .context("Failed to read discovery records from parquet file")?;
                (Arc::new(EagerRecordProvider::new(records)), false)
            }
            Err(memory_error) => {
                eprintln!(
                    "[NEAT-AI-Discovery] Insufficient memory for full pre-load in focus ranking. \
                     Using lazy-loading mode (slower but memory-efficient)."
                );
                if verbose_enabled() {
                    eprintln!("[NEAT-AI-Discovery][verbose] Memory check failed: {memory_error}");
                }
                (Arc::new(LazyRecordProvider::new(parquet_file)), true)
            }
        };

    if is_lazy_mode && verbose_enabled() {
        eprintln!(
            "[NEAT-AI-Discovery][verbose] Lazy record cache initialised (cached: {} neurons)",
            records_provider.len()
        );
    }

    // Verify that all selectable neurons have records (restore old error behaviour)
    for neuron in &selectable {
        get_records_or_error(records_provider.as_ref(), &neuron.uuid)
            .context("Failed to read discovery records for all selectable neurons")?;
    }

    let output_neurons: Vec<&NeuronJson> = creature
        .neurons
        .iter()
        .filter(|neuron| neuron.neuron_type == "output")
        .collect();

    let max_output_error = if output_neurons.is_empty() {
        0.0
    } else {
        let errors: Vec<f32> = output_neurons
            .iter()
            .map(|neuron| {
                let records = get_records_or_error(records_provider.as_ref(), &neuron.uuid)?;
                Ok(average_absolute_error_from_records(&records))
            })
            .collect::<Result<Vec<_>>>()?;

        errors.into_iter().fold(0.0, f32::max)
    };

    // Use activation-based impact calculation for more accurate MIN/MAX/IF statistics
    let impact_map = compute_impacts_with_activations(creature, records_provider.as_ref())?;

    // Now we can safely unwrap since we've verified all selectable neurons have records
    // Use parallel iteration for faster processing on multi-core systems
    let mut neurons: Vec<RankedNeuron> = selectable
        .par_iter()
        .map(|neuron| -> Result<RankedNeuron> {
            let records = get_records_or_error(records_provider.as_ref(), &neuron.uuid)?;
            let raw_error = average_absolute_error_from_records(&records);
            let structural_impact = *impact_map.get(&neuron.uuid).unwrap_or(&0.0);
            let mean_activation = mean_absolute_activation_from_records(&records);

            // Activation-weighted impact reflects the ACTUAL contribution during inference.
            // A neuron with tiny structural impact but massive activations still contributes
            // significantly: actual_contribution ≈ weight × activation
            //
            // Only neurons with BOTH low structural impact AND low activation should be
            // removal candidates. If either is high, the neuron is contributing.
            let activation_weighted_impact = structural_impact * mean_activation;

            let total_error = if max_output_error > 0.0 {
                raw_error.min(max_output_error)
            } else {
                raw_error
            };
            Ok(RankedNeuron {
                neuron_uuid: neuron.uuid.clone(),
                total_error,
                raw_error,
                impact: structural_impact,
                mean_activation,
                activation_weighted_impact,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Sort by weighted score (error × impact) to prioritise neurons that:
    // 1. Have high error (potential for improvement)
    // 2. Have high impact (changes will affect output)
    // This ensures output neurons and neurons close to outputs are prioritised
    // over high-error hidden neurons with minimal impact on the creature's score.
    //
    // Dec 2025: We deliberately soften (but do not remove) the output bias by applying a
    // sub-linear exponent to impact. This increases exploration of hidden neurons without
    // letting low-impact neurons dominate purely due to noisy per-neuron errors.
    const IMPACT_EPSILON: f32 = 0.0001;
    const IMPACT_GAMMA: f32 = 0.8;
    neurons.sort_by(|a, b| {
        let a_weighted = a.total_error * (a.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);
        let b_weighted = b.total_error * (b.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);
        b_weighted
            .partial_cmp(&a_weighted)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.impact.partial_cmp(&a.impact).unwrap_or(Ordering::Equal))
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    // Identify removal candidates: neurons with activation_weighted_impact < costOfGrowth.
    //
    // activation_weighted_impact = structural_impact × mean_activation
    // where structural_impact = NORMALISED impact through the network (Issue #130 fix)
    //
    // With normalised impact, structural_impact represents the FRACTION of influence
    // a neuron has on outputs (attribution), always in range [0, 1] per output.
    // This correctly represents how much removing the neuron would affect results.
    //
    // Neurons with impact below costOfGrowth are net negative - removing them
    // reduces complexity more than it affects error.
    //
    // We provide complexity savings info for each neuron based on NEAT-AI's formula:
    //   savings = growthCost × (1 + (N + M) / 10)
    // where N = incoming synapses, M = outgoing synapses
    //
    // Issue #132: costOfGrowth should be passed from NEAT-AI, not hardcoded.
    // The correct default is 1e-7 (per hidden neuron) as per NEAT-AI's Score.ts formula.
    // v0.1.145 incorrectly changed this to 0.01 which caused 418 false removal candidates.
    const DEFAULT_COST_OF_GROWTH: f32 = 1e-7;
    let cost_of_growth_threshold = cost_of_growth.unwrap_or(DEFAULT_COST_OF_GROWTH);

    // Issue #235: Return ALL neurons where removal improves the creature's score.
    //
    // The only filter is: "will this candidate improve the creature's score?"
    // A removal improves score when: removal_savings > activation_weighted_impact
    //
    // Previously, we filtered on activation_weighted_impact < cost_of_growth_threshold,
    // but this missed neurons where removal_savings (due to many synapses) exceeds
    // the neuron's contribution even when activation_weighted_impact is above threshold.
    //
    // Use parallel iteration for faster processing on multi-core systems
    let mut removal_candidates: Vec<RemovalCandidate> = neurons
        .par_iter()
        .filter_map(|n| {
            let (incoming, outgoing) = count_synapses_for_neuron(&n.neuron_uuid, creature);
            let savings = calculate_removal_savings(incoming, outgoing, cost_of_growth_threshold);

            // Issue #235: Filter on savings > impact (removal improves score)
            // instead of impact < threshold (may miss valid candidates)
            if savings <= n.activation_weighted_impact {
                return None;
            }

            // Issue #117: expected_error_reduction should be based on activation_weighted_impact,
            // NOT total_error. The activation_weighted_impact represents the actual contribution
            // this neuron makes to the output. Removing it changes error by approximately this amount.
            //
            // Previously, total_error (neuron's average error) was incorrectly used, leading to
            // predictions like 27% when actual reduction was ~0% (for low-impact neurons).
            let expected_error_reduction = n.activation_weighted_impact;

            // Net score improvement = savings - impact
            let net_improvement = savings - n.activation_weighted_impact;

            Some(RemovalCandidate {
                neuron_uuid: n.neuron_uuid.clone(),
                total_error: n.total_error,
                impact: n.impact,
                mean_activation: n.mean_activation,
                activation_weighted_impact: n.activation_weighted_impact,
                incoming_synapses: incoming,
                outgoing_synapses: outgoing,
                removal_savings: savings,
                expected_error_reduction,
                reason: format!(
                    "Removal improves score: saves {:.2e} > impact {:.2e} (net +{:.2e}), {} synapses, costOfGrowth={:.2e}",
                    savings,
                    n.activation_weighted_impact,
                    net_improvement,
                    incoming + outgoing,
                    cost_of_growth_threshold,
                ),
            })
        })
        .collect();

    // Extra candidates: high-error exploratory ablations
    //
    // Rationale: A neuron can have very high recorded error yet still be high-impact. Removing
    // such a neuron is NOT a "safe prune", but it can be a worthwhile ablation test: clone the
    // creature, remove/disable the neuron, then re-score on the full training set. Keep only
    // if the score improves.
    //
    // We intentionally keep these candidates limited in count and clearly labelled so callers
    // can treat them as exploratory.
    const EXPLORATORY_ABLATION_MAX: usize = 5;
    const EXPLORATORY_ERROR_MULTIPLIER: f32 = 10.0;

    if max_output_error > 0.0 {
        let neuron_types: HashMap<&str, &str> = creature
            .neurons
            .iter()
            .map(|n: &NeuronJson| (n.uuid.as_str(), n.neuron_type.as_str()))
            .collect();

        let already_selected: HashSet<&str> = removal_candidates
            .iter()
            .map(|c| c.neuron_uuid.as_str())
            .collect();

        let mut high_error_neurons: Vec<&RankedNeuron> = neurons
            .iter()
            .filter(|n| !already_selected.contains(n.neuron_uuid.as_str()))
            // Only propose exploratory removals for hidden neurons.
            .filter(|n| neuron_types.get(n.neuron_uuid.as_str()) == Some(&"hidden"))
            // Only when error is meaningfully larger than output error scale.
            .filter(|n| n.raw_error >= max_output_error * EXPLORATORY_ERROR_MULTIPLIER)
            .collect();

        high_error_neurons.sort_by(|a, b| {
            b.raw_error
                .partial_cmp(&a.raw_error)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.impact.partial_cmp(&a.impact).unwrap_or(Ordering::Equal))
                .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
        });

        high_error_neurons.truncate(EXPLORATORY_ABLATION_MAX);

        for n in high_error_neurons {
            let (incoming, outgoing) = count_synapses_for_neuron(&n.neuron_uuid, creature);
            let savings = calculate_removal_savings(incoming, outgoing, cost_of_growth_threshold);

            removal_candidates.push(RemovalCandidate {
                neuron_uuid: n.neuron_uuid.clone(),
                total_error: n.total_error,
                impact: n.impact,
                mean_activation: n.mean_activation,
                activation_weighted_impact: n.activation_weighted_impact,
                incoming_synapses: incoming,
                outgoing_synapses: outgoing,
                removal_savings: savings,
                // Exploratory candidates must not claim a predicted improvement. The controller
                // will run an ablation test on the full training set to validate.
                expected_error_reduction: 0.0,
                reason: format!(
                    "Exploratory ablation candidate (high error): raw_error {:.2e} (clamped {:.2e}), \
                     activation_weighted_impact {:.2e} >= costOfGrowth ({:.2e}). \
                     This is NOT a safe prune - validate by full-dataset ablation test.",
                    n.raw_error,
                    n.total_error,
                    n.activation_weighted_impact,
                    cost_of_growth_threshold
                ),
            });
        }
    }

    // Issue #235: Sort by net improvement (removal_savings - activation_weighted_impact).
    // Higher net improvement = better candidate (removing it saves more than its contribution).
    // Exploratory candidates (from high-error section) have expected_error_reduction = 0.0,
    // so they sort last (their net improvement calculation uses impact directly).
    removal_candidates.sort_by(|a, b| {
        // Calculate net improvement for each candidate
        let a_net = a.removal_savings - a.activation_weighted_impact;
        let b_net = b.removal_savings - b.activation_weighted_impact;

        // Sort by descending net improvement (best candidates first)
        b_net
            .partial_cmp(&a_net)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                // For ties, prefer lower impact (safer removal)
                a.activation_weighted_impact
                    .partial_cmp(&b.activation_weighted_impact)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    if let Some(limit) = max_results {
        if neurons.len() > limit {
            neurons.truncate(limit);
        }
    }

    // Issue #306: Detect constant-value neurons and create coordinated structural candidates
    // that remove the neuron and adjust downstream biases.
    //
    // A neuron with near-zero activation variance is "constant" - it always outputs roughly
    // the same value regardless of input. Removing it is equivalent to adjusting the biases
    // of downstream neurons by: bias_adjustment = synapse_weight × mean_activation
    //
    // This is a win because:
    // 1. We reduce complexity (one less neuron and its synapses)
    // 2. We preserve the network's behaviour (downstream biases compensate)
    // 3. The constant neuron wasn't contributing useful signal anyway
    let neuron_types: HashMap<&str, &str> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.neuron_type.as_str()))
        .collect();

    let neuron_biases: HashMap<&str, f32> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.bias))
        .collect();

    // Build outgoing synapse map: from_uuid -> [(to_uuid, weight)]
    let outgoing_synapses: HashMap<&str, Vec<(&str, f32)>> =
        creature
            .synapses
            .iter()
            .fold(HashMap::new(), |mut map, syn| {
                map.entry(syn.from_uuid.as_str())
                    .or_default()
                    .push((syn.to_uuid.as_str(), syn.weight));
                map
            });

    // Find constant hidden neurons and create coordinated removal candidates
    let constant_neuron_removals: Vec<CoordinatedStructuralCandidateJson> = selectable
        .par_iter()
        .filter_map(|neuron| {
            // Only consider hidden neurons for constant removal
            let neuron_type = neuron_types.get(neuron.uuid.as_str())?;
            if *neuron_type != "hidden" {
                return None;
            }

            // Get records and compute variance
            let records = get_records_or_error(records_provider.as_ref(), &neuron.uuid).ok()?;
            let (mean_activation, variance) = activation_mean_and_variance_from_records(&records);

            // Check if variance is below threshold (constant neuron)
            if variance > CONSTANT_VARIANCE_THRESHOLD {
                return None;
            }

            // Get outgoing synapses for this neuron
            let outgoing = outgoing_synapses.get(neuron.uuid.as_str())?;
            if outgoing.is_empty() {
                return None;
            }

            // Build coordinated structural candidate:
            // 1. SetBias operations for all downstream neurons
            // 2. RemoveNeuron operation
            let mut operations = Vec::with_capacity(outgoing.len() + 1);

            // Add SetBias operations for all downstream neurons
            for (to_uuid, weight) in outgoing {
                let old_bias = neuron_biases.get(to_uuid).copied().unwrap_or(0.0);
                let bias_adjustment = weight * mean_activation;
                let new_bias = old_bias + bias_adjustment;

                if new_bias.is_finite() {
                    operations.push(CoordinatedStructuralOpJson::SetBias {
                        neuron_uuid: to_uuid.to_string(),
                        bias: new_bias,
                    });
                }
            }

            // Add RemoveNeuron operation (must be last so bias adjustments happen first)
            operations.push(CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: neuron.uuid.clone(),
            });

            // Calculate expected improvement: removal savings (complexity reduction)
            let (incoming_count, outgoing_count) =
                count_synapses_for_neuron(&neuron.uuid, creature);
            let removal_savings =
                calculate_removal_savings(incoming_count, outgoing_count, cost_of_growth_threshold);

            Some(CoordinatedStructuralCandidateJson {
                operations,
                expected_creature_score_gain: removal_savings,
                comment: Some(format!(
                    "Issue #306: Constant neuron removal with bias adjustments. \
                     mean_activation={mean_activation:.6}, variance={variance:.2e}, \
                     {} downstream neurons, savings={removal_savings:.2e}",
                    outgoing.len()
                )),
            })
        })
        .collect();

    Ok(RankFocusStats {
        neurons,
        removal_candidates,
        constant_neuron_removals,
        max_output_error,
        processed_neurons: total_neurons,
        total_neurons,
        duration_ms: start.elapsed().as_millis(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn lazy_provider_defers_loading_and_bounds_cache() -> Result<()> {
        let loads = Arc::new(AtomicUsize::new(0));
        let provider = LazyRecordProvider::with_loader_for_tests("unused.parquet", 2, {
            let loads = Arc::clone(&loads);
            Arc::new(move |_file, neuron_uuid| {
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(vec![DiscoverRecord {
                    obs_index: 0,
                    neuron_uuid: neuron_uuid.to_string(),
                    value: None,
                    activation: 0.0,
                    errors: vec![0.0],
                }])
            })
        });

        // No eager loads during initialisation
        assert_eq!(0, loads.load(Ordering::SeqCst));

        // First load hits the loader, subsequent load for same neuron is cached
        provider.get("a")?.expect("records should be present");
        assert_eq!(1, loads.load(Ordering::SeqCst));
        provider.get("a")?.expect("records should be cached");
        assert_eq!(1, loads.load(Ordering::SeqCst));

        // Loading a second neuron increments once and cache remains bounded
        provider.get("b")?.expect("records should be present");
        assert_eq!(2, loads.load(Ordering::SeqCst));
        assert!(provider.len() <= 2);
        Ok(())
    }

    #[test]
    fn lazy_provider_returns_loader_errors_with_context() {
        let provider = LazyRecordProvider::with_loader_for_tests("failing.parquet", 2, {
            Arc::new(|file, neuron_uuid| {
                Err(anyhow!(
                    "Simulated parquet read failure for {neuron_uuid} in {file}"
                ))
            })
        });

        let err = provider
            .get("hidden-1")
            .expect_err("loader error should surface");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Simulated parquet read failure"),
            "expected loader error, got: {msg}"
        );
        assert!(
            msg.contains("hidden-1"),
            "neuron context should be present: {msg}"
        );
        assert!(
            msg.contains("failing.parquet"),
            "file context should be present: {msg}"
        );
    }
}

// NOTE: Tests for focus module have been moved to tests/focus.rs
// following the testing philosophy documented in README.md
// (prefer tests/ over inline unit tests for tests that use public APIs).
