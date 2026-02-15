//! Neuron ranking and selection.
//!
//! Core types and functions for ranking neurons by their discovery potential.
//! Includes record providers (eager and lazy), ranking metrics, removal
//! candidate identification, and constant neuron removal.

use super::gradient::{
    GradientFlowStats, build_squash_map, compute_gradient_flow_factor,
    compute_gradient_flow_for_neuron,
};
use super::impact::compute_impacts_with_activations;
use crate::analysis::utils::lock_or_bail;
use crate::analysis::{check_memory_for_parquet, verbose_enabled};
use crate::discovery_history::DiscoveryHistory;
use crate::parquet_format::{read_all_records_grouped_by_neuron, read_records_from_parquet};
use crate::types::DiscoverRecord;
use crate::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson,
};
use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Statistics for selection-based neurons (MINIMUM, MAXIMUM, IF).
/// Maps (from_uuid, to_uuid) -> win probability (0.0 to 1.0).
/// For MIN/MAX: probability this synapse provides the min/max value.
/// For IF: probability this synapse's branch is taken (condition always 1.0).
pub type SelectionStats = HashMap<(String, String), f32>;

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

pub(super) struct EagerRecordProvider {
    records: HashMap<String, Arc<Vec<DiscoverRecord>>>,
}

impl EagerRecordProvider {
    pub(super) fn new(records: HashMap<String, Vec<DiscoverRecord>>) -> Self {
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

pub(super) struct LazyRecordProvider {
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

    pub(super) fn new(parquet_file: &str) -> Self {
        Self {
            parquet_file: parquet_file.to_string(),
            cache: Mutex::new(LazyCache::new(Self::DEFAULT_CACHE_CAPACITY)),
            loader: Arc::new(read_records_from_parquet),
        }
    }

    #[cfg(test)]
    pub(super) fn with_loader_for_tests(
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
            let cache = lock_or_bail(&self.cache, "lazy record cache")?;
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

        let mut cache = lock_or_bail(&self.cache, "lazy record cache")?;
        cache.insert(neuron_uuid.to_string(), Arc::clone(&arc_records));
        Ok(Some(arc_records))
    }

    fn len(&self) -> usize {
        // len() is diagnostic-only; recover data from a poisoned mutex if needed
        match self.cache.lock() {
            Ok(cache) => cache.entries.len(),
            Err(poisoned) => poisoned.into_inner().entries.len(),
        }
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
    /// Issue #206: Gradient flow statistics for this neuron.
    ///
    /// These metrics help identify neurons with high learning potential:
    /// - `avg_gradient_magnitude`: How much error signal can flow through
    /// - `saturation_ratio`: % of samples in saturated activation region
    /// - `dead_ratio`: % of samples with zero gradient (ReLU dead zones)
    pub gradient_flow: GradientFlowStats,
    /// Issue #204: Activation frequency (proportion of samples where neuron fires).
    ///
    /// Calculated as: count_nonzero_activations / total_samples
    /// where "fires" means |activation| > small threshold (avoiding floating point issues).
    ///
    /// This helps identify neurons with extreme firing patterns:
    /// - activation_frequency < 0.1: Rarely fires, limited influence on most samples
    /// - activation_frequency > 0.9: Always fires, behaves like a constant (no discriminative power)
    /// - 0.1 <= activation_frequency <= 0.9: "Sweet spot" with good discriminative power
    pub activation_frequency: f32,
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
    ///
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

pub(super) fn is_selectable_type(neuron_type: &str) -> bool {
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

/// Pre-computed synapse counts for efficient O(1) lookup.
///
/// Issue #208: Pre-compute synapse counts to eliminate O(n×m) complexity.
///
/// Previously, `count_synapses_for_neuron` performed a linear O(m) scan through ALL synapses
/// twice (incoming + outgoing) for each neuron being ranked. With n neurons and m synapses,
/// this was O(n × m) complexity.
///
/// This struct pre-builds two HashMaps during initialisation in O(m) time, then provides
/// O(1) lookup for any neuron's synapse counts. Total complexity is O(n + m).
///
/// # Example performance improvement
///
/// For a creature with 500 neurons and 10,000 synapses:
/// - Previous: 500 × 10,000 × 2 = **10 million** iterations
/// - With SynapseCounts: 10,000 + 500 = **10,500** iterations
/// - **~1000x improvement**
#[derive(Debug)]
pub struct SynapseCounts {
    /// Map from neuron UUID to count of synapses pointing TO that neuron
    incoming: HashMap<String, usize>,
    /// Map from neuron UUID to count of synapses pointing FROM that neuron
    outgoing: HashMap<String, usize>,
}

impl SynapseCounts {
    /// Create a new SynapseCounts by scanning all synapses once.
    ///
    /// Time complexity: O(m) where m is the number of synapses.
    /// Space complexity: O(n) where n is the number of unique neurons with synapses.
    pub fn new(creature: &CreatureJson) -> Self {
        let mut incoming: HashMap<String, usize> = HashMap::new();
        let mut outgoing: HashMap<String, usize> = HashMap::new();

        for synapse in &creature.synapses {
            *incoming.entry(synapse.to_uuid.clone()).or_default() += 1;
            *outgoing.entry(synapse.from_uuid.clone()).or_default() += 1;
        }

        Self { incoming, outgoing }
    }

    /// Get the synapse counts for a neuron in O(1) time.
    ///
    /// # Arguments
    /// * `neuron_uuid` - The UUID of the neuron to look up
    ///
    /// # Returns
    /// A tuple of (incoming_count, outgoing_count). Returns (0, 0) if the neuron
    /// has no synapses or doesn't exist in the creature.
    pub fn get(&self, neuron_uuid: &str) -> (usize, usize) {
        (
            self.incoming.get(neuron_uuid).copied().unwrap_or(0),
            self.outgoing.get(neuron_uuid).copied().unwrap_or(0),
        )
    }
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

    if count == 0 { 0.0 } else { sum / count as f32 }
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

    if count == 0 { 0.0 } else { sum / count as f32 }
}

/// Issue #204: Compute activation frequency from discovery records.
///
/// Activation frequency = count_nonzero_activations / total_samples
/// where "fires" means |activation| > threshold to avoid floating-point issues.
///
/// # Arguments
/// * `records` - Discovery records for a single neuron
///
/// # Returns
/// A value in [0.0, 1.0] representing the proportion of samples where the neuron fires.
/// Returns 0.0 if there are no valid (finite) records.
///
/// # Rationale
/// - Neurons that rarely fire (frequency < 0.1) have limited influence on most samples
/// - Neurons that always fire (frequency > 0.9) behave like constants with no discriminative power
/// - Neurons with moderate frequency (0.1 to 0.9) are in the "sweet spot" for discovery
const ACTIVATION_FIRING_THRESHOLD: f32 = 1e-6;

fn activation_frequency_from_records(records: &[DiscoverRecord]) -> f32 {
    if records.is_empty() {
        return 0.0;
    }

    let mut firing_count: u32 = 0;
    let mut total_count: u32 = 0;

    for record in records {
        if record.activation.is_finite() {
            total_count += 1;
            // A neuron "fires" when its absolute activation exceeds the threshold
            if record.activation.abs() > ACTIVATION_FIRING_THRESHOLD {
                firing_count += 1;
            }
        }
    }

    if total_count == 0 {
        0.0
    } else {
        firing_count as f32 / total_count as f32
    }
}

/// Issue #204: Compute frequency factor for focus neuron ranking.
///
/// Neurons with extreme activation frequencies (rarely or always firing) are
/// less useful for discovery analysis:
/// - Rarely-firing neurons (< 10%) have limited influence on most samples
/// - Always-firing neurons (> 90%) behave like constants with no discriminative power
///
/// # Arguments
/// * `activation_frequency` - The proportion of samples where the neuron fires [0.0, 1.0]
///
/// # Returns
/// * 0.8 if activation_frequency < 0.1 (rarely fires) - 20% penalty
/// * 0.8 if activation_frequency > 0.9 (always fires) - 20% penalty
/// * 1.0 otherwise (moderate frequency) - no penalty
const FREQUENCY_LOW_THRESHOLD: f32 = 0.1;
const FREQUENCY_HIGH_THRESHOLD: f32 = 0.9;
const FREQUENCY_PENALTY_FACTOR: f32 = 0.8;

fn compute_frequency_factor(activation_frequency: f32) -> f32 {
    if (FREQUENCY_LOW_THRESHOLD..=FREQUENCY_HIGH_THRESHOLD).contains(&activation_frequency) {
        1.0 // No penalty for moderate frequency (10-90%)
    } else {
        FREQUENCY_PENALTY_FACTOR // 0.8x penalty for extreme frequencies
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

const DEFAULT_COST_OF_GROWTH: f32 = 1e-7;
const IMPACT_EPSILON: f32 = 0.0001;
const IMPACT_GAMMA: f32 = 0.8;

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
                tracing::warn!(
                    "Insufficient memory for full pre-load in focus ranking. \
                     Using lazy-loading mode (slower but memory-efficient)."
                );
                if verbose_enabled() {
                    tracing::debug!(error = %memory_error, "Memory check failed");
                }
                (Arc::new(LazyRecordProvider::new(parquet_file)), true)
            }
        };

    if is_lazy_mode && verbose_enabled() {
        tracing::debug!(
            cached_neurons = records_provider.len(),
            "Lazy record cache initialised"
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

    // Issue #208: Pre-compute synapse counts to eliminate O(n×m) complexity.
    // Previously, count_synapses_for_neuron scanned ALL synapses twice (incoming + outgoing)
    // for each neuron being ranked. With n neurons and m synapses, this was O(n × m).
    // By pre-computing counts into HashMaps, we reduce to O(m) for init + O(1) per lookup.
    let synapse_counts = SynapseCounts::new(creature);

    // Issue #206: Build squash map for gradient flow analysis
    let squash_map = build_squash_map(creature);

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

            // Issue #206: Compute gradient flow stats for this neuron
            let gradient_flow =
                compute_gradient_flow_for_neuron(&neuron.uuid, &squash_map, &records);

            // Issue #204: Compute activation frequency for focus neuron ranking
            let activation_frequency = activation_frequency_from_records(&records);

            Ok(RankedNeuron {
                neuron_uuid: neuron.uuid.clone(),
                total_error,
                raw_error,
                impact: structural_impact,
                mean_activation,
                activation_weighted_impact,
                gradient_flow,
                activation_frequency,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Sort by weighted score (error × impact × gradient_factor × frequency_factor) to prioritise neurons that:
    // 1. Have high error (potential for improvement)
    // 2. Have high impact (changes will affect output)
    // 3. Have good gradient flow (can actually learn - Issue #206)
    //
    // Dec 2025: We deliberately soften (but do not remove) the output bias by applying a
    // sub-linear exponent to impact. This increases exploration of hidden neurons without
    // letting low-impact neurons dominate purely due to noisy per-neuron errors.
    //
    // Jan 2026 (Issue #206): We further adjust ranking by gradient flow factor:
    // - Neurons stuck in saturation (high saturation_ratio) are de-prioritised
    // - Dead ReLU neurons (high dead_ratio) are de-prioritised
    // - Neurons with good gradient flow get higher priority
    //
    // Jan 2026 (Issue #204): We also adjust ranking by activation frequency factor:
    // - Rarely-firing neurons (< 10% activation rate) are de-prioritised (0.8x penalty)
    // - Always-firing neurons (> 90% activation rate) are de-prioritised (0.8x penalty)
    // - Moderate-frequency neurons (10-90%) get no penalty
    neurons.sort_by(|a, b| {
        // Base weighted score: error × impact^gamma
        let a_base = a.total_error * (a.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);
        let b_base = b.total_error * (b.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);

        // Issue #206: Apply gradient flow factor
        // Factor = (1 - saturation_ratio) × (1 - dead_ratio) × (0.5 + avg_gradient_magnitude/2)
        // This gives higher scores to neurons with:
        // - Low saturation (more room to learn)
        // - Low dead ratio (not stuck at zero)
        // - Higher gradient magnitude (better error propagation)
        let a_gradient_factor = compute_gradient_flow_factor(&a.gradient_flow);
        let b_gradient_factor = compute_gradient_flow_factor(&b.gradient_flow);

        // Issue #204: Apply activation frequency factor
        // Penalises neurons that rarely fire (< 10%) or always fire (> 90%)
        let a_frequency_factor = compute_frequency_factor(a.activation_frequency);
        let b_frequency_factor = compute_frequency_factor(b.activation_frequency);

        let a_weighted = a_base * a_gradient_factor * a_frequency_factor;
        let b_weighted = b_base * b_gradient_factor * b_frequency_factor;

        b_weighted
            .total_cmp(&a_weighted)
            .then_with(|| b.impact.total_cmp(&a.impact))
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
            // Issue #208: Use pre-computed synapse counts for O(1) lookup
            let (incoming, outgoing) = synapse_counts.get(&n.neuron_uuid);
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

    // Issue #414: High-error exploratory ablation DISABLED
    //
    // Previously, neurons with raw_error >= 10× max_output_error were returned as
    // "exploratory ablation candidates". This discovery type had a 0% success rate
    // (0 successes from 2 attempts) because the fundamental assumption was flawed:
    //
    // **High error ≠ harmful neuron**
    //
    // A neuron with high recorded error is often:
    // 1. Handling the most difficult samples (it's the only path for hard cases)
    // 2. Receiving bad inputs from upstream (the error is a symptom, not a cause)
    // 3. Fighting against incorrect biases elsewhere in the network
    //
    // Removing such neurons typically makes performance WORSE because:
    // - The difficult samples lose their only computation path
    // - The network loses the only neuron attempting to handle a specific pattern
    //
    // Error magnitude measures how WRONG the neuron's output is, not how HARMFUL
    // the neuron is to the network's overall score. This is why predicted
    // improvements (based on error magnitude) did not match actual outcomes.
    //
    // The legitimate removal candidate detection (based on activation_weighted_impact
    // < costOfGrowth) remains active and has a 17.6% success rate.

    // Issue #235: Sort by net improvement (removal_savings - activation_weighted_impact).
    // Higher net improvement = better candidate (removing it saves more than its contribution).
    removal_candidates.sort_by(|a, b| {
        // Calculate net improvement for each candidate
        let a_net = a.removal_savings - a.activation_weighted_impact;
        let b_net = b.removal_savings - b.activation_weighted_impact;

        // Sort by descending net improvement (best candidates first)
        b_net
            .total_cmp(&a_net)
            .then_with(|| {
                // For ties, prefer lower impact (safer removal)
                a.activation_weighted_impact
                    .total_cmp(&b.activation_weighted_impact)
            })
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    if let Some(limit) = max_results
        && neurons.len() > limit
    {
        neurons.truncate(limit);
    }

    // Issue #306: Detect constant-value neurons and create coordinated structural candidates
    // that remove the neuron and adjust downstream biases.
    let constant_neuron_removals = detect_constant_neuron_removals(
        &selectable,
        &records_provider,
        &synapse_counts,
        creature,
        cost_of_growth_threshold,
    );

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

/// Rank focus neurons with optional historical discovery success data.
///
/// Issue #227: By tracking which neurons have historically led to successful discoveries
/// (candidates that survived ablation testing), we can prioritise them in future runs,
/// improving the discovery hit rate.
///
/// This function behaves identically to `rank_focus_neurons` when no history is provided.
/// When history is provided, the ranking score is adjusted to favour neurons with
/// higher historical success rates using a Bayesian scoring approach:
///
/// ```text
/// combined_score = base_score × history_factor
/// ```
///
/// where:
/// - `base_score` = error × impact^gamma (same as rank_focus_neurons)
/// - `history_factor` = bayesian_score from history (0.0 to 1.0)
/// - For neurons not in history, `history_factor` = 0.5 (neutral prior)
///
/// # Arguments
///
/// * `parquet_file` - Path to the parquet file containing discovery records
/// * `creature` - The creature to rank neurons for
/// * `max_results` - Optional maximum number of neurons to return
/// * `cost_of_growth` - Optional cost of growth threshold (default: 1e-7)
/// * `history` - Optional discovery history for historical success data
///
/// # Returns
///
/// Returns `RankFocusStats` with neurons sorted by combined score (error × impact × history).
///
/// # Example
///
/// ```ignore
/// use neat_ai_discovery::discovery_history::DiscoveryHistory;
/// use neat_ai_discovery::focus::rank_focus_neurons_with_history;
///
/// // Create history from previous runs
/// let mut history = DiscoveryHistory::new();
/// history.record("hidden-1", true, Some(epoch));  // Success
/// history.record("hidden-2", false, None);         // Failure
///
/// // Rank neurons, prioritising those with higher historical success
/// let result = rank_focus_neurons_with_history(
///     "records.parquet",
///     &creature,
///     Some(10),      // max_results
///     None,          // cost_of_growth (use default)
///     Some(&history),
/// )?;
/// ```
pub fn rank_focus_neurons_with_history(
    parquet_file: &str,
    creature: &CreatureJson,
    max_results: Option<usize>,
    cost_of_growth: Option<f32>,
    history: Option<&DiscoveryHistory>,
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
                tracing::warn!(
                    "Insufficient memory for full pre-load in focus ranking. \
                     Using lazy-loading mode (slower but memory-efficient)."
                );
                if verbose_enabled() {
                    tracing::debug!(error = %memory_error, "Memory check failed");
                }
                (Arc::new(LazyRecordProvider::new(parquet_file)), true)
            }
        };

    if is_lazy_mode && verbose_enabled() {
        tracing::debug!(
            cached_neurons = records_provider.len(),
            "Lazy record cache initialised"
        );
    }

    // Verify that all selectable neurons have records
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

    // Pre-compute synapse counts for O(1) lookup
    let synapse_counts = SynapseCounts::new(creature);

    // Issue #206: Build squash map for gradient flow analysis
    let squash_map = build_squash_map(creature);

    // Build neurons with base metrics
    let mut neurons: Vec<RankedNeuron> = selectable
        .par_iter()
        .map(|neuron| -> Result<RankedNeuron> {
            let records = get_records_or_error(records_provider.as_ref(), &neuron.uuid)?;
            let raw_error = average_absolute_error_from_records(&records);
            let structural_impact = *impact_map.get(&neuron.uuid).unwrap_or(&0.0);
            let mean_activation = mean_absolute_activation_from_records(&records);
            let activation_weighted_impact = structural_impact * mean_activation;

            let total_error = if max_output_error > 0.0 {
                raw_error.min(max_output_error)
            } else {
                raw_error
            };

            // Issue #206: Compute gradient flow stats for this neuron
            let gradient_flow =
                compute_gradient_flow_for_neuron(&neuron.uuid, &squash_map, &records);

            // Issue #204: Compute activation frequency for focus neuron ranking
            let activation_frequency = activation_frequency_from_records(&records);

            Ok(RankedNeuron {
                neuron_uuid: neuron.uuid.clone(),
                total_error,
                raw_error,
                impact: structural_impact,
                mean_activation,
                activation_weighted_impact,
                gradient_flow,
                activation_frequency,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Sort by weighted score with optional history factor, gradient flow factor, and frequency factor
    // Issue #227: Incorporate historical success rate into ranking
    // Issue #206: Incorporate gradient flow analysis into ranking
    neurons.sort_by(|a, b| {
        let a_base = a.total_error * (a.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);
        let b_base = b.total_error * (b.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);

        // Issue #206: Apply gradient flow factor
        let a_gradient_factor = compute_gradient_flow_factor(&a.gradient_flow);
        let b_gradient_factor = compute_gradient_flow_factor(&b.gradient_flow);

        // Issue #204: Apply activation frequency factor
        let a_frequency_factor = compute_frequency_factor(a.activation_frequency);
        let b_frequency_factor = compute_frequency_factor(b.activation_frequency);

        let a_with_gradient = a_base * a_gradient_factor * a_frequency_factor;
        let b_with_gradient = b_base * b_gradient_factor * b_frequency_factor;

        // Apply history factor if available
        // History factor is in [0, 1], where 0.5 is neutral
        // We scale it so that:
        // - 0.5 (neutral) → multiplier of 1.0 (no change)
        // - 1.0 (perfect success) → multiplier of 1.5 (50% boost)
        // - 0.0 (complete failure) → multiplier of 0.5 (50% penalty)
        // Formula: multiplier = 0.5 + history_score
        let (a_weighted, b_weighted) = if let Some(h) = history {
            let a_history = h.bayesian_score_for(&a.neuron_uuid) as f32;
            let b_history = h.bayesian_score_for(&b.neuron_uuid) as f32;
            let a_multiplier = 0.5 + a_history;
            let b_multiplier = 0.5 + b_history;
            (
                a_with_gradient * a_multiplier,
                b_with_gradient * b_multiplier,
            )
        } else {
            (a_with_gradient, b_with_gradient)
        };

        b_weighted
            .total_cmp(&a_weighted)
            .then_with(|| b.impact.total_cmp(&a.impact))
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    // Identify removal candidates (same logic as rank_focus_neurons)
    let cost_of_growth_threshold = cost_of_growth.unwrap_or(DEFAULT_COST_OF_GROWTH);

    let mut removal_candidates: Vec<RemovalCandidate> = neurons
        .par_iter()
        .filter_map(|n| {
            let (incoming, outgoing) = synapse_counts.get(&n.neuron_uuid);
            let savings = calculate_removal_savings(incoming, outgoing, cost_of_growth_threshold);

            if savings <= n.activation_weighted_impact {
                return None;
            }

            let expected_error_reduction = n.activation_weighted_impact;
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

    // Issue #414: High-error exploratory ablation DISABLED (see rank_focus_neurons for rationale)

    // Sort removal candidates by net improvement
    removal_candidates.sort_by(|a, b| {
        let a_net = a.removal_savings - a.activation_weighted_impact;
        let b_net = b.removal_savings - b.activation_weighted_impact;

        b_net
            .total_cmp(&a_net)
            .then_with(|| {
                a.activation_weighted_impact
                    .total_cmp(&b.activation_weighted_impact)
            })
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    if let Some(limit) = max_results
        && neurons.len() > limit
    {
        neurons.truncate(limit);
    }

    // Constant neuron removals (same as rank_focus_neurons)
    let constant_neuron_removals = detect_constant_neuron_removals(
        &selectable,
        &records_provider,
        &synapse_counts,
        creature,
        cost_of_growth_threshold,
    );

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

/// Detect constant-value neurons and create coordinated structural candidates
/// that remove the neuron and adjust downstream biases.
///
/// A neuron with near-zero activation variance is "constant" - it always outputs roughly
/// the same value regardless of input. Removing it is equivalent to adjusting the biases
/// of downstream neurons by: bias_adjustment = synapse_weight × mean_activation
fn detect_constant_neuron_removals(
    selectable: &[&NeuronJson],
    records_provider: &Arc<dyn RecordProvider>,
    synapse_counts: &SynapseCounts,
    creature: &CreatureJson,
    cost_of_growth_threshold: f32,
) -> Vec<CoordinatedStructuralCandidateJson> {
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
    selectable
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
            // Issue #208: Use pre-computed synapse counts for O(1) lookup
            let (incoming_count, outgoing_count) = synapse_counts.get(&neuron.uuid);
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
        .collect()
}

// NOTE: Tests for focus module have been moved to tests/focus.rs
// following the testing philosophy documented in README.md
// (prefer tests/ over inline unit tests for tests that use public APIs).
