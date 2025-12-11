use crate::parquet_format::{read_all_records_grouped_by_neuron, read_records_from_parquet};
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};
use anyhow::{anyhow, Context, Result};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
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

#[derive(Debug)]
pub struct RankedNeuron {
    pub neuron_uuid: String,
    pub total_error: f32,
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
    pub reason: String,
}

#[derive(Debug)]
pub struct RankFocusStats {
    pub neurons: Vec<RankedNeuron>,
    /// Neurons with impact below costOfGrowth - candidates for removal
    pub removal_candidates: Vec<RemovalCandidate>,
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

#[allow(dead_code)]
fn average_absolute_error(parquet_file: &str, neuron_uuid: &str) -> Result<f32> {
    let records: Vec<DiscoverRecord> = read_records_from_parquet(parquet_file, neuron_uuid)
        .with_context(|| format!("Failed to read discovery records for neuron {neuron_uuid}"))?;

    Ok(average_absolute_error_from_records(&records))
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
    grouped_records: &HashMap<String, Vec<DiscoverRecord>>,
) -> SelectionStats {
    let mut stats = SelectionStats::new();
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

    for target_neuron in selection_neurons {
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
            continue;
        }

        match squash {
            "MINIMUM" => {
                compute_min_stats(&incoming_synapses, grouped_records, &mut stats);
            }
            "MAXIMUM" => {
                compute_max_stats(&incoming_synapses, grouped_records, &mut stats);
            }
            "IF" => {
                compute_if_stats(&incoming_synapses, grouped_records, &mut stats);
            }
            _ => {}
        }
    }

    stats
}

/// Compute selection statistics for a MINIMUM neuron.
/// Counts how often each synapse provides the minimum weighted activation.
fn compute_min_stats(
    synapses: &[&SynapseJson],
    grouped_records: &HashMap<String, Vec<DiscoverRecord>>,
    stats: &mut SelectionStats,
) {
    if synapses.is_empty() {
        return;
    }

    // Build a map of obs_index -> Vec<(synapse_key, weighted_activation)>
    let mut obs_contributions: ObservationContributions = HashMap::new();

    for synapse in synapses {
        if let Some(records) = grouped_records.get(&synapse.from_uuid) {
            for record in records {
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
}

/// Compute selection statistics for a MAXIMUM neuron.
/// Counts how often each synapse provides the maximum weighted activation.
fn compute_max_stats(
    synapses: &[&SynapseJson],
    grouped_records: &HashMap<String, Vec<DiscoverRecord>>,
    stats: &mut SelectionStats,
) {
    if synapses.is_empty() {
        return;
    }

    // Build a map of obs_index -> Vec<(synapse_key, weighted_activation)>
    let mut obs_contributions: ObservationContributions = HashMap::new();

    for synapse in synapses {
        if let Some(records) = grouped_records.get(&synapse.from_uuid) {
            for record in records {
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
    grouped_records: &HashMap<String, Vec<DiscoverRecord>>,
    stats: &mut SelectionStats,
) {
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
        return;
    }

    // Condition synapses are always active
    for synapse in &condition_synapses {
        let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
        stats.insert(key, 1.0);
    }

    // Compute condition sum for each observation to determine positive/negative branch usage
    let mut obs_condition_sums: HashMap<u32, f32> = HashMap::new();

    for synapse in &condition_synapses {
        if let Some(records) = grouped_records.get(&synapse.from_uuid) {
            for record in records {
                if record.activation.is_finite() {
                    let contribution = synapse.weight * record.activation;
                    *obs_condition_sums.entry(record.obs_index).or_insert(0.0) += contribution;
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
        return;
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
    squash_map: HashMap<String, String>,
    outputs: HashSet<String>,
    /// Selection statistics from activation records.
    /// When available, provides actual win probabilities for MIN/MAX/IF synapses
    /// instead of the conservative 1/N equal probability fallback.
    selection_stats: Option<SelectionStats>,
}

fn compute_impacts_internal(creature: &CreatureJson) -> HashMap<String, f32> {
    compute_impacts_internal_with_stats(creature, None)
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
    grouped_records: Option<&HashMap<String, Vec<DiscoverRecord>>>,
) -> HashMap<String, f32> {
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

    let outputs: HashSet<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone())
        .collect();

    // Compute selection statistics if activation records are available
    let selection_stats = grouped_records.map(|records| compute_selection_stats(creature, records));

    let ctx = ImpactContext {
        adjacency,
        inbound_count,
        squash_map,
        outputs,
        selection_stats,
    };

    let mut cache: HashMap<String, f32> = HashMap::new();

    for neuron in &creature.neurons {
        if !is_selectable_type(&neuron.neuron_type) {
            continue;
        }
        let mut visiting = HashSet::new();
        compute_impact_recursive(&neuron.uuid, &ctx, &mut cache, &mut visiting);
    }

    cache
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
    grouped_records: &HashMap<String, Vec<DiscoverRecord>>,
) -> HashMap<String, f32> {
    compute_impacts_internal_with_stats(creature, Some(grouped_records))
}

fn compute_impact_recursive(
    uuid: &str,
    ctx: &ImpactContext,
    cache: &mut HashMap<String, f32>,
    visiting: &mut HashSet<String>,
) -> f32 {
    if let Some(value) = cache.get(uuid) {
        return *value;
    }

    if !visiting.insert(uuid.to_string()) {
        // Cycle detected; treat as zero contribution
        return 0.0;
    }

    let impact = if ctx.outputs.contains(uuid) {
        1.0
    } else if let Some(edges) = ctx.adjacency.get(uuid) {
        // Sum across all outgoing edges (neuron may connect to multiple targets)
        let mut total_impact = 0.0;
        for (to_uuid, weight) in edges {
            let child_impact = compute_impact_recursive(to_uuid, ctx, cache, visiting);
            if child_impact <= 0.0 {
                continue;
            }

            // Get the target neuron's squash category
            let squash = ctx
                .squash_map
                .get(to_uuid)
                .map(|s| s.as_str())
                .unwrap_or("IDENTITY");
            let category = SquashCategory::from_squash(squash);

            let contribution = match category {
                SquashCategory::Linear => {
                    // ABSOLUTE impact: what is the actual contribution magnitude?
                    //
                    // v0.1.145: CRITICAL FIX - normalisation was causing massive underestimation.
                    //
                    // Previously we computed `|weight| / total_inbound × child_impact`, which
                    // gave "what fraction of inputs is this?" but this is WRONG for removal
                    // prediction. Production data showed impact underestimated by up to
                    // 145 BILLION times (calculated 1e-12, actual 20% error increase).
                    //
                    // The correct formula for removal impact is:
                    //   impact = |weight| × child_impact
                    //
                    // This gives the actual magnitude of change when the neuron is removed.
                    // Values are no longer bounded to [0,1] but that's fine - we only care
                    // about relative ordering for removal candidates.
                    //
                    // The old normalisation made sense for "blame assignment" but not for
                    // predicting what happens when a neuron is removed from the network.
                    weight.abs() * child_impact
                }
                SquashCategory::Threshold => {
                    // THRESHOLD impact (STEP/BIPOLAR): Any synapse could flip the output!
                    //
                    // For STEP/BIPOLAR neurons, even tiny weights can cause the full output
                    // swing if they push the neuron across the threshold (0).
                    //
                    // Conservative approach: Don't normalise by total_inbound. Instead,
                    // use the full child_impact as if this synapse alone determines output.
                    //
                    // This may overestimate impact, but it's better than underestimating
                    // (which causes incorrect removal candidates).
                    //
                    // See docs/IMPACT_CALCULATION.md for detailed explanation.
                    child_impact
                }
                SquashCategory::Selection => {
                    // SELECTION impact (MINIMUM/MAXIMUM/IF): Only one synapse "wins"!
                    //
                    // For MINIMUM/MAXIMUM neurons, only the min/max synapse contributes to
                    // the output. The others have zero contribution at any given time.
                    //
                    // When activation data is available (selection_stats), we use the actual
                    // win probability for this synapse. Otherwise, we fall back to the
                    // conservative 1/N equal probability approach.
                    //
                    // See docs/IMPACT_CALCULATION.md for detailed explanation.
                    let synapse_key = (uuid.to_string(), to_uuid.to_string());

                    if let Some(ref stats) = ctx.selection_stats {
                        // Use actual win probability from activation records
                        if let Some(&probability) = stats.get(&synapse_key) {
                            probability * child_impact
                        } else {
                            // Synapse not in stats - use equal probability fallback
                            let count = ctx.inbound_count.get(to_uuid).copied().unwrap_or(1).max(1);
                            child_impact / count as f32
                        }
                    } else {
                        // No activation data - use conservative equal probability
                        let count = ctx.inbound_count.get(to_uuid).copied().unwrap_or(1).max(1);
                        child_impact / count as f32
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
    cache.insert(uuid.to_string(), impact);
    impact
}

pub fn rank_focus_neurons(
    parquet_file: &str,
    creature: &CreatureJson,
    max_results: Option<usize>,
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
            max_output_error: 0.0,
            processed_neurons: 0,
            total_neurons: 0,
            duration_ms: start.elapsed().as_millis(),
        });
    }

    // Read all records once and group by neuron UUID for efficient access
    // This avoids reading the parquet file 460+ times (once per neuron)
    let grouped_records = read_all_records_grouped_by_neuron(parquet_file)
        .context("Failed to read discovery records from parquet file")?;

    // Verify that all selectable neurons have records (restore old error behavior)
    // This maintains data integrity by failing fast if records are missing
    for neuron in &selectable {
        if !grouped_records.contains_key(&neuron.uuid) {
            return Err(anyhow!(
                "Missing discovery records for selectable neuron: {}",
                neuron.uuid
            )
            .context("Failed to read discovery records for all selectable neurons"));
        }
    }

    let output_neurons: Vec<&NeuronJson> = creature
        .neurons
        .iter()
        .filter(|neuron| neuron.neuron_type == "output")
        .collect();

    let max_output_error = if output_neurons.is_empty() {
        0.0
    } else {
        output_neurons
            .iter()
            .map(|neuron| {
                let records = grouped_records
                    .get(&neuron.uuid)
                    .expect("Output neuron should have records (already verified above)");
                average_absolute_error_from_records(records)
            })
            .fold(0.0, f32::max)
    };

    // Use activation-based impact calculation for more accurate MIN/MAX/IF statistics
    let impact_map = compute_impacts_with_activations(creature, &grouped_records);

    // Now we can safely unwrap since we've verified all selectable neurons have records
    let mut neurons = selectable
        .iter()
        .map(|neuron| {
            let records = grouped_records
                .get(&neuron.uuid)
                .expect("Selectable neuron should have records (already verified above)");
            let avg_error = average_absolute_error_from_records(records);
            let structural_impact = *impact_map.get(&neuron.uuid).unwrap_or(&0.0);
            let mean_activation = mean_absolute_activation_from_records(records);

            // Activation-weighted impact reflects the ACTUAL contribution during inference.
            // A neuron with tiny structural impact but massive activations still contributes
            // significantly: actual_contribution ≈ weight × activation
            //
            // Only neurons with BOTH low structural impact AND low activation should be
            // removal candidates. If either is high, the neuron is contributing.
            let activation_weighted_impact = structural_impact * mean_activation;

            let total_error = if max_output_error > 0.0 {
                avg_error.min(max_output_error)
            } else {
                avg_error
            };
            RankedNeuron {
                neuron_uuid: neuron.uuid.clone(),
                total_error,
                impact: structural_impact,
                mean_activation,
                activation_weighted_impact,
            }
        })
        .collect::<Vec<_>>();

    // Sort by weighted score (error × impact) to prioritise neurons that:
    // 1. Have high error (potential for improvement)
    // 2. Have high impact (changes will affect output)
    // This ensures output neurons and neurons close to outputs are prioritised
    // over high-error hidden neurons with minimal impact on the creature's score.
    const IMPACT_EPSILON: f32 = 0.0001;
    neurons.sort_by(|a, b| {
        let a_weighted = a.total_error * (a.impact + IMPACT_EPSILON);
        let b_weighted = b.total_error * (b.impact + IMPACT_EPSILON);
        b_weighted
            .partial_cmp(&a_weighted)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.impact.partial_cmp(&a.impact).unwrap_or(Ordering::Equal))
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    // Identify removal candidates: neurons with activation_weighted_impact < costOfGrowth.
    //
    // activation_weighted_impact = structural_impact × mean_activation
    // where structural_impact = ABSOLUTE impact through the network (v0.1.145 fix)
    //
    // Neurons with impact below costOfGrowth are net negative - removing them
    // reduces complexity more than it affects error.
    //
    // We provide complexity savings info for each neuron based on NEAT-AI's formula:
    //   savings = growthCost × (1 + (N + M) / 10)
    // where N = incoming synapses, M = outgoing synapses
    //
    // v0.1.145: Changed from 1e-7 to 0.01 to match TypeScript default.
    // The old 1e-7 was calibrated for NORMALISED impacts which massively
    // underestimated actual impact (by up to 145 billion times in production).
    const COST_OF_GROWTH: f32 = 0.01;

    // Return ALL neurons with impact below costOfGrowth as removal candidates
    let mut removal_candidates: Vec<RemovalCandidate> = neurons
        .iter()
        .filter(|n| n.activation_weighted_impact < COST_OF_GROWTH)
        .map(|n| {
            let (incoming, outgoing) = count_synapses_for_neuron(&n.neuron_uuid, creature);
            let savings = calculate_removal_savings(incoming, outgoing, COST_OF_GROWTH);

            RemovalCandidate {
                neuron_uuid: n.neuron_uuid.clone(),
                total_error: n.total_error,
                impact: n.impact,
                mean_activation: n.mean_activation,
                activation_weighted_impact: n.activation_weighted_impact,
                incoming_synapses: incoming,
                outgoing_synapses: outgoing,
                removal_savings: savings,
                reason: format!(
                    "Impact {:.2e} < costOfGrowth ({:.0e}), {} synapses, saves {:.2e}",
                    n.activation_weighted_impact,
                    COST_OF_GROWTH,
                    incoming + outgoing,
                    savings
                ),
            }
        })
        .collect();

    // Sort by activation_weighted_impact ascending (lowest impact = best candidates)
    removal_candidates.sort_by(|a, b| {
        a.activation_weighted_impact
            .partial_cmp(&b.activation_weighted_impact)
            .unwrap_or(Ordering::Equal)
    });

    if let Some(limit) = max_results {
        if neurons.len() > limit {
            neurons.truncate(limit);
        }
    }

    Ok(RankFocusStats {
        neurons,
        removal_candidates,
        max_output_error,
        processed_neurons: total_neurons,
        total_neurons,
        duration_ms: start.elapsed().as_millis(),
    })
}

// NOTE: Tests for focus module have been moved to tests/focus.rs
// following the testing philosophy documented in README.md
// (prefer tests/ over inline unit tests for tests that use public APIs).
