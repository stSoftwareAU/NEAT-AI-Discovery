//! Impact calculation for neurons.
//!
//! Computes the structural impact of each neuron on outputs using weighted
//! path products. Handles squash-aware impact for STEP/BIPOLAR/MINIMUM/MAXIMUM
//! neurons and activation-based selection statistics.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::ranking::{RecordProvider, SelectionStats};
use crate::{CreatureJson, NeuronJson, SynapseJson};
use anyhow::Result;
use rayon::prelude::*;

use dashmap::DashMap;
use std::collections::{HashMap, HashSet};

/// Categorise squash functions for impact calculation.
/// See `docs/IMPACT_CALCULATION.md` for detailed explanation.
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
    /// Categorise a squash function name.
    ///
    /// # Performance (Issue #211)
    /// Uses `eq_ignore_ascii_case` for zero-allocation case-insensitive comparison.
    fn from_squash(squash: &str) -> Self {
        if squash.eq_ignore_ascii_case("STEP") || squash.eq_ignore_ascii_case("BIPOLAR") {
            Self::Threshold
        } else if squash.eq_ignore_ascii_case("MINIMUM")
            || squash.eq_ignore_ascii_case("MAXIMUM")
            || squash.eq_ignore_ascii_case("IF")
        {
            Self::Selection
        } else {
            Self::Linear
        }
    }
}

/// Compute selection statistics for MINIMUM, MAXIMUM, and IF neurons using activation records.
///
/// For each selection-based neuron, this function analyses the recorded activations to determine
/// which synapse "wins" (provides the min/max value) for each observation. The result is a map
/// from (`from_uuid`, `to_uuid`) to the probability (0.0 to 1.0) that synapse wins.
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
/// Map from (`from_uuid`, `to_uuid`) to win probability for selection-based synapses
pub fn compute_selection_stats(
    creature: &CreatureJson,
    grouped_records: &dyn RecordProvider,
) -> Result<SelectionStats> {
    let squash_map = super::gradient::build_squash_map(creature);

    // Find all selection-based neurons (MINIMUM, MAXIMUM, IF)
    let selection_neurons: Vec<&NeuronJson> = creature
        .neurons
        .iter()
        .filter(|n| {
            let squash = squash_map
                .get(&n.uuid)
                .map_or("", std::string::String::as_str);
            matches!(squash, "MINIMUM" | "MAXIMUM" | "IF")
        })
        .collect();

    // Process each selection neuron in parallel and collect local stats
    let partial_stats: Vec<SelectionStats> = selection_neurons
        .par_iter()
        .map(|target_neuron| -> Result<Option<SelectionStats>> {
            let squash = squash_map
                .get(&target_neuron.uuid)
                .map_or("", std::string::String::as_str);

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
                    compute_min_stats(&incoming_synapses, grouped_records, &mut local_stats)?;
                }
                "MAXIMUM" => {
                    compute_max_stats(&incoming_synapses, grouped_records, &mut local_stats)?;
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

/// A synapse key paired with its weighted activation contribution.
/// Used internally for tracking which synapse wins in MIN/MAX calculations.
type SynapseContribution = ((String, String), f32);

/// Map from observation index to list of synapse contributions for that observation.
/// Used to determine which synapse wins (has min/max value) for each observation.
type ObservationContributions = HashMap<u32, Vec<SynapseContribution>>;

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

/// Public version of `compute_impacts` for use in add-neuron analysis.
/// Computes the structural impact of each neuron on outputs (path weight products).
/// Output neurons have impact = 1.0, hidden neurons have impact in [0, 1] based on
/// their weighted paths to outputs.
///
/// NOTE: This function is squash-aware. For neurons feeding into STEP/BIPOLAR/MINIMUM/MAXIMUM
/// targets, the impact calculation uses special handling to avoid underestimation.
/// See `docs/IMPACT_CALCULATION.md` for detailed explanation.
pub fn compute_impacts_public(creature: &CreatureJson) -> HashMap<String, f32> {
    compute_impacts_internal(creature)
}

/// Context for impact calculation, containing pre-computed lookup tables.
/// This struct groups related parameters to avoid `clippy::too_many_arguments`.
struct ImpactContext {
    adjacency: HashMap<String, Vec<(String, f32)>>,
    inbound_count: HashMap<String, usize>,
    /// Sum of |weight| for all synapses INTO each target neuron.
    /// Used for normalising Linear squash impact: |w| / `total_inbound_weight` × `child_impact`
    /// This ensures hidden neurons always have impact < 1.0 (Issue #130).
    total_inbound_weight: HashMap<String, f32>,
    squash_map: HashMap<String, String>,
    outputs: HashSet<String>,
    /// Selection statistics from activation records.
    /// When available, provides actual win probabilities for MIN/MAX/IF synapses
    /// instead of the conservative 1/N equal probability fallback.
    selection_stats: Option<SelectionStats>,
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

fn compute_impacts_internal(creature: &CreatureJson) -> HashMap<String, f32> {
    // Issue #525: Replace expect() with graceful fallback.
    // This call with None records should never fail (no I/O, no external data),
    // but if it does we return an empty map rather than panicking in FFI context.
    debug_assert!(
        compute_impacts_internal_with_stats(creature, None).is_ok(),
        "impact computation without records should not fail"
    );
    compute_impacts_internal_with_stats(creature, None).unwrap_or_default()
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
    let squash_map = super::gradient::build_squash_map(creature);

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

    // Issue #835: Use DashMap for lock-free concurrent reads during recursive
    // impact computation. This eliminates the Mutex contention that occurred when
    // many parallel threads repeatedly acquired the lock on every recursive call.
    let shared_cache: DashMap<String, f32> = DashMap::new();

    all_neurons.par_iter().for_each(|neuron| {
        // Check if already computed (another thread might have done it).
        if shared_cache.contains_key(&neuron.uuid) {
            return;
        }

        // Compute with a local visiting set (cycle detection is per-path)
        let mut visiting = HashSet::new();

        // We need to compute recursively, but with shared cache access
        let impact =
            compute_impact_with_shared_cache(&neuron.uuid, &ctx, &shared_cache, &mut visiting);

        // Store result
        shared_cache.insert(neuron.uuid.clone(), impact);
    });

    Ok(shared_cache.into_iter().collect())
}

/// Compute impact with a shared cache for parallel execution.
fn compute_impact_with_shared_cache(
    uuid: &str,
    ctx: &ImpactContext,
    shared_cache: &DashMap<String, f32>,
    visiting: &mut HashSet<String>,
) -> f32 {
    // Check cache first (lock-free read via DashMap sharding).
    if let Some(value) = shared_cache.get(uuid) {
        return *value;
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
                .map_or("IDENTITY", std::string::String::as_str);
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

    // Cache the result (lock-free write via DashMap sharding).
    shared_cache.insert(uuid.to_string(), impact);

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
