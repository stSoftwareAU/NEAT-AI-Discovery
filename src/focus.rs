use crate::parquet_format::{read_all_records_grouped_by_neuron, read_records_from_parquet};
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson};
use anyhow::{anyhow, Context, Result};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

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

// NOTE: build_inbound_weights was removed in v0.1.126 as part of the impact
// calculation fix. The normalisation it supported was causing massive
// underestimation of neuron impact (see regression_v0_1_126.rs tests).

fn compute_impacts(creature: &CreatureJson) -> HashMap<String, f32> {
    compute_impacts_internal(creature)
}

/// Public version of compute_impacts for use in add-neuron analysis.
/// Computes the structural impact of each neuron on outputs (path weight products).
/// Output neurons have impact = 1.0, hidden neurons have impact in [0, 1] based on
/// their weighted paths to outputs.
pub fn compute_impacts_public(creature: &CreatureJson) -> HashMap<String, f32> {
    compute_impacts_internal(creature)
}

fn compute_impacts_internal(creature: &CreatureJson) -> HashMap<String, f32> {
    let adjacency = build_adjacency(creature);

    // Build total inbound |weight| for each neuron (for normalisation)
    let total_inbound: HashMap<String, f32> = {
        let mut map: HashMap<String, f32> = HashMap::new();
        for synapse in &creature.synapses {
            *map.entry(synapse.to_uuid.clone()).or_insert(0.0) += synapse.weight.abs();
        }
        map
    };

    let output_neurons: HashSet<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone())
        .collect();
    let mut cache: HashMap<String, f32> = HashMap::new();

    for neuron in &creature.neurons {
        if !is_selectable_type(&neuron.neuron_type) {
            continue;
        }
        let mut visiting = HashSet::new();
        compute_impact_recursive(
            &neuron.uuid,
            &adjacency,
            &total_inbound,
            &output_neurons,
            &mut cache,
            &mut visiting,
        );
    }

    cache
}

fn compute_impact_recursive(
    uuid: &str,
    adjacency: &HashMap<String, Vec<(String, f32)>>,
    total_inbound: &HashMap<String, f32>,
    outputs: &HashSet<String>,
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

    let impact = if outputs.contains(uuid) {
        1.0
    } else if let Some(edges) = adjacency.get(uuid) {
        // Sum across all outgoing edges (neuron may connect to multiple targets)
        let mut total_impact = 0.0;
        for (to_uuid, weight) in edges {
            let child_impact = compute_impact_recursive(
                to_uuid,
                adjacency,
                total_inbound,
                outputs,
                cache,
                visiting,
            );
            if child_impact <= 0.0 {
                continue;
            }

            // NORMALISED impact: what fraction of the target's input does this neuron provide?
            //
            // If target has 100 incoming synapses with total |weight| = 343,
            // and this synapse has |weight| = 3, then this neuron contributes:
            //   3 / 343 ≈ 0.9% of the target's input
            //
            // This is recursive - the fraction propagates through the network.
            // A neuron 2 hops from output, going through a target with 100 inputs,
            // has its impact diluted by that factor.
            let target_total = total_inbound
                .get(to_uuid)
                .copied()
                .unwrap_or(1.0)
                .max(1e-10);
            let fraction = weight.abs() / target_total;
            let contribution = fraction * child_impact;
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

    let impact_map = compute_impacts(creature);

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

    // Identify removal candidates: ALL neurons sorted by activation_weighted_impact.
    //
    // activation_weighted_impact = structural_impact × mean_activation
    // where structural_impact = product of |weights| on path(s) to output(s)
    //
    // The lowest impact neurons are the best candidates for removal.
    // NO THRESHOLD FILTERING - let NEAT-AI decide based on cost/benefit analysis.
    //
    // We provide complexity savings info for each neuron based on NEAT-AI's formula:
    //   savings = growthCost × (1 + (N + M) / 10)
    // where N = incoming synapses, M = outgoing synapses
    const COST_OF_GROWTH: f32 = 1e-7;

    // Return ALL neurons as potential removal candidates, sorted by impact
    let mut removal_candidates: Vec<RemovalCandidate> = neurons
        .iter()
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
                    "Impact {:.2e} (structural {:.2e} × activation {:.2e}), {} synapses, saves {:.2e}",
                    n.activation_weighted_impact, n.impact, n.mean_activation,
                    incoming + outgoing, savings
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

    // Return only top 10 candidates (lowest impact = highest chance of success)
    removal_candidates.truncate(10);

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
