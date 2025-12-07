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
/// This captures the actual magnitude of signals flowing through a neuron.
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
        // IMPORTANT: Use SUM (not max) to accumulate impact across all outgoing edges.
        // A neuron connecting to multiple outputs affects ALL of them, so removing it
        // has a cumulative effect. Using max previously underestimated impact for neurons
        // with multiple outgoing synapses - e.g. a neuron connecting to 2 outputs directly
        // would show impact=1.0 (max) instead of impact=2.0 (sum), causing incorrect
        // "low-impact" classification and failed removal predictions.
        let mut total_impact = 0.0;
        for (to_uuid, weight) in edges {
            let child_impact =
                compute_impact_recursive(to_uuid, adjacency, outputs, cache, visiting);
            if child_impact <= 0.0 {
                continue;
            }

            // Use ABSOLUTE weight for contribution calculation.
            //
            // BUG FIX (v0.1.126): Previously normalised by total_inbound, which computed
            // "fraction of downstream's input from this neuron" rather than "absolute
            // contribution to output". This caused massive underestimation:
            //
            // Example: neuron → target (weight 0.001), other → target (weight 100)
            // - Old (normalised): 0.001 / 100.001 × 1.0 ≈ 1e-5
            // - New (absolute): 0.001 × 1.0 = 0.001
            //
            // The normalised formula was 100x too pessimistic, leading to neurons being
            // incorrectly flagged as "low-impact" when they actually contributed
            // measurably to error. Production data showed ~75% of "low-impact" removals
            // actually increased error.
            //
            // The correct formula for contribution to output is:
            //   contribution = activation × weight × downstream_impact
            // NOT:
            //   contribution = activation × (weight / total_inbound) × downstream_impact
            let contribution = weight.abs() * child_impact;
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

    // Identify removal candidates: neurons where removing them likely improves score.
    //
    // Based on NEAT-AI's Score.ts formula:
    //   score = error + complexityPenalty
    //   complexityPenalty = hiddenNeuronCount × growthCost +
    //                       synapseCount × growthCost / 10 +
    //                       penalty × growthCost / 100
    //
    // Removing a neuron with N incoming and M outgoing synapses saves:
    //   savings = growthCost × (1 + (N + M) / 10)
    //
    // UNIT CONVERSION: activation_weighted_impact is in OUTPUT units (contribution to
    // output), while savings is in SCORE units (error + complexity). For MSE error,
    // a contribution `c` to output can increase error by at most `c²`. So:
    //
    //   c² < savings  →  c < sqrt(savings)
    //
    // A neuron is a removal candidate when:
    //   activation_weighted_impact < sqrt(savings)
    //
    // For savings ≈ 1.5e-7 (typical), threshold = sqrt(1.5e-7) ≈ 4e-4 (0.04%)
    // This catches neurons contributing less than 0.04% to output.
    const COST_OF_GROWTH: f32 = 1e-7;

    // Track statistics for verbose logging
    let mut min_impact: f32 = f32::MAX;
    let mut max_impact: f32 = f32::MIN;
    let mut min_impact_uuid = String::new();

    let mut removal_candidates: Vec<RemovalCandidate> = neurons
        .iter()
        .filter_map(|n| {
            let (incoming, outgoing) = count_synapses_for_neuron(&n.neuron_uuid, creature);
            let savings = calculate_removal_savings(incoming, outgoing, COST_OF_GROWTH);

            // Track min/max for verbose logging
            if n.activation_weighted_impact < min_impact {
                min_impact = n.activation_weighted_impact;
                min_impact_uuid = n.neuron_uuid.clone();
            }
            if n.activation_weighted_impact > max_impact {
                max_impact = n.activation_weighted_impact;
            }

            // Use sqrt(savings) as threshold to account for MSE error relationship
            let threshold = savings.sqrt();

            if n.activation_weighted_impact < threshold {
                Some(RemovalCandidate {
                    neuron_uuid: n.neuron_uuid.clone(),
                    total_error: n.total_error,
                    impact: n.impact,
                    mean_activation: n.mean_activation,
                    activation_weighted_impact: n.activation_weighted_impact,
                    incoming_synapses: incoming,
                    outgoing_synapses: outgoing,
                    removal_savings: savings,
                    reason: format!(
                        "Activation-weighted impact ({:.2e}) < threshold ({:.2e}) for {} synapses - removal likely improves score",
                        n.activation_weighted_impact, threshold, incoming + outgoing
                    ),
                })
            } else {
                None
            }
        })
        .collect();

    // Verbose logging to help debug removal candidate detection
    if std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok() && !neurons.is_empty() {
        let typical_savings = COST_OF_GROWTH * 1.5; // ~15 synapses
        let typical_threshold = typical_savings.sqrt();
        eprintln!(
            "[NEAT-AI-Discovery][verbose] Removal candidate check: {} neurons, impact range [{:.2e}, {:.2e}], \
             threshold sqrt({:.2e})={:.2e}, found {} candidates. Lowest impact: {} ({:.2e})",
            neurons.len(),
            min_impact,
            max_impact,
            typical_savings,
            typical_threshold,
            removal_candidates.len(),
            min_impact_uuid,
            min_impact
        );
    }

    // Sort by (impact - savings) ascending: most beneficial removals first
    // Lower values = bigger gap between savings and impact = safer to remove
    removal_candidates.sort_by(|a, b| {
        let a_benefit = a.removal_savings - a.activation_weighted_impact;
        let b_benefit = b.removal_savings - b.activation_weighted_impact;
        b_benefit.partial_cmp(&a_benefit).unwrap_or(Ordering::Equal)
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
