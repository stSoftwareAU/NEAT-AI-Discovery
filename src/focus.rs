use crate::parquet_format::{read_all_records_grouped_by_neuron, read_records_from_parquet};
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson};
use anyhow::{Context, Result};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Debug)]
pub struct RankedNeuron {
    pub neuron_uuid: String,
    pub total_error: f32,
    pub impact: f32,
}

#[derive(Debug)]
pub struct RankFocusStats {
    pub neurons: Vec<RankedNeuron>,
    pub max_output_error: f32,
    pub processed_neurons: usize,
    pub total_neurons: usize,
    pub duration_ms: u128,
}

fn is_selectable_type(neuron_type: &str) -> bool {
    neuron_type != "input" && neuron_type != "constant"
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

/// Build a map of total incoming absolute weights for each neuron.
/// This is used to normalize connection contributions when calculating impact.
fn build_inbound_weights(creature: &CreatureJson) -> HashMap<String, f32> {
    let mut inbound_weights: HashMap<String, f32> = HashMap::new();
    for synapse in &creature.synapses {
        *inbound_weights
            .entry(synapse.to_uuid.clone())
            .or_insert(0.0) += synapse.weight.abs();
    }
    inbound_weights
}

fn compute_impacts(creature: &CreatureJson) -> HashMap<String, f32> {
    let adjacency = build_adjacency(creature);
    let inbound_weights = build_inbound_weights(creature);
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
            &inbound_weights,
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
    inbound_weights: &HashMap<String, f32>,
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
        let mut max_value = 0.0;
        for (to_uuid, weight) in edges {
            let child_impact = compute_impact_recursive(
                to_uuid,
                adjacency,
                inbound_weights,
                outputs,
                cache,
                visiting,
            );
            if child_impact <= 0.0 {
                continue;
            }

            // Normalise contribution by the total incoming weight to the target neuron.
            // This ensures that impact is properly shared across multiple paths and that
            // connections with genuinely zero total inbound weight (all weights zero)
            // do not contribute spurious impact.
            let total_inbound = inbound_weights.get(to_uuid).copied().unwrap_or(0.0);
            let normalized_weight = if total_inbound > 0.0 {
                weight.abs() / total_inbound
            } else {
                // If there is no effective inbound signal (all weights zero), treat the
                // connection as having no impact to avoid division-by-zero artefacts.
                0.0
            };

            let contribution = normalized_weight * child_impact;
            if contribution > max_value {
                max_value = contribution;
            }
        }
        max_value
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
            .filter_map(|neuron| {
                grouped_records
                    .get(&neuron.uuid)
                    .map(|records| average_absolute_error_from_records(records))
            })
            .fold(0.0, f32::max)
    };

    let impact_map = compute_impacts(creature);

    let mut neurons = selectable
        .iter()
        .filter_map(|neuron| {
            grouped_records.get(&neuron.uuid).map(|records| {
                let avg_error = average_absolute_error_from_records(records);
                let impact = *impact_map.get(&neuron.uuid).unwrap_or(&0.0);
                let total_error = if max_output_error > 0.0 {
                    avg_error.min(max_output_error)
                } else {
                    avg_error
                };
                RankedNeuron {
                    neuron_uuid: neuron.uuid.clone(),
                    total_error,
                    impact,
                }
            })
        })
        .collect::<Vec<_>>();

    neurons.sort_by(|a, b| {
        b.total_error
            .partial_cmp(&a.total_error)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.impact.partial_cmp(&a.impact).unwrap_or(Ordering::Equal))
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    if let Some(limit) = max_results {
        if neurons.len() > limit {
            neurons.truncate(limit);
        }
    }

    Ok(RankFocusStats {
        neurons,
        max_output_error,
        processed_neurons: total_neurons,
        total_neurons,
        duration_ms: start.elapsed().as_millis(),
    })
}
