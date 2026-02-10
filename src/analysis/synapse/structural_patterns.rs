//! Structural pattern discovery for synapse analysis
//!
//! This module detects coordinated structural changes in the creature topology:
//! - Noisy vs trusted input folding (Issue #165)
//! - Collapse 1-in/1-out hidden neurons into direct synapses (Issue #425)
//!
//! Extracted from mod.rs as part of Issue #482.

use super::scoring::compute_synapse_improvement_and_count;
use crate::analysis::cache::RecordCache;
use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT;
use crate::analysis::diagnostics::TargetMap;
use crate::analysis::samples::{EPSILON, HelpfulSample};
use crate::analysis::weights::calculate_optimal_outgoing_weight;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, SynapseJson};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Noisy vs Trusted Input Folding (Issue #165)
// =============================================================================

/// Detect noisy vs trusted input pairs feeding the same target neuron.
///
/// When two input signals have the same mean and the same starting synapse weight,
/// but one is much noisier (higher activation variance), the intended coordinated fix is:
/// - Remove the noisy synapse
/// - Remove the trusted synapse
/// - Add the trusted synapse back with a higher weight (typically doubled)
///
/// Returns `Some(candidate)` if a beneficial noisy-vs-trusted pair is found.
pub(crate) fn detect_noisy_vs_trusted(
    target_uuid: &str,
    synapses_by_target: &[SynapseJson],
    cache: &RecordCache,
    target_map: &TargetMap,
    neuron_squash_map: &HashMap<String, String>,
) -> Option<CoordinatedStructuralCandidateJson> {
    fn activation_mean_and_variance(records: &[DiscoverRecord]) -> Option<(f32, f32)> {
        let mut n = 0.0f32;
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for r in records {
            if r.activation.is_finite() {
                n += 1.0;
                sum += r.activation;
                sum_sq += r.activation * r.activation;
            }
        }
        if n <= 0.0 {
            return None;
        }
        let mean = sum / n;
        let var = (sum_sq / n) - (mean * mean);
        Some((mean, var.max(0.0)))
    }

    fn activation_map(records: &[DiscoverRecord]) -> HashMap<u32, f32> {
        let mut map = HashMap::with_capacity(records.len());
        for r in records {
            if r.activation.is_finite() {
                map.insert(r.obs_index, r.activation);
            }
        }
        map
    }

    #[derive(Clone)]
    struct IncomingInput {
        from_uuid: String,
        weight: f32,
        mean: f32,
        var: f32,
    }

    let mut incoming_inputs: Vec<IncomingInput> = Vec::new();
    for syn in synapses_by_target.iter() {
        if !syn.from_uuid.starts_with("input-") {
            continue;
        }
        let Ok(from_records_arc) = cache.get(&syn.from_uuid) else {
            continue;
        };
        if from_records_arc.is_empty() {
            continue;
        }
        let Some((mean, var)) = activation_mean_and_variance(from_records_arc.as_ref()) else {
            continue;
        };
        incoming_inputs.push(IncomingInput {
            from_uuid: syn.from_uuid.clone(),
            weight: syn.weight,
            mean,
            var,
        });
    }

    if incoming_inputs.len() < 2 || target_map.map.is_empty() {
        return None;
    }

    // Strict matching for the simple-case test: same weights and same means.
    const WEIGHT_EPS: f32 = 1e-6;
    const MEAN_EPS: f32 = 1e-3;
    const MIN_VAR_RATIO: f32 = 10.0;

    let target_squash = neuron_squash_map.get(target_uuid).map(|s| s.as_str());

    let mut best: Option<(IncomingInput, IncomingInput, f32)> = None; // (noisy, trusted, gain)

    for i in 0..incoming_inputs.len() {
        for j in (i + 1)..incoming_inputs.len() {
            let a = incoming_inputs[i].clone();
            let b = incoming_inputs[j].clone();

            if (a.weight - b.weight).abs() > WEIGHT_EPS {
                continue;
            }
            if (a.mean - b.mean).abs() > MEAN_EPS {
                continue;
            }

            let (noisy, trusted) = if a.var >= b.var { (a, b) } else { (b, a) };
            let ratio = noisy.var / trusted.var.max(EPSILON);
            if ratio < MIN_VAR_RATIO {
                continue;
            }

            let Ok(noisy_records_arc) = cache.get(&noisy.from_uuid) else {
                continue;
            };
            let Ok(trusted_records_arc) = cache.get(&trusted.from_uuid) else {
                continue;
            };

            let noisy_map = activation_map(noisy_records_arc.as_ref());
            let trusted_map = activation_map(trusted_records_arc.as_ref());

            let mut delta_samples: Vec<HelpfulSample> = Vec::with_capacity(target_map.map.len());
            for (obs_index, target) in target_map.map.iter() {
                let Some(noisy_act) = noisy_map.get(obs_index) else {
                    continue;
                };
                let Some(trusted_act) = trusted_map.get(obs_index) else {
                    continue;
                };
                let Some(activation) =
                    crate::analysis::weights::coordinated_structural_activation_delta(
                        *trusted_act,
                        *noisy_act,
                        noisy.weight,
                        trusted.weight,
                    )
                else {
                    continue;
                };
                if !activation.is_finite() || !target.avg_error.is_finite() {
                    continue;
                }
                delta_samples.push(HelpfulSample {
                    activation,
                    avg_error: target.avg_error,
                    target_value: target.value,
                    target_activation: Some(target.activation),
                });
            }

            if delta_samples.is_empty() {
                continue;
            }

            let baseline_sq: f32 = delta_samples
                .iter()
                .map(|s| s.avg_error * s.avg_error)
                .sum();

            // Move the noisy weight onto the trusted input:
            // Δoutput = w_noisy * (trusted - noisy)
            let moved_weight = noisy.weight;
            let (improvement, _, _, _) = compute_synapse_improvement_and_count(
                delta_samples.as_slice(),
                moved_weight,
                baseline_sq,
                target_squash,
            );

            if improvement <= 0.0 {
                continue;
            }

            match &best {
                Some((_, _, best_gain)) if *best_gain >= improvement => {}
                _ => best = Some((noisy, trusted, improvement)),
            }
        }
    }

    best.map(|(noisy, trusted, gain)| {
        let new_weight = trusted.weight + noisy.weight;
        CoordinatedStructuralCandidateJson {
            operations: vec![
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: noisy.from_uuid,
                    to_neuron_uuid: target_uuid.to_string(),
                },
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: trusted.from_uuid.clone(),
                    to_neuron_uuid: target_uuid.to_string(),
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: trusted.from_uuid,
                    to_neuron_uuid: target_uuid.to_string(),
                    weight: new_weight,
                },
            ],
            expected_creature_score_gain: gain,
            comment: Some(
                "Coordinated: prune noisy input (high variance), strengthen trusted input"
                    .to_string(),
            ),
        }
    })
}

// =============================================================================
// Collapse Hidden Neuron (Issue #425)
// =============================================================================

/// Detect hidden neurons that form simple 1-in/1-out chains and propose collapsing
/// them into direct synapses.
///
/// When a hidden neuron `h` has exactly one incoming synapse (a → h) and one
/// outgoing synapse (h → b), we propose removing `h` and replacing the chain
/// with a direct synapse (a → b). This is emitted as a coordinated-structural
/// candidate group with 4 operations: RemoveSynapse(a→h), RemoveSynapse(h→b),
/// RemoveNeuron(h), AddSynapse(a→b).
pub(crate) fn detect_collapsible_hidden_neurons(
    input: &crate::AnalyzeSynapsesInput,
    cache: &RecordCache,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::new();

    // Build incoming/outgoing synapse lists per neuron.
    let mut incoming: HashMap<String, Vec<SynapseJson>> = HashMap::new();
    let mut outgoing: HashMap<String, Vec<SynapseJson>> = HashMap::new();
    for s in &input.creature.synapses {
        incoming
            .entry(s.to_uuid.clone())
            .or_default()
            .push(s.clone());
        outgoing
            .entry(s.from_uuid.clone())
            .or_default()
            .push(s.clone());
    }

    // Quick neuron-type lookup (only creature.neurons; inputs are not here).
    let neuron_type_map_local: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.neuron_type.clone()))
        .collect();

    // Precompute existing direct synapses so we don't propose duplicates.
    let mut existing_edges: HashSet<(String, String)> = HashSet::new();
    for s in &input.creature.synapses {
        existing_edges.insert((s.from_uuid.clone(), s.to_uuid.clone()));
    }

    for neuron in &input.creature.neurons {
        if neuron.neuron_type != "hidden" {
            continue;
        }
        let h = neuron.uuid.as_str();
        let Some(ins) = incoming.get(h) else { continue };
        let Some(outs) = outgoing.get(h) else {
            continue;
        };
        if ins.len() != 1 || outs.len() != 1 {
            continue;
        }

        let a_syn = &ins[0];
        let b_syn = &outs[0];
        let a = a_syn.from_uuid.as_str();
        let b = b_syn.to_uuid.as_str();

        // Skip degenerate / non-actionable cases.
        if a == b || a == h || b == h {
            continue;
        }
        if existing_edges.contains(&(a.to_string(), b.to_string())) {
            // A direct synapse already exists; collapsing would need additional ops (future work).
            continue;
        }

        // Ensure the target exists (either input-* or a neuron) so the op is not stale.
        let target_is_known = b.starts_with("input-") || neuron_type_map_local.contains_key(b);
        if !target_is_known {
            continue;
        }

        // Build samples: correlate a's activation to b's adjusted error after removing h→b.
        let Ok(a_records) = cache.get(a) else {
            continue;
        };
        let Ok(h_records) = cache.get(h) else {
            continue;
        };
        let Ok(b_records) = cache.get(b) else {
            continue;
        };
        if a_records.is_empty() || h_records.is_empty() || b_records.is_empty() {
            continue;
        }

        let target_map_b = TargetMap::from_records(b_records.as_ref());
        if target_map_b.map.is_empty() {
            continue;
        }
        let build_act_map = |records: &[DiscoverRecord]| -> HashMap<u32, f32> {
            let mut map: HashMap<u32, f32> = HashMap::with_capacity(records.len());
            for r in records {
                if r.activation.is_finite() {
                    map.insert(r.obs_index, r.activation);
                }
            }
            map
        };
        let a_map = build_act_map(a_records.as_ref());
        let h_map = build_act_map(h_records.as_ref());

        let mut samples: Vec<HelpfulSample> = Vec::with_capacity(target_map_b.map.len());
        for (obs_index, target) in target_map_b.map.iter() {
            let Some(a_act) = a_map.get(obs_index) else {
                continue;
            };
            let Some(h_act) = h_map.get(obs_index) else {
                continue;
            };
            if !a_act.is_finite() || !h_act.is_finite() || !target.avg_error.is_finite() {
                continue;
            }
            let adjusted_error = target.avg_error + b_syn.weight * (*h_act);
            if !adjusted_error.is_finite() {
                continue;
            }
            samples.push(HelpfulSample {
                activation: *a_act,
                avg_error: adjusted_error,
                target_value: None,
                target_activation: None,
            });
        }

        if samples.len() < MIN_NEURON_SAMPLE_COUNT {
            continue;
        }

        let mut sum_act_sq = 0.0f32;
        let mut sum_err_act = 0.0f32;
        let mut baseline_sq = 0.0f32;
        for s in &samples {
            sum_act_sq += s.activation * s.activation;
            sum_err_act += s.activation * s.avg_error;
            baseline_sq += s.avg_error * s.avg_error;
        }
        if baseline_sq <= EPSILON {
            continue;
        }

        let Some(weight) = calculate_optimal_outgoing_weight(sum_err_act, sum_act_sq, 1.0) else {
            continue;
        };

        let (improvement, _improved, _worsened, _total) =
            compute_synapse_improvement_and_count(&samples, weight, baseline_sq, None);
        if improvement <= 0.0 {
            continue;
        }

        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: a.to_string(),
                    to_neuron_uuid: h.to_string(),
                },
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: h.to_string(),
                    to_neuron_uuid: b.to_string(),
                },
                CoordinatedStructuralOpJson::RemoveNeuron {
                    neuron_uuid: h.to_string(),
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: a.to_string(),
                    to_neuron_uuid: b.to_string(),
                    weight,
                },
            ],
            expected_creature_score_gain: improvement,
            comment: Some(
                "Coordinated collapse: remove 1-in/1-out hidden neuron and add bypass synapse"
                    .to_string(),
            ),
        });
    }

    results
}
