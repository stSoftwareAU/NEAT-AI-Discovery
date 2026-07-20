//! Compound parameter degradation detection (Issue #929).
//!
//! Detects coordinated degradations where multiple parameters (bias + weight)
//! need simultaneous correction. Neither fix alone fully explains the
//! performance loss — both must be restored together.
//!
//! ## Detection Criteria
//!
//! A compound degradation is detected when:
//! 1. A hidden neuron has consistent, non-zero mean error suggesting bias drift.
//! 2. A synapse has error correlated with its source activation, suggesting
//!    the weight is wrong.
//! 3. Both corrections are on the same forward path to an output neuron.
//! 4. The combined correction is predicted to improve the creature's score.
//!
//! ## Recommended Actions
//!
//! - `setBias` — correct the neuron's operating point.
//! - `setWeight` — correct the synapse's contribution.
//! - Both applied atomically as a coordinated-structural candidate.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use super::helpers::build_record_map;
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;
use crate::analysis::quantised_error::is_quantised_zero_one;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};
use std::collections::{HashMap, HashSet};

/// Minimum mean absolute error to consider a neuron for bias correction.
const MIN_BIAS_ERROR: f32 = 0.02;

/// Minimum absolute bias delta to emit a correction.
const MIN_BIAS_DELTA: f32 = 0.05;

/// Minimum absolute weight delta to consider a synapse degraded.
const MIN_WEIGHT_DELTA: f32 = 0.05;

/// A detected bias correction for a hidden neuron.
#[derive(Debug, Clone)]
struct BiasCorrection {
    neuron_uuid: String,
    recommended_bias: f32,
    estimated_improvement: f32,
}

/// A detected weight correction for a synapse.
#[derive(Debug, Clone)]
struct WeightCorrection {
    from_neuron_uuid: String,
    to_neuron_uuid: String,
    recommended_weight: f32,
    estimated_improvement: f32,
}

/// Approximate the squash function derivative at a given pre-activation value.
///
/// Used to convert error signals back to pre-activation deltas for bias
/// correction estimation.
fn squash_derivative(squash: &str, pre_activation: f32) -> f32 {
    match squash {
        "TANH" | "HARD_TANH" => {
            let t = pre_activation.tanh();
            1.0 - t * t
        }
        "ReLU" | "RELU" => {
            if pre_activation > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        "LOGISTIC" => {
            let s = 1.0 / (1.0 + (-pre_activation).exp());
            s * (1.0 - s)
        }
        "IDENTITY" => 1.0,
        "GELU" => {
            // Approximate GELU derivative
            let cdf = 0.5
                * (1.0 + (0.7978846 * (pre_activation + 0.044715 * pre_activation.powi(3))).tanh());
            let pdf = (-0.5 * pre_activation * pre_activation).exp()
                / (2.0 * std::f32::consts::PI).sqrt();
            cdf + pre_activation * pdf
        }
        _ => 1.0, // Default: assume identity-like
    }
}

/// Detect compound bias+weight degradations across the network.
///
/// Analyses hidden neurons for consistent error (bias drift) and synapses
/// for activation-error correlation (weight degradation). When both a bias
/// and weight correction lie on the same forward path and improve the
/// network, emits a coordinated candidate with `setBias` + `setWeight`.
///
/// # Arguments
/// * `creature` - The creature topology.
/// * `neuron_records` - Slice of `(uuid, records)` pairs with observation data.
///
/// # Returns
/// Vector of compound degradation candidates, sorted by estimated improvement.
pub fn detect_compound_bias_weight_degradations(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<CompoundDegradationCandidate> {
    let records_map = build_record_map(neuron_records);

    // Build neuron lookup maps
    let neuron_squash: HashMap<&str, &str> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.squash.as_str()))
        .collect();
    let neuron_bias: HashMap<&str, f32> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.bias))
        .collect();

    // Build connectivity: which neurons are downstream of which
    let mut downstream_of: HashMap<&str, HashSet<&str>> = HashMap::new();
    for syn in &creature.synapses {
        downstream_of
            .entry(syn.from_uuid.as_str())
            .or_default()
            .insert(syn.to_uuid.as_str());
    }

    // Phase 1: Detect bias corrections for hidden neurons
    let bias_corrections =
        detect_bias_corrections(creature, &records_map, &neuron_squash, &neuron_bias);

    // Phase 2: Detect weight corrections for synapses
    let weight_corrections = detect_weight_corrections(creature, &records_map);

    // Phase 3: Combine related corrections
    combine_corrections(
        &bias_corrections,
        &weight_corrections,
        &downstream_of,
        creature,
    )
}

/// Detect hidden neurons with consistent error suggesting bias drift.
fn detect_bias_corrections(
    creature: &CreatureJson,
    records_map: &HashMap<&str, &[DiscoverRecord]>,
    neuron_squash: &HashMap<&str, &str>,
    neuron_bias: &HashMap<&str, f32>,
) -> Vec<BiasCorrection> {
    let mut corrections = Vec::new();

    for neuron in &creature.neurons {
        if neuron.neuron_type != "hidden" {
            continue;
        }

        let Some(records) = records_map.get(neuron.uuid.as_str()) else {
            continue;
        };
        if records.len() < MIN_SAMPLES {
            continue;
        }

        // Compute mean error (signed — a consistent offset indicates bias drift)
        let n = records.len() as f32;
        let mean_error: f32 = records
            .iter()
            .map(|r| r.errors.first().copied().unwrap_or(0.0))
            .sum::<f32>()
            / n;

        if mean_error.abs() < MIN_BIAS_ERROR {
            continue;
        }

        // Compute mean pre-activation for derivative estimation
        let mean_pre_act: f32 = records.iter().filter_map(|r| r.value).sum::<f32>()
            / records.iter().filter(|r| r.value.is_some()).count().max(1) as f32;

        let squash = neuron_squash
            .get(neuron.uuid.as_str())
            .copied()
            .unwrap_or("IDENTITY");
        let deriv = squash_derivative(squash, mean_pre_act);

        // Skip if derivative is near zero (saturated region)
        if deriv.abs() < 1e-4 {
            continue;
        }

        // Bias correction: error ≈ f'(pre_act) × Δbias → Δbias ≈ error / f'(pre_act)
        let bias_delta = mean_error / deriv;
        let old_bias = neuron_bias
            .get(neuron.uuid.as_str())
            .copied()
            .unwrap_or(0.0);
        let new_bias = old_bias + bias_delta;

        if bias_delta.abs() < MIN_BIAS_DELTA || !new_bias.is_finite() {
            continue;
        }

        // Estimated improvement: proportional to error corrected
        let estimated_improvement = mean_error.abs() * 0.3;

        corrections.push(BiasCorrection {
            neuron_uuid: neuron.uuid.clone(),
            recommended_bias: new_bias,
            estimated_improvement,
        });
    }

    corrections
}

/// Detect synapses with error correlated to source activation (weight degradation).
fn detect_weight_corrections(
    creature: &CreatureJson,
    records_map: &HashMap<&str, &[DiscoverRecord]>,
) -> Vec<WeightCorrection> {
    let mut corrections = Vec::new();

    for syn in &creature.synapses {
        let Some(source_records) = records_map.get(syn.from_uuid.as_str()) else {
            continue;
        };
        let Some(target_records) = records_map.get(syn.to_uuid.as_str()) else {
            continue;
        };

        if source_records.len() < MIN_SAMPLES || target_records.len() < MIN_SAMPLES {
            continue;
        }

        // Build per-observation maps
        let source_map: HashMap<u32, f32> = source_records
            .iter()
            .filter(|r| r.activation.is_finite())
            .map(|r| (r.obs_index, r.activation))
            .collect();

        let target_error_map: HashMap<u32, f32> = target_records
            .iter()
            .filter(|r| r.errors.first().copied().unwrap_or(0.0).is_finite())
            .map(|r| (r.obs_index, r.errors.first().copied().unwrap_or(0.0)))
            .collect();

        // Issue #1249: under `CATEGORICAL_ERROR` the target error is a
        // quantised `{0, 1}` misclassification flag, so the
        // `baseline_error_sq − corrected_error_sq` SSE-improvement
        // estimate below collapses to the misclassification count
        // rather than a loss reduction. Skip the weight correction —
        // the emitted `setWeight` magnitude would not correspond to
        // any real improvement.
        let target_error_values: Vec<f32> = target_error_map.values().copied().collect();
        if is_quantised_zero_one(&target_error_values) {
            continue;
        }

        // Compute correlation between source activation and target error
        // Optimal weight delta via least squares: Δw = Σ(act × err) / Σ(act²)
        let mut sum_act_sq = 0.0f32;
        let mut sum_act_err = 0.0f32;
        let mut count = 0u32;

        for (obs, src_act) in &source_map {
            if let Some(tgt_err) = target_error_map.get(obs) {
                sum_act_sq += src_act * src_act;
                sum_act_err += src_act * tgt_err;
                count += 1;
            }
        }

        if (count as usize) < MIN_SAMPLES || sum_act_sq < 1e-8 {
            continue;
        }

        let weight_delta = sum_act_err / sum_act_sq;
        let new_weight = syn.weight + weight_delta;

        if weight_delta.abs() < MIN_WEIGHT_DELTA || !new_weight.is_finite() {
            continue;
        }

        // Estimated improvement from weight correction
        let baseline_error_sq: f32 = target_error_map.values().map(|e| e * e).sum::<f32>();
        let corrected_error_sq: f32 = source_map
            .iter()
            .filter_map(|(obs, act)| {
                target_error_map.get(obs).map(|err| {
                    let corrected = err - weight_delta * act;
                    corrected * corrected
                })
            })
            .sum();

        let improvement = (baseline_error_sq - corrected_error_sq) / count as f32;
        if improvement <= 0.0 {
            continue;
        }

        corrections.push(WeightCorrection {
            from_neuron_uuid: syn.from_uuid.clone(),
            to_neuron_uuid: syn.to_uuid.clone(),
            recommended_weight: new_weight,
            estimated_improvement: improvement,
        });
    }

    corrections
}

/// Combine bias and weight corrections on the same forward path into
/// coordinated candidates.
fn combine_corrections(
    bias_corrections: &[BiasCorrection],
    weight_corrections: &[WeightCorrection],
    downstream_of: &HashMap<&str, HashSet<&str>>,
    creature: &CreatureJson,
) -> Vec<CompoundDegradationCandidate> {
    let mut candidates = Vec::new();

    // Build a set of output neuron UUIDs for path validation
    let output_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    for bias_corr in bias_corrections {
        for weight_corr in weight_corrections {
            // Skip if bias and weight target the same neuron (not a compound case)
            if bias_corr.neuron_uuid == weight_corr.to_neuron_uuid
                && bias_corr.neuron_uuid == weight_corr.from_neuron_uuid
            {
                continue;
            }

            // Check if both corrections are on connected forward paths.
            // Either: bias neuron feeds into weight target, or weight target
            // feeds downstream from bias neuron, or both feed to same output.
            let on_same_path = are_on_connected_path(
                &bias_corr.neuron_uuid,
                &weight_corr.from_neuron_uuid,
                &weight_corr.to_neuron_uuid,
                downstream_of,
                &output_uuids,
            );

            if !on_same_path {
                continue;
            }

            // Combined improvement: conservative estimate
            let combined_improvement =
                bias_corr.estimated_improvement + weight_corr.estimated_improvement * 0.7;

            if combined_improvement <= 0.0 {
                continue;
            }

            candidates.push(CompoundDegradationCandidate {
                bias_neuron_uuid: bias_corr.neuron_uuid.clone(),
                recommended_bias: bias_corr.recommended_bias,
                weight_from_uuid: weight_corr.from_neuron_uuid.clone(),
                weight_to_uuid: weight_corr.to_neuron_uuid.clone(),
                recommended_weight: weight_corr.recommended_weight,
                estimated_improvement: combined_improvement,
            });
        }
    }

    // Sort by estimated improvement descending
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));
    candidates
}

/// Check whether two corrections are on a connected forward path.
///
/// Returns true if:
/// - The bias neuron feeds (directly or indirectly) into the weight target neuron
/// - The weight synapse feeds from or into the bias neuron's downstream
/// - Both corrections affect neurons that share a downstream output
fn are_on_connected_path(
    bias_neuron: &str,
    weight_from: &str,
    weight_to: &str,
    downstream_of: &HashMap<&str, HashSet<&str>>,
    output_uuids: &HashSet<&str>,
) -> bool {
    // Direct connection: bias neuron feeds into weight target
    if let Some(downs) = downstream_of.get(bias_neuron)
        && (downs.contains(weight_to) || downs.contains(weight_from))
    {
        return true;
    }

    // Weight target feeds from bias neuron's downstream
    if let Some(downs) = downstream_of.get(weight_from)
        && downs.contains(bias_neuron)
    {
        return true;
    }

    // Both on path to same output: check if bias neuron and weight target
    // share a common downstream output (BFS depth 1-2)
    let bias_reaches = reachable_outputs(bias_neuron, downstream_of, output_uuids, 3);
    let weight_reaches = reachable_outputs(weight_to, downstream_of, output_uuids, 3);

    // If both reach the same output, they're on connected paths
    bias_reaches.intersection(&weight_reaches).next().is_some()
}

/// Find output neurons reachable within `max_depth` hops from a starting neuron.
fn reachable_outputs<'a>(
    start: &str,
    downstream_of: &HashMap<&str, HashSet<&'a str>>,
    output_uuids: &HashSet<&str>,
    max_depth: usize,
) -> HashSet<&'a str> {
    let mut reached = HashSet::new();
    let mut frontier: Vec<&str> = vec![start];

    for _depth in 0..max_depth {
        let mut next_frontier = Vec::new();
        for node in &frontier {
            if let Some(downs) = downstream_of.get(*node) {
                for &down in downs {
                    if output_uuids.contains(down) {
                        reached.insert(down);
                    }
                    next_frontier.push(down);
                }
            }
        }
        frontier = next_frontier;
        if frontier.is_empty() {
            break;
        }
    }

    reached
}

/// A detected compound degradation requiring both bias and weight correction.
#[derive(Debug, Clone)]
pub struct CompoundDegradationCandidate {
    /// UUID of the neuron needing bias correction.
    pub bias_neuron_uuid: String,
    /// Recommended bias value.
    pub recommended_bias: f32,
    /// UUID of the source neuron for the degraded synapse.
    pub weight_from_uuid: String,
    /// UUID of the target neuron for the degraded synapse.
    pub weight_to_uuid: String,
    /// Recommended synapse weight.
    pub recommended_weight: f32,
    /// Estimated combined improvement.
    pub estimated_improvement: f32,
}

/// Convert compound degradation candidates to coordinated structural candidates.
///
/// Each compound candidate generates a two-operation coordinated group with
/// `setBias` and `setWeight` applied atomically.
pub fn compound_degradations_to_coordinated_candidates(
    candidates: &[CompoundDegradationCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    candidates
        .iter()
        .map(|c| CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            operations: vec![
                CoordinatedStructuralOpJson::SetBias {
                    neuron_uuid: c.bias_neuron_uuid.clone(),
                    bias: c.recommended_bias,
                },
                CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: c.weight_from_uuid.clone(),
                    to_neuron_uuid: c.weight_to_uuid.clone(),
                    weight: c.recommended_weight,
                },
            ],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Compound degradation: setBias({}) {:.3} and setWeight({} → {}) {:.3}. \
                 Neither fix alone fully restores performance. (Issue #929)",
                c.bias_neuron_uuid,
                c.recommended_bias,
                c.weight_from_uuid,
                c.weight_to_uuid,
                c.recommended_weight,
            )),
        })
        .collect()
}
