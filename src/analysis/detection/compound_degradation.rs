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
//!
//! ## Scan bounds (Issue #2367)
//!
//! The combination step is a pair scan over bias corrections `B` and weight
//! corrections `W`, each pair needing a reachability check. Before the pair
//! loop only the strongest [`MAX_COMPOUND_BIAS_CORRECTIONS`] bias and
//! [`MAX_COMPOUND_WEIGHT_CORRECTIONS`] weight corrections (by estimated
//! improvement) are kept, bounding pairs to 256·256 = 65,536. Reachable-output
//! sets are computed once per distinct neuron with a visited set (`O(V+E)`
//! each), not once per pair. Emitted candidates are capped at
//! [`MAX_COMPOUND_DEGRADATION_CANDIDATES`]. The deadline-aware entry point
//! [`detect_compound_bias_weight_degradations_with_deadline`] checks the shared
//! analysis deadline and global cancellation once per precomputed neuron and
//! once per outer (bias) row, and reports any early stop or cap through
//! [`ScanTruncation`](crate::analysis::recommendation::epistatic::ScanTruncation).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use super::helpers::build_record_map;
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;
use crate::analysis::quantised_error::is_quantised_zero_one;
use crate::analysis::recommendation::epistatic::ScanTruncation;
use crate::analysis::utils::deadline_passed;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};
use std::collections::{HashMap, HashSet};
use std::time::SystemTime;

/// Minimum mean absolute error to consider a neuron for bias correction.
const MIN_BIAS_ERROR: f32 = 0.02;

/// Minimum absolute bias delta to emit a correction.
const MIN_BIAS_DELTA: f32 = 0.05;

/// Minimum absolute weight delta to consider a synapse degraded.
const MIN_WEIGHT_DELTA: f32 = 0.05;

/// Ceiling on the bias corrections admitted to the pair scan (Issue #2367).
///
/// Only the strongest corrections by estimated improvement are kept, so the
/// pair loop is bounded to `MAX_COMPOUND_BIAS_CORRECTIONS ×
/// MAX_COMPOUND_WEIGHT_CORRECTIONS` = 65,536 pairs regardless of creature size.
pub const MAX_COMPOUND_BIAS_CORRECTIONS: usize = 256;

/// Ceiling on the weight corrections admitted to the pair scan (Issue #2367).
///
/// Only the strongest corrections by estimated improvement are kept (see
/// [`MAX_COMPOUND_BIAS_CORRECTIONS`]).
pub const MAX_COMPOUND_WEIGHT_CORRECTIONS: usize = 256;

/// Ceiling on the compound candidates emitted by the scan (Issue #2367).
///
/// Candidates are sorted by estimated improvement (descending) before the
/// ceiling is applied, so the strongest candidates are kept.
pub const MAX_COMPOUND_DEGRADATION_CANDIDATES: usize = 256;

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
/// Applies the Issue #2367 correction and candidate ceilings but no deadline:
/// equivalent to [`detect_compound_bias_weight_degradations_with_deadline`]
/// with a `None` deadline, keeping only `.candidates`.
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
    detect_compound_bias_weight_degradations_with_deadline(creature, neuron_records, &None)
        .candidates
}

/// Deadline- and cancellation-aware variant of
/// [`detect_compound_bias_weight_degradations`] (Issue #2367).
///
/// Applies the ceilings [`MAX_COMPOUND_BIAS_CORRECTIONS`],
/// [`MAX_COMPOUND_WEIGHT_CORRECTIONS`] and [`MAX_COMPOUND_DEGRADATION_CANDIDATES`],
/// and checks `deadline` once per precomputed neuron and once per outer (bias)
/// row. When the deadline has passed (or cancellation was requested) the scan
/// returns the candidates found so far, sorted, with
/// `truncation = Some(ScanTruncation::DeadlinePassed)`. If the deadline fires
/// while reachable sets are still being precomputed, no pair has been
/// evaluated and the candidate list is empty. When a ceiling dropped entries
/// and no deadline fired, `truncation` is `Some(ScanTruncation::CandidateCeiling)`.
pub fn detect_compound_bias_weight_degradations_with_deadline(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    deadline: &Option<SystemTime>,
) -> CompoundDegradationScan {
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
    let mut bias_corrections =
        detect_bias_corrections(creature, &records_map, &neuron_squash, &neuron_bias);

    // Phase 2: Detect weight corrections for synapses
    let mut weight_corrections = detect_weight_corrections(creature, &records_map);

    // Issue #2367: keep only the strongest corrections before the pair loop.
    // Sorting happens only when a cap is hit, so uncapped inputs keep their
    // detection order.
    let mut corrections_capped = false;
    if bias_corrections.len() > MAX_COMPOUND_BIAS_CORRECTIONS {
        bias_corrections
            .sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));
        bias_corrections.truncate(MAX_COMPOUND_BIAS_CORRECTIONS);
        corrections_capped = true;
    }
    if weight_corrections.len() > MAX_COMPOUND_WEIGHT_CORRECTIONS {
        weight_corrections
            .sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));
        weight_corrections.truncate(MAX_COMPOUND_WEIGHT_CORRECTIONS);
        corrections_capped = true;
    }

    // Build a set of output neuron UUIDs for path validation
    let output_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    // Phase 3a: memoise each distinct neuron's reachable-output set once,
    // rather than walking the graph for every (bias, weight) pair (Issue #2367).
    let mut reachable: HashMap<&str, HashSet<&str>> = HashMap::new();
    let precompute_targets = bias_corrections
        .iter()
        .map(|c| c.neuron_uuid.as_str())
        .chain(weight_corrections.iter().map(|c| c.to_neuron_uuid.as_str()));
    for uuid in precompute_targets {
        if reachable.contains_key(uuid) {
            continue;
        }
        if deadline_passed(deadline) {
            return CompoundDegradationScan {
                candidates: Vec::new(),
                truncation: Some(ScanTruncation::DeadlinePassed),
            };
        }
        reachable.insert(
            uuid,
            reachable_outputs(uuid, &downstream_of, &output_uuids, 3),
        );
    }

    // Phase 3b: Combine related corrections
    let mut scan = combine_corrections(
        &bias_corrections,
        &weight_corrections,
        &downstream_of,
        &reachable,
        deadline,
    );
    if scan.truncation.is_none() && corrections_capped {
        scan.truncation = Some(ScanTruncation::CandidateCeiling);
    }
    scan
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
        // Issue #2351: squared-error sums can overflow to +inf (NaN improvement).
        if !improvement.is_finite() || improvement <= 0.0 {
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
/// coordinated candidates (Issue #2367: deadline-aware and capped).
///
/// `reachable` must hold the precomputed reachable-output set for every bias
/// neuron and weight target passed in. The deadline is checked once per outer
/// (bias) row; on a pass the candidates found so far are returned with
/// `DeadlinePassed`. Candidates are sorted by estimated improvement (descending)
/// and capped at [`MAX_COMPOUND_DEGRADATION_CANDIDATES`].
fn combine_corrections(
    bias_corrections: &[BiasCorrection],
    weight_corrections: &[WeightCorrection],
    downstream_of: &HashMap<&str, HashSet<&str>>,
    reachable: &HashMap<&str, HashSet<&str>>,
    deadline: &Option<SystemTime>,
) -> CompoundDegradationScan {
    let mut candidates = Vec::new();
    let mut truncation = None;

    for bias_corr in bias_corrections {
        // Issue #2367: check once per outer row so a long scan can be stopped.
        if deadline_passed(deadline) {
            truncation = Some(ScanTruncation::DeadlinePassed);
            break;
        }

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
                reachable,
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

    // Issue #2367: cap the emitted candidates, keeping the strongest.
    if candidates.len() > MAX_COMPOUND_DEGRADATION_CANDIDATES {
        candidates.truncate(MAX_COMPOUND_DEGRADATION_CANDIDATES);
        truncation = truncation.or(Some(ScanTruncation::CandidateCeiling));
    }

    CompoundDegradationScan {
        candidates,
        truncation,
    }
}

/// Check whether two corrections are on a connected forward path.
///
/// Returns true if:
/// - The bias neuron feeds (directly or indirectly) into the weight target neuron
/// - The weight synapse feeds from or into the bias neuron's downstream
/// - Both corrections affect neurons that share a downstream output
///
/// The shared-output test reads the precomputed `reachable` sets (Issue #2367)
/// rather than walking the graph per pair. `combine_corrections` is only ever
/// given a map that holds both neurons, so the lookup cannot miss; a missing
/// entry is treated as no shared output.
fn are_on_connected_path(
    bias_neuron: &str,
    weight_from: &str,
    weight_to: &str,
    downstream_of: &HashMap<&str, HashSet<&str>>,
    reachable: &HashMap<&str, HashSet<&str>>,
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
    // share a common downstream output (precomputed within depth 3)
    let (Some(bias_reaches), Some(weight_reaches)) =
        (reachable.get(bias_neuron), reachable.get(weight_to))
    else {
        return false;
    };

    // If both reach the same output, they're on connected paths
    bias_reaches.intersection(weight_reaches).next().is_some()
}

/// Find output neurons reachable within `max_depth` hops from a starting neuron.
///
/// Breadth-first, and never re-pushes an already-visited neuron into the
/// frontier, so each neuron is expanded at most once and a call is `O(V+E)`
/// (Issue #2367). The result is the set of outputs reached at the end of any
/// walk of 1..=`max_depth` edges from `start`, cycles included. `start` is not
/// pre-marked as visited, so a walk that returns to it within the depth limit
/// still reports it when it is an output.
fn reachable_outputs<'a>(
    start: &str,
    downstream_of: &HashMap<&str, HashSet<&'a str>>,
    output_uuids: &HashSet<&str>,
    max_depth: usize,
) -> HashSet<&'a str> {
    let mut reached = HashSet::new();
    // Neurons already queued for expansion; prevents re-expansion of fan-in nodes.
    let mut visited = HashSet::new();
    let mut frontier: Vec<&str> = vec![start];

    for _depth in 0..max_depth {
        let mut next_frontier = Vec::new();
        for node in &frontier {
            if let Some(downs) = downstream_of.get(*node) {
                for &down in downs {
                    if output_uuids.contains(down) {
                        reached.insert(down);
                    }
                    if visited.insert(down) {
                        next_frontier.push(down);
                    }
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

/// Result of a deadline- and ceiling-bounded compound degradation scan
/// (Issue #2367).
#[derive(Debug, Clone)]
pub struct CompoundDegradationScan {
    /// Compound candidates found before the scan finished or stopped, sorted
    /// by estimated improvement (highest first).
    pub candidates: Vec<CompoundDegradationCandidate>,
    /// `Some` when the scan stopped early (deadline or cancellation) or when a
    /// ceiling dropped entries.
    pub truncation: Option<ScanTruncation>,
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
            constant_neuron_bias_fold: None,
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

#[cfg(test)]
#[path = "compound_degradation_tests.rs"]
mod tests;
