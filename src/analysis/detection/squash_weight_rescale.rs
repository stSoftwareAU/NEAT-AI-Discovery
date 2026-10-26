//! Squash + weight rescale detection module (Issue #548).
//!
//! Coordinates multi-neuron squash exploration with compensating weight
//! adjustments. When recommending an activation function change, this module
//! computes the weight rescaling needed to preserve the neuron's current
//! operating point, then bundles `changeSquash` with `setWeight` adjustments
//! as a single coordinated structural candidate.
//!
//! ## Rationale
//!
//! Changing a single neuron's activation function might not be enough to escape
//! a local minimum if the surrounding synapses are tuned for the old activation
//! function. A coordinated change of activation function + re-scaled weights is
//! more likely to succeed because it gives the network a better starting point
//! after the structural change.
//!
//! ## Weight Rescaling Strategy
//!
//! For each candidate squash change, we find the optimal rescale factor `f` that
//! minimises the mean squared difference between `old_squash(x)` and
//! `new_squash(f * x)` across all observed pre-activation values. Incoming
//! weights are multiplied by this factor so the neuron's post-activation output
//! is approximately preserved.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::activations::{apply_scalar_squash, is_aggregate_squash};
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum mean absolute error to consider a neuron for squash change.
/// Below this, the neuron is performing well enough with its current squash.
const MIN_ERROR_THRESHOLD: f32 = 0.03;

/// Minimum absolute rescale factor change to include weight adjustments.
/// If the rescale factor is within this of 1.0, weights don't need changing.
const MIN_RESCALE_DEVIATION: f32 = 0.05;

/// Maximum rescale factor magnitude to prevent extreme weight adjustments.
const MAX_RESCALE_FACTOR: f32 = 5.0;

/// Candidate squash functions to evaluate for replacement.
const CANDIDATE_SQUASHES: &[&str] = &[
    "TANH",
    "LOGISTIC",
    "IDENTITY",
    "SOFTSIGN",
    "HARD_TANH",
    "RELU",
    "SELU",
    "MISH",
    "SWISH",
    "ELU",
];

/// Rescale factors to search over when finding the best match.
/// We use a coarse grid search rather than numerical optimisation
/// for simplicity and robustness.
const RESCALE_CANDIDATES: &[f32] = &[
    0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.33, 1.5, 2.0, 3.0, 4.0,
];

/// Boost factor for coordinated candidates over standalone squash changes.
/// Coordinated changes preserve the operating point, so they have a higher
/// likelihood of immediate benefit.
const COORDINATED_BOOST: f32 = 1.5;

/// Returns true if the candidate neuron's output feeds a downstream aggregate
/// (MAX/MIN/IF/MEAN/HYPOT) selection neuron (Issue #1713).
///
/// The expected-error-reduction estimator simulates each candidate neuron in
/// isolation as `f(x)` and compares MAE against that neuron's *own* target. That
/// is only meaningful when the neuron's activation flows to the output through
/// pure `f(x)` neurons. When the branch instead feeds a selection aggregate, the
/// aggregate — not the neuron — decides whether the branch's value reaches the
/// output, so a change that flips the branch's activation sign/range changes
/// which branch is selected. The local estimate cannot see that effect and
/// systematically mispredicts the gain (e.g. SELU→ABSOLUTE predicted `+4.2e-10`
/// but measured `−8.7e-4`). Until a proper propagation model exists, gate such
/// candidates out rather than emit a misleading estimate.
fn feeds_downstream_aggregate(creature: &CreatureJson, neuron_uuid: &str) -> bool {
    creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid == neuron_uuid)
        .any(|s| {
            creature
                .neurons
                .iter()
                .any(|n| n.uuid == s.to_uuid && is_aggregate_squash(&n.squash))
        })
}

/// A detected squash + weight rescale candidate.
#[derive(Debug, Clone)]
pub struct SquashWeightRescaleCandidate {
    /// UUID of the neuron to change.
    pub neuron_uuid: String,
    /// Current activation function.
    pub current_squash: String,
    /// Recommended replacement activation function.
    pub recommended_squash: String,
    /// Rescaled incoming weights: (`from_uuid`, `to_uuid`, `new_weight`).
    pub rescaled_weights: Vec<(String, String, f32)>,
    /// Estimated error reduction from the coordinated change.
    pub estimated_improvement: f32,
    /// Human-readable explanation.
    pub reason: String,
}

/// Evaluate the mean absolute error for a candidate squash with a given
/// rescale factor, compared to the targets derived from recorded activations.
///
/// Returns `(candidate_mae, current_mae)` — the mean absolute error of the
/// candidate squash and the current squash against the target.
fn evaluate_squash_errors(
    records: &[DiscoverRecord],
    current_squash: &str,
    candidate_squash: &str,
    rescale_factor: f32,
) -> Option<(f32, f32)> {
    let mut current_total = 0.0f32;
    let mut candidate_total = 0.0f32;
    let mut count = 0u32;

    for r in records {
        let Some(pre_act) = r.value else {
            continue;
        };

        let Some(_current_output) = apply_scalar_squash(current_squash, pre_act) else {
            continue;
        };
        let Some(candidate_output) =
            apply_scalar_squash(candidate_squash, pre_act * rescale_factor)
        else {
            continue;
        };

        // target = activation - error (since error = activation - target)
        let signed_error = r.errors.first().copied().unwrap_or(0.0);
        let target = r.activation - signed_error;

        current_total += (r.activation - target).abs();
        candidate_total += (candidate_output - target).abs();
        count += 1;
    }

    if count == 0 {
        return None;
    }

    let n = count as f32;
    Some((candidate_total / n, current_total / n))
}

/// Find the best rescale factor for a candidate squash by grid search.
///
/// Returns `(best_factor, improvement)` where improvement is the mean error
/// reduction compared to the current squash.
fn find_best_rescale_factor(
    records: &[DiscoverRecord],
    current_squash: &str,
    candidate_squash: &str,
) -> Option<(f32, f32)> {
    let mut best_factor = 1.0f32;
    let mut best_improvement = f32::NEG_INFINITY;

    for &factor in RESCALE_CANDIDATES {
        let Some((candidate_mae, current_mae)) =
            evaluate_squash_errors(records, current_squash, candidate_squash, factor)
        else {
            continue;
        };

        let improvement = current_mae - candidate_mae;
        if improvement > best_improvement {
            best_improvement = improvement;
            best_factor = factor;
        }
    }

    if best_improvement > 0.0 {
        Some((best_factor, best_improvement))
    } else {
        None
    }
}

/// Detect neurons that would benefit from a coordinated squash change with
/// compensating weight rescaling.
///
/// # Arguments
/// * `creature` - The creature topology (for synapse lookup).
/// * `hidden_neurons` - Slice of `(uuid, squash, bias)` tuples for hidden neurons.
/// * `neuron_records` - Slice of `(uuid, records)` pairs with observation data.
///
/// # Returns
/// Vector of detected candidates, sorted by estimated improvement (best first).
pub fn detect_squash_weight_rescale_candidates(
    creature: &CreatureJson,
    hidden_neurons: &[(String, String, f32)],
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<SquashWeightRescaleCandidate> {
    let mut candidates = Vec::with_capacity(hidden_neurons.len());

    for (uuid, current_squash, _bias) in hidden_neurons {
        // Skip aggregate squashes — they cannot be simulated as f(x)
        if is_aggregate_squash(current_squash) {
            continue;
        }

        // Gate out candidates whose branch feeds a downstream aggregate
        // selection (MAX/MIN/IF/…). The local f(x) estimate cannot model how a
        // changed activation range alters which branch the aggregate selects,
        // so it systematically mispredicts the whole-creature gain (Issue #1713).
        if feeds_downstream_aggregate(creature, uuid) {
            continue;
        }

        let Some((_id, records)) = neuron_records.iter().find(|(u, _)| u == uuid) else {
            continue;
        };
        let records = records.as_ref();

        if records.len() < MIN_SAMPLES {
            continue;
        }

        // Require pre-activation values for rescaling computation
        let pre_act_count = records.iter().filter(|r| r.value.is_some()).count();
        if pre_act_count < MIN_SAMPLES {
            continue;
        }

        // Check if error is high enough to warrant a change
        let mean_abs_error: f32 = records
            .iter()
            .map(|r| r.errors.first().copied().unwrap_or(0.0).abs())
            .sum::<f32>()
            / records.len() as f32;

        if mean_abs_error < MIN_ERROR_THRESHOLD {
            continue;
        }

        // Find incoming synapses for this neuron
        let incoming_synapses: Vec<(&str, f32)> = creature
            .synapses
            .iter()
            .filter(|s| s.to_uuid == *uuid)
            .map(|s| (s.from_uuid.as_str(), s.weight))
            .collect();

        if incoming_synapses.is_empty() {
            continue;
        }

        // Evaluate each candidate squash with optimal rescale factor
        let mut best_candidate: Option<(String, f32, f32)> = None; // (squash, factor, improvement)

        for &candidate_squash in CANDIDATE_SQUASHES {
            // Skip if same as current
            if candidate_squash.eq_ignore_ascii_case(current_squash) {
                continue;
            }

            let Some((factor, improvement)) =
                find_best_rescale_factor(records, current_squash, candidate_squash)
            else {
                continue;
            };

            // Validate factor is within acceptable range
            if factor.abs() > MAX_RESCALE_FACTOR || !factor.is_finite() {
                continue;
            }

            if let Some((_, _, best_imp)) = &best_candidate {
                if improvement > *best_imp {
                    best_candidate = Some((candidate_squash.to_string(), factor, improvement));
                }
            } else {
                best_candidate = Some((candidate_squash.to_string(), factor, improvement));
            }
        }

        let Some((recommended_squash, rescale_factor, improvement)) = best_candidate else {
            continue;
        };

        // Compute rescaled weights for all incoming synapses
        let needs_rescaling = (rescale_factor - 1.0).abs() >= MIN_RESCALE_DEVIATION;
        let rescaled_weights: Vec<(String, String, f32)> = incoming_synapses
            .iter()
            .map(|(from_uuid, old_weight)| {
                let new_weight = if needs_rescaling {
                    old_weight * rescale_factor
                } else {
                    *old_weight
                };
                (from_uuid.to_string(), uuid.clone(), new_weight)
            })
            .collect();

        // Boosted improvement for coordinated change
        let boosted_improvement = improvement * COORDINATED_BOOST;

        let weight_info = if needs_rescaling {
            format!(" Weight rescale factor: {rescale_factor:.3}.")
        } else {
            String::new()
        };

        candidates.push(SquashWeightRescaleCandidate {
            neuron_uuid: uuid.clone(),
            current_squash: current_squash.clone(),
            recommended_squash: recommended_squash.clone(),
            rescaled_weights,
            estimated_improvement: boosted_improvement,
            reason: format!(
                "Hidden neuron {uuid}: coordinated {current_squash} → {recommended_squash} with compensating weight adjustments.{weight_info} Mean error: {mean_abs_error:.3}. (Issue #548)",
            ),
        });
    }

    // Sort by estimated improvement descending
    candidates.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Convert squash + weight rescale candidates to coordinated structural candidates.
///
/// Each candidate becomes a coordinated group containing:
/// - One `changeSquash` operation for the neuron
/// - One `setWeight` operation per incoming synapse (with rescaled weight)
pub fn squash_weight_rescale_to_coordinated_candidates(
    candidates: &[SquashWeightRescaleCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    candidates
        .iter()
        .map(|c| {
            let mut operations = vec![CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: c.neuron_uuid.clone(),
                squash: c.recommended_squash.clone(),
            }];

            // Add setWeight for each incoming synapse
            for (from_uuid, to_uuid, new_weight) in &c.rescaled_weights {
                operations.push(CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: from_uuid.clone(),
                    to_neuron_uuid: to_uuid.clone(),
                    weight: *new_weight,
                });
            }

            CoordinatedStructuralCandidateJson {
                remove_neuron_compensation: None,
                constant_neuron_bias_fold: None,
                operations,
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(c.reason.clone()),
            }
        })
        .collect()
}
