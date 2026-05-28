//! Output bias drift detection module (Issue #361).
//!
//! Identifies output neurons with a consistent error sign bias — that is, neurons
//! whose errors are predominantly positive (predicting too low) or predominantly
//! negative (predicting too high) across training samples. This systematic bias
//! indicates the neuron's bias parameter needs adjustment.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Output Bias Drift Detection" for full documentation.
//!
//! ## Detection Criteria
//!
//! An output neuron has "bias drift" if:
//! 1. **Consistent error sign**: More than a threshold fraction of errors share the
//!    same sign (e.g., > 70% positive or > 70% negative).
//! 2. **Meaningful mean error**: The absolute mean error is above a minimum threshold
//!    (not just noise).
//! 3. **Sufficient samples**: Enough recorded samples for statistical reliability.
//! 4. **Only output neurons**: Hidden and input neurons are excluded (their errors
//!    are indirect).
//!
//! ## Recommended Actions
//!
//! When bias drift is detected, we recommend:
//! 1. **Set bias**: Adjust the output neuron's bias by the negative of the mean error
//!    to centre the predictions.
//!
//! This is emitted as a `CoordinatedStructuralCandidateJson` with a `SetBias` operation.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashMap;

use crate::analysis::task_descriptor::{TargetTopology, TaskDescriptor};
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

// MIN_SAMPLES_FOR_BIAS_DRIFT moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_BIAS_DRIFT;

/// Minimum fraction of errors that must share the same sign to consider
/// the output biased. 0.7 means 70%+ must be positive or negative.
const MIN_MAJORITY_SIGN_FRACTION: f32 = 0.7;

/// Minimum absolute mean error to distinguish from noise.
const MIN_MEAN_ERROR_MAGNITUDE: f32 = 0.01;

/// Capacity-starvation threshold (Issue #1316).
///
/// A class is considered "positively supported" when the recorded target
/// value is above this threshold. For `OneHot` / `Simplex` topologies the
/// targets sit on `[0, 1]` and the "on" class records as 1.0, so any
/// threshold in the middle of the unit interval reliably separates "on"
/// from "off" while tolerating soft labels.
const POSITIVE_SUPPORT_TARGET_THRESHOLD: f32 = 0.5;

/// Saturating activation threshold (Issue #1316).
///
/// Mirrors the bounded-unipolar saturation threshold used by the saturated
/// neuron detector. An output neuron whose maximum activation on its
/// positive-support class never reaches this value is treated as
/// capacity-starved for that class.
const SATURATING_ACTIVATION_THRESHOLD: f32 = 0.85;

/// Multiplier applied to the estimated improvement of a candidate that is
/// also flagged as capacity-starved under a `OneHot` / `Simplex` task
/// (Issue #1316). Picked to lift the candidate above peers without
/// overwhelming the rest of the ranking.
const CAPACITY_STARVED_GAIN_BOOST: f32 = 2.0;

/// Result of detecting output bias drift.
#[derive(Debug, Clone)]
pub struct OutputBiasDriftCandidate {
    /// UUID of the output neuron with bias drift.
    pub neuron_uuid: String,
    /// Current bias of the neuron.
    pub current_bias: f32,
    /// Mean error across all samples (positive = predicting too low).
    pub mean_error: f32,
    /// Fraction of samples with positive error.
    pub positive_error_fraction: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Recommended bias adjustment (negative of mean error).
    pub recommended_bias_delta: f32,
    /// Estimated creature score improvement from fixing the bias.
    pub estimated_improvement: f32,
    /// Whether this candidate was flagged as capacity-starved under a
    /// `OneHot` / `Simplex` task descriptor (Issue #1316). Always `false`
    /// for the legacy `detect_output_bias_drift` entry point; only the
    /// role-aware `detect_output_bias_drift_with_descriptor` path can
    /// set this to `true`.
    pub capacity_starved: bool,
}

/// Detect output neurons with bias drift from recorded activations.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `OutputBiasDriftCandidate` for output neurons with systematic bias,
/// sorted by estimated improvement (best first).
pub fn detect_output_bias_drift(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<OutputBiasDriftCandidate> {
    // Identify output neurons with their bias
    let output_neurons: HashMap<&str, f32> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| (n.uuid.as_str(), n.bias))
        .collect();

    // Build records lookup
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    let mut candidates = Vec::new();

    for (&uuid, &current_bias) in &output_neurons {
        let Some(records) = records_map.get(uuid) else {
            continue;
        };

        if records.len() < MIN_SAMPLES_FOR_BIAS_DRIFT {
            continue;
        }

        // Collect all first-error values (error index 0 for this output)
        let error_values: Vec<f32> = records
            .iter()
            .filter_map(|r| r.errors.first().copied())
            .collect();

        if error_values.len() < MIN_SAMPLES_FOR_BIAS_DRIFT {
            continue;
        }

        let n = error_values.len() as f32;

        // Compute mean error
        let sum_error: f32 = error_values.iter().sum();
        let mean_error = sum_error / n;

        if mean_error.abs() < MIN_MEAN_ERROR_MAGNITUDE {
            continue;
        }

        // Count positive and negative errors
        let positive_count = error_values.iter().filter(|&&e| e > 0.0).count();
        let positive_error_fraction = positive_count as f32 / n;

        // Check if one sign dominates
        let majority_fraction = positive_error_fraction.max(1.0 - positive_error_fraction);
        if majority_fraction < MIN_MAJORITY_SIGN_FRACTION {
            continue;
        }

        // Recommended bias adjustment: shift by negative of mean error
        // to centre predictions (reduce systematic bias)
        let recommended_bias_delta = -mean_error;

        // Estimate improvement: proportional to mean error magnitude and consistency
        let consistency = majority_fraction - 0.5; // 0.0 to 0.5
        let estimated_improvement = mean_error.abs() * consistency * 0.1;

        candidates.push(OutputBiasDriftCandidate {
            neuron_uuid: uuid.to_string(),
            current_bias,
            mean_error,
            positive_error_fraction,
            sample_count: error_values.len(),
            recommended_bias_delta,
            estimated_improvement,
            capacity_starved: false,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Convert output bias drift candidates into coordinated structural candidates.
///
/// Each output neuron with bias drift produces a `SetBias` coordinated candidate.
pub fn output_bias_drift_to_coordinated_candidates(
    candidates: &[OutputBiasDriftCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: c.neuron_uuid.clone(),
                bias: c.current_bias + c.recommended_bias_delta,
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Output bias drift {}: mean error {:.4} ({:.0}% same sign), current bias {:.4} → adjust bias by {:.4} to centre predictions",
                c.neuron_uuid,
                c.mean_error,
                c.positive_error_fraction.max(1.0 - c.positive_error_fraction) * 100.0,
                c.current_bias,
                c.recommended_bias_delta
            )),
        });
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    results
}

// =============================================================================
// Role-aware output bias drift (Issue #1316)
// =============================================================================
//
// Under `OneHot` / `Simplex` topologies the recorded target marks exactly one
// "on" class per sample. An output neuron whose maximum activation on the
// samples that *belong to its class* never crosses the saturating threshold is
// suffering capacity starvation — its parameters cannot push the prediction
// high enough even when the class is well supported. The role-aware path runs
// the legacy detector first (so unrelated callers are unaffected) and then,
// when the descriptor topology is `OneHot` or `Simplex`, boosts the
// estimated improvement of any matching candidate and synthesises a fresh
// candidate for capacity-starved output neurons that the legacy detector
// missed.

/// Whether a descriptor's topology gates the role-aware bias-drift path.
fn role_aware_topology(descriptor: &TaskDescriptor) -> bool {
    matches!(
        descriptor.target_topology,
        TargetTopology::OneHot | TargetTopology::Simplex
    )
}

/// Diagnostics about an output neuron's positive-support class.
struct PositiveSupportStats {
    /// Count of records on the positive-support class (target above the
    /// `POSITIVE_SUPPORT_TARGET_THRESHOLD`).
    count: usize,
    /// Maximum activation observed on the positive-support records.
    max_activation: f32,
    /// Mean activation observed on the positive-support records.
    mean_activation: f32,
    /// Mean error on the positive-support records (target − activation under
    /// linear-residual costs).
    mean_error: f32,
}

/// Summarise the positive-support behaviour of an output neuron given its
/// records. Returns `None` when the recorded targets carry no positive
/// support (no `value > POSITIVE_SUPPORT_TARGET_THRESHOLD`).
fn summarise_positive_support(records: &[DiscoverRecord]) -> Option<PositiveSupportStats> {
    let mut count = 0_usize;
    let mut max_activation = f32::NEG_INFINITY;
    let mut sum_activation = 0.0_f32;
    let mut sum_error = 0.0_f32;

    for r in records {
        // Targets are recorded in `value` (Option<f32>). Records without a
        // recorded target cannot contribute to positive-support reasoning.
        let Some(target) = r.value else { continue };
        if target <= POSITIVE_SUPPORT_TARGET_THRESHOLD {
            continue;
        }
        count += 1;
        sum_activation += r.activation;
        if r.activation > max_activation {
            max_activation = r.activation;
        }
        if let Some(&e) = r.errors.first() {
            sum_error += e;
        }
    }

    if count == 0 {
        return None;
    }

    let n = count as f32;
    Some(PositiveSupportStats {
        count,
        max_activation,
        mean_activation: sum_activation / n,
        mean_error: sum_error / n,
    })
}

/// Determine whether the positive-support stats indicate capacity starvation:
/// a well-supported class whose activations never cross the saturating
/// threshold.
fn is_capacity_starved(stats: &PositiveSupportStats) -> bool {
    stats.count >= MIN_SAMPLES_FOR_BIAS_DRIFT
        && stats.max_activation < SATURATING_ACTIVATION_THRESHOLD
}

/// Role-aware output bias drift detection (Issue #1316).
///
/// Behaves identically to [`detect_output_bias_drift`] when the descriptor
/// reports a topology other than `OneHot` / `Simplex` — neutral, `OTHER`,
/// `Independent`, `Margin`, and any future variant fall back to the legacy
/// detector. For `OneHot` / `Simplex` descriptors, output neurons with
/// positive class support whose activations never cross the saturating
/// activation threshold are flagged as capacity-starved and their estimated
/// improvement is boosted by a fixed gain multiplier. Capacity-starved
/// neurons that the legacy detector missed receive a fresh candidate whose
/// `recommended_bias_delta` matches the legacy convention (`−mean_error` on
/// the positive-support records).
pub fn detect_output_bias_drift_with_descriptor(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
    descriptor: &TaskDescriptor,
) -> Vec<OutputBiasDriftCandidate> {
    let mut candidates = detect_output_bias_drift(creature, neuron_records);

    if !role_aware_topology(descriptor) {
        return candidates;
    }

    let output_neurons: HashMap<&str, f32> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| (n.uuid.as_str(), n.bias))
        .collect();

    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    for (&uuid, &current_bias) in &output_neurons {
        let Some(records) = records_map.get(uuid) else {
            continue;
        };
        let Some(stats) = summarise_positive_support(records) else {
            continue;
        };
        if !is_capacity_starved(&stats) {
            continue;
        }

        if let Some(existing) = candidates.iter_mut().find(|c| c.neuron_uuid == uuid) {
            existing.capacity_starved = true;
            existing.estimated_improvement *= CAPACITY_STARVED_GAIN_BOOST;
            continue;
        }

        // No legacy candidate for this neuron — synthesise one. Picking the
        // bias delta from the positive-support mean error matches the legacy
        // convention (`recommended_bias_delta = −mean_error`) and falls back
        // to closing the gap to saturation when the recorded errors are
        // unusable.
        let recommended_bias_delta = if stats.mean_error.is_finite() && stats.mean_error.abs() > 0.0
        {
            -stats.mean_error
        } else {
            SATURATING_ACTIVATION_THRESHOLD - stats.mean_activation
        };
        let gap = (SATURATING_ACTIVATION_THRESHOLD - stats.max_activation).max(0.0);
        let estimated_improvement = gap * 0.1 * CAPACITY_STARVED_GAIN_BOOST;

        candidates.push(OutputBiasDriftCandidate {
            neuron_uuid: uuid.to_string(),
            current_bias,
            mean_error: stats.mean_error,
            positive_error_fraction: 0.0,
            sample_count: stats.count,
            recommended_bias_delta,
            estimated_improvement,
            capacity_starved: true,
        });
    }

    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));
    candidates
}
