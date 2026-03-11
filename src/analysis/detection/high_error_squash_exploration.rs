//! High-error squash exploration detection module (Issue #788).
//!
//! Proactively explores alternative activation functions for hidden neurons
//! that exhibit high prediction error. Unlike reactive modules (saturation,
//! mismatch), this module triggers on **error magnitude** — if a neuron's
//! mean absolute error is above a threshold, it simulates what each candidate
//! activation function would produce from the neuron's pre-activation values
//! and recommends the one that best reduces error.
//!
//! This increases change-squash candidate volume for the high-success-rate
//! change-squash candidate type (65% success rate in GRQ-sampler analysis).
//!
//! ## Detection Criteria
//!
//! - Pre-activation (`value`) data available for ≥ `MIN_SAMPLES` records
//! - Mean absolute error ≥ `MIN_MEAN_ERROR_THRESHOLD`
//! - At least one alternative activation function reduces error by ≥
//!   `MIN_ERROR_REDUCTION_FRACTION` relative to the current activation
//! - The neuron's current activation is not IDENTITY (already linear)

use crate::activations::apply_scalar_squash;
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Minimum mean absolute error required to consider a neuron for exploration.
///
/// Neurons with error below this threshold are performing well enough that
/// changing the activation function is unlikely to yield meaningful improvement.
const MIN_MEAN_ERROR_THRESHOLD: f32 = 0.10;

/// Minimum relative error reduction to recommend a squash replacement.
///
/// The alternative must reduce mean absolute error by at least this fraction
/// compared to the current activation. Set conservatively to maintain a high
/// success rate (target: >30%).
const MIN_ERROR_REDUCTION_FRACTION: f32 = 0.15;

/// Candidate squash functions to evaluate as replacements.
const CANDIDATE_SQUASHES: &[&str] = &[
    "TANH",
    "LOGISTIC",
    "IDENTITY",
    "SOFTSIGN",
    "HARD_TANH",
    "RELU",
    "ELU",
    "SELU",
    "MISH",
    "SWISH",
];

/// Squash functions that should not trigger this module (already flexible).
fn is_skip_squash(squash: &str) -> bool {
    matches!(squash, "IDENTITY")
}

/// Result of detecting a high-error squash exploration candidate.
#[derive(Debug, Clone)]
pub struct HighErrorSquashCandidate {
    /// UUID of the neuron.
    pub neuron_uuid: String,
    /// Current activation function.
    pub current_squash: String,
    /// Recommended replacement activation function.
    pub recommended_squash: Option<String>,
    /// Mean absolute error of the neuron with its current activation.
    pub mean_absolute_error: f32,
    /// Fraction of error reduction achieved by the recommended activation.
    pub error_reduction_fraction: f32,
    /// Estimated improvement from switching activation.
    pub estimated_improvement: f32,
    /// Human-readable explanation.
    pub reason: String,
}

/// Detect neurons with high prediction error that could benefit from a
/// different activation function.
///
/// For each neuron with sufficient pre-activation data and high error, this
/// function simulates every candidate activation function and selects the
/// one that most reduces the mean absolute error.
///
/// # Arguments
/// * `neurons` - List of `(uuid, squash, bias)` tuples for hidden neurons.
/// * `neuron_records` - List of `(uuid, records)` tuples with recorded samples.
///
/// # Returns
/// A list of `HighErrorSquashCandidate` sorted by estimated improvement
/// (descending).
pub fn detect_high_error_squash_candidates(
    neurons: &[(String, String, f32)],
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<HighErrorSquashCandidate> {
    let records_map: std::collections::HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    let mut candidates = Vec::new();

    for (uuid, squash, _bias) in neurons {
        if is_skip_squash(squash) {
            continue;
        }

        let Some(records) = records_map.get(uuid.as_str()) else {
            continue;
        };

        if records.len() < MIN_SAMPLES {
            continue;
        }

        if let Some(candidate) = evaluate_neuron(uuid, squash, records) {
            candidates.push(candidate);
        }
    }

    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));
    candidates
}

/// Evaluate a single neuron for high-error squash exploration.
///
/// Uses inferred targets: for each sample, `target = activation + error`,
/// representing what the neuron should be outputting. The current squash's
/// MAE against this target equals the mean absolute error. We then check
/// whether an alternative squash applied to the same pre-activation values
/// would get closer to the target.
fn evaluate_neuron(
    uuid: &str,
    current_squash: &str,
    records: &[DiscoverRecord],
) -> Option<HighErrorSquashCandidate> {
    // Collect samples with pre-activation values, errors, and inferred targets
    let samples: Vec<(f32, f32)> = records
        .iter()
        .filter_map(|r| {
            let pre_act = r.value?;
            let error = r.errors.first().copied().unwrap_or(0.0);
            let target = r.activation + error;
            Some((pre_act, target))
        })
        .collect();

    if samples.len() < MIN_SAMPLES {
        return None;
    }

    // The current activation's MAE equals the mean absolute error
    // (since target = activation + error, |activation - target| = |error|)
    let current_mae = compute_mae_for_squash(current_squash, &samples);

    if current_mae < MIN_MEAN_ERROR_THRESHOLD {
        return None;
    }

    // Try each candidate squash and find the best improvement
    let mut best_squash: Option<&str> = None;
    let mut best_mae = current_mae;

    for &candidate_name in CANDIDATE_SQUASHES {
        if candidate_name.eq_ignore_ascii_case(current_squash) {
            continue;
        }

        let candidate_mae = compute_mae_for_squash(candidate_name, &samples);

        if candidate_mae < best_mae {
            best_mae = candidate_mae;
            best_squash = Some(candidate_name);
        }
    }

    let best_squash = best_squash?;

    let reduction_fraction = (current_mae - best_mae) / current_mae;
    if reduction_fraction < MIN_ERROR_REDUCTION_FRACTION {
        return None;
    }

    // Scale improvement conservatively to maintain high success rate
    let estimated_improvement = reduction_fraction * current_mae * 0.01;

    Some(HighErrorSquashCandidate {
        neuron_uuid: uuid.to_string(),
        current_squash: current_squash.to_string(),
        recommended_squash: Some(best_squash.to_string()),
        mean_absolute_error: current_mae,
        error_reduction_fraction: reduction_fraction,
        estimated_improvement,
        reason: format!(
            "High error neuron (MAE={:.3}) — {} reduces simulated error by {:.0}% vs {}",
            current_mae,
            best_squash,
            reduction_fraction * 100.0,
            current_squash,
        ),
    })
}

/// Compute mean absolute error for a given squash function applied to
/// pre-activation values, compared against inferred targets.
///
/// Each sample is `(pre_activation, target)` where
/// `target = observed_activation + error`. The MAE measures how far
/// `apply_squash(pre_act)` is from the inferred target.
fn compute_mae_for_squash(squash: &str, samples: &[(f32, f32)]) -> f32 {
    let mut total_error = 0.0f32;
    let mut valid_count = 0usize;

    for &(pre_act, target) in samples {
        let Some(simulated) = apply_scalar_squash(squash, pre_act) else {
            continue;
        };

        if !simulated.is_finite() {
            continue;
        }

        total_error += (simulated - target).abs();
        valid_count += 1;
    }

    if valid_count < MIN_SAMPLES {
        return f32::INFINITY;
    }

    total_error / valid_count as f32
}

/// Convert high-error squash exploration candidates to coordinated structural
/// candidates.
///
/// Each candidate becomes a single `ChangeSquash` operation.
pub fn high_error_squash_to_coordinated_candidates(
    candidates: &[HighErrorSquashCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    candidates
        .iter()
        .filter_map(|c| {
            let recommended = c.recommended_squash.as_ref()?;

            Some(CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
                    neuron_uuid: c.neuron_uuid.clone(),
                    squash: recommended.clone(),
                }],
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "High-error squash exploration (Issue #788): {} → {}. {}",
                    c.current_squash, recommended, c.reason,
                )),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(
        uuid: &str,
        idx: u32,
        value: Option<f32>,
        activation: f32,
        error: f32,
    ) -> DiscoverRecord {
        DiscoverRecord {
            obs_index: idx,
            neuron_uuid: uuid.to_string(),
            value,
            activation,
            errors: vec![error],
        }
    }

    #[test]
    fn test_is_skip_squash() {
        assert!(is_skip_squash("IDENTITY"));
        assert!(!is_skip_squash("TANH"));
        assert!(!is_skip_squash("RELU"));
    }

    #[test]
    fn test_high_error_triggers_detection() {
        // TANH neuron where target is linear — error = pre_act - tanh(pre_act)
        let records: Vec<DiscoverRecord> = (0..50)
            .map(|i| {
                let pre_act = (i as f32 - 25.0) / 8.0;
                let activation = pre_act.tanh();
                let error = pre_act - activation;
                make_record("h1", i, Some(pre_act), activation, error)
            })
            .collect();

        let neurons = vec![("h1".to_string(), "TANH".to_string(), 0.0)];
        let neuron_records = vec![("h1".to_string(), records)];

        let result = detect_high_error_squash_candidates(&neurons, &neuron_records);

        assert!(
            !result.is_empty(),
            "High error TANH neuron should trigger exploration"
        );
    }

    #[test]
    fn test_low_error_skips() {
        let records: Vec<DiscoverRecord> = (0..50)
            .map(|i| {
                let pre_act = (i as f32 - 25.0) / 50.0;
                make_record("h2", i, Some(pre_act), pre_act.tanh(), 0.001)
            })
            .collect();

        let neurons = vec![("h2".to_string(), "TANH".to_string(), 0.0)];
        let neuron_records = vec![("h2".to_string(), records)];

        let result = detect_high_error_squash_candidates(&neurons, &neuron_records);

        assert!(
            result.is_empty(),
            "Low error neuron should not trigger exploration"
        );
    }
}
