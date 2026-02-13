//! Error stagnation plateau detection module (Issue #545 / #547).
//!
//! Detects when output neuron error distributions show stagnation patterns —
//! consistently high error with low variance — indicating the network is stuck
//! in a local minimum and needs structural changes (not just weight/bias tuning)
//! to escape.
//!
//! ## Detection Criteria
//!
//! A plateau is identified when:
//! 1. Mean absolute error exceeds `MIN_PLATEAU_ERROR` (not already converged)
//! 2. Error coefficient of variation (std_dev / mean) is below
//!    `MAX_COEFFICIENT_OF_VARIATION` (tightly clustered = flat error surface)
//! 3. Sufficient sample count for statistical confidence
//!
//! ## Recommended Actions
//!
//! For plateau neurons, the module recommends `changeSquash` candidates to
//! fundamentally alter the error landscape and potentially escape the local minimum.

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Minimum mean absolute error to consider a neuron as having a plateau problem.
/// Below this threshold, the neuron is considered sufficiently converged.
const MIN_PLATEAU_ERROR: f32 = 0.05;

/// Maximum coefficient of variation (std_dev / mean) for error to be considered
/// a plateau. Low CV means errors are tightly clustered around the mean.
const MAX_COEFFICIENT_OF_VARIATION: f32 = 0.3;

/// A detected error plateau candidate.
#[derive(Debug, Clone)]
pub struct ErrorPlateauCandidate {
    /// UUID of the plateau neuron.
    pub neuron_uuid: String,
    /// Current activation function.
    pub current_squash: String,
    /// Recommended replacement activation function.
    pub recommended_squash: String,
    /// Mean absolute error across observations.
    pub mean_error: f32,
    /// Standard deviation of errors.
    pub error_std_dev: f32,
    /// Coefficient of variation (std_dev / mean).
    pub error_coefficient_of_variation: f32,
    /// Confidence in the plateau detection (0.0 to 1.0).
    pub confidence: f32,
    /// Estimated improvement from structural change.
    pub estimated_improvement: f32,
    /// Human-readable explanation.
    pub reason: String,
}

/// Select a replacement squash function to escape the plateau.
///
/// The goal is to fundamentally change the error landscape by switching to a
/// qualitatively different activation function.
fn recommend_plateau_escape(current_squash: &str, records: &[DiscoverRecord]) -> String {
    let upper = current_squash.to_ascii_uppercase();

    // Check if activations are predominantly in a symmetric range
    let has_negative = records.iter().any(|r| r.activation < -0.01);
    let min_activation = records
        .iter()
        .map(|r| r.activation)
        .fold(f32::INFINITY, f32::min);
    let max_activation = records
        .iter()
        .map(|r| r.activation)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = max_activation - min_activation;

    match upper.as_str() {
        // Piecewise-linear bounded → smooth bounded
        "HARD_TANH" | "CLIPPED" => "TANH".to_string(),
        // Smooth bounded → different smooth bounded or unbounded
        "TANH" | "BIPOLAR_SIGMOID" => {
            if range < 0.5 {
                // Very restricted range — try unbounded
                "IDENTITY".to_string()
            } else {
                "SOFTSIGN".to_string()
            }
        }
        // Logistic → symmetric if needed
        "LOGISTIC" => {
            if has_negative {
                "TANH".to_string()
            } else {
                "SOFTSIGN".to_string()
            }
        }
        // Threshold/binary → smooth
        "BIPOLAR" | "STEP" => "TANH".to_string(),
        // RELU family → smooth bounded
        "RELU" | "RELU6" | "LEAKYRELU" => "TANH".to_string(),
        // Unbounded → bounded
        "IDENTITY" => "TANH".to_string(),
        // Default: try TANH as a general-purpose smooth bounded function
        _ => "TANH".to_string(),
    }
}

/// Detect error stagnation plateaus in output neurons.
///
/// # Arguments
/// * `output_neurons` - Slice of `(uuid, squash, bias)` tuples for output neurons.
/// * `neuron_records` - Slice of `(uuid, records)` pairs with observation data.
///
/// # Returns
/// Vector of detected plateau candidates, sorted by estimated improvement (best first).
pub fn detect_error_plateaus(
    output_neurons: &[(String, String, f32)],
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<ErrorPlateauCandidate> {
    let mut candidates = Vec::new();

    for (uuid, squash, _bias) in output_neurons {
        let Some((_id, records)) = neuron_records.iter().find(|(u, _)| u == uuid) else {
            continue;
        };

        if records.len() < MIN_SAMPLES {
            continue;
        }

        // Compute error statistics
        let errors: Vec<f32> = records
            .iter()
            .map(|r| r.errors.first().copied().unwrap_or(0.0).abs())
            .collect();

        let n = errors.len() as f32;
        let mean_error: f32 = errors.iter().sum::<f32>() / n;

        // Skip if error is already low (converged)
        if mean_error < MIN_PLATEAU_ERROR {
            continue;
        }

        // Compute standard deviation
        let variance: f32 = errors.iter().map(|e| (e - mean_error).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        // Coefficient of variation: std_dev / mean
        let cv = if mean_error > 1e-6 {
            std_dev / mean_error
        } else {
            f32::INFINITY
        };

        // Plateau = high error + low variance (tightly clustered around a non-zero mean)
        if cv > MAX_COEFFICIENT_OF_VARIATION {
            continue;
        }

        let recommended = recommend_plateau_escape(squash, records);

        // Don't recommend same squash
        if recommended.eq_ignore_ascii_case(squash) {
            continue;
        }

        // Confidence based on sample size and how tight the plateau is
        let sample_confidence = (records.len() as f32 / 100.0).min(1.0);
        let plateau_tightness = (1.0 - cv / MAX_COEFFICIENT_OF_VARIATION).max(0.0);
        let confidence =
            (sample_confidence * 0.4 + plateau_tightness * 0.4 + mean_error.min(1.0) * 0.2)
                .clamp(0.0, 1.0);

        // Estimated improvement: proportional to plateau error and tightness
        let estimated_improvement = mean_error * plateau_tightness * 0.3;

        candidates.push(ErrorPlateauCandidate {
            neuron_uuid: uuid.clone(),
            current_squash: squash.clone(),
            recommended_squash: recommended.clone(),
            mean_error,
            error_std_dev: std_dev,
            error_coefficient_of_variation: cv,
            confidence,
            estimated_improvement,
            reason: format!(
                "Output neuron {uuid} shows error plateau: mean error {mean_error:.3} with CV {cv:.3} (tight clustering around non-zero error indicates local minimum). Recommending {recommended} to escape plateau. (Issue #545)",
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

/// Convert error plateau candidates to coordinated structural candidates.
pub fn error_plateaus_to_coordinated_candidates(
    candidates: &[ErrorPlateauCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    candidates
        .iter()
        .map(|c| CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: c.neuron_uuid.clone(),
                squash: c.recommended_squash.clone(),
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(c.reason.clone()),
        })
        .collect()
}
