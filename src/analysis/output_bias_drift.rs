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

use std::collections::HashMap;

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum samples required for reliable bias drift detection.
const MIN_SAMPLES_FOR_BIAS_DRIFT: usize = 20;

/// Minimum fraction of errors that must share the same sign to consider
/// the output biased. 0.7 means 70%+ must be positive or negative.
const MIN_MAJORITY_SIGN_FRACTION: f32 = 0.7;

/// Minimum absolute mean error to distinguish from noise.
const MIN_MEAN_ERROR_MAGNITUDE: f32 = 0.01;

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
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DiscoverRecord;
    use crate::NeuronJson;

    fn make_output_neuron(uuid: &str, bias: f32) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias,
        }
    }

    fn make_creature(neurons: Vec<NeuronJson>) -> CreatureJson {
        CreatureJson {
            neurons,
            synapses: vec![],
            input: 0,
            output: 1,
        }
    }

    fn make_record(uuid: &str, obs_index: u32, error: f32) -> DiscoverRecord {
        DiscoverRecord::new(obs_index, uuid.to_string(), None, 0.0, vec![error])
    }

    // ── detect_output_bias_drift ───────────────────────────────────────

    #[test]
    fn consistent_positive_errors_detected() {
        let creature = make_creature(vec![make_output_neuron("out-1", 0.0)]);
        // 80% positive errors with meaningful magnitude
        let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
            "out-1".to_string(),
            (0..100)
                .map(|i| {
                    let err = if i < 80 { 0.5 } else { -0.2 };
                    make_record("out-1", i, err)
                })
                .collect(),
        )];

        let result = detect_output_bias_drift(&creature, &records);
        assert_eq!(result.len(), 1, "Should detect bias drift");
        assert!(
            result[0].recommended_bias_delta < 0.0,
            "Should recommend negative delta for positive error bias"
        );
    }

    #[test]
    fn balanced_errors_not_flagged() {
        let creature = make_creature(vec![make_output_neuron("out-1", 0.0)]);
        // 50/50 positive/negative errors
        let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
            "out-1".to_string(),
            (0..100)
                .map(|i| {
                    let err = if i % 2 == 0 { 0.5 } else { -0.5 };
                    make_record("out-1", i, err)
                })
                .collect(),
        )];

        let result = detect_output_bias_drift(&creature, &records);
        assert!(
            result.is_empty(),
            "Balanced errors should not trigger bias drift"
        );
    }

    #[test]
    fn hidden_neurons_excluded() {
        let creature = CreatureJson {
            neurons: vec![NeuronJson {
                uuid: "h-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            }],
            synapses: vec![],
            input: 0,
            output: 0,
        };

        let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
            "h-1".to_string(),
            (0..100).map(|i| make_record("h-1", i, 0.5)).collect(),
        )];

        let result = detect_output_bias_drift(&creature, &records);
        assert!(result.is_empty(), "Hidden neurons should be excluded");
    }

    #[test]
    fn noise_level_errors_not_flagged() {
        let creature = make_creature(vec![make_output_neuron("out-1", 0.0)]);
        // All positive but very small magnitude (below MIN_MEAN_ERROR_MAGNITUDE)
        let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
            "out-1".to_string(),
            (0..100).map(|i| make_record("out-1", i, 0.001)).collect(),
        )];

        let result = detect_output_bias_drift(&creature, &records);
        assert!(
            result.is_empty(),
            "Tiny errors should not trigger bias drift"
        );
    }

    #[test]
    fn insufficient_samples_not_detected() {
        let creature = make_creature(vec![make_output_neuron("out-1", 0.0)]);
        let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
            "out-1".to_string(),
            (0..10).map(|i| make_record("out-1", i, 0.5)).collect(),
        )];

        let result = detect_output_bias_drift(&creature, &records);
        assert!(result.is_empty(), "Should require minimum sample count");
    }

    // ── output_bias_drift_to_coordinated_candidates ────────────────────

    #[test]
    fn conversion_produces_set_bias_with_correct_value() {
        let candidate = OutputBiasDriftCandidate {
            neuron_uuid: "out-1".to_string(),
            current_bias: 0.5,
            mean_error: 0.3,
            positive_error_fraction: 0.8,
            sample_count: 100,
            recommended_bias_delta: -0.3,
            estimated_improvement: 0.006,
        };

        let coordinated = output_bias_drift_to_coordinated_candidates(&[candidate]);
        assert_eq!(coordinated.len(), 1);
        assert!(matches!(
            &coordinated[0].operations[0],
            CoordinatedStructuralOpJson::SetBias { neuron_uuid, bias }
                if neuron_uuid == "out-1" && (*bias - 0.2).abs() < 0.01
        ));
    }
}
