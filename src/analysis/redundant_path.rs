//! Redundant path pruning with renormalisation (Issue #164).
//!
//! Detects that two existing subnetworks (paths) feeding the same output compute
//! effectively the same thing, so one can be pruned and the other's weight scaled
//! to compensate.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Redundant Path Pruning" for full documentation.
//!
//! ## Detection signals
//!
//! 1. **Highly correlated activations** – Two sources feeding the same target have
//!    Pearson correlation ≥ `MIN_ACTIVATION_CORRELATION`.
//! 2. **Anti-correlated error gradients** – The product of error sensitivities is
//!    negative (both push the error in the same direction, so removing one and
//!    scaling the other preserves the correction).
//! 3. **Shared downstream synapses** – Both sources feed into the same target.
//!
//! ## Discovery type
//!
//! `COORDINATED_PRUNE_AND_REWEIGHT` – a coordinated structural candidate that
//! removes one synapse and adjusts the weight of the surviving synapse.

use crate::CoordinatedStructuralCandidateJson;
use crate::CoordinatedStructuralOpJson;
use crate::analysis::samples::HelpfulSample;

/// Minimum Pearson correlation between activation patterns to consider two paths redundant.
const MIN_ACTIVATION_CORRELATION: f64 = 0.85;

/// Minimum samples required for reliable redundant path detection.
const MIN_SAMPLES_FOR_REDUNDANT_PATH: usize = 30;

/// Minimum combined absolute weight for the pair to be worth pruning.
/// Very small weights are not worth the structural change.
const MIN_COMBINED_WEIGHT: f32 = 0.01;

/// Result of detecting a redundant path pair.
#[derive(Debug, Clone)]
pub struct RedundantPathCandidate {
    /// Source neuron UUID of the path to keep (the survivor).
    pub keep_source_uuid: String,
    /// Source neuron UUID of the path to prune.
    pub prune_source_uuid: String,
    /// Target neuron UUID (shared downstream).
    pub target_uuid: String,
    /// Existing weight of the survivor synapse.
    pub keep_weight: f32,
    /// Existing weight of the synapse to prune.
    pub prune_weight: f32,
    /// New weight for the survivor after renormalisation.
    pub renormalised_weight: f32,
    /// Pearson correlation between the two activation patterns.
    pub activation_correlation: f64,
    /// Estimated improvement from pruning (fraction of error reduced).
    pub estimated_improvement: f32,
    /// Human-readable reason for the candidate.
    pub reason: String,
}

/// A source feeding a target via an existing synapse, with recorded activations.
#[derive(Debug, Clone)]
pub struct ExistingPathContribution {
    /// Source neuron UUID.
    pub source_uuid: String,
    /// The existing synapse weight from source to target.
    pub existing_weight: f32,
    /// Recorded activation samples for this source, aligned by obs_index with the target.
    pub samples: Vec<HelpfulSample>,
}

/// Detect redundant paths feeding the same target neuron (Issue #164).
///
/// Two existing synapses `(A → T, weight_a)` and `(B → T, weight_b)` are redundant when:
/// - Their activation patterns are highly correlated (r ≥ 0.85)
/// - Both have non-trivial weight
///
/// When detected, we propose pruning the weaker synapse and scaling the survivor's weight
/// so the combined contribution is preserved.
///
/// # Arguments
/// * `target_uuid` – The target neuron UUID.
/// * `paths` – Existing synapse contributions feeding this target.
/// * `target_impact` – Impact factor for the target neuron (for discounting).
///
/// # Returns
/// A list of redundant path candidates, sorted by estimated improvement (descending).
pub fn detect_redundant_paths(
    target_uuid: &str,
    paths: &[ExistingPathContribution],
    target_impact: f32,
) -> Vec<RedundantPathCandidate> {
    if paths.len() < 2 {
        return Vec::new();
    }

    // Filter to paths with enough samples
    let valid_paths: Vec<&ExistingPathContribution> = paths
        .iter()
        .filter(|p| p.samples.len() >= MIN_SAMPLES_FOR_REDUNDANT_PATH)
        .collect();

    if valid_paths.len() < 2 {
        return Vec::new();
    }

    let mut candidates = Vec::new();

    // Check pairs for redundancy
    for i in 0..valid_paths.len() {
        for j in (i + 1)..valid_paths.len() {
            let path_a = valid_paths[i];
            let path_b = valid_paths[j];

            if let Some(candidate) =
                evaluate_pair_for_redundancy(target_uuid, path_a, path_b, target_impact)
            {
                candidates.push(candidate);
            }
        }
    }

    // Sort by estimated improvement (descending)
    candidates.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Evaluate a pair of existing paths for redundancy.
///
/// Returns `Some(RedundantPathCandidate)` if the pair is redundant and pruning one
/// while scaling the other is expected to improve (or at least preserve) fitness.
fn evaluate_pair_for_redundancy(
    target_uuid: &str,
    path_a: &ExistingPathContribution,
    path_b: &ExistingPathContribution,
    target_impact: f32,
) -> Option<RedundantPathCandidate> {
    let n_samples = path_a.samples.len().min(path_b.samples.len());
    if n_samples < MIN_SAMPLES_FOR_REDUNDANT_PATH {
        return None;
    }

    // Check combined weight is non-trivial
    let combined_abs_weight = path_a.existing_weight.abs() + path_b.existing_weight.abs();
    if combined_abs_weight < MIN_COMBINED_WEIGHT {
        return None;
    }

    // Compute activation correlation
    let correlation = compute_activation_correlation(&path_a.samples, &path_b.samples, n_samples);

    if correlation < MIN_ACTIVATION_CORRELATION {
        return None;
    }

    // Check for anti-correlated error gradients (both push error same direction)
    let error_gradient_product =
        compute_error_gradient_product(&path_a.samples, &path_b.samples, n_samples);

    // Determine which path to prune (the weaker one)
    let (keep, prune) = if path_a.existing_weight.abs() >= path_b.existing_weight.abs() {
        (path_a, path_b)
    } else {
        (path_b, path_a)
    };

    // Compute renormalised weight for the survivor
    // When activations are highly correlated, the combined effect of both paths is
    // approximately: keep_weight * activation + prune_weight * activation
    //             = (keep_weight + prune_weight) * activation
    // So the survivor should take on the sum of both weights.
    let renormalised_weight = keep.existing_weight + prune.existing_weight;

    // Estimate improvement from pruning
    // The benefit comes from reducing network complexity with minimal fitness cost.
    // With high correlation, removing the redundant path and compensating the weight
    // should preserve (or slightly improve) error. The improvement is proportional to
    // the structural simplification benefit.
    let estimated_improvement =
        estimate_pruning_improvement(keep, prune, renormalised_weight, correlation, target_impact);

    // Only emit candidate if we expect improvement
    if estimated_improvement <= 0.0 {
        return None;
    }

    let reason = if error_gradient_product > 0.0 {
        format!(
            "Redundant paths: activation correlation {:.2}, anti-correlated error gradients, \
             prune weaker path ({}) and scale survivor ({}) weight {:.4} → {:.4}",
            correlation,
            prune.source_uuid,
            keep.source_uuid,
            keep.existing_weight,
            renormalised_weight
        )
    } else {
        format!(
            "Redundant paths: activation correlation {:.2}, \
             prune weaker path ({}) and scale survivor ({}) weight {:.4} → {:.4}",
            correlation,
            prune.source_uuid,
            keep.source_uuid,
            keep.existing_weight,
            renormalised_weight
        )
    };

    Some(RedundantPathCandidate {
        keep_source_uuid: keep.source_uuid.clone(),
        prune_source_uuid: prune.source_uuid.clone(),
        target_uuid: target_uuid.to_string(),
        keep_weight: keep.existing_weight,
        prune_weight: prune.existing_weight,
        renormalised_weight,
        activation_correlation: correlation,
        estimated_improvement,
        reason,
    })
}

/// Compute Pearson correlation coefficient between two activation patterns.
///
/// Returns a value in [0, 1] (absolute correlation) where:
/// - 1 means perfectly correlated (identical or perfectly anti-correlated patterns)
/// - 0 means no correlation
fn compute_activation_correlation(
    samples_a: &[HelpfulSample],
    samples_b: &[HelpfulSample],
    n_samples: usize,
) -> f64 {
    if n_samples < 3 {
        return 0.0;
    }

    // Compute means
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;

    for i in 0..n_samples {
        sum_a += samples_a[i].activation as f64;
        sum_b += samples_b[i].activation as f64;
    }

    let mean_a = sum_a / n_samples as f64;
    let mean_b = sum_b / n_samples as f64;

    // Compute covariance and variances
    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;

    for i in 0..n_samples {
        let da = samples_a[i].activation as f64 - mean_a;
        let db = samples_b[i].activation as f64 - mean_b;

        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denominator = (var_a * var_b).sqrt();
    if denominator < 1e-10 {
        return 0.0;
    }

    // Return absolute correlation – both positive and negative correlation
    // indicate redundancy (anti-correlated paths cancel each other).
    (cov / denominator).abs()
}

/// Compute the average product of error gradients for two paths.
///
/// A positive product means both paths push the error in the same direction
/// (anti-correlated error gradients relative to each other), strengthening
/// the case for redundancy.
fn compute_error_gradient_product(
    samples_a: &[HelpfulSample],
    samples_b: &[HelpfulSample],
    n_samples: usize,
) -> f64 {
    if n_samples == 0 {
        return 0.0;
    }

    let mut product_sum = 0.0f64;

    for i in 0..n_samples {
        let error_a = samples_a[i].avg_error as f64;
        let error_b = samples_b[i].avg_error as f64;
        product_sum += error_a * error_b;
    }

    product_sum / n_samples as f64
}

/// Estimate improvement from pruning a redundant path and renormalising.
///
/// The improvement comes from:
/// 1. Reducing structural complexity (one fewer synapse to maintain)
/// 2. The fact that highly correlated paths are carrying duplicate information
///
/// We estimate by computing how well the renormalised single path approximates the
/// original two-path contribution, comparing residual error.
fn estimate_pruning_improvement(
    keep: &ExistingPathContribution,
    prune: &ExistingPathContribution,
    renormalised_weight: f32,
    correlation: f64,
    target_impact: f32,
) -> f32 {
    let n_samples = keep.samples.len().min(prune.samples.len());
    if n_samples == 0 {
        return 0.0;
    }

    // Compute the error of the two-path system vs the renormalised single-path system.
    //
    // Original contribution:  keep_weight * act_keep + prune_weight * act_prune
    // Renormalised:           renormalised_weight * act_keep
    //
    // Residual difference per sample:
    //   delta = (keep_weight * act_keep + prune_weight * act_prune) - (renormalised_weight * act_keep)
    //         = prune_weight * act_prune - prune_weight * act_keep  (since renorm = keep + prune)
    //         = prune_weight * (act_prune - act_keep)
    //
    // With high correlation, (act_prune ≈ k * act_keep), so delta ≈ 0.
    // The smaller the delta, the better the approximation.

    let mut original_error_sq_sum = 0.0f64;
    let mut renormalised_error_sq_sum = 0.0f64;

    for i in 0..n_samples {
        let act_keep = keep.samples[i].activation as f64;
        let act_prune = prune.samples[i].activation as f64;
        let target_error = keep.samples[i].avg_error as f64;

        // Original two-path contribution
        let original_contribution =
            keep.existing_weight as f64 * act_keep + prune.existing_weight as f64 * act_prune;

        // Renormalised single-path contribution
        let renormalised_contribution = renormalised_weight as f64 * act_keep;

        // Error after applying original two-path correction
        let error_with_original = target_error - original_contribution;
        // Error after applying renormalised single-path correction
        let error_with_renormalised = target_error - renormalised_contribution;

        original_error_sq_sum += error_with_original * error_with_original;
        renormalised_error_sq_sum += error_with_renormalised * error_with_renormalised;
    }

    // If original error is near zero, there's nothing to improve
    if original_error_sq_sum < 1e-10 && renormalised_error_sq_sum < 1e-10 {
        // Both are effectively zero – the renormalised version is equally good.
        // We still benefit from pruning (structural simplification).
        // Use correlation as a proxy for how confident we are.
        return (correlation as f32 * 0.01) * target_impact;
    }

    // Improvement = fraction of original error that's preserved or improved
    // If renormalised_error <= original_error, the simplification is free or beneficial.
    // We also add a small structural simplification bonus for high-correlation pairs.
    let base_error = original_error_sq_sum.max(renormalised_error_sq_sum);
    if base_error < 1e-10 {
        return 0.0;
    }

    let error_ratio = renormalised_error_sq_sum / base_error;
    // If renormalised is worse, error_ratio > 1 and improvement is negative
    // If renormalised is better or equal, improvement is positive
    let error_improvement = (1.0 - error_ratio) as f32;

    // Add structural simplification bonus proportional to correlation
    // Higher correlation → more confident that pruning is safe
    let structural_bonus = (correlation as f32 - MIN_ACTIVATION_CORRELATION as f32).max(0.0) * 0.01;

    (error_improvement + structural_bonus) * target_impact
}

/// Convert redundant path candidates to coordinated structural candidates.
///
/// Each candidate becomes a `COORDINATED_PRUNE_AND_REWEIGHT` coordinated structural candidate
/// with two operations:
/// 1. `RemoveSynapse` – remove the redundant (pruned) synapse
/// 2. `SetWeight` – adjust the survivor's weight to compensate
pub fn redundant_paths_to_coordinated_candidates(
    candidates: &[RedundantPathCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    candidates
        .iter()
        .map(|c| CoordinatedStructuralCandidateJson {
            operations: vec![
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: c.prune_source_uuid.clone(),
                    to_neuron_uuid: c.target_uuid.clone(),
                },
                CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: c.keep_source_uuid.clone(),
                    to_neuron_uuid: c.target_uuid.clone(),
                    weight: c.renormalised_weight,
                },
            ],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!("Redundant path pruning (Issue #164): {}", c.reason)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(activation: f32, avg_error: f32) -> HelpfulSample {
        HelpfulSample {
            activation,
            avg_error,
            target_value: None,
            target_activation: None,
        }
    }

    #[test]
    fn test_identical_activations_detected_as_redundant() {
        // Two sources with identical activation patterns → should be detected as redundant
        let samples: Vec<HelpfulSample> =
            (0..50).map(|i| make_sample(i as f32 / 50.0, 0.1)).collect();

        let paths = vec![
            ExistingPathContribution {
                source_uuid: "source-a".to_string(),
                existing_weight: 0.5,
                samples: samples.clone(),
            },
            ExistingPathContribution {
                source_uuid: "source-b".to_string(),
                existing_weight: 0.3,
                samples,
            },
        ];

        let candidates = detect_redundant_paths("target-0", &paths, 1.0);
        assert!(
            !candidates.is_empty(),
            "Identical activation patterns should be detected as redundant"
        );

        let c = &candidates[0];
        assert_eq!(c.keep_source_uuid, "source-a"); // Stronger weight kept
        assert_eq!(c.prune_source_uuid, "source-b");
        assert!(c.activation_correlation > 0.99);
        // Renormalised weight should be sum of both
        assert!(
            (c.renormalised_weight - 0.8).abs() < 0.01,
            "Renormalised weight should be ~0.8 (0.5 + 0.3), got {}",
            c.renormalised_weight
        );
    }

    #[test]
    fn test_uncorrelated_activations_not_detected() {
        // Two sources with uncorrelated activation patterns → should NOT be detected
        let samples_a: Vec<HelpfulSample> =
            (0..50).map(|i| make_sample(i as f32 / 50.0, 0.1)).collect();
        let samples_b: Vec<HelpfulSample> = (0..50)
            .map(|i| make_sample(((i * 7 + 13) % 50) as f32 / 50.0, 0.1))
            .collect();

        let paths = vec![
            ExistingPathContribution {
                source_uuid: "source-a".to_string(),
                existing_weight: 0.5,
                samples: samples_a,
            },
            ExistingPathContribution {
                source_uuid: "source-b".to_string(),
                existing_weight: 0.3,
                samples: samples_b,
            },
        ];

        let candidates = detect_redundant_paths("target-0", &paths, 1.0);
        assert!(
            candidates.is_empty(),
            "Uncorrelated activation patterns should not be detected as redundant"
        );
    }

    #[test]
    fn test_anti_correlated_activations_detected() {
        // Two sources with perfectly anti-correlated activations (one goes up, other goes down)
        // should also be detected (correlation is checked as absolute value)
        let samples_a: Vec<HelpfulSample> =
            (0..50).map(|i| make_sample(i as f32 / 50.0, 0.1)).collect();
        let samples_b: Vec<HelpfulSample> = (0..50)
            .map(|i| make_sample(1.0 - i as f32 / 50.0, 0.1))
            .collect();

        let paths = vec![
            ExistingPathContribution {
                source_uuid: "source-a".to_string(),
                existing_weight: 0.5,
                samples: samples_a,
            },
            ExistingPathContribution {
                source_uuid: "source-b".to_string(),
                existing_weight: -0.3,
                samples: samples_b,
            },
        ];

        let candidates = detect_redundant_paths("target-0", &paths, 1.0);
        assert!(
            !candidates.is_empty(),
            "Anti-correlated activations should be detected as redundant (absolute correlation)"
        );
    }

    #[test]
    fn test_insufficient_samples_returns_empty() {
        // Too few samples → should return empty
        let samples: Vec<HelpfulSample> =
            (0..5).map(|i| make_sample(i as f32 / 5.0, 0.1)).collect();

        let paths = vec![
            ExistingPathContribution {
                source_uuid: "source-a".to_string(),
                existing_weight: 0.5,
                samples: samples.clone(),
            },
            ExistingPathContribution {
                source_uuid: "source-b".to_string(),
                existing_weight: 0.3,
                samples,
            },
        ];

        let candidates = detect_redundant_paths("target-0", &paths, 1.0);
        assert!(
            candidates.is_empty(),
            "Insufficient samples should return empty"
        );
    }

    #[test]
    fn test_single_path_returns_empty() {
        let samples: Vec<HelpfulSample> =
            (0..50).map(|i| make_sample(i as f32 / 50.0, 0.1)).collect();

        let paths = vec![ExistingPathContribution {
            source_uuid: "source-a".to_string(),
            existing_weight: 0.5,
            samples,
        }];

        let candidates = detect_redundant_paths("target-0", &paths, 1.0);
        assert!(candidates.is_empty(), "Single path should return empty");
    }

    #[test]
    fn test_very_small_weights_skipped() {
        // Both weights are negligible → not worth pruning
        let samples: Vec<HelpfulSample> =
            (0..50).map(|i| make_sample(i as f32 / 50.0, 0.1)).collect();

        let paths = vec![
            ExistingPathContribution {
                source_uuid: "source-a".to_string(),
                existing_weight: 0.001,
                samples: samples.clone(),
            },
            ExistingPathContribution {
                source_uuid: "source-b".to_string(),
                existing_weight: 0.001,
                samples,
            },
        ];

        let candidates = detect_redundant_paths("target-0", &paths, 1.0);
        assert!(
            candidates.is_empty(),
            "Very small weights should be skipped"
        );
    }

    #[test]
    fn test_redundant_paths_to_coordinated_candidates() {
        let candidates = vec![RedundantPathCandidate {
            keep_source_uuid: "source-a".to_string(),
            prune_source_uuid: "source-b".to_string(),
            target_uuid: "output-0".to_string(),
            keep_weight: 0.5,
            prune_weight: 0.3,
            renormalised_weight: 0.8,
            activation_correlation: 0.95,
            estimated_improvement: 0.05,
            reason: "Test redundant paths".to_string(),
        }];

        let coordinated = redundant_paths_to_coordinated_candidates(&candidates);
        assert_eq!(coordinated.len(), 1);
        assert_eq!(coordinated[0].operations.len(), 2);

        // First op: RemoveSynapse for the pruned path
        let remove_op = &coordinated[0].operations[0];
        match remove_op {
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
            } => {
                assert_eq!(from_neuron_uuid, "source-b");
                assert_eq!(to_neuron_uuid, "output-0");
            }
            _ => panic!("Expected RemoveSynapse operation"),
        }

        // Second op: SetWeight for the survivor
        let set_weight_op = &coordinated[0].operations[1];
        match set_weight_op {
            CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid,
                to_neuron_uuid,
                weight,
            } => {
                assert_eq!(from_neuron_uuid, "source-a");
                assert_eq!(to_neuron_uuid, "output-0");
                assert!((weight - 0.8).abs() < 0.01);
            }
            _ => panic!("Expected SetWeight operation"),
        }

        // Check comment mentions Issue #164
        let comment = coordinated[0].comment.as_ref().unwrap();
        assert!(
            comment.contains("164"),
            "Comment should reference Issue #164: {comment}"
        );
    }

    #[test]
    fn test_compute_activation_correlation_identical() {
        let samples: Vec<HelpfulSample> =
            (0..50).map(|i| make_sample(i as f32 / 50.0, 0.0)).collect();
        let corr = compute_activation_correlation(&samples, &samples, 50);
        assert!(
            (corr - 1.0).abs() < 0.01,
            "Identical samples should have correlation ~1.0, got {corr}"
        );
    }

    #[test]
    fn test_compute_activation_correlation_anti_correlated() {
        let samples_a: Vec<HelpfulSample> =
            (0..50).map(|i| make_sample(i as f32 / 50.0, 0.0)).collect();
        let samples_b: Vec<HelpfulSample> = (0..50)
            .map(|i| make_sample(1.0 - i as f32 / 50.0, 0.0))
            .collect();
        let corr = compute_activation_correlation(&samples_a, &samples_b, 50);
        // Absolute correlation should be ~1.0
        assert!(
            (corr - 1.0).abs() < 0.01,
            "Anti-correlated samples should have absolute correlation ~1.0, got {corr}"
        );
    }

    #[test]
    fn test_stronger_weight_is_kept() {
        // source-a has weight 0.3, source-b has weight 0.5
        // source-b should be kept (stronger)
        let samples: Vec<HelpfulSample> =
            (0..50).map(|i| make_sample(i as f32 / 50.0, 0.1)).collect();

        let paths = vec![
            ExistingPathContribution {
                source_uuid: "source-a".to_string(),
                existing_weight: 0.3,
                samples: samples.clone(),
            },
            ExistingPathContribution {
                source_uuid: "source-b".to_string(),
                existing_weight: 0.5,
                samples,
            },
        ];

        let candidates = detect_redundant_paths("target-0", &paths, 1.0);
        assert!(!candidates.is_empty());

        let c = &candidates[0];
        assert_eq!(
            c.keep_source_uuid, "source-b",
            "Stronger weight should be kept"
        );
        assert_eq!(
            c.prune_source_uuid, "source-a",
            "Weaker weight should be pruned"
        );
    }

    #[test]
    fn test_target_impact_affects_improvement() {
        let samples: Vec<HelpfulSample> =
            (0..50).map(|i| make_sample(i as f32 / 50.0, 0.1)).collect();

        let paths = vec![
            ExistingPathContribution {
                source_uuid: "source-a".to_string(),
                existing_weight: 0.5,
                samples: samples.clone(),
            },
            ExistingPathContribution {
                source_uuid: "source-b".to_string(),
                existing_weight: 0.3,
                samples,
            },
        ];

        let candidates_high = detect_redundant_paths("target-0", &paths, 1.0);
        let candidates_low = detect_redundant_paths("target-0", &paths, 0.1);

        // Both should detect the redundancy
        assert!(!candidates_high.is_empty());
        assert!(!candidates_low.is_empty());

        // Higher impact should give higher improvement
        assert!(
            candidates_high[0].estimated_improvement >= candidates_low[0].estimated_improvement,
            "Higher target impact should give higher improvement"
        );
    }
}
