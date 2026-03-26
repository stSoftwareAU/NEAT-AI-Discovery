//! Synergistic discovery via residual analysis (Issue #189).
//!
//! This module implements individual operation pre-screening before pairing:
//! 1. Find the best single-source candidate for the target
//! 2. Compute residual error after applying that candidate
//! 3. Search for a second source that reduces the residual
//! 4. Return as synergistic candidate if combined > individual
//!
//! This is O(2n) instead of O(n²) for pairwise analysis.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::CoordinatedStructuralCandidateJson;
use crate::CoordinatedStructuralOpJson;
use crate::analysis::samples::HelpfulSample;

// Issue #508: Individual operation pre-screen threshold
use crate::analysis::constants::MAX_INDIVIDUAL_HARM_FOR_PAIRING;

// Issue #897: Conservative weight scale for coordinated estimation
use crate::analysis::constants::COORDINATED_ESTIMATION_WEIGHT_SCALE;

use super::{SourceContribution, SynergisticCandidate};

/// Minimum residual reduction ratio for synergistic detection (Issue #189).
/// The second source must reduce residual error by at least this fraction.
const MIN_RESIDUAL_REDUCTION_RATIO: f32 = 0.1;

/// Minimum synergistic benefit ratio (Issue #189).
/// Combined improvement must exceed max(individual) * this factor.
const MIN_SYNERGISTIC_BENEFIT_RATIO: f32 = 1.1;

/// Minimum samples for reliable residual analysis.
const MIN_SAMPLES_FOR_RESIDUAL_ANALYSIS: usize = 30;

/// Detect synergistic candidates using residual analysis (Issue #189).
///
/// This implements Option 2 from the issue: Residual Analysis
/// 1. Find the best single-source candidate for the target
/// 2. Compute residual error after applying that candidate
/// 3. Search for a second source that reduces the residual
/// 4. Return as synergistic candidate if combined > individual
///
/// This is O(2n) instead of O(n²) for pairwise analysis.
///
/// # Arguments
/// * `target_uuid` - The target neuron UUID
/// * `contributions` - List of source contributions (samples and stats for each source)
/// * `target_impact` - Impact factor of the target neuron (for discounting)
///
/// # Returns
/// A list of synergistic candidates that should be returned as coordinated structural candidates.
pub fn detect_synergistic_candidates(
    target_uuid: &str,
    contributions: &[SourceContribution],
    target_impact: f32,
    target_squash: Option<&str>,
) -> Vec<SynergisticCandidate> {
    if contributions.len() < 2 {
        return Vec::new();
    }

    // Issue #897: Resolve target activation function for saturation-aware simulation
    let target_activation_fn = target_squash.and_then(crate::activations::target_simulation_fn);

    // Filter to sources with enough samples and non-harmful individual improvement (Issue #508)
    let valid_sources: Vec<&SourceContribution> = contributions
        .iter()
        .filter(|c| {
            c.samples.len() >= MIN_SAMPLES_FOR_RESIDUAL_ANALYSIS
                && c.individual_improvement >= MAX_INDIVIDUAL_HARM_FOR_PAIRING
        })
        .collect();

    if valid_sources.len() < 2 {
        return Vec::new();
    }

    // Step 1: Find the best single-source candidate
    let best_primary = valid_sources.iter().max_by(|a, b| {
        a.individual_improvement
            .total_cmp(&b.individual_improvement)
    });

    let Some(primary) = best_primary else {
        return Vec::new();
    };

    // Step 2: Compute residual errors after applying primary source
    let residuals = compute_residual_errors(
        &primary.samples,
        primary.optimal_weight,
        target_activation_fn,
    );

    // Step 3: Search for complementary sources that reduce the residual
    let mut candidates = Vec::new();

    for source in &valid_sources {
        // Skip the primary source itself
        if source.source_uuid == primary.source_uuid {
            continue;
        }

        // Evaluate how well this source reduces the residual error
        if let Some(candidate) = evaluate_residual_reduction(
            target_uuid,
            primary,
            source,
            &residuals,
            target_impact,
            target_activation_fn,
        ) {
            candidates.push(candidate);
        }
    }

    // Sort by combined improvement (descending)
    candidates.sort_by(|a, b| b.combined_improvement.total_cmp(&a.combined_improvement));

    candidates
}

/// Compute residual errors after applying a source with given weight (Issue #897).
///
/// When `target_activation_fn` is provided and samples have target data, uses
/// saturation-aware simulation in activation domain. Otherwise uses linear
/// approximation: `residual_error` = `original_error` - (weight × activation).
fn compute_residual_errors(
    samples: &[HelpfulSample],
    weight: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> Vec<ResidualSample> {
    let use_activation = target_activation_fn.is_some()
        && samples
            .iter()
            .all(|s| s.target_value.is_some() && s.target_activation.is_some());

    samples
        .iter()
        .filter_map(|s| {
            if use_activation {
                // Gracefully skip samples missing target data (Issue #940).
                let target_fn = target_activation_fn?;
                let target_value = s.target_value?;
                let target_activation = s.target_activation?;
                let desired_value = target_value + s.avg_error;
                let expected = target_fn(desired_value);

                // Baseline error in activation domain
                let original_error = expected - target_activation;

                // Residual after applying primary source through activation
                let new_input = target_value + weight * s.activation;
                let new_output = target_fn(new_input);
                let residual_error = expected - new_output;

                Some(ResidualSample {
                    original_error,
                    residual_error,
                    new_target_value: Some(new_input),
                    desired_value: Some(desired_value),
                })
            } else {
                let contribution = weight * s.activation;
                let residual_error = s.avg_error - contribution;
                Some(ResidualSample {
                    original_error: s.avg_error,
                    residual_error,
                    new_target_value: None,
                    desired_value: None,
                })
            }
        })
        .collect()
}

/// Internal structure for residual analysis (Issue #897: extended with target data).
#[derive(Debug, Clone)]
struct ResidualSample {
    original_error: f32,
    residual_error: f32,
    /// Target neuron pre-activation value after applying primary contribution (for activation-aware simulation).
    new_target_value: Option<f32>,
    /// Desired pre-activation value for computing expected output.
    desired_value: Option<f32>,
}

/// Compute Pearson correlation coefficient between two activation patterns.
///
/// Returns a value in [-1, 1] where:
/// - 1 means perfect positive correlation (identical patterns)
/// - 0 means no correlation
/// - -1 means perfect negative correlation (anti-correlated)
fn compute_activation_correlation(
    primary_samples: &[HelpfulSample],
    complement_samples: &[HelpfulSample],
    n_samples: usize,
) -> f64 {
    crate::analysis::detection::stats::pearson_correlation_samples(
        primary_samples,
        complement_samples,
        n_samples,
    )
}

/// Evaluate how well a complement source reduces the residual error (Issue #897).
///
/// When `target_activation_fn` is provided and residual samples carry target data,
/// uses saturation-aware simulation for the complement contribution.
fn evaluate_residual_reduction(
    target_uuid: &str,
    primary: &SourceContribution,
    complement: &SourceContribution,
    residuals: &[ResidualSample],
    target_impact: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> Option<SynergisticCandidate> {
    let min_samples = residuals.len().min(complement.samples.len());
    if min_samples < MIN_SAMPLES_FOR_RESIDUAL_ANALYSIS {
        return None;
    }

    // Check activation pattern correlation between primary and complement
    let activation_correlation =
        compute_activation_correlation(&primary.samples, &complement.samples, min_samples);

    // Skip if activations are too similar (correlation > 0.9)
    if activation_correlation > 0.9 {
        return None;
    }

    // Issue #897: Check if activation-aware simulation is available
    let use_activation = target_activation_fn.is_some()
        && residuals
            .iter()
            .take(min_samples)
            .all(|r| r.new_target_value.is_some() && r.desired_value.is_some());

    // Compute optimal weight for complement source against residual errors
    // Using least squares: w = Σ(activation × residual_error) / Σ(activation²)
    let mut sum_act_residual = 0.0f64;
    let mut sum_act_squared = 0.0f64;

    for (complement_sample, residual_sample) in complement
        .samples
        .iter()
        .zip(residuals.iter())
        .take(min_samples)
    {
        let activation = complement_sample.activation as f64;
        let residual = residual_sample.residual_error as f64;

        sum_act_residual += activation * residual;
        sum_act_squared += activation * activation;
    }

    if sum_act_squared < 1e-10 {
        return None;
    }

    let complement_weight = (sum_act_residual / sum_act_squared) as f32;

    // Compute error metrics
    let mut original_error_sum = 0.0f64;
    let mut residual_error_sum = 0.0f64;
    let mut combined_error_sum = 0.0f64;

    for (complement_sample, residual_sample) in complement
        .samples
        .iter()
        .zip(residuals.iter())
        .take(min_samples)
    {
        let original = residual_sample.original_error as f64;
        let residual = residual_sample.residual_error as f64;

        let combined_residual = if use_activation {
            // Issue #897: Saturation-aware simulation for complement contribution.
            // Gracefully skip samples missing target data (Issue #940).
            let (Some(target_fn), Some(new_target_value), Some(desired_value)) = (
                target_activation_fn,
                residual_sample.new_target_value,
                residual_sample.desired_value,
            ) else {
                continue;
            };
            let new_target_value = new_target_value as f64;
            let expected = target_fn(desired_value) as f64;

            // Add complement contribution through activation function
            let complement_contribution =
                complement_weight as f64 * complement_sample.activation as f64;
            let combined_input = new_target_value + complement_contribution;
            let combined_output = target_fn(combined_input as f32) as f64;
            expected - combined_output
        } else {
            // Linear approximation fallback
            let activation = complement_sample.activation as f64;
            residual - (complement_weight as f64 * activation)
        };

        original_error_sum += original * original;
        residual_error_sum += residual * residual;
        combined_error_sum += combined_residual * combined_residual;
    }

    if original_error_sum < 1e-10 {
        return None;
    }

    // Calculate improvements
    let primary_improvement = 1.0 - (residual_error_sum / original_error_sum);
    let combined_improvement = 1.0 - (combined_error_sum / original_error_sum);
    let residual_reduction = if residual_error_sum > 1e-10 {
        1.0 - (combined_error_sum / residual_error_sum)
    } else {
        0.0
    };

    // Check if this is a true synergistic candidate
    let best_individual = primary_improvement.max(complement.individual_improvement as f64);
    let synergy_ratio = if best_individual > 0.0 {
        combined_improvement / best_individual
    } else if combined_improvement > 0.0 {
        // True synergy: combined is positive when individuals are zero/negative
        f64::INFINITY
    } else {
        0.0
    };

    // Filter criteria:
    // 1. Residual reduction must be significant
    // 2. Combined improvement must be better than best individual
    // 3. Combined improvement must be positive
    let is_synergistic = residual_reduction >= MIN_RESIDUAL_REDUCTION_RATIO as f64
        && synergy_ratio >= MIN_SYNERGISTIC_BENEFIT_RATIO as f64
        && combined_improvement > 0.0;

    if !is_synergistic {
        return None;
    }

    // Generate reason based on the type of synergy
    let reason = if primary_improvement <= 0.0 && complement.individual_improvement <= 0.0 {
        format!(
            "XOR-like pattern: neither source improves alone, combined {:.1}% improvement",
            combined_improvement * 100.0
        )
    } else if residual_reduction > 0.5 {
        format!(
            "Strong residual reduction: complement reduces remaining error by {:.1}%",
            residual_reduction * 100.0
        )
    } else {
        format!(
            "Synergistic: combined {:.1}% > max individual {:.1}% (synergy ratio {:.2}x)",
            combined_improvement * 100.0,
            best_individual * 100.0,
            synergy_ratio
        )
    };

    // Issue #897: Apply conservative weight scale to reported gain.
    // During evaluation, coordinated candidate weights are scaled to 0.2× variants,
    // so the reported gain should reflect this more conservative configuration.
    let weight_scale = COORDINATED_ESTIMATION_WEIGHT_SCALE as f64;

    Some(SynergisticCandidate {
        primary_source_uuid: primary.source_uuid.clone(),
        complement_source_uuid: complement.source_uuid.clone(),
        target_uuid: target_uuid.to_string(),
        primary_weight: primary.optimal_weight,
        complement_weight,
        combined_improvement: (combined_improvement * target_impact as f64 * weight_scale) as f32,
        primary_improvement: (primary_improvement * target_impact as f64) as f32,
        complement_improvement: (complement.individual_improvement as f64 * target_impact as f64)
            as f32,
        residual_reduction: residual_reduction as f32,
        synergy_ratio: synergy_ratio as f32,
        reason,
    })
}

/// Convert synergistic candidates to coordinated structural candidates.
pub fn synergistic_to_coordinated_candidates(
    candidates: &[SynergisticCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    candidates
        .iter()
        .map(|c| CoordinatedStructuralCandidateJson {
            operations: vec![
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: c.primary_source_uuid.clone(),
                    to_neuron_uuid: c.target_uuid.clone(),
                    weight: c.primary_weight,
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: c.complement_source_uuid.clone(),
                    to_neuron_uuid: c.target_uuid.clone(),
                    weight: c.complement_weight,
                },
            ],
            expected_creature_score_gain: c.combined_improvement,
            comment: Some(format!(
                "Synergistic: {} (combined {:.2}%, synergy ratio {:.2}x)",
                c.reason,
                c.combined_improvement * 100.0,
                c.synergy_ratio
            )),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::recommendation::epistatic::candidate_generation::build_source_contribution;
    use crate::analysis::samples::HelpfulStats;

    #[test]
    fn test_prescreen_rejects_strongly_harmful_source() {
        // Source with individual_improvement below MAX_INDIVIDUAL_HARM_FOR_PAIRING
        // should be excluded from valid_sources, so no pairs are formed.
        let samples: Vec<HelpfulSample> = (0..64)
            .map(|i| HelpfulSample {
                activation: if i < 32 { 1.0 } else { 0.0 },
                avg_error: 0.3,
                target_value: None,
                target_activation: None,
            })
            .collect();
        let complement_samples: Vec<HelpfulSample> = (0..64)
            .map(|i| HelpfulSample {
                activation: if i >= 32 { 1.0 } else { 0.0 },
                avg_error: 0.3,
                target_value: None,
                target_activation: None,
            })
            .collect();

        let contributions = vec![
            build_source_contribution("harmful", samples, HelpfulStats::default(), 0.1, -0.05),
            build_source_contribution(
                "neutral",
                complement_samples,
                HelpfulStats::default(),
                0.1,
                0.02,
            ),
        ];

        let synergistic = detect_synergistic_candidates("output-0", &contributions, 1.0, None);

        // Should not include the harmful source
        assert!(
            synergistic.iter().all(
                |c| c.primary_source_uuid != "harmful" && c.complement_source_uuid != "harmful"
            ),
            "Harmful source should be pre-screened from synergistic candidates"
        );
    }
}
