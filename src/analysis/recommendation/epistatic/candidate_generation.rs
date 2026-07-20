//! Epistatic candidate pair generation and filtering.
//!
//! This module handles:
//! - Detecting epistatic neuron pairs from source contributions
//! - Evaluating pairs for complementary activation patterns
//! - Computing complementarity scores and combined improvements
//! - Converting epistatic pairs to coordinated structural candidates
//! - Building source contributions from samples and stats

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::CoordinatedStructuralCandidateJson;
use crate::CoordinatedStructuralOpJson;
use crate::analysis::samples::HelpfulSample;
use crate::analysis::samples::HelpfulStats;

use std::collections::HashSet;

// MIN_SAMPLES_FOR_EPISTATIC_DETECTION moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_EPISTATIC_DETECTION;

// Issue #508: Individual operation pre-screen threshold
use crate::analysis::constants::MAX_INDIVIDUAL_HARM_FOR_PAIRING;

// Issue #897: Conservative weight scale for coordinated estimation
use crate::analysis::constants::COORDINATED_ESTIMATION_WEIGHT_SCALE;

use super::{EpistaticPairCandidate, SourceContribution};

/// Minimum activation threshold to consider a neuron "firing" for pattern detection.
const ACTIVATION_FIRING_THRESHOLD: f32 = 0.5;

/// Minimum complementarity ratio to consider a pair epistatic.
/// A ratio of 0.7 means 70% of samples are covered by one neuron but not the other.
const MIN_COMPLEMENTARITY_RATIO: f32 = 0.7;

/// Detect epistatic neuron pairs targeting the same output.
///
/// This function analyses pairs of source neurons that could benefit from being
/// added together (as a coordinated structural change) even if neither would
/// improve the score individually.
///
/// # Arguments
/// * `target_uuid` - The target neuron UUID
/// * `contributions` - List of source contributions (samples and stats for each source)
/// * `target_impact` - Impact factor of the target neuron (for discounting)
///
/// # Returns
/// A list of epistatic pair candidates that should be returned as coordinated structural candidates.
pub fn detect_epistatic_pairs(
    target_uuid: &str,
    contributions: &[SourceContribution],
    target_impact: f32,
    target_squash: Option<&str>,
) -> Vec<EpistaticPairCandidate> {
    if contributions.len() < 2 {
        return Vec::new();
    }

    // Issue #897: Resolve target activation function for saturation-aware simulation
    let target_activation_fn = target_squash.and_then(crate::activations::target_simulation_fn);

    // Filter to sources with enough samples and non-harmful individual improvement (Issue #508)
    let valid_sources: Vec<&SourceContribution> = contributions
        .iter()
        .filter(|c| {
            c.samples.len() >= MIN_SAMPLES_FOR_EPISTATIC_DETECTION
                && c.individual_improvement >= MAX_INDIVIDUAL_HARM_FOR_PAIRING
        })
        .collect();

    if valid_sources.len() < 2 {
        return Vec::new();
    }

    let mut candidates = Vec::new();

    // Check pairs for complementary activation patterns
    for i in 0..valid_sources.len() {
        for j in (i + 1)..valid_sources.len() {
            let source_a = valid_sources[i];
            let source_b = valid_sources[j];

            if let Some(candidate) = evaluate_pair_for_epistasis(
                target_uuid,
                source_a,
                source_b,
                target_impact,
                target_activation_fn,
            ) {
                candidates.push(candidate);
            }
        }
    }

    // Sort by combined improvement (descending)
    candidates.sort_by(|a, b| b.combined_improvement.total_cmp(&a.combined_improvement));

    candidates
}

/// Evaluate a pair of sources for epistatic relationship (Issue #731).
///
/// Returns Some if the pair shows genuine epistatic characteristics:
/// - Complementary activation patterns (each fires when the other doesn't)
/// - No cross-harm: source A does not hurt samples where source B helps
/// - Super-additive: combined improvement exceeds sum of individual improvements
/// - Cross-validated: benefit holds on both halves of the sample set
fn evaluate_pair_for_epistasis(
    target_uuid: &str,
    source_a: &SourceContribution,
    source_b: &SourceContribution,
    target_impact: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> Option<EpistaticPairCandidate> {
    // Compute complementarity: how much do the activation patterns not overlap?
    let complementarity =
        compute_complementarity(&source_a.firing_indices, &source_b.firing_indices);

    if complementarity < MIN_COMPLEMENTARITY_RATIO {
        return None;
    }

    // Issue #731: Check sample-wise harm — reject if one source hurts samples
    // where the other fires (false complementarity)
    if has_cross_sample_harm(source_a, source_b) {
        return None;
    }

    // Issue #731/#897: Compute combined improvement from sample-level data using
    // saturation-aware simulation when available, not linear approximation.
    // Uses conservative weight scaling (0.2×) for realistic gain estimation.
    let combined_improvement = compute_combined_improvement_from_samples(
        source_a,
        source_b,
        target_impact,
        target_activation_fn,
    );

    // Issue #731: Require super-additivity — combined must exceed sum of individuals.
    // Issue #897: Both combined and individual estimates use conservative weight scaling,
    // so the comparison is fair at the same scale.
    let sum_of_individuals = source_a.individual_improvement + source_b.individual_improvement;

    let is_super_additive =
        combined_improvement > sum_of_individuals * target_impact && combined_improvement > 0.0;

    if !is_super_additive {
        return None;
    }

    // Issue #731: Cross-validation — verify benefit holds on both halves
    if !cross_validate_pair(source_a, source_b, target_impact, target_activation_fn) {
        return None;
    }

    let reason = if source_a.individual_improvement <= 0.0 && source_b.individual_improvement <= 0.0
    {
        format!(
            "True epistasis: neither improves alone, combined {:.1}% (super-additive)",
            combined_improvement * 100.0
        )
    } else if complementarity > 0.9 {
        format!(
            "Highly complementary patterns ({:.0}% non-overlap), super-additive",
            complementarity * 100.0
        )
    } else {
        format!(
            "Complementary patterns ({:.0}% non-overlap), super-additive benefit",
            complementarity * 100.0
        )
    };

    // Issue #897: Apply conservative weight scale to reported gain.
    // During evaluation, coordinated candidate weights are scaled to 0.2× variants,
    // so the reported gain should reflect this more conservative configuration.
    let reported_improvement = combined_improvement * COORDINATED_ESTIMATION_WEIGHT_SCALE;

    Some(EpistaticPairCandidate {
        source_a_uuid: source_a.source_uuid.clone(),
        source_b_uuid: source_b.source_uuid.clone(),
        target_uuid: target_uuid.to_string(),
        weight_a: source_a.optimal_weight,
        weight_b: source_b.optimal_weight,
        combined_improvement: reported_improvement,
        individual_improvement_a: source_a.individual_improvement,
        individual_improvement_b: source_b.individual_improvement,
        complementarity_score: complementarity,
        reason,
    })
}

/// Check for cross-sample harm between two sources (Issue #731).
///
/// Returns true if source A has negative contribution on samples where source B
/// fires, or vice versa. This detects false complementarity where the sources
/// have non-overlapping firing patterns but each hurts the other's samples.
fn has_cross_sample_harm(source_a: &SourceContribution, source_b: &SourceContribution) -> bool {
    let min_samples = source_a.samples.len().min(source_b.samples.len());
    if min_samples < 10 {
        return false;
    }

    let mut a_harm_on_b_samples = 0.0f64;
    let mut b_harm_on_a_samples = 0.0f64;
    let mut b_firing_count = 0usize;
    let mut a_firing_count = 0usize;

    for i in 0..min_samples {
        let a_fires = source_a.samples[i].activation.abs() >= ACTIVATION_FIRING_THRESHOLD;
        let b_fires = source_b.samples[i].activation.abs() >= ACTIVATION_FIRING_THRESHOLD;

        // When B fires: check if A's contribution would be harmful
        // A contribution = weight_a * activation_a — if this has opposite sign to error,
        // it makes the error worse
        if b_fires {
            b_firing_count += 1;
            let a_contribution =
                source_a.optimal_weight as f64 * source_a.samples[i].activation as f64;
            let error = source_a.samples[i].avg_error as f64;
            // Harm = contribution that increases error magnitude
            if error.abs() > 1e-10 && (a_contribution * error) < 0.0 {
                a_harm_on_b_samples += (a_contribution * error).abs();
            }
        }

        // When A fires: check if B's contribution would be harmful
        if a_fires {
            a_firing_count += 1;
            let b_contribution =
                source_b.optimal_weight as f64 * source_b.samples[i].activation as f64;
            let error = source_b.samples[i].avg_error as f64;
            if error.abs() > 1e-10 && (b_contribution * error) < 0.0 {
                b_harm_on_a_samples += (b_contribution * error).abs();
            }
        }
    }

    // Reject if average harm per sample exceeds a small threshold
    let avg_a_harm = if b_firing_count > 0 {
        a_harm_on_b_samples / b_firing_count as f64
    } else {
        0.0
    };
    let avg_b_harm = if a_firing_count > 0 {
        b_harm_on_a_samples / a_firing_count as f64
    } else {
        0.0
    };

    // Threshold: if average harm exceeds 1% of typical error magnitude, reject
    avg_a_harm > 0.01 || avg_b_harm > 0.01
}

/// Cross-validate epistatic pair by splitting samples (Issue #731).
///
/// Computes combined improvement on each half of the samples independently.
/// Both halves must show positive combined improvement for the pair to pass.
fn cross_validate_pair(
    source_a: &SourceContribution,
    source_b: &SourceContribution,
    target_impact: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> bool {
    let min_samples = source_a.samples.len().min(source_b.samples.len());
    if min_samples < 20 {
        // Not enough samples to split — skip cross-validation
        return true;
    }

    let mid = min_samples / 2;

    // Compute combined improvement on first half
    let improvement_first = compute_combined_improvement_on_range(
        source_a,
        source_b,
        target_impact,
        target_activation_fn,
        0,
        mid,
    );

    // Compute combined improvement on second half
    let improvement_second = compute_combined_improvement_on_range(
        source_a,
        source_b,
        target_impact,
        target_activation_fn,
        mid,
        min_samples,
    );

    // Both halves must show positive improvement
    improvement_first > 0.0 && improvement_second > 0.0
}

/// Compute complementarity score between two sets of firing indices.
///
/// Returns a value in [0, 1] where:
/// - 0 means complete overlap (both fire on same samples)
/// - 1 means complete complementarity (never fire together)
pub(crate) fn compute_complementarity(a_indices: &HashSet<u32>, b_indices: &HashSet<u32>) -> f32 {
    if a_indices.is_empty() || b_indices.is_empty() {
        return 0.0;
    }

    let intersection_size = a_indices.intersection(b_indices).count();
    let union_size = a_indices.union(b_indices).count();

    if union_size == 0 {
        return 0.0;
    }

    // Complementarity = 1 - (intersection / union)
    // High complementarity means low overlap
    1.0 - (intersection_size as f32 / union_size as f32)
}

/// Compute firing indices for a set of samples.
///
/// A neuron is considered "firing" on a sample if its activation exceeds the threshold.
pub fn compute_firing_indices(samples: &[HelpfulSample], threshold: f32) -> HashSet<u32> {
    // We need obs_index information - this is typically stored in the sample building process
    // For now, we use sample index as a proxy (matches obs_index for single-sample-per-obs cases)
    samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.activation.abs() >= threshold)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Compute combined improvement from sample-level data (Issue #731).
///
/// Instead of naively summing pre-computed individual improvements, this computes
/// the actual error reduction when both synapses are active simultaneously.
/// This accounts for downstream non-linear effects like weight conflicts.
fn compute_combined_improvement_from_samples(
    source_a: &SourceContribution,
    source_b: &SourceContribution,
    target_impact: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> f32 {
    let min_samples = source_a.samples.len().min(source_b.samples.len());
    compute_combined_improvement_on_range(
        source_a,
        source_b,
        target_impact,
        target_activation_fn,
        0,
        min_samples,
    )
}

/// Compute combined improvement for a range of samples (Issue #731, #897).
///
/// Used for both full-set computation and cross-validation halves.
///
/// Issue #897: When `target_activation_fn` is provided and samples have target data,
/// uses saturation-aware simulation through the target neuron's actual activation
/// function. Otherwise falls back to linear approximation.
///
/// Uses conservative weight scaling (0.2×) to align estimation with the most likely
/// tested configuration during evaluation.
fn compute_combined_improvement_on_range(
    source_a: &SourceContribution,
    source_b: &SourceContribution,
    target_impact: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
    start: usize,
    end: usize,
) -> f32 {
    if end <= start {
        return 0.0;
    }

    let weight_a = source_a.optimal_weight as f64;
    let weight_b = source_b.optimal_weight as f64;

    // Issue #897: Check if saturation-aware simulation is possible
    let use_activation_simulation = target_activation_fn.is_some()
        && (start..end).all(|i| {
            source_a.samples[i].target_value.is_some()
                && source_a.samples[i].target_activation.is_some()
        });

    let mut original_error_sq = 0.0f64;
    let mut combined_error_sq = 0.0f64;

    for i in start..end {
        let sample = &source_a.samples[i];

        if use_activation_simulation {
            // Issue #897: Saturation-aware simulation in ACTIVATION domain
            // Gracefully skip samples missing target data (Issue #940).
            let (Some(target_fn), Some(tv), Some(ta)) = (
                target_activation_fn,
                sample.target_value,
                sample.target_activation,
            ) else {
                continue;
            };
            let target_value = tv as f64;
            let target_activation = ta as f64;

            // Reconstruct expected output: what the target should produce
            let desired_value = target_value + sample.avg_error as f64;
            let expected = target_fn(desired_value as f32) as f64;

            // Baseline error in activation domain
            let baseline_err = expected - target_activation;

            // Simulate adding both contributions through the activation function
            let contribution_a = weight_a * sample.activation as f64;
            let contribution_b = weight_b * source_b.samples[i].activation as f64;
            let new_input = target_value + contribution_a + contribution_b;
            let new_output = target_fn(new_input as f32) as f64;
            let new_err = expected - new_output;

            original_error_sq += baseline_err * baseline_err;
            combined_error_sq += new_err * new_err;
        } else {
            // Linear approximation fallback (original behaviour)
            let error = sample.avg_error as f64;
            let contribution_a = weight_a * sample.activation as f64;
            let contribution_b = weight_b * source_b.samples[i].activation as f64;
            let new_error = error - contribution_a - contribution_b;

            original_error_sq += error * error;
            combined_error_sq += new_error * new_error;
        }
    }

    if original_error_sq < 1e-10 {
        return 0.0;
    }

    let improvement = (1.0 - (combined_error_sq / original_error_sq)) as f32;
    if improvement.is_finite() {
        improvement * target_impact
    } else {
        0.0
    }
}

/// Convert epistatic pair candidates to coordinated structural candidates.
pub fn epistatic_pairs_to_coordinated_candidates(
    pairs: &[EpistaticPairCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    pairs
        .iter()
        .map(|pair| CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: pair.source_a_uuid.clone(),
                    to_neuron_uuid: pair.target_uuid.clone(),
                    weight: pair.weight_a,
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: pair.source_b_uuid.clone(),
                    to_neuron_uuid: pair.target_uuid.clone(),
                    weight: pair.weight_b,
                },
            ],
            expected_creature_score_gain: pair.combined_improvement,
            comment: Some(format!(
                "Epistatic pair: {} (combined improvement {:.2}%, complementarity {:.0}%)",
                pair.reason,
                pair.combined_improvement * 100.0,
                pair.complementarity_score * 100.0
            )),
        })
        .collect()
}

/// Build source contribution from samples and stats.
pub fn build_source_contribution(
    source_uuid: &str,
    samples: Vec<HelpfulSample>,
    stats: HelpfulStats,
    optimal_weight: f32,
    individual_improvement: f32,
) -> SourceContribution {
    let firing_indices = compute_firing_indices(&samples, ACTIVATION_FIRING_THRESHOLD);

    SourceContribution {
        source_uuid: source_uuid.to_string(),
        samples,
        optimal_weight,
        individual_improvement,
        firing_indices,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_complementarity_full_overlap() {
        let a: HashSet<u32> = [1, 2, 3, 4, 5].into_iter().collect();
        let b: HashSet<u32> = [1, 2, 3, 4, 5].into_iter().collect();
        let c = compute_complementarity(&a, &b);
        assert!(c < 0.01, "Full overlap should have ~0 complementarity: {c}");
    }

    #[test]
    fn test_compute_complementarity_no_overlap() {
        let a: HashSet<u32> = [1, 2, 3].into_iter().collect();
        let b: HashSet<u32> = [4, 5, 6].into_iter().collect();
        let c = compute_complementarity(&a, &b);
        assert!(c > 0.99, "No overlap should have ~1 complementarity: {c}");
    }

    #[test]
    fn test_compute_complementarity_partial_overlap() {
        let a: HashSet<u32> = [1, 2, 3, 4].into_iter().collect();
        let b: HashSet<u32> = [3, 4, 5, 6].into_iter().collect();
        let c = compute_complementarity(&a, &b);
        // Intersection: {3, 4} = 2, Union: {1,2,3,4,5,6} = 6
        // Complementarity = 1 - 2/6 = 0.666...
        assert!(
            (c - 0.666).abs() < 0.01,
            "Partial overlap complementarity: {c}"
        );
    }

    #[test]
    fn test_compute_complementarity_empty_sets() {
        let a: HashSet<u32> = HashSet::new();
        let b: HashSet<u32> = [1, 2, 3].into_iter().collect();
        assert_eq!(compute_complementarity(&a, &b), 0.0);
        assert_eq!(compute_complementarity(&b, &a), 0.0);
    }

    #[test]
    fn test_compute_firing_indices() {
        let samples = vec![
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.2,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.9,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
        ];

        let firing = compute_firing_indices(&samples, 0.5);
        assert_eq!(firing.len(), 2);
        assert!(firing.contains(&0));
        assert!(!firing.contains(&1));
        assert!(firing.contains(&2));
    }

    #[test]
    fn test_detect_epistatic_pairs_insufficient_sources() {
        let contributions = vec![SourceContribution {
            source_uuid: "input-0".to_string(),
            samples: vec![],
            optimal_weight: 0.1,
            individual_improvement: 0.0,
            firing_indices: HashSet::new(),
            stats: HelpfulStats::default(),
        }];

        let pairs = detect_epistatic_pairs("output-0", &contributions, 1.0, None);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_epistatic_pairs_to_coordinated_candidates() {
        let pairs = vec![EpistaticPairCandidate {
            source_a_uuid: "input-0".to_string(),
            source_b_uuid: "input-1".to_string(),
            target_uuid: "output-0".to_string(),
            weight_a: 0.3,
            weight_b: 0.3,
            combined_improvement: 0.1,
            individual_improvement_a: 0.02,
            individual_improvement_b: 0.02,
            complementarity_score: 0.9,
            reason: "Test reason".to_string(),
        }];

        let coordinated = epistatic_pairs_to_coordinated_candidates(&pairs);
        assert_eq!(coordinated.len(), 1);
        assert_eq!(coordinated[0].operations.len(), 2);

        // Check that both addSynapse operations are present
        let has_input_0 = coordinated[0].operations.iter().any(|op| {
            matches!(op, CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid, ..
            } if from_neuron_uuid == "input-0")
        });
        let has_input_1 = coordinated[0].operations.iter().any(|op| {
            matches!(op, CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid, ..
            } if from_neuron_uuid == "input-1")
        });
        assert!(has_input_0 && has_input_1);
    }
}
