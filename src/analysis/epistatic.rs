//! Epistatic neuron pair detection module (Issue #202) and synergistic discovery (Issue #189).
//!
//! Epistatic changes are structural modifications where no single operation improves
//! the score, but a group of operations does. This module pre-detects such relationships
//! during analysis, rather than discovering them post-hoc.
//!
//! ## Detection Strategies
//!
//! 1. **Correlation analysis**: For neuron pairs (A, B) targeting the same output,
//!    compute correlation between their predicted improvements. Flag pairs where
//!    individual improvements are low but combined might be high.
//!
//! 2. **Shared error pattern detection**: Find neurons that improve on complementary
//!    sample subsets. These are candidates for combined structural changes.
//!
//! 3. **Residual analysis** (Issue #189): Find the best single-source candidate, compute
//!    residual error after applying it, then search for a second source that reduces the
//!    residual. This is O(2n) instead of O(n²) and detects XOR-like patterns.
//!
//! ## Key Functions
//!
//! - `detect_epistatic_pairs` - Main entry point for epistatic detection (Issue #202)
//! - `detect_synergistic_candidates` - Residual-based synergistic discovery (Issue #189)
//! - `compute_pair_correlation` - Compute correlation between neuron pair improvements
//! - `detect_complementary_patterns` - Find neurons with complementary activation patterns

use crate::analysis::samples::{HelpfulSample, HelpfulStats};
use crate::CoordinatedStructuralCandidateJson;
use crate::CoordinatedStructuralOpJson;

use std::collections::HashSet;

/// Minimum samples required for reliable epistatic detection.
const MIN_SAMPLES_FOR_EPISTATIC_DETECTION: usize = 20;

/// Minimum activation threshold to consider a neuron "firing" for pattern detection.
const ACTIVATION_FIRING_THRESHOLD: f32 = 0.5;

/// Minimum complementarity ratio to consider a pair epistatic.
/// A ratio of 0.7 means 70% of samples are covered by one neuron but not the other.
const MIN_COMPLEMENTARITY_RATIO: f32 = 0.7;

/// Minimum residual reduction ratio for synergistic detection (Issue #189).
/// The second source must reduce residual error by at least this fraction.
const MIN_RESIDUAL_REDUCTION_RATIO: f32 = 0.1;

/// Minimum synergistic benefit ratio (Issue #189).
/// Combined improvement must exceed max(individual) * this factor.
const MIN_SYNERGISTIC_BENEFIT_RATIO: f32 = 1.1;

/// Minimum samples for reliable residual analysis.
const MIN_SAMPLES_FOR_RESIDUAL_ANALYSIS: usize = 30;

/// Result of evaluating a potential epistatic pair.
#[derive(Debug, Clone)]
pub struct EpistaticPairCandidate {
    /// First source neuron UUID
    pub source_a_uuid: String,
    /// Second source neuron UUID
    pub source_b_uuid: String,
    /// Target neuron UUID
    pub target_uuid: String,
    /// Optimal weight for source A's synapse
    pub weight_a: f32,
    /// Optimal weight for source B's synapse
    pub weight_b: f32,
    /// Expected combined improvement
    pub combined_improvement: f32,
    /// Individual improvement for source A alone
    pub individual_improvement_a: f32,
    /// Individual improvement for source B alone
    pub individual_improvement_b: f32,
    /// Complementarity score (0 to 1, higher = more complementary)
    pub complementarity_score: f32,
    /// Description of why this pair is epistatic
    pub reason: String,
}

/// Result of synergistic candidate detection via residual analysis (Issue #189).
///
/// A synergistic candidate is a pair of sources where:
/// - Neither source alone provides strong improvement
/// - Together they reduce error better than either alone
/// - This is detected via residual analysis: apply best source, find second source for residual
#[derive(Debug, Clone)]
pub struct SynergisticCandidate {
    /// Primary source neuron UUID (best single-source candidate)
    pub primary_source_uuid: String,
    /// Complementary source neuron UUID (reduces residual error)
    pub complement_source_uuid: String,
    /// Target neuron UUID
    pub target_uuid: String,
    /// Optimal weight for primary source's synapse
    pub primary_weight: f32,
    /// Optimal weight for complement source's synapse
    pub complement_weight: f32,
    /// Expected combined improvement (fraction of error reduced)
    pub combined_improvement: f32,
    /// Primary source individual improvement
    pub primary_improvement: f32,
    /// Complement source individual improvement
    pub complement_improvement: f32,
    /// Residual reduction achieved by complement source (fraction)
    pub residual_reduction: f32,
    /// Synergy ratio: combined / max(individual)
    pub synergy_ratio: f32,
    /// Description of why this pair is synergistic
    pub reason: String,
}

/// Represents a source neuron's contribution to a target.
#[derive(Debug, Clone)]
pub struct SourceContribution {
    /// Source neuron UUID
    pub source_uuid: String,
    /// Samples where this source contributes to the target
    pub samples: Vec<HelpfulSample>,
    /// Computed optimal weight for this source
    pub optimal_weight: f32,
    /// Individual improvement prediction
    pub individual_improvement: f32,
    /// Set of obs_indices where source fires (activation > threshold)
    pub firing_indices: HashSet<u32>,
    /// GPU evaluation stats
    pub stats: HelpfulStats,
}

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
) -> Vec<EpistaticPairCandidate> {
    if contributions.len() < 2 {
        return Vec::new();
    }

    // Filter to sources with enough samples
    let valid_sources: Vec<&SourceContribution> = contributions
        .iter()
        .filter(|c| c.samples.len() >= MIN_SAMPLES_FOR_EPISTATIC_DETECTION)
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

            if let Some(candidate) =
                evaluate_pair_for_epistasis(target_uuid, source_a, source_b, target_impact)
            {
                candidates.push(candidate);
            }
        }
    }

    // Sort by combined improvement (descending)
    candidates.sort_by(|a, b| {
        b.combined_improvement
            .partial_cmp(&a.combined_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Evaluate a pair of sources for epistatic relationship.
///
/// Returns Some if the pair shows epistatic characteristics:
/// - Complementary activation patterns (each fires when the other doesn't)
/// - Combined improvement exceeds sum of individual improvements
fn evaluate_pair_for_epistasis(
    target_uuid: &str,
    source_a: &SourceContribution,
    source_b: &SourceContribution,
    target_impact: f32,
) -> Option<EpistaticPairCandidate> {
    // Compute complementarity: how much do the activation patterns not overlap?
    let complementarity =
        compute_complementarity(&source_a.firing_indices, &source_b.firing_indices);

    if complementarity < MIN_COMPLEMENTARITY_RATIO {
        return None;
    }

    // Compute combined improvement
    // When patterns are complementary, combined effect is roughly additive
    let combined_improvement = compute_combined_improvement(source_a, source_b, target_impact);

    // Calculate sum of individual improvements for comparison
    let _sum_of_individuals = source_a.individual_improvement + source_b.individual_improvement;
    let best_individual = source_a
        .individual_improvement
        .max(source_b.individual_improvement);

    // Epistatic pairs are valuable when:
    // 1. Both have positive individual improvements with complementary patterns
    //    (combined > best individual, since they help different samples)
    // 2. OR individuals are low/zero but combined is positive
    //    (true epistasis - neither helps alone but together they do)
    //
    // For complementary patterns, the combined improvement should be roughly
    // the sum of individuals (since they help non-overlapping samples).
    // We want to detect cases where combining them is significantly beneficial.

    let is_truly_epistatic = source_a.individual_improvement <= 0.0
        && source_b.individual_improvement <= 0.0
        && combined_improvement > 0.0;

    let is_complementary_beneficial = complementarity >= MIN_COMPLEMENTARITY_RATIO
        && combined_improvement > best_individual
        && combined_improvement > 0.0;

    if !is_truly_epistatic && !is_complementary_beneficial {
        return None;
    }

    let reason = if source_a.individual_improvement <= 0.0 && source_b.individual_improvement <= 0.0
    {
        "Both neurons have non-positive individual improvement but combined improvement is positive"
            .to_string()
    } else if complementarity > 0.9 {
        format!(
            "Highly complementary patterns ({:.0}% non-overlap)",
            complementarity * 100.0
        )
    } else {
        format!(
            "Complementary patterns ({:.0}% non-overlap) with combined benefit",
            complementarity * 100.0
        )
    };

    Some(EpistaticPairCandidate {
        source_a_uuid: source_a.source_uuid.clone(),
        source_b_uuid: source_b.source_uuid.clone(),
        target_uuid: target_uuid.to_string(),
        weight_a: source_a.optimal_weight,
        weight_b: source_b.optimal_weight,
        combined_improvement,
        individual_improvement_a: source_a.individual_improvement,
        individual_improvement_b: source_b.individual_improvement,
        complementarity_score: complementarity,
        reason,
    })
}

/// Compute complementarity score between two sets of firing indices.
///
/// Returns a value in [0, 1] where:
/// - 0 means complete overlap (both fire on same samples)
/// - 1 means complete complementarity (never fire together)
fn compute_complementarity(a_indices: &HashSet<u32>, b_indices: &HashSet<u32>) -> f32 {
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

/// Compute combined improvement for an epistatic pair.
///
/// This estimates what the improvement would be if both synapses were added together.
fn compute_combined_improvement(
    source_a: &SourceContribution,
    source_b: &SourceContribution,
    target_impact: f32,
) -> f32 {
    // For complementary patterns where neurons fire on different samples,
    // the combined improvement should be roughly the sum of individual improvements.
    //
    // Key insight: Each source's individual_improvement is calculated over the WHOLE
    // sample set. When source A fires on 50% of samples and achieves X% improvement
    // overall, and source B fires on the other 50% and also achieves X% improvement,
    // the combined effect is 2*X% (since they help non-overlapping samples).
    //
    // More generally: combined = A.improvement + B.improvement * (1 - overlap_fraction)
    // where overlap_fraction = intersection_size / max(firing_a, firing_b)

    let intersection_size = source_a
        .firing_indices
        .intersection(&source_b.firing_indices)
        .count() as f32;
    let firing_a = source_a.firing_indices.len() as f32;
    let firing_b = source_b.firing_indices.len() as f32;

    // Compute overlap fraction relative to the larger firing set
    let max_firing = firing_a.max(firing_b);
    let overlap_fraction = if max_firing > 0.0 {
        intersection_size / max_firing
    } else {
        1.0
    };

    // Combined improvement: A's improvement + B's improvement scaled by non-overlap
    // For perfect complementarity (overlap_fraction = 0), this is A + B
    // For complete overlap (overlap_fraction = 1), this is max(A, B)
    let combined_improvement = source_a.individual_improvement
        + source_b.individual_improvement * (1.0 - overlap_fraction);

    // Apply target impact discount
    combined_improvement * target_impact
}

/// Convert epistatic pair candidates to coordinated structural candidates.
pub fn epistatic_pairs_to_coordinated_candidates(
    pairs: &[EpistaticPairCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    pairs
        .iter()
        .map(|pair| CoordinatedStructuralCandidateJson {
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

// ============================================================================
// Issue #189: Synergistic Discovery via Residual Analysis
// ============================================================================

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
) -> Vec<SynergisticCandidate> {
    if contributions.len() < 2 {
        return Vec::new();
    }

    // Filter to sources with enough samples
    let valid_sources: Vec<&SourceContribution> = contributions
        .iter()
        .filter(|c| c.samples.len() >= MIN_SAMPLES_FOR_RESIDUAL_ANALYSIS)
        .collect();

    if valid_sources.len() < 2 {
        return Vec::new();
    }

    // Step 1: Find the best single-source candidate
    let best_primary = valid_sources.iter().max_by(|a, b| {
        a.individual_improvement
            .partial_cmp(&b.individual_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let Some(primary) = best_primary else {
        return Vec::new();
    };

    // Step 2: Compute residual errors after applying primary source
    // For each sample, residual = original_error - (primary_weight * primary_activation)
    let residuals = compute_residual_errors(&primary.samples, primary.optimal_weight);

    // Step 3: Search for complementary sources that reduce the residual
    let mut candidates = Vec::new();

    for source in &valid_sources {
        // Skip the primary source itself
        if source.source_uuid == primary.source_uuid {
            continue;
        }

        // Evaluate how well this source reduces the residual error
        if let Some(candidate) =
            evaluate_residual_reduction(target_uuid, primary, source, &residuals, target_impact)
        {
            candidates.push(candidate);
        }
    }

    // Sort by combined improvement (descending)
    candidates.sort_by(|a, b| {
        b.combined_improvement
            .partial_cmp(&a.combined_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Compute residual errors after applying a source with given weight.
///
/// Residual error = original_error - (weight × activation)
fn compute_residual_errors(samples: &[HelpfulSample], weight: f32) -> Vec<ResidualSample> {
    samples
        .iter()
        .enumerate()
        .map(|(idx, s)| {
            let contribution = weight * s.activation;
            let residual_error = s.avg_error - contribution;
            ResidualSample {
                sample_index: idx,
                original_error: s.avg_error,
                residual_error,
                primary_contribution: contribution,
            }
        })
        .collect()
}

/// Internal structure for residual analysis.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for debugging and future extensions
struct ResidualSample {
    sample_index: usize,
    original_error: f32,
    residual_error: f32,
    primary_contribution: f32,
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
    if n_samples < 3 {
        return 0.0;
    }

    // Compute means
    let mut sum_primary = 0.0f64;
    let mut sum_complement = 0.0f64;

    for i in 0..n_samples {
        sum_primary += primary_samples[i].activation as f64;
        sum_complement += complement_samples[i].activation as f64;
    }

    let mean_primary = sum_primary / n_samples as f64;
    let mean_complement = sum_complement / n_samples as f64;

    // Compute covariance and variances
    let mut cov = 0.0f64;
    let mut var_primary = 0.0f64;
    let mut var_complement = 0.0f64;

    for i in 0..n_samples {
        let p = primary_samples[i].activation as f64 - mean_primary;
        let c = complement_samples[i].activation as f64 - mean_complement;

        cov += p * c;
        var_primary += p * p;
        var_complement += c * c;
    }

    // Compute correlation coefficient
    let denominator = (var_primary * var_complement).sqrt();
    if denominator < 1e-10 {
        return 0.0;
    }

    cov / denominator
}

/// Evaluate how well a complement source reduces the residual error.
fn evaluate_residual_reduction(
    target_uuid: &str,
    primary: &SourceContribution,
    complement: &SourceContribution,
    residuals: &[ResidualSample],
    target_impact: f32,
) -> Option<SynergisticCandidate> {
    // We need to match samples between primary and complement
    // Build a map of sample activations for the complement source
    // Note: samples may not have the same indices, so we need to be careful

    // For simplicity, we assume samples are aligned (same obs_indices)
    // In practice, we should use obs_index for matching

    let min_samples = residuals.len().min(complement.samples.len());
    if min_samples < MIN_SAMPLES_FOR_RESIDUAL_ANALYSIS {
        return None;
    }

    // Check activation pattern correlation between primary and complement
    // If they have very high correlation, adding both is redundant, not synergistic
    let activation_correlation =
        compute_activation_correlation(&primary.samples, &complement.samples, min_samples);

    // Skip if activations are too similar (correlation > 0.9)
    // This prevents false positives where both sources are essentially the same
    if activation_correlation > 0.9 {
        return None;
    }

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
        let activation = complement_sample.activation as f64;
        let original = residual_sample.original_error as f64;
        let residual = residual_sample.residual_error as f64;
        let combined_residual = residual - (complement_weight as f64 * activation);

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

    Some(SynergisticCandidate {
        primary_source_uuid: primary.source_uuid.clone(),
        complement_source_uuid: complement.source_uuid.clone(),
        target_uuid: target_uuid.to_string(),
        primary_weight: primary.optimal_weight,
        complement_weight,
        combined_improvement: (combined_improvement * target_impact as f64) as f32,
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

        let pairs = detect_epistatic_pairs("output-0", &contributions, 1.0);
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
