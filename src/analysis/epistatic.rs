//! Epistatic neuron pair detection module (Issue #202).
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
//! ## Key Functions
//!
//! - `detect_epistatic_pairs` - Main entry point for epistatic detection
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
