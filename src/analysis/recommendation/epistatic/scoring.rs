//! Epistatic candidate interference filtering (Issue #415).
//!
//! This module handles:
//! - Filtering epistatic and synergistic candidates whose sources are redundant
//! - Computing activation correlations between sample sets

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::analysis::samples::HelpfulSample;

use super::{EpistaticPairCandidate, SourceContribution, SynergisticCandidate};

/// Minimum correlation threshold to consider two sources as redundant.
/// Correlation >= 0.9 indicates the sources compute essentially the same thing.
const REDUNDANCY_CORRELATION_THRESHOLD: f64 = 0.9;

/// Compute Pearson correlation between two sets of activation samples.
pub(crate) fn compute_sample_correlation(
    samples_a: &[HelpfulSample],
    samples_b: &[HelpfulSample],
) -> f64 {
    let n = samples_a.len().min(samples_b.len());
    crate::analysis::detection::stats::pearson_correlation_samples(samples_a, samples_b, n)
}

/// Filter epistatic pairs to remove those that would interfere (Issue #415).
///
/// This function takes detected epistatic pairs and removes any that show
/// interference patterns that would cause combo-successful to fail.
pub fn filter_interfering_epistatic_pairs(
    pairs: Vec<EpistaticPairCandidate>,
    contributions: &[SourceContribution],
) -> Vec<EpistaticPairCandidate> {
    pairs
        .into_iter()
        .filter(|pair| {
            // Find the contributions for this pair
            let contrib_a = contributions
                .iter()
                .find(|c| c.source_uuid == pair.source_a_uuid);
            let contrib_b = contributions
                .iter()
                .find(|c| c.source_uuid == pair.source_b_uuid);

            let (Some(a), Some(b)) = (contrib_a, contrib_b) else {
                return true; // Keep if we can't find contributions
            };

            // Check for redundancy
            let correlation = compute_sample_correlation(&a.samples, &b.samples);
            if correlation >= REDUNDANCY_CORRELATION_THRESHOLD {
                return false; // Filter out redundant pairs
            }

            true
        })
        .collect()
}

/// Filter synergistic candidates to remove those that would interfere (Issue #415).
///
/// This function takes detected synergistic candidates and removes any that show
/// interference patterns that would cause combo-successful to fail.
pub fn filter_interfering_synergistic_candidates(
    candidates: Vec<SynergisticCandidate>,
    contributions: &[SourceContribution],
) -> Vec<SynergisticCandidate> {
    candidates
        .into_iter()
        .filter(|candidate| {
            // Find the contributions for this candidate
            let primary = contributions
                .iter()
                .find(|c| c.source_uuid == candidate.primary_source_uuid);
            let complement = contributions
                .iter()
                .find(|c| c.source_uuid == candidate.complement_source_uuid);

            let (Some(p), Some(c)) = (primary, complement) else {
                return true; // Keep if we can't find contributions
            };

            // Check for redundancy
            let correlation = compute_sample_correlation(&p.samples, &c.samples);
            if correlation >= REDUNDANCY_CORRELATION_THRESHOLD {
                return false; // Filter out redundant pairs
            }

            true
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::samples::HelpfulStats;
    use std::collections::HashSet;

    #[test]
    fn test_compute_sample_correlation_identical() {
        let samples: Vec<HelpfulSample> = (0..64)
            .map(|i| HelpfulSample {
                activation: i as f32,
                avg_error: 0.0,
                target_value: None,
                target_activation: None,
            })
            .collect();

        let corr = compute_sample_correlation(&samples, &samples);
        assert!(
            corr > 0.99,
            "Identical samples should have correlation ~1.0: {corr}"
        );
    }

    #[test]
    fn test_compute_sample_correlation_anti_correlated() {
        let samples_a: Vec<HelpfulSample> = (0..64)
            .map(|i| HelpfulSample {
                activation: i as f32,
                avg_error: 0.0,
                target_value: None,
                target_activation: None,
            })
            .collect();

        let samples_b: Vec<HelpfulSample> = (0..64)
            .map(|i| HelpfulSample {
                activation: 64.0 - i as f32, // Reverse order
                avg_error: 0.0,
                target_value: None,
                target_activation: None,
            })
            .collect();

        let corr = compute_sample_correlation(&samples_a, &samples_b);
        assert!(
            corr < -0.99,
            "Anti-correlated samples should have correlation ~-1.0: {corr}"
        );
    }

    #[test]
    fn test_filter_interfering_epistatic_pairs() {
        // Create contributions with identical samples (redundant)
        let samples: Vec<HelpfulSample> = (0..64)
            .map(|i| HelpfulSample {
                activation: i as f32 / 64.0,
                avg_error: 0.3,
                target_value: None,
                target_activation: None,
            })
            .collect();

        let contributions = vec![
            SourceContribution {
                source_uuid: "input-0".to_string(),
                samples: samples.clone(),
                optimal_weight: 0.5,
                individual_improvement: 0.1,
                firing_indices: HashSet::new(),
                stats: HelpfulStats::default(),
            },
            SourceContribution {
                source_uuid: "input-1".to_string(),
                samples,
                optimal_weight: 0.5,
                individual_improvement: 0.1,
                firing_indices: HashSet::new(),
                stats: HelpfulStats::default(),
            },
        ];

        let pairs = vec![EpistaticPairCandidate {
            source_a_uuid: "input-0".to_string(),
            source_b_uuid: "input-1".to_string(),
            target_uuid: "output-0".to_string(),
            weight_a: 0.5,
            weight_b: 0.5,
            combined_improvement: 0.1,
            individual_improvement_a: 0.05,
            individual_improvement_b: 0.05,
            complementarity_score: 0.0, // Would be wrong due to identical patterns
            reason: "Test".to_string(),
        }];

        let filtered = filter_interfering_epistatic_pairs(pairs, &contributions);
        assert!(
            filtered.is_empty(),
            "Redundant pairs should be filtered out"
        );
    }

    #[test]
    fn test_prescreen_rejects_strongly_harmful_source_in_epistatic() {
        // Source with individual_improvement below MAX_INDIVIDUAL_HARM_FOR_PAIRING
        // should be excluded from valid_sources, so no pairs are formed.
        use crate::analysis::recommendation::epistatic::candidate_generation::{
            build_source_contribution, detect_epistatic_pairs,
        };

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

        let pairs = detect_epistatic_pairs("output-0", &contributions, 1.0, None);

        // Neither should include the harmful source
        assert!(
            pairs
                .iter()
                .all(|p| p.source_a_uuid != "harmful" && p.source_b_uuid != "harmful"),
            "Harmful source should be pre-screened from epistatic pairs"
        );
    }

    #[test]
    fn test_prescreen_rejects_mildly_negative_source() {
        // Issue #731: Threshold tightened from -0.01 to 0.0.
        // A source with individual_improvement = -0.005 (mildly negative) should
        // now be pre-screened out, since the threshold is 0.0.
        use crate::analysis::recommendation::epistatic::candidate_generation::{
            build_source_contribution, detect_epistatic_pairs,
        };

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
            build_source_contribution("mildly-neg", samples, HelpfulStats::default(), 0.1, -0.005),
            build_source_contribution(
                "positive",
                complement_samples,
                HelpfulStats::default(),
                0.1,
                0.03,
            ),
        ];

        // With threshold at 0.0, the mildly-negative source is rejected by pre-screen.
        // Only one valid source remains, so no pairs can be formed.
        let pairs = detect_epistatic_pairs("output-0", &contributions, 1.0, None);
        assert!(
            pairs
                .iter()
                .all(|p| p.source_a_uuid != "mildly-neg" && p.source_b_uuid != "mildly-neg"),
            "Mildly negative source should be pre-screened out with threshold 0.0"
        );
    }
}
