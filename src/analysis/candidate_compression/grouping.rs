//! Candidate grouping and deduplication by target neuron.
//!
//! Groups synapse candidates by their `to_neuron_uuid` and deduplicates by
//! source neuron, keeping the best candidate per source.

use std::collections::HashMap;

use crate::CandidateSynapseJson;

use super::CompressibleGroup;
use crate::analysis::constants::MIN_COMPRESSED_SOURCES;

/// Detect groups of helpful synapse candidates that can be compressed.
///
/// Groups candidates by `to_neuron_uuid`, then filters for groups with
/// ≥ `MIN_COMPRESSED_SOURCES` candidates from distinct `from_neuron_uuid` sources.
pub fn detect_compressible_groups(
    helpful_synapses: &[CandidateSynapseJson],
) -> Vec<CompressibleGroup> {
    // Group by target neuron.
    let mut by_target: HashMap<String, Vec<CandidateSynapseJson>> = HashMap::new();
    for candidate in helpful_synapses {
        by_target
            .entry(candidate.to_neuron_uuid.clone())
            .or_default()
            .push(candidate.clone());
    }

    let mut groups = Vec::new();
    for (to_uuid, candidates) in by_target {
        // Deduplicate by from_neuron_uuid — keep the best candidate per source.
        let mut best_by_source: HashMap<String, CandidateSynapseJson> = HashMap::new();
        for c in candidates {
            best_by_source
                .entry(c.from_neuron_uuid.clone())
                .and_modify(|existing| {
                    if c.expected_creature_score_gain > existing.expected_creature_score_gain {
                        *existing = c.clone();
                    }
                })
                .or_insert(c);
        }

        let distinct_candidates: Vec<CandidateSynapseJson> = best_by_source.into_values().collect();

        if distinct_candidates.len() >= MIN_COMPRESSED_SOURCES {
            groups.push(CompressibleGroup {
                to_neuron_uuid: to_uuid,
                candidates: distinct_candidates,
            });
        }
    }

    // Sort groups by target UUID for deterministic output.
    groups.sort_by(|a, b| a.to_neuron_uuid.cmp(&b.to_neuron_uuid));
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(from: &str, to: &str, weight: f32, gain: f32) -> CandidateSynapseJson {
        CandidateSynapseJson {
            from_neuron_uuid: from.to_string(),
            to_neuron_uuid: to.to_string(),
            from_neuron_index: None,
            to_neuron_index: None,
            weight,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: gain,
            expected_creature_score_gain: gain,
            improved_count: 80,
            total_count: 100,
            improvement_magnitude_ratio: None,
            target_neuron_stats: None,
            outlier_reduction_info: None,
            prediction_confidence: 0.8,
            expected_score_gain_confidence_interval: [gain * 0.5, gain * 1.5],
            comment: None,
        }
    }

    #[test]
    fn test_detect_compressible_groups_basic() {
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.02),
            make_candidate("input-b", "output-1", 0.5, 0.03),
        ];

        let groups = detect_compressible_groups(&candidates);
        assert_eq!(groups.len(), 1, "Should detect one compressible group");
        assert_eq!(groups[0].to_neuron_uuid, "output-1");
        assert_eq!(groups[0].candidates.len(), 2);
    }

    #[test]
    fn test_detect_no_groups_for_single_candidate() {
        let candidates = vec![make_candidate("input-a", "output-1", 0.3, 0.02)];

        let groups = detect_compressible_groups(&candidates);
        assert!(
            groups.is_empty(),
            "Single candidate per target should not form a group"
        );
    }

    #[test]
    fn test_detect_no_groups_for_same_source() {
        // Two candidates from the same source to the same target — only one distinct source.
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.02),
            make_candidate("input-a", "output-1", 0.5, 0.03),
        ];

        let groups = detect_compressible_groups(&candidates);
        assert!(
            groups.is_empty(),
            "Same source should not form a compressible group"
        );
    }

    #[test]
    fn test_detect_multiple_groups() {
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.02),
            make_candidate("input-b", "output-1", 0.5, 0.03),
            make_candidate("input-c", "output-2", 0.4, 0.01),
            make_candidate("input-d", "output-2", 0.6, 0.04),
        ];

        let groups = detect_compressible_groups(&candidates);
        assert_eq!(groups.len(), 2, "Should detect two compressible groups");
    }

    #[test]
    fn test_detect_empty_input() {
        let groups = detect_compressible_groups(&[]);
        assert!(groups.is_empty(), "Empty input should produce no groups");
    }
}
