//! Dominant-neuron deduplication logic (Issue #509).
//!
//! When many epistatic or synergistic pairs share the same dominant source neuron,
//! this module caps pairs per group to free candidate budget for alternative sources.

use std::cmp::Ordering;
use std::collections::HashMap;

use super::{EpistaticPairCandidate, SynergisticCandidate};

/// Maximum number of coordinated-structural pairs per dominant neuron.
///
/// When many epistatic or synergistic pairs share the same dominant source neuron
/// (the one with the larger individual improvement), only this many diverse pairs
/// are kept per group. The rest are discarded to free candidate budget for
/// alternative sources, activation functions, or weight ranges.
///
/// Production discovery-cache analysis (creature b2ff6e45) showed
/// 10 pairs sharing the same dominant neuron, all failing identically. Capping
/// to 3 frees 7 candidate slots with no loss of coverage.
const MAX_PAIRS_PER_DOMINANT_NEURON: usize = 3;

/// Total order for epistatic pairs: combined improvement descending, ties
/// broken by UUIDs so output never depends on `HashMap` seed (Issue #2191).
fn cmp_epistatic(a: &EpistaticPairCandidate, b: &EpistaticPairCandidate) -> Ordering {
    b.combined_improvement
        .total_cmp(&a.combined_improvement)
        .then_with(|| a.source_a_uuid.cmp(&b.source_a_uuid))
        .then_with(|| a.source_b_uuid.cmp(&b.source_b_uuid))
        .then_with(|| a.target_uuid.cmp(&b.target_uuid))
}

/// Total order for synergistic candidates: combined improvement descending,
/// ties broken by UUIDs so output never depends on `HashMap` seed (Issue #2191).
fn cmp_synergistic(a: &SynergisticCandidate, b: &SynergisticCandidate) -> Ordering {
    b.combined_improvement
        .total_cmp(&a.combined_improvement)
        .then_with(|| a.primary_source_uuid.cmp(&b.primary_source_uuid))
        .then_with(|| a.complement_source_uuid.cmp(&b.complement_source_uuid))
        .then_with(|| a.target_uuid.cmp(&b.target_uuid))
}

/// Deduplicate epistatic pairs that share a common dominant neuron (Issue #509).
///
/// Groups pairs by the neuron with the larger individual improvement (the
/// "dominant" neuron). From each group, keeps at most
/// `MAX_PAIRS_PER_DOMINANT_NEURON` pairs, selecting by highest combined
/// improvement to maximise diversity and expected benefit.
///
/// # Arguments
/// * `pairs` - Epistatic pair candidates to deduplicate.
///
/// # Returns
/// A deduplicated list of pairs, capped per dominant neuron group.
pub fn deduplicate_by_dominant_neuron(
    mut pairs: Vec<EpistaticPairCandidate>,
) -> Vec<EpistaticPairCandidate> {
    if pairs.len() <= MAX_PAIRS_PER_DOMINANT_NEURON {
        return pairs;
    }

    // Group by dominant neuron (higher individual improvement determines dominance)
    let mut groups: HashMap<String, Vec<EpistaticPairCandidate>> = HashMap::new();
    for pair in pairs.drain(..) {
        let dominant = if pair.individual_improvement_a >= pair.individual_improvement_b {
            pair.source_a_uuid.clone()
        } else {
            pair.source_b_uuid.clone()
        };
        groups.entry(dominant).or_default().push(pair);
    }

    let mut result = Vec::new();
    for (_dominant, mut group) in groups {
        // Best combined improvement first, UUID tie-break — keep the best
        group.sort_by(cmp_epistatic);
        group.truncate(MAX_PAIRS_PER_DOMINANT_NEURON);
        result.extend(group);
    }

    // Overall order: combined improvement, then UUIDs (never hash order)
    result.sort_by(cmp_epistatic);
    result
}

/// Deduplicate synergistic candidates that share a common dominant neuron (Issue #509).
///
/// For synergistic candidates, the primary source is the dominant neuron (it was
/// chosen as the best single-source candidate). Groups by primary source and
/// keeps at most `MAX_PAIRS_PER_DOMINANT_NEURON` per group.
///
/// # Arguments
/// * `candidates` - Synergistic candidates to deduplicate.
///
/// # Returns
/// A deduplicated list of candidates, capped per primary source group.
pub fn deduplicate_synergistic_by_dominant_neuron(
    mut candidates: Vec<SynergisticCandidate>,
) -> Vec<SynergisticCandidate> {
    if candidates.len() <= MAX_PAIRS_PER_DOMINANT_NEURON {
        return candidates;
    }

    let mut groups: HashMap<String, Vec<SynergisticCandidate>> = HashMap::new();
    for candidate in candidates.drain(..) {
        groups
            .entry(candidate.primary_source_uuid.clone())
            .or_default()
            .push(candidate);
    }

    let mut result = Vec::new();
    for (_primary, mut group) in groups {
        group.sort_by(cmp_synergistic);
        group.truncate(MAX_PAIRS_PER_DOMINANT_NEURON);
        result.extend(group);
    }

    result.sort_by(cmp_synergistic);
    result
}
