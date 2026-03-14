//! Candidate generation: sample locality grouping and ordered neuron building
//!
//! This module handles sample building, locality optimisation (Issue #221),
//! and neuron ordering for synapse analysis.

use crate::analysis::diagnostics::TargetMap;
use crate::analysis::samples::HelpfulSample;
use crate::analysis::utils::OrderedNeuron;
use crate::types::DiscoverRecord;
use std::collections::HashSet;
use std::sync::Arc;

// =============================================================================
// Sample Locality Grouping (Issue #221)
// =============================================================================

/// Minimum number of sources in a group to make shared sample building worthwhile.
/// Below this threshold, the overhead of grouping exceeds the benefit.
pub(crate) const MIN_GROUP_SIZE_FOR_LOCALITY: usize = 3;

/// Minimum overlap fraction required to group sources together.
/// Sources are grouped if they share at least this fraction of their obs_indices.
const MIN_LOCALITY_OVERLAP: f32 = 0.8;

/// Represents a group of sources with similar obs_index coverage.
/// Sources in the same group can share sample building overhead.
pub(crate) struct SampleLocalityGroup<'a> {
    /// The source neurons in this group
    pub(crate) sources: Vec<(&'a OrderedNeuron, Arc<Vec<DiscoverRecord>>)>,
}

/// Extract obs_indices from source records.
pub(crate) fn extract_obs_indices(records: &[DiscoverRecord]) -> HashSet<u32> {
    records
        .iter()
        .filter(|r| r.activation.is_finite())
        .map(|r| r.obs_index)
        .collect()
}

/// Compute the overlap fraction between two sets of obs_indices.
/// Returns a value in [0, 1] representing what fraction of the smaller set
/// is contained in the larger set.
pub(crate) fn compute_obs_index_overlap(a: &HashSet<u32>, b: &HashSet<u32>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection_size = a.intersection(b).count();
    let min_size = a.len().min(b.len());
    intersection_size as f32 / min_size as f32
}

/// Group sources by sample locality for efficient batch processing.
///
/// Sources with high obs_index overlap (≥80%) are grouped together so that
/// sample building can be done in a single pass through the target data
/// rather than N separate passes.
///
/// # Issue #221: Sample Locality for Correlated Source Neurons
///
/// Expected benefits for typical creatures:
/// - 100 sources, same obs_indices: 1 group (100x reduction in target lookups)
/// - 100 sources, 80% overlap: ~5 groups (20x reduction)
/// - 100 sources, no overlap: 100 groups (no change)
pub(crate) fn group_sources_by_locality<'a>(
    sources: &[(&'a OrderedNeuron, Arc<Vec<DiscoverRecord>>)],
) -> Vec<SampleLocalityGroup<'a>> {
    if sources.len() < MIN_GROUP_SIZE_FOR_LOCALITY {
        // Not enough sources to benefit from grouping
        return sources
            .iter()
            .map(|(neuron, records)| SampleLocalityGroup {
                sources: vec![(neuron, Arc::clone(records))],
            })
            .collect();
    }

    // Extract obs_indices for each source (done once, reused for grouping)
    let source_indices: Vec<HashSet<u32>> = sources
        .iter()
        .map(|(_, records)| extract_obs_indices(records))
        .collect();

    let mut groups: Vec<SampleLocalityGroup<'a>> = Vec::new();
    let mut assigned: Vec<bool> = vec![false; sources.len()];

    for i in 0..sources.len() {
        if assigned[i] {
            continue;
        }

        let my_indices = &source_indices[i];
        if my_indices.is_empty() {
            // Source has no valid indices - put it in its own group
            assigned[i] = true;
            groups.push(SampleLocalityGroup {
                sources: vec![(sources[i].0, Arc::clone(&sources[i].1))],
            });
            continue;
        }

        // Start a new group with this source
        let mut group_sources = vec![(sources[i].0, Arc::clone(&sources[i].1))];
        assigned[i] = true;

        // Find all other sources with high overlap
        for j in (i + 1)..sources.len() {
            if assigned[j] {
                continue;
            }

            let other_indices = &source_indices[j];
            let overlap = compute_obs_index_overlap(my_indices, other_indices);

            if overlap >= MIN_LOCALITY_OVERLAP {
                group_sources.push((sources[j].0, Arc::clone(&sources[j].1)));
                assigned[j] = true;
            }
        }

        groups.push(SampleLocalityGroup {
            sources: group_sources,
        });
    }

    groups
}

/// Build samples for all sources in a locality group efficiently.
///
/// This uses `TargetMap::build_samples_for_group` to build samples for all
/// sources in a single pass through the target data.
///
/// Returns `&str` references to source UUIDs from the locality group,
/// avoiding per-source String allocations in the hot path (Issue #808).
pub(crate) fn build_samples_for_locality_group<'a>(
    group: &SampleLocalityGroup<'a>,
    target_map: &TargetMap,
) -> Vec<(&'a str, Vec<HelpfulSample>, usize)> {
    if group.sources.len() == 1 {
        // Single source - use standard path (no overhead)
        let (source, records) = &group.sources[0];
        let samples = target_map.build_samples_from(records);
        return vec![(source.uuid.as_str(), samples, records.len())];
    }

    // Multiple sources - use batched sample building
    let source_records: Vec<(&str, &[DiscoverRecord])> = group
        .sources
        .iter()
        .map(|(source, records)| (source.uuid.as_str(), records.as_slice()))
        .collect();

    let samples_batch = target_map.build_samples_for_group(&source_records);

    group
        .sources
        .iter()
        .zip(samples_batch)
        .map(|((source, records), samples)| (source.uuid.as_str(), samples, records.len()))
        .collect()
}

// =============================================================================
// Build Ordered Neurons
// =============================================================================

/// Build an ordered list of neurons for the creature.
/// This includes input neurons (named "input-0", "input-1", etc.) followed by
/// all neurons from the creature definition.
pub(crate) fn build_ordered_neurons(creature: &crate::CreatureJson) -> Vec<OrderedNeuron> {
    let mut ordered = Vec::with_capacity(creature.input + creature.neurons.len());

    for input_index in 0..creature.input {
        ordered.push(OrderedNeuron {
            uuid: format!("input-{input_index}"),
            index: input_index,
        });
    }

    for (offset, neuron) in creature.neurons.iter().enumerate() {
        ordered.push(OrderedNeuron {
            uuid: neuron.uuid.clone(),
            index: creature.input + offset,
        });
    }

    ordered
}

// =============================================================================
// Build Samples (test helper)
// =============================================================================

/// Build samples for testing. In production, use TargetMap::from_records() and
/// TargetMap::build_samples_from() for better performance when processing
/// multiple sources against the same target.
#[cfg(test)]
pub(crate) fn build_samples(
    target_records: &[DiscoverRecord],
    from_records: &[DiscoverRecord],
) -> Vec<HelpfulSample> {
    if target_records.is_empty() || from_records.is_empty() {
        return Vec::new();
    }

    // Build map from obs_index to target data (error, value, activation)
    let target_map = TargetMap::from_records(target_records);

    if target_map.map.is_empty() {
        return Vec::new();
    }

    target_map.build_samples_from(from_records)
}
