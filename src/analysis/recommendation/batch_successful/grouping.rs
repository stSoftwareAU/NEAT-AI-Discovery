//! Batch grouping and conversion to coordinated structural candidates.
//!
//! Groups individually successful candidates into non-conflicting batches
//! of 2–4 operations and converts them to `CoordinatedStructuralCandidateJson`.

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use super::detection::detect_individually_successful;
use super::{BatchSuccessfulGroup, IndividualCandidate};

/// Minimum number of candidates in a batch.
const MIN_BATCH_SIZE: usize = 2;

/// Maximum number of candidates in a batch.
const MAX_BATCH_SIZE: usize = 4;

/// Maximum number of batch groups to return.
const MAX_BATCHES: usize = 20;

/// Conservative scaling factor for batch improvement estimation.
///
/// Individual improvements are estimated from per-target least-squares, which
/// over-estimates creature-level impact. This factor scales the sum of
/// individual improvements to a more realistic combined estimate.
const BATCH_IMPROVEMENT_SCALE: f32 = 0.01;

/// Check whether two individual candidates have a structural conflict.
///
/// Two candidates conflict if they represent the same operation — i.e., adding
/// a synapse from the same source to the same target.
pub fn has_structural_conflict(a: &IndividualCandidate, b: &IndividualCandidate) -> bool {
    a.source_uuid == b.source_uuid && a.target_uuid == b.target_uuid
}

/// Group individually successful candidates into non-conflicting batches.
///
/// Candidates are sorted by improvement (best first) and greedily grouped
/// into batches of 2–4 operations. Each batch contains candidates that do
/// not structurally conflict with each other.
///
/// # Arguments
/// * `candidates` — Individually successful candidates (should be pre-sorted
///   by improvement descending).
///
/// # Returns
/// Batch groups with combined improvement estimates.
pub fn group_into_batches(candidates: &[IndividualCandidate]) -> Vec<BatchSuccessfulGroup> {
    if candidates.len() < MIN_BATCH_SIZE {
        return Vec::new();
    }

    let mut batches: Vec<Vec<&IndividualCandidate>> = Vec::new();

    // Greedy batch formation: try to add each candidate to an existing batch,
    // or start a new batch.
    for candidate in candidates {
        let mut added = false;

        for batch in &mut batches {
            if batch.len() >= MAX_BATCH_SIZE {
                continue;
            }

            // Check for conflicts with all existing members.
            let conflicts = batch
                .iter()
                .any(|member| has_structural_conflict(candidate, member));

            if !conflicts {
                batch.push(candidate);
                added = true;
                break;
            }
        }

        if !added && batches.len() < MAX_BATCHES {
            batches.push(vec![candidate]);
        }
    }

    // Filter to batches with at least MIN_BATCH_SIZE candidates.
    batches
        .into_iter()
        .filter(|batch| batch.len() >= MIN_BATCH_SIZE)
        .map(|batch| {
            let combined_improvement: f32 =
                batch.iter().map(|c| c.improvement).sum::<f32>() * BATCH_IMPROVEMENT_SCALE;

            // Issue #983: Collect references for formatting instead of cloning strings.
            let source_list: Vec<&str> = batch.iter().map(|c| c.source_uuid.as_str()).collect();
            let target_list: Vec<&str> = batch
                .iter()
                .map(|c| c.target_uuid.as_str())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            let reason = format!(
                "Batch-successful: {} individually proven sources ({}) targeting {} — \
                 combined for batch testing",
                batch.len(),
                source_list.join(", "),
                target_list.join(", "),
            );

            BatchSuccessfulGroup {
                candidates: batch.into_iter().cloned().collect(),
                combined_improvement,
                reason,
            }
        })
        .collect()
}

/// Main entry point: detect individually successful candidates and group
/// them into batches.
///
/// This function is designed for use with the `discovery_spec!` macro's
/// detect closure.
pub fn detect_batch_successful_groups(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<BatchSuccessfulGroup> {
    let individuals = detect_individually_successful(creature, neuron_records);
    group_into_batches(&individuals)
}

/// Convert batch-successful groups to coordinated structural candidates.
///
/// Each batch becomes a single `CoordinatedStructuralCandidateJson` with
/// `AddSynapse` operations for each candidate in the batch.
///
/// The per-op-count empirical discount is applied automatically
/// during the merge step in `merge_coordinated_structural_replacements`.
pub fn batch_successful_to_coordinated_candidates(
    groups: &[BatchSuccessfulGroup],
) -> Vec<CoordinatedStructuralCandidateJson> {
    groups
        .iter()
        .filter(|g| g.combined_improvement > 0.0)
        .map(|group| {
            let operations: Vec<CoordinatedStructuralOpJson> = group
                .candidates
                .iter()
                .map(|c| CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: c.source_uuid.clone(),
                    to_neuron_uuid: c.target_uuid.clone(),
                    weight: c.weight,
                })
                .collect();

            CoordinatedStructuralCandidateJson {
                operations,
                expected_creature_score_gain: group.combined_improvement,
                comment: Some(group.reason.clone()),
            }
        })
        .collect()
}
