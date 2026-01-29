//! Candidate clustering module (Issue #224).
//!
//! Groups similar discovery candidates to reduce redundant ablation tests. When
//! discovery returns many candidates targeting the same neuron from similar source
//! regions, clustering identifies these groups so the TypeScript controller can test
//! a representative candidate first and skip the rest if it fails.
//!
//! ## Clustering Criteria
//!
//! Candidates are grouped by:
//! 1. **Same target neuron** (`to_neuron_uuid`) — candidates must target the same neuron.
//! 2. **Same source type** (input vs hidden) — different neuron types have different
//!    signal characteristics and should not be mixed.
//! 3. **Similar improvement prediction** — candidates with very different expected
//!    improvements are unlikely to be truly redundant.
//!
//! ## Output
//!
//! Each cluster contains:
//! - A **representative** (the highest-improvement candidate in the cluster).
//! - A list of **member UUIDs** (all candidates in the cluster).
//! - An **internal correlation** score (0.0 to 1.0) indicating how similar the
//!   members' improvements are.
//!
//! The controller tests the representative first. If it fails, all members are
//! skipped (high correlation implies they would also fail). If it succeeds, the
//! controller may test additional members with lower priority.

use serde::Serialize;
use std::collections::HashMap;

/// Maximum ratio between the best and worst improvement within a cluster.
/// Candidates whose improvement differs by more than this factor are split
/// into separate sub-clusters.
const IMPROVEMENT_SIMILARITY_RATIO: f32 = 5.0;

/// Minimum number of candidates in a group to form a cluster.
/// A single candidate has no redundancy to reduce.
const MIN_CLUSTER_SIZE: usize = 2;

/// Input representation for a candidate that can be clustered.
///
/// This is a simplified view of a synapse or structural candidate, containing
/// only the fields needed for clustering decisions.
#[derive(Debug, Clone)]
pub struct ClusterableCandidate {
    /// UUID of the source neuron.
    pub from_neuron_uuid: String,
    /// UUID of the target neuron.
    pub to_neuron_uuid: String,
    /// Expected improvement from this candidate.
    pub expected_improvement: f32,
    /// Type of the source neuron (e.g., "input", "hidden").
    pub neuron_type: String,
}

/// JSON output for a candidate cluster.
///
/// Tells the controller which candidates are redundant and can be skipped
/// if the representative fails the ablation test.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateClusterJson {
    /// Source UUID of the representative (highest-improvement) candidate.
    pub representative_from_uuid: String,
    /// Target UUID of the representative candidate.
    pub representative_to_uuid: String,
    /// Expected improvement of the representative candidate.
    pub representative_improvement: f32,
    /// Total number of candidates in this cluster.
    pub member_count: usize,
    /// Source UUIDs of all candidates in this cluster (ordered by improvement, best first).
    pub member_from_uuids: Vec<String>,
    /// Similarity score within the cluster (0.0 to 1.0).
    /// Higher values mean the candidates are more likely to share the same outcome.
    pub internal_correlation: f32,
}

/// Cluster candidates by target neuron, source type, and improvement similarity.
///
/// Returns a list of clusters, each containing at least [`MIN_CLUSTER_SIZE`] members.
/// Clusters are sorted by representative improvement (best first).
///
/// # Arguments
/// * `candidates` - Slice of clusterable candidates to group.
///
/// # Returns
/// A list of [`CandidateClusterJson`] for groups with ≥ 2 members.
pub fn cluster_candidates(candidates: &[ClusterableCandidate]) -> Vec<CandidateClusterJson> {
    if candidates.len() < MIN_CLUSTER_SIZE {
        return Vec::new();
    }

    // Step 1: Group by (to_neuron_uuid, neuron_type)
    let mut groups: HashMap<(&str, &str), Vec<&ClusterableCandidate>> = HashMap::new();
    for c in candidates {
        groups
            .entry((c.to_neuron_uuid.as_str(), c.neuron_type.as_str()))
            .or_default()
            .push(c);
    }

    let mut clusters: Vec<CandidateClusterJson> = Vec::new();

    for group in groups.values() {
        if group.len() < MIN_CLUSTER_SIZE {
            continue;
        }

        // Step 2: Sort by improvement (best first) within each group
        let mut sorted: Vec<&ClusterableCandidate> = group.clone();
        sorted.sort_by(|a, b| {
            b.expected_improvement
                .partial_cmp(&a.expected_improvement)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 3: Sub-cluster by improvement similarity.
        // Walk through sorted candidates and split when the ratio between the
        // current best and the next candidate exceeds the threshold.
        let sub_clusters = split_by_improvement_similarity(&sorted);

        for sub in sub_clusters {
            if sub.len() < MIN_CLUSTER_SIZE {
                continue;
            }

            let representative = sub[0];
            let internal_correlation = compute_internal_correlation(&sub);

            clusters.push(CandidateClusterJson {
                representative_from_uuid: representative.from_neuron_uuid.clone(),
                representative_to_uuid: representative.to_neuron_uuid.clone(),
                representative_improvement: representative.expected_improvement,
                member_count: sub.len(),
                member_from_uuids: sub.iter().map(|c| c.from_neuron_uuid.clone()).collect(),
                internal_correlation,
            });
        }
    }

    // Sort clusters by representative improvement (best first)
    clusters.sort_by(|a, b| {
        b.representative_improvement
            .partial_cmp(&a.representative_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    clusters
}

/// Split a sorted (best-first) group of candidates into sub-clusters where
/// all members have similar improvement values.
///
/// Uses a ratio-based criterion: if the best improvement in the current sub-cluster
/// is more than [`IMPROVEMENT_SIMILARITY_RATIO`]× the worst, a new sub-cluster starts.
fn split_by_improvement_similarity<'a>(
    sorted: &[&'a ClusterableCandidate],
) -> Vec<Vec<&'a ClusterableCandidate>> {
    if sorted.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<Vec<&'a ClusterableCandidate>> = Vec::new();
    let mut current_sub: Vec<&'a ClusterableCandidate> = vec![sorted[0]];
    let mut current_best = sorted[0].expected_improvement.abs().max(f32::EPSILON);

    for &c in &sorted[1..] {
        let c_improvement = c.expected_improvement.abs().max(f32::EPSILON);

        // If the ratio between the sub-cluster's best and this candidate is too large,
        // start a new sub-cluster.
        if current_best / c_improvement > IMPROVEMENT_SIMILARITY_RATIO {
            result.push(std::mem::take(&mut current_sub));
            current_best = c_improvement;
        }

        current_sub.push(c);
    }

    if !current_sub.is_empty() {
        result.push(current_sub);
    }

    result
}

/// Compute the internal correlation score for a sub-cluster.
///
/// Uses the coefficient of variation (CV) of improvements to estimate how
/// similar the candidates are. A low CV (< 0.1) yields correlation ≈ 1.0;
/// a high CV (> 1.0) yields correlation ≈ 0.0.
///
/// Formula: `correlation = 1.0 - min(cv, 1.0)` where `cv = std_dev / mean`.
fn compute_internal_correlation(candidates: &[&ClusterableCandidate]) -> f32 {
    if candidates.len() < 2 {
        return 1.0;
    }

    let improvements: Vec<f32> = candidates
        .iter()
        .map(|c| c.expected_improvement.abs())
        .collect();

    let n = improvements.len() as f32;
    let mean = improvements.iter().sum::<f32>() / n;

    if mean < f32::EPSILON {
        // All near-zero — treat as perfectly correlated
        return 1.0;
    }

    let variance = improvements
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>()
        / n;
    let std_dev = variance.sqrt();
    let cv = std_dev / mean;

    // Map CV to correlation: CV=0 → 1.0, CV≥1 → 0.0
    (1.0 - cv.min(1.0)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_by_improvement_similarity_basic() {
        let c1 = ClusterableCandidate {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "t".to_string(),
            expected_improvement: 0.10,
            neuron_type: "input".to_string(),
        };
        let c2 = ClusterableCandidate {
            from_neuron_uuid: "b".to_string(),
            to_neuron_uuid: "t".to_string(),
            expected_improvement: 0.09,
            neuron_type: "input".to_string(),
        };

        let sorted = vec![&c1, &c2];
        let subs = split_by_improvement_similarity(&sorted);

        assert_eq!(subs.len(), 1, "Similar improvements should stay together");
        assert_eq!(subs[0].len(), 2);
    }

    #[test]
    fn test_compute_internal_correlation_identical() {
        let c1 = ClusterableCandidate {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "t".to_string(),
            expected_improvement: 0.10,
            neuron_type: "input".to_string(),
        };
        let c2 = ClusterableCandidate {
            from_neuron_uuid: "b".to_string(),
            to_neuron_uuid: "t".to_string(),
            expected_improvement: 0.10,
            neuron_type: "input".to_string(),
        };

        let candidates = vec![&c1, &c2];
        let corr = compute_internal_correlation(&candidates);

        assert!(
            (corr - 1.0).abs() < f32::EPSILON,
            "Identical improvements should yield correlation ≈ 1.0"
        );
    }
}
