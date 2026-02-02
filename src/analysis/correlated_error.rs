//! Correlated error pattern detection module (Issue #344).
//!
//! Identifies groups of output neurons that exhibit correlated error patterns across
//! samples, suggesting they share a common missing cause. Recommends structural changes
//! that address the shared cause rather than treating each output independently.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Correlated Error Pattern Detection" for full documentation.
//!
//! ## Detection Method
//!
//! 1. **Compute error correlation matrix**: For each pair of output neurons, compute the
//!    Pearson correlation of their per-sample errors.
//! 2. **Cluster correlated outputs**: Group outputs with correlation > threshold (0.7).
//! 3. **Identify shared error samples**: Find samples where all neurons in a cluster err
//!    in the same direction.
//! 4. **Find predictive inputs**: Identify which input neuron activations predict the
//!    shared error pattern.
//!
//! ## Recommended Actions
//!
//! When correlated errors are detected, we recommend:
//! 1. **Add shared hidden neuron**: A new neuron connecting predictive inputs to all outputs
//!    in the correlated group, addressing the shared missing cause.
//!
//! These are emitted as `CoordinatedStructuralCandidateJson` with `AddNeuron` and
//! `AddSynapse` operations.

use std::collections::{HashMap, HashSet};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum samples required for reliable correlation estimation.
const MIN_SAMPLES_FOR_CORRELATION: usize = 20;

/// Minimum Pearson correlation to consider two outputs as correlated.
const CORRELATION_THRESHOLD: f32 = 0.7;

/// Minimum absolute correlation between an input activation and the shared error
/// to consider the input "predictive".
const PREDICTIVE_INPUT_THRESHOLD: f32 = 0.4;

/// Minimum fraction of samples where the group's errors share the same sign
/// to count as "shared error samples".
const SHARED_ERROR_SIGN_THRESHOLD: f32 = 0.0;

/// Result of detecting a correlated error group among output neurons.
#[derive(Debug, Clone)]
pub struct CorrelatedErrorGroup {
    /// UUIDs of output neurons in this correlated group.
    pub output_neuron_uuids: Vec<String>,
    /// Mean pairwise Pearson correlation within the group.
    pub mean_correlation: f32,
    /// Number of samples where all neurons in the group err in the same direction.
    pub shared_error_sample_count: usize,
    /// Total number of samples analysed.
    pub total_sample_count: usize,
    /// UUIDs of input neurons whose activations predict the shared error pattern.
    pub predictive_input_uuids: Vec<String>,
    /// Estimated creature score improvement from addressing the shared cause.
    pub estimated_improvement: f32,
}

/// Detect correlated error patterns among output neurons.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations
///   and errors. Should include both output and input neuron records.
///
/// # Returns
/// A list of `CorrelatedErrorGroup` for groups of output neurons with correlated errors,
/// sorted by estimated improvement (best first).
pub fn detect_correlated_error_patterns(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<CorrelatedErrorGroup> {
    // Identify output neuron UUIDs
    let output_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    // Skip if only one output neuron — nothing to correlate
    if output_uuids.len() < 2 {
        return Vec::new();
    }

    // Identify input neuron UUIDs (for predictive input analysis)
    let input_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();

    // Build records lookup
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    // Collect output neurons that have sufficient records with errors
    let mut output_neurons_with_errors: Vec<&str> = Vec::new();
    for uuid in &output_uuids {
        if let Some(records) = records_map.get(uuid) {
            if records.len() >= MIN_SAMPLES_FOR_CORRELATION
                && records.iter().any(|r| !r.errors.is_empty())
            {
                output_neurons_with_errors.push(uuid);
            }
        }
    }
    output_neurons_with_errors.sort(); // deterministic ordering

    if output_neurons_with_errors.len() < 2 {
        return Vec::new();
    }

    // Build per-sample error vectors for each output neuron, indexed by obs_index.
    // Use the first (primary) error value for correlation.
    let mut error_by_obs: HashMap<&str, HashMap<u32, f32>> = HashMap::new();
    for &uuid in &output_neurons_with_errors {
        if let Some(records) = records_map.get(uuid) {
            let mut obs_map = HashMap::new();
            for r in records.iter() {
                if let Some(&err) = r.errors.first() {
                    obs_map.insert(r.obs_index, err);
                }
            }
            error_by_obs.insert(uuid, obs_map);
        }
    }

    // Find shared obs_indices across all output pairs
    // Build pairwise correlation matrix
    let n_outputs = output_neurons_with_errors.len();
    let mut correlation_matrix: Vec<Vec<f32>> = vec![vec![0.0; n_outputs]; n_outputs];

    for i in 0..n_outputs {
        correlation_matrix[i][i] = 1.0; // self-correlation
        for j in (i + 1)..n_outputs {
            let uuid_i = output_neurons_with_errors[i];
            let uuid_j = output_neurons_with_errors[j];

            let corr = compute_pearson_correlation(
                error_by_obs.get(uuid_i).unwrap(),
                error_by_obs.get(uuid_j).unwrap(),
            );

            correlation_matrix[i][j] = corr;
            correlation_matrix[j][i] = corr;
        }
    }

    // Cluster correlated outputs using greedy single-linkage:
    // Start from each pair above threshold, merge into groups where all pairwise
    // correlations exceed the threshold (complete-linkage criterion).
    let groups = cluster_correlated_outputs(
        &output_neurons_with_errors,
        &correlation_matrix,
        CORRELATION_THRESHOLD,
    );

    // For each group, compute statistics and find predictive inputs
    let mut results: Vec<CorrelatedErrorGroup> = Vec::new();

    for group_indices in &groups {
        let group_uuids: Vec<&str> = group_indices
            .iter()
            .map(|&i| output_neurons_with_errors[i])
            .collect();

        // Compute mean pairwise correlation within the group
        let mut corr_sum = 0.0;
        let mut corr_count = 0;
        for (pos_i, &i) in group_indices.iter().enumerate() {
            for &j in group_indices.iter().skip(pos_i + 1) {
                corr_sum += correlation_matrix[i][j];
                corr_count += 1;
            }
        }
        let mean_correlation = if corr_count > 0 {
            corr_sum / corr_count as f32
        } else {
            0.0
        };

        // Find shared obs_indices across all neurons in the group
        let shared_obs = find_shared_obs_indices(&group_uuids, &error_by_obs);
        let total_sample_count = shared_obs.len();

        if total_sample_count < MIN_SAMPLES_FOR_CORRELATION {
            continue;
        }

        // Count samples where all neurons err in the same direction
        let shared_error_sample_count =
            count_shared_error_samples(&group_uuids, &error_by_obs, &shared_obs);

        if (shared_error_sample_count as f32 / total_sample_count as f32)
            <= SHARED_ERROR_SIGN_THRESHOLD
        {
            continue;
        }

        // Find predictive input neurons
        let predictive_inputs = find_predictive_inputs(
            &group_uuids,
            &input_uuids,
            &records_map,
            &error_by_obs,
            &shared_obs,
        );

        // Estimate improvement: based on correlation strength, group size, and error magnitude
        let mean_abs_error = compute_mean_abs_error(&group_uuids, &error_by_obs, &shared_obs);
        let estimated_improvement =
            mean_correlation * mean_abs_error * (group_uuids.len() as f32) * 0.01;

        results.push(CorrelatedErrorGroup {
            output_neuron_uuids: group_uuids.iter().map(|s| s.to_string()).collect(),
            mean_correlation,
            shared_error_sample_count,
            total_sample_count,
            predictive_input_uuids: predictive_inputs,
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    results.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Compute Pearson correlation between two error vectors indexed by obs_index.
fn compute_pearson_correlation(errors_a: &HashMap<u32, f32>, errors_b: &HashMap<u32, f32>) -> f32 {
    // Find shared obs_indices
    let shared: Vec<u32> = errors_a
        .keys()
        .filter(|k| errors_b.contains_key(k))
        .copied()
        .collect();

    let n = shared.len();
    if n < MIN_SAMPLES_FOR_CORRELATION {
        return 0.0;
    }

    let n_f = n as f32;

    // Compute means
    let mean_a: f32 = shared.iter().map(|k| errors_a[k]).sum::<f32>() / n_f;
    let mean_b: f32 = shared.iter().map(|k| errors_b[k]).sum::<f32>() / n_f;

    // Compute covariance and standard deviations
    let mut cov = 0.0_f32;
    let mut var_a = 0.0_f32;
    let mut var_b = 0.0_f32;

    for &k in &shared {
        let da = errors_a[&k] - mean_a;
        let db = errors_b[&k] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 {
        return 0.0; // No variance — undefined correlation
    }

    cov / denom
}

/// Cluster correlated outputs using complete-linkage clustering.
///
/// Returns groups of indices into `output_uuids` where all pairwise correlations
/// within each group exceed `threshold`.
fn cluster_correlated_outputs(
    output_uuids: &[&str],
    correlation_matrix: &[Vec<f32>],
    threshold: f32,
) -> Vec<Vec<usize>> {
    let n = output_uuids.len();
    let mut assigned: Vec<bool> = vec![false; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for i in 0..n {
        if assigned[i] {
            continue;
        }

        // Start a new potential group with neuron i
        let mut group = vec![i];
        assigned[i] = true;

        // Try to add other unassigned neurons that correlate with ALL current members
        for j in (i + 1)..n {
            if assigned[j] {
                continue;
            }

            // Check complete-linkage: j must correlate above threshold with all in group
            let all_correlated = group
                .iter()
                .all(|&member| correlation_matrix[member][j] >= threshold);

            if all_correlated {
                group.push(j);
                assigned[j] = true;
            }
        }

        // Only keep groups with at least 2 members
        if group.len() >= 2 {
            groups.push(group);
        } else {
            // Unassign the single neuron so it can be picked up by another group
            assigned[i] = false;
        }
    }

    groups
}

/// Find obs_indices shared across all neurons in the group.
fn find_shared_obs_indices(
    group_uuids: &[&str],
    error_by_obs: &HashMap<&str, HashMap<u32, f32>>,
) -> Vec<u32> {
    if group_uuids.is_empty() {
        return Vec::new();
    }

    let first = error_by_obs.get(group_uuids[0]);
    if first.is_none() {
        return Vec::new();
    }

    let mut shared: HashSet<u32> = first.unwrap().keys().copied().collect();

    for &uuid in &group_uuids[1..] {
        if let Some(obs_map) = error_by_obs.get(uuid) {
            shared.retain(|k| obs_map.contains_key(k));
        } else {
            return Vec::new();
        }
    }

    let mut result: Vec<u32> = shared.into_iter().collect();
    result.sort();
    result
}

/// Count samples where all neurons in the group err in the same direction.
fn count_shared_error_samples(
    group_uuids: &[&str],
    error_by_obs: &HashMap<&str, HashMap<u32, f32>>,
    shared_obs: &[u32],
) -> usize {
    shared_obs
        .iter()
        .filter(|&&obs| {
            let errors: Vec<f32> = group_uuids
                .iter()
                .filter_map(|uuid| error_by_obs.get(uuid)?.get(&obs).copied())
                .collect();

            if errors.len() < group_uuids.len() {
                return false;
            }

            // All errors have the same sign (all positive or all negative)
            let all_positive = errors.iter().all(|&e| e > 0.0);
            let all_negative = errors.iter().all(|&e| e < 0.0);
            all_positive || all_negative
        })
        .count()
}

/// Find input neurons whose activations predict the shared error pattern.
///
/// For each input neuron, compute the correlation between its activation and the
/// average error across the correlated group. Inputs with |correlation| > threshold
/// are considered predictive.
fn find_predictive_inputs(
    group_uuids: &[&str],
    input_uuids: &HashSet<&str>,
    records_map: &HashMap<&str, &Vec<DiscoverRecord>>,
    error_by_obs: &HashMap<&str, HashMap<u32, f32>>,
    shared_obs: &[u32],
) -> Vec<String> {
    if shared_obs.is_empty() {
        return Vec::new();
    }

    // Compute average error across the group for each shared obs_index
    let mut avg_error: HashMap<u32, f32> = HashMap::new();
    for &obs in shared_obs {
        let mut sum = 0.0;
        let mut count = 0;
        for &uuid in group_uuids {
            if let Some(err) = error_by_obs.get(uuid).and_then(|m| m.get(&obs)) {
                sum += err;
                count += 1;
            }
        }
        if count > 0 {
            avg_error.insert(obs, sum / count as f32);
        }
    }

    // For each input neuron, compute correlation with the average error
    let mut predictive: Vec<(String, f32)> = Vec::new();

    for &input_uuid in input_uuids {
        let Some(input_records) = records_map.get(input_uuid) else {
            continue;
        };

        // Build activation-by-obs map for this input
        let input_activations: HashMap<u32, f32> = input_records
            .iter()
            .map(|r| (r.obs_index, r.activation))
            .collect();

        // Compute correlation between input activation and average error
        let corr = compute_pearson_correlation_vecs(&input_activations, &avg_error);

        if corr.abs() >= PREDICTIVE_INPUT_THRESHOLD {
            predictive.push((input_uuid.to_string(), corr.abs()));
        }
    }

    // Sort by correlation strength (strongest first)
    predictive.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    predictive.into_iter().map(|(uuid, _)| uuid).collect()
}

/// Compute Pearson correlation between two f32 vectors indexed by u32 keys.
fn compute_pearson_correlation_vecs(vec_a: &HashMap<u32, f32>, vec_b: &HashMap<u32, f32>) -> f32 {
    let shared: Vec<u32> = vec_a
        .keys()
        .filter(|k| vec_b.contains_key(k))
        .copied()
        .collect();

    let n = shared.len();
    if n < MIN_SAMPLES_FOR_CORRELATION {
        return 0.0;
    }

    let n_f = n as f32;

    let mean_a: f32 = shared.iter().map(|k| vec_a[k]).sum::<f32>() / n_f;
    let mean_b: f32 = shared.iter().map(|k| vec_b[k]).sum::<f32>() / n_f;

    let mut cov = 0.0_f32;
    let mut var_a = 0.0_f32;
    let mut var_b = 0.0_f32;

    for &k in &shared {
        let da = vec_a[&k] - mean_a;
        let db = vec_b[&k] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    cov / denom
}

/// Compute mean absolute error across all neurons in the group for shared samples.
fn compute_mean_abs_error(
    group_uuids: &[&str],
    error_by_obs: &HashMap<&str, HashMap<u32, f32>>,
    shared_obs: &[u32],
) -> f32 {
    if shared_obs.is_empty() || group_uuids.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0_f32;
    let mut count = 0_usize;

    for &obs in shared_obs {
        for &uuid in group_uuids {
            if let Some(err) = error_by_obs.get(uuid).and_then(|m| m.get(&obs)) {
                sum += err.abs();
                count += 1;
            }
        }
    }

    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}

/// Generate a deterministic UUID for a shared hidden neuron.
fn shared_neuron_uuid(group_uuids: &[String], index: usize) -> String {
    let mut key = format!("correlated-shared|{index}");
    for uuid in group_uuids {
        key.push('|');
        key.push_str(uuid);
    }
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("cs-{hash:016x}")
}

/// Convert correlated error groups into coordinated structural candidates.
///
/// Each group produces a candidate that adds a shared hidden neuron connecting
/// predictive inputs to all correlated outputs, addressing the shared missing cause.
///
/// # Arguments
/// * `groups` - Detected correlated error groups.
/// * `creature` - The creature topology (needed for neuron lookup).
pub fn correlated_errors_to_coordinated_candidates(
    groups: &[CorrelatedErrorGroup],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    // Find the first output neuron UUID for insert_before placement
    let first_output_uuid = creature
        .neurons
        .iter()
        .find(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone());

    let mut results = Vec::new();

    for (idx, group) in groups.iter().enumerate() {
        let new_uuid = shared_neuron_uuid(&group.output_neuron_uuids, idx);

        let mut operations = Vec::new();

        // Add the shared hidden neuron (placed before the first output neuron)
        operations.push(CoordinatedStructuralOpJson::AddNeuron {
            neuron_uuid: new_uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
            insert_before_neuron_uuid: first_output_uuid.clone(),
        });

        // Connect predictive inputs to the new shared neuron
        let input_sources: &[String] = if group.predictive_input_uuids.is_empty() {
            // If no predictive inputs found, use the first input neuron as fallback
            &[]
        } else {
            &group.predictive_input_uuids
        };

        for input_uuid in input_sources {
            operations.push(CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: input_uuid.clone(),
                to_neuron_uuid: new_uuid.clone(),
                weight: 0.5,
            });
        }

        // If no predictive inputs, connect from a generic input
        if input_sources.is_empty() {
            if let Some(first_input) = creature.neurons.iter().find(|n| n.neuron_type == "input") {
                operations.push(CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: first_input.uuid.clone(),
                    to_neuron_uuid: new_uuid.clone(),
                    weight: 0.5,
                });
            }
        }

        // Connect the shared neuron to all correlated outputs
        for output_uuid in &group.output_neuron_uuids {
            operations.push(CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: new_uuid.clone(),
                to_neuron_uuid: output_uuid.clone(),
                weight: 0.1, // Start with a small weight — ablation testing will validate
            });
        }

        let output_list = group.output_neuron_uuids.join(", ");
        let input_list = if group.predictive_input_uuids.is_empty() {
            "auto-selected".to_string()
        } else {
            group.predictive_input_uuids.join(", ")
        };

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: group.estimated_improvement,
            comment: Some(format!(
                "Correlated error group [{}]: mean correlation {:.2}, {}/{} shared error samples → add shared hidden neuron from [{}]",
                output_list, group.mean_correlation, group.shared_error_sample_count, group.total_sample_count, input_list
            )),
        });
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── compute_pearson_correlation ─────────────────────────────────────

    #[test]
    fn perfect_positive_correlation() {
        let a: HashMap<u32, f32> = (0..30).map(|i| (i, i as f32)).collect();
        let b: HashMap<u32, f32> = (0..30).map(|i| (i, i as f32 * 2.0)).collect();
        let corr = compute_pearson_correlation(&a, &b);
        assert!(
            (corr - 1.0).abs() < 0.01,
            "Perfectly correlated vectors should have r ≈ 1.0, got {corr}"
        );
    }

    #[test]
    fn perfect_negative_correlation() {
        let a: HashMap<u32, f32> = (0..30).map(|i| (i, i as f32)).collect();
        let b: HashMap<u32, f32> = (0..30).map(|i| (i, -(i as f32))).collect();
        let corr = compute_pearson_correlation(&a, &b);
        assert!(
            (corr + 1.0).abs() < 0.01,
            "Perfectly anti-correlated vectors should have r ≈ -1.0, got {corr}"
        );
    }

    #[test]
    fn insufficient_shared_samples_returns_zero() {
        let a: HashMap<u32, f32> = (0..5).map(|i| (i, i as f32)).collect();
        let b: HashMap<u32, f32> = (0..5).map(|i| (i, i as f32)).collect();
        let corr = compute_pearson_correlation(&a, &b);
        assert_eq!(corr, 0.0, "Should return 0.0 with insufficient samples");
    }

    #[test]
    fn no_overlap_returns_zero() {
        let a: HashMap<u32, f32> = (0..30).map(|i| (i, i as f32)).collect();
        let b: HashMap<u32, f32> = (100..130).map(|i| (i, i as f32)).collect();
        let corr = compute_pearson_correlation(&a, &b);
        assert_eq!(corr, 0.0, "No overlapping obs indices should return 0.0");
    }

    #[test]
    fn zero_variance_returns_zero() {
        let a: HashMap<u32, f32> = (0..30).map(|i| (i, 5.0)).collect();
        let b: HashMap<u32, f32> = (0..30).map(|i| (i, 5.0)).collect();
        let corr = compute_pearson_correlation(&a, &b);
        assert_eq!(
            corr, 0.0,
            "Zero variance should return 0.0 (undefined correlation)"
        );
    }

    // ── cluster_correlated_outputs ─────────────────────────────────────

    #[test]
    fn two_highly_correlated_outputs_form_one_group() {
        let output_uuids = vec!["out-1", "out-2"];
        let corr_matrix = vec![vec![1.0, 0.9], vec![0.9, 1.0]];
        let groups = cluster_correlated_outputs(&output_uuids, &corr_matrix, 0.7);
        assert_eq!(groups.len(), 1, "Should form one group");
        assert_eq!(groups[0].len(), 2, "Group should contain both outputs");
    }

    #[test]
    fn independent_outputs_form_no_group() {
        let output_uuids = vec!["out-1", "out-2"];
        let corr_matrix = vec![vec![1.0, 0.1], vec![0.1, 1.0]];
        let groups = cluster_correlated_outputs(&output_uuids, &corr_matrix, 0.7);
        assert!(
            groups.is_empty(),
            "Independent outputs should form no groups"
        );
    }

    // ── count_shared_error_samples ─────────────────────────────────────

    #[test]
    fn all_same_sign_errors_counted() {
        let group_uuids = vec!["out-1", "out-2"];
        let error_by_obs: HashMap<&str, HashMap<u32, f32>> = [
            ("out-1", [(0_u32, 1.0), (1, 2.0)].into_iter().collect()),
            ("out-2", [(0_u32, 0.5), (1, 1.5)].into_iter().collect()),
        ]
        .into_iter()
        .collect();
        let shared_obs = vec![0, 1];
        let count = count_shared_error_samples(&group_uuids, &error_by_obs, &shared_obs);
        assert_eq!(count, 2, "Both samples have all-positive errors");
    }

    #[test]
    fn mixed_sign_errors_not_counted() {
        let group_uuids = vec!["out-1", "out-2"];
        let error_by_obs: HashMap<&str, HashMap<u32, f32>> = [
            ("out-1", [(0_u32, 1.0)].into_iter().collect()),
            ("out-2", [(0_u32, -0.5)].into_iter().collect()),
        ]
        .into_iter()
        .collect();
        let shared_obs = vec![0];
        let count = count_shared_error_samples(&group_uuids, &error_by_obs, &shared_obs);
        assert_eq!(count, 0, "Mixed-sign errors should not be counted");
    }

    // ── shared_neuron_uuid ─────────────────────────────────────────────

    #[test]
    fn shared_neuron_uuid_is_deterministic() {
        let uuids = vec!["out-1".to_string(), "out-2".to_string()];
        let id1 = shared_neuron_uuid(&uuids, 0);
        let id2 = shared_neuron_uuid(&uuids, 0);
        assert_eq!(id1, id2);
    }

    #[test]
    fn shared_neuron_uuid_has_cs_prefix() {
        let uuids = vec!["out-1".to_string()];
        let id = shared_neuron_uuid(&uuids, 0);
        assert!(id.starts_with("cs-"), "Should start with 'cs-' prefix");
    }
}
