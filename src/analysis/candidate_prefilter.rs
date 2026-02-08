//! Candidate pre-filtering module (Issue #429).
//!
//! Extends the early termination system beyond GPU evaluation to the candidate
//! generation pipeline. Provides four improvements:
//!
//! 1. **Hierarchical candidate filtering** — removes candidates with near-zero or
//!    negative expected improvement before they reach the controller.
//! 2. **Budget-aware prioritisation** — limits the number of candidates returned
//!    when a budget is specified, keeping only the highest-value ones.
//! 3. **Incremental confidence** — sorts candidates by expected improvement so
//!    the best candidates are evaluated first by the controller.
//! 4. **Cross-module deduplication** — when multiple discovery modules propose
//!    similar candidates targeting the same neuron with the same operation
//!    signature, keeps only the best one per group.

use crate::CoordinatedStructuralCandidateJson;
use crate::CoordinatedStructuralOpJson;
use std::collections::HashMap;

/// Configuration for candidate pre-filtering.
#[derive(Debug, Clone)]
pub struct CandidatePrefilterConfig {
    /// Minimum expected improvement for a candidate to survive filtering.
    /// Candidates below this threshold are considered low-value noise.
    pub min_improvement_threshold: f32,

    /// Maximum number of candidates to return. `None` means no limit.
    pub max_candidates: Option<usize>,
}

impl Default for CandidatePrefilterConfig {
    fn default() -> Self {
        Self {
            // Conservative threshold — only removes clearly worthless candidates.
            // 0.001 means the candidate must predict at least 0.1% improvement.
            min_improvement_threshold: 0.001,
            max_candidates: None,
        }
    }
}

/// Classify a single operation by its type name.
fn classify_single_op(op: &CoordinatedStructuralOpJson) -> &'static str {
    match op {
        CoordinatedStructuralOpJson::RemoveSynapse { .. } => "remove_synapse",
        CoordinatedStructuralOpJson::AddSynapse { .. } => "add_synapse",
        CoordinatedStructuralOpJson::AddNeuron { .. } => "add_neuron",
        CoordinatedStructuralOpJson::RemoveNeuron { .. } => "remove_neuron",
        CoordinatedStructuralOpJson::ChangeSquash { .. } => "change_squash",
        CoordinatedStructuralOpJson::SetBias { .. } => "set_bias",
        CoordinatedStructuralOpJson::SetWeight { .. } => "set_weight",
    }
}

/// Classify a single operation with detail for deduplication.
///
/// Returns a string that includes the operation type and any distinguishing
/// detail (e.g., activation function for AddNeuron, source UUID for synapse
/// operations) so that structurally different candidates are not collapsed.
fn classify_op_with_detail(op: &CoordinatedStructuralOpJson) -> String {
    match op {
        CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid,
            to_neuron_uuid,
        } => format!("remove_synapse({from_neuron_uuid}->{to_neuron_uuid})"),
        CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid,
            to_neuron_uuid,
            ..
        } => format!("add_synapse({from_neuron_uuid}->{to_neuron_uuid})"),
        CoordinatedStructuralOpJson::AddNeuron { squash, .. } => {
            format!("add_neuron({squash})")
        }
        CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } => {
            format!("remove_neuron({neuron_uuid})")
        }
        CoordinatedStructuralOpJson::ChangeSquash {
            neuron_uuid,
            squash,
        } => format!("change_squash({neuron_uuid},{squash})"),
        CoordinatedStructuralOpJson::SetBias { neuron_uuid, .. } => {
            format!("set_bias({neuron_uuid})")
        }
        CoordinatedStructuralOpJson::SetWeight {
            from_neuron_uuid,
            to_neuron_uuid,
            ..
        } => format!("set_weight({from_neuron_uuid}->{to_neuron_uuid})"),
    }
}

/// Build an operation signature for deduplication.
///
/// For single-operation candidates, this is the operation type with detail.
/// For coordinated (multi-operation) candidates, this concatenates all
/// operation details to form a composite signature so that structurally
/// different candidates (e.g., adding a ReLU vs a LOGISTIC neuron) are
/// not incorrectly deduplicated.
fn operation_signature(ops: &[CoordinatedStructuralOpJson]) -> String {
    if ops.is_empty() {
        return "unknown".to_string();
    }
    if ops.len() == 1 {
        return classify_single_op(&ops[0]).to_string();
    }
    // Multi-operation: join sorted detailed operation descriptions.
    let mut op_details: Vec<String> = ops.iter().map(classify_op_with_detail).collect();
    op_details.sort_unstable();
    op_details.join("+")
}

/// Extract the primary target neuron UUID from a candidate's operations.
///
/// The "target" is the neuron being acted upon:
/// - For remove/add synapse: the `to_neuron_uuid`
/// - For add/remove neuron: the `neuron_uuid` or `insert_before_neuron_uuid`
/// - For set bias/change squash: the `neuron_uuid`
/// - For set weight: the `to_neuron_uuid`
fn extract_target_uuid(ops: &[CoordinatedStructuralOpJson]) -> Option<&str> {
    if ops.is_empty() {
        return None;
    }
    match &ops[0] {
        CoordinatedStructuralOpJson::RemoveSynapse { to_neuron_uuid, .. } => {
            Some(to_neuron_uuid.as_str())
        }
        CoordinatedStructuralOpJson::AddSynapse { to_neuron_uuid, .. } => {
            Some(to_neuron_uuid.as_str())
        }
        CoordinatedStructuralOpJson::AddNeuron {
            insert_before_neuron_uuid,
            neuron_uuid,
            ..
        } => insert_before_neuron_uuid
            .as_deref()
            .or(Some(neuron_uuid.as_str())),
        CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid, .. } => Some(neuron_uuid.as_str()),
        CoordinatedStructuralOpJson::ChangeSquash { neuron_uuid, .. } => Some(neuron_uuid.as_str()),
        CoordinatedStructuralOpJson::SetBias { neuron_uuid, .. } => Some(neuron_uuid.as_str()),
        CoordinatedStructuralOpJson::SetWeight { to_neuron_uuid, .. } => {
            Some(to_neuron_uuid.as_str())
        }
    }
}

/// Filter out candidates with near-zero or negative expected improvement.
///
/// This is the "hierarchical candidate filtering" step — a quick pre-filter
/// that removes clearly poor candidates before any further processing.
///
/// # Arguments
/// * `candidates` - Slice of candidates to filter.
/// * `config` - Pre-filter configuration with threshold.
///
/// # Returns
/// Candidates that exceed the minimum improvement threshold.
pub fn filter_low_value_candidates(
    candidates: &[CoordinatedStructuralCandidateJson],
    config: &CandidatePrefilterConfig,
) -> Vec<CoordinatedStructuralCandidateJson> {
    candidates
        .iter()
        .filter(|c| c.expected_creature_score_gain >= config.min_improvement_threshold)
        .cloned()
        .collect()
}

/// Deduplicate candidates that target the same neuron with the same operation
/// signature.
///
/// When multiple discovery modules independently propose similar candidates
/// (e.g., two modules both suggest removing a synapse to the same target neuron),
/// only the best candidate per (target, operation_signature) pair is kept.
///
/// Coordinated structural candidates with different operation compositions
/// (e.g., "remove_synapse + add_neuron" vs "remove_synapse" alone) are treated
/// as distinct and not deduplicated against each other.
///
/// # Arguments
/// * `candidates` - Slice of candidates to deduplicate.
/// * `_config` - Pre-filter configuration (reserved for future tuning).
///
/// # Returns
/// Deduplicated candidates, keeping the highest-improvement candidate per group.
pub fn deduplicate_candidates(
    candidates: &[CoordinatedStructuralCandidateJson],
    _config: &CandidatePrefilterConfig,
) -> Vec<CoordinatedStructuralCandidateJson> {
    if candidates.len() <= 1 {
        return candidates.to_vec();
    }

    // Group by (target_uuid, operation_signature) and keep the best per group.
    let mut best_per_group: HashMap<(String, String), &CoordinatedStructuralCandidateJson> =
        HashMap::new();

    for candidate in candidates {
        let op_sig = operation_signature(&candidate.operations);
        let target = extract_target_uuid(&candidate.operations)
            .unwrap_or("unknown")
            .to_string();
        let key = (target, op_sig);

        let entry = best_per_group.entry(key).or_insert(candidate);
        if candidate.expected_creature_score_gain > entry.expected_creature_score_gain {
            *entry = candidate;
        }
    }

    best_per_group.into_values().cloned().collect()
}

/// Apply the full candidate pre-filter pipeline.
///
/// Combines all four improvements in sequence:
/// 1. Filter out low-value candidates (hierarchical filtering)
/// 2. Deduplicate across modules (cross-module deduplication)
/// 3. Sort by expected improvement, best first (incremental confidence)
/// 4. Apply budget limit (budget-aware prioritisation)
///
/// # Arguments
/// * `candidates` - Slice of candidates to process.
/// * `config` - Pre-filter configuration.
///
/// # Returns
/// Filtered, deduplicated, sorted, and budget-limited candidates.
pub fn prefilter_candidates(
    candidates: &[CoordinatedStructuralCandidateJson],
    config: &CandidatePrefilterConfig,
) -> Vec<CoordinatedStructuralCandidateJson> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Step 1: Filter out low-value candidates
    let filtered = filter_low_value_candidates(candidates, config);

    // Step 2: Deduplicate across modules
    let deduped = deduplicate_candidates(&filtered, config);

    // Step 3: Sort by expected improvement (best first)
    let mut sorted = deduped;
    sorted.sort_by(|a, b| {
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 4: Apply budget limit
    if let Some(max) = config.max_candidates {
        sorted.truncate(max);
    }

    sorted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_single_op_variants() {
        assert_eq!(
            classify_single_op(&CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "a".into(),
                to_neuron_uuid: "b".into(),
            }),
            "remove_synapse"
        );
        assert_eq!(
            classify_single_op(&CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: "a".into(),
                bias: 0.1,
            }),
            "set_bias"
        );
    }

    #[test]
    fn test_operation_signature_single() {
        let ops = vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".into(),
            to_neuron_uuid: "b".into(),
        }];
        assert_eq!(operation_signature(&ops), "remove_synapse");
    }

    #[test]
    fn test_operation_signature_multi() {
        let ops = vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "a".into(),
                to_neuron_uuid: "b".into(),
            },
            CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: "n".into(),
                neuron_type: "hidden".into(),
                squash: "ReLU".into(),
                bias: 0.0,
                insert_before_neuron_uuid: None,
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "a".into(),
                to_neuron_uuid: "n".into(),
                weight: 1.0,
            },
        ];
        let sig = operation_signature(&ops);
        // Multi-op signature includes detailed operation info
        assert!(sig.contains("add_neuron(ReLU)"), "sig = {sig}");
        assert!(sig.contains("add_synapse(a->n)"), "sig = {sig}");
        assert!(sig.contains("remove_synapse(a->b)"), "sig = {sig}");
    }

    #[test]
    fn test_operation_signature_differs_by_squash() {
        let ops_relu = vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "a".into(),
                to_neuron_uuid: "b".into(),
            },
            CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: "n1".into(),
                neuron_type: "hidden".into(),
                squash: "ReLU".into(),
                bias: 0.0,
                insert_before_neuron_uuid: None,
            },
        ];
        let ops_logistic = vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "a".into(),
                to_neuron_uuid: "b".into(),
            },
            CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: "n2".into(),
                neuron_type: "hidden".into(),
                squash: "LOGISTIC".into(),
                bias: 0.0,
                insert_before_neuron_uuid: None,
            },
        ];
        assert_ne!(
            operation_signature(&ops_relu),
            operation_signature(&ops_logistic),
            "Different activation functions should produce different signatures"
        );
    }

    #[test]
    fn test_operation_signature_empty() {
        assert_eq!(operation_signature(&[]), "unknown");
    }

    #[test]
    fn test_extract_target_uuid_variants() {
        assert_eq!(
            extract_target_uuid(&[CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "a".into(),
                to_neuron_uuid: "b".into(),
            }]),
            Some("b")
        );
        assert_eq!(
            extract_target_uuid(&[CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: "x".into(),
                bias: 0.1,
            }]),
            Some("x")
        );
        assert_eq!(extract_target_uuid(&[]), None);
    }
}
