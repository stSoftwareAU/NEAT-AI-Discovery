//! Cross-detection candidate synthesis for co-flagged neurons (Issue #963).
//!
//! When multiple detection modules independently flag the same neuron,
//! this module synthesises combined remediation candidates that address
//! multiple issues simultaneously rather than generating independent
//! single-issue candidates.
//!
//! ## Synthesis Rules
//!
//! Compatible operation combinations that produce synthesised candidates:
//! - `ChangeSquash` + `SetBias` → single coordinated candidate (e.g., saturation + restricted range)
//! - `ChangeSquash` + `SetWeight` → single coordinated candidate (e.g., saturation + weight magnitude)
//! - `RemoveNeuron` + `RemoveSynapse` → single coordinated candidate (e.g., dead neuron + dormant synapse)
//! - `SetBias` + `SetWeight` → single coordinated candidate (e.g., bias perturbation + weight issue)
//!
//! Incompatible combinations (not synthesised):
//! - `RemoveNeuron` + any modification (`ChangeSquash`, `SetBias`, `SetWeight`)
//!   — cannot modify a neuron that is being removed
//!
//! ## Integration
//!
//! Called from `orchestration::analyze_all` after discovery modules run and before
//! ensemble scoring. Synthesised candidates are added alongside (not replacing)
//! the individual candidates from each module.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use std::collections::HashMap;

use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Result of cross-detection candidate synthesis.
pub struct SynthesisResult {
    /// All candidates: original individuals plus any synthesised combined candidates.
    pub candidates: Vec<CoordinatedStructuralCandidateJson>,
    /// Number of synthesised (combined) candidates that were added.
    pub synthesised_count: usize,
}

/// Extract the primary neuron UUID targeted by a single-operation candidate.
///
/// Returns `None` for multi-operation candidates (which are already coordinated).
fn primary_neuron_uuid(candidate: &CoordinatedStructuralCandidateJson) -> Option<String> {
    if candidate.operations.len() != 1 {
        return None;
    }

    match &candidate.operations[0] {
        CoordinatedStructuralOpJson::ChangeSquash { neuron_uuid, .. }
        | CoordinatedStructuralOpJson::SetBias { neuron_uuid, .. }
        | CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid }
        | CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. } => Some(neuron_uuid.clone()),
        CoordinatedStructuralOpJson::SetWeight { to_neuron_uuid, .. } => {
            Some(to_neuron_uuid.clone())
        }
        CoordinatedStructuralOpJson::RemoveSynapse { to_neuron_uuid, .. } => {
            Some(to_neuron_uuid.clone())
        }
        CoordinatedStructuralOpJson::AddSynapse { to_neuron_uuid, .. } => {
            Some(to_neuron_uuid.clone())
        }
    }
}

/// Group single-operation candidates by their target neuron UUID.
///
/// Multi-operation candidates (already coordinated) are excluded from grouping.
/// Returns a map from neuron UUID to the indices of candidates targeting that neuron.
pub fn group_candidates_by_neuron(
    candidates: &[CoordinatedStructuralCandidateJson],
) -> HashMap<String, Vec<usize>> {
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, candidate) in candidates.iter().enumerate() {
        if let Some(uuid) = primary_neuron_uuid(candidate) {
            groups.entry(uuid).or_default().push(i);
        }
    }

    groups
}

/// Classify a single operation as either "removal" or "modification".
fn is_removal_op(op: &CoordinatedStructuralOpJson) -> bool {
    matches!(
        op,
        CoordinatedStructuralOpJson::RemoveNeuron { .. }
            | CoordinatedStructuralOpJson::RemoveSynapse { .. }
    )
}

/// Check whether a set of operations are compatible for synthesis.
///
/// Rules:
/// - All removals are compatible with each other.
/// - All modifications are compatible with each other.
/// - Removals are NOT compatible with modifications (cannot modify a removed neuron).
fn are_operations_compatible(ops: &[&CoordinatedStructuralOpJson]) -> bool {
    if ops.len() <= 1 {
        return true;
    }

    let has_removal = ops.iter().any(|op| is_removal_op(op));
    let has_modification = ops.iter().any(|op| !is_removal_op(op));

    // Incompatible: mixing removals with modifications
    !(has_removal && has_modification)
}

/// Extract the module name from a candidate's comment for attribution.
fn extract_module_name(candidate: &CoordinatedStructuralCandidateJson) -> String {
    candidate.comment.as_ref().map_or_else(
        || "unknown".to_string(),
        |c| {
            c.split(&[':', '|'][..])
                .next()
                .unwrap_or("unknown")
                .trim()
                .to_string()
        },
    )
}

/// Synthesise cross-detection candidates for co-flagged neurons (Issue #963).
///
/// Groups single-operation candidates by their target neuron UUID. When 2+
/// detection modules flag the same neuron with compatible operations, a
/// combined `CoordinatedStructuralCandidateJson` is synthesised containing
/// all operations.
///
/// Individual candidates are always preserved — synthesis is additive.
pub fn synthesise_cross_detection_candidates(
    candidates: Vec<CoordinatedStructuralCandidateJson>,
) -> SynthesisResult {
    if candidates.is_empty() {
        return SynthesisResult {
            candidates: Vec::new(),
            synthesised_count: 0,
        };
    }

    let groups = group_candidates_by_neuron(&candidates);

    let mut synthesised: Vec<CoordinatedStructuralCandidateJson> = Vec::new();

    for indices in groups.values() {
        // Only synthesise when 2+ modules flag the same neuron
        if indices.len() < 2 {
            continue;
        }

        // Collect the single operations from each candidate in this group
        let ops: Vec<&CoordinatedStructuralOpJson> = indices
            .iter()
            .map(|&i| &candidates[i].operations[0])
            .collect();

        // Check compatibility before merging
        if !are_operations_compatible(&ops) {
            continue;
        }

        // Build the synthesised candidate with all operations combined
        let combined_ops: Vec<CoordinatedStructuralOpJson> = indices
            .iter()
            .map(|&i| candidates[i].operations[0].clone())
            .collect();

        // Use the best individual gain as the base for the synthesised candidate
        let best_gain = indices
            .iter()
            .map(|&i| candidates[i].expected_creature_score_gain)
            .fold(f32::NEG_INFINITY, f32::max);

        // Collect module names for attribution
        let module_names: Vec<String> = indices
            .iter()
            .map(|&i| extract_module_name(&candidates[i]))
            .collect();
        let modules_str = module_names.join(" + ");

        let comment = format!(
            "cross-detection synthesis: combined {} modules ({modules_str})",
            indices.len()
        );

        synthesised.push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: combined_ops,
            expected_creature_score_gain: best_gain,
            comment: Some(comment),
        });
    }

    let synthesised_count = synthesised.len();

    // Preserve all original candidates and append synthesised ones
    let mut result = candidates;
    result.extend(synthesised);

    SynthesisResult {
        candidates: result,
        synthesised_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_neuron_uuid_for_change_squash() {
        let c = CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: "abc".to_string(),
                squash: "RELU".to_string(),
            }],
            expected_creature_score_gain: 0.01,
            comment: None,
        };
        assert_eq!(primary_neuron_uuid(&c), Some("abc".to_string()));
    }

    #[test]
    fn primary_neuron_uuid_for_set_weight_uses_target() {
        let c = CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid: "source".to_string(),
                to_neuron_uuid: "target".to_string(),
                weight: 0.5,
            }],
            expected_creature_score_gain: 0.01,
            comment: None,
        };
        assert_eq!(primary_neuron_uuid(&c), Some("target".to_string()));
    }

    #[test]
    fn primary_neuron_uuid_none_for_multi_op() {
        let c = CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![
                CoordinatedStructuralOpJson::ChangeSquash {
                    neuron_uuid: "a".to_string(),
                    squash: "RELU".to_string(),
                },
                CoordinatedStructuralOpJson::SetBias {
                    neuron_uuid: "a".to_string(),
                    bias: 0.1,
                },
            ],
            expected_creature_score_gain: 0.01,
            comment: None,
        };
        assert_eq!(primary_neuron_uuid(&c), None);
    }

    #[test]
    fn compatible_modifications() {
        let ops = [
            CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: "a".to_string(),
                squash: "RELU".to_string(),
            },
            CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: "a".to_string(),
                bias: 0.1,
            },
        ];
        let refs: Vec<&CoordinatedStructuralOpJson> = ops.iter().collect();
        assert!(are_operations_compatible(&refs));
    }

    #[test]
    fn compatible_removals() {
        let ops = [
            CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: "a".to_string(),
            },
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "b".to_string(),
                to_neuron_uuid: "a".to_string(),
            },
        ];
        let refs: Vec<&CoordinatedStructuralOpJson> = ops.iter().collect();
        assert!(are_operations_compatible(&refs));
    }

    #[test]
    fn incompatible_removal_with_modification() {
        let ops = [
            CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: "a".to_string(),
            },
            CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: "a".to_string(),
                squash: "RELU".to_string(),
            },
        ];
        let refs: Vec<&CoordinatedStructuralOpJson> = ops.iter().collect();
        assert!(!are_operations_compatible(&refs));
    }
}
