//! Ensemble candidate scoring — combine predictions across discovery modules (Issue #572).
//!
//! After all discovery modules have produced candidates, this module identifies
//! candidates targeting the same neuron/synapse from different modules and
//! combines their evidence:
//!
//! - **Agreement boost**: When multiple independent modules suggest the same type
//!   of change for the same target, the combined score is boosted above the best
//!   individual prediction.
//! - **Disagreement penalty**: When modules suggest conflicting changes (e.g. one
//!   says remove a neuron, another says change its squash), both candidates are
//!   penalised.
//! - **Single-module passthrough**: Candidates from only one module are returned
//!   unchanged.
//!
//! ## Weighted averaging
//!
//! When combining scores, per-module historical success rates (from
//! `ModuleOutcomeTracker`) are used as weights. Modules with higher success rates
//! contribute more to the ensemble score.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashMap;

use crate::CoordinatedStructuralCandidateJson;
use crate::CoordinatedStructuralOpJson;

use super::module_weights::ModuleOutcomeTracker;

/// Agreement boost multiplier applied when multiple modules agree.
///
/// The ensemble score is: `weighted_average * (1.0 + AGREEMENT_BOOST_FACTOR * (n - 1) / n)`
/// where n is the number of agreeing modules. This gives a ~15% boost for 2 modules,
/// scaling up with more agreement.
const AGREEMENT_BOOST_FACTOR: f32 = 0.3;

/// Disagreement penalty multiplier applied to conflicting candidates.
///
/// Each conflicting candidate's score is multiplied by this factor.
const DISAGREEMENT_PENALTY: f32 = 0.7;

/// Result of ensemble scoring.
pub struct EnsembleScoringResult {
    /// The scored candidates (ensemble-adjusted where applicable).
    pub candidates: Vec<CoordinatedStructuralCandidateJson>,
    /// Number of candidates that were produced via ensemble combination.
    pub ensemble_candidates: usize,
    /// Number of candidates that were from a single module (unchanged).
    pub single_module_candidates: usize,
}

/// Apply ensemble scoring to a set of coordinated structural candidates.
///
/// Groups candidates by their target (neuron UUID or synapse from→to pair),
/// then:
/// 1. Single-module groups pass through unchanged.
/// 2. Multi-module groups with the same operation type are combined via
///    weighted average with an agreement boost.
/// 3. Multi-module groups with conflicting operation types have each
///    candidate penalised.
pub fn apply_ensemble_scoring(
    candidates: Vec<CoordinatedStructuralCandidateJson>,
    tracker: &ModuleOutcomeTracker,
) -> EnsembleScoringResult {
    if candidates.is_empty() {
        return EnsembleScoringResult {
            candidates: Vec::new(),
            ensemble_candidates: 0,
            single_module_candidates: 0,
        };
    }

    // Group candidates by their primary target key.
    let mut groups: HashMap<String, Vec<CoordinatedStructuralCandidateJson>> = HashMap::new();
    for c in candidates {
        let key = target_key(&c);
        groups.entry(key).or_default().push(c);
    }

    let mut result_candidates: Vec<CoordinatedStructuralCandidateJson> = Vec::new();
    let mut ensemble_count: usize = 0;
    let mut single_count: usize = 0;

    for (_key, group) in groups {
        if group.len() == 1 {
            // Single-module candidate — pass through unchanged.
            single_count += 1;
            result_candidates.extend(group);
        } else {
            // Multi-module group — classify as agreement or disagreement.
            let op_types = classify_operation_types(&group);

            if op_types.len() == 1 {
                // Agreement: all candidates suggest the same type of operation.
                let combined = combine_agreeing_candidates(group, tracker);
                ensemble_count += 1;
                result_candidates.push(combined);
            } else {
                // Disagreement: candidates suggest conflicting operations.
                let penalised = penalise_conflicting_candidates(group);
                single_count += penalised.len();
                result_candidates.extend(penalised);
            }
        }
    }

    // Sort by expected gain descending.
    result_candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    EnsembleScoringResult {
        candidates: result_candidates,
        ensemble_candidates: ensemble_count,
        single_module_candidates: single_count,
    }
}

/// Extract a target key from a candidate's operations.
///
/// For single-operation candidates, uses the target neuron UUID (or from→to
/// for synapse operations). For multi-operation candidates, concatenates all
/// target UUIDs to form a composite key.
fn target_key(candidate: &CoordinatedStructuralCandidateJson) -> String {
    let mut keys: Vec<String> = Vec::with_capacity(candidate.operations.len());

    for op in &candidate.operations {
        let part = match op {
            CoordinatedStructuralOpJson::SetBias { neuron_uuid, .. }
            | CoordinatedStructuralOpJson::ChangeSquash { neuron_uuid, .. }
            | CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid }
            | CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. } => {
                format!("n:{neuron_uuid}")
            }
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                ..
            }
            | CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
            } => {
                format!("s:{from_neuron_uuid}->{to_neuron_uuid}")
            }
            CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid,
                to_neuron_uuid,
                ..
            } => {
                format!("s:{from_neuron_uuid}->{to_neuron_uuid}")
            }
        };
        keys.push(part);
    }

    keys.sort();
    keys.join("|")
}

/// Classify the types of operations in a group of candidates.
///
/// Returns a set of unique operation type names. If all candidates have the
/// same operation type, the set will have exactly one element (agreement).
fn classify_operation_types(group: &[CoordinatedStructuralCandidateJson]) -> Vec<String> {
    let mut types: Vec<String> = Vec::new();

    for candidate in group {
        for op in &candidate.operations {
            let op_type = match op {
                CoordinatedStructuralOpJson::SetBias { .. } => "SetBias",
                CoordinatedStructuralOpJson::ChangeSquash { .. } => "ChangeSquash",
                CoordinatedStructuralOpJson::RemoveNeuron { .. } => "RemoveNeuron",
                CoordinatedStructuralOpJson::AddNeuron { .. } => "AddNeuron",
                CoordinatedStructuralOpJson::AddSynapse { .. } => "AddSynapse",
                CoordinatedStructuralOpJson::RemoveSynapse { .. } => "RemoveSynapse",
                CoordinatedStructuralOpJson::SetWeight { .. } => "SetWeight",
            }
            .to_string();

            if !types.contains(&op_type) {
                types.push(op_type);
            }
        }
    }

    types
}

/// Combine agreeing candidates using weighted scoring with agreement boost.
///
/// The combined score starts from the best individual score, then applies a
/// boost based on the number of agreeing modules. The boost reflects increased
/// confidence when independent modules confirm the same change.
///
/// Per-module success rates from `ModuleOutcomeTracker` influence which
/// candidate's parameters are used as the template (the one from the highest-
/// weighted module).
fn combine_agreeing_candidates(
    group: Vec<CoordinatedStructuralCandidateJson>,
    tracker: &ModuleOutcomeTracker,
) -> CoordinatedStructuralCandidateJson {
    let n = group.len() as f32;

    // Find the best candidate by module-weight-adjusted score.
    let mut best_candidate_idx: usize = 0;
    let mut best_adjusted_score: f64 = f64::NEG_INFINITY;
    let mut best_raw_gain: f32 = f32::NEG_INFINITY;
    let mut module_names: Vec<String> = Vec::new();

    for (i, candidate) in group.iter().enumerate() {
        let module_name = extract_module_name(candidate);
        let weight = if tracker.is_empty() {
            1.0
        } else {
            tracker.module_boost(&module_name).max(0.5)
        };

        let adjusted = weight * candidate.expected_creature_score_gain as f64;
        if adjusted > best_adjusted_score {
            best_adjusted_score = adjusted;
            best_candidate_idx = i;
        }

        if candidate.expected_creature_score_gain > best_raw_gain {
            best_raw_gain = candidate.expected_creature_score_gain;
        }

        module_names.push(module_name);
    }

    // Apply agreement boost to the best raw score.
    // boost = 1.0 + AGREEMENT_BOOST_FACTOR * (n - 1) / n
    // For n=2: 1.15, n=3: 1.20, n=4: 1.225, etc.
    //
    // Issue #732: Cap the boost for multi-operation candidates. Complex coordinated
    // candidates (multiple operations) have compounding prediction uncertainty, so
    // the ensemble agreement boost is reduced proportionally to the operation count.
    let max_ops = group.iter().map(|c| c.operations.len()).max().unwrap_or(1);
    let base_boost = AGREEMENT_BOOST_FACTOR * (n - 1.0) / n;
    let capped_boost = if max_ops > 1 {
        // Scale down boost: 1 op = full boost, 4 ops = ~33% of boost
        base_boost / (max_ops as f32)
    } else {
        base_boost
    };
    let boost = 1.0 + capped_boost;
    let ensemble_score = best_raw_gain * boost;

    // Use the best-weighted candidate as the template, update its score and comment.
    let mut result = group.into_iter().nth(best_candidate_idx).unwrap();
    result.expected_creature_score_gain = ensemble_score;

    let modules_str = module_names.join(", ");
    let ensemble_comment = format!(
        "Ensemble ({} modules agree: {})",
        module_names.len(),
        modules_str
    );
    result.comment = Some(match result.comment {
        Some(existing) => format!("{existing} | {ensemble_comment}"),
        None => ensemble_comment,
    });

    result
}

/// Penalise candidates with conflicting operations on the same target.
///
/// Each candidate's score is reduced by the disagreement penalty factor.
fn penalise_conflicting_candidates(
    group: Vec<CoordinatedStructuralCandidateJson>,
) -> Vec<CoordinatedStructuralCandidateJson> {
    group
        .into_iter()
        .map(|mut c| {
            c.expected_creature_score_gain *= DISAGREEMENT_PENALTY;
            let penalty_comment = "Ensemble: penalised (module disagreement)".to_string();
            c.comment = Some(match c.comment {
                Some(existing) => format!("{existing} | {penalty_comment}"),
                None => penalty_comment,
            });
            c
        })
        .collect()
}

/// Extract the module name from a candidate's comment.
///
/// Discovery modules typically set the comment to their module name or a
/// description prefixed by the module name. If no comment is present,
/// returns "unknown".
fn extract_module_name(candidate: &CoordinatedStructuralCandidateJson) -> String {
    candidate.comment.as_ref().map_or_else(
        || "unknown".to_string(),
        |c| {
            // Take the first part before any colon or pipe separator.
            c.split(&[':', '|'][..])
                .next()
                .unwrap_or("unknown")
                .trim()
                .to_string()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_key_for_set_bias() {
        let c = CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: "abc".to_string(),
                bias: 0.5,
            }],
            expected_creature_score_gain: 0.01,
            comment: None,
        };
        assert_eq!(target_key(&c), "n:abc");
    }

    #[test]
    fn target_key_for_add_synapse() {
        let c = CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "a".to_string(),
                to_neuron_uuid: "b".to_string(),
                weight: 0.1,
            }],
            expected_creature_score_gain: 0.01,
            comment: None,
        };
        assert_eq!(target_key(&c), "s:a->b");
    }

    #[test]
    fn classify_agreement() {
        let group = vec![
            CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::SetBias {
                    neuron_uuid: "a".to_string(),
                    bias: 0.1,
                }],
                expected_creature_score_gain: 0.01,
                comment: None,
            },
            CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::SetBias {
                    neuron_uuid: "a".to_string(),
                    bias: 0.2,
                }],
                expected_creature_score_gain: 0.02,
                comment: None,
            },
        ];
        let types = classify_operation_types(&group);
        assert_eq!(types.len(), 1);
        assert_eq!(types[0], "SetBias");
    }

    #[test]
    fn classify_disagreement() {
        let group = vec![
            CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
                    neuron_uuid: "a".to_string(),
                }],
                expected_creature_score_gain: 0.01,
                comment: None,
            },
            CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
                    neuron_uuid: "a".to_string(),
                    squash: "RELU".to_string(),
                }],
                expected_creature_score_gain: 0.02,
                comment: None,
            },
        ];
        let types = classify_operation_types(&group);
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn extract_module_name_from_comment() {
        let c = CoordinatedStructuralCandidateJson {
            operations: vec![],
            expected_creature_score_gain: 0.01,
            comment: Some("saturation detection: found 3 saturated neurons".to_string()),
        };
        assert_eq!(extract_module_name(&c), "saturation detection");
    }

    #[test]
    fn extract_module_name_no_comment() {
        let c = CoordinatedStructuralCandidateJson {
            operations: vec![],
            expected_creature_score_gain: 0.01,
            comment: None,
        };
        assert_eq!(extract_module_name(&c), "unknown");
    }
}
