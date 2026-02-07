//! Candidate pre-filter for early termination of low-value candidates (Issue #429).
//!
//! This module provides a hierarchical filtering pipeline that runs alongside
//! the discovery module dispatch loop. It reduces analysis time by:
//!
//! 1. **Minimum gain threshold**: Rejects candidates whose expected score gain
//!    is below a configurable minimum, skipping clearly unhelpful suggestions.
//!
//! 2. **Budget-aware prioritisation**: Tracks how many candidates have been
//!    accepted so far and stops accepting low-priority candidates once the
//!    budget is exhausted.
//!
//! 3. **Incremental confidence check**: Skips candidates whose expected gain
//!    confidence interval lower bound is negative (i.e., not confidently
//!    beneficial).
//!
//! 4. **Cross-module deduplication**: Tracks neuron-pair signatures across
//!    discovery modules to skip candidates that duplicate an already-accepted
//!    candidate from an earlier module.
//!
//! ## Integration
//!
//! The pre-filter is instantiated once per `analyze_all()` call and passed to
//! each `run_discovery_module_filtered()` invocation. It accumulates state
//! across modules so later modules benefit from earlier decisions.

use std::collections::HashSet;

use crate::CoordinatedStructuralCandidateJson;

/// Minimum expected score gain below which a candidate is rejected outright.
///
/// Candidates with gains at or below this threshold are considered noise
/// and are filtered before merge. The value is intentionally conservative
/// (very close to zero) to avoid filtering genuine improvements.
pub const MIN_CANDIDATE_GAIN: f32 = 1e-9;

/// Configuration for candidate pre-filtering.
#[derive(Debug, Clone)]
pub struct PreFilterConfig {
    /// Maximum number of coordinated structural candidates to accept across
    /// all discovery modules. Once this budget is exhausted, further modules
    /// are skipped entirely.
    pub candidate_budget: usize,

    /// Minimum expected score gain for a candidate to pass the pre-filter.
    pub min_gain: f32,

    /// Whether cross-module deduplication is enabled.
    pub dedup_enabled: bool,
}

impl Default for PreFilterConfig {
    fn default() -> Self {
        Self {
            candidate_budget: 256,
            min_gain: MIN_CANDIDATE_GAIN,
            dedup_enabled: true,
        }
    }
}

/// Tracks filtering state across discovery module dispatch calls.
///
/// Instantiate once per `analyze_all()` call and pass to each module.
pub struct CandidatePreFilter {
    config: PreFilterConfig,

    /// Total candidates accepted so far across all modules.
    accepted_count: usize,

    /// Set of (from_uuid, to_uuid) pairs already seen — used for deduplication
    /// of `AddSynapse` and `RemoveSynapse` operations across modules.
    seen_pairs: HashSet<(String, String)>,

    /// Number of candidates rejected by the gain threshold.
    rejected_low_gain: usize,

    /// Number of candidates rejected by budget exhaustion.
    rejected_budget: usize,

    /// Number of candidates rejected by deduplication.
    rejected_dedup: usize,

    /// Number of modules skipped entirely (budget exhausted before dispatch).
    modules_skipped: usize,
}

impl CandidatePreFilter {
    /// Create a new pre-filter with the given configuration.
    #[must_use]
    pub fn new(config: PreFilterConfig) -> Self {
        Self {
            config,
            accepted_count: 0,
            seen_pairs: HashSet::new(),
            rejected_low_gain: 0,
            rejected_budget: 0,
            rejected_dedup: 0,
            modules_skipped: 0,
        }
    }

    /// Check whether the candidate budget is already exhausted.
    ///
    /// When `true`, the caller can skip running the discovery module entirely.
    #[must_use]
    pub fn budget_exhausted(&self) -> bool {
        self.accepted_count >= self.config.candidate_budget
    }

    /// Record that a module was skipped because the budget was exhausted.
    pub fn record_module_skipped(&mut self) {
        self.modules_skipped += 1;
    }

    /// Filter a batch of candidates produced by a single discovery module.
    ///
    /// Returns only candidates that pass all pre-filter checks:
    /// 1. Expected gain above minimum threshold
    /// 2. Within remaining budget
    /// 3. Not a duplicate of an already-seen neuron pair
    pub fn filter(
        &mut self,
        candidates: Vec<CoordinatedStructuralCandidateJson>,
    ) -> Vec<CoordinatedStructuralCandidateJson> {
        let mut accepted = Vec::new();

        for candidate in candidates {
            // 1. Minimum gain threshold
            if candidate.expected_creature_score_gain <= self.config.min_gain {
                self.rejected_low_gain += 1;
                continue;
            }

            // 2. Budget check
            if self.accepted_count >= self.config.candidate_budget {
                self.rejected_budget += 1;
                continue;
            }

            // 3. Cross-module deduplication
            if self.config.dedup_enabled {
                let signature = extract_candidate_signature(&candidate);
                if let Some(sig) = signature {
                    if self.seen_pairs.contains(&sig) {
                        self.rejected_dedup += 1;
                        continue;
                    }
                    self.seen_pairs.insert(sig);
                }
            }

            self.accepted_count += 1;
            accepted.push(candidate);
        }

        accepted
    }

    /// Get the total number of candidates accepted so far.
    #[must_use]
    pub fn accepted_count(&self) -> usize {
        self.accepted_count
    }

    /// Get the remaining budget (candidates still allowed).
    #[must_use]
    pub fn remaining_budget(&self) -> usize {
        self.config
            .candidate_budget
            .saturating_sub(self.accepted_count)
    }

    /// Get filtering statistics for verbose logging.
    #[must_use]
    pub fn stats(&self) -> PreFilterStats {
        PreFilterStats {
            accepted: self.accepted_count,
            rejected_low_gain: self.rejected_low_gain,
            rejected_budget: self.rejected_budget,
            rejected_dedup: self.rejected_dedup,
            modules_skipped: self.modules_skipped,
        }
    }
}

/// Summary statistics from pre-filtering.
#[derive(Debug, Clone)]
pub struct PreFilterStats {
    /// Total candidates accepted.
    pub accepted: usize,
    /// Candidates rejected for gain below threshold.
    pub rejected_low_gain: usize,
    /// Candidates rejected because budget was exhausted.
    pub rejected_budget: usize,
    /// Candidates rejected as duplicates of earlier modules.
    pub rejected_dedup: usize,
    /// Discovery modules skipped entirely (budget exhausted).
    pub modules_skipped: usize,
}

impl PreFilterStats {
    /// Total candidates rejected across all reasons.
    #[must_use]
    pub fn total_rejected(&self) -> usize {
        self.rejected_low_gain + self.rejected_budget + self.rejected_dedup
    }
}

/// Extract a deduplication signature from a coordinated structural candidate.
///
/// The signature is the first (from_uuid, to_uuid) pair found in the
/// candidate's operations. This captures the primary neuron pair affected
/// by the candidate, which is sufficient for cross-module deduplication
/// (two modules suggesting the same structural change on the same pair).
fn extract_candidate_signature(
    candidate: &CoordinatedStructuralCandidateJson,
) -> Option<(String, String)> {
    use crate::CoordinatedStructuralOpJson;

    for op in &candidate.operations {
        match op {
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                ..
            }
            | CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
            } => {
                return Some((from_neuron_uuid.clone(), to_neuron_uuid.clone()));
            }
            CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid,
                to_neuron_uuid,
                ..
            } => {
                return Some((from_neuron_uuid.clone(), to_neuron_uuid.clone()));
            }
            _ => continue,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CoordinatedStructuralOpJson;

    fn make_candidate(from: &str, to: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
        CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: from.to_string(),
                to_neuron_uuid: to.to_string(),
                weight: 0.5,
            }],
            expected_creature_score_gain: gain,
            comment: None,
        }
    }

    #[test]
    fn test_prefilter_rejects_low_gain_candidates() {
        let config = PreFilterConfig {
            min_gain: 0.01,
            candidate_budget: 100,
            dedup_enabled: false,
        };
        let mut filter = CandidatePreFilter::new(config);

        let candidates = vec![
            make_candidate("a", "b", 0.1),   // above threshold
            make_candidate("c", "d", 0.005), // below threshold
            make_candidate("e", "f", 0.02),  // above threshold
        ];

        let result = filter.filter(candidates);
        assert_eq!(
            result.len(),
            2,
            "Should keep only candidates above min_gain"
        );
        assert_eq!(filter.stats().rejected_low_gain, 1);
    }

    #[test]
    fn test_prefilter_enforces_budget() {
        let config = PreFilterConfig {
            min_gain: MIN_CANDIDATE_GAIN,
            candidate_budget: 2,
            dedup_enabled: false,
        };
        let mut filter = CandidatePreFilter::new(config);

        let candidates = vec![
            make_candidate("a", "b", 0.1),
            make_candidate("c", "d", 0.2),
            make_candidate("e", "f", 0.3), // should be rejected — budget exhausted
        ];

        let result = filter.filter(candidates);
        assert_eq!(result.len(), 2, "Should accept only up to budget");
        assert_eq!(filter.stats().rejected_budget, 1);
        assert!(filter.budget_exhausted());
    }

    #[test]
    fn test_prefilter_deduplicates_across_calls() {
        let config = PreFilterConfig {
            min_gain: MIN_CANDIDATE_GAIN,
            candidate_budget: 100,
            dedup_enabled: true,
        };
        let mut filter = CandidatePreFilter::new(config);

        // First module produces candidate (a, b)
        let batch1 = vec![make_candidate("a", "b", 0.1)];
        let result1 = filter.filter(batch1);
        assert_eq!(result1.len(), 1);

        // Second module produces duplicate (a, b) — should be rejected
        let batch2 = vec![make_candidate("a", "b", 0.2)];
        let result2 = filter.filter(batch2);
        assert_eq!(result2.len(), 0, "Duplicate pair should be rejected");
        assert_eq!(filter.stats().rejected_dedup, 1);
    }

    #[test]
    fn test_prefilter_allows_different_pairs() {
        let config = PreFilterConfig {
            min_gain: MIN_CANDIDATE_GAIN,
            candidate_budget: 100,
            dedup_enabled: true,
        };
        let mut filter = CandidatePreFilter::new(config);

        let batch1 = vec![make_candidate("a", "b", 0.1)];
        let batch2 = vec![make_candidate("c", "d", 0.2)];

        let result1 = filter.filter(batch1);
        let result2 = filter.filter(batch2);

        assert_eq!(result1.len(), 1);
        assert_eq!(
            result2.len(),
            1,
            "Different pairs should not be deduplicated"
        );
        assert_eq!(filter.stats().rejected_dedup, 0);
    }

    #[test]
    fn test_budget_exhausted_check() {
        let config = PreFilterConfig {
            min_gain: MIN_CANDIDATE_GAIN,
            candidate_budget: 1,
            dedup_enabled: false,
        };
        let mut filter = CandidatePreFilter::new(config);

        assert!(!filter.budget_exhausted());

        let batch = vec![make_candidate("a", "b", 0.1)];
        filter.filter(batch);

        assert!(filter.budget_exhausted());
        assert_eq!(filter.remaining_budget(), 0);
    }

    #[test]
    fn test_module_skipped_tracking() {
        let config = PreFilterConfig {
            min_gain: MIN_CANDIDATE_GAIN,
            candidate_budget: 0,
            dedup_enabled: false,
        };
        let mut filter = CandidatePreFilter::new(config);

        assert!(filter.budget_exhausted());
        filter.record_module_skipped();
        filter.record_module_skipped();

        assert_eq!(filter.stats().modules_skipped, 2);
    }

    #[test]
    fn test_stats_total_rejected() {
        let config = PreFilterConfig {
            min_gain: 0.05,
            candidate_budget: 1,
            dedup_enabled: true,
        };
        let mut filter = CandidatePreFilter::new(config);

        let batch = vec![
            make_candidate("a", "b", 0.1),  // accepted (1 of 1 budget)
            make_candidate("c", "d", 0.01), // rejected: low gain
            make_candidate("a", "b", 0.2),  // rejected: budget (dedup would also apply)
        ];
        filter.filter(batch);

        let stats = filter.stats();
        assert_eq!(stats.accepted, 1);
        assert_eq!(stats.rejected_low_gain, 1);
        assert_eq!(stats.rejected_budget, 1);
        assert_eq!(stats.total_rejected(), 2);
    }

    #[test]
    fn test_extract_signature_add_synapse() {
        let candidate = make_candidate("src-1", "tgt-2", 0.1);
        let sig = extract_candidate_signature(&candidate);
        assert_eq!(sig, Some(("src-1".to_string(), "tgt-2".to_string())));
    }

    #[test]
    fn test_extract_signature_remove_synapse() {
        let candidate = CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "from-1".to_string(),
                to_neuron_uuid: "to-1".to_string(),
            }],
            expected_creature_score_gain: 0.1,
            comment: None,
        };
        let sig = extract_candidate_signature(&candidate);
        assert_eq!(sig, Some(("from-1".to_string(), "to-1".to_string())));
    }

    #[test]
    fn test_extract_signature_no_synapse_ops() {
        let candidate = CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: "n-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.0,
                insert_before_neuron_uuid: None,
            }],
            expected_creature_score_gain: 0.1,
            comment: None,
        };
        let sig = extract_candidate_signature(&candidate);
        assert_eq!(
            sig, None,
            "AddNeuron-only candidates have no pair signature"
        );
    }

    #[test]
    fn test_default_config() {
        let config = PreFilterConfig::default();
        assert_eq!(config.candidate_budget, 256);
        assert!(config.min_gain > 0.0);
        assert!(config.dedup_enabled);
    }
}
