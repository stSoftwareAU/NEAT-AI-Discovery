//! Cross-stack failure-cache handshake (Issue #1447).
//!
//! Issue #1423 added Rust-side novelty/diversification escalation for plateaued
//! creatures, but production stayed in drought because of a **cross-stack gap**:
//! NEAT-AI (TypeScript) builds creatures from Rust output and then drops every
//! candidate whose identity matches the per-creature failure cache *before*
//! Phase-1 evaluation. Rust proposed "novel" candidates; TS suppressed them all
//! and the operator saw `Built 0 candidates`.
//!
//! This module computes the Rust half of the handshake so NEAT-AI can both
//! observe and act on the gap:
//!
//! 1. [`count_suppressed`] reports how many of the candidates Rust is returning
//!    match a failure-cache identity — i.e. how many TS will drop. This is the
//!    `failure_cache_suppressed_count` surfaced on analysis metadata.
//! 2. [`evaluate`] folds that count into the existing
//!    [`decide_escalation`] gate to
//!    produce `novelty_escalation_active`. When set, NEAT-AI is expected to
//!    bypass its failure-cache filter for the top-K candidates (mirroring the
//!    #1423 intent) so at least one candidate reaches Phase-1 evaluation.
//!
//! The matched count also wires the previously-dead
//! [`REJECTION_DUPLICATE_OF_FAILURE_CACHE`](super::diagnostics::rejection_reasons::REJECTION_DUPLICATE_OF_FAILURE_CACHE)
//! reason so duplicate suppression is no longer invisible on the Rust side.
//!
//! Pure logic only — no GPU, Parquet, or FFI dependencies — mirroring the
//! self-contained design of [`novelty_escalation`](super::novelty_escalation).

use super::novelty_escalation::{EscalationDecision, decide_escalation};
use super::scoring::calibration_correction::FailureCacheEntry;

/// Stable change-type identifier for add-synapse candidates (matches the
/// NEAT-AI failure-cache `changeType`).
pub const CHANGE_TYPE_ADD_SYNAPSES: &str = "add-synapses";
/// Stable change-type identifier for add-neuron candidates.
pub const CHANGE_TYPE_ADD_NEURONS: &str = "add-neurons";
/// Stable change-type identifier for coordinated-structural candidates.
pub const CHANGE_TYPE_COORDINATED_STRUCTURAL: &str = "coordinated-structural";

/// Identity of a returned candidate, used to match against the failure cache.
///
/// Mirrors the identity fields the NEAT-AI failure cache carries: the
/// `change_type`, the target neuron UUID, and (for squash-bearing candidates)
/// the target squash. `None` fields are treated as wildcards during matching so
/// a coarse failure-cache entry still suppresses a more specific candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateIdentity {
    /// Candidate change type (e.g. `"add-synapses"`).
    pub change_type: String,
    /// Target neuron UUID, when the candidate type has a single target.
    pub target_uuid: Option<String>,
    /// Target squash function, when the candidate carries one (add-neurons).
    pub target_squash: Option<String>,
}

impl CandidateIdentity {
    /// Convenience constructor.
    #[must_use]
    pub fn new(
        change_type: impl Into<String>,
        target_uuid: Option<String>,
        target_squash: Option<String>,
    ) -> Self {
        Self {
            change_type: change_type.into(),
            target_uuid,
            target_squash,
        }
    }
}

/// Whether `entry` suppresses `candidate`.
///
/// The change type must match exactly. A field present on the failure-cache
/// entry (`target_uuid`, `target_squash`) must equal the candidate's; a field
/// absent on the entry acts as a wildcard so a target-agnostic entry still
/// matches. This mirrors NEAT-AI `isCandidateCached`, which keys on the same
/// identity tuple and treats missing target metadata as "any".
#[must_use]
fn entry_matches(entry: &FailureCacheEntry, candidate: &CandidateIdentity) -> bool {
    if entry.change_type != candidate.change_type {
        return false;
    }
    if let Some(entry_uuid) = entry.target_uuid.as_deref()
        && candidate.target_uuid.as_deref() != Some(entry_uuid)
    {
        return false;
    }
    if let Some(entry_squash) = entry.target_squash.as_deref()
        && candidate.target_squash.as_deref() != Some(entry_squash)
    {
        return false;
    }
    true
}

/// Count how many `candidates` match at least one failure-cache entry.
///
/// Returns `0` when the failure cache is empty. The count is the number of
/// candidates NEAT-AI's failure-cache filter will drop before Phase-1
/// evaluation.
#[must_use]
pub fn count_suppressed(
    candidates: &[CandidateIdentity],
    failure_cache: &[FailureCacheEntry],
) -> usize {
    if failure_cache.is_empty() {
        return 0;
    }
    candidates
        .iter()
        .filter(|c| failure_cache.iter().any(|e| entry_matches(e, c)))
        .count()
}

/// Outcome of the cross-stack handshake evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandshakeOutcome {
    /// Number of returned candidates whose identity matches the failure cache —
    /// the count NEAT-AI's filter will suppress before Phase-1 evaluation.
    pub failure_cache_suppressed_count: usize,
    /// Whether novelty escalation is engaged this pass. When `true`, NEAT-AI
    /// should bypass its failure-cache filter for the top-K candidates so at
    /// least one candidate reaches Phase-1 evaluation.
    pub novelty_escalation_active: bool,
    /// The underlying escalation decision (echoed for diagnostics).
    pub escalation: EscalationDecision,
}

/// Evaluate the handshake for a set of returned candidates.
///
/// `novelty_escalation_active` engages only when the creature is genuinely
/// plateaued (`rolling_success_rate < low_threshold`) **and** the suppressed
/// fraction of the returned candidates meets `suppression_ratio_threshold` —
/// the exact situation in which every built candidate would otherwise be
/// dropped by the TS failure-cache filter, leaving the operator at
/// `Built 0 candidates`.
#[must_use]
pub fn evaluate(
    candidates: &[CandidateIdentity],
    failure_cache: &[FailureCacheEntry],
    rolling_success_rate: f32,
    low_threshold: f32,
    suppression_ratio_threshold: f64,
) -> HandshakeOutcome {
    let suppressed = count_suppressed(candidates, failure_cache);
    let escalation = decide_escalation(
        rolling_success_rate,
        low_threshold,
        suppressed,
        candidates.len(),
        suppression_ratio_threshold,
    );
    HandshakeOutcome {
        failure_cache_suppressed_count: suppressed,
        novelty_escalation_active: escalation.engaged,
        escalation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::discovery_mode::DEFAULT_LOW_SUCCESS_RATE_THRESHOLD;
    use crate::analysis::novelty_escalation::DEFAULT_SUPPRESSION_RATIO_THRESHOLD;

    /// Build a minimal failure-cache entry for matching tests.
    fn entry(
        change_type: &str,
        target_uuid: Option<&str>,
        target_squash: Option<&str>,
    ) -> FailureCacheEntry {
        FailureCacheEntry {
            change_type: change_type.to_string(),
            expected_error_reduction: 0.0,
            actual_error_reduction: 0.0,
            target_squash: target_squash.map(str::to_string),
            variant_key: None,
            target_uuid: target_uuid.map(str::to_string),
            improved_count: None,
            total_count: None,
        }
    }

    fn ident(change_type: &str, uuid: Option<&str>, squash: Option<&str>) -> CandidateIdentity {
        CandidateIdentity::new(
            change_type,
            uuid.map(str::to_string),
            squash.map(str::to_string),
        )
    }

    #[test]
    fn empty_cache_suppresses_nothing() {
        let cands = vec![ident(CHANGE_TYPE_ADD_SYNAPSES, Some("n1"), None)];
        assert_eq!(count_suppressed(&cands, &[]), 0);
    }

    #[test]
    fn change_type_must_match() {
        let cache = vec![entry(CHANGE_TYPE_COORDINATED_STRUCTURAL, None, None)];
        let cands = vec![ident(CHANGE_TYPE_ADD_SYNAPSES, Some("n1"), None)];
        assert_eq!(count_suppressed(&cands, &cache), 0);
    }

    #[test]
    fn entry_target_uuid_must_match_when_present() {
        let cache = vec![entry(CHANGE_TYPE_ADD_SYNAPSES, Some("n1"), None)];
        let matching = vec![ident(CHANGE_TYPE_ADD_SYNAPSES, Some("n1"), None)];
        let other = vec![ident(CHANGE_TYPE_ADD_SYNAPSES, Some("n2"), None)];
        assert_eq!(count_suppressed(&matching, &cache), 1);
        assert_eq!(count_suppressed(&other, &cache), 0);
    }

    #[test]
    fn absent_entry_target_acts_as_wildcard() {
        // A coordinated-structural entry with no target uuid suppresses any
        // coordinated-structural candidate regardless of its target.
        let cache = vec![entry(CHANGE_TYPE_COORDINATED_STRUCTURAL, None, None)];
        let cands = vec![
            ident(CHANGE_TYPE_COORDINATED_STRUCTURAL, None, None),
            ident(CHANGE_TYPE_COORDINATED_STRUCTURAL, Some("anything"), None),
        ];
        assert_eq!(count_suppressed(&cands, &cache), 2);
    }

    #[test]
    fn target_squash_discriminates_add_neurons() {
        let cache = vec![entry(CHANGE_TYPE_ADD_NEURONS, Some("n1"), Some("RELU"))];
        let same = vec![ident(CHANGE_TYPE_ADD_NEURONS, Some("n1"), Some("RELU"))];
        let diff = vec![ident(CHANGE_TYPE_ADD_NEURONS, Some("n1"), Some("TANH"))];
        assert_eq!(count_suppressed(&same, &cache), 1);
        assert_eq!(count_suppressed(&diff, &cache), 0);
    }

    #[test]
    fn evaluate_engages_escalation_when_all_built_candidates_are_cached() {
        // The GRQ-3 scenario: a plateaued creature whose every returned
        // candidate matches the failure cache.
        let cache = vec![entry(CHANGE_TYPE_COORDINATED_STRUCTURAL, None, None)];
        let cands = vec![
            ident(CHANGE_TYPE_COORDINATED_STRUCTURAL, Some("a"), None),
            ident(CHANGE_TYPE_COORDINATED_STRUCTURAL, Some("b"), None),
        ];
        let out = evaluate(
            &cands,
            &cache,
            0.05,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
        );
        assert_eq!(out.failure_cache_suppressed_count, 2);
        assert!(out.novelty_escalation_active);
    }

    #[test]
    fn evaluate_inert_when_not_plateaued() {
        // Fully suppressed but a healthy success rate — escalation stays off so
        // steady-state behaviour is unchanged.
        let cache = vec![entry(CHANGE_TYPE_COORDINATED_STRUCTURAL, None, None)];
        let cands = vec![ident(CHANGE_TYPE_COORDINATED_STRUCTURAL, Some("a"), None)];
        let out = evaluate(
            &cands,
            &cache,
            0.9,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
        );
        assert_eq!(out.failure_cache_suppressed_count, 1);
        assert!(!out.novelty_escalation_active);
    }

    #[test]
    fn evaluate_inert_when_no_candidates() {
        let out = evaluate(
            &[],
            &[entry(CHANGE_TYPE_ADD_SYNAPSES, None, None)],
            0.0,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
        );
        assert_eq!(out.failure_cache_suppressed_count, 0);
        assert!(!out.novelty_escalation_active);
    }
}
