//! Drought diagnostic for sustained empty-discovery passes (Issue #1202).
//!
//! When the rolling outcome log shows N consecutive trailing failures, the
//! orchestrator emits a single structured `tracing::warn!` event and attaches
//! a [`DroughtDiagnostic`] payload to the FFI metadata. The diagnostic
//! consolidates suppression signals from the candidate cache, the per-target
//! cooldown tracker, the rejection breakdown, and the discovery mode so
//! operators can root-cause "no successful candidates for a while" without
//! re-running analysis under elevated logging.
//!
//! # Diagnostic shape
//!
//! See [`DroughtDiagnostic`] for the field-level schema. The same payload is
//! serialised as `synapseMetadata.droughtDiagnostic` and
//! `neuronMetadata.droughtDiagnostic` (camelCase JSON) in the FFI response.
//!
//! # Emission rule
//!
//! [`emit_drought_diagnostic`] returns `Some(payload)` only when
//! `consecutive_failures >= drought_threshold`. The orchestrator calls it once
//! per `analyze_all` invocation, so the warn log fires at most once per pass.

use serde::{Deserialize, Serialize};

use super::candidate_cache::CandidateOutcomeCache;
use super::diagnostics::RejectionBreakdown;
use super::discovery_mode::DiscoveryMode;
use super::target_failure_tracker::TargetFailureTracker;

/// Structured payload describing the current drought state (Issue #1202).
///
/// Serialised as `droughtDiagnostic` on both `synapseMetadata` and
/// `neuronMetadata`. All fields use camelCase in JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DroughtDiagnostic {
    /// Number of consecutive trailing empty passes that triggered the drought.
    pub consecutive_failures: u32,
    /// Rolling success rate over the most recent
    /// [`super::discovery_mode::ROLLING_WINDOW`] passes.
    pub rolling_success_rate: f32,
    /// Current creature-level discovery mode (`"normal"` / `"conservative"`).
    pub discovery_mode: DiscoveryMode,
    /// Total entries in the candidate outcome cache (success and failure).
    pub candidate_cache_size: usize,
    /// Failed candidate cache entries still inside the staleness window —
    /// these are actively suppressing candidate generation.
    pub candidate_cache_suppressed_count: usize,
    /// Number of target neurons currently in cooldown via
    /// [`TargetFailureTracker`].
    pub target_cooldown_active_count: usize,
    /// Targets dropped by the per-target cooldown filter during the most
    /// recent run (best-effort; `0` when not tracked by the orchestrator).
    pub target_cooldown_skipped: u32,
    /// Stable rejection-reason name with the highest count (e.g.
    /// `"no_eligible_sources"`). `None` when no rejections were recorded.
    pub dominant_rejection_reason: Option<String>,
    /// Count for [`Self::dominant_rejection_reason`].
    pub dominant_rejection_count: u32,
    /// Total candidates that were considered (returned + rejected). Lets
    /// callers reason about "N of M" without needing the breakdown details.
    pub total_candidates_considered: u32,
    /// Total candidates rejected across all reasons.
    pub total_candidates_rejected: u32,
}

/// Inputs gathered by the orchestrator before drought emission.
///
/// All counts use the same `current_epoch` so the diagnostic reports a
/// consistent snapshot.
#[derive(Debug, Clone, Copy)]
pub struct DroughtInputs<'a> {
    pub consecutive_failures: u32,
    pub rolling_success_rate: f32,
    pub discovery_mode: DiscoveryMode,
    pub candidate_cache: Option<&'a CandidateOutcomeCache>,
    pub target_tracker: Option<&'a TargetFailureTracker>,
    pub current_epoch: u64,
    pub target_cooldown_skipped: u32,
    pub rejection_breakdown: &'a RejectionBreakdown,
    pub candidates_returned: u32,
}

/// Build a [`DroughtDiagnostic`] when the trailing-failure streak has crossed
/// `drought_threshold`. Emits a single structured `tracing::warn!` event as a
/// side effect.
///
/// Returns `None` when no drought is active so the caller can leave
/// `droughtDiagnostic` unset on the FFI metadata.
#[must_use]
pub fn emit_drought_diagnostic(
    inputs: &DroughtInputs<'_>,
    drought_threshold: u32,
) -> Option<DroughtDiagnostic> {
    if inputs.consecutive_failures < drought_threshold {
        return None;
    }

    let candidate_cache_size = inputs.candidate_cache.map_or(0, CandidateOutcomeCache::len);
    let candidate_cache_suppressed_count = inputs
        .candidate_cache
        .map_or(0, |c| c.suppressed_count(inputs.current_epoch));
    let target_cooldown_active_count = inputs
        .target_tracker
        .map_or(0, |t| t.active_cooldown_count(inputs.current_epoch));

    let (dominant_rejection_reason, dominant_rejection_count) =
        match inputs.rejection_breakdown.dominant_reason() {
            Some((reason, count)) => (Some(reason.to_string()), count),
            None => (None, 0),
        };
    let total_candidates_rejected = inputs.rejection_breakdown.total();
    let total_candidates_considered =
        total_candidates_rejected.saturating_add(inputs.candidates_returned);

    let diagnostic = DroughtDiagnostic {
        consecutive_failures: inputs.consecutive_failures,
        rolling_success_rate: inputs.rolling_success_rate,
        discovery_mode: inputs.discovery_mode,
        candidate_cache_size,
        candidate_cache_suppressed_count,
        target_cooldown_active_count,
        target_cooldown_skipped: inputs.target_cooldown_skipped,
        dominant_rejection_reason,
        dominant_rejection_count,
        total_candidates_considered,
        total_candidates_rejected,
    };

    tracing::warn!(
        consecutive_failures = diagnostic.consecutive_failures,
        rolling_success_rate = diagnostic.rolling_success_rate,
        discovery_mode = inputs.discovery_mode.as_str(),
        candidate_cache_size = diagnostic.candidate_cache_size,
        candidate_cache_suppressed_count = diagnostic.candidate_cache_suppressed_count,
        target_cooldown_active_count = diagnostic.target_cooldown_active_count,
        target_cooldown_skipped = diagnostic.target_cooldown_skipped,
        dominant_rejection_reason = diagnostic
            .dominant_rejection_reason
            .as_deref()
            .unwrap_or(""),
        dominant_rejection_count = diagnostic.dominant_rejection_count,
        total_candidates_considered = diagnostic.total_candidates_considered,
        total_candidates_rejected = diagnostic.total_candidates_rejected,
        drought_threshold,
        "Issue #1202: discovery drought — no successful candidates for {} consecutive passes",
        diagnostic.consecutive_failures
    );

    Some(diagnostic)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn breakdown_with(reason: &'static str, count: u32) -> RejectionBreakdown {
        let mut b = RejectionBreakdown::new();
        b.record_many(reason, count);
        b
    }

    #[test]
    fn no_diagnostic_below_threshold() {
        let breakdown = RejectionBreakdown::new();
        let inputs = DroughtInputs {
            consecutive_failures: 4,
            rolling_success_rate: 0.0,
            discovery_mode: DiscoveryMode::Normal,
            candidate_cache: None,
            target_tracker: None,
            current_epoch: 0,
            target_cooldown_skipped: 0,
            rejection_breakdown: &breakdown,
            candidates_returned: 0,
        };
        assert!(emit_drought_diagnostic(&inputs, 5).is_none());
    }

    #[test]
    fn diagnostic_emits_at_threshold() {
        let breakdown = breakdown_with("no_eligible_sources", 7);
        let inputs = DroughtInputs {
            consecutive_failures: 5,
            rolling_success_rate: 0.0,
            discovery_mode: DiscoveryMode::Conservative,
            candidate_cache: None,
            target_tracker: None,
            current_epoch: 0,
            target_cooldown_skipped: 2,
            rejection_breakdown: &breakdown,
            candidates_returned: 0,
        };
        let diag = emit_drought_diagnostic(&inputs, 5).expect("at threshold");
        assert_eq!(diag.consecutive_failures, 5);
        assert_eq!(diag.discovery_mode, DiscoveryMode::Conservative);
        assert_eq!(
            diag.dominant_rejection_reason.as_deref(),
            Some("no_eligible_sources")
        );
        assert_eq!(diag.dominant_rejection_count, 7);
        assert_eq!(diag.total_candidates_rejected, 7);
        assert_eq!(diag.total_candidates_considered, 7);
        assert_eq!(diag.target_cooldown_skipped, 2);
    }

    #[test]
    fn diagnostic_aggregates_cache_and_tracker_counts() {
        let mut cache = CandidateOutcomeCache::new();
        cache.record("src1", "tgt1", "addSynapse", false, 0);
        cache.record("src2", "tgt2", "addSynapse", false, 0);
        cache.record("src3", "tgt3", "addSynapse", true, 0);

        let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
        tracker.record_failure("tgt-x", 0);
        tracker.record_failure("tgt-x", 1);

        let breakdown = breakdown_with("below_threshold", 3);
        let inputs = DroughtInputs {
            consecutive_failures: 6,
            rolling_success_rate: 0.0,
            discovery_mode: DiscoveryMode::Normal,
            candidate_cache: Some(&cache),
            target_tracker: Some(&tracker),
            current_epoch: 1,
            target_cooldown_skipped: 0,
            rejection_breakdown: &breakdown,
            candidates_returned: 1,
        };

        let diag = emit_drought_diagnostic(&inputs, 5).expect("emits");
        assert_eq!(diag.candidate_cache_size, 3);
        assert_eq!(diag.candidate_cache_suppressed_count, 2);
        assert_eq!(diag.target_cooldown_active_count, 1);
        assert_eq!(diag.total_candidates_considered, 4); // 3 rejected + 1 returned.
    }
}
