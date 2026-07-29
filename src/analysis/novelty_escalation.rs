//! Novelty / diversification escalation for plateaued creatures (Issue #1423).
//!
//! Follow-up to the drought-diagnostic work (#1418). The drought escape hatch
//! (#1205) clears the *memory* of past rejections, but on a genuinely
//! search-exhausted creature the next pass simply re-proposes the same losers
//! once the suppression ages out. This module decides when to escalate **what**
//! is proposed rather than merely forgetting what failed.
//!
//! When the search is plateaued — the rolling success rate has fallen below the
//! conservative-mode threshold **and** most of the returned candidate pool is
//! currently suppressed — [`decide_escalation`] engages and two levers follow:
//!
//! 1. **Failure-cache bypass.** The handshake
//!    ([`super::failure_cache_handshake`]) reports `noveltyEscalationActive`
//!    so NEAT-AI bypasses its failure-cache filter for the top-K candidates
//!    and at least one candidate reaches Phase-1 evaluation.
//! 2. **Gain-floor relaxation.** The coordinated-structural expected-gain floor
//!    is loosened ([`gain_floor_multiplier`]) so structurally-novel candidates
//!    that conservative mode would otherwise gate are allowed through.
//!
//! Both levers are **inert** when the creature is not plateaued, so a healthy,
//! steadily-accepting creature sees no change in behaviour.
//!
//! Issue #1792: this module also carried `rank_source_types_by_novelty` and
//! `seed_forced_novel_candidates`, which took a `&CandidateOutcomeCache`. That
//! cache was never constructed outside tests, so neither entry point was
//! reachable from production; both were deleted with the cache.

/// Default fraction of the considered candidate pool that must be
/// suppressed before escalation engages.
///
/// At `0.8`, escalation only fires once at least 80 % of the candidates the
/// search would normally propose are currently suppressed — i.e. the search is
/// genuinely re-treading old ground rather than merely having a slow pass.
pub const DEFAULT_SUPPRESSION_RATIO_THRESHOLD: f64 = 0.8;

/// Default multiplier applied to the coordinated-structural expected-gain floor
/// when escalation engages.
///
/// Values below `1.0` *relax* the floor (let more structural candidates
/// through); `0.5` halves it. This deliberately opposes conservative mode,
/// which tightens the same floor — under sustained drought, widening the search
/// is the less-bad option.
pub const DEFAULT_GAIN_FLOOR_RELAXATION: f32 = 0.5;

/// Outcome of the escalation decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EscalationDecision {
    /// Whether novelty/diversification escalation should engage this pass.
    pub engaged: bool,
    /// Fraction of the considered pool that was suppressed (`0.0` when nothing
    /// was considered).
    pub suppression_ratio: f64,
    /// Rolling success rate that fed the decision (echoed for diagnostics).
    pub rolling_success_rate: f32,
}

/// Decide whether escalation should engage for this pass.
///
/// Escalation engages when **all** of the following hold:
///
/// 1. At least one candidate was considered (`considered_count > 0`).
/// 2. The rolling success rate is strictly below `low_threshold` (the same
///    threshold that drives conservative mode).
/// 3. The suppressed fraction (`suppressed_count / considered_count`) is at or
///    above `suppression_ratio_threshold`.
///
/// A non-plateaued creature fails condition 2, so escalation stays inert and
/// steady-state behaviour is unchanged.
#[must_use]
pub fn decide_escalation(
    rolling_success_rate: f32,
    low_threshold: f32,
    suppressed_count: usize,
    considered_count: usize,
    suppression_ratio_threshold: f64,
) -> EscalationDecision {
    let suppression_ratio = if considered_count == 0 {
        0.0
    } else {
        // Counts are bounded by the candidate pool; saturate to u32 before the
        // exact `f64::from` conversion so the ratio never loses precision.
        let suppressed = f64::from(u32::try_from(suppressed_count).unwrap_or(u32::MAX));
        let considered = f64::from(u32::try_from(considered_count).unwrap_or(u32::MAX));
        suppressed / considered
    };

    let engaged = considered_count > 0
        && rolling_success_rate < low_threshold
        && suppression_ratio >= suppression_ratio_threshold;

    EscalationDecision {
        engaged,
        suppression_ratio,
        rolling_success_rate,
    }
}

/// Multiplier to apply to the coordinated-structural expected-gain floor.
///
/// Returns `relaxation` (clamped to `(0.0, 1.0]`) when escalation is engaged so
/// the floor is loosened, otherwise `1.0` (no change). The clamp guarantees the
/// floor is never raised by this lever and never collapses to zero.
#[must_use]
pub fn gain_floor_multiplier(engaged: bool, relaxation: f32) -> f32 {
    if !engaged {
        return 1.0;
    }
    if !relaxation.is_finite() {
        return DEFAULT_GAIN_FLOOR_RELAXATION;
    }
    relaxation.clamp(f32::MIN_POSITIVE, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::discovery_mode::DEFAULT_LOW_SUCCESS_RATE_THRESHOLD;

    #[test]
    fn decide_escalation_engages_when_plateaued_and_suppressed() {
        let d = decide_escalation(
            0.1,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            9,
            10,
            DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
        );
        assert!(d.engaged);
        assert!((d.suppression_ratio - 0.9).abs() < 1e-9);
    }

    #[test]
    fn decide_escalation_inert_when_not_plateaued() {
        // Healthy success rate, fully suppressed — must NOT engage (steady state).
        let d = decide_escalation(
            0.6,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            10,
            10,
            DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
        );
        assert!(!d.engaged);
    }

    #[test]
    fn decide_escalation_inert_when_few_suppressed() {
        // Plateaued but most candidates still fresh — escalation not yet needed.
        let d = decide_escalation(
            0.1,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            2,
            10,
            DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
        );
        assert!(!d.engaged);
    }

    #[test]
    fn decide_escalation_inert_with_empty_pool() {
        let d = decide_escalation(0.0, 0.2, 0, 0, 0.8);
        assert!(!d.engaged);
        assert_eq!(d.suppression_ratio, 0.0);
    }

    #[test]
    fn gain_floor_multiplier_relaxes_only_when_engaged() {
        assert!((gain_floor_multiplier(false, 0.5) - 1.0).abs() < 1e-9);
        assert!((gain_floor_multiplier(true, 0.5) - 0.5).abs() < 1e-9);
        // Never raised above 1.0.
        assert!((gain_floor_multiplier(true, 4.0) - 1.0).abs() < 1e-9);
        // Never collapses to zero or below.
        assert!(gain_floor_multiplier(true, 0.0) > 0.0);
        assert!(gain_floor_multiplier(true, -1.0) > 0.0);
    }
}
