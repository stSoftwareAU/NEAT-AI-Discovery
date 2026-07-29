//! Issue #1423 — novelty / diversification escalation for plateaued creatures.
//!
//! Behaviour tests against the escalation decision and the gain-floor lever.
//!
//! Issue #1792 note: the original AC1/AC2 tests drove
//! `seed_forced_novel_candidates` through a `CandidateOutcomeCache`. That cache
//! was never constructed outside tests, so the seeder was unreachable from
//! production and both were deleted. The surviving production path is
//! `decide_escalation` → `failure_cache_handshake::evaluate` →
//! `noveltyEscalationActive`, which is what these tests now cover; the
//! failure-cache-driven engagement itself is covered end-to-end by
//! `tests/analysis/issue_1781_failure_cache_expiry.rs`.

use neat_ai_discovery::analysis::discovery_mode::DEFAULT_LOW_SUCCESS_RATE_THRESHOLD;
use neat_ai_discovery::analysis::novelty_escalation::{
    DEFAULT_GAIN_FLOOR_RELAXATION, DEFAULT_SUPPRESSION_RATIO_THRESHOLD, decide_escalation,
    gain_floor_multiplier,
};

/// A plateaued creature whose candidate pool is fully suppressed escalates.
#[test]
fn escalation_engages_on_a_full_plateau() {
    let decision = decide_escalation(
        0.0,
        DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
        4,
        4,
        DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
    );
    assert!(decision.engaged, "escalation must engage on a full plateau");
    assert!((decision.suppression_ratio - 1.0).abs() < 1e-9);
}

/// AC3: on a non-plateaued creature (healthy rolling success rate) escalation
/// stays inert, even when candidates happen to be suppressed — steady-state
/// behaviour is unchanged.
#[test]
fn ac3_no_escalation_on_non_plateaued_creature() {
    let decision = decide_escalation(
        0.6,
        DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
        4,
        4,
        DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
    );
    assert!(
        !decision.engaged,
        "escalation must stay inert on a non-plateaued creature"
    );

    // The gain floor is unchanged (multiplier 1.0) when escalation is inert.
    let multiplier = gain_floor_multiplier(decision.engaged, DEFAULT_GAIN_FLOOR_RELAXATION);
    assert!((multiplier - 1.0).abs() < 1e-9);
}

/// The gain floor is relaxed (multiplier < 1.0) only when escalation engages.
#[test]
fn gain_floor_relaxed_only_under_escalation() {
    let engaged = decide_escalation(0.0, 0.2, 10, 10, 0.8);
    assert!(engaged.engaged);
    let m = gain_floor_multiplier(engaged.engaged, DEFAULT_GAIN_FLOOR_RELAXATION);
    assert!(
        m < 1.0 && m > 0.0,
        "floor must be loosened, not zeroed: {m}"
    );
}
