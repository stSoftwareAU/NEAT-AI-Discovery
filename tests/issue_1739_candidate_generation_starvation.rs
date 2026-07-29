//! Issue #1739 — improve candidate generation for the large converged
//! production network.
//!
//! The milestone-#1736 diagnosis (`docs/analysis/rejection-diagnosis-1737.md`,
//! Issue #1737) established that the converged production network is **not**
//! candidate-starved:
//! its generators emit candidates in quantity, which are then discarded at the
//! expected-gain gate (the estimator/threshold work of #1738 / #1740). Blindly
//! widening generation on that profile cannot lift the accepted rate and would
//! only add noise on the wider discovery corpus.
//!
//! This issue therefore delivers the scope's *decision* piece: classify a run
//! as candidate-starved vs proposal-rich-but-over-rejected from the diagnosis
//! output ([`RejectionBreakdown`]) and gate generation widening (novelty
//! escalation) to the starved case only. These behaviour tests pin:
//!
//! 1. The converged over-rejected profile classifies as proposal-rich and does
//!    **not** trigger widening (guards regression/noise on the wider corpus).
//! 2. A genuinely candidate-starved profile classifies as starved and **does**
//!    trigger widening (more candidates warranted).
//! 3. The starvation gate only fires when escalation was already engaged (no
//!    unsafe acceptance path is touched — #1623 evaluate-before-accept is
//!    independent of this hint).
//! 4. Every stable rejection reason is classified, so a silent regression back
//!    to candidate starvation cannot escape the partition.

use neat_ai_discovery::analysis::candidate_starvation::{
    ABUNDANCE_REJECTION_REASONS, GATE_SIDE_REJECTION_REASONS, StarvationClass, StarvationConfig,
    UPSTREAM_REJECTION_REASONS, classify, gate_escalation, recommend_widening,
    signals_from_breakdown,
};
use neat_ai_discovery::analysis::diagnostics::RejectionBreakdown;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    ALL_REJECTION_REASONS, REJECTION_BELOW_EXPECTED_GAIN_FLOOR, REJECTION_BELOW_MULTI_OP_FLOOR,
    REJECTION_NO_ELIGIBLE_SOURCES, REJECTION_NO_TARGET_RECORDS, REJECTION_NON_POSITIVE_GAIN,
    REJECTION_TARGET_SATURATED,
};
use std::collections::HashSet;

/// Build a breakdown from `(reason, count)` pairs.
fn breakdown(entries: &[(&'static str, u32)]) -> RejectionBreakdown {
    let mut b = RejectionBreakdown::new();
    for &(reason, count) in entries {
        b.record_many(reason, count);
    }
    b
}

#[test]
fn converged_production_profile_does_not_trigger_widening() {
    // Faithful to the #1737 diagnosis: thousands of candidates reach the accept
    // gate and are rejected there (gain collapse); ~2 acceptances survive.
    let b = breakdown(&[
        (REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 1800),
        (REJECTION_BELOW_MULTI_OP_FLOOR, 600),
        (REJECTION_NON_POSITIVE_GAIN, 400),
    ]);
    let signals = signals_from_breakdown(&b, 2);
    let class = classify(&signals, &StarvationConfig::default());

    assert_eq!(
        class,
        StarvationClass::ProposalRichOverRejected,
        "the converged production profile is proposal-rich but over-rejected, not starved"
    );
    assert!(
        !recommend_widening(class),
        "widening generation must not be recommended for the over-rejected production profile"
    );
    // Even if novelty escalation would otherwise engage this pass, the
    // starvation gate suppresses the wasted widening hint.
    assert!(
        !gate_escalation(true, class),
        "escalation must be gated off on the over-rejected production profile"
    );
}

#[test]
fn starved_profile_triggers_widening() {
    // Almost nothing reaches the gate; upstream generation-side filters
    // dominate — generation genuinely is the limiting factor.
    let b = breakdown(&[
        (REJECTION_NO_ELIGIBLE_SOURCES, 120),
        (REJECTION_NO_TARGET_RECORDS, 30),
        (REJECTION_TARGET_SATURATED, 45),
    ]);
    let signals = signals_from_breakdown(&b, 0);
    let class = classify(&signals, &StarvationConfig::default());

    assert_eq!(class, StarvationClass::CandidateStarved);
    assert!(
        recommend_widening(class),
        "widening must be recommended when generation is starved"
    );
    // Widening fires only if escalation had already engaged on its own
    // plateau/suppression preconditions.
    assert!(gate_escalation(true, class));
    assert!(
        !gate_escalation(false, class),
        "the gate never manufactures escalation that was not engaged"
    );
}

#[test]
fn widening_is_measurably_more_targeted_than_ungated() {
    // Acceptance criterion, expressed offline against the two representative
    // profiles: the diagnosis-driven gate widens generation on the starved
    // profile while withholding it on the over-rejected profile. An ungated
    // escalation (the prior behaviour) would widen on both — the second being
    // wasted effort / noise on the wider discovery corpus.
    let over_rejected = classify(
        &signals_from_breakdown(
            &breakdown(&[(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 2500)]),
            1,
        ),
        &StarvationConfig::default(),
    );
    let starved = classify(
        &signals_from_breakdown(&breakdown(&[(REJECTION_NO_ELIGIBLE_SOURCES, 200)]), 0),
        &StarvationConfig::default(),
    );

    let engaged = true; // both plateaued with high suppression
    let gated_widen = [over_rejected, starved]
        .iter()
        .filter(|&&c| gate_escalation(engaged, c))
        .count();
    let ungated_widen = 2; // prior behaviour: escalation fires on both

    assert_eq!(gated_widen, 1, "gate widens only the genuinely-starved run");
    assert!(
        gated_widen < ungated_widen,
        "the gate strictly reduces wasted widening versus the ungated path"
    );
}

#[test]
fn every_rejection_reason_is_classified() {
    // A newly-added rejection reason that escapes the partition would let a
    // starvation regression go unclassified — assert exhaustive coverage.
    let mut seen: HashSet<&str> = HashSet::new();
    for &r in GATE_SIDE_REJECTION_REASONS {
        assert!(seen.insert(r), "{r} classified twice");
    }
    for &r in UPSTREAM_REJECTION_REASONS {
        assert!(seen.insert(r), "{r} classified twice");
    }
    for &r in ABUNDANCE_REJECTION_REASONS {
        assert!(seen.insert(r), "{r} classified twice");
    }
    for &r in ALL_REJECTION_REASONS {
        assert!(seen.contains(r), "{r} is not classified into any partition");
    }
    assert_eq!(seen.len(), ALL_REJECTION_REASONS.len());
}

#[test]
fn empty_pass_classifies_as_starved() {
    let class = classify(
        &signals_from_breakdown(&RejectionBreakdown::new(), 0),
        &StarvationConfig::default(),
    );
    assert_eq!(class, StarvationClass::CandidateStarved);
}
