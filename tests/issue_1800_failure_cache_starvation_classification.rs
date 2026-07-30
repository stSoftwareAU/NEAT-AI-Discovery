//! Issue #1800 — fold failure-cache suppression into the breakdown *before*
//! the starvation classifier reads it (sub-issue of #1782, root cause B3).
//!
//! The FFI layer counted the cross-stack failure-cache suppression (#1447) but
//! only added it to the *surfaced* per-surface wire maps, after the classifier
//! had already run. `REJECTION_DUPLICATE_OF_FAILURE_CACHE` was therefore listed
//! as upstream (starvation) evidence yet could never contribute to the
//! classification: any pass with >= `DEFAULT_MIN_FORMED_PROPOSALS` gate-side
//! rejections was judged `ProposalRichOverRejected` and the novelty bypass
//! stayed disabled no matter how much suppression was occurring.
//!
//! These tests pin the fixed ordering end to end:
//!
//! 1. The classifier input carries `duplicate_of_failure_cache` exactly once,
//!    with the pass total.
//! 2. A pass whose candidates are all failure-cache suppressed classifies as
//!    `CandidateStarved` and recommends widening — where the same pass without
//!    the fold classified as `ProposalRichOverRejected`.

use neat_ai_discovery::analysis::candidate_starvation::{
    DEFAULT_MIN_FORMED_PROPOSALS, StarvationClass, StarvationConfig, classify, gate_escalation,
    recommend_widening, signals_from_breakdown,
};
use neat_ai_discovery::analysis::diagnostics::RejectionBreakdown;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    REJECTION_BELOW_EXPECTED_GAIN_FLOOR, REJECTION_DUPLICATE_OF_FAILURE_CACHE,
    REJECTION_FINGERPRINT_UNCHANGED,
};
use neat_ai_discovery::starvation_classifier_breakdown;

/// Candidates suppressed by the cross-stack failure cache for this pass.
const SUPPRESSED: usize = 9;
/// Gate-side rejections, at the formed-proposal floor that used to force a
/// `ProposalRichOverRejected` verdict on its own.
const GATE_SIDE: u32 = DEFAULT_MIN_FORMED_PROPOSALS;

/// Build a breakdown from `(reason, count)` pairs.
fn breakdown(entries: &[(&'static str, u32)]) -> RejectionBreakdown {
    let mut b = RejectionBreakdown::new();
    for &(reason, count) in entries {
        b.record_many(reason, count);
    }
    b
}

#[test]
fn suppression_appears_exactly_once_in_the_classifier_input() {
    let synapse = breakdown(&[(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, GATE_SIDE)]);
    let combined = starvation_classifier_breakdown(
        Some(&synapse),
        None,
        &RejectionBreakdown::new(),
        SUPPRESSED,
    );

    let suppressed = u32::try_from(SUPPRESSED).expect("count fits u32");
    assert_eq!(
        combined
            .counts()
            .get(REJECTION_DUPLICATE_OF_FAILURE_CACHE)
            .copied(),
        Some(suppressed),
        "the classifier input must carry the failure-cache suppression count"
    );
    assert_eq!(
        combined.total(),
        GATE_SIDE + suppressed,
        "the suppression count must be folded in exactly once — no double count"
    );
}

#[test]
fn both_surfaces_and_pass_drops_fold_into_one_classifier_input() {
    // Two surfaces plus the pass-level breakdown (#1781), with the pass total
    // suppression count (synapse + neuron) folded in once.
    let synapse = breakdown(&[(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 2)]);
    let neuron = breakdown(&[(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 2)]);
    let pass = breakdown(&[(REJECTION_FINGERPRINT_UNCHANGED, 1)]);

    let combined =
        starvation_classifier_breakdown(Some(&synapse), Some(&neuron), &pass, SUPPRESSED);

    let suppressed = u32::try_from(SUPPRESSED).expect("count fits u32");
    assert_eq!(
        combined
            .counts()
            .get(REJECTION_DUPLICATE_OF_FAILURE_CACHE)
            .copied(),
        Some(suppressed),
    );
    assert_eq!(
        combined
            .counts()
            .get(REJECTION_BELOW_EXPECTED_GAIN_FLOOR)
            .copied(),
        Some(GATE_SIDE),
        "both surfaces' gate-side counts merge into the classifier input"
    );
    assert_eq!(combined.total(), GATE_SIDE + 1 + suppressed);
}

#[test]
fn no_suppression_leaves_the_reason_absent() {
    let synapse = breakdown(&[(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, GATE_SIDE)]);
    let combined =
        starvation_classifier_breakdown(Some(&synapse), None, &RejectionBreakdown::new(), 0);

    assert!(
        !combined
            .counts()
            .contains_key(REJECTION_DUPLICATE_OF_FAILURE_CACHE),
        "a pass with no suppression must not gain a present-and-zero reason"
    );
}

#[test]
fn suppressed_pass_with_gate_side_rejections_is_candidate_starved() {
    let synapse = breakdown(&[(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, GATE_SIDE)]);
    let combined = starvation_classifier_breakdown(
        Some(&synapse),
        None,
        &RejectionBreakdown::new(),
        SUPPRESSED,
    );

    // Zero candidates survive the pass.
    let signals = signals_from_breakdown(&combined, 0);
    assert_eq!(
        signals.upstream_rejections,
        u32::try_from(SUPPRESSED).expect("count fits u32")
    );
    assert_eq!(signals.gate_side_rejections, GATE_SIDE);

    let class = classify(&signals, &StarvationConfig::default());
    assert_eq!(
        class,
        StarvationClass::CandidateStarved,
        "failure-cache suppression outweighing the formed proposals is starvation evidence"
    );
    assert!(
        recommend_widening(class),
        "the bypass must be allowed to engage on a suppression-dominated pass"
    );
    assert!(gate_escalation(true, class));
}

#[test]
fn without_the_fold_the_same_pass_was_over_rejected() {
    // The pre-#1800 ordering: the classifier saw the breakdown with the
    // suppression count still missing, so the bypass stayed disabled. This pins
    // the regression the fold repairs.
    let unfolded = breakdown(&[(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, GATE_SIDE)]);
    let class = classify(
        &signals_from_breakdown(&unfolded, 0),
        &StarvationConfig::default(),
    );

    assert_eq!(class, StarvationClass::ProposalRichOverRejected);
    assert!(!recommend_widening(class));
}
