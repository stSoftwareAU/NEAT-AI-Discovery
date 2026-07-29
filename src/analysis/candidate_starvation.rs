//! Candidate-starvation classifier for the large converged production network
//! (Issue #1739).
//!
//! The milestone-#1736 diagnosis (`docs/analysis/rejection-diagnosis-1737.md`,
//! Issue #1737) established that the large converged production creature is
//! **not**
//! candidate-starved: its generators emit `change-squash`, coordinated multi-op
//! `change-squash`, and `remove-neuron` candidates in quantity — they are then
//! discarded at the expected-gain gate because the gain estimate has collapsed
//! to (near) zero (the estimator/threshold work of #1738 / #1740). Widening
//! *generation* on that profile cannot lift the accepted rate and would only add
//! noise: the run is **proposal-rich but over-rejected**, not starved.
//!
//! Yet the existing novelty/diversification escalation
//! ([`super::novelty_escalation::decide_escalation`]) engages
//! on `low rolling success rate + high cache-suppression` alone. On that
//! converged profile the success rate is ~0 and much of the pool is suppressed,
//! so escalation would fire and tell the host to widen generation — exactly the
//! wasted widening the diagnosis warns against, and a regression/noise risk on
//! any other converged creature of the same shape (the wider discovery corpus).
//!
//! This module supplies the missing decision from the issue's scope: **is the
//! run candidate-starved (few proposals ever reach the accept gate) or
//! proposal-rich but over-rejected (many reach the gate and are rejected
//! there)?** It reads the diagnosis output — the
//! [`RejectionBreakdown`] already
//! emitted on analysis metadata (Issue #1129) — and classifies the run, so
//! generation widening is gated to the *starved* case only.
//!
//! # How the classification is derived
//!
//! Every stable rejection reason is partitioned into exactly one of three
//! buckets ([`GATE_SIDE_REJECTION_REASONS`], [`UPSTREAM_REJECTION_REASONS`],
//! [`ABUNDANCE_REJECTION_REASONS`]) — a partition pinned exhaustively against
//! [`reasons::ALL_REJECTION_REASONS`]
//! by a unit test so a newly-added reason cannot silently escape classification:
//!
//! - **Gate-side** — the candidate reached the expected-gain / acceptance /
//!   scoring gate and was rejected or discounted *there*. These are
//!   over-rejection evidence.
//! - **Upstream** — the candidate never reached the gate: too little recorded
//!   data, no eligible sources, a structural pre-check, a saturated target, or
//!   stale-proposal (failure-cache) suppression. These are starvation evidence.
//! - **Abundance** — the candidate *was* generated but was capped or truncated
//!   because there were already too many proposals. These are proof the
//!   generator is *not* starved.
//!
//! A candidate that reaches the gate is either accepted (survives) or counted
//! gate-side, so `reaching_gate = accepted + gate_side_rejections`. Proposals
//! that were formed but never scored (abundance) still prove generation is
//! productive, so `proposals_formed = reaching_gate + abundance_rejections`.
//!
//! # Safety
//!
//! Pure logic only — no GPU, Parquet, or FFI dependencies, mirroring the
//! self-contained design of [`super::novelty_escalation`].
//! It **never accepts a candidate**: it only decides whether the
//! generation-widening *hint* may fire, so the #1623 evaluate-before-accept
//! gate is entirely untouched.

use super::diagnostics::RejectionBreakdown;
use super::diagnostics::rejection_reasons as reasons;

/// Rejection reasons recorded when a candidate reached the expected-gain /
/// acceptance / scoring gate and was rejected or discounted *there*.
///
/// A run dominated by these is **proposal-rich but over-rejected** — the
/// generators are working; the accept gate (estimator / threshold) is the
/// bottleneck. This is the converged production profile from the #1737
/// diagnosis.
pub const GATE_SIDE_REJECTION_REASONS: &[&str] = &[
    reasons::REJECTION_BELOW_EXPECTED_GAIN_FLOOR,
    reasons::REJECTION_BELOW_MULTI_OP_FLOOR,
    reasons::REJECTION_NON_POSITIVE_GAIN,
    reasons::REJECTION_NON_FINITE_GAIN,
    reasons::REJECTION_SATURATION_DISCOUNTED_TO_ZERO,
    reasons::REJECTION_PESSIMISM_DISCOUNTED_TO_ZERO,
    reasons::REJECTION_INTERFERENCE_FILTERED,
    reasons::REJECTION_BELOW_IMPROVED_RATIO,
    reasons::REJECTION_ZERO_IMPROVEMENT,
    reasons::REJECTION_BELOW_THRESHOLD,
    reasons::REJECTION_REMOVAL_BELOW_NOISE_FLOOR,
    reasons::REJECTION_REMOVE_NEURON_DROUGHT_DEPRIORITISED,
];

/// Rejection reasons recorded when a candidate never reached the gate: too
/// little recorded data, no eligible sources, a structural pre-check, a
/// saturated target, or stale-proposal (failure-cache) suppression.
///
/// A run dominated by these is **candidate-starved** — generation is the
/// limiting factor, so widening (novelty escalation) is warranted.
pub const UPSTREAM_REJECTION_REASONS: &[&str] = &[
    reasons::REJECTION_DUPLICATE_OF_FAILURE_CACHE,
    reasons::REJECTION_ADD_SYNAPSE_GATED,
    reasons::REJECTION_CPU_PRE_REJECT_NO_SIGNAL,
    reasons::REJECTION_NO_SAMPLES,
    reasons::REJECTION_NO_TARGET_RECORDS,
    reasons::REJECTION_INSUFFICIENT_RECORDING,
    reasons::REJECTION_NO_ELIGIBLE_SOURCES,
    reasons::REJECTION_INPUT_NEURON_FILTERED,
    reasons::REJECTION_HIDDEN_NEURON_FILTERED,
    reasons::REJECTION_CONSTANT_NEURON_FILTERED,
    reasons::REJECTION_NO_DIAGNOSTICS,
    reasons::REJECTION_TARGET_SATURATED,
    // Issue #1781: focus neurons skipped by the structural fingerprint cache
    // were never analysed at all, so no proposal could reach the gate.
    reasons::REJECTION_FINGERPRINT_UNCHANGED,
];

/// Rejection reasons recorded when a candidate *was* generated but was capped or
/// truncated because too many proposals already existed.
///
/// These are proof the generator is **not** starved — they corroborate
/// proposal-richness rather than either failure mode.
pub const ABUNDANCE_REJECTION_REASONS: &[&str] = &[
    reasons::REJECTION_BUDGET_TRUNCATED,
    reasons::REJECTION_PER_TARGET_CAP,
    reasons::REJECTION_SAME_TARGET_SQUASH_DUPLICATE,
    reasons::REJECTION_COORDINATED_TARGET_CAP_EXCEEDED,
    reasons::REJECTION_COORDINATED_COLLAPSE_BYPASS_WEIGHT_BELOW_FLOOR,
];

/// Default minimum number of *formed* proposals below which a run with a
/// collapsed accept rate is judged candidate-starved rather than over-rejected.
///
/// Generation-limited passes form (near) zero proposals; a converged network
/// that is merely over-rejected still forms many. The exact value is not
/// sensitive — the two profiles sit orders of magnitude apart — so a small,
/// conservative floor cleanly separates them.
pub const DEFAULT_MIN_FORMED_PROPOSALS: u32 = 4;

/// Default surviving/accepted fraction of gate-reaching candidates at or above
/// which a run is judged healthy (accepting normally) rather than in either
/// failure mode.
///
/// The converged production profile accepts ~2 candidates out of thousands that reach the gate
/// (rate ~1e-4), far below this floor, so it is never misclassified as healthy;
/// a creature genuinely accepting improvements sits well above it.
pub const DEFAULT_HEALTHY_ACCEPT_RATE: f32 = 0.02;

/// Thresholds for [`classify`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StarvationConfig {
    /// See [`DEFAULT_MIN_FORMED_PROPOSALS`].
    pub min_formed_proposals: u32,
    /// See [`DEFAULT_HEALTHY_ACCEPT_RATE`].
    pub healthy_accept_rate: f32,
}

impl Default for StarvationConfig {
    fn default() -> Self {
        Self {
            min_formed_proposals: DEFAULT_MIN_FORMED_PROPOSALS,
            healthy_accept_rate: DEFAULT_HEALTHY_ACCEPT_RATE,
        }
    }
}

/// Generation signals extracted from a discovery pass for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GenerationSignals {
    /// Candidates that survived the Rust accept gate (returned to the host).
    pub accepted: u32,
    /// Candidates rejected *at* the gate ([`GATE_SIDE_REJECTION_REASONS`]).
    pub gate_side_rejections: u32,
    /// Candidates dropped *before* the gate ([`UPSTREAM_REJECTION_REASONS`]).
    pub upstream_rejections: u32,
    /// Candidates formed but capped/truncated ([`ABUNDANCE_REJECTION_REASONS`]).
    pub abundance_rejections: u32,
}

impl GenerationSignals {
    /// Candidates that reached the accept gate (accepted or rejected there).
    #[must_use]
    pub fn reaching_gate(&self) -> u32 {
        self.accepted.saturating_add(self.gate_side_rejections)
    }

    /// Candidates the generator provably formed (reached the gate, or were
    /// capped/truncated as surplus).
    #[must_use]
    pub fn proposals_formed(&self) -> u32 {
        self.reaching_gate()
            .saturating_add(self.abundance_rejections)
    }
}

/// Extract [`GenerationSignals`] from a diagnosis [`RejectionBreakdown`] plus the
/// surviving/accepted count for the pass.
///
/// Reasons outside the three partitions are ignored (there are none while the
/// partition-coverage test passes). `accepted` is the number of candidates
/// returned to the host — those that survived the Rust accept gate.
#[must_use]
pub fn signals_from_breakdown(breakdown: &RejectionBreakdown, accepted: u32) -> GenerationSignals {
    let sum = |group: &[&str]| -> u32 {
        group
            .iter()
            .map(|r| breakdown.counts().get(*r).copied().unwrap_or(0))
            .fold(0u32, u32::saturating_add)
    };
    GenerationSignals {
        accepted,
        gate_side_rejections: sum(GATE_SIDE_REJECTION_REASONS),
        upstream_rejections: sum(UPSTREAM_REJECTION_REASONS),
        abundance_rejections: sum(ABUNDANCE_REJECTION_REASONS),
    }
}

/// Classification of a discovery pass's candidate-generation health.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StarvationClass {
    /// The pass is accepting candidates at a healthy rate — neither failure
    /// mode applies.
    Healthy,
    /// Few proposals ever reach the accept gate: generation is the limiting
    /// factor and widening is warranted.
    CandidateStarved,
    /// Many proposals reach the gate but are rejected there: the accept gate
    /// (estimator / threshold) is the bottleneck, not generation. This is the
    /// converged production profile — widening cannot help and would only add
    /// noise.
    ProposalRichOverRejected,
}

/// Classify a discovery pass from its generation signals.
///
/// 1. If the pass is accepting candidates at or above
///    [`StarvationConfig::healthy_accept_rate`], it is [`Healthy`].
/// 2. Otherwise, with the accept rate collapsed, the run is
///    [`ProposalRichOverRejected`] when the generator provably formed at least
///    [`StarvationConfig::min_formed_proposals`] proposals, and
///    [`CandidateStarved`] when it did not.
///
/// [`Healthy`]: StarvationClass::Healthy
/// [`ProposalRichOverRejected`]: StarvationClass::ProposalRichOverRejected
/// [`CandidateStarved`]: StarvationClass::CandidateStarved
#[must_use]
pub fn classify(signals: &GenerationSignals, cfg: &StarvationConfig) -> StarvationClass {
    let reaching = signals.reaching_gate();
    if signals.accepted > 0 && reaching > 0 {
        let rate = f64::from(signals.accepted) / f64::from(reaching);
        if rate >= f64::from(cfg.healthy_accept_rate) {
            return StarvationClass::Healthy;
        }
    }

    if signals.proposals_formed() >= cfg.min_formed_proposals {
        StarvationClass::ProposalRichOverRejected
    } else {
        StarvationClass::CandidateStarved
    }
}

/// Whether generation widening (novelty escalation) is warranted for `class`.
///
/// Only a genuinely [`CandidateStarved`](StarvationClass::CandidateStarved) run
/// benefits from widening. On a
/// [`ProposalRichOverRejected`](StarvationClass::ProposalRichOverRejected) run
/// the accept gate is the bottleneck, so widening cannot lift the accepted rate
/// and would only add noise (the #1737 diagnosis conclusion for the large
/// converged production network).
#[must_use]
pub fn recommend_widening(class: StarvationClass) -> bool {
    matches!(class, StarvationClass::CandidateStarved)
}

/// Gate an escalation decision by the starvation classification.
///
/// Returns `true` only when escalation was already `engaged` **and** the run is
/// genuinely candidate-starved. This lets the existing
/// [`super::novelty_escalation`] gate keep its plateau/
/// suppression preconditions while adding the diagnosis-driven guard that
/// prevents wasted widening on proposal-rich-but-over-rejected profiles.
#[must_use]
pub fn gate_escalation(engaged: bool, class: StarvationClass) -> bool {
    engaged && recommend_widening(class)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn breakdown(entries: &[(&'static str, u32)]) -> RejectionBreakdown {
        let mut b = RejectionBreakdown::new();
        for &(reason, count) in entries {
            b.record_many(reason, count);
        }
        b
    }

    #[test]
    fn partitions_cover_every_reason_exactly_once() {
        let mut seen: HashSet<&str> = HashSet::new();
        for group in [
            GATE_SIDE_REJECTION_REASONS,
            UPSTREAM_REJECTION_REASONS,
            ABUNDANCE_REJECTION_REASONS,
        ] {
            for &reason in group {
                assert!(
                    seen.insert(reason),
                    "reason {reason} appears in more than one partition"
                );
            }
        }
        for &reason in reasons::ALL_REJECTION_REASONS {
            assert!(
                seen.contains(reason),
                "reason {reason} is not classified into any starvation partition"
            );
        }
        assert_eq!(
            seen.len(),
            reasons::ALL_REJECTION_REASONS.len(),
            "partitions classify a reason not present in ALL_REJECTION_REASONS"
        );
    }

    #[test]
    fn converged_production_profile_is_proposal_rich_over_rejected() {
        // The #1737 diagnosis shape: thousands of candidates reach the gate and
        // are rejected there (gain collapse); a rare acceptance survives.
        let b = breakdown(&[
            (reasons::REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 1800),
            (reasons::REJECTION_BELOW_MULTI_OP_FLOOR, 600),
            (reasons::REJECTION_NON_POSITIVE_GAIN, 400),
        ]);
        let signals = signals_from_breakdown(&b, 2);
        assert_eq!(signals.gate_side_rejections, 2800);
        assert_eq!(signals.upstream_rejections, 0);
        assert!(signals.reaching_gate() >= 2800);

        let class = classify(&signals, &StarvationConfig::default());
        assert_eq!(class, StarvationClass::ProposalRichOverRejected);
        assert!(
            !recommend_widening(class),
            "widening must not be recommended on the over-rejected production profile"
        );
    }

    #[test]
    fn starved_profile_is_candidate_starved() {
        // Almost nothing reaches the gate; upstream filters dominate.
        let b = breakdown(&[
            (reasons::REJECTION_NO_ELIGIBLE_SOURCES, 120),
            (reasons::REJECTION_NO_TARGET_RECORDS, 30),
            (reasons::REJECTION_TARGET_SATURATED, 45),
        ]);
        let signals = signals_from_breakdown(&b, 0);
        assert_eq!(signals.gate_side_rejections, 0);
        assert_eq!(signals.upstream_rejections, 195);
        assert_eq!(signals.proposals_formed(), 0);

        let class = classify(&signals, &StarvationConfig::default());
        assert_eq!(class, StarvationClass::CandidateStarved);
        assert!(recommend_widening(class));
    }

    #[test]
    fn empty_pass_is_candidate_starved() {
        // Nothing generated, nothing accepted, no rejections recorded.
        let class = classify(&GenerationSignals::default(), &StarvationConfig::default());
        assert_eq!(class, StarvationClass::CandidateStarved);
    }

    #[test]
    fn healthy_pass_is_neither_failure_mode() {
        let b = breakdown(&[(reasons::REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 10)]);
        let signals = signals_from_breakdown(&b, 5); // 5 of 15 reaching gate accepted
        let class = classify(&signals, &StarvationConfig::default());
        assert_eq!(class, StarvationClass::Healthy);
        assert!(!recommend_widening(class));
    }

    #[test]
    fn abundance_counts_as_proposals_formed_not_starved() {
        // Candidates were truncated because there were too many — the generator
        // is productive even though nothing survived (e.g. all over budget).
        let b = breakdown(&[(reasons::REJECTION_BUDGET_TRUNCATED, 500)]);
        let signals = signals_from_breakdown(&b, 0);
        assert_eq!(signals.abundance_rejections, 500);
        assert_eq!(signals.proposals_formed(), 500);
        let class = classify(&signals, &StarvationConfig::default());
        assert_eq!(class, StarvationClass::ProposalRichOverRejected);
    }

    #[test]
    fn gate_escalation_only_fires_when_starved() {
        assert!(gate_escalation(true, StarvationClass::CandidateStarved));
        // Engaged but over-rejected → suppressed (the over-rejected-profile guard).
        assert!(!gate_escalation(
            true,
            StarvationClass::ProposalRichOverRejected
        ));
        assert!(!gate_escalation(true, StarvationClass::Healthy));
        // Never fires when escalation was not engaged in the first place.
        assert!(!gate_escalation(false, StarvationClass::CandidateStarved));
    }

    #[test]
    fn classify_respects_custom_thresholds() {
        // Raise the formed-proposal floor so a modest proposal count is judged
        // starved rather than over-rejected.
        let b = breakdown(&[(reasons::REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 3)]);
        let signals = signals_from_breakdown(&b, 0);
        let strict = StarvationConfig {
            min_formed_proposals: 10,
            ..StarvationConfig::default()
        };
        assert_eq!(
            classify(&signals, &strict),
            StarvationClass::CandidateStarved
        );
        // Under the default floor (4) the same 3 proposals stay starved too.
        assert_eq!(
            classify(&signals, &StarvationConfig::default()),
            StarvationClass::CandidateStarved
        );
    }
}
