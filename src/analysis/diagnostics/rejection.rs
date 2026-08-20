//! Synapse rejection tracking and diagnostic reporting.
//!
//! Tracks why synapse candidates were rejected during analysis, including:
//! - No overlapping samples between source and target
//! - Zero consistent improvement from GPU evaluation
//! - Expected improvement below threshold
//!
//! Uses `DashMap` for lock-free concurrent access from parallel analysis threads.

use dashmap::DashMap;
use std::fmt;

use crate::analysis::shared::{
    SynapseNoCandidateDetail, SynapseNoCandidateReason, SynapseNoCandidateSummary,
};
use crate::analysis::utils::verbose_enabled;

// =============================================================================
// Synapse Rejection Tracking
// =============================================================================

/// Why a synapse candidate was rejected during analysis.
#[derive(Clone, Copy)]
pub(crate) enum RejectionReason {
    /// No overlapping discovery samples between source and target.
    NoSamples,
    /// No consistent improvement in GPU stats.
    ZeroImprovement,
    /// Expected improvement below threshold.
    BelowThreshold,
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RejectionReason::NoSamples => write!(f, "no overlapping discovery samples"),
            RejectionReason::ZeroImprovement => write!(f, "no consistent improvement in GPU stats"),
            RejectionReason::BelowThreshold => write!(f, "expected improvement below threshold"),
        }
    }
}

/// Detailed rejection information for a synapse candidate.
#[derive(Clone)]
pub(crate) struct RejectionDetail {
    pub(crate) source_uuid: String,
    pub(crate) reason: RejectionReason,
    pub(crate) sample_count: usize,
    pub(crate) source_record_count: usize,
    pub(crate) improved_count: u32,
    pub(crate) worsened_count: u32,
    pub(crate) expected_improvement: f32,
    pub(crate) threshold: f32,
    pub(crate) weight: Option<f32>,
}

impl RejectionDetail {
    pub(crate) fn score(&self) -> f32 {
        self.expected_improvement
    }
}

/// Threshold calculation context for recording below-threshold rejections.
pub(crate) struct ThresholdContext {
    pub(crate) sample_count: usize,
    pub(crate) expected_improvement: f32,
    pub(crate) threshold: f32,
    pub(crate) improved_count: u32,
    pub(crate) worsened_count: u32,
    pub(crate) weight: f32,
}

/// Per-target diagnostic entry for synapse analysis.
#[derive(Clone)]
pub(crate) struct TargetDiagnosticEntry {
    pub(crate) target_uuid: String,
    pub(crate) target_record_count: usize,
    pub(crate) evaluated_candidates: u32,
    pub(crate) candidates_with_samples: u32,
    pub(crate) total_eligible_sources: u32,
    pub(crate) input_neuron_count: u32,
    pub(crate) already_connected_count: u32,
    pub(crate) record_load_failures: u32,
    pub(crate) had_candidate: bool,
    pub(crate) best_rejection: Option<RejectionDetail>,
    /// Issue #1018: Count of candidates accepted via Metropolis-Hastings
    /// probabilistic acceptance despite being below the threshold.
    pub(crate) accepted_below_threshold_count: u32,
}

impl TargetDiagnosticEntry {
    pub(crate) fn new(target_uuid: &str) -> Self {
        Self {
            target_uuid: target_uuid.to_string(),
            target_record_count: 0,
            evaluated_candidates: 0,
            candidates_with_samples: 0,
            total_eligible_sources: 0,
            input_neuron_count: 0,
            already_connected_count: 0,
            record_load_failures: 0,
            had_candidate: false,
            best_rejection: None,
            accepted_below_threshold_count: 0,
        }
    }

    pub(crate) fn update_best(&mut self, detail: RejectionDetail) {
        match &self.best_rejection {
            Some(current) => {
                if detail.score() > current.score()
                    || (detail.score() == current.score()
                        && detail.sample_count > current.sample_count)
                {
                    self.best_rejection = Some(detail);
                }
            }
            None => self.best_rejection = Some(detail),
        }
    }
}

/// Collection of target diagnostics for synapse analysis.
///
/// Uses `DashMap` internally for lock-free concurrent access (Issue #216).
/// All methods take `&self` instead of `&mut self` to allow concurrent updates
/// from multiple threads without external synchronisation.
pub(crate) struct TargetDiagnostics {
    log_enabled: bool,
    /// Lock-free concurrent map for diagnostic entries.
    /// Each focus neuron is processed by a separate thread, and diagnostics
    /// are recorded without contention using `DashMap`'s sharded internal structure.
    pub(crate) entries: DashMap<String, TargetDiagnosticEntry>,
    /// Count of new add-synapse sources skipped because the target neuron
    /// is saturated (Issue #1143). Surfaced via the
    /// `REJECTION_TARGET_SATURATED` entry on `synapseMetadata`.
    target_saturated_drops: std::sync::atomic::AtomicU32,
    /// Count of helpful add-synapse candidates dropped by the CPU pre-reject
    /// screen before GPU submit (Issue #1544). Surfaced via the
    /// `REJECTION_CPU_PRE_REJECT_NO_SIGNAL` entry on `synapseMetadata`.
    cpu_pre_reject_no_signal_drops: std::sync::atomic::AtomicU32,
}

impl TargetDiagnostics {
    pub(crate) fn new(targets: &[&String]) -> Self {
        let log_enabled = verbose_enabled();
        let entries = DashMap::new();
        for target in targets {
            entries.insert((*target).clone(), TargetDiagnosticEntry::new(target));
        }
        Self {
            log_enabled,
            entries,
            target_saturated_drops: std::sync::atomic::AtomicU32::new(0),
            cpu_pre_reject_no_signal_drops: std::sync::atomic::AtomicU32::new(0),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_tests(targets: &[&str]) -> Self {
        let entries = DashMap::new();
        for target in targets {
            entries.insert((*target).to_string(), TargetDiagnosticEntry::new(target));
        }
        Self {
            log_enabled: true,
            entries,
            target_saturated_drops: std::sync::atomic::AtomicU32::new(0),
            cpu_pre_reject_no_signal_drops: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Record that `count` new add-synapse sources were dropped because the
    /// target neuron is saturated (Issue #1143).
    pub(crate) fn record_target_saturated_drops(&self, count: u32) {
        if count > 0 {
            self.target_saturated_drops
                .fetch_add(count, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Snapshot of the target-saturated-drop counter (Issue #1143).
    pub(crate) fn target_saturated_drop_count(&self) -> u32 {
        self.target_saturated_drops
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record that `count` helpful add-synapse candidates were dropped by the
    /// CPU pre-reject screen before GPU submit (Issue #1544).
    pub(crate) fn record_cpu_pre_reject_no_signal(&self, count: u32) {
        if count > 0 {
            self.cpu_pre_reject_no_signal_drops
                .fetch_add(count, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Snapshot of the CPU pre-reject no-signal drop counter (Issue #1544).
    pub(crate) fn cpu_pre_reject_no_signal_drop_count(&self) -> u32 {
        self.cpu_pre_reject_no_signal_drops
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn set_target_record_count(&self, target_uuid: &str, count: usize) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.target_record_count = count;
        }
    }

    pub(crate) fn set_total_eligible_sources(&self, target_uuid: &str, count: u32) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.total_eligible_sources = count;
        }
    }

    pub(crate) fn set_input_neuron_count(&self, target_uuid: &str, count: u32) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.input_neuron_count = count;
        }
    }

    pub(crate) fn record_already_connected(&self, target_uuid: &str) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.already_connected_count += 1;
        }
    }

    pub(crate) fn record_load_failure(&self, target_uuid: &str) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.record_load_failures += 1;
        }
    }

    pub(crate) fn record_candidate_attempt(&self, target_uuid: &str, had_samples: bool) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.evaluated_candidates += 1;
            if had_samples {
                entry.candidates_with_samples += 1;
            }
        }
    }

    pub(crate) fn record_no_samples(
        &self,
        target_uuid: &str,
        source_uuid: &str,
        source_record_count: usize,
    ) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::NoSamples,
                sample_count: 0,
                source_record_count,
                improved_count: 0,
                worsened_count: 0,
                expected_improvement: f32::NEG_INFINITY,
                threshold: 0.0,
                weight: None,
            });
        }
    }

    pub(crate) fn record_zero_improvement(
        &self,
        target_uuid: &str,
        source_uuid: &str,
        sample_count: usize,
        positive_count: u32,
        negative_count: u32,
    ) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::ZeroImprovement,
                sample_count,
                source_record_count: sample_count,
                improved_count: positive_count.max(negative_count),
                worsened_count: positive_count.min(negative_count),
                expected_improvement: 0.0,
                threshold: 0.0,
                weight: None,
            });
        }
    }

    pub(crate) fn record_below_threshold(
        &self,
        target_uuid: &str,
        source_uuid: &str,
        context: ThresholdContext,
    ) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::BelowThreshold,
                sample_count: context.sample_count,
                source_record_count: context.sample_count,
                improved_count: context.improved_count,
                worsened_count: context.worsened_count,
                expected_improvement: context.expected_improvement,
                threshold: context.threshold,
                weight: Some(context.weight),
            });
        }
    }

    /// Issue #1018: Record that a candidate was accepted below threshold
    /// via Metropolis-Hastings probabilistic acceptance.
    pub(crate) fn record_accepted_below_threshold(&self, target_uuid: &str) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.accepted_below_threshold_count += 1;
        }
    }

    pub(crate) fn mark_candidate_selected(&self, target_uuid: &str) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.had_candidate = true;
        }
    }

    /// Snapshot one per-target verdict per target for the global cooldown
    /// tracker (Issue #1791).
    ///
    /// A target counts as evaluated once at least one candidate source was
    /// actually tried against it. Targets the pass never reached (deadline,
    /// cooldown) contribute no evidence and must not move their streak.
    pub(crate) fn pass_outcomes(
        &self,
    ) -> Vec<crate::analysis::target_pass_outcomes::TargetPassOutcome> {
        self.entries
            .iter()
            .map(|entry_ref| {
                let entry = entry_ref.value();
                crate::analysis::target_pass_outcomes::TargetPassOutcome::new(
                    entry.target_uuid.clone(),
                    entry.had_candidate,
                    entry.evaluated_candidates > 0,
                )
            })
            .collect()
    }

    pub(crate) fn emit_logs(&self) {
        if !self.log_enabled {
            return;
        }

        for entry_ref in &self.entries {
            let entry = entry_ref.value();

            // Issue #1018: Log Metropolis-Hastings acceptance counts
            if entry.accepted_below_threshold_count > 0 {
                tracing::trace!(
                    target_uuid = %entry.target_uuid,
                    accepted_below_threshold = entry.accepted_below_threshold_count,
                    "Metropolis-Hastings accepted candidates below threshold"
                );
            }

            if entry.had_candidate {
                continue;
            }

            // Issue #1101: Log a specific warning when the target neuron has zero
            // Parquet records — the recording phase likely timed out.
            if entry.target_record_count == 0 && entry.evaluated_candidates == 0 {
                tracing::warn!(
                    target_uuid = %entry.target_uuid,
                    "Target neuron has zero activation records in Parquet — \
                     recording phase may have timed out or produced insufficient data"
                );
                continue;
            }

            if entry.total_eligible_sources == 0 {
                // This should never happen - input/constant neurons are skipped early
                // and hidden/output neurons should always have at least input neurons as eligible sources
                // Skip logging to avoid cluttering logs with impossible conditions
                continue;
            }

            // Check if neuron is fully connected (all eligible sources already have synapses)
            // Eligible sources include: ALL input neurons (input-0 through input-(creature.input-1))
            // AND ALL prior hidden/output neurons (with index < target_index, excluding constants)
            // This condition is rare - only occurs when neuron is connected to all possible sources
            if entry.already_connected_count == entry.total_eligible_sources {
                tracing::trace!(
                    target_uuid = %entry.target_uuid,
                    eligible_sources = entry.total_eligible_sources,
                    input_neuron_count = entry.input_neuron_count,
                    "Target is fully connected: all eligible upstream sources already have synapses"
                );
                continue;
            }

            // Check for record loading failures (this indicates a bug)
            if entry.record_load_failures > 0 {
                tracing::warn!(
                    target_uuid = %entry.target_uuid,
                    record_load_failures = entry.record_load_failures,
                    "Target had record loading failures (this may indicate a bug — records exist but couldn't be loaded)"
                );
            }

            if entry.evaluated_candidates == 0 {
                tracing::trace!(
                    target_uuid = %entry.target_uuid,
                    eligible_sources = entry.total_eligible_sources,
                    already_connected = entry.already_connected_count,
                    record_load_failures = entry.record_load_failures,
                    "Target had eligible upstream neurons but none were evaluated"
                );
                continue;
            }

            let best = match &entry.best_rejection {
                Some(detail) => detail,
                None => {
                    tracing::trace!(
                        target_uuid = %entry.target_uuid,
                        evaluated_candidates = entry.evaluated_candidates,
                        "Target evaluated potential synapses but recorded no diagnostics"
                    );
                    continue;
                }
            };

            match best.reason {
                RejectionReason::NoSamples => {
                    tracing::trace!(
                        target_uuid = %entry.target_uuid,
                        source_uuid = %best.source_uuid,
                        source_record_count = best.source_record_count,
                        target_record_count = entry.target_record_count,
                        "Target skipped candidate — no aligned samples were available"
                    );
                }
                RejectionReason::ZeroImprovement => {
                    tracing::trace!(
                        target_uuid = %entry.target_uuid,
                        sample_count = best.sample_count,
                        source_uuid = %best.source_uuid,
                        improved_count = best.improved_count,
                        worsened_count = best.worsened_count,
                        "Target saw aligned samples but GPU stats reported zero consistent improvements"
                    );
                }
                RejectionReason::BelowThreshold => {
                    tracing::trace!(
                        target_uuid = %entry.target_uuid,
                        source_uuid = %best.source_uuid,
                        expected_improvement = best.expected_improvement,
                        threshold = best.threshold,
                        improved_count = best.improved_count,
                        worsened_count = best.worsened_count,
                        suggested_weight = best.weight,
                        "Target best candidate improved but remained below threshold"
                    );
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn entry_for(&self, target_uuid: &str) -> Option<TargetDiagnosticEntry> {
        self.entries.get(target_uuid).map(|r| r.value().clone())
    }

    pub(crate) fn no_candidate_summaries(&self) -> Vec<SynapseNoCandidateSummary> {
        self.entries
            .iter()
            .filter(|entry_ref| !entry_ref.value().had_candidate)
            .map(|entry_ref| {
                let entry = entry_ref.value();

                // Issue #1101: When the target neuron has zero records in the Parquet
                // file, report the specific reason rather than the misleading
                // "NoEligibleSources". This typically occurs when the recording phase
                // timed out before capturing data for this neuron.
                if entry.target_record_count == 0 && entry.evaluated_candidates == 0 {
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: SynapseNoCandidateReason::NoTargetRecords,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                // Only report "no eligible sources" if both total_eligible_sources and evaluated_candidates are 0
                // This handles the case where total_eligible_sources might be 0 in tests but evaluated_candidates > 0
                if entry.total_eligible_sources == 0 && entry.evaluated_candidates == 0 {
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: SynapseNoCandidateReason::NoEligibleSources,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                // Check if neuron is fully connected (all eligible sources already have synapses)
                // Eligible sources include: ALL input neurons (input-0 through input-(creature.input-1))
                // AND ALL prior hidden/output neurons (with index < target_index, excluding constants)
                // This condition is rare - only occurs when neuron is connected to all possible sources
                if entry.total_eligible_sources > 0
                    && entry.already_connected_count == entry.total_eligible_sources
                    && entry.evaluated_candidates == 0
                {
                    // Neuron is fully connected - all eligible sources (all inputs + all prior hidden neurons) already have synapses
                    // This is legitimate but rare, and we report it as "no eligible sources"
                    // since there are no NEW sources to evaluate
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: SynapseNoCandidateReason::NoEligibleSources,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                if let Some(best) = &entry.best_rejection {
                    let reason = match best.reason {
                        RejectionReason::NoSamples => SynapseNoCandidateReason::NoSamples,
                        RejectionReason::ZeroImprovement => {
                            SynapseNoCandidateReason::ZeroImprovement
                        }
                        RejectionReason::BelowThreshold => SynapseNoCandidateReason::BelowThreshold,
                    };
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: Some(SynapseNoCandidateDetail {
                            source_uuid: Some(best.source_uuid.clone()),
                            sample_count: Some(best.sample_count),
                            source_record_count: Some(best.source_record_count),
                            improved_count: Some(best.improved_count),
                            worsened_count: Some(best.worsened_count),
                            expected_improvement: Some(best.expected_improvement),
                            threshold: Some(best.threshold),
                            suggested_weight: best.weight,
                        }),
                    };
                }

                SynapseNoCandidateSummary {
                    target_uuid: entry.target_uuid.clone(),
                    reason: SynapseNoCandidateReason::NoDiagnostics,
                    evaluated_candidates: entry.evaluated_candidates,
                    candidates_with_samples: entry.candidates_with_samples,
                    target_record_count: entry.target_record_count,
                    detail: None,
                }
            })
            .collect()
    }
}

// =============================================================================
// Target-saturation early abort (Issue #4140)
// =============================================================================

/// Minimum rejection sample before a saturation-dominant pass may abort.
///
/// Chosen from the recorded 40-minute zero-candidate diagnostics: the
/// pass had `upstream_rejections=3080` with `dominant_rejection_reason=
/// "target_saturated"`. Eighty rejections is large enough to be a meaningful
/// sample and small enough to trip well before a tens-of-minutes deadline.
pub const TARGET_SATURATED_EARLY_EXIT_MIN_SAMPLE: u32 = 80;

/// Fraction of the rejection sample that must be `target_saturated` to abort.
///
/// 80 % is "overwhelming majority": a productive mix of other rejections keeps
/// the pass running, while a pass that is almost entirely saturation-blocked
/// stops immediately. `within_batch_target_short_circuit` is **not** counted
/// here — it only ends a batch, not the whole pass.
pub const TARGET_SATURATED_EARLY_EXIT_RATIO: f64 = 0.80;

/// Minimum candidates the pass must have evaluated before it may abort.
///
/// A saturated target drops **every** eligible source at once, and the source
/// budget is unlimited by default, so one heavily-connected saturated target
/// can push `target_saturated_drop_count()` past the min-sample floor before
/// the pass has evaluated a single candidate. Aborting there would skip every
/// remaining target on the evidence of one — the over-aggression Issue #4140's
/// second acceptance criterion forbids. Requiring the pass to have considered
/// real candidates first keeps the abort meaning "we tried and it is hopeless".
///
/// 32 sits well below the recorded regression (`proposals_formed=52`), so the
/// 40-minute pass still trips, and well above zero, so a first-target drop
/// cannot end the pass on its own.
pub const TARGET_SATURATED_EARLY_EXIT_MIN_CONSIDERED: u32 = 32;

/// Whether a pass should abort because `target_saturated` dominates.
///
/// `target_saturated_drops` must come from
/// [`TargetDiagnostics::target_saturated_drop_count`] /
/// [`crate::analysis::diagnostics::NeuronDiagnostics::target_saturated_drop_count`]
/// — do not introduce a parallel counter. `total_rejections` must **exclude**
/// `within_batch_target_short_circuit`. `candidates_considered` is the pass's
/// [`crate::analysis::candidate_reconciliation::CandidateLedger::considered`]
/// count. Returns `false` when `candidates_kept` is true so a productive
/// fixture is unchanged.
#[must_use]
pub fn target_saturated_should_abort_pass(
    target_saturated_drops: u32,
    total_rejections: u32,
    candidates_considered: u32,
    candidates_kept: bool,
) -> bool {
    if candidates_kept {
        return false;
    }
    if candidates_considered < TARGET_SATURATED_EARLY_EXIT_MIN_CONSIDERED {
        return false;
    }
    if target_saturated_drops < TARGET_SATURATED_EARLY_EXIT_MIN_SAMPLE {
        return false;
    }
    if total_rejections == 0 {
        return false;
    }
    let ratio = f64::from(target_saturated_drops) / f64::from(total_rejections);
    ratio >= TARGET_SATURATED_EARLY_EXIT_RATIO
}

#[cfg(test)]
mod saturation_early_exit_tests {
    use super::*;

    /// Enough evaluated candidates that the "we actually tried" floor is met,
    /// so each test below exercises the signal it names.
    const TRIED: u32 = TARGET_SATURATED_EARLY_EXIT_MIN_CONSIDERED;

    #[test]
    fn saturated_pass_trips_from_target_saturated_drop_count() {
        let diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.record_target_saturated_drops(TARGET_SATURATED_EARLY_EXIT_MIN_SAMPLE);
        let saturated = diagnostics.target_saturated_drop_count();
        assert!(
            target_saturated_should_abort_pass(saturated, saturated, TRIED, false),
            "trip threshold must be computed from target_saturated_drop_count()"
        );
    }

    #[test]
    fn within_batch_short_circuit_does_not_abort_the_pass() {
        // A batch-local skip is not terminal. Even a large within-batch count
        // with zero saturation drops must not trip.
        assert!(!target_saturated_should_abort_pass(0, 0, TRIED, false));
        assert!(!target_saturated_should_abort_pass(10, 200, TRIED, false));
    }

    #[test]
    fn productive_pass_does_not_abort() {
        assert!(!target_saturated_should_abort_pass(
            TARGET_SATURATED_EARLY_EXIT_MIN_SAMPLE,
            TARGET_SATURATED_EARLY_EXIT_MIN_SAMPLE,
            TRIED,
            true,
        ));
    }

    /// One heavily-connected saturated target drops every eligible source at
    /// once. That alone must not end a pass which has not yet evaluated
    /// anything — otherwise every remaining target is skipped on the evidence
    /// of one.
    #[test]
    fn a_single_saturated_target_does_not_abort_before_the_pass_has_tried() {
        assert!(
            !target_saturated_should_abort_pass(500, 500, 0, false),
            "500 drops from one target with nothing evaluated is not a verdict"
        );
        assert!(!target_saturated_should_abort_pass(
            500,
            500,
            TARGET_SATURATED_EARLY_EXIT_MIN_CONSIDERED - 1,
            false
        ));
        assert!(
            target_saturated_should_abort_pass(
                500,
                500,
                TARGET_SATURATED_EARLY_EXIT_MIN_CONSIDERED,
                false
            ),
            "once the pass has evaluated candidates and kept none, saturation is terminal"
        );
    }

    /// Recorded shape: 3080 `target_saturated` rejections, 52 proposals,
    /// zero candidates. Must trip well before a tens-of-minutes deadline.
    #[test]
    fn recorded_saturation_shape_trips() {
        let saturated = 3080;
        let proposals_formed = 52;
        let total_rejections = saturated.max(proposals_formed);
        assert!(target_saturated_should_abort_pass(
            saturated,
            total_rejections,
            proposals_formed,
            false
        ));
    }
}
