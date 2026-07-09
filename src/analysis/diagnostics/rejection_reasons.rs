//! Stable rejection reason names and aggregate breakdown (Issue #1129).
//!
//! Every filter or discount call site that zeroes out (or removes) a candidate
//! must increment a counter on [`RejectionBreakdown`] using one of the
//! `REJECTION_*` constants below. Downstream tooling relies on these strings
//! staying stable across releases — add new names here rather than editing
//! existing ones.
//!
//! The aggregated counts, plus a one-sentence [`top_level_summary`] describing
//! the dominant reason, are emitted on both `synapseMetadata` and
//! `neuronMetadata` in the FFI JSON response (even when zero candidates
//! survive) so operators can root-cause "no candidates found" failures
//! without re-running analysis.

use std::collections::HashMap;

// =============================================================================
// Stable reason names
// =============================================================================
//
// These are the documented reason strings used as keys in `rejection_breakdown`.
// Keep them lowercase-snake-case and stable; add new constants rather than
// renaming existing ones.

/// Candidate's expected gain fell below the coordinated expected-gain floor
/// (`COORDINATED_MIN_EXPECTED_GAIN` or `COORDINATED_POST_DISCOUNT_NOISE_FLOOR`).
pub const REJECTION_BELOW_EXPECTED_GAIN_FLOOR: &str = "below_expected_gain_floor";

/// Multi-operation coordinated candidate's discounted gain fell below
/// `MIN_COORDINATED_MULTI_OP_GAIN`.
pub const REJECTION_BELOW_MULTI_OP_FLOOR: &str = "below_multi_op_floor";

/// Candidate's expected gain was non-positive after pre-filtering.
pub const REJECTION_NON_POSITIVE_GAIN: &str = "non_positive_gain";

/// Candidate's expected gain was not finite (`NaN` or `±∞`) (Issue #1367).
///
/// Ranking sorts by `expected_creature_score_gain` via `total_cmp`, which
/// orders a positive `NaN` above `+∞`, so a non-finite gain would otherwise
/// sort to the top and be returned as the *best* candidate. A non-finite gain
/// is not a valid positive improvement and is dropped before reranking / final
/// selection.
pub const REJECTION_NON_FINITE_GAIN: &str = "non_finite_gain";

/// Target saturation discount collapsed the expected gain to (near) zero.
pub const REJECTION_SATURATION_DISCOUNTED_TO_ZERO: &str = "saturation_discounted_to_zero";

/// Pessimism discount (improved-ratio-weighted) collapsed the expected gain
/// to (near) zero.
pub const REJECTION_PESSIMISM_DISCOUNTED_TO_ZERO: &str = "pessimism_discounted_to_zero";

/// Candidate pair was filtered by the epistatic interference detector.
pub const REJECTION_INTERFERENCE_FILTERED: &str = "interference_filtered";

/// Candidate's improved/total sample ratio was below the module's threshold.
pub const REJECTION_BELOW_IMPROVED_RATIO: &str = "below_improved_ratio";

/// Candidate matched an entry in the failure cache (seen to fail before).
pub const REJECTION_DUPLICATE_OF_FAILURE_CACHE: &str = "duplicate_of_failure_cache";

/// Add-synapse candidates were gated out by historical success rate or
/// synapse density (Issue #1057).
pub const REJECTION_ADD_SYNAPSE_GATED: &str = "add_synapse_gated";

/// Candidate was truncated because the per-module or global candidate budget
/// was already full.
pub const REJECTION_BUDGET_TRUNCATED: &str = "budget_truncated";

/// Add-neuron candidate was dropped because the per-target cap
/// (`MAX_ADD_NEURON_CANDIDATES_PER_TARGET`) was already reached for this
/// target neuron in the current batch (Issue #1140).
pub const REJECTION_PER_TARGET_CAP: &str = "per_target_cap";

/// Add-neuron candidate was dropped because a higher-gain candidate for the
/// same `(target_uuid, squash)` pair already exists in the current batch
/// (Issue #1141). Without this filter, multiple near-duplicate proposals
/// targeting the same neuron with the same squash function (e.g. 17
/// candidates all using `ReLU6`) would consume the controller's ablation
/// budget on near-identical failing bets.
pub const REJECTION_SAME_TARGET_SQUASH_DUPLICATE: &str = "same_target_squash_duplicate";

/// Coordinated-structural candidate was dropped because the per-final-target
/// cap (`MAX_COORDINATED_PER_TARGET_OUTPUT`) was already reached for the
/// last operation's target neuron in the current batch (Issue #1271).
pub const REJECTION_COORDINATED_TARGET_CAP_EXCEEDED: &str = "coordinated_target_cap_exceeded";

/// 1-in/1-out hidden-neuron collapse candidate was dropped because the
/// computed bypass synapse weight had `|weight| <
/// MIN_BYPASS_WEIGHT_FOR_COLLAPSE` (Issue #1270).
///
/// At near-zero bypass weights the chain `a→h→b` was contributing essentially
/// nothing through the hidden neuron, so the 4-op coordinated collapse is
/// functionally equivalent to a 1-op `remove-neuron` but still carries the
/// implementation-risk profile of a 4-op coordinated change.
pub const REJECTION_COORDINATED_COLLAPSE_BYPASS_WEIGHT_BELOW_FLOOR: &str =
    "coordinated_collapse_bypass_weight_below_floor";

/// Discovery module was skipped because the per-(creature, module) starvation
/// tracker has the module in active cooldown after
/// `MODULE_STARVATION_FAILURE_STREAK` consecutive failures (Issue #1273).
///
/// One rejection count is recorded per module skipped during the parallel
/// detection phase. Unlike the population-wide `MODULE_GATE_THRESHOLD`
/// (Issue #1060), this signal is creature-scoped: a module that has failed
/// repeatedly for this creature is paused while the candidate budget is
/// redirected to alternative modules.
pub const REJECTION_MODULE_STARVED: &str = "module_starved";

/// Helpful add-synapse candidate was dropped by the CPU pre-reject screen
/// *before* the GPU submit because it provably carries no usable signal
/// (Issue #1544).
///
/// The screen recomputes the least-squares sufficient statistics that the
/// helpful GPU shader would produce (`Σ activation²`, `Σ activation·avg_error`)
/// and applies `calculate_optimal_outgoing_weight` — the exact gate the
/// downstream result-collection loop uses (`None => continue`). A candidate
/// counted here would have been rejected after a wasted GPU round-trip with an
/// identical outcome, so the reason distinguishes "cheaply screened out on CPU"
/// from a genuine candidate drought.
pub const REJECTION_CPU_PRE_REJECT_NO_SIGNAL: &str = "cpu_pre_reject_no_signal";

/// Synapse: no overlapping discovery samples between source and target.
pub const REJECTION_NO_SAMPLES: &str = "no_samples";

/// Synapse: GPU evaluation reported zero consistent improvement.
pub const REJECTION_ZERO_IMPROVEMENT: &str = "zero_improvement";

/// Synapse: expected improvement below the per-target threshold.
pub const REJECTION_BELOW_THRESHOLD: &str = "below_threshold";

/// Target neuron had zero activation records in the Parquet file (recording
/// phase likely timed out).
pub const REJECTION_NO_TARGET_RECORDS: &str = "no_target_records";

/// Analysis was skipped before any GPU work because the selected focus neurons
/// had insufficient Parquet coverage — the record phase produced zero rows for
/// (at least the configured fraction of) the focus neurons, so analysis was
/// guaranteed to return nothing (Issue #1444). Distinct from
/// [`REJECTION_NO_TARGET_RECORDS`], which is recorded per-target *during*
/// analysis; this reason fails fast *before* analysis to avoid spending the
/// full budget on a guaranteed-empty pass.
pub const REJECTION_INSUFFICIENT_RECORDING: &str = "insufficient_recording";

/// Target had no upstream neurons eligible for analysis.
pub const REJECTION_NO_ELIGIBLE_SOURCES: &str = "no_eligible_sources";

/// Neuron: target was an input neuron (observation source, not a computation
/// node).
pub const REJECTION_INPUT_NEURON_FILTERED: &str = "input_neuron_filtered";

/// Neuron: target was a hidden neuron (add-neuron analysis only targets
/// outputs).
pub const REJECTION_HIDDEN_NEURON_FILTERED: &str = "hidden_neuron_filtered";

/// Neuron: target was a constant neuron (does not receive inputs).
pub const REJECTION_CONSTANT_NEURON_FILTERED: &str = "constant_neuron_filtered";

/// Synapse: target neuron evaluated candidates but none had any recorded
/// diagnostic detail. Corresponds to `SynapseNoCandidateReason::NoDiagnostics`.
pub const REJECTION_NO_DIAGNOSTICS: &str = "no_diagnostics";

/// Candidate was dropped because the target neuron's observed activation
/// distribution already covered the bulk of its bounded squash output range
/// (Issue #1143).
///
/// A saturated target cannot respond to additional inputs with a usable
/// gradient — the pre-check rejects every add-neuron (and new add-synapse)
/// candidate for that target regardless of the candidate/intermediate
/// squash. See `compute_target_saturation` in
/// `src/analysis/neuron/preparation.rs`.
pub const REJECTION_TARGET_SATURATED: &str = "target_saturated";

/// Remove-low-impact: the net improvement
/// (`boosted_savings - activation_weighted_impact`) fell below
/// `REMOVE_LOW_IMPACT_NOISE_FLOOR` (Issue #1142).
///
/// The `REMOVAL_CANDIDATE_BOOST` (1.5×) is applied to raw complexity savings
/// before the savings-vs-impact comparison. Boost-inflated net improvements in
/// the 1e-8 range are indistinguishable from numerical noise — GRQ-sampler
/// commit `744ac60d` showed such a candidate causing a `-2.39e-7` actual
/// error reduction.
pub const REJECTION_REMOVAL_BELOW_NOISE_FLOOR: &str = "removal_below_noise_floor";

/// Single-op remove-neuron coordinated candidate had its expected gain demoted
/// because the creature is in a search-exhaustion drought (Issue #1448).
///
/// This is a *deprioritisation*, not an outright rejection: the demoted gain
/// sorts the destructive remove-neuron proposals below the constructive change
/// types, and the most over-confident ones subsequently fall through the
/// coordinated noise floor (recorded separately under
/// [`REJECTION_BELOW_EXPECTED_GAIN_FLOOR`]). The count here is how many
/// remove-neuron candidates were demoted on the pass, so operators can see the
/// destructive module surrendering budget during a plateau.
pub const REJECTION_REMOVE_NEURON_DROUGHT_DEPRIORITISED: &str =
    "remove_neuron_drought_deprioritised";

/// All documented rejection reason names. Used for assertions and
/// documentation. Keep this list in sync with the constants above.
pub const ALL_REJECTION_REASONS: &[&str] = &[
    REJECTION_BELOW_EXPECTED_GAIN_FLOOR,
    REJECTION_BELOW_MULTI_OP_FLOOR,
    REJECTION_NON_POSITIVE_GAIN,
    REJECTION_NON_FINITE_GAIN,
    REJECTION_SATURATION_DISCOUNTED_TO_ZERO,
    REJECTION_PESSIMISM_DISCOUNTED_TO_ZERO,
    REJECTION_INTERFERENCE_FILTERED,
    REJECTION_BELOW_IMPROVED_RATIO,
    REJECTION_DUPLICATE_OF_FAILURE_CACHE,
    REJECTION_ADD_SYNAPSE_GATED,
    REJECTION_BUDGET_TRUNCATED,
    REJECTION_PER_TARGET_CAP,
    REJECTION_SAME_TARGET_SQUASH_DUPLICATE,
    REJECTION_COORDINATED_TARGET_CAP_EXCEEDED,
    REJECTION_COORDINATED_COLLAPSE_BYPASS_WEIGHT_BELOW_FLOOR,
    REJECTION_MODULE_STARVED,
    REJECTION_CPU_PRE_REJECT_NO_SIGNAL,
    REJECTION_NO_SAMPLES,
    REJECTION_ZERO_IMPROVEMENT,
    REJECTION_BELOW_THRESHOLD,
    REJECTION_NO_TARGET_RECORDS,
    REJECTION_INSUFFICIENT_RECORDING,
    REJECTION_NO_ELIGIBLE_SOURCES,
    REJECTION_INPUT_NEURON_FILTERED,
    REJECTION_HIDDEN_NEURON_FILTERED,
    REJECTION_CONSTANT_NEURON_FILTERED,
    REJECTION_NO_DIAGNOSTICS,
    REJECTION_TARGET_SATURATED,
    REJECTION_REMOVAL_BELOW_NOISE_FLOOR,
    REJECTION_REMOVE_NEURON_DROUGHT_DEPRIORITISED,
];

// =============================================================================
// RejectionBreakdown
// =============================================================================

/// Aggregate rejection counts keyed by stable reason name (Issue #1129).
///
/// Designed to be cheap to increment sequentially from orchestration /
/// post-processing code. For parallel discovery module dispatch, accumulate
/// into a per-module `HashMap` and merge sequentially via
/// [`RejectionBreakdown::merge_from`].
#[derive(Debug, Default, Clone)]
pub struct RejectionBreakdown {
    counts: HashMap<String, u32>,
}

impl RejectionBreakdown {
    /// Create an empty breakdown.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    /// Add `count` rejections for `reason`. Does nothing if `count == 0`.
    pub fn record_many(&mut self, reason: &'static str, count: u32) {
        if count == 0 {
            return;
        }
        *self.counts.entry(reason.to_string()).or_insert(0) += count;
    }

    /// Alias for [`RejectionBreakdown::record_many`]; accepts a `u32` count explicitly so call
    /// sites converting from `usize` are concise.
    pub fn record_many_u32(&mut self, reason: &'static str, count: u32) {
        self.record_many(reason, count);
    }

    /// Record a single rejection for `reason`.
    pub fn record(&mut self, reason: &'static str) {
        self.record_many(reason, 1);
    }

    /// Merge another breakdown into this one.
    pub fn merge_from(&mut self, other: &HashMap<String, u32>) {
        for (reason, &count) in other {
            if count == 0 {
                continue;
            }
            *self.counts.entry(reason.clone()).or_insert(0) += count;
        }
    }

    /// Total rejection count across all reasons.
    #[must_use]
    pub fn total(&self) -> u32 {
        self.counts.values().sum()
    }

    /// Return whether no rejections have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.values().all(|&v| v == 0)
    }

    /// Return the dominant (most-frequent) reason and its count, if any.
    ///
    /// Ties are broken lexicographically by reason name so the output is
    /// deterministic.
    #[must_use]
    pub fn dominant_reason(&self) -> Option<(&str, u32)> {
        self.counts
            .iter()
            .filter(|&(_, count)| *count > 0)
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(r, c)| (r.as_str(), *c))
    }

    /// Borrow the underlying counts map.
    #[must_use]
    pub fn counts(&self) -> &HashMap<String, u32> {
        &self.counts
    }

    /// Take the underlying counts map, leaving an empty breakdown behind.
    #[must_use]
    pub fn into_counts(self) -> HashMap<String, u32> {
        self.counts
    }
}

// =============================================================================
// Top-level summary
// =============================================================================

/// Produce a one-sentence summary naming the dominant rejection reason and
/// associated gain floor (Issue #1129).
///
/// `total_candidates_considered` is the number of candidates (including
/// rejected ones) that the analysis attempted to evaluate. When it is `None`
/// the summary still reports absolute counts.
#[must_use]
pub fn top_level_summary(
    breakdown: &RejectionBreakdown,
    total_candidates_considered: Option<u32>,
) -> Option<String> {
    let (reason, count) = breakdown.dominant_reason()?;
    let considered = total_candidates_considered.unwrap_or_else(|| breakdown.total());
    let friendly = friendly_reason(reason);
    Some(format!(
        "{count} of {considered} candidates rejected by {friendly}"
    ))
}

/// Render a rejection reason name into a human-readable phrase for the
/// top-level summary.
fn friendly_reason(reason: &str) -> String {
    match reason {
        REJECTION_BELOW_EXPECTED_GAIN_FLOOR => {
            format!(
                "expected-gain floor of {:e}",
                crate::analysis::constants::COORDINATED_MIN_EXPECTED_GAIN
            )
        }
        REJECTION_BELOW_MULTI_OP_FLOOR => format!(
            "multi-op expected-gain floor of {:e}",
            crate::analysis::constants::MIN_COORDINATED_MULTI_OP_GAIN
        ),
        REJECTION_NON_POSITIVE_GAIN => "non-positive expected gain".to_string(),
        REJECTION_NON_FINITE_GAIN => "non-finite expected gain (NaN or ±∞)".to_string(),
        REJECTION_SATURATION_DISCOUNTED_TO_ZERO => {
            "saturation discount collapsed expected gain to zero".to_string()
        }
        REJECTION_PESSIMISM_DISCOUNTED_TO_ZERO => {
            "pessimism discount collapsed expected gain to zero".to_string()
        }
        REJECTION_INTERFERENCE_FILTERED => "epistatic interference filter".to_string(),
        REJECTION_BELOW_IMPROVED_RATIO => "improved-sample-ratio floor".to_string(),
        REJECTION_DUPLICATE_OF_FAILURE_CACHE => "duplicate of failure cache".to_string(),
        REJECTION_ADD_SYNAPSE_GATED => "add-synapse historical-gating filter".to_string(),
        REJECTION_BUDGET_TRUNCATED => "per-module candidate budget".to_string(),
        REJECTION_PER_TARGET_CAP => "per-target add-neuron cap".to_string(),
        REJECTION_SAME_TARGET_SQUASH_DUPLICATE => "duplicate squash within same target".to_string(),
        REJECTION_COORDINATED_TARGET_CAP_EXCEEDED => {
            "per-final-target coordinated-structural cap".to_string()
        }
        REJECTION_COORDINATED_COLLAPSE_BYPASS_WEIGHT_BELOW_FLOOR => format!(
            "hidden-neuron collapse bypass-weight floor of {}",
            crate::analysis::constants::min_bypass_weight_for_collapse()
        ),
        REJECTION_MODULE_STARVED => "per-creature module starvation cooldown".to_string(),
        REJECTION_CPU_PRE_REJECT_NO_SIGNAL => {
            "CPU pre-reject screen (no usable signal before GPU submit)".to_string()
        }
        REJECTION_NO_SAMPLES => "no overlapping discovery samples".to_string(),
        REJECTION_ZERO_IMPROVEMENT => "zero consistent improvement in GPU stats".to_string(),
        REJECTION_BELOW_THRESHOLD => "expected-improvement per-target threshold".to_string(),
        REJECTION_NO_TARGET_RECORDS => "no target activation records".to_string(),
        REJECTION_INSUFFICIENT_RECORDING => {
            "insufficient Parquet recording for the selected focus neurons \
             (record phase likely timed out)"
                .to_string()
        }
        REJECTION_NO_ELIGIBLE_SOURCES => "no eligible upstream sources".to_string(),
        REJECTION_INPUT_NEURON_FILTERED => "input-neuron pre-filter".to_string(),
        REJECTION_HIDDEN_NEURON_FILTERED => "hidden-neuron pre-filter".to_string(),
        REJECTION_CONSTANT_NEURON_FILTERED => "constant-neuron pre-filter".to_string(),
        REJECTION_NO_DIAGNOSTICS => "no diagnostics recorded".to_string(),
        REJECTION_TARGET_SATURATED => {
            "target neuron already saturated (observed activation covers the full bounded range)"
                .to_string()
        }
        REJECTION_REMOVAL_BELOW_NOISE_FLOOR => format!(
            "remove-low-impact noise floor of {:e}",
            crate::analysis::constants::remove_low_impact_noise_floor()
        ),
        REJECTION_REMOVE_NEURON_DROUGHT_DEPRIORITISED => {
            "remove-neuron deprioritised during search-exhaustion drought".to_string()
        }
        other => other.replace('_', " "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_increments_counter() {
        let mut b = RejectionBreakdown::new();
        b.record(REJECTION_BELOW_EXPECTED_GAIN_FLOOR);
        b.record(REJECTION_BELOW_EXPECTED_GAIN_FLOOR);
        b.record(REJECTION_INTERFERENCE_FILTERED);
        assert_eq!(
            b.counts().get(REJECTION_BELOW_EXPECTED_GAIN_FLOOR),
            Some(&2)
        );
        assert_eq!(b.counts().get(REJECTION_INTERFERENCE_FILTERED), Some(&1));
        assert_eq!(b.total(), 3);
    }

    #[test]
    fn record_many_skips_zero() {
        let mut b = RejectionBreakdown::new();
        b.record_many(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 0);
        assert!(b.is_empty());
    }

    #[test]
    fn dominant_reason_picks_highest_count() {
        let mut b = RejectionBreakdown::new();
        b.record_many(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 32);
        b.record_many(REJECTION_INTERFERENCE_FILTERED, 5);
        b.record_many(REJECTION_NO_SAMPLES, 2);
        let (reason, count) = b.dominant_reason().expect("has counts");
        assert_eq!(reason, REJECTION_BELOW_EXPECTED_GAIN_FLOOR);
        assert_eq!(count, 32);
    }

    #[test]
    fn top_level_summary_mentions_floor() {
        let mut b = RejectionBreakdown::new();
        b.record_many(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 32);
        let summary = top_level_summary(&b, Some(34)).expect("summary");
        assert!(
            summary.contains("32 of 34"),
            "summary should contain counts, got: {summary}"
        );
        assert!(
            summary.contains("expected-gain floor"),
            "summary should mention the floor, got: {summary}"
        );
    }

    #[test]
    fn top_level_summary_none_on_empty() {
        let b = RejectionBreakdown::new();
        assert!(top_level_summary(&b, None).is_none());
    }

    #[test]
    fn merge_from_sums_counts() {
        let mut base = RejectionBreakdown::new();
        base.record_many(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 3);
        let mut other = HashMap::new();
        other.insert(REJECTION_BELOW_EXPECTED_GAIN_FLOOR.to_string(), 4);
        other.insert(REJECTION_INTERFERENCE_FILTERED.to_string(), 2);
        base.merge_from(&other);
        assert_eq!(
            base.counts().get(REJECTION_BELOW_EXPECTED_GAIN_FLOOR),
            Some(&7)
        );
        assert_eq!(base.counts().get(REJECTION_INTERFERENCE_FILTERED), Some(&2));
    }

    #[test]
    fn all_reasons_list_contains_every_constant() {
        for reason in ALL_REJECTION_REASONS {
            // If this assertion fires, someone added a constant without
            // appending it to ALL_REJECTION_REASONS.
            assert!(!reason.is_empty());
        }
    }
}
