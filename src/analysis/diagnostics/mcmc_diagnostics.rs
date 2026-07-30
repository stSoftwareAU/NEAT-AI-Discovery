//! MCMC diagnostics: acceptance rate tracking and chain convergence metrics.
//!
//! Tracks Markov Chain Monte Carlo metrics for the candidate selection pipeline:
//! - **Acceptance rate**: Fraction of proposed candidates accepted, by candidate type
//! - **Proposal quality**: Distribution of improvement values (min, max, mean, median)
//! - **Diversity metric**: Unique source/target neurons in accepted vs evaluated candidates
//!
//! All tracking is zero-overhead when verbose mode is disabled — no allocations
//! occur for improvement value collection.
//!
//! Issue #1021

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for ratio computation

use crate::analysis::scoring::calibration_correction::FailureCacheEntry;
use crate::analysis::utils::verbose_enabled;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};

/// Maximum number of `CalibrationMissEntry` records retained in verbose mode
/// (Issue #1165). The counter is unbounded; only the per-entry vector is
/// capped to bound memory.
pub const CALIBRATION_MISS_VEC_CAP: usize = 1000;

/// Default threshold for the `actual / expected` ratio above which (or whose
/// reciprocal exceeds) a calibration mismatch is logged (Issue #1165).
///
/// A miss is reported when `|ratio| > threshold` or `|ratio| < 1 / threshold`.
/// Overridable via `NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD`.
pub const DEFAULT_CALIBRATION_MISS_THRESHOLD: f32 = 10.0;

/// A single prediction-vs-actual calibration mismatch (Issue #1165).
///
/// Captured when the recorded `actual_error_reduction` differs from the
/// `expected_error_reduction` by more than the configured threshold (default
/// 10×). The structured tuple lets future tuning of calibration constants
/// in `analysis::constants::candidate_scoring` work from data instead of a
/// handful of failure JSON files.
#[derive(Debug, Clone)]
pub struct CalibrationMissEntry {
    /// Candidate classification — same key used for `CalibrationCorrection`.
    pub change_type: String,
    /// Target neuron's activation function, when supplied by the upstream
    /// emitter.
    pub target_squash: Option<String>,
    /// Variant key identifying the variant generator (Issue #1163).
    pub variant_key: Option<String>,
    /// What was predicted at proposal time.
    pub expected: f32,
    /// What actually happened after applying the candidate.
    pub actual: f32,
    /// `actual / expected` (finite, non-zero divisor).
    pub ratio: f32,
}

// =============================================================================
// Per-candidate-type acceptance counters (lock-free atomics)
// =============================================================================

/// Lock-free acceptance counters for a single candidate type.
///
/// Uses atomics so parallel rayon threads can increment without contention.
pub(crate) struct AcceptanceCounters {
    pub(crate) proposed: AtomicU32,
    pub(crate) accepted: AtomicU32,
}

impl AcceptanceCounters {
    pub(crate) fn new() -> Self {
        Self {
            proposed: AtomicU32::new(0),
            accepted: AtomicU32::new(0),
        }
    }

    pub(crate) fn record_proposed(&self) {
        self.proposed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_accepted(&self) {
        self.accepted.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> AcceptanceSnapshot {
        AcceptanceSnapshot {
            proposed: self.proposed.load(Ordering::Relaxed),
            accepted: self.accepted.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of acceptance counters.
#[derive(Debug, Clone, Copy)]
pub struct AcceptanceSnapshot {
    pub proposed: u32,
    pub accepted: u32,
}

impl AcceptanceSnapshot {
    /// Acceptance rate as a fraction in [0, 1]. Returns 0.0 if no proposals.
    pub fn rate(&self) -> f32 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f32 / self.proposed as f32
        }
    }
}

// =============================================================================
// MCMC Diagnostics Tracker
// =============================================================================

/// Collects MCMC diagnostics during an analysis run (Issue #1021).
///
/// Lock-free atomic counters for acceptance rates; guarded `Vec` collection
/// for improvement values (only allocated when verbose mode is enabled).
pub(crate) struct McmcDiagnosticsTracker {
    /// Acceptance counters broken down by candidate type.
    pub(crate) synapse_counters: AcceptanceCounters,
    pub(crate) neuron_counters: AcceptanceCounters,
    pub(crate) coordinated_counters: AcceptanceCounters,

    /// Whether to collect per-candidate improvement values and diversity data.
    /// When false, the `Vec`s remain empty — zero allocation overhead.
    collect_details: bool,

    /// Improvement values for accepted candidates (only populated when verbose).
    /// Protected by a mutex for thread-safe appending.
    improvement_values: std::sync::Mutex<Vec<f32>>,

    /// Unique source neuron UUIDs among evaluated candidates.
    evaluated_sources: std::sync::Mutex<HashSet<String>>,
    /// Unique target neuron UUIDs among evaluated candidates.
    evaluated_targets: std::sync::Mutex<HashSet<String>>,
    /// Unique source neuron UUIDs among accepted candidates.
    accepted_sources: std::sync::Mutex<HashSet<String>>,
    /// Unique target neuron UUIDs among accepted candidates.
    accepted_targets: std::sync::Mutex<HashSet<String>>,

    /// Total count of calibration mismatches detected this run (Issue #1165).
    /// Always tracked, regardless of verbose mode, so the rate is visible at
    /// a glance via [`McmcDiagnosticsSummary::calibration_miss_count`].
    calibration_miss_count: AtomicU32,
    /// Per-entry calibration mismatch records (Issue #1165). Only populated in
    /// verbose mode and capped at [`CALIBRATION_MISS_VEC_CAP`] to bound memory.
    calibration_misses: std::sync::Mutex<Vec<CalibrationMissEntry>>,
}

impl McmcDiagnosticsTracker {
    /// Create a new tracker. Detail collection is enabled when verbose mode is on.
    pub(crate) fn new() -> Self {
        let collect = verbose_enabled();
        Self {
            synapse_counters: AcceptanceCounters::new(),
            neuron_counters: AcceptanceCounters::new(),
            coordinated_counters: AcceptanceCounters::new(),
            collect_details: collect,
            improvement_values: std::sync::Mutex::new(Vec::new()),
            evaluated_sources: std::sync::Mutex::new(HashSet::new()),
            evaluated_targets: std::sync::Mutex::new(HashSet::new()),
            accepted_sources: std::sync::Mutex::new(HashSet::new()),
            accepted_targets: std::sync::Mutex::new(HashSet::new()),
            calibration_miss_count: AtomicU32::new(0),
            calibration_misses: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Create a tracker that always collects details (for testing).
    #[cfg(test)]
    pub(crate) fn new_verbose() -> Self {
        Self {
            synapse_counters: AcceptanceCounters::new(),
            neuron_counters: AcceptanceCounters::new(),
            coordinated_counters: AcceptanceCounters::new(),
            collect_details: true,
            improvement_values: std::sync::Mutex::new(Vec::new()),
            evaluated_sources: std::sync::Mutex::new(HashSet::new()),
            evaluated_targets: std::sync::Mutex::new(HashSet::new()),
            accepted_sources: std::sync::Mutex::new(HashSet::new()),
            accepted_targets: std::sync::Mutex::new(HashSet::new()),
            calibration_miss_count: AtomicU32::new(0),
            calibration_misses: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Record an evaluated candidate (proposed but not yet accepted/rejected).
    pub(crate) fn record_evaluated(
        &self,
        source_uuid: &str,
        target_uuid: &str,
        candidate_type: CandidateType,
    ) {
        self.counters_for(candidate_type).record_proposed();

        if self.collect_details {
            if let Ok(mut set) = self.evaluated_sources.lock() {
                set.insert(source_uuid.to_string());
            }
            if let Ok(mut set) = self.evaluated_targets.lock() {
                set.insert(target_uuid.to_string());
            }
        }
    }

    /// Record an accepted candidate with its improvement value.
    pub(crate) fn record_accepted(
        &self,
        source_uuid: &str,
        target_uuid: &str,
        improvement: f32,
        candidate_type: CandidateType,
    ) {
        self.counters_for(candidate_type).record_accepted();

        if self.collect_details {
            if let Ok(mut vals) = self.improvement_values.lock() {
                vals.push(improvement);
            }
            if let Ok(mut set) = self.accepted_sources.lock() {
                set.insert(source_uuid.to_string());
            }
            if let Ok(mut set) = self.accepted_targets.lock() {
                set.insert(target_uuid.to_string());
            }
        }
    }

    /// Inspect a failure cache for prediction-vs-actual calibration mismatches
    /// and record any that are out of tolerance (Issue #1165).
    ///
    /// A miss is reported when the per-entry ratio
    /// `actual / expected` satisfies `|ratio| > threshold` or
    /// `|ratio| < 1 / threshold`. For each miss, this method:
    ///
    /// 1. Emits a `tracing::warn!` with the structured tuple
    ///    `(change_type, target_squash, variant_key, expected, actual, ratio)`.
    /// 2. Increments the calibration-miss counter.
    /// 3. In verbose mode, appends a [`CalibrationMissEntry`] to the bounded
    ///    in-memory vector (capped at [`CALIBRATION_MISS_VEC_CAP`]).
    ///
    /// Entries with `expected_error_reduction == 0` or non-finite ratios are
    /// skipped (no defined ratio). Threshold values `<= 1.0` are clamped to
    /// `1.0` so the upper and lower bounds remain meaningful.
    pub(crate) fn record_calibration_misses_from_cache(
        &self,
        cache: &[FailureCacheEntry],
        threshold: f32,
    ) {
        let threshold = if threshold.is_finite() && threshold > 1.0 {
            threshold
        } else {
            1.0
        };
        let lower = 1.0 / threshold;
        for entry in cache {
            if entry.expected_error_reduction == 0.0 {
                continue;
            }
            let ratio = entry.actual_error_reduction / entry.expected_error_reduction;
            if !ratio.is_finite() {
                continue;
            }
            let abs_ratio = ratio.abs();
            if abs_ratio <= threshold && abs_ratio >= lower {
                continue;
            }

            tracing::warn!(
                target: "neat_ai_discovery::observability",
                change_type = %entry.change_type,
                target_squash = entry.target_squash.as_deref().unwrap_or("<none>"),
                variant_key = entry.variant_key.as_deref().unwrap_or("<none>"),
                expected = entry.expected_error_reduction,
                actual = entry.actual_error_reduction,
                ratio,
                threshold,
                "prediction-vs-actual calibration mismatch (Issue #1165)"
            );

            self.calibration_miss_count.fetch_add(1, Ordering::Relaxed);

            if self.collect_details
                && let Ok(mut vec) = self.calibration_misses.lock()
                && vec.len() < CALIBRATION_MISS_VEC_CAP
            {
                vec.push(CalibrationMissEntry {
                    change_type: entry.change_type.clone(),
                    target_squash: entry.target_squash.clone(),
                    variant_key: entry.variant_key.clone(),
                    expected: entry.expected_error_reduction,
                    actual: entry.actual_error_reduction,
                    ratio,
                });
            }
        }
    }

    fn counters_for(&self, candidate_type: CandidateType) -> &AcceptanceCounters {
        match candidate_type {
            CandidateType::Synapse => &self.synapse_counters,
            CandidateType::Neuron => &self.neuron_counters,
            CandidateType::Coordinated => &self.coordinated_counters,
        }
    }

    /// Build the final diagnostics summary.
    pub(crate) fn build_summary(&self) -> McmcDiagnosticsSummary {
        let synapse = self.synapse_counters.snapshot();
        let neuron = self.neuron_counters.snapshot();
        let coordinated = self.coordinated_counters.snapshot();

        let total_proposed = synapse.proposed + neuron.proposed + coordinated.proposed;
        let total_accepted = synapse.accepted + neuron.accepted + coordinated.accepted;

        let proposal_quality = if self.collect_details {
            self.compute_proposal_quality()
        } else {
            None
        };

        let diversity = if self.collect_details {
            self.compute_diversity()
        } else {
            None
        };

        let calibration_miss_count = self.calibration_miss_count.load(Ordering::Relaxed);
        let calibration_misses = if self.collect_details {
            let vec = self
                .calibration_misses
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if vec.is_empty() {
                None
            } else {
                Some(vec.clone())
            }
        } else {
            None
        };

        McmcDiagnosticsSummary {
            synapse_acceptance: synapse,
            neuron_acceptance: neuron,
            coordinated_acceptance: coordinated,
            total_proposed,
            total_accepted,
            proposal_quality,
            diversity,
            calibration_miss_count,
            calibration_misses,
        }
    }

    fn compute_proposal_quality(&self) -> Option<ProposalQuality> {
        let vals = self
            .improvement_values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if vals.is_empty() {
            return None;
        }

        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let sum: f64 = sorted.iter().map(|v| *v as f64).sum();
        let mean = (sum / sorted.len() as f64) as f32;
        let median = if sorted.len().is_multiple_of(2) {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        Some(ProposalQuality {
            count: sorted.len(),
            min,
            max,
            mean,
            median,
        })
    }

    fn compute_diversity(&self) -> Option<DiversityMetric> {
        let eval_sources = self
            .evaluated_sources
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let eval_targets = self
            .evaluated_targets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let acc_sources = self
            .accepted_sources
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let acc_targets = self
            .accepted_targets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        if eval_sources.is_empty() && eval_targets.is_empty() {
            return None;
        }

        Some(DiversityMetric {
            evaluated_unique_sources: eval_sources.len(),
            evaluated_unique_targets: eval_targets.len(),
            accepted_unique_sources: acc_sources.len(),
            accepted_unique_targets: acc_targets.len(),
        })
    }
}

// =============================================================================
// Candidate type classification
// =============================================================================

/// Classification of candidate types for per-type acceptance tracking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Neuron variant is available for neuron analysis integration
pub(crate) enum CandidateType {
    Synapse,
    Neuron,
    Coordinated,
}

// =============================================================================
// Summary types (output)
// =============================================================================

/// Complete MCMC diagnostics summary for an analysis run (Issue #1021).
#[derive(Debug, Clone)]
pub struct McmcDiagnosticsSummary {
    /// Per-type acceptance rates.
    pub synapse_acceptance: AcceptanceSnapshot,
    pub neuron_acceptance: AcceptanceSnapshot,
    pub coordinated_acceptance: AcceptanceSnapshot,
    /// Aggregate counts across all types.
    pub total_proposed: u32,
    pub total_accepted: u32,
    /// Distribution of improvement values among accepted candidates (verbose only).
    pub proposal_quality: Option<ProposalQuality>,
    /// Source/target diversity among evaluated vs accepted candidates (verbose only).
    pub diversity: Option<DiversityMetric>,
    /// Total number of prediction-vs-actual calibration mismatches detected
    /// during this run (Issue #1165). Always populated; surfaced in JSON so
    /// operators can see the rate at a glance.
    pub calibration_miss_count: u32,
    /// Per-entry calibration mismatches (Issue #1165). Only populated in
    /// verbose mode (capped at [`CALIBRATION_MISS_VEC_CAP`]).
    pub calibration_misses: Option<Vec<CalibrationMissEntry>>,
}

impl McmcDiagnosticsSummary {
    /// Overall acceptance rate across all candidate types.
    pub fn overall_acceptance_rate(&self) -> f32 {
        if self.total_proposed == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_proposed as f32
        }
    }
}

/// Distribution statistics for improvement values of accepted candidates.
#[derive(Debug, Clone)]
pub struct ProposalQuality {
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
}

/// Diversity of source and target neurons in evaluated vs accepted candidates.
#[derive(Debug, Clone)]
pub struct DiversityMetric {
    pub evaluated_unique_sources: usize,
    pub evaluated_unique_targets: usize,
    pub accepted_unique_sources: usize,
    pub accepted_unique_targets: usize,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acceptance_counters_empty() {
        let counters = AcceptanceCounters::new();
        let snap = counters.snapshot();
        assert_eq!(snap.proposed, 0);
        assert_eq!(snap.accepted, 0);
        assert_eq!(snap.rate(), 0.0);
    }

    #[test]
    fn test_acceptance_counters_tracking() {
        let counters = AcceptanceCounters::new();
        counters.record_proposed();
        counters.record_proposed();
        counters.record_proposed();
        counters.record_accepted();

        let snap = counters.snapshot();
        assert_eq!(snap.proposed, 3);
        assert_eq!(snap.accepted, 1);
        let rate = snap.rate();
        assert!(
            (rate - 1.0 / 3.0).abs() < 1e-6,
            "Expected ~0.333, got {rate}"
        );
    }

    #[test]
    fn test_tracker_per_type_counting() {
        let tracker = McmcDiagnosticsTracker::new_verbose();

        // Propose and accept synapse candidates
        tracker.record_evaluated("src-1", "tgt-1", CandidateType::Synapse);
        tracker.record_evaluated("src-2", "tgt-1", CandidateType::Synapse);
        tracker.record_accepted("src-1", "tgt-1", 0.05, CandidateType::Synapse);

        // Propose neuron candidates
        tracker.record_evaluated("src-3", "tgt-2", CandidateType::Neuron);
        tracker.record_accepted("src-3", "tgt-2", 0.10, CandidateType::Neuron);

        // Propose coordinated candidates
        tracker.record_evaluated("src-4", "tgt-3", CandidateType::Coordinated);

        let summary = tracker.build_summary();

        assert_eq!(summary.synapse_acceptance.proposed, 2);
        assert_eq!(summary.synapse_acceptance.accepted, 1);
        assert_eq!(summary.neuron_acceptance.proposed, 1);
        assert_eq!(summary.neuron_acceptance.accepted, 1);
        assert_eq!(summary.coordinated_acceptance.proposed, 1);
        assert_eq!(summary.coordinated_acceptance.accepted, 0);
        assert_eq!(summary.total_proposed, 4);
        assert_eq!(summary.total_accepted, 2);
        assert!((summary.overall_acceptance_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_proposal_quality_statistics() {
        let tracker = McmcDiagnosticsTracker::new_verbose();

        tracker.record_accepted("s1", "t1", 0.10, CandidateType::Synapse);
        tracker.record_accepted("s2", "t2", 0.20, CandidateType::Synapse);
        tracker.record_accepted("s3", "t3", 0.30, CandidateType::Synapse);
        tracker.record_accepted("s4", "t4", 0.40, CandidateType::Synapse);

        let summary = tracker.build_summary();
        let quality = summary
            .proposal_quality
            .expect("Should have proposal quality");

        assert_eq!(quality.count, 4);
        assert!((quality.min - 0.10).abs() < 1e-6);
        assert!((quality.max - 0.40).abs() < 1e-6);
        assert!((quality.mean - 0.25).abs() < 1e-6);
        // Median of [0.10, 0.20, 0.30, 0.40] = (0.20 + 0.30) / 2 = 0.25
        assert!((quality.median - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_proposal_quality_odd_count() {
        let tracker = McmcDiagnosticsTracker::new_verbose();

        tracker.record_accepted("s1", "t1", 0.10, CandidateType::Synapse);
        tracker.record_accepted("s2", "t2", 0.30, CandidateType::Synapse);
        tracker.record_accepted("s3", "t3", 0.50, CandidateType::Synapse);

        let summary = tracker.build_summary();
        let quality = summary
            .proposal_quality
            .expect("Should have proposal quality");

        assert_eq!(quality.count, 3);
        // Median of [0.10, 0.30, 0.50] = 0.30
        assert!((quality.median - 0.30).abs() < 1e-6);
    }

    #[test]
    fn test_diversity_metric() {
        let tracker = McmcDiagnosticsTracker::new_verbose();

        // Evaluate 4 candidates with 3 unique sources, 2 unique targets
        tracker.record_evaluated("src-1", "tgt-1", CandidateType::Synapse);
        tracker.record_evaluated("src-2", "tgt-1", CandidateType::Synapse);
        tracker.record_evaluated("src-3", "tgt-2", CandidateType::Synapse);
        tracker.record_evaluated("src-1", "tgt-2", CandidateType::Synapse);

        // Accept 2 candidates with 2 unique sources, 1 unique target
        tracker.record_accepted("src-1", "tgt-1", 0.05, CandidateType::Synapse);
        tracker.record_accepted("src-2", "tgt-1", 0.03, CandidateType::Synapse);

        let summary = tracker.build_summary();
        let diversity = summary.diversity.expect("Should have diversity metric");

        assert_eq!(diversity.evaluated_unique_sources, 3);
        assert_eq!(diversity.evaluated_unique_targets, 2);
        assert_eq!(diversity.accepted_unique_sources, 2);
        assert_eq!(diversity.accepted_unique_targets, 1);
    }

    #[test]
    fn test_no_proposals_returns_zero_rate() {
        let tracker = McmcDiagnosticsTracker::new_verbose();
        let summary = tracker.build_summary();

        assert_eq!(summary.total_proposed, 0);
        assert_eq!(summary.total_accepted, 0);
        assert_eq!(summary.overall_acceptance_rate(), 0.0);
        assert!(summary.proposal_quality.is_none());
    }

    // =========================================================================
    // Issue #1165 — calibration mismatch logging
    // =========================================================================

    fn miss_entry(
        change_type: &str,
        expected: f32,
        actual: f32,
        target_squash: Option<&str>,
        variant_key: Option<&str>,
    ) -> FailureCacheEntry {
        FailureCacheEntry {
            change_type: change_type.to_string(),
            expected_error_reduction: expected,
            actual_error_reduction: actual,
            target_squash: target_squash.map(str::to_string),
            variant_key: variant_key.map(str::to_string),
            target_uuid: None,
            improved_count: None,
            total_count: None,
            age_epochs: None,
        }
    }

    #[test]
    fn calibration_miss_in_tolerance_records_nothing() {
        let tracker = McmcDiagnosticsTracker::new_verbose();
        // Ratio = 0.5, well within +/-10×.
        let cache = vec![miss_entry("add-neurons", 1.0, 0.5, Some("SELU"), None)];
        tracker.record_calibration_misses_from_cache(&cache, 10.0);
        let summary = tracker.build_summary();
        assert_eq!(summary.calibration_miss_count, 0);
        assert!(summary.calibration_misses.is_none());
    }

    #[test]
    fn calibration_miss_above_threshold_is_recorded() {
        let tracker = McmcDiagnosticsTracker::new_verbose();
        // Ratio = 100, way above the 10× threshold.
        let cache = vec![miss_entry("add-synapses", 0.001, 0.1, None, None)];
        tracker.record_calibration_misses_from_cache(&cache, 10.0);
        let summary = tracker.build_summary();
        assert_eq!(summary.calibration_miss_count, 1);
        let misses = summary
            .calibration_misses
            .expect("verbose mode should populate the vec");
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].change_type, "add-synapses");
        assert!((misses[0].ratio - 100.0).abs() < 1e-3);
    }

    #[test]
    fn calibration_miss_below_reciprocal_threshold_is_recorded() {
        let tracker = McmcDiagnosticsTracker::new_verbose();
        // Ratio = 1/3000, far below 1/10 = 0.1 — the canonical "Gentle Nudge"
        // case from Issue #1160.
        let cache = vec![miss_entry(
            "add-neurons",
            0.003,
            0.000_001,
            Some("SELU"),
            Some("gentle-nudge"),
        )];
        tracker.record_calibration_misses_from_cache(&cache, 10.0);
        let summary = tracker.build_summary();
        assert_eq!(summary.calibration_miss_count, 1);
        let misses = summary
            .calibration_misses
            .expect("verbose mode should populate the vec");
        assert_eq!(misses[0].target_squash.as_deref(), Some("SELU"));
        assert_eq!(misses[0].variant_key.as_deref(), Some("gentle-nudge"));
    }

    #[test]
    fn calibration_miss_skips_zero_or_non_finite_entries() {
        let tracker = McmcDiagnosticsTracker::new_verbose();
        let cache = vec![
            // expected = 0 → undefined ratio, skip.
            miss_entry("add-neurons", 0.0, 0.5, None, None),
            // expected = MIN_POSITIVE, actual = MAX → non-finite, skip.
            miss_entry("add-neurons", f32::MIN_POSITIVE, f32::MAX, None, None),
        ];
        tracker.record_calibration_misses_from_cache(&cache, 10.0);
        let summary = tracker.build_summary();
        assert_eq!(summary.calibration_miss_count, 0);
    }

    #[test]
    fn calibration_miss_non_verbose_mode_skips_vec_but_counts() {
        // Default tracker — collect_details follows verbose_enabled() which is
        // false in the test environment. Regardless, the count must be tracked.
        let tracker = McmcDiagnosticsTracker {
            synapse_counters: AcceptanceCounters::new(),
            neuron_counters: AcceptanceCounters::new(),
            coordinated_counters: AcceptanceCounters::new(),
            collect_details: false,
            improvement_values: std::sync::Mutex::new(Vec::new()),
            evaluated_sources: std::sync::Mutex::new(HashSet::new()),
            evaluated_targets: std::sync::Mutex::new(HashSet::new()),
            accepted_sources: std::sync::Mutex::new(HashSet::new()),
            accepted_targets: std::sync::Mutex::new(HashSet::new()),
            calibration_miss_count: AtomicU32::new(0),
            calibration_misses: std::sync::Mutex::new(Vec::new()),
        };
        let cache = vec![
            miss_entry("add-neurons", 0.001, 0.1, None, None), // ratio 100
            miss_entry("add-neurons", 1.0, 0.000_01, None, None), // ratio 1e-5
        ];
        tracker.record_calibration_misses_from_cache(&cache, 10.0);
        let summary = tracker.build_summary();
        assert_eq!(summary.calibration_miss_count, 2);
        assert!(
            summary.calibration_misses.is_none(),
            "non-verbose mode must not populate the per-entry vec"
        );
    }

    #[test]
    fn calibration_miss_vec_capped_at_max_size() {
        let tracker = McmcDiagnosticsTracker::new_verbose();
        // Generate CAP + 25 misses; vector must cap, count must reflect all.
        let total = CALIBRATION_MISS_VEC_CAP + 25;
        let cache: Vec<FailureCacheEntry> = (0..total)
            .map(|_| miss_entry("add-neurons", 1.0, 100.0, None, None))
            .collect();
        tracker.record_calibration_misses_from_cache(&cache, 10.0);
        let summary = tracker.build_summary();
        assert_eq!(summary.calibration_miss_count as usize, total);
        let misses = summary
            .calibration_misses
            .expect("verbose mode should populate the vec");
        assert_eq!(misses.len(), CALIBRATION_MISS_VEC_CAP);
    }

    #[test]
    fn calibration_miss_threshold_clamps_invalid_inputs() {
        let tracker = McmcDiagnosticsTracker::new_verbose();
        let cache = vec![miss_entry("add-neurons", 1.0, 0.5, None, None)];
        // Threshold <= 1.0 collapses to threshold = 1.0 so the symmetric
        // window degenerates to exactly 1.0, marking any non-unit ratio as a
        // miss. This guards against silently disabling the check on bad input.
        tracker.record_calibration_misses_from_cache(&cache, -5.0);
        let summary = tracker.build_summary();
        assert_eq!(summary.calibration_miss_count, 1);
    }

    #[test]
    fn test_concurrent_counter_updates() {
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(McmcDiagnosticsTracker::new_verbose());
        let num_threads = 8;
        let proposals_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let tracker = Arc::clone(&tracker);
                thread::spawn(move || {
                    for i in 0..proposals_per_thread {
                        let src = format!("src-{t}-{i}");
                        let tgt = format!("tgt-{t}-{i}");
                        tracker.record_evaluated(&src, &tgt, CandidateType::Synapse);
                        if i % 3 == 0 {
                            tracker.record_accepted(&src, &tgt, 0.01, CandidateType::Synapse);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let summary = tracker.build_summary();
        let expected_proposed = num_threads * proposals_per_thread;
        // Every 3rd is accepted: 0, 3, 6, ... 99 → 34 per thread
        let expected_accepted = num_threads * 34;

        assert_eq!(
            summary.synapse_acceptance.proposed, expected_proposed as u32,
            "Total proposed should match"
        );
        assert_eq!(
            summary.synapse_acceptance.accepted, expected_accepted as u32,
            "Total accepted should match"
        );
    }
}
