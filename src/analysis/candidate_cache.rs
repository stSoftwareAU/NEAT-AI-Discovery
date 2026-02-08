//! Candidate outcome cache for tracking discovery candidate success/failure (Issue #465).
//!
//! This module provides a cache that records which candidates have been suggested
//! and whether they succeeded or failed ablation testing. By tracking per-candidate
//! outcomes, we can:
//!
//! 1. **Suppress recently-failed candidates** — avoid wasting the discovery budget
//!    on candidates that have already been tried and rejected.
//! 2. **Re-enable candidates after a staleness window** — the creature evolves over
//!    time, so a previously-failed candidate may become viable after structural changes.
//! 3. **Track per-source-type success rates** — input neurons as synapse sources
//!    have historically higher success rates (36.2% vs ~3% for hidden neurons in
//!    GRQ-sampler data). This enables source-type-aware boosting.
//!
//! # Key Design Decisions
//!
//! - Candidates are keyed by `(source_uuid, target_uuid, operation_type)`.
//! - Only the **most recent** outcome is stored per candidate key (not a history).
//! - The staleness window is configurable and defaults to [`DEFAULT_STALENESS_WINDOW`].
//! - Source-type stats use Bayesian scoring (same approach as `DiscoveryHistory`).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::constants::MIN_BOOST_SAMPLES;

/// Default staleness window in epochs.
///
/// Failed candidates become re-eligible after this many epochs have passed,
/// allowing them to be re-evaluated if the creature has changed structurally.
pub const DEFAULT_STALENESS_WINDOW: u64 = 100;

/// Outcome of a single candidate's ablation test.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CandidateOutcome {
    /// Whether the candidate improved the creature's score.
    pub succeeded: bool,
    /// Epoch at which this outcome was recorded.
    pub epoch: u64,
}

/// Per-source-type success/failure statistics.
///
/// Tracks how many candidates from each source type (input, hidden, etc.)
/// have succeeded or failed ablation testing. Uses Bayesian scoring for
/// robust estimates with low sample counts.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SourceTypeStats {
    /// Total number of candidates from this source type that were tested.
    pub attempts: u32,
    /// Number of candidates from this source type that succeeded.
    pub successes: u32,
}

impl SourceTypeStats {
    /// Returns the Bayesian success rate using Beta distribution posterior mean.
    ///
    /// Uses the same Beta(1,1) prior as `DiscoveryHistory::bayesian_score()`:
    /// - No data → 0.5 (neutral prior)
    /// - Converges to raw success rate with many samples
    /// - Never returns exactly 0.0 or 1.0
    pub fn success_rate(&self) -> f64 {
        if self.attempts == 0 {
            return 0.5;
        }
        let alpha = self.successes as f64 + 1.0;
        let beta = (self.attempts - self.successes) as f64 + 1.0;
        alpha / (alpha + beta)
    }
}

/// Cache of candidate outcomes for suppressing repeated failures and tracking
/// source-type success rates.
///
/// # Persistence
///
/// The cache is serialisable to JSON for storage alongside the creature,
/// enabling cross-run candidate memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CandidateOutcomeCache {
    /// Per-candidate outcomes keyed by "(source_uuid, target_uuid, operation_type)".
    outcomes: HashMap<String, CandidateOutcome>,
    /// Per-source-type success/failure statistics.
    source_type_stats: HashMap<String, SourceTypeStats>,
    /// Staleness window in epochs — failed candidates become re-eligible after this.
    staleness_window: u64,
}

impl Default for CandidateOutcomeCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CandidateOutcomeCache {
    /// Creates a new empty cache with the default staleness window.
    pub fn new() -> Self {
        Self {
            outcomes: HashMap::new(),
            source_type_stats: HashMap::new(),
            staleness_window: DEFAULT_STALENESS_WINDOW,
        }
    }

    /// Creates a new empty cache with a custom staleness window.
    pub fn with_staleness_window(staleness_window: u64) -> Self {
        Self {
            outcomes: HashMap::new(),
            source_type_stats: HashMap::new(),
            staleness_window,
        }
    }

    /// Returns true if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    /// Returns the number of candidate outcome entries.
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    /// Returns the configured staleness window in epochs.
    pub fn staleness_window(&self) -> u64 {
        self.staleness_window
    }

    /// Builds the composite key for a candidate.
    fn make_key(source_uuid: &str, target_uuid: &str, operation: &str) -> String {
        format!("{source_uuid}|{target_uuid}|{operation}")
    }

    /// Records the outcome of a candidate's ablation test.
    ///
    /// If the candidate has a previous outcome, it is replaced by the new one.
    pub fn record(
        &mut self,
        source_uuid: &str,
        target_uuid: &str,
        operation: &str,
        succeeded: bool,
        epoch: u64,
    ) {
        let key = Self::make_key(source_uuid, target_uuid, operation);
        self.outcomes
            .insert(key, CandidateOutcome { succeeded, epoch });
    }

    /// Records a candidate outcome and also updates source-type statistics.
    pub fn record_with_source_type(
        &mut self,
        source_uuid: &str,
        target_uuid: &str,
        operation: &str,
        succeeded: bool,
        epoch: u64,
        source_type: &str,
    ) {
        self.record(source_uuid, target_uuid, operation, succeeded, epoch);

        let stats = self
            .source_type_stats
            .entry(source_type.to_string())
            .or_default();
        stats.attempts += 1;
        if succeeded {
            stats.successes += 1;
        }
    }

    /// Returns the most recent outcome for a candidate, if any.
    pub fn get_outcome(
        &self,
        source_uuid: &str,
        target_uuid: &str,
        operation: &str,
    ) -> Option<&CandidateOutcome> {
        let key = Self::make_key(source_uuid, target_uuid, operation);
        self.outcomes.get(&key)
    }

    /// Returns true if the candidate should be suppressed at the given epoch.
    ///
    /// A candidate is suppressed if:
    /// 1. It has a recorded **failed** outcome, AND
    /// 2. The failure occurred within the staleness window (epoch - failure_epoch < staleness_window)
    ///
    /// Successful candidates and unknown candidates are never suppressed.
    pub fn is_suppressed(
        &self,
        source_uuid: &str,
        target_uuid: &str,
        operation: &str,
        current_epoch: u64,
    ) -> bool {
        let Some(outcome) = self.get_outcome(source_uuid, target_uuid, operation) else {
            return false;
        };

        if outcome.succeeded {
            return false;
        }

        // Failed candidate: check if within staleness window
        current_epoch < outcome.epoch + self.staleness_window
    }

    /// Returns the success statistics for a given source type.
    ///
    /// Returns default (empty) stats for unknown source types.
    pub fn source_type_stats(&self, source_type: &str) -> SourceTypeStats {
        self.source_type_stats
            .get(source_type)
            .cloned()
            .unwrap_or_default()
    }

    /// Returns a scoring boost factor for candidates from the given source type.
    ///
    /// The boost is based on the Bayesian success rate of candidates from this
    /// source type. Source types with higher success rates get a larger boost.
    ///
    /// Returns 1.0 (neutral) when:
    /// - The source type is unknown (no data)
    /// - There are fewer than [`MIN_BOOST_SAMPLES`] samples (insufficient data)
    ///
    /// The boost factor is computed as: `2.0 * success_rate`, clamped to [0.5, 2.0].
    /// This means:
    /// - 50% success rate → 1.0 (neutral)
    /// - 75% success rate → 1.5 (moderate boost)
    /// - 25% success rate → 0.5 (penalty)
    pub fn source_type_boost(&self, source_type: &str) -> f64 {
        let Some(stats) = self.source_type_stats.get(source_type) else {
            return 1.0;
        };

        if (stats.attempts as usize) < MIN_BOOST_SAMPLES {
            return 1.0;
        }

        let rate = stats.success_rate();
        (2.0 * rate).clamp(0.5, 2.0)
    }

    /// Removes all entries with epochs older than the given threshold.
    ///
    /// This prevents the cache from growing unboundedly over long training runs.
    pub fn prune_before_epoch(&mut self, min_epoch: u64) {
        self.outcomes
            .retain(|_, outcome| outcome.epoch >= min_epoch);
    }
}
