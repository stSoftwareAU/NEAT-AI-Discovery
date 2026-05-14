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
//!
//! # Operator playbook
//!
//! See [`docs/DROUGHT_PLAYBOOK.md`](https://github.com/stSoftwareAU/NEAT-AI-Discovery/blob/main/docs/DROUGHT_PLAYBOOK.md)
//! for the end-to-end drought diagnostic walkthrough — how this cache, the
//! target cooldown tracker, conservative-mode bias, and post-processing
//! rejection filters interact, and which env var to reach for when
//! `candidateCacheSuppressedCount` dominates the diagnostic.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::constants::{
    MIN_BOOST_SAMPLES, STALENESS_WINDOW_FLOOR, staleness_conservative_divisor,
    staleness_extended_drought_divisor,
};
use super::discovery_mode::DiscoveryMode;
use crate::config::conservative_mode_max_epochs;

/// Default staleness window in epochs.
///
/// Failed candidates become re-eligible after this many epochs have passed,
/// allowing them to be re-evaluated if the creature has changed structurally.
pub const DEFAULT_STALENESS_WINDOW: u64 = 100;

/// Sentinel value indicating no effective window has been observed yet.
///
/// Used by `last_effective_window` to suppress the "transition" log on the
/// very first call (there is no prior state to compare against, so it is
/// not a transition).
const NO_PRIOR_EFFECTIVE_WINDOW: u64 = u64::MAX;

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
///
/// # Adaptive staleness window (Issue #1203)
///
/// The cache exposes both the configured `staleness_window()` and an
/// adaptive [`Self::effective_staleness_window`] that shrinks the window when
/// the discovery pipeline is struggling:
///
/// | Mode + drought                                    | Effective window         |
/// |---------------------------------------------------|--------------------------|
/// | `Normal` (or drought below the conservative cap)  | `staleness_window`       |
/// | `Conservative`, drought `< conservative_mode_max` | `staleness_window / 2`   |
/// | drought `>= conservative_mode_max` (extended)     | `max(staleness_window / 4, 5)` |
///
/// ## Worked example
///
/// With the default `staleness_window = 100` and
/// `conservative_mode_max_epochs = 20`:
///
/// - A candidate that fails at epoch 0 is suppressed at epoch 60 in `Normal`
///   mode (60 < 100). In `Conservative` mode the effective window is
///   `100 / 2 = 50`, so the same candidate is **re-eligible** at epoch 60
///   (60 >= 50). The suppression boundary moves from epoch 100 down to
///   epoch 50.
/// - If `drought_failures` then crosses 20 (conservative mode reverts to
///   Normal), the effective window drops to `100 / 4 = 25`, so any
///   candidate that failed at epoch 0 is re-eligible from epoch 25 onward.
///
/// `is_suppressed` always resolves the effective window via
/// [`Self::effective_staleness_window`]; callers must not bypass the adaptive
/// logic with the raw `staleness_window` field.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateOutcomeCache {
    /// Per-candidate outcomes keyed by "(`source_uuid`, `target_uuid`, `operation_type`)".
    outcomes: HashMap<String, CandidateOutcome>,
    /// Per-source-type success/failure statistics.
    source_type_stats: HashMap<String, SourceTypeStats>,
    /// Staleness window in epochs — failed candidates become re-eligible after this.
    staleness_window: u64,
    /// Last effective staleness window observed, used to emit a single
    /// `tracing::info!` log per transition (Issue #1203). Not serialised — a
    /// freshly deserialised cache always logs its first transition.
    #[serde(skip, default = "default_last_effective_window")]
    last_effective_window: AtomicU64,
    /// Epoch at which the drought-driven one-shot reset last fired (Issue
    /// #1205). `None` until the first reset; cleared by `record`/
    /// `record_with_source_type` when a successful outcome arrives so the
    /// next future drought can re-arm the lever.
    #[serde(default)]
    tombstone_reset_epoch: Option<u64>,
}

fn default_last_effective_window() -> AtomicU64 {
    AtomicU64::new(NO_PRIOR_EFFECTIVE_WINDOW)
}

impl Clone for CandidateOutcomeCache {
    fn clone(&self) -> Self {
        Self {
            outcomes: self.outcomes.clone(),
            source_type_stats: self.source_type_stats.clone(),
            staleness_window: self.staleness_window,
            // Reset the transition tracker so a cloned cache re-logs its
            // first transition. Cloning is rare (config snapshots, tests) and
            // the next call resolves the same effective window deterministically.
            last_effective_window: AtomicU64::new(
                self.last_effective_window.load(Ordering::Relaxed),
            ),
            tombstone_reset_epoch: self.tombstone_reset_epoch,
        }
    }
}

impl PartialEq for CandidateOutcomeCache {
    fn eq(&self, other: &Self) -> bool {
        // `last_effective_window` is observational state (a log de-duplicator)
        // and intentionally excluded from equality — two caches with the same
        // outcomes, stats, and window are semantically equal regardless of
        // their log history.
        self.outcomes == other.outcomes
            && self.source_type_stats == other.source_type_stats
            && self.staleness_window == other.staleness_window
            && self.tombstone_reset_epoch == other.tombstone_reset_epoch
    }
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
            last_effective_window: AtomicU64::new(NO_PRIOR_EFFECTIVE_WINDOW),
            tombstone_reset_epoch: None,
        }
    }

    /// Creates a new empty cache with a custom staleness window.
    pub fn with_staleness_window(staleness_window: u64) -> Self {
        Self {
            outcomes: HashMap::new(),
            source_type_stats: HashMap::new(),
            staleness_window,
            last_effective_window: AtomicU64::new(NO_PRIOR_EFFECTIVE_WINDOW),
            tombstone_reset_epoch: None,
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
    ///
    /// A successful outcome clears the drought-reset tombstone (Issue #1205)
    /// so the next future drought can re-arm the one-shot reset.
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
        if succeeded {
            self.tombstone_reset_epoch = None;
        }
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

    /// Returns the adaptive effective staleness window for the given mode and
    /// drought failure count (Issue #1203).
    ///
    /// The window shrinks while the pipeline is in conservative mode or
    /// experiencing an extended drought, so previously-failed candidates
    /// become re-eligible sooner. See the [`CandidateOutcomeCache`] doc
    /// comment for the full table and a worked example.
    ///
    /// Emits a one-shot `tracing::info!` log whenever the effective window
    /// changes from the value seen on the previous call (mode transition or
    /// crossing the extended-drought boundary). Repeated calls with the same
    /// (mode, drought) inputs do not log.
    #[must_use]
    pub fn effective_staleness_window(&self, mode: DiscoveryMode, drought_failures: u32) -> u64 {
        let conservative_max = conservative_mode_max_epochs();
        let effective = if drought_failures >= conservative_max {
            // Extended drought: quartered window with a hard floor of 5.
            let divisor = staleness_extended_drought_divisor().max(1);
            (self.staleness_window / divisor).max(STALENESS_WINDOW_FLOOR)
        } else if matches!(mode, DiscoveryMode::Conservative) {
            // Conservative mode without extended drought: halved window,
            // still respecting the floor for very small base windows.
            let divisor = staleness_conservative_divisor().max(1);
            (self.staleness_window / divisor).max(STALENESS_WINDOW_FLOOR)
        } else {
            // Normal mode, no drought: full configured window.
            self.staleness_window
        };

        // Emit a single tracing::info! per transition. The first call after
        // construction or deserialisation is not treated as a transition.
        let prior = self
            .last_effective_window
            .swap(effective, Ordering::Relaxed);
        if prior != NO_PRIOR_EFFECTIVE_WINDOW && prior != effective {
            tracing::info!(
                old_window = prior,
                new_window = effective,
                mode = mode.as_str(),
                drought_failures,
                conservative_max,
                "Candidate-cache effective staleness window changed"
            );
        }

        effective
    }

    /// Returns true if the candidate should be suppressed at the given epoch.
    ///
    /// A candidate is suppressed if:
    /// 1. It has a recorded **failed** outcome, AND
    /// 2. The failure occurred within the **effective** staleness window
    ///    (Issue #1203) resolved from `mode` and `drought_failures` via
    ///    [`Self::effective_staleness_window`].
    ///
    /// Successful candidates and unknown candidates are never suppressed.
    pub fn is_suppressed(
        &self,
        source_uuid: &str,
        target_uuid: &str,
        operation: &str,
        current_epoch: u64,
        mode: DiscoveryMode,
        drought_failures: u32,
    ) -> bool {
        let Some(outcome) = self.get_outcome(source_uuid, target_uuid, operation) else {
            return false;
        };

        if outcome.succeeded {
            return false;
        }

        // Failed candidate: check if within the effective (adaptive) window.
        let window = self.effective_staleness_window(mode, drought_failures);
        current_epoch < outcome.epoch + window
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

    /// Returns the number of failed candidate entries still within the
    /// configured staleness window at `current_epoch` (Issue #1202).
    ///
    /// Used by the drought diagnostic to surface how many cached failures are
    /// actively suppressing new candidate generation. Successful entries and
    /// entries whose failure timestamp has aged past the staleness window are
    /// not counted.
    ///
    /// The check uses the **base** [`Self::staleness_window`] rather than the
    /// adaptive [`Self::effective_staleness_window`]. The diagnostic reports
    /// the conservative upper bound (the count visible under default,
    /// non-shrunk behaviour) so operators can reason about the worst-case
    /// suppression footprint without entangling the count with mode/drought
    /// state already reported alongside it.
    #[must_use]
    pub fn suppressed_count(&self, current_epoch: u64) -> usize {
        let window = self.staleness_window;
        self.outcomes
            .values()
            .filter(|outcome| {
                !outcome.succeeded && current_epoch < outcome.epoch.saturating_add(window)
            })
            .count()
    }

    /// Returns the epoch at which the drought-driven reset was last fired,
    /// if any (Issue #1205). `None` means the lever is currently re-armed
    /// (either never fired, or a successful pass cleared the tombstone).
    #[must_use]
    pub fn drought_reset_tombstone(&self) -> Option<u64> {
        self.tombstone_reset_epoch
    }

    /// Clear the drought-reset tombstone explicitly (Issue #1205).
    ///
    /// Normal usage relies on [`Self::record`] / [`Self::record_with_source_type`]
    /// to clear the tombstone when a successful outcome arrives. Callers that
    /// observe a successful pass without recording a per-candidate outcome
    /// (e.g. the orchestrator's outcome-log path) can use this method to
    /// re-arm the lever directly.
    pub fn clear_drought_reset_tombstone(&mut self) {
        self.tombstone_reset_epoch = None;
    }

    /// Remove all failed outcome entries from the cache (Issue #1205).
    ///
    /// Operator-controlled escape hatch invoked after a configurable drought.
    /// Successful entries and per-source-type statistics are preserved — the
    /// "institutional memory" that informs scoring stays intact. The
    /// drought-reset tombstone is set to `current_epoch` so the same streak
    /// cannot trigger a second reset; the next successful pass (via
    /// [`Self::record`]) re-arms the lever.
    ///
    /// Returns the number of failed entries removed.
    pub fn clear_failed_entries(&mut self, current_epoch: u64) -> usize {
        let before = self.outcomes.len();
        self.outcomes.retain(|_, outcome| outcome.succeeded);
        let removed = before.saturating_sub(self.outcomes.len());
        self.tombstone_reset_epoch = Some(current_epoch);
        removed
    }
}
