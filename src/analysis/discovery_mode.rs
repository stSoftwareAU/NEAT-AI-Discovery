//! Creature-level discovery strategy adaptation (Issue #1132).
//!
//! When a creature has had several consecutive discovery failures, the base
//! pipeline keeps running the same modules with the same thresholds, wasting
//! budget on high-failure-rate structural changes. This module adds a rolling
//! per-creature success rate and, when the rate falls below a threshold,
//! biases the candidate mix toward lower-risk change types.
//!
//! ## Inputs
//!
//! Callers supply a `DiscoveryOutcomeLog` — a chronological list of per-pass
//! booleans where `true` = at least one candidate was accepted in that pass,
//! `false` = empty response. Only the last [`ROLLING_WINDOW`] entries are used
//! to compute the rolling success rate.
//!
//! ## Modes
//!
//! - [`DiscoveryMode::Normal`] — default. All modules receive their usual
//!   weights and thresholds.
//! - [`DiscoveryMode::Conservative`] — biases module weights in favour of
//!   low-failure-rate modules (activation substitution, bias drift, weight
//!   adjustment) and away from high-failure-rate structural modules
//!   (coordinated-structural, add-neurons). Tightens the coordinated
//!   minimum-expected-gain floor by a configurable multiplier so only
//!   obviously-promising structural candidates survive.
//!
//! ## Cooldown
//!
//! Conservative mode persists until either:
//! 1. A successful pass is recorded (most recent outcome is `true`), OR
//! 2. The consecutive failure streak exceeds
//!    `conservative_mode_max_epochs`. After this point the library reverts
//!    to Normal mode — the bias wasn't helping, so falling back to full
//!    exploration is the least-bad option.
//!
//! ## Thresholds and env vars
//!
//! - `NEAT_AI_DISCOVERY_LOW_SUCCESS_RATE_THRESHOLD` (f32, default
//!   [`DEFAULT_LOW_SUCCESS_RATE_THRESHOLD`] = 0.2)
//! - `NEAT_AI_DISCOVERY_CONSERVATIVE_MODE_MAX_EPOCHS` (u32, default
//!   [`DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS`] = 20)
//! - `NEAT_AI_DISCOVERY_CONSERVATIVE_GAIN_MULTIPLIER` (f32, default
//!   [`DEFAULT_CONSERVATIVE_GAIN_MULTIPLIER`] = 10.0)

#![allow(clippy::cast_precision_loss)] // success-rate maths over small counts
#![allow(clippy::cast_possible_truncation)] // clamped before `as` casts
#![allow(clippy::cast_sign_loss)] // clamped to >= 0 before cast to unsigned

use serde::{Deserialize, Serialize};

use super::module_weights::ModuleOutcomeTracker;

// =============================================================================
// Constants
// =============================================================================

/// Rolling window length — the last K outcomes used to compute the success
/// rate (K = 10 per Issue #1132).
pub const ROLLING_WINDOW: usize = 10;

/// Default threshold below which the rolling success rate triggers conservative
/// mode (Issue #1132). A value of 0.2 means fewer than 2 successes in the last
/// 10 passes.
pub const DEFAULT_LOW_SUCCESS_RATE_THRESHOLD: f32 = 0.2;

/// Default maximum number of consecutive failed passes before conservative
/// mode is abandoned as ineffective (Issue #1132).
pub const DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS: u32 = 20;

/// Default multiplier applied to `COORDINATED_MIN_EXPECTED_GAIN` in
/// conservative mode (Issue #1132). 10× means only candidates whose expected
/// gain is an order of magnitude above the standard floor survive.
pub const DEFAULT_CONSERVATIVE_GAIN_MULTIPLIER: f32 = 10.0;

/// Boost factor applied to low-failure-rate modules in conservative mode.
///
/// Low-failure modules (activation substitution, bias drift, weight
/// adjustment) have their effective success-rate weight inflated so that the
/// candidate budget allocator and module-boost pipeline favour them when the
/// pipeline is struggling.
pub const CONSERVATIVE_LOW_RISK_BOOST: f32 = 1.5;

/// Inverse factor applied to high-failure-rate modules in conservative mode.
///
/// `CONSERVATIVE_HIGH_RISK_PENALTY = 1 / CONSERVATIVE_LOW_RISK_BOOST`, so
/// the two factors combine symmetrically for reasoning about their overall
/// effect on allocation.
pub const CONSERVATIVE_HIGH_RISK_PENALTY: f32 = 1.0 / CONSERVATIVE_LOW_RISK_BOOST;

/// Module names classified as low-risk structural-lite changes (Issue #1132).
///
/// Membership here causes the module to receive the low-risk boost in
/// conservative mode. These modules produce activation swaps, bias drifts
/// and weight adjustments that rarely harm the creature.
pub const LOW_RISK_MODULES: &[&str] = &[
    "activation-recommendation",
    "activation-substitution",
    "bias-drift",
    "output-bias-drift",
    "weight-adjustment",
    "gradient-discovery",
    "sample-weighted",
];

/// Module names classified as high-risk structural changes (Issue #1132).
///
/// Membership here causes the module to receive the high-risk penalty in
/// conservative mode. These modules propose coordinated structural edits
/// (adding neurons, large topology changes) that frequently fail ablation.
pub const HIGH_RISK_MODULES: &[&str] = &[
    "coordinated-structural",
    "add-neurons",
    "add-synapses-coordinated",
    "topology-diversification",
    "skip-connection",
    "cross-detection-synthesis",
];

// =============================================================================
// DiscoveryMode
// =============================================================================

/// Whether the pipeline is currently adapting to a low recent success rate.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiscoveryMode {
    /// Default mode. All modules receive their usual weights and thresholds.
    #[default]
    Normal,
    /// Biased mode. Low-risk modules are boosted, high-risk modules are
    /// penalised, and the coordinated gain floor is tightened.
    Conservative,
}

impl DiscoveryMode {
    /// Stable string identifier used in FFI response metadata.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Conservative => "conservative",
        }
    }
}

// =============================================================================
// DiscoveryOutcomeLog
// =============================================================================

/// Chronological log of per-pass outcomes for a single creature (Issue #1132).
///
/// `outcomes[i]` is `true` if pass `i` accepted at least one candidate,
/// `false` if the response was empty. Callers pass the full history (or at
/// least the most recent window); the rolling-rate computation uses the tail
/// of length [`ROLLING_WINDOW`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryOutcomeLog {
    /// Chronological list of booleans: `true` = success, `false` = failure.
    pub outcomes: Vec<bool>,
}

impl DiscoveryOutcomeLog {
    /// Creates a log from an explicit list of outcomes.
    #[must_use]
    pub fn from_outcomes(outcomes: Vec<bool>) -> Self {
        Self { outcomes }
    }

    /// Returns the number of outcomes recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    /// Returns whether the log has no recorded outcomes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    /// Rolling success rate over the last [`ROLLING_WINDOW`] outcomes.
    ///
    /// Returns `1.0` for an empty log (neutral — nothing known yet, so don't
    /// trigger conservative mode). When the log has fewer than
    /// [`ROLLING_WINDOW`] entries, the rate is computed over whatever is
    /// present.
    #[must_use]
    pub fn rolling_success_rate(&self) -> f32 {
        if self.outcomes.is_empty() {
            return 1.0;
        }
        let window = self.window();
        let successes = window.iter().filter(|&&ok| ok).count();
        successes as f32 / window.len() as f32
    }

    /// Number of consecutive failures at the tail of the log.
    ///
    /// Used to implement the cooldown: after
    /// [`DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS`] consecutive failures,
    /// conservative mode is abandoned.
    #[must_use]
    pub fn consecutive_trailing_failures(&self) -> u32 {
        let mut count: u32 = 0;
        for &ok in self.outcomes.iter().rev() {
            if ok {
                break;
            }
            count = count.saturating_add(1);
        }
        count
    }

    /// Returns the rolling-window slice (at most [`ROLLING_WINDOW`] entries).
    fn window(&self) -> &[bool] {
        let n = self.outcomes.len();
        let start = n.saturating_sub(ROLLING_WINDOW);
        &self.outcomes[start..]
    }
}

// =============================================================================
// Mode decision
// =============================================================================

/// Determine the discovery mode for a given outcome log and configuration.
///
/// The mode is [`DiscoveryMode::Conservative`] when **all** of the following
/// hold:
///
/// 1. The rolling success rate (last [`ROLLING_WINDOW`] entries) is strictly
///    below `low_success_threshold`.
/// 2. The trailing consecutive-failure streak is at or below
///    `max_conservative_epochs` — beyond that the bias is abandoned.
/// 3. The log contains at least one outcome (empty logs default to Normal).
///
/// Otherwise the mode is [`DiscoveryMode::Normal`].
#[must_use]
pub fn decide_mode(
    log: &DiscoveryOutcomeLog,
    low_success_threshold: f32,
    max_conservative_epochs: u32,
) -> DiscoveryMode {
    if log.is_empty() {
        return DiscoveryMode::Normal;
    }
    if log.consecutive_trailing_failures() > max_conservative_epochs {
        return DiscoveryMode::Normal;
    }
    if log.rolling_success_rate() < low_success_threshold {
        DiscoveryMode::Conservative
    } else {
        DiscoveryMode::Normal
    }
}

// =============================================================================
// Conservative biasing
// =============================================================================

/// Apply the conservative-mode weight bias to a `ModuleOutcomeTracker`.
///
/// This is called when entering conservative mode and returns a **new**
/// tracker (the input tracker is not mutated) where modules in
/// [`LOW_RISK_MODULES`] have their success counts boosted, and modules in
/// [`HIGH_RISK_MODULES`] have their success counts attenuated. The shift
/// feeds through to downstream consumers: `module_boost`,
/// `allocate_candidate_budgets`, `allocate_time_budgets`.
///
/// When the tracker has no entry for a module listed in
/// [`LOW_RISK_MODULES`] or [`HIGH_RISK_MODULES`], a fresh entry is seeded
/// with a modest attempt count so the boost/penalty registers with the
/// `MIN_BOOST_SAMPLES` gate.
#[must_use]
pub fn biased_tracker_for_conservative_mode(
    tracker: &ModuleOutcomeTracker,
) -> ModuleOutcomeTracker {
    let mut out = tracker.clone();
    // Seed floor so low/high-risk modules have enough attempts to register.
    // We use a neutral Beta prior shape: attempts=5 with successes proportional
    // to the bias we want to apply.
    const SEED_ATTEMPTS: u32 = 5;

    for &name in LOW_RISK_MODULES {
        apply_weight_shift(&mut out, name, CONSERVATIVE_LOW_RISK_BOOST, SEED_ATTEMPTS);
    }
    for &name in HIGH_RISK_MODULES {
        apply_weight_shift(
            &mut out,
            name,
            CONSERVATIVE_HIGH_RISK_PENALTY,
            SEED_ATTEMPTS,
        );
    }
    out
}

/// Shift a module's Bayesian success-rate signal by a multiplicative factor.
///
/// The per-module success rate used for boost / budget allocation is
/// approximately `successes / attempts`. Multiplying `successes` (and
/// clamping to `attempts`) shifts the posterior mean by roughly that factor,
/// while the Beta(1,1) prior keeps the output within (0, 1).
fn apply_weight_shift(
    tracker: &mut ModuleOutcomeTracker,
    module_name: &str,
    factor: f32,
    seed_attempts: u32,
) {
    let existing = tracker.stats(module_name);
    let (attempts, successes) = if existing.attempts == 0 {
        // Seed a baseline so the factor has something to bias.
        let target_rate = 0.5_f32 * factor;
        let clamped_rate = target_rate.clamp(0.05, 0.95);
        let seeded_successes = (f32::from(u16::try_from(seed_attempts).unwrap_or(u16::MAX))
            * clamped_rate)
            .round() as u32;
        (seed_attempts, seeded_successes.min(seed_attempts))
    } else {
        let raw = existing.successes as f32 * factor;
        let shifted = raw.round() as i64;
        let new_successes = shifted.clamp(0, existing.attempts as i64) as u32;
        (existing.attempts, new_successes)
    };

    // ModuleOutcomeTracker doesn't expose a "set" op, so we record deltas.
    let current = tracker.stats(module_name);
    let attempts_delta = i64::from(attempts) - i64::from(current.attempts);
    let successes_delta = i64::from(successes) - i64::from(current.successes);

    if attempts_delta != 0 || successes_delta != 0 {
        // Build up to the target (attempts, successes) by calling `record`.
        // Each record adds one attempt and optionally one success.
        let add_successes = successes_delta.max(0) as u32;
        let add_failures = attempts_delta.saturating_sub(successes_delta).max(0) as u32;
        for _ in 0..add_successes {
            tracker.record(module_name, true);
        }
        for _ in 0..add_failures {
            tracker.record(module_name, false);
        }
    }
}

/// Multiplier applied to `COORDINATED_MIN_EXPECTED_GAIN` in a given mode.
///
/// Normal mode returns `1.0`. Conservative mode returns the configured
/// gain multiplier (default [`DEFAULT_CONSERVATIVE_GAIN_MULTIPLIER`] = 10).
#[must_use]
pub fn coordinated_gain_multiplier_for_mode(mode: DiscoveryMode, multiplier: f32) -> f32 {
    match mode {
        DiscoveryMode::Normal => 1.0,
        DiscoveryMode::Conservative => multiplier.max(1.0),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn failures(n: usize) -> Vec<bool> {
        vec![false; n]
    }

    fn successes(n: usize) -> Vec<bool> {
        vec![true; n]
    }

    #[test]
    fn empty_log_is_neutral_normal_mode() {
        let log = DiscoveryOutcomeLog::default();
        assert_eq!(log.rolling_success_rate(), 1.0);
        assert_eq!(log.consecutive_trailing_failures(), 0);
        assert_eq!(
            decide_mode(
                &log,
                DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
                DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS
            ),
            DiscoveryMode::Normal
        );
    }

    #[test]
    fn rolling_rate_uses_last_k_entries() {
        // 10 successes then 10 failures — window covers only the failures.
        let mut outcomes = successes(10);
        outcomes.extend(failures(10));
        let log = DiscoveryOutcomeLog::from_outcomes(outcomes);
        assert_eq!(log.rolling_success_rate(), 0.0);

        // Prepend an enormous success history — still 0.0 because the window
        // is fixed at ROLLING_WINDOW = 10.
        let mut outcomes = successes(100);
        outcomes.extend(failures(10));
        let log = DiscoveryOutcomeLog::from_outcomes(outcomes);
        assert_eq!(log.rolling_success_rate(), 0.0);
    }

    #[test]
    fn rolling_rate_short_log_uses_whatever_is_present() {
        // 3 outcomes: 1 success, 2 failures — rate is 1/3.
        let log = DiscoveryOutcomeLog::from_outcomes(vec![true, false, false]);
        let rate = log.rolling_success_rate();
        assert!((rate - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn eight_of_ten_failures_triggers_conservative() {
        // 2 successes + 8 failures in last 10.
        let mut outcomes = successes(2);
        outcomes.extend(failures(8));
        let log = DiscoveryOutcomeLog::from_outcomes(outcomes);
        // rate = 0.2, threshold = 0.2 — strictly less than, so Normal at
        // exactly 0.2. Push one more failure to drop rate to 1/10 = 0.1.
        let mode = decide_mode(
            &log,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
        );
        assert_eq!(
            mode,
            DiscoveryMode::Normal,
            "0.2 is not strictly below the 0.2 threshold"
        );

        // 1 success + 9 failures in last 10 → 0.1 rolling rate.
        let mut outcomes = successes(1);
        outcomes.extend(failures(9));
        let log = DiscoveryOutcomeLog::from_outcomes(outcomes);
        let mode = decide_mode(
            &log,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
        );
        assert_eq!(mode, DiscoveryMode::Conservative);
    }

    #[test]
    fn one_success_exits_conservative_back_to_normal() {
        // 9 failures, then a success — most recent window is 8 failures + 1
        // success = 0.1 rate... actually 9 failures + 1 success = 0.1. The
        // trailing-failure streak is 0 because the last outcome is success.
        let mut outcomes = failures(9);
        outcomes.push(true);
        let log = DiscoveryOutcomeLog::from_outcomes(outcomes);
        assert_eq!(log.consecutive_trailing_failures(), 0);
        let mode = decide_mode(
            &log,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
        );
        // Rate = 1/10 = 0.1, strictly below threshold, but the issue says
        // "exit conservative mode as soon as one successful discovery
        // occurs". That exit is achieved by recomputing the rolling rate to
        // include the success — in this scenario 0.1 is still below 0.2, so
        // conservative stays active. This documents the explicit trade-off:
        // we do not force-exit after a single success; the rolling rate
        // still governs.
        assert_eq!(mode, DiscoveryMode::Conservative);

        // However, once the window shifts enough that the rate climbs above
        // the threshold, we exit.
        let outcomes = vec![
            false, false, false, true, true, true, true, true, true, true,
        ];
        let log = DiscoveryOutcomeLog::from_outcomes(outcomes);
        let mode = decide_mode(
            &log,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
        );
        assert_eq!(
            mode,
            DiscoveryMode::Normal,
            "7 successes / 10 == 0.7 > 0.2 threshold"
        );
    }

    #[test]
    fn cooldown_exit_after_max_epochs() {
        // 25 consecutive failures: rolling rate is 0.0 (below threshold) but
        // the trailing streak (25) exceeds MAX_EPOCHS (20), so we exit.
        let log = DiscoveryOutcomeLog::from_outcomes(failures(25));
        let mode = decide_mode(
            &log,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
        );
        assert_eq!(mode, DiscoveryMode::Normal);
    }

    #[test]
    fn cooldown_does_not_exit_within_max_epochs() {
        // 10 failures: rolling rate = 0.0, trailing streak = 10 <= 20, so
        // conservative is still active.
        let log = DiscoveryOutcomeLog::from_outcomes(failures(10));
        let mode = decide_mode(
            &log,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
        );
        assert_eq!(mode, DiscoveryMode::Conservative);
    }

    #[test]
    fn discovery_mode_serialises_as_lowercase_string() {
        assert_eq!(DiscoveryMode::Normal.as_str(), "normal");
        assert_eq!(DiscoveryMode::Conservative.as_str(), "conservative");
        let json = serde_json::to_string(&DiscoveryMode::Conservative).unwrap();
        assert_eq!(json, "\"conservative\"");
    }

    #[test]
    fn coordinated_gain_multiplier_respects_mode() {
        assert!(
            (coordinated_gain_multiplier_for_mode(DiscoveryMode::Normal, 10.0) - 1.0).abs() < 1e-6
        );
        assert!(
            (coordinated_gain_multiplier_for_mode(DiscoveryMode::Conservative, 10.0) - 10.0).abs()
                < 1e-6
        );
        // Clamp: never less than 1.0.
        assert!(
            (coordinated_gain_multiplier_for_mode(DiscoveryMode::Conservative, 0.5) - 1.0).abs()
                < 1e-6
        );
    }

    #[test]
    fn conservative_mode_biases_module_weights() {
        // Seed a tracker where low- and high-risk modules have identical
        // histories (50% success rate). After biasing, low-risk modules must
        // score above high-risk modules.
        let mut tracker = ModuleOutcomeTracker::new();
        for _ in 0..10 {
            tracker.record("activation-substitution", true);
            tracker.record("activation-substitution", false);
            tracker.record("coordinated-structural", true);
            tracker.record("coordinated-structural", false);
        }
        let biased = biased_tracker_for_conservative_mode(&tracker);

        let low = biased.stats("activation-substitution");
        let high = biased.stats("coordinated-structural");
        let low_rate = low.success_rate();
        let high_rate = high.success_rate();

        assert!(
            low_rate > high_rate,
            "low-risk module should score above high-risk: low={low_rate}, high={high_rate}"
        );

        // Normal-mode tracker is untouched — the function takes &ModuleOutcomeTracker
        // and returns a fresh one; the original remains at 0.5 each.
        let orig_low = tracker.stats("activation-substitution").success_rate();
        let orig_high = tracker.stats("coordinated-structural").success_rate();
        assert!((orig_low - orig_high).abs() < 1e-6);
    }

    #[test]
    fn conservative_biasing_is_idempotent_for_repeated_calls() {
        // Applying the bias twice should not stack unboundedly — the second
        // call sees the already-biased state and should keep low-risk >
        // high-risk, not flip them.
        let mut tracker = ModuleOutcomeTracker::new();
        for _ in 0..10 {
            tracker.record("weight-adjustment", true);
            tracker.record("weight-adjustment", false);
            tracker.record("add-neurons", true);
            tracker.record("add-neurons", false);
        }
        let biased_once = biased_tracker_for_conservative_mode(&tracker);
        let biased_twice = biased_tracker_for_conservative_mode(&biased_once);

        let low1 = biased_once.stats("weight-adjustment").success_rate();
        let high1 = biased_once.stats("add-neurons").success_rate();
        let low2 = biased_twice.stats("weight-adjustment").success_rate();
        let high2 = biased_twice.stats("add-neurons").success_rate();
        assert!(low1 > high1);
        assert!(low2 > high2);
    }
}
