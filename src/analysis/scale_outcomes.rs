//! Per-scale success rate tracking for weight variant selection (Issue #964).
//!
//! Extends the module outcome tracking system to record success rates per weight
//! scale magnitude (Conservative, Gentle Nudge, Micro-Nudge, Feather-Touch, Whisper)
//! per module type. Uses this data to learn which weight scales work best and bias
//! future variant generation toward historically successful scales.
//!
//! ## Design
//!
//! - Outcomes are keyed by `(module_name, weight_scale_tier)`.
//! - Uses the same Bayesian Beta(1,1) prior and smoothing as `ModuleOutcomeTracker`.
//! - Per-scale boost factors are clamped to [0.5, 2.0].
//! - Decay factor prevents stale data from permanently biasing scale selection.
//! - Serialisable to JSON for persistence alongside the creature.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for neural network computation (Issue #873)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::constants::MIN_BOOST_SAMPLES;

/// Per-scale success/failure statistics.
///
/// Tracks how many candidates at a particular weight scale tier have been
/// accepted or rejected during ablation testing.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ScaleStats {
    /// Total number of candidates at this scale that were ablation-tested.
    pub attempts: u32,
    /// Number of candidates at this scale that passed ablation testing.
    pub successes: u32,
}

impl ScaleStats {
    /// Returns the Bayesian success rate using Beta distribution posterior mean.
    ///
    /// Uses Beta(1,1) prior (uniform):
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

/// Tracker for per-(module, scale-tier) outcomes.
///
/// Serialisable to JSON for persistence alongside the creature, enabling
/// cross-run learning of which weight scales work best for each module.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ScaleOutcomeTracker {
    /// Outer key: module name, inner key: scale tier name.
    modules: HashMap<String, HashMap<String, ScaleStats>>,
}

impl Default for ScaleOutcomeTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ScaleOutcomeTracker {
    /// Creates a new empty tracker.
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    /// Returns true if no outcomes have been tracked.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Returns the number of distinct modules being tracked.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Records an accept/reject outcome for a specific (module, scale tier).
    pub fn record(&mut self, module_name: &str, scale_tier: &str, succeeded: bool) {
        let scales = self.modules.entry(module_name.to_string()).or_default();
        let stats = scales.entry(scale_tier.to_string()).or_default();
        stats.attempts += 1;
        if succeeded {
            stats.successes += 1;
        }
    }

    /// Returns the statistics for a given (module, scale tier) combination.
    ///
    /// Returns default (empty) stats for unknown combinations.
    pub fn stats(&self, module_name: &str, scale_tier: &str) -> ScaleStats {
        self.modules
            .get(module_name)
            .and_then(|scales| scales.get(scale_tier))
            .cloned()
            .unwrap_or_default()
    }

    /// Returns all module statistics as a nested map.
    pub fn all_stats(&self) -> &HashMap<String, HashMap<String, ScaleStats>> {
        &self.modules
    }

    /// Applies a decay factor to all scale statistics.
    ///
    /// Multiplies `attempts` and `successes` by the given `factor` (clamped to [0.0, 1.0]),
    /// rounding to nearest. This prevents historical data from permanently biasing
    /// scale selection by gradually reducing the weight of old outcomes.
    ///
    /// - `factor = 1.0` — no change (preserve all history)
    /// - `factor = 0.5` — halve all counts (moderate decay)
    /// - `factor = 0.0` — clear all counts (full reset)
    pub fn apply_decay(&mut self, factor: f64) {
        let factor = factor.clamp(0.0, 1.0);
        for scales in self.modules.values_mut() {
            for stats in scales.values_mut() {
                stats.attempts = (stats.attempts as f64 * factor).round() as u32;
                stats.successes = (stats.successes as f64 * factor).round() as u32;
                // Ensure successes never exceeds attempts after rounding.
                if stats.successes > stats.attempts {
                    stats.successes = stats.attempts;
                }
            }
        }
    }

    /// Returns a scoring boost factor for a specific (module, scale tier).
    ///
    /// The boost is based on the Bayesian success rate. Scale tiers with higher
    /// success rates get a larger boost.
    ///
    /// Returns 1.0 (neutral) when:
    /// - The combination is unknown (no data)
    /// - There are fewer than [`MIN_BOOST_SAMPLES`] attempts (insufficient data)
    ///
    /// The boost factor is computed as: `2.0 * success_rate`, clamped to [0.5, 2.0].
    pub fn scale_boost(&self, module_name: &str, scale_tier: &str) -> f64 {
        let Some(scales) = self.modules.get(module_name) else {
            return 1.0;
        };
        let Some(stats) = scales.get(scale_tier) else {
            return 1.0;
        };

        if (stats.attempts as usize) < MIN_BOOST_SAMPLES {
            return 1.0;
        }

        let rate = stats.success_rate();
        (2.0 * rate).clamp(0.5, 2.0)
    }
}

/// Known weight scale tier names matching variant generation comments.
const KNOWN_SCALE_TIERS: [&str; 5] = [
    "Conservative",
    "Gentle Nudge",
    "Micro-Nudge",
    "Feather-Touch",
    "Whisper",
];

/// Extract the weight scale tier from a candidate comment string.
///
/// Returns `None` if the comment does not match any known tier.
pub fn extract_scale_tier(comment: &str) -> Option<&'static str> {
    for tier in &KNOWN_SCALE_TIERS {
        if comment.starts_with(tier) || comment.contains(&format!("{tier} variant")) {
            return Some(tier);
        }
    }
    None
}

/// Extract the module name from a candidate comment string.
///
/// Module names are typically the prefix before `:` or `|` in comments.
fn extract_module_name(comment: &str) -> &str {
    comment
        .split(&[':', '|'][..])
        .next()
        .unwrap_or("unknown")
        .trim()
}

/// Apply per-scale boost factors to coordinated structural candidate expected gains (Issue #964).
///
/// For each candidate, extracts the module name and scale tier from the comment,
/// then multiplies `expected_creature_score_gain` by the tracker's `scale_boost()`
/// for that (module, tier) combination. Candidates from unknown modules/tiers or
/// an empty tracker receive a neutral boost (1.0).
pub fn apply_scale_boosts_to_candidates(
    candidates: &mut [crate::CoordinatedStructuralCandidateJson],
    tracker: &ScaleOutcomeTracker,
) {
    if tracker.is_empty() {
        return;
    }

    for candidate in candidates.iter_mut() {
        let comment = match candidate.comment.as_ref() {
            Some(c) => c.as_str(),
            None => continue,
        };

        let scale_tier = match extract_scale_tier(comment) {
            Some(tier) => tier,
            None => continue,
        };

        let module_name = extract_module_name(comment);
        let boost = tracker.scale_boost(module_name, scale_tier);
        if (boost - 1.0).abs() > f64::EPSILON {
            candidate.expected_creature_score_gain *= boost as f32;
        }
    }
}
