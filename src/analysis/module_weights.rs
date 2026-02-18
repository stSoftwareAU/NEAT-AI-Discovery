//! Per-module success rate tracking for adaptive discovery weighting (Issue #485).
//!
//! Tracks which discovery modules produce candidates that are accepted or rejected
//! by NEAT-AI's ablation testing. Uses Bayesian smoothing (Beta(1,1) prior) to
//! provide robust success rate estimates even with limited data.
//!
//! ## Design
//!
//! - Each discovery module (saturation, dead neuron, bottleneck, etc.) gets its own
//!   success/failure counter.
//! - Boost factors are computed from Bayesian success rates, clamped to [0.5, 2.0].
//! - Modules with insufficient data receive a neutral boost (1.0).
//! - No module is ever entirely starved — the minimum boost is 0.5, not 0.0.
//!
//! ## Usage
//!
//! The tracker is populated during `run_discovery_modules_parallel()` with the number
//! of candidates each module produced. Accept/reject outcomes are recorded externally
//! by the NEAT-AI controller when ablation results are available.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::constants::MIN_BOOST_SAMPLES;

/// Per-module success/failure statistics.
///
/// Tracks how many candidates from each discovery module have been accepted or
/// rejected, plus total candidates produced.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModuleStats {
    /// Total number of candidates from this module that were ablation-tested.
    pub attempts: u32,
    /// Number of candidates from this module that passed ablation testing.
    pub successes: u32,
    /// Total candidates produced by this module (may exceed attempts if some
    /// candidates have not yet been tested).
    pub candidates_produced: u32,
}

impl ModuleStats {
    /// Returns the Bayesian success rate using Beta distribution posterior mean.
    ///
    /// Uses the same Beta(1,1) prior as `CandidateOutcomeCache::SourceTypeStats`:
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

/// Tracker for per-discovery-module outcomes.
///
/// Serialisable to JSON for persistence alongside the creature, enabling
/// cross-run module effectiveness memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModuleOutcomeTracker {
    modules: HashMap<String, ModuleStats>,
}

impl Default for ModuleOutcomeTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleOutcomeTracker {
    /// Creates a new empty tracker.
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    /// Returns true if no modules have been tracked.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Returns the number of distinct modules being tracked.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Records an accept/reject outcome for a module's candidate.
    pub fn record(&mut self, module_name: &str, succeeded: bool) {
        let stats = self.modules.entry(module_name.to_string()).or_default();
        stats.attempts += 1;
        if succeeded {
            stats.successes += 1;
        }
    }

    /// Records the number of candidates produced by a module in a single run.
    pub fn record_candidates(&mut self, module_name: &str, count: usize) {
        let stats = self.modules.entry(module_name.to_string()).or_default();
        stats.candidates_produced += count as u32;
    }

    /// Returns the statistics for a given module.
    ///
    /// Returns default (empty) stats for unknown modules.
    pub fn stats(&self, module_name: &str) -> ModuleStats {
        self.modules.get(module_name).cloned().unwrap_or_default()
    }

    /// Returns all module statistics as a map.
    pub fn all_stats(&self) -> &HashMap<String, ModuleStats> {
        &self.modules
    }

    /// Applies a decay factor to all module statistics (Issue #603).
    ///
    /// Multiplies `attempts` and `successes` by the given `factor` (clamped to [0.0, 1.0]),
    /// rounding down. This prevents historical data from permanently biasing allocation
    /// by gradually reducing the weight of old outcomes.
    ///
    /// - `factor = 1.0` — no change (preserve all history)
    /// - `factor = 0.5` — halve all counts (moderate decay)
    /// - `factor = 0.0` — clear all counts (full reset)
    ///
    /// The success rate is approximately preserved because both numerator and denominator
    /// are scaled by the same factor.
    pub fn apply_decay(&mut self, factor: f64) {
        let factor = factor.clamp(0.0, 1.0);
        for stats in self.modules.values_mut() {
            stats.attempts = (stats.attempts as f64 * factor).round() as u32;
            stats.successes = (stats.successes as f64 * factor).round() as u32;
            stats.candidates_produced = (stats.candidates_produced as f64 * factor).round() as u32;
            // Ensure successes never exceeds attempts after rounding.
            if stats.successes > stats.attempts {
                stats.successes = stats.attempts;
            }
        }
    }

    /// Returns a scoring boost factor for candidates from the given module.
    ///
    /// The boost is based on the Bayesian success rate. Modules with higher
    /// success rates get a larger boost.
    ///
    /// Returns 1.0 (neutral) when:
    /// - The module is unknown (no data)
    /// - There are fewer than [`MIN_BOOST_SAMPLES`] attempts (insufficient data)
    ///
    /// The boost factor is computed as: `2.0 * success_rate`, clamped to [0.5, 2.0].
    pub fn module_boost(&self, module_name: &str) -> f64 {
        let Some(stats) = self.modules.get(module_name) else {
            return 1.0;
        };

        if (stats.attempts as usize) < MIN_BOOST_SAMPLES {
            return 1.0;
        }

        let rate = stats.success_rate();
        (2.0 * rate).clamp(0.5, 2.0)
    }
}

/// Allocate time budgets to discovery modules based on historical yield (Issue #603).
///
/// Returns a map from module name to allocated time in milliseconds. The allocation
/// strategy is:
///
/// 1. **Sufficient history**: Modules with >= [`MIN_BOOST_SAMPLES`] attempts get time
///    proportional to their Bayesian success rate, with a minimum floor of 0.5 to
///    prevent starvation.
/// 2. **Sparse history**: Modules with fewer attempts get a neutral weight (1.0),
///    equivalent to proportional allocation.
/// 3. **Decay factor**: The `decay_factor` parameter (0.0–1.0) blends between equal
///    allocation (0.0) and fully history-driven allocation (1.0). This prevents
///    historical data from permanently biasing the allocation.
///
/// # Arguments
///
/// * `module_names` — The names of modules to allocate time for.
/// * `tracker` — Historical outcome tracker with per-module success/failure data.
/// * `total_ms` — Total time budget to distribute across all modules.
/// * `decay_factor` — How much historical data influences allocation (0.0 = equal, 1.0 = full).
pub fn allocate_time_budgets(
    module_names: &[String],
    tracker: &ModuleOutcomeTracker,
    total_ms: u64,
    decay_factor: f64,
) -> HashMap<String, u64> {
    let mut budgets = HashMap::new();
    if module_names.is_empty() || total_ms == 0 {
        return budgets;
    }

    let decay_factor = decay_factor.clamp(0.0, 1.0);

    // Compute a weight for each module based on historical yield.
    // Modules with sufficient data get a weight derived from their Bayesian success rate.
    // Modules without sufficient data get a neutral weight (1.0).
    let equal_weight = 1.0;
    let min_adaptive_weight = 0.5;

    let weights: Vec<f64> = module_names
        .iter()
        .map(|name| {
            let stats = tracker.stats(name);
            let adaptive_weight = if (stats.attempts as usize) >= MIN_BOOST_SAMPLES {
                // Use success rate as the weight, with a floor to prevent starvation.
                stats.success_rate().max(min_adaptive_weight)
            } else {
                // Insufficient data — neutral weight.
                equal_weight
            };
            // Blend between equal and adaptive based on decay factor.
            equal_weight * (1.0 - decay_factor) + adaptive_weight * decay_factor
        })
        .collect();

    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        // Safety: shouldn't happen with the floor, but fall back to equal.
        let per_module = total_ms / module_names.len() as u64;
        for name in module_names {
            budgets.insert(name.clone(), per_module);
        }
        return budgets;
    }

    // Distribute time proportionally, handling rounding.
    let mut allocated: u64 = 0;
    for (i, name) in module_names.iter().enumerate() {
        let share = if i == module_names.len() - 1 {
            // Last module gets the remainder to avoid rounding loss.
            total_ms - allocated
        } else {
            let fraction = weights[i] / total_weight;
            (total_ms as f64 * fraction).round() as u64
        };
        budgets.insert(name.clone(), share);
        allocated += share;
    }

    budgets
}

/// Per-module statistics for JSON metadata output.
///
/// A flattened representation of module stats suitable for inclusion in
/// the analysis metadata JSON response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryModuleStatsJson {
    /// Name of the discovery module.
    pub module_name: String,
    /// Number of candidates this module produced in the current run.
    pub candidates_produced: usize,
    /// Number of candidates previously tested via ablation (from tracker).
    pub attempts: u32,
    /// Number of previously tested candidates that succeeded (from tracker).
    pub successes: u32,
    /// Bayesian success rate (0.0–1.0). 0.5 when no data available.
    pub success_rate: f64,
}
