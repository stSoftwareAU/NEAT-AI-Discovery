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
