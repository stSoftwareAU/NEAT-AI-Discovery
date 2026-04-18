//! Discovery History Tracking for Issue #227.
//!
//! This module provides types for tracking which neurons have historically led to
//! successful discoveries (candidates that survived ablation testing). By tracking
//! this history, we can prioritise neurons with higher success rates in future
//! discovery runs, improving the discovery hit rate.
//!
//! # Key Features
//!
//! - **Bayesian scoring**: Uses Beta distribution posterior mean for robust scoring
//!   that handles low sample sizes appropriately.
//! - **Neutral prior**: New neurons (not yet attempted) get a 0.5 (50%) prior,
//!   giving them a fair chance while still prioritising proven neurons.
//! - **Persistence**: History can be serialised to JSON for storage alongside
//!   the creature.
//!
//! # Example
//!
//! ```
//! use neat_ai_discovery::discovery_history::DiscoveryHistory;
//!
//! let mut history = DiscoveryHistory::new();
//!
//! // Record some ablation test results
//! history.record("hidden-1", true, Some(12345));  // Success at epoch 12345
//! history.record("hidden-1", false, None);         // Failure
//! history.record("hidden-2", true, Some(12346));   // Success
//!
//! // Get Bayesian scores for ranking
//! let score_1 = history.bayesian_score_for("hidden-1");  // ~0.5 (1 success, 1 failure)
//! let score_2 = history.bayesian_score_for("hidden-2");  // ~0.67 (1 success, 0 failures)
//! let score_unknown = history.bayesian_score_for("unknown");  // 0.5 (neutral prior)
//!
//! assert!(score_2 > score_1);
//! ```

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Calibration Tracking (Issue #605)
// =============================================================================

/// Build a string key from module name and candidate type for `HashMap` lookup.
fn calibration_key(module_name: &str, candidate_type: &str) -> String {
    format!("{module_name}::{candidate_type}")
}

/// Parse a calibration key back into (`module_name`, `candidate_type`).
fn parse_calibration_key(key: &str) -> (&str, &str) {
    key.split_once("::").unwrap_or((key, ""))
}

/// A single predicted vs actual improvement observation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct CalibrationObservation {
    predicted: f64,
    actual: f64,
}

/// Summary of calibration metrics for a single module/candidate-type combination.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationSummaryEntry {
    /// Discovery module name (e.g. "saturation", "bottleneck").
    pub module_name: String,
    /// Candidate type (e.g. "addSynapse", "addNeuron").
    pub candidate_type: String,
    /// Number of recorded predictions.
    pub sample_count: usize,
    /// Mean absolute error between predicted and actual improvement.
    pub mean_absolute_error: f64,
    /// Bias direction: positive means over-prediction, negative means under-prediction.
    /// Computed as mean(predicted - actual).
    pub bias: f64,
    /// Calibration factor: multiply future predictions by this to correct bias.
    /// Computed as mean(actual / predicted), clamped to [0.1, 10.0].
    pub calibration_factor: f64,
}

/// Tracks predicted vs actual improvement accuracy per discovery module and
/// candidate type (Issue #605).
///
/// Records observations of (predicted improvement, actual improvement) and
/// computes calibration metrics: mean absolute error, bias direction, and
/// per-module calibration factors.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationTracker {
    /// Observations keyed by "`module_name::candidate_type`".
    observations: HashMap<String, Vec<CalibrationObservation>>,
}

/// Minimum number of observations before calibration factor is applied.
const MIN_CALIBRATION_SAMPLES: usize = 1;

/// Clamp range for calibration factors to prevent extreme corrections.
const CALIBRATION_FACTOR_MIN: f64 = 0.1;
const CALIBRATION_FACTOR_MAX: f64 = 10.0;

impl CalibrationTracker {
    /// Creates a new empty calibration tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a predicted vs actual improvement observation.
    ///
    /// # Arguments
    ///
    /// * `module_name` - The discovery module that produced the candidate
    /// * `candidate_type` - The type of candidate (e.g. "addSynapse", "addNeuron")
    /// * `predicted` - The predicted improvement (expected creature score gain)
    /// * `actual` - The actual improvement observed after ablation testing
    pub fn record_prediction(
        &mut self,
        module_name: &str,
        candidate_type: &str,
        predicted: f64,
        actual: f64,
    ) {
        let key = calibration_key(module_name, candidate_type);
        self.observations
            .entry(key)
            .or_default()
            .push(CalibrationObservation { predicted, actual });
    }

    /// Returns a calibration factor for the given module and candidate type.
    ///
    /// Multiply future predictions by this factor to correct for systematic bias.
    /// Returns 1.0 (no correction) if no observations exist for the combination.
    pub fn calibration_factor(&self, module_name: &str, candidate_type: &str) -> f64 {
        let key = calibration_key(module_name, candidate_type);

        let observations = match self.observations.get(&key) {
            Some(obs) if obs.len() >= MIN_CALIBRATION_SAMPLES => obs,
            _ => return 1.0,
        };

        compute_calibration_factor(observations)
    }

    /// Returns a summary of calibration metrics for all tracked modules.
    ///
    /// Results are sorted by sample count (descending) for easy review.
    pub fn calibration_summary(&self) -> Vec<CalibrationSummaryEntry> {
        let mut entries: Vec<CalibrationSummaryEntry> = self
            .observations
            .iter()
            .map(|(key, obs)| {
                let (module_name, candidate_type) = parse_calibration_key(key);
                let n = obs.len() as f64;
                let mae: f64 = obs
                    .iter()
                    .map(|o| (o.predicted - o.actual).abs())
                    .sum::<f64>()
                    / n;
                let bias: f64 = obs.iter().map(|o| o.predicted - o.actual).sum::<f64>() / n;
                let factor = compute_calibration_factor(obs);

                CalibrationSummaryEntry {
                    module_name: module_name.to_string(),
                    candidate_type: candidate_type.to_string(),
                    sample_count: obs.len(),
                    mean_absolute_error: mae,
                    bias,
                    calibration_factor: factor,
                }
            })
            .collect();

        entries.sort_by_key(|b| std::cmp::Reverse(b.sample_count));
        entries
    }
}

/// Compute calibration factor from observations as mean(actual / predicted),
/// clamped to a reasonable range.
fn compute_calibration_factor(observations: &[CalibrationObservation]) -> f64 {
    if observations.is_empty() {
        return 1.0;
    }

    // Use ratio-based calibration: mean(actual / predicted)
    // Skip entries where predicted is near zero to avoid division issues.
    let epsilon = 1e-10;
    let mut ratio_sum = 0.0;
    let mut ratio_count = 0u32;

    for obs in observations {
        if obs.predicted.abs() > epsilon {
            ratio_sum += obs.actual / obs.predicted;
            ratio_count += 1;
        } else {
            // Predicted was ~0 but actual may be non-zero.
            // Treat as a large under-prediction: cap ratio.
            ratio_sum += CALIBRATION_FACTOR_MAX;
            ratio_count += 1;
        }
    }

    if ratio_count == 0 {
        return 1.0;
    }

    let mean_ratio = ratio_sum / ratio_count as f64;
    mean_ratio.clamp(CALIBRATION_FACTOR_MIN, CALIBRATION_FACTOR_MAX)
}

/// Tracks discovery history for a single neuron.
///
/// This struct records how many times a neuron has been selected as a focus target
/// and how many of those attempts resulted in successful discoveries (candidates
/// that survived ablation testing).
///
/// # Bayesian Scoring
///
/// Rather than using raw success rate (which can be misleading with few samples),
/// we use a Bayesian approach with a Beta distribution prior:
///
/// - Prior: Beta(1, 1) - equivalent to 1 pseudo-success and 1 pseudo-failure
/// - Posterior: Beta(successes + 1, failures + 1)
/// - Score: posterior mean = (successes + 1) / (attempts + 2)
///
/// This provides several benefits:
/// - New neurons get score 0.5 (neutral), not 0.0 or undefined
/// - A single success doesn't give score 1.0 (would be (1+1)/(1+2) = 0.67)
/// - A single failure doesn't give score 0.0 (would be (0+1)/(1+2) = 0.33)
/// - With many samples, the score converges to the true success rate
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NeuronDiscoveryHistory {
    /// The neuron's UUID
    uuid: String,
    /// Total number of times this neuron was selected for discovery
    attempts: u32,
    /// Number of successful discoveries (ablation tests that improved score)
    successes: u32,
    /// Epoch timestamp of the most recent successful discovery
    #[serde(skip_serializing_if = "Option::is_none")]
    last_success_epoch: Option<u64>,
}

impl NeuronDiscoveryHistory {
    /// Creates a new history entry for a neuron with no prior attempts.
    pub fn new(uuid: String) -> Self {
        Self {
            uuid,
            attempts: 0,
            successes: 0,
            last_success_epoch: None,
        }
    }

    /// Returns the neuron's UUID.
    pub fn uuid(&self) -> &str {
        &self.uuid
    }

    /// Returns the total number of discovery attempts for this neuron.
    pub fn attempts(&self) -> u32 {
        self.attempts
    }

    /// Returns the number of successful discoveries.
    pub fn successes(&self) -> u32 {
        self.successes
    }

    /// Returns the epoch timestamp of the last successful discovery, if any.
    pub fn last_success_epoch(&self) -> Option<u64> {
        self.last_success_epoch
    }

    /// Returns the raw success rate (successes / attempts).
    ///
    /// For neurons with no attempts, returns 0.5 (neutral prior).
    /// For most purposes, prefer `bayesian_score()` which handles
    /// low sample sizes more appropriately.
    pub fn success_rate(&self) -> f64 {
        if self.attempts == 0 {
            0.5 // Neutral prior for new neurons
        } else {
            self.successes as f64 / self.attempts as f64
        }
    }

    /// Returns the Bayesian score using Beta distribution posterior mean.
    ///
    /// This is the recommended scoring method as it:
    /// - Handles low sample sizes appropriately
    /// - Gives new neurons a neutral prior (0.5)
    /// - Converges to true success rate with many samples
    /// - Never returns exactly 0.0 or 1.0 (regularised)
    ///
    /// # Formula
    ///
    /// ```text
    /// score = (successes + 1) / (attempts + 2)
    /// ```
    ///
    /// This is the posterior mean of Beta(α, β) where:
    /// - α = successes + 1 (prior: 1 pseudo-success)
    /// - β = failures + 1 (prior: 1 pseudo-failure)
    pub fn bayesian_score(&self) -> f64 {
        let alpha = self.successes as f64 + 1.0;
        let beta = (self.attempts - self.successes) as f64 + 1.0;
        alpha / (alpha + beta)
    }

    /// Records a discovery attempt result.
    ///
    /// # Arguments
    ///
    /// * `success` - Whether the discovery attempt resulted in a candidate that
    ///   improved the creature's score (survived ablation testing).
    /// * `epoch` - Optional epoch timestamp. If the attempt was successful, this
    ///   updates `last_success_epoch`.
    pub fn record_attempt(&mut self, success: bool, epoch: Option<u64>) {
        self.attempts += 1;
        if success {
            self.successes += 1;
            if let Some(e) = epoch {
                self.last_success_epoch = Some(e);
            }
        }
    }
}

/// Container for discovery history across all neurons in a creature.
///
/// This struct provides methods to:
/// - Record discovery attempt results for any neuron
/// - Retrieve history for a specific neuron
/// - Get Bayesian scores for ranking (with neutral prior for unknown neurons)
/// - Serialise/deserialise history for persistence
///
/// # Storage Format
///
/// When serialised to JSON, the history is stored as:
///
/// ```json
/// {
///   "neurons": {
///     "hidden-1": { "uuid": "hidden-1", "attempts": 10, "successes": 3, "lastSuccessEpoch": 12345 },
///     "hidden-2": { "uuid": "hidden-2", "attempts": 5, "successes": 4, "lastSuccessEpoch": 12340 }
///   }
/// }
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryHistory {
    /// History entries keyed by neuron UUID
    neurons: HashMap<String, NeuronDiscoveryHistory>,
    /// Calibration tracker for predicted vs actual improvement (Issue #605)
    #[serde(default)]
    calibration: CalibrationTracker,
}

impl DiscoveryHistory {
    /// Creates a new empty discovery history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if the history contains no entries.
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Returns the number of neurons with history entries.
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Returns the history for a specific neuron, if it exists.
    pub fn get(&self, neuron_uuid: &str) -> Option<&NeuronDiscoveryHistory> {
        self.neurons.get(neuron_uuid)
    }

    /// Records a discovery attempt result for a neuron.
    ///
    /// If the neuron doesn't have an existing history entry, one is created.
    ///
    /// # Arguments
    ///
    /// * `neuron_uuid` - The UUID of the neuron that was selected for discovery
    /// * `success` - Whether the discovery resulted in a candidate that improved
    ///   the creature's score (survived ablation testing)
    /// * `epoch` - Optional epoch timestamp for tracking when successes occurred
    pub fn record(&mut self, neuron_uuid: &str, success: bool, epoch: Option<u64>) {
        let entry = self
            .neurons
            .entry(neuron_uuid.to_string())
            .or_insert_with(|| NeuronDiscoveryHistory::new(neuron_uuid.to_string()));
        entry.record_attempt(success, epoch);
    }

    /// Returns the Bayesian score for a neuron.
    ///
    /// For neurons not in the history, returns 0.5 (neutral prior), giving
    /// new neurons a fair chance during focus selection.
    pub fn bayesian_score_for(&self, neuron_uuid: &str) -> f64 {
        self.neurons
            .get(neuron_uuid)
            .map_or(0.5, NeuronDiscoveryHistory::bayesian_score) // Neutral prior for unknown neurons
    }

    /// Returns an iterator over all neuron history entries.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &NeuronDiscoveryHistory)> {
        self.neurons.iter()
    }

    /// Clears all history entries.
    pub fn clear(&mut self) {
        self.neurons.clear();
    }

    /// Records a calibration observation (predicted vs actual improvement).
    ///
    /// Delegates to the internal `CalibrationTracker`.
    pub fn record_calibration(
        &mut self,
        module_name: &str,
        candidate_type: &str,
        predicted: f64,
        actual: f64,
    ) {
        self.calibration
            .record_prediction(module_name, candidate_type, predicted, actual);
    }

    /// Returns the calibration factor for a given module and candidate type.
    ///
    /// Returns 1.0 if no calibration data exists for the combination.
    pub fn calibration_factor(&self, module_name: &str, candidate_type: &str) -> f64 {
        self.calibration
            .calibration_factor(module_name, candidate_type)
    }

    /// Returns a summary of calibration metrics for all tracked modules.
    pub fn calibration_summary(&self) -> Vec<CalibrationSummaryEntry> {
        self.calibration.calibration_summary()
    }

    /// Removes history for neurons that are no longer in the creature.
    ///
    /// This is useful when a creature has been mutated and some neurons
    /// have been removed. Call this after mutation to prevent the history
    /// from growing unboundedly.
    ///
    /// # Arguments
    ///
    /// * `current_neuron_uuids` - UUIDs of neurons currently in the creature
    pub fn prune(&mut self, current_neuron_uuids: &[&str]) {
        let current_set: std::collections::HashSet<&str> =
            current_neuron_uuids.iter().copied().collect();
        self.neurons
            .retain(|uuid, _| current_set.contains(uuid.as_str()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_history_new() {
        let h = NeuronDiscoveryHistory::new("test".to_string());
        assert_eq!(h.uuid(), "test");
        assert_eq!(h.attempts(), 0);
        assert_eq!(h.successes(), 0);
        assert!(h.last_success_epoch().is_none());
    }

    #[test]
    fn test_bayesian_score_progression() {
        let mut h = NeuronDiscoveryHistory::new("test".to_string());

        // No attempts: neutral prior
        assert!((h.bayesian_score() - 0.5).abs() < 0.001);

        // 1 success: (1+1)/(1+2) = 2/3 ≈ 0.667
        h.record_attempt(true, None);
        assert!((h.bayesian_score() - 2.0 / 3.0).abs() < 0.001);

        // 1 success, 1 failure: (1+1)/(2+2) = 2/4 = 0.5
        h.record_attempt(false, None);
        assert!((h.bayesian_score() - 0.5).abs() < 0.001);

        // 2 successes, 1 failure: (2+1)/(3+2) = 3/5 = 0.6
        h.record_attempt(true, None);
        assert!((h.bayesian_score() - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_discovery_history_prune() {
        let mut history = DiscoveryHistory::new();
        history.record("keep-1", true, None);
        history.record("keep-2", true, None);
        history.record("remove-1", true, None);
        history.record("remove-2", true, None);

        assert_eq!(history.len(), 4);

        history.prune(&["keep-1", "keep-2"]);

        assert_eq!(history.len(), 2);
        assert!(history.get("keep-1").is_some());
        assert!(history.get("keep-2").is_some());
        assert!(history.get("remove-1").is_none());
        assert!(history.get("remove-2").is_none());
    }
}
