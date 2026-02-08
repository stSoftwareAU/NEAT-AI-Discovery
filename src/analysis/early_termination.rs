//! Early termination for candidate evaluation using Sequential Probability Ratio Test (SPRT).
//!
//! Issue #219: Implements statistical early termination for clearly beneficial/harmful candidates.
//! This allows the GPU evaluation to stop early when a candidate is clearly good or clearly bad,
//! rather than evaluating all samples.
//!
//! ## Statistical Foundation
//!
//! The Sequential Probability Ratio Test (SPRT) is a statistical method that allows for
//! early termination of hypothesis testing while controlling error rates. It tests:
//!
//! - H0: The true improvement rate is at or below the threshold (candidate is not beneficial)
//! - H1: The true improvement rate exceeds the threshold (candidate is beneficial)
//!
//! The SPRT computes a log-likelihood ratio after each observation and compares it against
//! bounds derived from the desired error rates (alpha and beta).
//!
//! ## References
//!
//! - Wald, A. (1945). Sequential Tests of Statistical Hypotheses
//! - <https://en.wikipedia.org/wiki/Sequential_probability_ratio_test>

/// Decision result from the sequential evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyTerminationDecision {
    /// Continue collecting samples - no decision yet
    Continue,
    /// Accept the candidate as beneficial
    Accept,
    /// Reject the candidate as not beneficial
    Reject,
}

/// Sequential evaluator using the Sequential Probability Ratio Test (SPRT).
///
/// This evaluator tracks positive and negative sample counts and computes whether
/// we have enough statistical evidence to make an early decision about a candidate.
///
/// ## Usage
///
/// ```rust,ignore
/// let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);
///
/// for sample in samples {
///     let is_positive = sample.improves_error();
///     evaluator.add_sample(is_positive);
///
///     match evaluator.should_stop() {
///         EarlyTerminationDecision::Accept => {
///             // Candidate is beneficial, stop evaluation
///             break;
///         }
///         EarlyTerminationDecision::Reject => {
///             // Candidate is not beneficial, stop evaluation
///             break;
///         }
///         EarlyTerminationDecision::Continue => {
///             // Need more samples
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SequentialEvaluator {
    /// Type I error rate (false positive rate) - probability of accepting a bad candidate
    /// (Stored for debugging and future extensions; bounds are precomputed)
    #[allow(dead_code)]
    alpha: f64,
    /// Type II error rate (false negative rate) - probability of rejecting a good candidate
    /// (Stored for debugging and future extensions; bounds are precomputed)
    #[allow(dead_code)]
    beta: f64,
    /// Minimum improvement threshold (H1 hypothesis boundary)
    /// A threshold of 0.0 means we're testing if the improvement rate > 0.5 (better than random)
    threshold: f64,

    /// Count of samples that showed improvement
    positive_count: u64,
    /// Count of samples that did not show improvement
    negative_count: u64,

    /// Pre-computed upper bound for log-likelihood ratio (accept threshold)
    upper_bound: f64,
    /// Pre-computed lower bound for log-likelihood ratio (reject threshold)
    lower_bound: f64,

    /// Minimum samples before making any decision (for statistical validity)
    min_samples: u64,
}

impl SequentialEvaluator {
    /// Create a new sequential evaluator with the given error rates and threshold.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Type I error rate (false positive rate). Typical value: 0.01 (1%)
    /// * `beta` - Type II error rate (false negative rate). Typical value: 0.01 (1%)
    /// * `threshold` - Minimum improvement rate to detect. A value of 0.0 tests for
    ///   improvement > 50% (better than random).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // 1% false positive/negative rate, testing for any improvement above random
    /// let evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);
    /// ```
    #[must_use]
    pub fn new(alpha: f64, beta: f64, threshold: f64) -> Self {
        // Clamp alpha and beta to valid range (avoiding division by zero and log(0))
        let alpha = alpha.clamp(1e-10, 1.0 - 1e-10);
        let beta = beta.clamp(1e-10, 1.0 - 1e-10);

        // SPRT bounds:
        // Upper bound A = ln((1 - beta) / alpha)
        // Lower bound B = ln(beta / (1 - alpha))
        let upper_bound = ((1.0 - beta) / alpha).ln();
        let lower_bound = (beta / (1.0 - alpha)).ln();

        Self {
            alpha,
            beta,
            threshold,
            positive_count: 0,
            negative_count: 0,
            upper_bound,
            lower_bound,
            min_samples: 30, // Require at least 30 samples for statistical validity
        }
    }

    /// Add a single sample observation.
    ///
    /// # Arguments
    ///
    /// * `is_positive` - Whether this sample showed improvement
    pub fn add_sample(&mut self, is_positive: bool) {
        if is_positive {
            self.positive_count += 1;
        } else {
            self.negative_count += 1;
        }
    }

    /// Add a batch of samples (GPU-compatible interface).
    ///
    /// This is useful when processing batches of samples from GPU results.
    ///
    /// # Arguments
    ///
    /// * `positive_count` - Number of samples that showed improvement
    /// * `negative_count` - Number of samples that did not show improvement
    pub fn add_batch(&mut self, positive_count: u32, negative_count: u32) {
        self.positive_count += u64::from(positive_count);
        self.negative_count += u64::from(negative_count);
    }

    /// Check if we should stop evaluation and return the decision.
    ///
    /// Returns `Accept` if there's sufficient evidence the candidate is beneficial,
    /// `Reject` if there's sufficient evidence the candidate is not beneficial,
    /// or `Continue` if more samples are needed.
    #[must_use]
    pub fn should_stop(&self) -> EarlyTerminationDecision {
        let total = self.sample_count();

        // Require minimum samples for statistical validity
        if total < self.min_samples {
            return EarlyTerminationDecision::Continue;
        }

        let llr = self.log_likelihood_ratio();

        if llr >= self.upper_bound {
            EarlyTerminationDecision::Accept
        } else if llr <= self.lower_bound {
            EarlyTerminationDecision::Reject
        } else {
            EarlyTerminationDecision::Continue
        }
    }

    /// Compute the log-likelihood ratio for the current observations.
    ///
    /// The SPRT compares H1 (improvement rate = p1) vs H0 (improvement rate = p0).
    /// For our use case:
    /// - p0 = 0.5 + threshold/2 (null hypothesis: improvement is at threshold)
    /// - p1 = 0.5 + threshold/2 + delta (alternative: improvement exceeds threshold)
    ///
    /// We use a simplified formulation where we test if the observed proportion
    /// significantly exceeds 0.5 (or 0.5 + threshold for non-zero threshold).
    #[must_use]
    pub fn log_likelihood_ratio(&self) -> f64 {
        let k = self.positive_count as f64;
        let n = self.sample_count() as f64;

        if n == 0.0 {
            return 0.0;
        }

        // For SPRT testing p > p0 vs p <= p0:
        // We use p0 = 0.5 (null hypothesis: no better than random)
        // and p1 = 0.6 (alternative: meaningfully better than random)
        //
        // The threshold parameter adjusts p0: p0 = 0.5 + threshold/2
        // This means threshold=0 tests against 50%, threshold=0.1 tests against 55%, etc.

        let p0 = 0.5 + self.threshold / 2.0;
        let p1 = p0 + 0.1; // Alternative hypothesis: 10% better than null

        // Clamp probabilities to valid range to avoid log(0)
        let p0 = p0.clamp(1e-10, 1.0 - 1e-10);
        let p1 = p1.clamp(1e-10, 1.0 - 1e-10);

        // Log-likelihood ratio:
        // LLR = k * ln(p1/p0) + (n-k) * ln((1-p1)/(1-p0))
        let log_ratio_positive = (p1 / p0).ln();
        let log_ratio_negative = ((1.0 - p1) / (1.0 - p0)).ln();

        k * log_ratio_positive + (n - k) * log_ratio_negative
    }

    /// Get the total number of samples observed.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.positive_count + self.negative_count
    }

    /// Get the count of positive (improved) samples.
    #[must_use]
    pub fn positive_count(&self) -> u64 {
        self.positive_count
    }

    /// Get the count of negative (not improved) samples.
    #[must_use]
    pub fn negative_count(&self) -> u64 {
        self.negative_count
    }

    /// Get the SPRT decision bounds.
    ///
    /// Returns (lower_bound, upper_bound) for the log-likelihood ratio.
    #[must_use]
    pub fn get_bounds(&self) -> (f64, f64) {
        (self.lower_bound, self.upper_bound)
    }

    /// Reset the evaluator for reuse with a new candidate.
    pub fn reset(&mut self) {
        self.positive_count = 0;
        self.negative_count = 0;
    }

    /// Get the current improvement ratio (positive / total).
    #[must_use]
    pub fn improvement_ratio(&self) -> f64 {
        let total = self.sample_count();
        if total == 0 {
            return 0.5; // Default to neutral
        }
        self.positive_count as f64 / total as f64
    }

    /// Check if the candidate appears to be strongly beneficial.
    ///
    /// This is a quick heuristic check (not SPRT-based) for use in
    /// deciding whether to continue GPU batch evaluation.
    #[must_use]
    pub fn is_strongly_beneficial(&self) -> bool {
        let total = self.sample_count();
        if total < self.min_samples {
            return false;
        }
        // More than 70% positive is considered "strongly beneficial"
        self.improvement_ratio() > 0.7
    }

    /// Check if the candidate appears to be strongly harmful.
    ///
    /// This is a quick heuristic check (not SPRT-based) for use in
    /// deciding whether to continue GPU batch evaluation.
    #[must_use]
    pub fn is_strongly_harmful(&self) -> bool {
        let total = self.sample_count();
        if total < self.min_samples {
            return false;
        }
        // Less than 30% positive is considered "strongly harmful"
        self.improvement_ratio() < 0.3
    }
}

impl Default for SequentialEvaluator {
    /// Create a default evaluator with conservative error rates (1% each).
    fn default() -> Self {
        Self::new(0.01, 0.01, 0.0)
    }
}

/// Configuration for early termination behaviour.
#[derive(Debug, Clone)]
pub struct EarlyTerminationConfig {
    /// Whether early termination is enabled
    pub enabled: bool,
    /// Type I error rate (false positive rate)
    pub alpha: f64,
    /// Type II error rate (false negative rate)
    pub beta: f64,
    /// Minimum improvement threshold
    pub threshold: f64,
    /// Minimum samples before allowing early termination
    pub min_samples: u64,
    /// Check interval (check every N samples for early termination)
    pub check_interval: usize,
}

impl Default for EarlyTerminationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            alpha: 0.01,
            beta: 0.01,
            threshold: 0.0,
            min_samples: 100,
            check_interval: 1024,
        }
    }
}

impl EarlyTerminationConfig {
    /// Create a disabled early termination config (evaluate all samples).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Create a conservative config with higher minimum samples.
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            enabled: true,
            alpha: 0.001,
            beta: 0.001,
            threshold: 0.0,
            min_samples: 500,
            check_interval: 1024,
        }
    }

    /// Create an evaluator from this config.
    #[must_use]
    pub fn create_evaluator(&self) -> SequentialEvaluator {
        let mut evaluator = SequentialEvaluator::new(self.alpha, self.beta, self.threshold);
        evaluator.min_samples = self.min_samples;
        evaluator
    }
}

/// Result of checking early termination for a batch of candidates.
#[derive(Debug, Clone)]
pub struct EarlyTerminationResult {
    /// Indices of candidates that should be accepted (clearly beneficial)
    pub accept_indices: Vec<usize>,
    /// Indices of candidates that should be rejected (clearly harmful)
    pub reject_indices: Vec<usize>,
    /// Indices of candidates that need more samples
    pub continue_indices: Vec<usize>,
}

impl EarlyTerminationResult {
    /// Check if any candidates reached a decision.
    #[must_use]
    pub fn has_decisions(&self) -> bool {
        !self.accept_indices.is_empty() || !self.reject_indices.is_empty()
    }

    /// Check if all candidates need more samples.
    #[must_use]
    pub fn all_continue(&self) -> bool {
        self.accept_indices.is_empty() && self.reject_indices.is_empty()
    }
}

/// Check early termination for a batch of candidates based on their current stats.
///
/// This function evaluates each candidate's current statistics and returns which
/// candidates can be terminated early (accepted/rejected) vs which need more samples.
///
/// # Arguments
///
/// * `stats` - Vector of current statistics for each candidate
/// * `config` - Early termination configuration
///
/// # Returns
///
/// An `EarlyTerminationResult` indicating which candidates can terminate early.
pub fn check_batch_early_termination(
    stats: &[crate::analysis::samples::HelpfulStats],
    config: &EarlyTerminationConfig,
) -> EarlyTerminationResult {
    let mut result = EarlyTerminationResult {
        accept_indices: Vec::new(),
        reject_indices: Vec::new(),
        continue_indices: Vec::new(),
    };

    if !config.enabled {
        // When disabled, all candidates continue
        result.continue_indices = (0..stats.len()).collect();
        return result;
    }

    for (i, stat) in stats.iter().enumerate() {
        let mut evaluator = config.create_evaluator();
        evaluator.add_batch(stat.positive_count, stat.negative_count);

        match evaluator.should_stop() {
            EarlyTerminationDecision::Accept => {
                result.accept_indices.push(i);
            }
            EarlyTerminationDecision::Reject => {
                result.reject_indices.push(i);
            }
            EarlyTerminationDecision::Continue => {
                result.continue_indices.push(i);
            }
        }
    }

    result
}

// =============================================================================
// Issue #429: Early termination improvements for low-value candidates
// =============================================================================

// ---------------------------------------------------------------------------
// 1. Hierarchical candidate pre-filtering
// ---------------------------------------------------------------------------

/// Result of the quick pre-filter classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreFilterResult {
    /// Candidate is clearly beneficial — skip full SPRT evaluation.
    Accept,
    /// Candidate is clearly poor — skip full SPRT evaluation.
    Reject,
    /// Candidate is marginal — requires full SPRT evaluation.
    NeedsFullEvaluation,
}

/// Result of pre-filtering a batch of candidates.
#[derive(Debug, Clone)]
pub struct PreFilterBatchResult {
    /// Indices of candidates that pass the quick accept threshold.
    pub accept_indices: Vec<usize>,
    /// Indices of candidates that fail the quick reject threshold.
    pub reject_indices: Vec<usize>,
    /// Indices of candidates that need full SPRT evaluation.
    pub needs_eval_indices: Vec<usize>,
}

/// Quick pre-filter for candidate evaluation (Issue #429).
///
/// Performs a cheap ratio-based check before running the full SPRT analysis.
/// Candidates with extreme positive/negative ratios are classified immediately,
/// saving the cost of creating and running a `SequentialEvaluator`.
///
/// Thresholds are deliberately conservative to avoid filtering out good candidates:
/// - Accept: > 75% positive (well above the 60% SPRT H1 threshold)
/// - Reject: < 25% positive (well below the 50% SPRT H0 threshold)
/// - Between: needs full evaluation
#[derive(Debug, Clone)]
pub struct CandidatePreFilter {
    /// Minimum positive ratio to immediately accept (default: 0.75).
    accept_threshold: f64,
    /// Maximum positive ratio to immediately reject (default: 0.25).
    reject_threshold: f64,
    /// Minimum samples before pre-filter decisions are made (default: 30).
    min_samples: u32,
}

impl Default for CandidatePreFilter {
    fn default() -> Self {
        Self {
            accept_threshold: 0.75,
            reject_threshold: 0.25,
            min_samples: 30,
        }
    }
}

impl CandidatePreFilter {
    /// Classify a single candidate based on its current statistics.
    ///
    /// This is a quick heuristic — candidates classified as `NeedsFullEvaluation`
    /// should be passed to the full SPRT evaluator for a rigorous decision.
    #[must_use]
    pub fn classify(&self, stats: &crate::analysis::samples::HelpfulStats) -> PreFilterResult {
        let total = stats.positive_count + stats.negative_count;
        if total < self.min_samples {
            return PreFilterResult::NeedsFullEvaluation;
        }

        let ratio = f64::from(stats.positive_count) / f64::from(total);

        if ratio >= self.accept_threshold {
            PreFilterResult::Accept
        } else if ratio <= self.reject_threshold {
            PreFilterResult::Reject
        } else {
            PreFilterResult::NeedsFullEvaluation
        }
    }

    /// Pre-filter a batch of candidates, returning indices grouped by classification.
    #[must_use]
    pub fn filter_batch(
        &self,
        stats: &[crate::analysis::samples::HelpfulStats],
    ) -> PreFilterBatchResult {
        let mut result = PreFilterBatchResult {
            accept_indices: Vec::new(),
            reject_indices: Vec::new(),
            needs_eval_indices: Vec::new(),
        };

        for (i, stat) in stats.iter().enumerate() {
            match self.classify(stat) {
                PreFilterResult::Accept => result.accept_indices.push(i),
                PreFilterResult::Reject => result.reject_indices.push(i),
                PreFilterResult::NeedsFullEvaluation => result.needs_eval_indices.push(i),
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// 2. Budget-aware prioritisation
// ---------------------------------------------------------------------------

/// Tracks candidate generation budget to avoid wasting analysis time on
/// low-priority candidates when the budget is already full (Issue #429).
///
/// When the maximum number of candidates has been generated, further candidates
/// are rejected unless they have higher expected improvement than the current
/// worst candidate in the budget.
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    /// Maximum number of candidates to keep.
    budget: usize,
    /// Currently tracked candidates: (id, expected_improvement).
    candidates: Vec<(String, f32)>,
}

impl BudgetTracker {
    /// Create a new budget tracker with the given capacity.
    #[must_use]
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            candidates: Vec::with_capacity(budget),
        }
    }

    /// Try to add a candidate. Returns `true` if accepted (within budget).
    pub fn try_add(&mut self, id: String, expected_improvement: f32) -> bool {
        if self.candidates.len() < self.budget {
            self.candidates.push((id, expected_improvement));
            true
        } else {
            false
        }
    }

    /// Try to add a candidate, displacing the worst existing candidate if the
    /// new one has higher expected improvement.
    ///
    /// Returns `true` if the candidate was added (either within budget or by
    /// displacing a lower-value candidate).
    pub fn try_add_with_displacement(&mut self, id: String, expected_improvement: f32) -> bool {
        if self.candidates.len() < self.budget {
            self.candidates.push((id, expected_improvement));
            return true;
        }

        // Find the worst candidate (lowest improvement)
        let worst_idx = self
            .candidates
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        if let Some(idx) = worst_idx {
            if expected_improvement > self.candidates[idx].1 {
                self.candidates[idx] = (id, expected_improvement);
                return true;
            }
        }

        false
    }

    /// Get the number of candidates currently tracked.
    #[must_use]
    pub fn count(&self) -> usize {
        self.candidates.len()
    }

    /// Get the remaining capacity before the budget is exhausted.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.budget.saturating_sub(self.candidates.len())
    }

    /// Check if the budget is fully exhausted.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.candidates.len() >= self.budget
    }

    /// Check if a candidate with the given ID is currently tracked.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.candidates.iter().any(|(cid, _)| cid == id)
    }
}

// ---------------------------------------------------------------------------
// 3. Incremental confidence (added to SequentialEvaluator)
// ---------------------------------------------------------------------------

impl SequentialEvaluator {
    /// Compute a confidence score (0.0 to 1.0) for the current observation.
    ///
    /// Issue #429: This provides a quick confidence measure that can be used
    /// for early exit decisions even before the full SPRT reaches a conclusion.
    ///
    /// The score is based on:
    /// - How far the observed ratio deviates from 0.5 (neutral)
    /// - How many samples have been observed (more samples = higher confidence)
    ///
    /// Returns 0.0 when the ratio is near 0.5 or sample count is low,
    /// and approaches 1.0 when the ratio is extreme with many samples.
    #[must_use]
    pub fn confidence_score(&self) -> f64 {
        let total = self.sample_count();
        if total == 0 {
            return 0.0;
        }

        let ratio = self.improvement_ratio();

        // How far from neutral (0.5)? Range: 0.0 (neutral) to 0.5 (extreme)
        let deviation = (ratio - 0.5).abs();

        // Scale deviation to 0.0–1.0 range (deviation of 0.5 → 1.0)
        let deviation_factor = (deviation * 2.0).min(1.0);

        // Sample count factor: confidence increases with sqrt(n)
        // Full confidence at ~500 samples
        let sample_factor = ((total as f64).sqrt() / 500.0_f64.sqrt()).min(1.0);

        // Combined: both factors must be high for high confidence
        deviation_factor * sample_factor
    }
}

// ---------------------------------------------------------------------------
// 4. Cross-module deduplication
// ---------------------------------------------------------------------------

/// Tracks candidates across discovery modules to avoid generating duplicates
/// that target the same (source, target, operation) combination (Issue #429).
///
/// In large creatures, multiple discovery modules may independently identify
/// the same candidate. This deduplicator lets modules check whether a candidate
/// has already been registered before spending computation on it.
#[derive(Debug, Clone)]
pub struct CrossModuleDeduplicator {
    /// Set of registered (source_uuid, target_uuid, op_type) tuples.
    seen: std::collections::HashSet<(String, String, String)>,
    /// Count of duplicate registrations.
    duplicates: usize,
}

impl CrossModuleDeduplicator {
    /// Create a new empty deduplicator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            seen: std::collections::HashSet::new(),
            duplicates: 0,
        }
    }

    /// Register a candidate and return whether it is new (not a duplicate).
    ///
    /// # Arguments
    /// * `source_uuid` - UUID of the source neuron
    /// * `target_uuid` - UUID of the target neuron
    /// * `op_type` - Operation type (e.g., "addSynapse", "removeSynapse")
    /// * `_expected_improvement` - Expected improvement (reserved for future ranking)
    pub fn register_candidate(
        &mut self,
        source_uuid: &str,
        target_uuid: &str,
        op_type: &str,
        _expected_improvement: f32,
    ) -> bool {
        let key = (
            source_uuid.to_string(),
            target_uuid.to_string(),
            op_type.to_string(),
        );
        if self.seen.insert(key) {
            true // new
        } else {
            self.duplicates += 1;
            false // duplicate
        }
    }

    /// Get the count of unique candidates registered.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.seen.len()
    }

    /// Get the count of duplicate registrations.
    #[must_use]
    pub fn duplicate_count(&self) -> usize {
        self.duplicates
    }
}

impl Default for CrossModuleDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_evaluator_starts_with_zero_counts() {
        let evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);
        assert_eq!(evaluator.sample_count(), 0);
        assert_eq!(evaluator.positive_count(), 0);
        assert_eq!(evaluator.negative_count(), 0);
    }

    #[test]
    fn test_add_sample_increments_counts() {
        let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

        evaluator.add_sample(true);
        assert_eq!(evaluator.positive_count(), 1);
        assert_eq!(evaluator.negative_count(), 0);

        evaluator.add_sample(false);
        assert_eq!(evaluator.positive_count(), 1);
        assert_eq!(evaluator.negative_count(), 1);
    }

    #[test]
    fn test_add_batch_adds_multiple_samples() {
        let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

        evaluator.add_batch(100, 50);
        assert_eq!(evaluator.positive_count(), 100);
        assert_eq!(evaluator.negative_count(), 50);
        assert_eq!(evaluator.sample_count(), 150);
    }

    #[test]
    fn test_reset_clears_counts() {
        let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

        evaluator.add_batch(100, 50);
        evaluator.reset();

        assert_eq!(evaluator.sample_count(), 0);
        assert_eq!(evaluator.positive_count(), 0);
        assert_eq!(evaluator.negative_count(), 0);
    }

    #[test]
    fn test_improvement_ratio_calculation() {
        let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

        // Empty evaluator returns 0.5
        assert!((evaluator.improvement_ratio() - 0.5).abs() < 1e-10);

        evaluator.add_batch(75, 25);
        assert!((evaluator.improvement_ratio() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_bounds_are_symmetric_for_equal_error_rates() {
        let evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);
        let (lower, upper) = evaluator.get_bounds();

        // With equal alpha and beta, bounds should be symmetric around 0
        assert!((lower + upper).abs() < 0.01);
    }

    #[test]
    fn test_should_stop_requires_minimum_samples() {
        let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

        // Even with 100% positive, shouldn't decide with too few samples
        for _ in 0..10 {
            evaluator.add_sample(true);
        }

        // Should continue until minimum samples reached
        assert!(matches!(
            evaluator.should_stop(),
            EarlyTerminationDecision::Continue
        ));
    }

    #[test]
    fn test_strongly_beneficial_with_enough_samples() {
        let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

        // Add 50 samples with 90% positive (need > 30 for is_strongly_beneficial)
        evaluator.add_batch(45, 5);

        assert!(evaluator.is_strongly_beneficial());
        assert!(!evaluator.is_strongly_harmful());
    }

    #[test]
    fn test_strongly_harmful_with_enough_samples() {
        let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

        // Add 50 samples with 10% positive
        evaluator.add_batch(5, 45);

        assert!(evaluator.is_strongly_harmful());
        assert!(!evaluator.is_strongly_beneficial());
    }

    #[test]
    fn test_log_likelihood_ratio_increases_with_positive_samples() {
        let mut evaluator1 = SequentialEvaluator::new(0.01, 0.01, 0.0);
        let mut evaluator2 = SequentialEvaluator::new(0.01, 0.01, 0.0);

        evaluator1.add_batch(80, 20);
        evaluator2.add_batch(20, 80);

        let llr1 = evaluator1.log_likelihood_ratio();
        let llr2 = evaluator2.log_likelihood_ratio();

        assert!(llr1 > llr2, "Higher positive ratio should have higher LLR");
        assert!(llr1 > 0.0, "80% positive should have positive LLR");
        assert!(llr2 < 0.0, "20% positive should have negative LLR");
    }
}
