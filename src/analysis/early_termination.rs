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

/// Minimum samples required for prefilter decisions.
///
/// Below this threshold, the improvement ratio is too noisy for reliable
/// pre-filtering. Candidates with fewer samples pass through to full SPRT.
const PREFILTER_MIN_SAMPLES: u32 = 20;

/// Improvement ratio below which a candidate is clearly poor.
///
/// Candidates with an improvement ratio below this threshold are rejected
/// by the prefilter without needing full SPRT evaluation.
const PREFILTER_REJECT_RATIO: f64 = 0.25;

/// Improvement ratio above which a candidate is clearly good.
///
/// Candidates with an improvement ratio above this threshold are accepted
/// by the prefilter without needing full SPRT evaluation.
const PREFILTER_ACCEPT_RATIO: f64 = 0.75;

/// Hierarchical candidate pre-filter (Issue #429).
///
/// Performs a quick pass over candidate statistics to identify clearly
/// poor or clearly good candidates before running the more expensive
/// SPRT evaluation. This reduces computation for large candidate sets
/// by filtering out obvious cases early.
///
/// Candidates with fewer than [`PREFILTER_MIN_SAMPLES`] samples are
/// always passed through to SPRT (they need more data).
///
/// # Arguments
/// * `stats` - Slice of candidate statistics to pre-filter.
///
/// # Returns
/// An [`EarlyTerminationResult`] with pre-filtered decisions.
pub fn prefilter_candidates(
    stats: &[crate::analysis::samples::HelpfulStats],
) -> EarlyTerminationResult {
    let mut result = EarlyTerminationResult {
        accept_indices: Vec::new(),
        reject_indices: Vec::new(),
        continue_indices: Vec::new(),
    };

    for (i, stat) in stats.iter().enumerate() {
        let total = stat.positive_count + stat.negative_count;
        if total < PREFILTER_MIN_SAMPLES {
            result.continue_indices.push(i);
            continue;
        }

        let ratio = f64::from(stat.positive_count) / f64::from(total);
        if ratio <= PREFILTER_REJECT_RATIO {
            result.reject_indices.push(i);
        } else if ratio >= PREFILTER_ACCEPT_RATIO {
            result.accept_indices.push(i);
        } else {
            result.continue_indices.push(i);
        }
    }

    result
}

/// Budget-aware candidate evaluation (Issue #429).
///
/// Evaluates candidates using SPRT but stops accepting new candidates once
/// the budget is exhausted. Candidates beyond the budget are rejected to
/// focus computational resources on high-value candidates.
///
/// The budget represents the maximum number of candidates that can be
/// accepted or left undecided (i.e., worth further evaluation).
///
/// # Arguments
/// * `stats` - Slice of candidate statistics to evaluate.
/// * `config` - Early termination configuration.
/// * `budget` - Maximum number of candidates to accept or continue evaluating.
///
/// # Returns
/// An [`EarlyTerminationResult`] with budget-constrained decisions.
pub fn budget_aware_evaluate(
    stats: &[crate::analysis::samples::HelpfulStats],
    config: &EarlyTerminationConfig,
    budget: usize,
) -> EarlyTerminationResult {
    let mut result = EarlyTerminationResult {
        accept_indices: Vec::new(),
        reject_indices: Vec::new(),
        continue_indices: Vec::new(),
    };

    let mut remaining_budget = budget;

    for (i, stat) in stats.iter().enumerate() {
        // Budget exhausted — reject remaining candidates
        if remaining_budget == 0 {
            result.reject_indices.push(i);
            continue;
        }

        if !config.enabled {
            result.continue_indices.push(i);
            remaining_budget = remaining_budget.saturating_sub(1);
            continue;
        }

        let mut evaluator = config.create_evaluator();
        evaluator.add_batch(stat.positive_count, stat.negative_count);

        match evaluator.should_stop() {
            EarlyTerminationDecision::Accept => {
                result.accept_indices.push(i);
                remaining_budget = remaining_budget.saturating_sub(1);
            }
            EarlyTerminationDecision::Reject => {
                result.reject_indices.push(i);
                // Rejected candidates don't consume budget
            }
            EarlyTerminationDecision::Continue => {
                result.continue_indices.push(i);
                remaining_budget = remaining_budget.saturating_sub(1);
            }
        }
    }

    result
}

/// Evaluate a single candidate with confidence scoring (Issue #429).
///
/// Combines SPRT evaluation with a confidence metric based on the
/// improvement ratio and sample count. This allows callers to prioritise
/// high-confidence decisions and skip candidates where the signal is weak.
///
/// # Arguments
/// * `stat` - Candidate statistics.
/// * `config` - Early termination configuration.
///
/// # Returns
/// A tuple of `(decision, confidence)` where confidence is in [0.0, 1.0].
pub fn evaluate_with_confidence(
    stat: &crate::analysis::samples::HelpfulStats,
    config: &EarlyTerminationConfig,
) -> (EarlyTerminationDecision, f64) {
    let mut evaluator = config.create_evaluator();
    evaluator.add_batch(stat.positive_count, stat.negative_count);

    let decision = if config.enabled {
        evaluator.should_stop()
    } else {
        EarlyTerminationDecision::Continue
    };

    // Compute confidence as a function of sample count and signal strength.
    // Confidence increases with more samples and with stronger deviation from 50%.
    let total = evaluator.sample_count();
    if total == 0 {
        return (decision, 0.0);
    }

    let ratio = evaluator.improvement_ratio();
    // Signal strength: how far from 50/50 (0.0 = no signal, 0.5 = max signal)
    let signal_strength = (ratio - 0.5).abs() * 2.0;
    // Sample factor: asymptotic approach to 1.0 with more samples
    let sample_factor = 1.0 - (-0.01 * total as f64).exp();
    let confidence = (signal_strength * sample_factor).clamp(0.0, 1.0);

    (decision, confidence)
}

// =============================================================================
// Cross-Module Deduplication (Issue #429)
// =============================================================================

/// Signature for identifying duplicate candidates across modules.
///
/// Two candidates are considered duplicates if they share the same
/// source neuron, target neuron, and candidate type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CandidateSignature {
    /// UUID of the source neuron.
    pub source_uuid: String,
    /// UUID of the target neuron.
    pub target_uuid: String,
    /// Type of candidate (e.g., "addSynapse", "removeSynapse").
    pub candidate_type: String,
}

/// Cross-module deduplicator for candidate discovery (Issue #429).
///
/// Tracks previously generated candidates by signature so that multiple
/// discovery modules can avoid producing redundant candidates. When a
/// module generates a candidate with a signature already registered,
/// it can skip or deprioritise that candidate.
pub struct CrossModuleDeduplicator {
    seen: std::collections::HashSet<CandidateSignature>,
}

impl CrossModuleDeduplicator {
    /// Create a new empty deduplicator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            seen: std::collections::HashSet::new(),
        }
    }

    /// Check whether a candidate signature has been seen before.
    ///
    /// Returns `true` if the candidate is novel (not seen), `false` if duplicate.
    #[must_use]
    pub fn is_novel(&self, sig: &CandidateSignature) -> bool {
        !self.seen.contains(sig)
    }

    /// Register a candidate signature as seen.
    ///
    /// Future calls to [`is_novel`](Self::is_novel) with the same signature
    /// will return `false`.
    pub fn register(&mut self, sig: &CandidateSignature) {
        self.seen.insert(sig.clone());
    }

    /// Get the number of unique registered signatures.
    #[must_use]
    pub fn registered_count(&self) -> usize {
        self.seen.len()
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
