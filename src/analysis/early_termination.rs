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
// Issue #429: Hierarchical Candidate Pre-Filtering
// =============================================================================

/// Configuration for candidate pre-filtering.
///
/// Controls thresholds for the quick statistical pre-filter that runs
/// before detailed GPU analysis.
#[derive(Debug, Clone)]
pub struct CandidatePreFilterConfig {
    /// Minimum source activation standard deviation to consider a source useful.
    /// Sources below this are constant-ish and unlikely to yield good candidates.
    pub min_source_std_dev: f32,
    /// Minimum sample count required for reliable analysis.
    pub min_sample_count: usize,
    /// Minimum absolute Pearson correlation between source activation and target error.
    /// Below this, the source has no predictive relationship with the error.
    pub min_error_correlation: f32,
}

impl Default for CandidatePreFilterConfig {
    fn default() -> Self {
        Self {
            min_source_std_dev: crate::analysis::constants::MIN_SOURCE_STD_DEV,
            min_sample_count: crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT,
            min_error_correlation: 0.05,
        }
    }
}

/// Statistics tracked by the pre-filter for observability.
#[derive(Debug, Clone, Default)]
pub struct PreFilterStatistics {
    /// Total candidates checked.
    pub total_checked: usize,
    /// Candidates that passed the pre-filter.
    pub total_passed: usize,
    /// Candidates filtered out.
    pub total_filtered: usize,
}

/// Quick hierarchical pre-filter for candidate generation (Issue #429).
///
/// Applies cheap statistical checks before expensive GPU analysis to skip
/// candidates that are unlikely to be beneficial. Checks are ordered from
/// cheapest to most expensive:
///
/// 1. **Sample count** — skip if too few samples for reliable statistics
/// 2. **Source variance** — skip constant-ish sources (near-zero std dev)
/// 3. **Error correlation** — skip sources with no predictive relationship
#[derive(Debug, Clone)]
pub struct CandidatePreFilter {
    config: CandidatePreFilterConfig,
    stats: PreFilterStatistics,
}

impl CandidatePreFilter {
    /// Create a new pre-filter with the given configuration.
    #[must_use]
    pub fn new(config: CandidatePreFilterConfig) -> Self {
        Self {
            config,
            stats: PreFilterStatistics::default(),
        }
    }

    /// Check if the sample count is sufficient for analysis.
    #[must_use]
    pub fn passes_sample_count_check(&self, sample_count: usize) -> bool {
        sample_count >= self.config.min_sample_count
    }

    /// Check if the source activation variance is sufficient.
    ///
    /// Constant or near-constant sources cannot predict error changes.
    #[must_use]
    pub fn passes_variance_check(&self, activations: &[f32]) -> bool {
        if activations.len() < 2 {
            return false;
        }
        let n = activations.len() as f32;
        let mean = activations.iter().sum::<f32>() / n;
        let variance = activations.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();
        std_dev >= self.config.min_source_std_dev
    }

    /// Check if source activations correlate with target errors.
    ///
    /// Uses Pearson correlation coefficient. Sources with near-zero correlation
    /// cannot predict error reduction.
    #[must_use]
    pub fn passes_error_correlation_check(&self, activations: &[f32], errors: &[f32]) -> bool {
        let n = activations.len().min(errors.len());
        if n < 2 {
            return false;
        }

        let n_f = n as f32;
        let mean_a = activations[..n].iter().sum::<f32>() / n_f;
        let mean_e = errors[..n].iter().sum::<f32>() / n_f;

        let mut cov = 0.0_f32;
        let mut var_a = 0.0_f32;
        let mut var_e = 0.0_f32;
        for i in 0..n {
            let da = activations[i] - mean_a;
            let de = errors[i] - mean_e;
            cov += da * de;
            var_a += da * da;
            var_e += de * de;
        }

        let denom = (var_a * var_e).sqrt();
        if denom < f32::EPSILON {
            return false;
        }

        let correlation = (cov / denom).abs();
        correlation >= self.config.min_error_correlation
    }

    /// Run all pre-filter checks in order (cheapest to most expensive).
    ///
    /// Returns `true` if the candidate should proceed to detailed analysis.
    #[must_use]
    pub fn should_analyse(&self, activations: &[f32], errors: &[f32]) -> bool {
        if !self.passes_sample_count_check(activations.len()) {
            return false;
        }
        if !self.passes_variance_check(activations) {
            return false;
        }
        if !self.passes_error_correlation_check(activations, errors) {
            return false;
        }
        true
    }

    /// Record the result of a check for statistics tracking.
    pub fn record_check(&mut self, passed: bool) {
        self.stats.total_checked += 1;
        if passed {
            self.stats.total_passed += 1;
        } else {
            self.stats.total_filtered += 1;
        }
    }

    /// Get current pre-filter statistics.
    #[must_use]
    pub fn statistics(&self) -> &PreFilterStatistics {
        &self.stats
    }
}

// =============================================================================
// Issue #429: Budget-Aware Prioritisation
// =============================================================================

/// Tracks candidate generation budget to stop producing low-priority
/// candidates when the budget is exhausted (Issue #429).
///
/// The budget represents the maximum number of candidates to generate.
/// As the budget is consumed, low-priority candidates are progressively
/// skipped to focus computation on high-value candidates.
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    /// Total budget (max candidates).
    total: usize,
    /// Amount consumed so far.
    consumed: usize,
}

impl BudgetTracker {
    /// Create a new budget tracker with the given total budget.
    #[must_use]
    pub fn new(total: usize) -> Self {
        Self { total, consumed: 0 }
    }

    /// Check if any budget remains.
    #[must_use]
    pub fn has_budget(&self) -> bool {
        self.consumed < self.total
    }

    /// Get the remaining budget.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.total.saturating_sub(self.consumed)
    }

    /// Consume some of the budget.
    pub fn consume(&mut self, amount: usize) {
        self.consumed = self.consumed.saturating_add(amount).min(self.total);
    }

    /// Check if a low-priority candidate should be skipped based on
    /// current budget utilisation.
    ///
    /// As budget fills up, the priority threshold for accepting new
    /// candidates increases. When > 80% consumed, only candidates with
    /// expected gain above the dynamic threshold are accepted.
    ///
    /// # Arguments
    /// * `expected_gain` - The expected score improvement from this candidate.
    #[must_use]
    pub fn should_skip_low_priority(&self, expected_gain: f32) -> bool {
        if self.total == 0 {
            return true;
        }
        let utilisation = self.consumed as f32 / self.total as f32;

        // Below 80% utilisation, accept everything
        if utilisation < 0.8 {
            return false;
        }

        // Above 80%, apply a rising threshold: at 80% require gain > 0.01,
        // at 100% require gain > 0.1. Linear interpolation between.
        let threshold = 0.01 + (utilisation - 0.8) * (0.09 / 0.2);
        expected_gain < threshold
    }
}

// =============================================================================
// Issue #429: Incremental Confidence Checking
// =============================================================================

/// Checks whether enough high-confidence candidates have been generated
/// to justify stopping early (Issue #429).
///
/// When the top candidates already exceed the confidence threshold,
/// continuing to generate more candidates yields diminishing returns.
#[derive(Debug, Clone)]
pub struct IncrementalConfidenceChecker {
    /// Confidence threshold — once the top-K candidates are all above this,
    /// we can stop generating.
    threshold: f32,
    /// Confidence values of candidates generated so far (sorted desc on query).
    confidences: Vec<f32>,
    /// Minimum number of candidates before we consider stopping.
    min_candidates: usize,
}

impl IncrementalConfidenceChecker {
    /// Create a new checker with the given confidence threshold.
    ///
    /// # Arguments
    /// * `threshold` - Confidence level (0.0–1.0) at which candidates are considered sufficient.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            confidences: Vec::new(),
            min_candidates: 3,
        }
    }

    /// Record a newly generated candidate's confidence.
    pub fn add_candidate(&mut self, confidence: f32) {
        self.confidences.push(confidence);
    }

    /// Check if we should stop generating more candidates.
    ///
    /// Returns `true` when at least `min_candidates` have been generated
    /// and the median confidence exceeds the threshold.
    #[must_use]
    pub fn should_stop_generating(&self) -> bool {
        if self.confidences.len() < self.min_candidates {
            return false;
        }

        let mut sorted = self.confidences.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Check the median of the top-K candidates
        let mid = sorted.len() / 2;
        sorted[mid] >= self.threshold
    }

    /// Get the highest confidence seen so far.
    #[must_use]
    pub fn best_confidence(&self) -> f32 {
        self.confidences
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

// =============================================================================
// Issue #429: Cross-Module Deduplication
// =============================================================================

/// Statistics for deduplication observability.
#[derive(Debug, Clone, Default)]
pub struct DeduplicationStatistics {
    /// Total candidates registered.
    pub total_registered: usize,
    /// Total duplicate checks performed.
    pub total_duplicate_checks: usize,
}

/// Deduplicates candidates across multiple discovery modules (Issue #429).
///
/// Discovery modules run independently and may produce overlapping candidates
/// for the same source-target pair with similar expected gains. This deduplicator
/// tracks generated candidates and allows checking for near-duplicates before
/// adding them to the result set.
///
/// Two candidates are considered duplicates when they share the same source
/// and target neuron UUIDs and their expected gains are within a similarity
/// tolerance.
#[derive(Debug, Clone)]
pub struct CrossModuleDeduplicator {
    /// Registered candidates: (source_uuid, target_uuid) -> list of gains.
    registered: std::collections::HashMap<(String, String), Vec<f32>>,
    /// Tolerance for gain similarity (relative).
    gain_tolerance: f32,
    stats: DeduplicationStatistics,
}

impl CrossModuleDeduplicator {
    /// Create a new deduplicator with default gain tolerance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registered: std::collections::HashMap::new(),
            gain_tolerance: 0.1, // 10% relative tolerance
            stats: DeduplicationStatistics::default(),
        }
    }

    /// Register a candidate as generated.
    pub fn register(&mut self, source_uuid: &str, target_uuid: &str, expected_gain: f32) {
        self.registered
            .entry((source_uuid.to_string(), target_uuid.to_string()))
            .or_default()
            .push(expected_gain);
        self.stats.total_registered += 1;
    }

    /// Check if a candidate is a near-duplicate of an already registered one.
    ///
    /// Returns `true` if a similar candidate (same source/target, similar gain)
    /// has already been registered.
    #[must_use]
    pub fn is_duplicate(
        &mut self,
        source_uuid: &str,
        target_uuid: &str,
        expected_gain: f32,
    ) -> bool {
        self.stats.total_duplicate_checks += 1;

        let key = (source_uuid.to_string(), target_uuid.to_string());
        let Some(gains) = self.registered.get(&key) else {
            return false;
        };

        let abs_gain = expected_gain.abs().max(f32::EPSILON);
        gains.iter().any(|&existing| {
            let abs_existing = existing.abs().max(f32::EPSILON);
            let diff = (abs_gain - abs_existing).abs();
            let max_val = abs_gain.max(abs_existing);
            diff / max_val <= self.gain_tolerance
        })
    }

    /// Register if unique, returning whether the candidate was new.
    ///
    /// Combines `is_duplicate` and `register` — registers only if not a duplicate.
    pub fn register_if_unique(
        &mut self,
        source_uuid: &str,
        target_uuid: &str,
        expected_gain: f32,
    ) -> bool {
        if self.is_duplicate(source_uuid, target_uuid, expected_gain) {
            return false;
        }
        self.register(source_uuid, target_uuid, expected_gain);
        true
    }

    /// Get current deduplication statistics.
    #[must_use]
    pub fn statistics(&self) -> &DeduplicationStatistics {
        &self.stats
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
