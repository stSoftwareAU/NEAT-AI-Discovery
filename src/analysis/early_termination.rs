//! Early termination for candidate evaluation using Sequential Probability Ratio Test (SPRT).
//!
//! Issue #219: Implements statistical early termination for clearly beneficial/harmful candidates.
//! This allows the GPU evaluation to stop early when a candidate is clearly good or clearly bad,
//! rather than evaluating all samples.
//!
//! Issue #429: Extends early termination to candidate generation with:
//! - Hierarchical candidate filtering (quick pre-filter before detailed analysis)
//! - Incremental confidence (exit early when confidence already exceeds threshold)
//! - Cross-module deduplication (skip candidates similar to already-generated ones)
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
// Issue #429: Hierarchical Candidate Filtering
// =============================================================================

/// Result of pre-filtering candidates based on quick heuristics.
///
/// Issue #429: This provides a fast first-pass filter to skip clearly low-value
/// candidates before investing in full GPU evaluation.
#[derive(Debug, Clone)]
pub struct CandidatePreFilterResult {
    /// Indices of candidates worth full evaluation
    pub evaluate_indices: Vec<usize>,
    /// Indices of candidates rejected by pre-filter
    pub skip_indices: Vec<usize>,
    /// Reason for skipping each index (for diagnostics)
    pub skip_reasons: Vec<PreFilterSkipReason>,
}

impl CandidatePreFilterResult {
    /// Create a new empty result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            evaluate_indices: Vec::new(),
            skip_indices: Vec::new(),
            skip_reasons: Vec::new(),
        }
    }

    /// Get the total number of candidates evaluated.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.evaluate_indices.len() + self.skip_indices.len()
    }

    /// Get the skip rate (fraction of candidates skipped).
    #[must_use]
    pub fn skip_rate(&self) -> f64 {
        let total = self.total_count();
        if total == 0 {
            return 0.0;
        }
        self.skip_indices.len() as f64 / total as f64
    }
}

impl Default for CandidatePreFilterResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Reason why a candidate was skipped in pre-filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreFilterSkipReason {
    /// Source activation has insufficient variance to provide meaningful signal
    LowSourceVariance,
    /// Sample count is too low for reliable statistics
    InsufficientSamples,
    /// Error-activation correlation is too weak
    WeakCorrelation,
    /// Similar candidate already exists (deduplication)
    DuplicateCandidate,
}

/// Configuration for hierarchical pre-filtering.
#[derive(Debug, Clone)]
pub struct PreFilterConfig {
    /// Whether pre-filtering is enabled
    pub enabled: bool,
    /// Minimum source activation variance for consideration
    pub min_source_variance: f32,
    /// Minimum sample count for reliable statistics
    pub min_sample_count: usize,
    /// Minimum absolute correlation for consideration
    pub min_correlation: f32,
}

impl Default for PreFilterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_source_variance: 0.001, // Very small threshold to catch only constant sources
            min_sample_count: 10,
            min_correlation: 0.05, // Weak threshold to avoid filtering good candidates
        }
    }
}

impl PreFilterConfig {
    /// Create a disabled pre-filter config.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Create an aggressive pre-filter config (more filtering).
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            min_source_variance: 0.01,
            min_sample_count: 20,
            min_correlation: 0.1,
        }
    }
}

/// Statistics for quick pre-filtering of candidates.
///
/// These lightweight stats can be computed quickly without GPU involvement
/// to identify candidates that are unlikely to be valuable.
#[derive(Debug, Clone, Default)]
pub struct PreFilterStats {
    /// Number of samples
    pub sample_count: usize,
    /// Variance of source activation values
    pub source_variance: f32,
    /// Correlation coefficient between source activation and target error
    pub correlation: f32,
}

impl PreFilterStats {
    /// Compute pre-filter stats from samples.
    ///
    /// This is a lightweight computation suitable for quick filtering.
    #[must_use]
    pub fn from_samples(samples: &[crate::analysis::samples::HelpfulSample]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        // Compute means
        let mut activation_sum = 0.0f64;
        let mut error_sum = 0.0f64;
        let mut valid_count = 0usize;

        for s in samples {
            if s.activation.is_finite() && s.avg_error.is_finite() {
                activation_sum += s.activation as f64;
                error_sum += s.avg_error as f64;
                valid_count += 1;
            }
        }

        if valid_count < 2 {
            return Self {
                sample_count: valid_count,
                source_variance: 0.0,
                correlation: 0.0,
            };
        }

        let valid_f = valid_count as f64;
        let act_mean = activation_sum / valid_f;
        let err_mean = error_sum / valid_f;

        // Compute variance and covariance
        let mut act_var_sum = 0.0f64;
        let mut err_var_sum = 0.0f64;
        let mut covar_sum = 0.0f64;

        for s in samples {
            if s.activation.is_finite() && s.avg_error.is_finite() {
                let act_diff = s.activation as f64 - act_mean;
                let err_diff = s.avg_error as f64 - err_mean;
                act_var_sum += act_diff * act_diff;
                err_var_sum += err_diff * err_diff;
                covar_sum += act_diff * err_diff;
            }
        }

        let source_variance = (act_var_sum / valid_f) as f32;
        let err_variance = err_var_sum / valid_f;

        // Pearson correlation coefficient
        let correlation = if source_variance > 0.0 && err_variance > 0.0 {
            let denom = (act_var_sum * err_var_sum).sqrt();
            if denom > 0.0 {
                (covar_sum / denom) as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        Self {
            sample_count: valid_count,
            source_variance,
            correlation,
        }
    }

    /// Check if this candidate passes the pre-filter.
    #[must_use]
    pub fn passes_filter(&self, config: &PreFilterConfig) -> Option<PreFilterSkipReason> {
        if !config.enabled {
            return None;
        }

        if self.sample_count < config.min_sample_count {
            return Some(PreFilterSkipReason::InsufficientSamples);
        }

        if self.source_variance < config.min_source_variance {
            return Some(PreFilterSkipReason::LowSourceVariance);
        }

        if self.correlation.abs() < config.min_correlation {
            return Some(PreFilterSkipReason::WeakCorrelation);
        }

        None
    }
}

/// Pre-filter a batch of candidates based on quick heuristics.
///
/// Issue #429: This provides fast first-pass filtering to skip low-value candidates
/// before full GPU evaluation, reducing analysis time for large creatures.
///
/// # Arguments
///
/// * `samples_batch` - Vector of sample slices, one per candidate
/// * `config` - Pre-filter configuration
///
/// # Returns
///
/// A `CandidatePreFilterResult` indicating which candidates should proceed to GPU evaluation.
pub fn prefilter_candidates(
    samples_batch: &[&[crate::analysis::samples::HelpfulSample]],
    config: &PreFilterConfig,
) -> CandidatePreFilterResult {
    let mut result = CandidatePreFilterResult::new();

    if !config.enabled {
        // When disabled, all candidates proceed
        result.evaluate_indices = (0..samples_batch.len()).collect();
        return result;
    }

    for (i, samples) in samples_batch.iter().enumerate() {
        let stats = PreFilterStats::from_samples(samples);

        if let Some(reason) = stats.passes_filter(config) {
            result.skip_indices.push(i);
            result.skip_reasons.push(reason);
        } else {
            result.evaluate_indices.push(i);
        }
    }

    result
}

// =============================================================================
// Issue #429: Cross-Module Deduplication
// =============================================================================

/// Candidate signature for deduplication.
///
/// This captures the essential characteristics of a candidate for similarity comparison.
#[derive(Debug, Clone)]
pub struct CandidateSignature {
    /// Source neuron identifier
    pub source_uuid: String,
    /// Target neuron identifier
    pub target_uuid: String,
    /// Estimated weight (quantised to reduce near-duplicates)
    pub weight_bucket: i32,
    /// Expected improvement bucket (quantised)
    pub improvement_bucket: i32,
}

impl CandidateSignature {
    /// Create a signature from candidate parameters.
    ///
    /// Weights and improvements are bucketed to identify near-duplicates.
    #[must_use]
    pub fn new(source_uuid: &str, target_uuid: &str, weight: f32, improvement: f32) -> Self {
        // Bucket weight into bins of 0.1
        let weight_bucket = (weight * 10.0).round() as i32;
        // Bucket improvement into bins of 0.01
        let improvement_bucket = (improvement * 100.0).round() as i32;

        Self {
            source_uuid: source_uuid.to_string(),
            target_uuid: target_uuid.to_string(),
            weight_bucket,
            improvement_bucket,
        }
    }

    /// Check if this signature is similar to another.
    ///
    /// Two candidates are similar if they have the same source and target
    /// and similar weight/improvement values.
    #[must_use]
    pub fn is_similar_to(&self, other: &Self) -> bool {
        self.source_uuid == other.source_uuid
            && self.target_uuid == other.target_uuid
            && (self.weight_bucket - other.weight_bucket).abs() <= 1
            && (self.improvement_bucket - other.improvement_bucket).abs() <= 2
    }
}

/// Deduplication tracker for cross-module candidate filtering.
///
/// Issue #429: This tracks candidate signatures across modules to avoid
/// generating redundant similar candidates.
#[derive(Debug)]
pub struct CandidateDeduplicator {
    /// Set of already-seen signatures
    signatures: std::collections::HashSet<String>,
}

impl CandidateDeduplicator {
    /// Create a new deduplicator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            signatures: std::collections::HashSet::new(),
        }
    }

    /// Create a new deduplicator with capacity hint.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            signatures: std::collections::HashSet::with_capacity(capacity),
        }
    }

    /// Check if a candidate is a duplicate and register it if not.
    ///
    /// Returns `true` if the candidate is a duplicate (should be skipped).
    pub fn is_duplicate(&mut self, signature: &CandidateSignature) -> bool {
        let key = format!(
            "{}:{}:{}:{}",
            signature.source_uuid,
            signature.target_uuid,
            signature.weight_bucket,
            signature.improvement_bucket
        );

        if self.signatures.contains(&key) {
            return true;
        }

        // Check for similar signatures (within bucket tolerance)
        for bucket_offset_w in -1..=1 {
            for bucket_offset_i in -2..=2 {
                let similar_key = format!(
                    "{}:{}:{}:{}",
                    signature.source_uuid,
                    signature.target_uuid,
                    signature.weight_bucket + bucket_offset_w,
                    signature.improvement_bucket + bucket_offset_i
                );
                if self.signatures.contains(&similar_key) {
                    return true;
                }
            }
        }

        // Not a duplicate - register it
        self.signatures.insert(key);
        false
    }

    /// Get the number of unique candidates registered.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.signatures.len()
    }

    /// Clear all registered signatures.
    pub fn clear(&mut self) {
        self.signatures.clear();
    }
}

impl Default for CandidateDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Issue #429: Incremental Confidence Early Exit
// =============================================================================

/// Incremental confidence tracker for early exit during batch processing.
///
/// Issue #429: This tracks confidence accumulation across batches to enable
/// early exit when we've found enough high-confidence candidates.
#[derive(Debug, Clone)]
pub struct IncrementalConfidenceTracker {
    /// Target number of high-confidence candidates
    pub target_count: usize,
    /// Current count of high-confidence candidates
    pub high_confidence_count: usize,
    /// Minimum confidence threshold for "high confidence"
    pub confidence_threshold: f64,
    /// Total candidates evaluated
    pub evaluated_count: usize,
}

impl IncrementalConfidenceTracker {
    /// Create a new tracker.
    #[must_use]
    pub fn new(target_count: usize, confidence_threshold: f64) -> Self {
        Self {
            target_count,
            high_confidence_count: 0,
            confidence_threshold,
            evaluated_count: 0,
        }
    }

    /// Record a candidate evaluation.
    ///
    /// Returns `true` if we should stop early (enough high-confidence candidates found).
    pub fn record(&mut self, confidence: f64) -> bool {
        self.evaluated_count += 1;

        if confidence >= self.confidence_threshold {
            self.high_confidence_count += 1;
        }

        self.should_stop()
    }

    /// Check if we should stop early.
    #[must_use]
    pub fn should_stop(&self) -> bool {
        self.high_confidence_count >= self.target_count
    }

    /// Get the current confidence rate.
    #[must_use]
    pub fn confidence_rate(&self) -> f64 {
        if self.evaluated_count == 0 {
            return 0.0;
        }
        self.high_confidence_count as f64 / self.evaluated_count as f64
    }
}

impl Default for IncrementalConfidenceTracker {
    fn default() -> Self {
        Self::new(50, 0.7) // Default: stop when we have 50 candidates with >70% confidence
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

    // =========================================================================
    // Issue #429: Pre-filter tests
    // =========================================================================

    #[test]
    fn test_prefilter_stats_from_empty_samples() {
        let samples: Vec<crate::analysis::samples::HelpfulSample> = vec![];
        let stats = PreFilterStats::from_samples(&samples);
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.source_variance, 0.0);
        assert_eq!(stats.correlation, 0.0);
    }

    #[test]
    fn test_prefilter_stats_from_constant_source() {
        // Constant source activation should have zero variance
        let samples: Vec<crate::analysis::samples::HelpfulSample> = (0..20)
            .map(|i| crate::analysis::samples::HelpfulSample {
                activation: 1.0, // Constant
                avg_error: if i % 2 == 0 { 0.3 } else { -0.3 },
                target_value: None,
                target_activation: None,
            })
            .collect();

        let stats = PreFilterStats::from_samples(&samples);
        assert_eq!(stats.sample_count, 20);
        assert!(
            stats.source_variance < 0.001,
            "Constant source should have ~0 variance"
        );
    }

    #[test]
    fn test_prefilter_stats_from_correlated_source() {
        // Source that correlates with error should have non-zero correlation
        let samples: Vec<crate::analysis::samples::HelpfulSample> = (0..20)
            .map(|i| crate::analysis::samples::HelpfulSample {
                activation: if i % 2 == 0 { 0.8 } else { -0.8 },
                avg_error: if i % 2 == 0 { 0.3 } else { -0.3 }, // Same pattern
                target_value: None,
                target_activation: None,
            })
            .collect();

        let stats = PreFilterStats::from_samples(&samples);
        assert_eq!(stats.sample_count, 20);
        assert!(
            stats.source_variance > 0.1,
            "Varying source should have variance"
        );
        assert!(
            stats.correlation > 0.9,
            "Correlated source/error should have high correlation"
        );
    }

    #[test]
    fn test_prefilter_rejects_constant_source() {
        let samples: Vec<crate::analysis::samples::HelpfulSample> = (0..20)
            .map(|i| crate::analysis::samples::HelpfulSample {
                activation: 1.0, // Constant
                avg_error: if i % 2 == 0 { 0.3 } else { -0.3 },
                target_value: None,
                target_activation: None,
            })
            .collect();

        let stats = PreFilterStats::from_samples(&samples);
        let config = PreFilterConfig::default();

        let reason = stats.passes_filter(&config);
        assert!(reason.is_some(), "Constant source should be filtered out");
    }

    #[test]
    fn test_prefilter_accepts_good_candidate() {
        let samples: Vec<crate::analysis::samples::HelpfulSample> = (0..20)
            .map(|i| crate::analysis::samples::HelpfulSample {
                activation: if i % 2 == 0 { 0.8 } else { -0.8 },
                avg_error: if i % 2 == 0 { 0.3 } else { -0.3 },
                target_value: None,
                target_activation: None,
            })
            .collect();

        let stats = PreFilterStats::from_samples(&samples);
        let config = PreFilterConfig::default();

        let reason = stats.passes_filter(&config);
        assert!(reason.is_none(), "Good candidate should pass filter");
    }

    #[test]
    fn test_prefilter_disabled_passes_all() {
        let samples: Vec<crate::analysis::samples::HelpfulSample> = (0..5)
            .map(|_| crate::analysis::samples::HelpfulSample {
                activation: 1.0,
                avg_error: 0.0,
                target_value: None,
                target_activation: None,
            })
            .collect();

        let stats = PreFilterStats::from_samples(&samples);
        let config = PreFilterConfig::disabled();

        let reason = stats.passes_filter(&config);
        assert!(
            reason.is_none(),
            "Disabled filter should pass all candidates"
        );
    }

    #[test]
    fn test_prefilter_candidates_batch() {
        let good_samples: Vec<crate::analysis::samples::HelpfulSample> = (0..20)
            .map(|i| crate::analysis::samples::HelpfulSample {
                activation: if i % 2 == 0 { 0.8 } else { -0.8 },
                avg_error: if i % 2 == 0 { 0.3 } else { -0.3 },
                target_value: None,
                target_activation: None,
            })
            .collect();

        let bad_samples: Vec<crate::analysis::samples::HelpfulSample> = (0..20)
            .map(|i| crate::analysis::samples::HelpfulSample {
                activation: 1.0, // Constant
                avg_error: if i % 2 == 0 { 0.3 } else { -0.3 },
                target_value: None,
                target_activation: None,
            })
            .collect();

        let batches: Vec<&[crate::analysis::samples::HelpfulSample]> =
            vec![good_samples.as_slice(), bad_samples.as_slice()];

        let result = prefilter_candidates(&batches, &PreFilterConfig::default());

        assert_eq!(
            result.evaluate_indices.len(),
            1,
            "One candidate should pass"
        );
        assert_eq!(
            result.skip_indices.len(),
            1,
            "One candidate should be skipped"
        );
        assert_eq!(
            result.evaluate_indices[0], 0,
            "Good candidate at index 0 should pass"
        );
        assert_eq!(
            result.skip_indices[0], 1,
            "Bad candidate at index 1 should be skipped"
        );
    }

    // =========================================================================
    // Issue #429: Deduplication tests
    // =========================================================================

    #[test]
    fn test_candidate_signature_similarity() {
        let sig1 = CandidateSignature::new("src-1", "tgt-1", 0.5, 0.1);
        let sig2 = CandidateSignature::new("src-1", "tgt-1", 0.51, 0.11); // Very similar
        let sig3 = CandidateSignature::new("src-2", "tgt-1", 0.5, 0.1); // Different source

        assert!(
            sig1.is_similar_to(&sig2),
            "Similar candidates should be detected"
        );
        assert!(
            !sig1.is_similar_to(&sig3),
            "Different source should not be similar"
        );
    }

    #[test]
    fn test_deduplicator_tracks_duplicates() {
        let mut dedup = CandidateDeduplicator::new();

        let sig1 = CandidateSignature::new("src-1", "tgt-1", 0.5, 0.1);
        let sig2 = CandidateSignature::new("src-1", "tgt-1", 0.51, 0.11); // Near-duplicate
        let sig3 = CandidateSignature::new("src-2", "tgt-1", 0.5, 0.1); // Different

        assert!(
            !dedup.is_duplicate(&sig1),
            "First signature should not be duplicate"
        );
        assert!(
            dedup.is_duplicate(&sig2),
            "Similar signature should be duplicate"
        );
        assert!(
            !dedup.is_duplicate(&sig3),
            "Different signature should not be duplicate"
        );

        assert_eq!(dedup.unique_count(), 2, "Should have 2 unique signatures");
    }

    #[test]
    fn test_deduplicator_clear() {
        let mut dedup = CandidateDeduplicator::new();
        let sig = CandidateSignature::new("src-1", "tgt-1", 0.5, 0.1);

        dedup.is_duplicate(&sig);
        assert_eq!(dedup.unique_count(), 1);

        dedup.clear();
        assert_eq!(dedup.unique_count(), 0);

        // After clear, same signature should not be duplicate
        assert!(!dedup.is_duplicate(&sig));
    }

    // =========================================================================
    // Issue #429: Incremental confidence tests
    // =========================================================================

    #[test]
    fn test_incremental_confidence_stops_at_target() {
        let mut tracker = IncrementalConfidenceTracker::new(3, 0.7);

        // Record some low-confidence candidates
        assert!(!tracker.record(0.5), "Should not stop yet");
        assert!(!tracker.record(0.6), "Should not stop yet");

        // Record high-confidence candidates
        assert!(!tracker.record(0.8), "Should not stop at 1 high-conf");
        assert!(!tracker.record(0.9), "Should not stop at 2 high-conf");
        assert!(tracker.record(0.75), "Should stop at 3 high-conf");
    }

    #[test]
    fn test_incremental_confidence_rate() {
        let mut tracker = IncrementalConfidenceTracker::new(10, 0.7);

        // 4 high confidence out of 8 total
        tracker.record(0.8);
        tracker.record(0.5);
        tracker.record(0.9);
        tracker.record(0.4);
        tracker.record(0.75);
        tracker.record(0.3);
        tracker.record(0.85);
        tracker.record(0.2);

        assert_eq!(tracker.evaluated_count, 8);
        assert_eq!(tracker.high_confidence_count, 4);
        assert!((tracker.confidence_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_incremental_confidence_default() {
        let tracker = IncrementalConfidenceTracker::default();
        assert_eq!(tracker.target_count, 50);
        assert!((tracker.confidence_threshold - 0.7).abs() < 0.001);
    }
}
