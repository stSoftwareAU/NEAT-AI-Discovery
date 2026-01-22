//! Sample data structures for NEAT-AI Discovery GPU pipeline.
//!
//! This module contains the core sample types and GPU-compatible data formats used
//! throughout the analysis pipeline. These structures are fundamental to the GPU
//! computation workflow:
//!
//! 1. `HelpfulSample` - Core evaluation unit passed between CPU and GPU
//! 2. GPU structs (`*Contribution`, `*Uniforms`) - Tightly coupled with shader code
//! 3. Statistics types - Aggregate GPU computation results
//!
//! The `#[repr(C)]` attribute and bytemuck derives are critical for GPU buffer
//! compatibility.

use crate::types::DiscoverRecord;
use bytemuck::{Pod, Zeroable};

/// Small epsilon value to prevent division by zero.
pub const EPSILON: f32 = 1e-8;

/// Default threshold for treating an add-synapse candidate as an effective bias change.
///
/// Issue #178 (7-Jan-2026): When the source activation range is ~0, adding a synapse
/// only contributes a near-constant offset to the target. This is better represented as
/// a `setBias` coordinated-structural operation than paying complexity cost for a new edge.
///
/// The heuristic is based on the *range* of the contribution:
///   effect_range approx |weight| * (max_activation - min_activation)
///
/// If `effect_range <= threshold`, we fold the synapse into `setBias`.
pub const DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD: f32 = 1e-7;

// =============================================================================
// Core Sample Types
// =============================================================================

/// Core evaluation sample unit.
///
/// Represents a single sample for evaluating potential synapse/neuron candidates.
/// For accurate HARD_TANH modelling, we need the target's pre-activation value
/// to properly simulate clamping behaviour. When `target_value` is `Some`, we can
/// compute the actual effect of adding a contribution rather than using the linear
/// approximation.
#[derive(Debug, Clone, Copy, Default)]
pub struct HelpfulSample {
    /// Source neuron's activation (what we're considering adding a connection FROM)
    pub activation: f32,
    /// Target neuron's average error (expected - actual output)
    pub avg_error: f32,
    /// Target neuron's pre-activation value (input sum before squash function).
    /// Used for accurate HARD_TANH/clamping calculations. None for GPU-matched samples.
    pub target_value: Option<f32>,
    /// Target neuron's post-activation output (after squash function).
    /// Note: avg_error is in VALUE domain, so expected = squash(target_value + avg_error)
    pub target_activation: Option<f32>,
}

/// Statistics computed from neuron error and activation samples.
#[derive(Debug, Clone)]
pub struct NeuronStats {
    pub mean_error: f32,
    pub error_variance: f32,
    pub mean_activation: f32,
    pub activation_variance: f32,
    pub error_spike_count: u32,
    pub activation_spike_count: u32,
    pub activation_min: f32,
    pub activation_max: f32,
}

impl NeuronStats {
    /// Compute statistics from a slice of discovery records (for target neurons).
    pub fn from_records(records: &[DiscoverRecord]) -> Option<Self> {
        if records.is_empty() {
            return None;
        }

        let mut samples = Vec::new();
        for record in records {
            if record.errors.is_empty() {
                continue;
            }
            // Compute average error for this record
            let mut error_sum = 0.0;
            let mut error_count = 0;
            for &err in &record.errors {
                if err.is_finite() {
                    error_sum += err;
                    error_count += 1;
                }
            }
            if error_count > 0 && record.activation.is_finite() {
                let avg_error = error_sum / error_count as f32;
                samples.push(HelpfulSample {
                    activation: record.activation,
                    avg_error,
                    target_value: record.value,
                    target_activation: Some(record.activation),
                });
            }
        }

        Self::from_samples(&samples)
    }

    /// Compute statistics from a slice of samples.
    pub fn from_samples(samples: &[HelpfulSample]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut error_sum = 0.0;
        let mut error_sq_sum = 0.0;
        let mut activation_sum = 0.0;
        let mut activation_sq_sum = 0.0;
        let mut error_spike_count = 0u32;
        let mut activation_spike_count = 0u32;
        let mut activation_min = f32::INFINITY;
        let mut activation_max = f32::NEG_INFINITY;
        let mut valid_count = 0usize;

        // Spike thresholds: 2 standard deviations (we'll approximate with mean + 2*mean for now)
        // We'll compute proper thresholds after we have the mean
        let mut error_abs_sum = 0.0;
        let mut activation_abs_sum = 0.0;

        for sample in samples {
            if !sample.avg_error.is_finite() || !sample.activation.is_finite() {
                continue;
            }
            valid_count += 1;
            let error_abs = sample.avg_error.abs();
            let activation_abs = sample.activation.abs();

            error_sum += sample.avg_error;
            error_sq_sum += sample.avg_error * sample.avg_error;
            error_abs_sum += error_abs;

            activation_sum += sample.activation;
            activation_sq_sum += sample.activation * sample.activation;
            activation_abs_sum += activation_abs;

            if activation_min > sample.activation {
                activation_min = sample.activation;
            }
            if activation_max < sample.activation {
                activation_max = sample.activation;
            }
        }

        if valid_count == 0 {
            return None;
        }

        let count_f = valid_count as f32;
        let mean_error = error_sum / count_f;
        let mean_activation = activation_sum / count_f;
        let mean_error_abs = error_abs_sum / count_f;
        let mean_activation_abs = activation_abs_sum / count_f;

        // Compute variance using E[X^2] - E[X]^2
        let error_variance = (error_sq_sum / count_f) - (mean_error * mean_error);
        let activation_variance =
            (activation_sq_sum / count_f) - (mean_activation * mean_activation);

        // Spike detection: count samples where error/activation exceeds 2x the mean absolute value
        // This is a simple heuristic; more sophisticated methods could use actual std dev
        let error_spike_threshold = mean_error_abs * 2.0;
        let activation_spike_threshold = mean_activation_abs * 2.0;

        for sample in samples {
            if !sample.avg_error.is_finite() || !sample.activation.is_finite() {
                continue;
            }
            if sample.avg_error.abs() > error_spike_threshold {
                error_spike_count += 1;
            }
            if sample.activation.abs() > activation_spike_threshold {
                activation_spike_count += 1;
            }
        }

        Some(Self {
            mean_error,
            error_variance: error_variance.max(0.0), // Variance should be non-negative
            mean_activation,
            activation_variance: activation_variance.max(0.0),
            error_spike_count,
            activation_spike_count,
            activation_min: if activation_min.is_finite() {
                activation_min
            } else {
                0.0
            },
            activation_max: if activation_max.is_finite() {
                activation_max
            } else {
                0.0
            },
        })
    }

    /// Convert to the JSON-serializable format.
    pub fn to_json(&self) -> crate::NeuronStatsJson {
        crate::NeuronStatsJson {
            mean_error: self.mean_error,
            error_variance: self.error_variance,
            mean_activation: self.mean_activation,
            activation_variance: self.activation_variance,
            error_spike_count: self.error_spike_count,
            activation_spike_count: self.activation_spike_count,
            activation_min: self.activation_min,
            activation_max: self.activation_max,
        }
    }
}

// =============================================================================
// GPU-Compatible Formats
// =============================================================================

/// GPU buffer format for helpful samples.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuHelpfulSample {
    pub activation: f32,
    pub avg_error: f32,
}

impl From<HelpfulSample> for GpuHelpfulSample {
    fn from(value: HelpfulSample) -> Self {
        Self {
            activation: value.activation,
            avg_error: value.avg_error,
        }
    }
}

/// GPU contribution data for helpful synapse evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HelpfulContribution {
    pub positive_flag: u32,
    pub negative_flag: u32,
    pub positive_improvement: f32,
    pub negative_improvement: f32,
    pub positive_activation: f32,
    pub negative_activation: f32,
    pub error_squared: f32,
    pub activation_squared: f32,
    pub error_activation: f32,
    pub pad0: f32,
    pub pad1: f32,
    pub pad2: f32,
}

/// GPU shader uniforms for helpful synapse evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HelpfulUniforms {
    pub length: u32,
    pub pad0: u32,
    pub epsilon: f32,
    pub pad1: f32,
}

/// GPU contribution data for harmful synapse evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HarmfulContribution {
    pub harmful_flag: u32,
    pub helpful_flag: u32,
    pub error_magnitude: f32,
    pub pad0: f32,
}

/// GPU shader uniforms for harmful synapse evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HarmfulUniforms {
    pub length: u32,
    pub pad0: u32,
    pub epsilon: f32,
    pub weight: f32,
}

/// GPU contribution data for ReLU activation evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ReluContribution {
    pub positive_activation_sq: f32,
    pub positive_error_activation: f32,
    pub positive_count: u32,
    pub negative_activation_sq: f32,
    pub negative_error_activation: f32,
    pub negative_count: u32,
    pub error_sq: f32,
    pub pad0: f32,
    pub pad1: u32,
    pub pad2: u32,
}

/// GPU shader uniforms for ReLU activation evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ReluUniforms {
    pub length: u32,
    pub threshold: f32,
    pub epsilon: f32,
    pub pad0: f32,
}

/// GPU result data for bias optimisation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BiasResult {
    pub bias_value: f32,
    pub error_reduction: f32,
    pub valid_sample_count: u32,
    pub pad0: u32,
}

impl BiasResult {
    /// Create a zeroed result.
    pub fn zeroed() -> Self {
        Self {
            bias_value: 0.0,
            error_reduction: 0.0,
            valid_sample_count: 0,
            pad0: 0,
        }
    }
}

/// GPU shader uniforms for bias optimisation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BiasUniforms {
    pub sample_count: u32,
    pub bias_count: u32,
    pub incoming_weight: f32,
    pub outgoing_weight: f32,
    pub activation_type: u32,
    pub epsilon: f32,
    pub min_sample_count: u32,
    pub pad0: u32,
}

/// GPU output data for activation function evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ActivationOutput {
    pub output: f32,
    pub output_sq: f32,
    pub error_output: f32,
    pub valid: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// GPU shader uniforms for activation function evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ActivationUniforms {
    pub sample_count: u32,
    pub orientation: f32,
    pub scale: f32,
    pub activation_type: u32,
    pub epsilon: f32,
    pub pad0: f32,
    pub pad1: f32,
}

/// GPU shader uniforms for workgroup reduction (Issue #218).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ReductionUniforms {
    /// Total number of contributions to reduce
    pub contribution_count: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

// =============================================================================
// Statistics Results
// =============================================================================

/// Computed statistics for helpful synapse evaluation.
#[derive(Debug, Clone, Default)]
pub struct HelpfulStats {
    pub positive_count: u32,
    pub negative_count: u32,
    pub positive_improvement_sum: f32,
    pub negative_improvement_sum: f32,
    pub positive_activation_sum: f32,
    pub negative_activation_sum: f32,
    pub error_sq_sum: f32,
    pub activation_sq_sum: f32,
    pub error_activation_sum: f32,
    /// Total samples evaluated (for early termination tracking - Issue #219).
    /// When using early termination, this may be less than the total available samples.
    pub samples_evaluated: u32,
    /// Whether evaluation was terminated early (Issue #219).
    pub early_terminated: bool,
}

impl HelpfulStats {
    /// Get the total count of samples evaluated.
    #[must_use]
    pub fn total_count(&self) -> u32 {
        self.positive_count + self.negative_count
    }

    /// Get the improvement ratio (positive_count / total_count).
    ///
    /// Returns 0.5 (neutral) if no samples have been evaluated.
    #[must_use]
    pub fn improvement_ratio(&self) -> f64 {
        let total = self.total_count();
        if total == 0 {
            return 0.5;
        }
        f64::from(self.positive_count) / f64::from(total)
    }

    /// Check if this candidate appears strongly beneficial based on current statistics.
    ///
    /// This is a quick heuristic for deciding whether to continue evaluation.
    /// A candidate with >70% positive samples is considered strongly beneficial.
    #[must_use]
    pub fn is_strongly_beneficial(&self) -> bool {
        let total = self.total_count();
        if total < 30 {
            return false;
        }
        self.improvement_ratio() > 0.7
    }

    /// Check if this candidate appears strongly harmful based on current statistics.
    ///
    /// This is a quick heuristic for deciding whether to continue evaluation.
    /// A candidate with <30% positive samples is considered strongly harmful.
    #[must_use]
    pub fn is_strongly_harmful(&self) -> bool {
        let total = self.total_count();
        if total < 30 {
            return false;
        }
        self.improvement_ratio() < 0.3
    }

    /// Merge another stats instance into this one (for incremental batch processing).
    pub fn merge(&mut self, other: &Self) {
        self.positive_count += other.positive_count;
        self.negative_count += other.negative_count;
        self.positive_improvement_sum += other.positive_improvement_sum;
        self.negative_improvement_sum += other.negative_improvement_sum;
        self.positive_activation_sum += other.positive_activation_sum;
        self.negative_activation_sum += other.negative_activation_sum;
        self.error_sq_sum += other.error_sq_sum;
        self.activation_sq_sum += other.activation_sq_sum;
        self.error_activation_sum += other.error_activation_sum;
        self.samples_evaluated += other.samples_evaluated;
        // Don't merge early_terminated - let caller decide
    }
}

/// ReLU split direction for neuron candidates.
#[derive(Clone, Copy)]
pub enum ReluOrientation {
    Positive,
    Negative,
}

/// ReLU evaluation results for neuron candidates.
pub struct ReluStats {
    pub orientation: ReluOrientation,
    pub samples: Vec<(f32, f32)>,
    pub activation_sq_sum: f32,
    pub error_activation_sum: f32,
}

impl ReluStats {
    /// Create new empty stats for the given orientation.
    pub fn new(orientation: ReluOrientation) -> Self {
        Self {
            orientation,
            samples: Vec::new(),
            activation_sq_sum: 0.0,
            error_activation_sum: 0.0,
        }
    }

    /// Evaluate this orientation and return a candidate if it passes the threshold.
    ///
    /// This method computes the optimal outgoing weight and expected improvement for
    /// a ReLU candidate, returning a candidate JSON if it exceeds the threshold.
    ///
    /// # Arguments
    /// * `source_uuid` - UUID of the source neuron
    /// * `target_uuid` - UUID of the target neuron
    /// * `threshold` - Minimum improvement required
    /// * `total_baseline_error_sq` - Total squared error baseline for normalisation
    /// * `original_samples` - Original samples for statistics computation
    ///
    /// **Extracted from implementation.rs as part of Issue #275**
    pub fn evaluate(
        &self,
        source_uuid: &str,
        target_uuid: &str,
        threshold: f32,
        total_baseline_error_sq: f32,
        original_samples: &[HelpfulSample],
    ) -> Option<crate::CandidateNeuronJson> {
        use crate::analysis::weights::MAX_OUTGOING_WEIGHT;

        const MIN_NEURON_SAMPLE_COUNT: usize = 10;

        let sample_count = self.samples.len();
        if sample_count < MIN_NEURON_SAMPLE_COUNT || self.activation_sq_sum <= EPSILON {
            return None;
        }

        let mut outgoing_weight = self.error_activation_sum / (self.activation_sq_sum + EPSILON);
        if !outgoing_weight.is_finite() || outgoing_weight.abs() <= EPSILON {
            return None;
        }
        outgoing_weight = outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        let mut improved_count = 0u32;
        for (relu_activation, error) in &self.samples {
            let new_error = error - outgoing_weight * relu_activation;
            if new_error.abs() + EPSILON < error.abs() {
                improved_count += 1;
            }
        }

        // Calculate improvement based on magnitude (reduction in squared error)
        // improvement = baseline_sq - new_sq
        // = 2*w*sum(ea) - w^2*sum(aa)
        let improvement_magnitude = 2.0 * outgoing_weight * self.error_activation_sum
            - outgoing_weight * outgoing_weight * self.activation_sq_sum;

        // Normalise by total baseline error of ALL samples (not just active ones)
        let expected_improvement = if total_baseline_error_sq > EPSILON {
            let result = improvement_magnitude / total_baseline_error_sq;
            if result.is_finite() {
                result
            } else {
                0.0
            }
        } else {
            0.0
        };

        if expected_improvement <= threshold {
            return None;
        }

        let incoming_weight = match self.orientation {
            ReluOrientation::Positive => 1.0,
            ReluOrientation::Negative => -1.0,
        };

        // For split-error ReLU evaluation, use bias=0.
        // The whole point of split-error is that the ReLU should fire for ONE subset
        // (positive or negative error samples) but NOT the other.
        // Optimising bias on the subset alone can find a large positive bias that makes
        // the ReLU fire for ALL samples, defeating the split-error approach.
        // With bias=0, the ReLU naturally fires only when source activation > 0.
        let optimal_bias = 0.0;

        let target_stats = NeuronStats::from_samples(original_samples).map(|s| s.to_json());
        let total_count = self.samples.len() as u32;

        // Issue #128: Use creature-level metrics instead of neuron-level percentage.
        // target_neuron_impact will be updated during impact discounting.
        Some(crate::CandidateNeuronJson {
            source_neuron_uuid: source_uuid.to_string(),
            target_neuron_uuid: target_uuid.to_string(),
            source_neuron_index: None, // Set during impact discounting
            target_neuron_index: None, // Set during impact discounting
            incoming_weight,
            outgoing_weight,
            squash: "ReLU".to_string(),
            bias: optimal_bias,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: expected_improvement,
            expected_creature_score_gain: expected_improvement,
            improved_count,
            total_count,
            target_neuron_stats: target_stats,
        })
    }
}

/// Computed statistics for harmful synapse evaluation.
#[derive(Default)]
pub struct HarmfulStats {
    pub harmful_count: u32,
    pub helpful_count: u32,
    pub harmful_error_sum: f32,
    /// Total samples evaluated (for early termination tracking - Issue #219).
    pub samples_evaluated: u32,
    /// Whether evaluation was terminated early (Issue #219).
    pub early_terminated: bool,
}

impl HarmfulStats {
    /// Get the total count of samples evaluated.
    #[must_use]
    pub fn total_count(&self) -> u32 {
        self.harmful_count + self.helpful_count
    }

    /// Get the harmful ratio (harmful_count / total_count).
    ///
    /// Returns 0.5 (neutral) if no samples have been evaluated.
    #[must_use]
    pub fn harmful_ratio(&self) -> f64 {
        let total = self.total_count();
        if total == 0 {
            return 0.5;
        }
        f64::from(self.harmful_count) / f64::from(total)
    }

    /// Check if this synapse appears strongly harmful based on current statistics.
    ///
    /// A synapse with >70% harmful samples is considered a good removal candidate.
    #[must_use]
    pub fn is_clearly_harmful(&self) -> bool {
        let total = self.total_count();
        if total < 30 {
            return false;
        }
        self.harmful_ratio() > 0.7
    }

    /// Merge another stats instance into this one (for incremental batch processing).
    pub fn merge(&mut self, other: &Self) {
        self.harmful_count += other.harmful_count;
        self.helpful_count += other.helpful_count;
        self.harmful_error_sum += other.harmful_error_sum;
        self.samples_evaluated += other.samples_evaluated;
        // Don't merge early_terminated - let caller decide
    }
}

// =============================================================================
// Supporting Functions
// =============================================================================

/// Compute source activation variance discount factor.
///
/// Issue #130 (v0.2.2): When a source neuron has constant or near-constant activation,
/// adding a connection from it cannot reduce error correlation - it only adds a constant
/// offset. The prediction model must discount improvements based on source variance.
///
/// **Key insight**: If source activation is constant, the new connection acts like a
/// bias change, not a meaningful signal. A constant cannot correlate with varying error.
///
/// **Production example**: input-1244 had variance 0.000000 (completely constant), yet
/// the model predicted 29.6% error reduction. Actual result was 0%.
///
/// # Returns
/// A discount factor in [0, 1]:
/// - 1.0: Source has high variance (no discount)
/// - 0.0: Source is constant (full discount -> zero improvement)
/// - Between: Proportional discount based on variance ratio
///
/// # Formula
/// `discount = min(1.0, source_std_dev / MIN_SOURCE_STD_DEV)`
///
/// Where MIN_SOURCE_STD_DEV = 0.05 (sources with std dev < 0.05 are progressively discounted)
pub fn compute_source_variance_discount(samples: &[HelpfulSample]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }

    // Minimum source standard deviation for full credit.
    // Sources with std dev below this are progressively discounted.
    // Value chosen based on production analysis: input-1064 had std dev 0.01 and caused
    // massive over-prediction. Sources should have at least 0.05 std dev for reliable correlation.
    const MIN_SOURCE_STD_DEV: f32 = 0.05;

    let mut activation_sum = 0.0f64;
    let mut activation_sq_sum = 0.0f64;
    let mut count = 0u32;

    for sample in samples {
        if sample.activation.is_finite() {
            let a = sample.activation as f64;
            activation_sum += a;
            activation_sq_sum += a * a;
            count += 1;
        }
    }

    if count < 2 {
        return 0.0;
    }

    let n = count as f64;
    let mean = activation_sum / n;
    let variance = (activation_sq_sum / n) - (mean * mean);
    let std_dev = variance.max(0.0).sqrt() as f32;

    // Linear discount: full credit at MIN_SOURCE_STD_DEV, zero at 0
    // Values above MIN_SOURCE_STD_DEV get full credit (capped at 1.0)
    (std_dev / MIN_SOURCE_STD_DEV).clamp(0.0, 1.0)
}

/// Optional threshold for folding constant/near-constant sources into `setBias`.
///
/// Controlled via `NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD`:
/// - unset / empty: enabled with default (`DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD`)
/// - `0`: disabled (never fold)
/// - `> 0`: enabled with the configured threshold
pub fn constant_source_effect_threshold_from_env() -> Option<f32> {
    use crate::analysis::utils::verbose_enabled;

    let raw = std::env::var("NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD").ok();
    let Some(raw) = raw else {
        return Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD);
    }
    match trimmed.parse::<f32>() {
        Ok(v) if v.is_finite() && v == 0.0 => None,
        Ok(v) if v.is_finite() && v > 0.0 => Some(v),
        _ => {
            if verbose_enabled() {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Ignoring invalid NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD={trimmed:?} (expected 0 or a finite number > 0)"
                );
            }
            Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD)
        }
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helpful_sample_default() {
        let sample = HelpfulSample::default();
        assert_eq!(sample.activation, 0.0);
        assert_eq!(sample.avg_error, 0.0);
        assert!(sample.target_value.is_none());
        assert!(sample.target_activation.is_none());
    }

    #[test]
    fn test_gpu_helpful_sample_from_helpful_sample() {
        let sample = HelpfulSample {
            activation: 1.5,
            avg_error: -0.3,
            target_value: Some(2.0),
            target_activation: Some(0.9),
        };
        let gpu_sample: GpuHelpfulSample = sample.into();
        assert_eq!(gpu_sample.activation, 1.5);
        assert_eq!(gpu_sample.avg_error, -0.3);
    }

    #[test]
    fn test_bias_result_zeroed() {
        let result = BiasResult::zeroed();
        assert_eq!(result.bias_value, 0.0);
        assert_eq!(result.error_reduction, 0.0);
        assert_eq!(result.valid_sample_count, 0);
    }

    #[test]
    fn test_relu_stats_new() {
        let stats = ReluStats::new(ReluOrientation::Positive);
        assert!(stats.samples.is_empty());
        assert_eq!(stats.activation_sq_sum, 0.0);
        assert_eq!(stats.error_activation_sum, 0.0);
    }

    #[test]
    fn test_harmful_stats_default() {
        let stats = HarmfulStats::default();
        assert_eq!(stats.harmful_count, 0);
        assert_eq!(stats.helpful_count, 0);
        assert_eq!(stats.harmful_error_sum, 0.0);
    }

    #[test]
    fn test_helpful_stats_default() {
        let stats = HelpfulStats::default();
        assert_eq!(stats.positive_count, 0);
        assert_eq!(stats.negative_count, 0);
        assert_eq!(stats.error_sq_sum, 0.0);
    }

    #[test]
    fn test_compute_source_variance_discount_empty() {
        let discount = compute_source_variance_discount(&[]);
        assert_eq!(discount, 0.0);
    }

    #[test]
    fn test_compute_source_variance_discount_single() {
        let samples = vec![HelpfulSample {
            activation: 1.0,
            avg_error: 0.0,
            target_value: None,
            target_activation: None,
        }];
        let discount = compute_source_variance_discount(&samples);
        assert_eq!(discount, 0.0);
    }

    #[test]
    fn test_compute_source_variance_discount_constant() {
        // All samples have the same activation - should be fully discounted
        let samples = vec![
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
        ];
        let discount = compute_source_variance_discount(&samples);
        assert!(
            discount < 0.01,
            "Constant source should be heavily discounted"
        );
    }

    #[test]
    fn test_compute_source_variance_discount_high_variance() {
        // Samples with high variance should get full credit
        let samples = vec![
            HelpfulSample {
                activation: -1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.0,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
        ];
        let discount = compute_source_variance_discount(&samples);
        assert!(
            discount > 0.9,
            "High variance source should get full credit"
        );
    }

    #[test]
    fn test_neuron_stats_from_samples_empty() {
        let result = NeuronStats::from_samples(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_neuron_stats_from_samples_valid() {
        let samples = vec![
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 2.0,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 3.0,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
        ];
        let stats = NeuronStats::from_samples(&samples).expect("Should compute stats");
        assert!((stats.mean_activation - 2.0).abs() < 0.001);
        assert!(stats.activation_variance > 0.0);
        assert_eq!(stats.activation_min, 1.0);
        assert_eq!(stats.activation_max, 3.0);
    }

    #[test]
    fn test_neuron_stats_to_json() {
        let stats = NeuronStats {
            mean_error: 0.1,
            error_variance: 0.01,
            mean_activation: 0.5,
            activation_variance: 0.25,
            error_spike_count: 2,
            activation_spike_count: 1,
            activation_min: -1.0,
            activation_max: 1.0,
        };
        let json = stats.to_json();
        assert_eq!(json.mean_error, 0.1);
        assert_eq!(json.error_variance, 0.01);
        assert_eq!(json.mean_activation, 0.5);
        assert_eq!(json.activation_variance, 0.25);
        assert_eq!(json.error_spike_count, 2);
        assert_eq!(json.activation_spike_count, 1);
        assert_eq!(json.activation_min, -1.0);
        assert_eq!(json.activation_max, 1.0);
    }

    #[test]
    fn test_gpu_structs_are_pod() {
        // These tests verify that GPU structs can be used with bytemuck
        // by checking they implement Pod and Zeroable
        let _ = bytemuck::bytes_of(&GpuHelpfulSample {
            activation: 0.0,
            avg_error: 0.0,
        });
        let _ = bytemuck::bytes_of(&HelpfulContribution {
            positive_flag: 0,
            negative_flag: 0,
            positive_improvement: 0.0,
            negative_improvement: 0.0,
            positive_activation: 0.0,
            negative_activation: 0.0,
            error_squared: 0.0,
            activation_squared: 0.0,
            error_activation: 0.0,
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        });
        let _ = bytemuck::bytes_of(&HelpfulUniforms {
            length: 0,
            pad0: 0,
            epsilon: 0.0,
            pad1: 0.0,
        });
        let _ = bytemuck::bytes_of(&HarmfulContribution {
            harmful_flag: 0,
            helpful_flag: 0,
            error_magnitude: 0.0,
            pad0: 0.0,
        });
        let _ = bytemuck::bytes_of(&HarmfulUniforms {
            length: 0,
            pad0: 0,
            epsilon: 0.0,
            weight: 0.0,
        });
        let _ = bytemuck::bytes_of(&ReluContribution {
            positive_activation_sq: 0.0,
            positive_error_activation: 0.0,
            positive_count: 0,
            negative_activation_sq: 0.0,
            negative_error_activation: 0.0,
            negative_count: 0,
            error_sq: 0.0,
            pad0: 0.0,
            pad1: 0,
            pad2: 0,
        });
        let _ = bytemuck::bytes_of(&ReluUniforms {
            length: 0,
            threshold: 0.0,
            epsilon: 0.0,
            pad0: 0.0,
        });
        let _ = bytemuck::bytes_of(&BiasResult::zeroed());
        let _ = bytemuck::bytes_of(&BiasUniforms {
            sample_count: 0,
            bias_count: 0,
            incoming_weight: 0.0,
            outgoing_weight: 0.0,
            activation_type: 0,
            epsilon: 0.0,
            min_sample_count: 0,
            pad0: 0,
        });
        let _ = bytemuck::bytes_of(&ActivationOutput {
            output: 0.0,
            output_sq: 0.0,
            error_output: 0.0,
            valid: 0,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        });
        let _ = bytemuck::bytes_of(&ActivationUniforms {
            sample_count: 0,
            orientation: 0.0,
            scale: 0.0,
            activation_type: 0,
            epsilon: 0.0,
            pad0: 0.0,
            pad1: 0.0,
        });
    }
}
