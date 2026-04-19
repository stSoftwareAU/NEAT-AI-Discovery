//! Statistics types for synapse and neuron evaluation results.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::{EPSILON, HelpfulSample};

// Import confidence interval calculations (Issue #194)
use crate::analysis::scoring::confidence::compute_confidence_metrics;

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
    pub fn from_records(records: &[crate::types::DiscoverRecord]) -> Option<Self> {
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

/// Computed statistics for helpful synapse evaluation.
#[derive(Debug, Clone, Copy, Default)]
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

    /// Get the improvement ratio (`positive_count` / `total_count`).
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

/// `ReLU` split direction for neuron candidates.
#[derive(Clone, Copy)]
pub enum ReluOrientation {
    Positive,
    Negative,
}

/// `ReLU` evaluation results for neuron candidates.
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
    /// a `ReLU` candidate, returning a candidate JSON if it exceeds the threshold.
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
        use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT;
        use crate::analysis::scoring::weights::MAX_OUTGOING_WEIGHT;

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
            if result.is_finite() { result } else { 0.0 }
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
        // Issue #194: Compute confidence metrics for this prediction
        let confidence_metrics = compute_confidence_metrics(
            original_samples,
            expected_improvement,
            None, // R² not available for neuron candidates
        );
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
            prediction_confidence: confidence_metrics.prediction_confidence,
            expected_score_gain_confidence_interval: confidence_metrics
                .expected_score_gain_confidence_interval,
            target_saturation_factor: None,
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

    /// Get the harmful ratio (`harmful_count` / `total_count`).
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
