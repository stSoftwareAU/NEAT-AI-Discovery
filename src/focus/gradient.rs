//! Gradient flow analysis (Issue #206).
//!
//! Computes gradient flow statistics for neurons to identify those with high
//! learning potential. Analyses activation function derivatives, saturation
//! ratios, and dead neuron ratios.

use crate::CreatureJson;
use crate::parquet_format::read_all_records_grouped_by_neuron;
use crate::types::DiscoverRecord;
use anyhow::{Context, Result};

use std::collections::HashMap;

use super::ranking::is_selectable_type;

/// Statistics about gradient flow through a neuron.
///
/// These metrics help identify neurons with high learning potential by analysing:
/// 1. How much error signal can flow through the neuron (gradient magnitude)
/// 2. Whether the neuron is stuck in a saturated region (saturation ratio)
/// 3. Whether the neuron is effectively dead (dead ratio for ReLU-family)
///
/// # Usage in Focus Ranking
///
/// Neurons with high gradient magnitude and low saturation/dead ratios have higher
/// learning potential and should be prioritised for discovery analysis.
///
/// # Example
///
/// ```ignore
/// let stats = GradientFlowStats {
///     avg_gradient_magnitude: 0.75,  // Good gradient flow
///     saturation_ratio: 0.1,          // 10% saturated samples
///     dead_ratio: 0.0,                // No dead samples (not ReLU)
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GradientFlowStats {
    /// Average gradient magnitude across all samples.
    ///
    /// For a neuron with activation function f(x), this is the average of |f'(x)|
    /// weighted by the error magnitude. Higher values indicate the neuron can
    /// effectively propagate error signals for learning.
    pub avg_gradient_magnitude: f32,

    /// Ratio of samples where the neuron is in a saturated region (0.0 to 1.0).
    ///
    /// Saturation thresholds by activation:
    /// - TANH: |value| > 3 (gradient < 0.01)
    /// - LOGISTIC: |value| > 5 (output near 0 or 1)
    /// - HARD_TANH: |value| > 1 (clamped to ±1)
    ///
    /// High saturation ratio means the neuron cannot effectively learn because
    /// the gradient is near zero in the saturated region.
    pub saturation_ratio: f32,

    /// Ratio of samples where the neuron has zero gradient (0.0 to 1.0).
    ///
    /// Primarily relevant for ReLU-family activations:
    /// - ReLU: value < 0 (completely dead - zero output and gradient)
    /// - RELU6: value < 0 or value > 6 (dead at both ends)
    /// - SELU: value < 0 has reduced but non-zero gradient
    ///
    /// High dead ratio means the neuron cannot learn from most samples.
    /// For non-ReLU activations, this will typically be 0.0.
    pub dead_ratio: f32,
}

impl Default for GradientFlowStats {
    fn default() -> Self {
        Self {
            avg_gradient_magnitude: 1.0, // Assume full gradient for unknown activations
            saturation_ratio: 0.0,       // Assume not saturated
            dead_ratio: 0.0,             // Assume not dead
        }
    }
}

/// Saturation thresholds for various activation functions.
/// These thresholds define when the gradient becomes effectively zero.
mod saturation_thresholds {
    /// TANH: |x| > 3 means tanh'(x) < 0.01 (effectively saturated)
    pub const TANH_THRESHOLD: f32 = 3.0;

    /// LOGISTIC: |x| > 5 means sigmoid'(x) < 0.007 (effectively saturated)
    pub const LOGISTIC_THRESHOLD: f32 = 5.0;

    /// HARD_TANH: |x| >= 1 means output is clamped (gradient = 0)
    pub const HARD_TANH_THRESHOLD: f32 = 1.0;

    /// ARCTAN: |x| > 10 means arctan'(x) < 0.01
    pub const ARCTAN_THRESHOLD: f32 = 10.0;

    /// SOFTSIGN: |x| > 10 means softsign'(x) < 0.008
    pub const SOFTSIGN_THRESHOLD: f32 = 10.0;

    /// Gradient threshold below which we consider the sample "saturated"
    pub const GRADIENT_SATURATION_THRESHOLD: f32 = 0.01;
}

/// Compute the gradient (derivative) of an activation function at a given input value.
///
/// Returns (gradient, is_dead) where:
/// - gradient: The derivative f'(value)
/// - is_dead: True if this is a "dead" neuron sample (zero gradient AND zero output)
///
/// # Arguments
/// * `squash` - Uppercase squash function name
/// * `value` - Pre-activation input value
fn compute_activation_gradient(squash: &str, value: f32) -> (f32, bool) {
    match squash {
        // ReLU family - can have dead neurons
        "RELU" => {
            if value > 0.0 {
                (1.0, false)
            } else {
                (0.0, true) // Dead: zero gradient AND zero output
            }
        }
        "RELU6" => {
            if value <= 0.0 {
                (0.0, true) // Dead at negative end
            } else if value >= 6.0 {
                (0.0, false) // Saturated but not "dead" (output is 6)
            } else {
                (1.0, false)
            }
        }
        "LEAKYRELU" => {
            // LeakyReLU never truly dead - always has 0.01 gradient for negative inputs
            if value >= 0.0 {
                (1.0, false)
            } else {
                (0.01, false)
            }
        }
        "ELU" => {
            if value >= 0.0 {
                (1.0, false)
            } else {
                // ELU'(x) = exp(x) for x < 0
                (value.exp(), false)
            }
        }
        "SELU" => {
            const ALPHA: f32 = 1.673_263_2;
            const LAMBDA: f32 = 1.050_701;
            if value >= 0.0 {
                (LAMBDA, false)
            } else {
                (LAMBDA * ALPHA * value.exp(), false)
            }
        }

        // Sigmoid-family - can saturate
        "TANH" => {
            let tanh_val = value.tanh();
            let gradient = 1.0 - tanh_val * tanh_val;
            (gradient, false)
        }
        "LOGISTIC" => {
            let sigmoid = if value >= 0.0 {
                1.0 / (1.0 + (-value).exp())
            } else {
                let exp_x = value.exp();
                exp_x / (1.0 + exp_x)
            };
            let gradient = sigmoid * (1.0 - sigmoid);
            (gradient, false)
        }
        "HARD_TANH" | "CLIPPED" => {
            if value.abs() >= saturation_thresholds::HARD_TANH_THRESHOLD {
                (0.0, false) // Saturated but not "dead"
            } else {
                (1.0, false)
            }
        }
        "BIPOLAR_SIGMOID" => {
            // 2 * sigmoid(x) - 1, derivative = 2 * sigmoid(x) * (1 - sigmoid(x))
            let sigmoid = if value >= 0.0 {
                1.0 / (1.0 + (-value).exp())
            } else {
                let exp_x = value.exp();
                exp_x / (1.0 + exp_x)
            };
            (2.0 * sigmoid * (1.0 - sigmoid), false)
        }
        "SOFTSIGN" => {
            // softsign'(x) = 1 / (1 + |x|)^2
            let denom = 1.0 + value.abs();
            (1.0 / (denom * denom), false)
        }

        // Other activations
        "IDENTITY" => (1.0, false),
        "ABSOLUTE" | "ABS" => {
            if value != 0.0 {
                (1.0, false) // |gradient| = 1 everywhere except at 0
            } else {
                (0.0, false) // Technically undefined at 0
            }
        }
        "ARCTAN" => {
            // arctan'(x) = 1 / (1 + x^2)
            (1.0 / (1.0 + value * value), false)
        }
        "BENT_IDENTITY" => {
            // bent_identity'(x) = x / (2 * sqrt(x^2 + 1)) + 1
            ((value / (2.0 * (value * value + 1.0).sqrt())) + 1.0, false)
        }
        "CUBE" => {
            // cube'(x) = 3x^2
            (3.0 * value * value, false)
        }
        "SQUARE" => {
            // square'(x) = 2x
            (2.0 * value, false)
        }
        "GAUSSIAN" => {
            // gaussian(x) = exp(-x^2), gaussian'(x) = -2x * exp(-x^2)
            let safe_x = value.abs().min(100.0);
            ((-2.0 * value * (-safe_x * safe_x).exp()).abs(), false)
        }
        "GELU" => {
            // GELU gradient approximation
            let x3 = value * value * value;
            let tanh_arg = 0.797_884_6_f32 * (value + 0.044_715_f32 * x3);
            let tanh_val = tanh_arg.tanh();
            let sech2 = 1.0 - tanh_val * tanh_val;
            let inner_deriv = 0.797_884_6_f32 * (1.0 + 3.0 * 0.044_715_f32 * value * value);
            let gradient = 0.5 * (1.0 + tanh_val) + 0.5 * value * sech2 * inner_deriv;
            (gradient, false)
        }
        "ISRU" => {
            // ISRU(x) = x / sqrt(1 + x^2)
            // ISRU'(x) = 1 / (1 + x^2)^(3/2)
            let denom = (1.0 + value * value).powf(1.5);
            (1.0 / denom, false)
        }
        "MISH" => {
            // Mish(x) = x * tanh(softplus(x))
            // Derivative is complex, approximate with numerical stability
            let sp = if value > 20.0 {
                value
            } else {
                (1.0 + value.exp()).ln()
            };
            let tanh_sp = sp.tanh();
            let sech2_sp = 1.0 - tanh_sp * tanh_sp;
            let sp_deriv = if value > 20.0 {
                1.0
            } else {
                value.exp() / (1.0 + value.exp())
            };
            (tanh_sp + value * sech2_sp * sp_deriv, false)
        }
        "SOFTPLUS" => {
            // softplus(x) = ln(1 + exp(x)), softplus'(x) = sigmoid(x)
            let sigmoid = if value >= 0.0 {
                1.0 / (1.0 + (-value).exp())
            } else {
                let exp_x = value.exp();
                exp_x / (1.0 + exp_x)
            };
            (sigmoid, false)
        }
        "SWISH" => {
            // Swish(x) = x * sigmoid(x)
            // Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            let sigmoid = if value >= 0.0 {
                1.0 / (1.0 + (-value).exp())
            } else {
                let exp_x = value.exp();
                exp_x / (1.0 + exp_x)
            };
            (sigmoid + value * sigmoid * (1.0 - sigmoid), false)
        }
        "SINE" | "SINUSOID" => (value.cos().abs(), false),
        "COSINE" => (value.sin().abs(), false),
        "TAN" => {
            let cos_val = value.cos();
            if cos_val.abs() < 1e-6 {
                (100.0, false) // Near asymptote, very high gradient
            } else {
                (1.0 / (cos_val * cos_val), false)
            }
        }
        "EXPONENTIAL" => {
            if value >= 88.0 {
                (f32::MAX, false)
            } else {
                (value.exp(), false)
            }
        }
        "LOGSIGMOID" => {
            // logsigmoid(x) = -ln(1 + exp(-x))
            // logsigmoid'(x) = 1 / (1 + exp(x)) = 1 - sigmoid(x)
            let sigmoid = if value >= 0.0 {
                1.0 / (1.0 + (-value).exp())
            } else {
                let exp_x = value.exp();
                exp_x / (1.0 + exp_x)
            };
            (1.0 - sigmoid, false)
        }
        "SQRT" => {
            if value > 0.0 {
                (0.5 / value.sqrt(), false)
            } else {
                (0.0, false) // Undefined for negative, zero at boundary
            }
        }
        "STDINVERSE" => {
            // 1/x, derivative = -1/x^2
            let eps = 1e-15_f32;
            let safe_x = if value.abs() < eps { eps } else { value.abs() };
            (1.0 / (safe_x * safe_x), false)
        }
        "COMPLEMENT" | "INVERSE" => {
            // 1 - x, derivative = -1 (absolute value = 1)
            (1.0, false)
        }

        // Threshold functions - gradient is zero almost everywhere
        // but can still propagate useful information at the threshold
        "STEP" | "BIPOLAR" => {
            // Technically gradient = 0 everywhere except at threshold
            // For gradient flow analysis, treat as having small gradient
            (0.0, false)
        }

        // Aggregate functions - cannot compute scalar gradient
        "MINIMUM" | "MAXIMUM" | "IF" | "MEAN" | "HYPOT" | "HYPOTV2" => {
            // For aggregate functions, assume gradient passes through
            // The actual gradient depends on which input "wins"
            (1.0, false)
        }

        // Unknown activation - assume identity-like behaviour
        _ => (1.0, false),
    }
}

/// Check if a neuron sample is in a saturated region based on its activation function.
///
/// Saturation means the gradient is effectively zero, preventing learning.
fn is_saturated(squash: &str, value: f32, gradient: f32) -> bool {
    use saturation_thresholds::*;

    // Use function-specific thresholds for common activations
    let saturated_by_threshold = match squash {
        "TANH" => value.abs() > TANH_THRESHOLD,
        "LOGISTIC" => value.abs() > LOGISTIC_THRESHOLD,
        "HARD_TANH" | "CLIPPED" => value.abs() >= HARD_TANH_THRESHOLD,
        "ARCTAN" => value.abs() > ARCTAN_THRESHOLD,
        "SOFTSIGN" => value.abs() > SOFTSIGN_THRESHOLD,
        "RELU6" => !(0.0..=6.0).contains(&value),
        // For other functions, rely on gradient threshold
        _ => false,
    };

    // Also check gradient magnitude as a fallback
    saturated_by_threshold || gradient < GRADIENT_SATURATION_THRESHOLD
}

/// Compute gradient flow statistics for all neurons in a creature.
///
/// This function analyses discovery records to compute gradient flow metrics
/// for each selectable neuron. The metrics help identify neurons with high
/// learning potential by measuring:
///
/// 1. **Gradient magnitude**: Average |f'(value)| across samples
/// 2. **Saturation ratio**: Fraction of samples in saturated region
/// 3. **Dead ratio**: Fraction of samples with zero gradient (ReLU-family)
///
/// # Arguments
/// * `parquet_file` - Path to parquet file containing discovery records
/// * `creature` - The creature to analyse
///
/// # Returns
/// Map from neuron UUID to GradientFlowStats
///
/// # Example
///
/// ```ignore
/// let stats = compute_gradient_flow_stats("records.parquet", &creature)?;
/// for (uuid, flow_stats) in &stats {
///     println!("{}: gradient={:.3}, saturated={:.1}%, dead={:.1}%",
///         uuid,
///         flow_stats.avg_gradient_magnitude,
///         flow_stats.saturation_ratio * 100.0,
///         flow_stats.dead_ratio * 100.0
///     );
/// }
/// ```
pub fn compute_gradient_flow_stats(
    parquet_file: &str,
    creature: &CreatureJson,
) -> Result<HashMap<String, GradientFlowStats>> {
    // Build squash map for efficient lookup
    let squash_map = build_squash_map(creature);

    // Load records grouped by neuron
    let records = read_all_records_grouped_by_neuron(parquet_file)
        .context("Failed to read discovery records for gradient flow analysis")?;

    // Compute stats for each neuron
    let stats: HashMap<String, GradientFlowStats> = creature
        .neurons
        .iter()
        .filter(|n| is_selectable_type(&n.neuron_type))
        .filter_map(|neuron| {
            let squash = squash_map.get(&neuron.uuid)?;
            let neuron_records = records.get(&neuron.uuid)?;

            if neuron_records.is_empty() {
                return Some((neuron.uuid.clone(), GradientFlowStats::default()));
            }

            let mut total_gradient = 0.0f64;
            let mut saturated_count = 0u32;
            let mut dead_count = 0u32;
            let mut valid_count = 0u32;

            for record in neuron_records {
                // Use pre-activation value if available, otherwise skip
                let value = match record.value {
                    Some(v) if v.is_finite() => v,
                    _ => continue,
                };

                let (gradient, is_dead) = compute_activation_gradient(squash, value);

                if !gradient.is_finite() {
                    continue;
                }

                valid_count += 1;
                total_gradient += gradient as f64;

                if is_dead {
                    dead_count += 1;
                }

                if is_saturated(squash, value, gradient) {
                    saturated_count += 1;
                }
            }

            if valid_count == 0 {
                return Some((neuron.uuid.clone(), GradientFlowStats::default()));
            }

            let avg_gradient_magnitude = (total_gradient / valid_count as f64) as f32;
            let saturation_ratio = saturated_count as f32 / valid_count as f32;
            let dead_ratio = dead_count as f32 / valid_count as f32;

            Some((
                neuron.uuid.clone(),
                GradientFlowStats {
                    avg_gradient_magnitude,
                    saturation_ratio,
                    dead_ratio,
                },
            ))
        })
        .collect();

    Ok(stats)
}

/// Compute gradient flow stats for a single neuron from its discovery records.
///
/// This is a helper function for use within `rank_focus_neurons` where records
/// are already loaded and we want to avoid re-loading the parquet file.
///
/// # Arguments
/// * `neuron_uuid` - UUID of the neuron
/// * `squash_map` - Map from neuron UUID to uppercase squash name
/// * `records` - Discovery records for this neuron
///
/// # Returns
/// GradientFlowStats for the neuron
pub(super) fn compute_gradient_flow_for_neuron(
    neuron_uuid: &str,
    squash_map: &HashMap<String, String>,
    records: &[DiscoverRecord],
) -> GradientFlowStats {
    let squash = match squash_map.get(neuron_uuid) {
        Some(s) => s.as_str(),
        None => return GradientFlowStats::default(),
    };

    if records.is_empty() {
        return GradientFlowStats::default();
    }

    let mut total_gradient = 0.0f64;
    let mut saturated_count = 0u32;
    let mut dead_count = 0u32;
    let mut valid_count = 0u32;

    for record in records {
        // Use pre-activation value if available, otherwise skip
        let value = match record.value {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };

        let (gradient, is_dead) = compute_activation_gradient(squash, value);

        if !gradient.is_finite() {
            continue;
        }

        valid_count += 1;
        total_gradient += gradient as f64;

        if is_dead {
            dead_count += 1;
        }

        if is_saturated(squash, value, gradient) {
            saturated_count += 1;
        }
    }

    if valid_count == 0 {
        return GradientFlowStats::default();
    }

    let avg_gradient_magnitude = (total_gradient / valid_count as f64) as f32;
    let saturation_ratio = saturated_count as f32 / valid_count as f32;
    let dead_ratio = dead_count as f32 / valid_count as f32;

    GradientFlowStats {
        avg_gradient_magnitude,
        saturation_ratio,
        dead_ratio,
    }
}

/// Compute a ranking factor from gradient flow statistics.
///
/// This factor adjusts the base ranking score (error × impact) to prioritise
/// neurons with high learning potential:
///
/// ```text
/// factor = (1 - saturation_ratio) × (1 - dead_ratio) × gradient_boost
/// ```
///
/// where `gradient_boost = 0.5 + 0.5 × clamp(avg_gradient_magnitude, 0, 1)`
/// ensures the factor is always positive and rewards higher gradient magnitudes.
///
/// # Returns
/// A factor in range (0.0, 1.0] that multiplies the base ranking score.
pub(super) fn compute_gradient_flow_factor(stats: &GradientFlowStats) -> f32 {
    // Factor for saturation: 1.0 when no saturation, 0.0 when fully saturated
    let saturation_factor = (1.0 - stats.saturation_ratio).max(0.0);

    // Factor for dead neurons: 1.0 when no dead samples, 0.0 when fully dead
    let dead_factor = (1.0 - stats.dead_ratio).max(0.0);

    // Factor for gradient magnitude: boost neurons with higher gradient
    // Clamp to [0, 1] range for consistent scaling, then apply 0.5 + 0.5*x
    // This ensures factor is always >= 0.5 (never completely kills the score)
    let clamped_gradient = stats.avg_gradient_magnitude.clamp(0.0, 1.0);
    let gradient_boost = 0.5 + 0.5 * clamped_gradient;

    // Combine factors multiplicatively
    // Minimum factor is ~0.01 to ensure neurons aren't completely hidden
    (saturation_factor * dead_factor * gradient_boost).max(0.01)
}

/// Build a map from neuron UUID to squash function name.
pub(super) fn build_squash_map(creature: &CreatureJson) -> HashMap<String, String> {
    creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.squash.to_uppercase()))
        .collect()
}
