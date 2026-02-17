//! Threshold computation and source variance analysis functions.

use crate::types::DiscoverRecord;

use super::{DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD, HelpfulSample};

// MIN_SOURCE_STD_DEV_REFERENCE moved to constants.rs as MIN_SOURCE_STD_DEV (Issue #424)
use crate::analysis::constants::MIN_SOURCE_STD_DEV as MIN_SOURCE_STD_DEV_REFERENCE;

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

    // MIN_SOURCE_STD_DEV moved to constants.rs (Issue #424)
    use crate::analysis::constants::MIN_SOURCE_STD_DEV;

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
///
/// Note: For dynamic threshold based on source variance profile, use
/// `compute_dynamic_constant_source_threshold()` instead.
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
                tracing::debug!(
                    raw_value = ?trimmed,
                    "Ignoring invalid NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD (expected 0 or a finite number > 0)"
                );
            }
            Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD)
        }
    }
}

/// Compute the dynamic constant-source effect threshold based on source variance profile.
///
/// Issue #199: The threshold for folding constant-source synapses into bias operations
/// should scale based on the overall source variance profile of the creature:
///
/// ```text
/// dynamic_threshold = DEFAULT_THRESHOLD × max(1.0, source_std_dev_avg / 0.05)
/// ```
///
/// This means:
/// - For creatures with mostly low-variance sources (avg std dev < 0.05): threshold stays at default
/// - For creatures with high-variance sources: threshold scales up proportionally
///
/// This captures more coordinated candidates in creatures where "constant" is relative
/// to the overall variance profile.
///
/// # Arguments
/// * `source_std_dev_avg` - Average standard deviation across all source activations
///
/// # Returns
/// The dynamic threshold value, always >= `DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD`
pub fn compute_dynamic_constant_source_threshold(source_std_dev_avg: f32) -> f32 {
    if !source_std_dev_avg.is_finite() || source_std_dev_avg <= 0.0 {
        return DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD;
    }

    let scaling_factor = (source_std_dev_avg / MIN_SOURCE_STD_DEV_REFERENCE).max(1.0);
    let dynamic_threshold = DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD * scaling_factor;

    if dynamic_threshold.is_finite() {
        dynamic_threshold
    } else {
        DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD
    }
}

/// Compute source activation standard deviation from discovery records.
///
/// Used to build the source variance profile for dynamic threshold calculation.
///
/// # Returns
/// The standard deviation of the activation values, or 0.0 if insufficient data
pub fn compute_source_std_dev(records: &[DiscoverRecord]) -> f32 {
    if records.len() < 2 {
        return 0.0;
    }

    let mut activation_sum = 0.0f64;
    let mut activation_sq_sum = 0.0f64;
    let mut count = 0u32;

    for record in records {
        if record.activation.is_finite() {
            let a = record.activation as f64;
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
    variance.max(0.0).sqrt() as f32
}

/// Get the constant source effect threshold, considering both env var override and dynamic calculation.
///
/// Issue #199: This function checks for an explicit env var override first. If no override is set,
/// it uses the dynamic threshold based on the source variance profile.
///
/// # Arguments
/// * `source_std_dev_avg` - Average standard deviation across all source activations.
///   Pass `None` to use only env var or default threshold.
///
/// # Returns
/// * `Some(threshold)` - The threshold to use for constant source detection
/// * `None` - Constant source folding is disabled (env var set to 0)
pub fn get_constant_source_threshold(source_std_dev_avg: Option<f32>) -> Option<f32> {
    use crate::analysis::utils::verbose_enabled;

    // First check for env var override
    let raw = std::env::var("NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD").ok();

    // If env var is set and valid, use it (explicit override takes precedence)
    if let Some(raw) = raw {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            match trimmed.parse::<f32>() {
                Ok(v) if v.is_finite() && v == 0.0 => return None, // Disabled
                Ok(v) if v.is_finite() && v > 0.0 => return Some(v), // Explicit override
                _ => {
                    if verbose_enabled() {
                        tracing::debug!(
                            raw_value = ?trimmed,
                            "Ignoring invalid NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD (expected 0 or a finite number > 0)"
                        );
                    }
                    // Fall through to dynamic calculation
                }
            }
        }
    }

    // No valid env var override - use dynamic threshold if source variance is provided
    match source_std_dev_avg {
        Some(avg) if avg.is_finite() && avg > 0.0 => {
            let threshold = compute_dynamic_constant_source_threshold(avg);
            if verbose_enabled() {
                tracing::debug!(
                    threshold = format_args!("{threshold:.2e}"),
                    source_std_dev_avg = format_args!("{avg:.4}"),
                    "Using dynamic constant-source threshold"
                );
            }
            Some(threshold)
        }
        _ => Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD),
    }
}
