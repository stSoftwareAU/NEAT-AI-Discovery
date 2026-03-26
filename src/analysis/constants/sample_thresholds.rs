//! Sample count thresholds for discovery analysis.
//!
//! Minimum sample requirements for reliable detection and evaluation,
//! including hold-out validation split parameters.

// =============================================================================
// Sample Count Thresholds
// =============================================================================

/// Minimum number of samples required for neuron candidate evaluation.
///
/// Used by weight calculation, synapse analysis, neuron analysis, and GPU
/// shader validation to ensure sufficient data for reliable predictions.
///
/// ## Valid Range
/// Must be >= 2 for statistical calculations. Values below 10 produce
/// unreliable least-squares fits.
pub const MIN_NEURON_SAMPLE_COUNT: usize = 10;

/// Minimum number of samples required for discovery module detection.
///
/// Used by detection modules (saturation, dead neuron, bottleneck, oscillation,
/// dormant synapse, opposing synapse, correlated error, etc.) to ensure
/// sufficient data for pattern recognition.
///
/// ## Valid Range
/// Must be >= `MIN_NEURON_SAMPLE_COUNT`. Values below 20 produce unreliable
/// pattern detection.
pub const MIN_DISCOVERY_SAMPLE_COUNT: usize = 20;

// =============================================================================
// Hold-Out Validation for Multi-Weight Search (Issue #893)
// =============================================================================

/// Minimum number of samples required to use hold-out validation.
///
/// Below this threshold, splitting into train/validate sets would leave
/// too few samples in each partition for reliable results. When the total
/// sample count is below this value, the current approach (full-sample
/// evaluation with pessimism discounting) is used as a fallback.
///
/// ## Valid Range
/// Must be >= `MIN_DISCOVERY_SAMPLE_COUNT`. Values below 20 produce
/// unreliable splits.
pub const HOLDOUT_MIN_SAMPLE_COUNT: usize = 20;

/// Fraction of samples reserved for the validation (hold-out) set.
///
/// The remaining samples (1 - this fraction) are used for training
/// (weight selection). A 70/30 split balances having enough training
/// data for reliable weight fitting while retaining a meaningful
/// validation set.
///
/// ## Valid Range
/// Must be in (0.1, 0.5). Values below 0.1 leave too few validation
/// samples. Values above 0.5 leave too few training samples.
pub const HOLDOUT_VALIDATION_FRACTION: f32 = 0.3;
