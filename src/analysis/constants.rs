//! Central constants module for discovery thresholds (Issue #424).
//!
//! This module is the single source of truth for all discovery detection
//! constants and thresholds. Previously these were duplicated across
//! individual analysis modules.
//!
//! ## Constant Categories
//!
//! - **Sample count thresholds**: Minimum samples required for reliable detection
//! - **Sentinel detection**: Constants for identifying sentinel/null values
//! - **Source variance**: Thresholds for source activation variance filtering
//! - **Candidate diversification**: Parameters for deadline-constrained exploration

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
/// Must be >= MIN_NEURON_SAMPLE_COUNT. Values below 20 produce unreliable
/// pattern detection.
pub const MIN_DISCOVERY_SAMPLE_COUNT: usize = 20;

// =============================================================================
// Sentinel Detection Constants
// =============================================================================

/// Candidate sentinel values to check for clusters.
///
/// These are the most common sentinel values in normalised data.
/// Used by observation range, bounded range, and sentinel gating modules.
pub const CANDIDATE_SENTINELS: [f32; 3] = [-1.0, 0.0, 1.0];

/// Minimum fraction of samples at a sentinel value to consider it a cluster.
///
/// If fewer than this fraction of samples cluster at a candidate sentinel
/// value, it is not considered a meaningful sentinel.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.10 may flag noise as sentinels.
pub const MIN_SENTINEL_FRACTION: f32 = 0.15;

/// Tolerance for grouping values into a sentinel cluster.
///
/// Values within this distance of a candidate sentinel are considered part
/// of the cluster. Also used as the default sentinel tolerance for
/// range-aware weight calculations.
///
/// ## Valid Range
/// Must be > 0.0. Values above 0.1 may merge distinct value groups.
pub const SENTINEL_TOLERANCE: f32 = 0.02;

/// Minimum gap between a sentinel cluster and the useful value range.
///
/// If the gap between the sentinel cluster and the nearest useful value is
/// smaller than this threshold, the values are too interleaved to separate
/// reliably.
///
/// ## Valid Range
/// Must be > SENTINEL_TOLERANCE. Values above 0.2 may miss valid sentinels.
pub const MIN_SENTINEL_GAP: f32 = 0.05;

// =============================================================================
// Source Variance Thresholds
// =============================================================================

/// Minimum source activation standard deviation for full credit.
///
/// Sources with std dev below this are progressively discounted to avoid
/// over-prediction from constant-ish sources. Also used as the reference
/// value for dynamic constant-source threshold scaling.
///
/// ## Context
/// Based on production analysis: input-1064 had std dev 0.01 and caused
/// massive over-prediction. Sources should have at least 0.05 std dev for
/// reliable correlation.
///
/// ## Valid Range
/// Must be > 0.0. Values above 0.1 may discard useful low-variance sources.
pub const MIN_SOURCE_STD_DEV: f32 = 0.05;

// =============================================================================
// Candidate Diversification
// =============================================================================

/// Number of top candidates to diversify when deadline-constrained.
///
/// When analysis runs under a deadline, the top-K candidates are shuffled
/// so that repeated runs explore different high-quality candidates over time.
/// This helps with failure caches and avoids category starvation.
///
/// ## Valid Range
/// Must be >= 1. Values above 128 may reduce the benefit of sorting by
/// expected improvement.
pub const DIVERSIFY_TOP_K: usize = 64;
