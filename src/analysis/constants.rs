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

// =============================================================================
// Source-Type Scoring (Issue #465)
// =============================================================================

/// Scoring boost multiplier for candidates from input-neuron sources.
///
/// GRQ-sampler analysis shows input neurons as synapse sources have a 36.2%
/// success rate compared to 2.8–3.3% for hidden neurons. This boost is applied
/// as a static multiplier to `expected_creature_score_gain` when the source
/// neuron is an input neuron and no historical data is available yet.
///
/// ## Valid Range
/// Must be > 1.0 (boost) and <= 3.0 (avoid over-biasing).
pub const INPUT_SOURCE_BOOST: f64 = 1.5;

/// Minimum number of recorded outcomes before applying source-type boost.
///
/// Below this threshold, the Bayesian estimate is too noisy to use for
/// boosting. The cache returns a neutral boost (1.0) until enough samples
/// have been collected.
///
/// ## Valid Range
/// Must be >= 5 to avoid noise and <= 50 to be responsive.
pub const MIN_BOOST_SAMPLES: usize = 10;

// =============================================================================
// Target-Type Scoring (Issue #468)
// =============================================================================

/// Scoring boost multiplier for candidates targeting existing hidden neurons.
///
/// GRQ-sampler analysis shows existing hidden neurons as targets have a 31.4%
/// success rate compared to 5.3–5.4% for output or discovery-hidden neurons.
/// This boost is applied as a static multiplier to `expected_creature_score_gain`
/// when the target neuron is an existing hidden neuron.
///
/// ## Valid Range
/// Must be > 1.0 (boost) and <= 3.0 (avoid over-biasing).
pub const EXISTING_HIDDEN_TARGET_BOOST: f64 = 1.5;

// =============================================================================
// Individual Operation Pre-Screen (Issue #508)
// =============================================================================

/// Maximum individual harm allowed for a source to participate in epistatic or
/// synergistic pairing.
///
/// Production analysis (creature b2ff6e45, GRQ-sampler commit a1340f8d) showed
/// that all 10 coordinated-structural candidates failed because they all included
/// the same harmful operation (e8480883 → output-0, weight 0.1) which degraded
/// the score by ~−0.042. The partner neuron varied but could never overcome that
/// dominant damage.
///
/// Before forming a coordinated pair, each individual operation is pre-screened:
/// if its `individual_improvement` is below this threshold, it is excluded from
/// pairing. Issue #731 tightened this from −0.01 to 0.0 because production data
/// showed that even mildly harmful sources (e.g. −0.005) consistently caused
/// combo-successful failures — the partner could never overcome the damage.
///
/// ## Valid Range
/// Must be >= 0.0 to exclude all harmful individual sources from pairing.
pub const MAX_INDIVIDUAL_HARM_FOR_PAIRING: f32 = 0.0;

// =============================================================================
// Pessimism Discount (Issue #506)
// =============================================================================

/// Minimum ratio of improved samples required for a synapse candidate to be accepted.
///
/// Issue #730: The add-synapses module had a 0% success rate because candidates
/// where more samples worsened than improved were still being proposed. This
/// threshold requires that at least this fraction of samples must improve before
/// a candidate is considered viable.
///
/// A ratio of 0.5 means at least half the samples must improve, filtering out
/// candidates that hurt more than they help.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.3 provide insufficient filtering.
/// Values above 0.75 may over-filter legitimate candidates.
pub const MIN_IMPROVED_RATIO: f32 = 0.5;

/// Minimum pessimism discount applied to all score predictions.
///
/// Production analysis (creature b2ff6e45, GRQ-sampler commit a1340f8d) showed
/// that raw improvement percentages are wildly over-estimated — the sole
/// successful candidate predicted +0.0205 but achieved only +0.0000011
/// (an 18,500× over-estimation). The improvement calculation measures the
/// fraction of a single target neuron's squared error explained by sampled
/// data, but this does not generalise directly to creature-level score gain.
///
/// The pessimism discount scales predictions down based on the ratio of
/// samples that actually improved (`improved_count / total_count`):
///
/// ```text
/// discount = FLOOR + (1 - FLOOR) × (improved_count / total_count)
/// discounted_gain = raw_gain × discount
/// ```
///
/// When all samples improve (ratio = 1.0), the discount equals 1.0 (only the
/// floor applies). When few samples improve, the discount approaches the floor.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.1 risk zeroing-out legitimate candidates.
/// Values above 0.5 provide insufficient correction.
pub const PESSIMISM_DISCOUNT_FLOOR: f32 = 0.15;

// =============================================================================
// NaN-safe Floating-Point Comparison Helpers (Issue #483)
// =============================================================================

/// NaN-safe descending comparison for `f32` values.
///
/// Uses `f32::total_cmp()` which provides a total ordering including NaN.
/// NaN values sort after all finite values (to the end of a descending sort).
///
/// Replaces the error-prone `partial_cmp().unwrap_or(Ordering::Equal)` pattern
/// which silently treats NaN as equal to any value, corrupting sort order.
#[inline]
pub fn cmp_f32_desc(a: &f32, b: &f32) -> std::cmp::Ordering {
    b.total_cmp(a)
}

/// NaN-safe ascending comparison for `f32` values.
///
/// Uses `f32::total_cmp()` which provides a total ordering including NaN.
/// NaN values sort after all finite values (to the end of an ascending sort).
#[inline]
pub fn cmp_f32_asc(a: &f32, b: &f32) -> std::cmp::Ordering {
    a.total_cmp(b)
}

/// NaN-safe descending comparison for `f64` values.
///
/// Uses `f64::total_cmp()` which provides a total ordering including NaN.
#[inline]
pub fn cmp_f64_desc(a: &f64, b: &f64) -> std::cmp::Ordering {
    b.total_cmp(a)
}
