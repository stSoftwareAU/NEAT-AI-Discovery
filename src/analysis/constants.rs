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

/// Minimum ratio of improved samples required for a neuron candidate to be accepted.
///
/// Issue #733: The add-neurons module had a 14.1% success rate. Neuron candidates
/// are inherently noisier than synapse candidates because they involve two new
/// connections (incoming + outgoing) rather than one. A slightly lower threshold
/// than `MIN_IMPROVED_RATIO` allows moderate-quality candidates through while
/// still filtering out clearly bad ones.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.3 provide insufficient filtering.
/// Values above MIN_IMPROVED_RATIO may be too strict for neuron candidates.
pub const NEURON_MIN_IMPROVED_RATIO: f32 = 0.4;

/// Minimum pessimism discount applied to all score predictions.
///
/// Production analysis (creature b2ff6e45, GRQ-sampler commit a1340f8d) showed
/// that raw improvement percentages are wildly over-estimated — the sole
/// successful candidate predicted +0.0205 but achieved only +0.0000011
/// (an 18,500× over-estimation). The improvement calculation measures the
/// fraction of a single target neuron's squared error explained by sampled
/// data, but this does not generalise directly to creature-level score gain.
///
/// The pessimism discount scales predictions down using a concave (power) curve
/// based on the ratio of samples that actually improved:
///
/// ```text
/// improved_ratio = improved_count / total_count
/// adjusted_ratio = improved_ratio ^ PESSIMISM_CURVE_EXPONENT
/// discount = FLOOR + (1 - FLOOR) × adjusted_ratio
/// discounted_gain = raw_gain × discount
/// ```
///
/// Issue #733: Changed from linear to concave curve. The linear formula was
/// too aggressive for add-neurons candidates (14.1% success rate), discounting
/// moderate-quality candidates (30-60% improved ratio) excessively. The concave
/// curve is more forgiving at moderate ratios while remaining aggressive at
/// very low ratios (<10%).
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.1 risk zeroing-out legitimate candidates.
/// Values above 0.5 provide insufficient correction.
pub const PESSIMISM_DISCOUNT_FLOOR: f32 = 0.15;

/// Exponent for the concave pessimism discount curve (Issue #733).
///
/// The improved ratio is raised to this power before being used in the discount
/// formula. An exponent < 1.0 produces a concave curve that is:
/// - More forgiving at moderate ratios (30-60%): retains add-neuron candidates
///   with genuine but moderate signal
/// - Still aggressive at very low ratios (<10%): filters out noise
///
/// With exponent 0.6 and floor 0.15 (discount = 0.15 + 0.85 × ratio^0.6):
/// - ratio 0.1 → 0.1^0.6 ≈ 0.251 → discount ≈ 0.363 → gain 0.05 × 0.363 ≈ 0.018
/// - ratio 0.4 → 0.4^0.6 ≈ 0.575 → discount ≈ 0.639 → gain 0.05 × 0.639 ≈ 0.032
/// - ratio 0.7 → 0.7^0.6 ≈ 0.802 → discount ≈ 0.832 → gain 0.05 × 0.832 ≈ 0.042
/// - ratio 1.0 → 1.0       → discount = 1.000 → gain 0.05 × 1.000 = 0.050
///
/// ## Valid Range
/// Must be in (0.0, 1.0]. Values below 0.3 may over-flatten the curve.
/// Values above 0.9 give near-linear behaviour with minimal benefit.
pub const PESSIMISM_CURVE_EXPONENT: f32 = 0.6;

// =============================================================================
// Neuron-Specific Pessimism Discount (Issue #791)
// =============================================================================

/// Minimum pessimism discount floor for neuron candidates (Issue #791).
///
/// GRQ-sampler analysis (Issue #787) shows add-neurons has a 15% success rate
/// (3,812 / 25,812) — substantially lower than synapse candidates. The generic
/// pessimism parameters (PESSIMISM_DISCOUNT_FLOOR = 0.15, PESSIMISM_CURVE_EXPONENT
/// = 0.6) are calibrated for the overall candidate pool and are too generous for
/// neuron candidates specifically.
///
/// A lower floor applies more aggressive base discounting to neuron predictions,
/// reducing the expected gain for candidates with few samples improving.
///
/// ## Valid Range
/// Must be in (0.0, PESSIMISM_DISCOUNT_FLOOR). Values below 0.05 risk
/// zeroing-out legitimate neuron candidates.
pub const NEURON_PESSIMISM_DISCOUNT_FLOOR: f32 = 0.10;

/// Exponent for the neuron-specific pessimism discount curve (Issue #791).
///
/// A higher exponent (closer to linear) produces a less forgiving curve at
/// moderate ratios compared to the generic exponent (0.6). This is appropriate
/// for neuron candidates because their 15% success rate suggests moderate
/// improved ratios (30–60%) are less reliable predictors of actual success
/// than they are for synapse candidates.
///
/// With exponent 0.75 and floor 0.10 (discount = 0.10 + 0.90 × ratio^0.75):
/// - ratio 0.1 → 0.1^0.75 ≈ 0.178 → discount ≈ 0.260
/// - ratio 0.4 → 0.4^0.75 ≈ 0.506 → discount ≈ 0.555
/// - ratio 0.7 → 0.7^0.75 ≈ 0.744 → discount ≈ 0.770
/// - ratio 1.0 → 1.0       → discount = 1.000
///
/// ## Valid Range
/// Must be in (PESSIMISM_CURVE_EXPONENT, 1.0]. Values above 0.9 give
/// near-linear behaviour.
pub const NEURON_PESSIMISM_CURVE_EXPONENT: f32 = 0.75;

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

// =============================================================================
// Coordinated-Structural Validation (Issue #732)
// =============================================================================

/// Per-operation compounding uncertainty discount for multi-operation candidates.
///
/// Production analysis shows coordinated-structural candidates have a 1.9% success
/// rate because each operation's prediction uncertainty compounds when combined.
/// For a candidate with N operations, the discount is:
///
/// ```text
/// discount = COORDINATED_OPERATION_DISCOUNT ^ (N - 1)
/// ```
///
/// With a factor of 0.8, a 4-operation candidate receives 0.8^3 = 0.512 discount,
/// roughly halving the predicted gain to account for inter-operation interference.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.5 may over-discount legitimate candidates.
/// Values above 0.95 provide insufficient correction.
pub const COORDINATED_OPERATION_DISCOUNT: f32 = 0.8;

/// Minimum absolute gain required for a multi-operation coordinated candidate.
///
/// Single-operation candidates are accepted with any positive gain, but
/// multi-operation candidates (>= 2 operations) must exceed this threshold
/// after discounting. This prevents near-zero predictions from generating
/// candidates that almost never succeed.
///
/// ## Valid Range
/// Must be > 0.0. Values above 1e-4 may filter too aggressively.
pub const MIN_COORDINATED_MULTI_OP_GAIN: f32 = 1e-5;
