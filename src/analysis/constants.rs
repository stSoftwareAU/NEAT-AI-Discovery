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
/// Must be >= `MIN_NEURON_SAMPLE_COUNT`. Values below 20 produce unreliable
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
/// Must be > `SENTINEL_TOLERANCE`. Values above 0.2 may miss valid sentinels.
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
/// where more samples worsened than improved were still being proposed.
///
/// Issue #789: Raised from 0.5 to 0.6. GRQ-sampler cache data (Issue #787) showed
/// all 31 candidates that passed the 0.5 threshold still failed ablation testing.
/// Requiring 60% of samples to improve filters out marginal candidates where the
/// multi-weight search found a local optimum that does not generalise.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.3 provide insufficient filtering.
/// Values above 0.75 may over-filter legitimate candidates.
pub const MIN_IMPROVED_RATIO: f32 = 0.6;

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
/// Values above `MIN_IMPROVED_RATIO` may be too strict for neuron candidates.
pub const NEURON_MIN_IMPROVED_RATIO: f32 = 0.4;

// =============================================================================
// Activation-Function-Aware Neuron Scoring (Issue #887)
// =============================================================================

// Per-activation-function boost/penalty multipliers for add-neuron candidate scoring.
//
// GRQ-sampler discovery cache reveals dramatic differences in success rates by
// activation function. These multipliers are derived from Bayesian-smoothed success
// rates (Beta posterior with prior centred on the baseline ~13.9% success rate,
// K=20 pseudo-observations) to handle small sample sizes.
//
// Methodology:
//
// For each activation function:
// 1. Compute Bayesian-smoothed rate: (successes + α) / (total + K) where
//    α = baseline × K = 0.139 × 20 = 2.78, K = 20
// 2. Compute ratio to baseline: smoothed_rate / baseline
// 3. Apply square-root dampening to compress extreme ratios: ratio^0.5
// 4. Clamp to [0.5, 2.0] to avoid over-biasing
//
// Cache Evidence (GRQ-sampler):
//
// | Activation     | Successes | Total | Raw Rate | Smoothed Rate | Boost |
// |----------------|-----------|-------|----------|---------------|-------|
// | GELU           | 111       | 185   | 60.0%    | 55.5%         | 2.0   |
// | ABSOLUTE       | 28        | 38    | 73.6%    | 53.1%         | 1.95  |
// | Mish           | 75        | 156   | 48.0%    | 44.2%         | 1.78  |
// | ReLU6          | 27        | 54    | 50.0%    | 40.3%         | 1.70  |
// | BENT_IDENTITY  | 57        | 155   | 36.7%    | 34.2%         | 1.57  |
// | ELU            | 72        | 219   | 32.8%    | 31.3%         | 1.50  |
// | Softplus       | 29        | 91    | 31.8%    | 28.6%         | 1.43  |
// | ArcTan         | 11        | 58    | 18.9%    | 17.7%         | 1.13  |
// | SOFTSIGN       | 5         | 28    | 17.8%    | 16.2%         | 1.08  |
// | CLIPPED        | 9         | 59    | 15.2%    | 14.9%         | 1.03  |
// | IDENTITY       | 41        | 274   | 14.9%    | 14.9%         | 1.03  |
// | TANH           | 6         | 43    | 13.9%    | 13.9%         | 1.00  |
// | BIPOLAR        | 8         | 66    | 12.1%    | 12.5%         | 0.95  |
// | HARD_TANH      | 4         | 55    | 7.2%     | 9.0%          | 0.80  |
//
// Valid Range:
// Each boost must be in [0.5, 2.0]. Values below 0.5 risk suppressing
// potentially valuable candidates. Values above 2.0 risk over-biasing
// toward historically successful activations at the expense of exploration.

/// Boost multiplier for GELU activation (60.0% raw, Bayesian-smoothed 2.0×).
pub const ACTIVATION_BOOST_GELU: f64 = 2.0;

/// Boost multiplier for ABSOLUTE activation (73.6% raw, Bayesian-smoothed 1.95×).
pub const ACTIVATION_BOOST_ABSOLUTE: f64 = 1.95;

/// Boost multiplier for `Mish` activation (48.0% raw, Bayesian-smoothed 1.78×).
pub const ACTIVATION_BOOST_MISH: f64 = 1.78;

/// Boost multiplier for `ReLU6` activation (50.0% raw, Bayesian-smoothed 1.70×).
pub const ACTIVATION_BOOST_RELU6: f64 = 1.70;

/// Boost multiplier for `BENT_IDENTITY` activation (36.7% raw, Bayesian-smoothed 1.57×).
pub const ACTIVATION_BOOST_BENT_IDENTITY: f64 = 1.57;

/// Boost multiplier for ELU activation (32.8% raw, Bayesian-smoothed 1.50×).
pub const ACTIVATION_BOOST_ELU: f64 = 1.50;

/// Boost multiplier for `Softplus` activation (31.8% raw, Bayesian-smoothed 1.43×).
pub const ACTIVATION_BOOST_SOFTPLUS: f64 = 1.43;

/// Boost multiplier for `ArcTan` activation (18.9% raw, Bayesian-smoothed 1.13×).
pub const ACTIVATION_BOOST_ARCTAN: f64 = 1.13;

/// Boost multiplier for SOFTSIGN activation (17.8% raw, Bayesian-smoothed 1.08×).
pub const ACTIVATION_BOOST_SOFTSIGN: f64 = 1.08;

/// Boost multiplier for CLIPPED activation (15.2% raw, Bayesian-smoothed 1.03×).
pub const ACTIVATION_BOOST_CLIPPED: f64 = 1.03;

/// Boost multiplier for IDENTITY activation (14.9% raw, Bayesian-smoothed 1.03×).
pub const ACTIVATION_BOOST_IDENTITY: f64 = 1.03;

/// Neutral multiplier for TANH activation (13.9% raw, matches baseline exactly).
pub const ACTIVATION_BOOST_TANH: f64 = 1.0;

/// Penalty multiplier for BIPOLAR activation (12.1% raw, Bayesian-smoothed 0.95×).
pub const ACTIVATION_BOOST_BIPOLAR: f64 = 0.95;

/// Penalty multiplier for `HARD_TANH` activation (7.2% raw, Bayesian-smoothed 0.80×).
pub const ACTIVATION_BOOST_HARD_TANH: f64 = 0.80;

/// Returns the activation-function-aware boost/penalty multiplier for add-neuron
/// candidate scoring (Issue #887).
///
/// This lookup maps activation function names to their Bayesian-smoothed boost
/// multipliers derived from GRQ-sampler cache success rates. Unknown activations
/// (including `ReLU`, which is evaluated separately) receive a neutral multiplier of 1.0.
#[inline]
pub fn activation_neuron_boost(squash_name: &str) -> f64 {
    match squash_name {
        "GELU" => ACTIVATION_BOOST_GELU,
        "ABSOLUTE" => ACTIVATION_BOOST_ABSOLUTE,
        "Mish" => ACTIVATION_BOOST_MISH,
        "ReLU6" => ACTIVATION_BOOST_RELU6,
        "BENT_IDENTITY" => ACTIVATION_BOOST_BENT_IDENTITY,
        "ELU" => ACTIVATION_BOOST_ELU,
        "Softplus" => ACTIVATION_BOOST_SOFTPLUS,
        "ArcTan" => ACTIVATION_BOOST_ARCTAN,
        "SOFTSIGN" => ACTIVATION_BOOST_SOFTSIGN,
        "CLIPPED" => ACTIVATION_BOOST_CLIPPED,
        "IDENTITY" => ACTIVATION_BOOST_IDENTITY,
        "TANH" => ACTIVATION_BOOST_TANH,
        "BIPOLAR" => ACTIVATION_BOOST_BIPOLAR,
        "HARD_TANH" => ACTIVATION_BOOST_HARD_TANH,
        _ => 1.0, // Neutral boost for unknown activations (including ReLU)
    }
}

/// Minimum activation boost value (lower clamp).
///
/// ## Valid Range
/// Must be > 0.0 and < 1.0.
pub const ACTIVATION_BOOST_MIN: f64 = 0.5;

/// Maximum activation boost value (upper clamp).
///
/// ## Valid Range
/// Must be > 1.0 and <= 3.0.
pub const ACTIVATION_BOOST_MAX: f64 = 2.0;

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
/// pessimism parameters (`PESSIMISM_DISCOUNT_FLOOR` = 0.15, `PESSIMISM_CURVE_EXPONENT`
/// = 0.6) are calibrated for the overall candidate pool and are too generous for
/// neuron candidates specifically.
///
/// A lower floor applies more aggressive base discounting to neuron predictions,
/// reducing the expected gain for candidates with few samples improving.
///
/// ## Valid Range
/// Must be in (0.0, `PESSIMISM_DISCOUNT_FLOOR`). Values below 0.05 risk
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
/// Must be in (`PESSIMISM_CURVE_EXPONENT`, 1.0]. Values above 0.9 give
/// near-linear behaviour.
pub const NEURON_PESSIMISM_CURVE_EXPONENT: f32 = 0.75;

// =============================================================================
// Synapse-Specific Pessimism Discount (Issue #789)
// =============================================================================

/// Minimum pessimism discount floor for synapse candidates (Issue #789).
///
/// GRQ-sampler analysis (Issue #787) shows add-synapses has a 0% success rate
/// (0 / 31) — the worst of all candidate types. The generic pessimism parameters
/// (`PESSIMISM_DISCOUNT_FLOOR` = 0.15, `PESSIMISM_CURVE_EXPONENT` = 0.6) and even the
/// neuron-specific parameters (0.10, 0.75) are too generous for synapse candidates.
///
/// A lower floor applies more aggressive base discounting to synapse predictions,
/// reducing the expected gain for candidates with marginal improved ratios. This
/// accounts for the multi-weight search (9 variants) creating selection bias that
/// overfits to sample data.
///
/// ## Valid Range
/// Must be in (0.0, `NEURON_PESSIMISM_DISCOUNT_FLOOR`]. Values below 0.02 risk
/// zeroing-out all synapse candidates.
pub const SYNAPSE_PESSIMISM_DISCOUNT_FLOOR: f32 = 0.05;

/// Exponent for the synapse-specific pessimism discount curve (Issue #789).
///
/// A higher exponent (closer to linear) produces a less forgiving curve at
/// moderate ratios. With a 0% success rate, synapse candidates need the most
/// aggressive discounting of all candidate types. The multi-weight search
/// (9 weight variants) creates selection bias where the best weight for the
/// sample data does not generalise to the full evaluation.
///
/// With exponent 0.85 and floor 0.05 (discount = 0.05 + 0.95 × ratio^0.85):
/// - ratio 0.1 → 0.1^0.85 ≈ 0.141 → discount ≈ 0.184
/// - ratio 0.4 → 0.4^0.85 ≈ 0.453 → discount ≈ 0.480
/// - ratio 0.6 → 0.6^0.85 ≈ 0.641 → discount ≈ 0.659
/// - ratio 0.7 → 0.7^0.85 ≈ 0.741 → discount ≈ 0.754
/// - ratio 1.0 → 1.0       → discount = 1.000
///
/// ## Valid Range
/// Must be in (`NEURON_PESSIMISM_CURVE_EXPONENT`, 1.0]. Values above 0.95 give
/// near-linear behaviour.
pub const SYNAPSE_PESSIMISM_CURVE_EXPONENT: f32 = 0.85;

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
/// Production analysis shows coordinated-structural candidates have a 2.3% success
/// rate (272 / 12,069 in GRQ-sampler cache, Issue #787) because each operation's
/// prediction uncertainty compounds when combined.
/// For a candidate with N operations, the discount is:
///
/// ```text
/// discount = COORDINATED_OPERATION_DISCOUNT ^ (N - 1)
/// ```
///
/// Issue #790: Reduced from 0.8 to 0.65. The 2.3% success rate and near-negligible
/// actual gains (~2.2e-14) on successful candidates demonstrate that multi-operation
/// predictions are far less reliable than previously assumed. With 0.65, a 4-operation
/// candidate receives 0.65^3 ≈ 0.274 discount, more aggressively filtering out
/// candidates whose compounding uncertainty makes success unlikely.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.5 may over-discount legitimate candidates.
/// Values above 0.95 provide insufficient correction.
pub const COORDINATED_OPERATION_DISCOUNT: f32 = 0.65;

/// Minimum absolute gain required for a multi-operation coordinated candidate.
///
/// Single-operation candidates are accepted with any positive gain, but
/// multi-operation candidates (>= 2 operations) must exceed this threshold
/// after discounting. This prevents near-zero predictions from generating
/// candidates that almost never succeed.
///
/// Issue #790: Raised from 1e-5 to 1e-3. GRQ-sampler cache analysis (Issue #787)
/// shows that successful coordinated candidates achieve only ~2.2e-14 actual gain,
/// demonstrating an enormous prediction-to-reality gap. The previous threshold of
/// 1e-5 allowed candidates with negligible predicted improvement through, wasting
/// ablation testing time. The raised threshold filters out marginal predictions
/// while still admitting candidates with meaningful expected gains.
///
/// ## Valid Range
/// Must be > 0.0. Values above 1e-2 may filter too aggressively.
pub const MIN_COORDINATED_MULTI_OP_GAIN: f32 = 1e-3;

// =============================================================================
// Coordinated-Structural Pessimism Discount (Issue #790)
// =============================================================================

/// Flat pessimism discount applied to all coordinated-structural candidates.
///
/// GRQ-sampler analysis (Issue #787) shows coordinated-structural candidates have
/// a 2.3% success rate (272 / 12,069), with successful candidates achieving only
/// near-negligible score deltas (~2.2e-14). Unlike synapse and neuron candidates
/// which have per-sample `improved_count/total_count` ratios, coordinated candidates
/// combine multiple operations whose individual predictions compound optimistically.
///
/// This flat multiplicative discount is applied to all coordinated-structural
/// candidates during post-processing, analogous to the pessimism discounts applied
/// to synapse and neuron candidates but using a fixed factor rather than a
/// ratio-based curve (since coordinated candidates lack per-sample counts).
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.05 risk zeroing-out all coordinated
/// candidates. Values above 0.30 provide insufficient correction given the
/// 2.3% success rate.
pub const COORDINATED_PESSIMISM_DISCOUNT: f32 = 0.15;
