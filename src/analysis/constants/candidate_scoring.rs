//! Candidate evaluation, scoring, and calibration constants.
//!
//! Covers diversification, source/target-type boosts, activation-function
//! boosts, pessimism discounts, prediction calibration, coordinated-structural
//! validation, and NaN-safe comparison helpers.

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

/// Scoring boost multiplier for candidates from hidden-neuron sources (Issue #910).
///
/// Hidden-to-hidden synapse candidates connect existing computational units into
/// more complex structures. While input sources have higher individual success
/// rates, hidden-to-hidden connections are essential for building deeper network
/// architectures. This modest boost ensures hidden-sourced candidates are not
/// entirely eclipsed by `INPUT_SOURCE_BOOST` during ranking.
///
/// ## Valid Range
/// Must be >= 1.0 and <= `INPUT_SOURCE_BOOST` (hidden sources should not
/// outrank input sources, just compete more fairly).
pub const HIDDEN_SOURCE_BOOST: f64 = 1.2;

/// Interleave interval for hidden sources during source ordering (Issue #907).
///
/// After every `HIDDEN_SOURCE_INTERLEAVE_INTERVAL` input sources, one hidden
/// source is inserted into the evaluation order. This ensures hidden-to-hidden
/// synapse candidates are evaluated even under tight deadline constraints.
///
/// With an interval of 3, approximately 25% of evaluation slots go to hidden
/// sources — enough to discover hidden-to-hidden connections without starving
/// the higher-success-rate input sources.
///
/// ## Valid Range
/// Must be >= 2 (to still prioritise inputs) and <= 5 (to ensure hidden
/// sources get meaningful evaluation time).
pub const HIDDEN_SOURCE_INTERLEAVE_INTERVAL: usize = 3;

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
// Activation-Function-Aware Neuron Scoring (Issue #887, Issue #909)
// =============================================================================

// Per-activation-function boost/penalty multipliers for add-neuron candidate scoring.
//
// GRQ-sampler discovery cache reveals dramatic differences in success rates by
// activation function. These multipliers are derived from Bayesian-smoothed success
// rates (Beta posterior with prior centred on the baseline ~13.9% success rate,
// K=20 pseudo-observations) to handle small sample sizes.
//
// Issue #909 recalibration: IDENTITY's 14.9% raw success rate is inflated because
// IDENTITY candidates dominate the candidate pool (274 total — more than any other
// activation). Per-candidate success is mediocre, and GRQ-sampler evidence (commit
// 7f15429) shows IDENTITY neurons are frequently substituted with non-linear
// activations like SINE for improvement. A penalty of 0.85× is applied to discourage
// IDENTITY dominance and encourage exploration of non-linear alternatives.
//
// Additionally, SINE is added as a supported activation with a modest boost, reflecting
// its demonstrated value as a substitution target for IDENTITY neurons.
//
// Methodology:
//
// For each activation function:
// 1. Compute Bayesian-smoothed rate: (successes + α) / (total + K) where
//    α = baseline × K = 0.139 × 20 = 2.78, K = 20
// 2. Compute ratio to baseline: smoothed_rate / baseline
// 3. Apply square-root dampening to compress extreme ratios: ratio^0.5
// 4. Clamp to [0.5, 2.0] to avoid over-biasing
// 5. Apply candidate-pool normalisation penalty for over-represented activations
//
// Cache Evidence (GRQ-sampler, with Issue #909 normalisation):
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
// | SINE           | —         | —     | —        | ~20%*         | 1.15  |
// | ArcTan         | 11        | 58    | 18.9%    | 17.7%         | 1.13  |
// | SOFTSIGN       | 5         | 28    | 17.8%    | 16.2%         | 1.08  |
// | CLIPPED        | 9         | 59    | 15.2%    | 14.9%         | 1.03  |
// | TANH           | 6         | 43    | 13.9%    | 13.9%         | 1.00  |
// | BIPOLAR        | 8         | 66    | 12.1%    | 12.5%         | 0.95  |
// | IDENTITY       | 41        | 274   | 14.9%    | 14.9%         | 0.85† |
// | HARD_TANH      | 4         | 55    | 7.2%     | 9.0%          | 0.80  |
//
// * SINE boost estimated from substitution evidence (commit 7f15429)
// † IDENTITY penalised (Issue #909): raw rate inflated by candidate-pool dominance;
//   neurons frequently substituted with non-linear activations post-addition
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

/// Boost multiplier for SINE activation (~20% estimated, 1.15×).
///
/// SINE is added based on GRQ-sampler evidence showing it successfully
/// substitutes IDENTITY neurons (commit 7f15429). The modest boost encourages
/// exploration of this non-linear activation.
pub const ACTIVATION_BOOST_SINE: f64 = 1.15;

/// Boost multiplier for `ArcTan` activation (18.9% raw, Bayesian-smoothed 1.13×).
pub const ACTIVATION_BOOST_ARCTAN: f64 = 1.13;

/// Boost multiplier for SOFTSIGN activation (17.8% raw, Bayesian-smoothed 1.08×).
pub const ACTIVATION_BOOST_SOFTSIGN: f64 = 1.08;

/// Boost multiplier for CLIPPED activation (15.2% raw, Bayesian-smoothed 1.03×).
pub const ACTIVATION_BOOST_CLIPPED: f64 = 1.03;

/// Penalty multiplier for IDENTITY activation (Issue #909).
///
/// Although IDENTITY has a 14.9% raw success rate (near the 13.9% baseline),
/// this rate is inflated by IDENTITY's dominance in the candidate pool (274
/// candidates — more than any other activation). GRQ-sampler evidence (commit
/// 7f15429) shows IDENTITY neurons are frequently substituted with non-linear
/// activations like SINE for improvement, indicating IDENTITY acts as a
/// placeholder rather than an optimal choice. The penalty discourages IDENTITY
/// dominance and encourages exploration of genuinely better non-linear activations.
pub const ACTIVATION_BOOST_IDENTITY: f64 = 0.85;

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
        "SINE" | "SINUSOID" => ACTIVATION_BOOST_SINE,
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

// =============================================================================
// Pessimism Discount (Issue #506)
// =============================================================================

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

/// Conservative weight scale used when estimating coordinated candidate gains (Issue #897).
///
/// During evaluation, coordinated candidate weights are scaled to 0.2×, 0.1×, or 0.05×
/// variants (see `variant_generation.rs`). The estimation should use the most likely
/// tested weight scale (0.2×) rather than the full optimal weight, because non-linear
/// activation functions mean scaling a weight does NOT proportionally scale improvement.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Should match `COORDINATED_CONSERVATIVE_WEIGHT_SCALE` in
/// `variant_generation.rs`.
pub const COORDINATED_ESTIMATION_WEIGHT_SCALE: f32 = 0.2;

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

// =============================================================================
// Scoring Boost Multipliers
// =============================================================================

/// Scoring boost multiplier for Micro-Nudge variant candidates (Issue #888).
///
/// GRQ-sampler cache evidence shows the Micro-Nudge pattern (incoming=2,
/// outgoing=0.001–0.005) dominates successes at ~90% of successful samples.
/// This boost is applied to Micro-Nudge variant candidates to prioritise
/// them in the ranking.
///
/// ## Valid Range
/// Must be > 1.0 (boost) and <= 2.0 (avoid over-biasing).
pub const MICRO_NUDGE_VARIANT_BOOST: f32 = 1.5;

/// Scoring boost multiplier for `remove-low-impact` candidates (Issue #892).
///
/// GRQ-sampler discovery cache shows `remove-low-impact` has the highest success
/// rate at 21.5% (440/2,043) — roughly double the overall 10.7% rate. This boost
/// is applied to the `removal_savings` score to ensure removal candidates are
/// ranked higher relative to other candidate types.
///
/// The boost is derived from the ratio of `remove-low-impact` success rate to the
/// overall baseline: 21.5% / 10.7% ≈ 2.0, dampened with square-root to 1.41,
/// then rounded to 1.5 for conservatism.
///
/// ## Valid Range
/// Must be >= 1.0 (boost) and <= 3.0 (avoid over-biasing).
pub const REMOVAL_CANDIDATE_BOOST: f32 = 1.5;

// =============================================================================
// Prediction Calibration Scaling (Issue #891)
// =============================================================================

// GRQ-sampler discovery cache reveals that `expectedCreatureScoreGain` overestimates
// actual outcomes by 100–10,000×. The magnitude of overestimation varies by candidate
// type, making cross-type comparisons unreliable. These per-type calibration factors
// are applied post-pessimism-discount to correct the systematic magnitude gap.
//
// Cache Evidence (GRQ-sampler):
//
// | Candidate Type | Predicted Gain    | Actual Gain       | Overestimation |
// |----------------|-------------------|-------------------|----------------|
// | Add-Neuron     | 0.003–0.01        | 1e-7 to 3e-6     | 100–10,000×    |
// | Add-Synapse    | 0.001–0.01        | (often negative)  | ~1,000×+       |
// | Coordinated    | 0.001–0.01        | ~2.2e-14          | ~10,000×+      |
//
// Calibration factors are conservative (slightly under-correcting) to avoid
// suppressing genuinely strong candidates. The pessimism discount already handles
// ratio-based corrections; these factors address the residual magnitude gap.

/// Prediction calibration factor for synapse candidates (Issue #891).
///
/// Synapse predictions overestimate actual score gains by approximately 1,000×.
/// Applied as a multiplicative factor to `expected_creature_score_gain` after
/// pessimism discounting and type-specific boosts.
///
/// ## Derivation
/// Actual/predicted ratio from cache evidence: ~0.001 (range 0.0001–0.01).
/// Conservative choice: 0.001 (corrects the median overestimation without
/// over-correcting edge cases).
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values above 0.01 provide insufficient correction.
/// Values below 0.0001 risk suppressing all synapse candidates.
pub const SYNAPSE_PREDICTION_CALIBRATION: f32 = 0.001;

/// Prediction calibration factor for neuron candidates (Issue #891).
///
/// Neuron predictions overestimate actual score gains by approximately 100×.
/// The overestimation is less severe than synapses because neuron candidates
/// involve more direct structural changes.
///
/// ## Derivation
/// Actual/predicted ratio from cache evidence: ~0.01 (range 0.001–0.1).
/// Conservative choice: 0.01 (corrects the median overestimation).
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values above 0.1 provide insufficient correction.
/// Values below 0.001 risk suppressing all neuron candidates.
pub const NEURON_PREDICTION_CALIBRATION: f32 = 0.01;

/// Prediction calibration factor for coordinated-structural candidates (Issue #891).
///
/// Coordinated predictions have the most severe overestimation (~10,000×)
/// because multi-operation predictions compound optimistically. Successful
/// coordinated candidates achieve only ~2.2e-14 actual gain despite
/// predictions in the 0.001–0.01 range.
///
/// ## Derivation
/// Actual/predicted ratio from cache evidence: ~0.0001 (range 0.00001–0.001).
/// Conservative choice: 0.0001 (corrects the median overestimation).
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values above 0.001 provide insufficient correction.
/// Values below 0.00001 risk suppressing all coordinated candidates.
pub const COORDINATED_PREDICTION_CALIBRATION: f32 = 0.0001;
