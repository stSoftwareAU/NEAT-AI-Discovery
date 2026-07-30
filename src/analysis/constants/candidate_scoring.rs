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
// Per-Target Add-Neuron Cap (Issue #1140)
// =============================================================================

/// Default maximum number of `add-neurons` candidates returned per target
/// neuron in a single discovery batch (Issue #1140).
///
/// Production discovery-cache analysis showed 17 of 19 `add-neurons` failures in one
/// submission targeted the same neuron, differing only by source input. All 17
/// failed. The cross-batch cooldown (Issue #1130) cannot fire within a single
/// batch, so a per-target within-batch cap is required to avoid wasting budget
/// on clearly-hopeless targets.
///
/// Overridable via the `NEAT_AI_DISCOVERY_MAX_ADD_NEURON_PER_TARGET`
/// environment variable.
///
/// ## Valid Range
/// Must be >= 1. Values above ~10 defeat the purpose of the cap.
pub const MAX_ADD_NEURON_CANDIDATES_PER_TARGET: usize = 3;

/// Minimum permitted per-target cap after env-var override clamping.
pub const MIN_ADD_NEURON_CANDIDATES_PER_TARGET: usize = 1;

/// Maximum permitted per-target cap after env-var override clamping.
pub const MAX_ADD_NEURON_CANDIDATES_PER_TARGET_CEILING: usize = 32;

/// Return the effective per-target add-neuron cap (Issue #1140).
///
/// Reads `NEAT_AI_DISCOVERY_MAX_ADD_NEURON_PER_TARGET` at call time so tests
/// can override the default. Values outside the permitted range are clamped
/// to `[MIN_ADD_NEURON_CANDIDATES_PER_TARGET,
/// MAX_ADD_NEURON_CANDIDATES_PER_TARGET_CEILING]`. Unparsable or missing
/// values fall back to `MAX_ADD_NEURON_CANDIDATES_PER_TARGET`.
#[must_use]
pub fn max_add_neuron_candidates_per_target() -> usize {
    std::env::var("NEAT_AI_DISCOVERY_MAX_ADD_NEURON_PER_TARGET")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(MAX_ADD_NEURON_CANDIDATES_PER_TARGET)
        .clamp(
            MIN_ADD_NEURON_CANDIDATES_PER_TARGET,
            MAX_ADD_NEURON_CANDIDATES_PER_TARGET_CEILING,
        )
}

// =============================================================================
// Per-Final-Target Coordinated-Structural Cap (Issue #1271)
// =============================================================================

/// Default maximum number of coordinated-structural candidates returned per
/// final-operation target neuron in a single discovery batch (Issue #1271).
///
/// Production discovery-cache analysis showed 41 consecutive
/// coordinated-structural failures whose final operation all targeted the same
/// output neuron `533d8616-037c-4278-b95c-3a2a1ce15ee6`. A single problematic
/// target consumed the entire coordinated-structural budget while other targets
/// in the creature went unexplored. This cap mirrors the per-target add-neuron
/// cap (Issue #1140) for the coordinated-structural pipeline.
///
/// The cap operates earlier in the pipeline than the cross-target diversity
/// spread (`MIN_DISTINCT_TARGETS_PER_BATCH`, Issue #1193): it constrains the
/// candidate pool *before* any diversity-aware reordering downstream.
///
/// Overridable via the `NEAT_AI_DISCOVERY_MAX_COORDINATED_PER_TARGET`
/// environment variable.
///
/// ## Valid Range
/// Must be >= 1. Values above ~10 defeat the purpose of the cap.
pub const MAX_COORDINATED_PER_TARGET_OUTPUT: usize = 3;

/// Minimum permitted per-final-target coordinated cap after env-var override
/// clamping.
pub const MIN_COORDINATED_PER_TARGET_OUTPUT: usize = 1;

/// Maximum permitted per-final-target coordinated cap after env-var override
/// clamping.
pub const MAX_COORDINATED_PER_TARGET_OUTPUT_CEILING: usize = 32;

/// Return the effective per-final-target coordinated-structural cap
/// (Issue #1271).
///
/// Reads `NEAT_AI_DISCOVERY_MAX_COORDINATED_PER_TARGET` at call time so tests
/// can override the default. Values outside the permitted range are clamped to
/// `[MIN_COORDINATED_PER_TARGET_OUTPUT,
/// MAX_COORDINATED_PER_TARGET_OUTPUT_CEILING]`. Unparsable or missing values
/// fall back to `MAX_COORDINATED_PER_TARGET_OUTPUT`.
#[must_use]
pub fn max_coordinated_per_target_output() -> usize {
    std::env::var("NEAT_AI_DISCOVERY_MAX_COORDINATED_PER_TARGET")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(MAX_COORDINATED_PER_TARGET_OUTPUT)
        .clamp(
            MIN_COORDINATED_PER_TARGET_OUTPUT,
            MAX_COORDINATED_PER_TARGET_OUTPUT_CEILING,
        )
}

// =============================================================================
// Cross-Target Diversity Spread (Issue #1193)
// =============================================================================

/// Default minimum number of distinct target neurons that an emitted batch
/// must include when the candidate pool supports it (Issue #1193).
///
/// The per-target cap (Issue #1140) only fires once three slots for one target
/// have already been consumed. When the top of the gain-sorted list is
/// dominated by candidates against a single problematic target, the cap saves
/// nothing — every slot is spent on that target before any other is
/// considered. Reordering the top of the list to cover at least
/// `MIN_DISTINCT_TARGETS_PER_BATCH` distinct targets first keeps a single
/// risky neuron from monopolising the budget while still letting the cap
/// admit up to three candidates per target where alternatives are scarce.
///
/// Overridable via the `NEAT_AI_DISCOVERY_MIN_DISTINCT_TARGETS_PER_BATCH`
/// environment variable.
///
/// ## Valid Range
/// Must be >= 1. Values above ~16 may starve high-gain targets.
pub const MIN_DISTINCT_TARGETS_PER_BATCH: usize = 3;

/// Minimum permitted distinct-target spread after env-var override clamping.
pub const MIN_DISTINCT_TARGETS_PER_BATCH_FLOOR: usize = 1;

/// Maximum permitted distinct-target spread after env-var override clamping.
pub const MIN_DISTINCT_TARGETS_PER_BATCH_CEILING: usize = 32;

/// Return the effective minimum distinct-target spread (Issue #1193).
///
/// Reads `NEAT_AI_DISCOVERY_MIN_DISTINCT_TARGETS_PER_BATCH` at call time so
/// tests and operators can override the default. Values outside the permitted
/// range are clamped to `[MIN_DISTINCT_TARGETS_PER_BATCH_FLOOR,
/// MIN_DISTINCT_TARGETS_PER_BATCH_CEILING]`. Unparsable or missing values
/// fall back to `MIN_DISTINCT_TARGETS_PER_BATCH`.
#[must_use]
pub fn min_distinct_targets_per_batch() -> usize {
    std::env::var("NEAT_AI_DISCOVERY_MIN_DISTINCT_TARGETS_PER_BATCH")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(MIN_DISTINCT_TARGETS_PER_BATCH)
        .clamp(
            MIN_DISTINCT_TARGETS_PER_BATCH_FLOOR,
            MIN_DISTINCT_TARGETS_PER_BATCH_CEILING,
        )
}

// =============================================================================
// Source-Type Scoring (Issue #465)
// =============================================================================

/// Scoring boost multiplier for candidates from input-neuron sources.
///
/// Production discovery-cache analysis shows input neurons as synapse sources have a 36.2%
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
// Module Gating (Issue #1060)
// =============================================================================

/// Success rate threshold below which a module is gated (skipped entirely).
///
/// When a module's Bayesian success rate drops below this threshold and the
/// module has at least `MIN_BOOST_SAMPLES` attempts, candidate generation for
/// that module is skipped entirely to save compute.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values above 0.05 may gate modules too eagerly.
pub const MODULE_GATE_THRESHOLD: f64 = 0.005;

/// Weight applied to pre-filtering failures when recording soft failures
/// in the `ModuleOutcomeTracker` (Issue #1060).
///
/// Candidates filtered out during post-processing (e.g., below threshold,
/// deduplicated, budget exceeded) are recorded as failures with this weight
/// relative to a real ablation failure (weight 1.0).
///
/// ## Valid Range
/// Must be in (0.0, 1.0]. Values close to 1.0 make pre-filtering failures
/// nearly as impactful as real ablation failures.
pub const SOFT_FAILURE_WEIGHT: f64 = 0.5;

// =============================================================================
// Quality-Based Module Skipping (Issue #1074)
// =============================================================================

/// Minimum expected gain for a candidate to be considered "high quality"
/// during quality-based module skipping.
///
/// During the sequential merge phase, if the accumulated candidates already
/// contain at least [`QUALITY_SKIP_MIN_CANDIDATES`] candidates whose
/// `expected_creature_score_gain` exceeds this threshold, remaining
/// lower-priority modules are skipped.
///
/// ## Valid Range
/// Must be > 0.0. Values above 1.0 may be too aggressive and skip useful
/// modules.
pub const QUALITY_SKIP_GAIN_THRESHOLD: f32 = 0.01;

/// Minimum number of high-quality candidates required before module skipping
/// is triggered.
///
/// Quality-based skipping only activates when this many candidates with
/// gain above [`QUALITY_SKIP_GAIN_THRESHOLD`] have been accumulated during
/// the merge phase. This prevents premature skipping when only a few
/// candidates have been found.
///
/// ## Valid Range
/// Must be >= 1. Values above 50 may prevent skipping from ever triggering.
pub const QUALITY_SKIP_MIN_CANDIDATES: usize = 15;

// =============================================================================
// Target-Type Scoring (Issue #468)
// =============================================================================

/// Scoring boost multiplier for candidates targeting existing hidden neurons.
///
/// Production discovery-cache analysis shows existing hidden neurons as targets have a 31.4%
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
// The production discovery cache reveals dramatic differences in success rates by
// activation function. These multipliers are derived from Bayesian-smoothed success
// rates (Beta posterior with prior centred on the baseline ~13.9% success rate,
// K=20 pseudo-observations) to handle small sample sizes.
//
// Issue #909 recalibration: IDENTITY's 14.9% raw success rate is inflated because
// IDENTITY candidates dominate the candidate pool (274 total — more than any other
// activation). Per-candidate success is mediocre, and production discovery-cache
// evidence shows IDENTITY neurons are frequently substituted with non-linear
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
// Cache Evidence (production discovery cache, with Issue #909 normalisation):
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
// * SINE boost estimated from substitution evidence
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
/// SINE is added based on production discovery-cache evidence showing it successfully
/// substitutes IDENTITY neurons. The modest boost encourages
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
/// candidates — more than any other activation). Production discovery-cache
/// evidence shows IDENTITY neurons are frequently substituted with non-linear
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
/// multipliers derived from production discovery-cache success rates. Unknown activations
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
// Risky Target-Squash Cold-Start Prior (Issue #1192)
// =============================================================================

/// Non-invertible / periodic target activations that warrant a conservative
/// cold-start calibration prior (Issue #1192).
///
/// `src/activations.rs` documents these as non-invertible: a small change in
/// the incoming weighted sum can flip the output direction entirely. That
/// makes any add-neuron / add-synapse candidate aimed at a target with one of
/// these squashes substantially riskier than a candidate aimed at a monotone
/// target (`ReLU` family, Sigmoid, Tanh).
///
/// The per-(`change_type`, `target_squash`) calibration introduced in
/// Issue #1162 only kicks in once at least three failure-cache entries exist
/// for a key. Before that threshold is reached, the lookup falls back to the
/// per-`change_type` value (or the global neutral 1.0 default), which is too
/// optimistic for these activations. `RISKY_TARGET_SQUASHES` lets the cold-
/// start path use a conservative prior instead — see [`risky_squash_prior`].
pub const RISKY_TARGET_SQUASHES: &[&str] = &["SINE", "COSINE", "GAUSSIAN", "SQUARE", "ABSOLUTE"];

/// Default conservative cold-start calibration prior for risky target squashes
/// (Issue #1192).
///
/// Applied only while the per-(`change_type`, `target_squash`) bucket has
/// fewer than three usable failure samples. Once the bucket reaches the
/// warmup threshold, the learnt EWMA value takes over.
///
/// ## Valid Range
/// Must be in `[MIN_RISKY_SQUASH_PRIOR, MAX_RISKY_SQUASH_PRIOR]`.
pub const RISKY_SQUASH_PRIOR_DEFAULT: f32 = 0.25;

/// Lower clamp for the risky-squash cold-start prior (Issue #1192).
///
/// Matches the existing `MIN_CALIBRATION_CORRECTION` floor used for learnt
/// corrections so the cold-start prior cannot collapse predictions to zero.
pub const MIN_RISKY_SQUASH_PRIOR: f32 = 0.001;

/// Upper clamp for the risky-squash cold-start prior (Issue #1192).
///
/// Matches `NEUTRAL_CORRECTION` — the prior must never inflate the base
/// calibration constant beyond the compiled value.
pub const MAX_RISKY_SQUASH_PRIOR: f32 = 1.0;

/// Returns true when the supplied target squash name is in
/// [`RISKY_TARGET_SQUASHES`] (Issue #1192).
///
/// Comparison is case-sensitive; callers should pass the canonical
/// upper-case squash name as it appears in the failure-cache JSON.
#[must_use]
pub fn is_risky_target_squash(squash: &str) -> bool {
    RISKY_TARGET_SQUASHES.contains(&squash)
}

/// Returns the effective risky-squash cold-start prior (Issue #1192).
///
/// Reads `NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR` at call time so tests and
/// operators can override the default without recompiling. Values outside
/// `[MIN_RISKY_SQUASH_PRIOR, MAX_RISKY_SQUASH_PRIOR]` are clamped. Unparsable,
/// non-finite, or missing values fall back to [`RISKY_SQUASH_PRIOR_DEFAULT`].
#[must_use]
pub fn risky_squash_prior() -> f32 {
    std::env::var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(RISKY_SQUASH_PRIOR_DEFAULT)
        .clamp(MIN_RISKY_SQUASH_PRIOR, MAX_RISKY_SQUASH_PRIOR)
}

// =============================================================================
// Saturation-Aware Prediction Discount (Issue #1112)
// =============================================================================

/// Discount floor for predictions targeting near-saturated neurons (Issue #1112).
///
/// When a target neuron is operating near its activation saturation bounds
/// (e.g., `HARD_TANH` at [-1, 1]), its output physically cannot move much in
/// response to small perturbations, so predictions are heavily over-estimated.
///
/// Production data shows predictions of ~0.01 for neuron candidates targeting
/// `HARD_TANH` at full range, while actual results are ~-0.00005 (negative —
/// making things worse). This discount is applied multiplicatively after
/// pessimism discounting and before logistic calibration.
///
/// The discount interpolates linearly from 1.0 (at the saturation threshold)
/// down to this floor (at full saturation, factor = 1.0).
///
/// ## Valid Range
/// Must be in (0.0, 0.5). Values below 0.05 risk zeroing-out all saturated
/// candidates. Values above 0.5 provide insufficient correction.
pub const SATURATION_DISCOUNT_AGGRESSIVE: f32 = 0.15;

// =============================================================================
// Pessimism Discount (Issue #506)
// =============================================================================

/// Minimum pessimism discount applied to all score predictions.
///
/// Production discovery-cache analysis (creature b2ff6e45) showed
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

/// Minimum pessimism discount floor for neuron candidates (Issue #791, #1056).
///
/// Production discovery-cache analysis shows add-neurons has a ~2.7% actual success rate
/// (28/1028 in latest cache) — far lower than predicted. The previous floor
/// of 0.10 was calibrated against a 15% success estimate (Issue #787) which
/// proved to be overestimated when measured against production outcomes.
///
/// A lower floor applies more aggressive base discounting to neuron predictions,
/// reducing the expected gain for candidates with few samples improving.
///
/// ## Valid Range
/// Must be in (0.0, `PESSIMISM_DISCOUNT_FLOOR`). Values below 0.05 risk
/// zeroing-out legitimate neuron candidates.
pub const NEURON_PESSIMISM_DISCOUNT_FLOOR: f32 = 0.08;

/// Exponent for the neuron-specific pessimism discount curve (Issue #791, #1056).
///
/// A higher exponent (closer to linear) produces a less forgiving curve at
/// moderate ratios compared to the generic exponent (0.6). Updated from 0.75
/// to 0.80 based on production discovery-cache data showing ~2.7% actual success rate
/// (28/1028), indicating moderate improved ratios are even less reliable
/// predictors of creature-level success than previously assumed.
///
/// With exponent 0.80 and floor 0.08 (discount = 0.08 + 0.92 × ratio^0.80):
/// - ratio 0.1 → 0.1^0.80 ≈ 0.158 → discount ≈ 0.226
/// - ratio 0.4 → 0.4^0.80 ≈ 0.476 → discount ≈ 0.518
/// - ratio 0.7 → 0.7^0.80 ≈ 0.745 → discount ≈ 0.765
/// - ratio 1.0 → 1.0       → discount = 1.000
///
/// ## Valid Range
/// Must be in (`PESSIMISM_CURVE_EXPONENT`, 1.0]. Values above 0.9 give
/// near-linear behaviour.
pub const NEURON_PESSIMISM_CURVE_EXPONENT: f32 = 0.80;

// =============================================================================
// Synapse-Specific Pessimism Discount (Issue #789)
// =============================================================================

/// Minimum pessimism discount floor for synapse candidates (Issue #789, #1056).
///
/// Production discovery-cache analysis shows add-synapses has a ~0.1% actual success rate
/// (3/1001 in latest cache). The multi-weight search creates selection bias
/// that overfits to sample data, contributing to massive overestimation.
/// Updated from 0.05 to 0.03 to match production reality.
///
/// ## Valid Range
/// Must be in (0.0, `NEURON_PESSIMISM_DISCOUNT_FLOOR`]. Values below 0.02 risk
/// zeroing-out all synapse candidates.
pub const SYNAPSE_PESSIMISM_DISCOUNT_FLOOR: f32 = 0.03;

/// Exponent for the synapse-specific pessimism discount curve (Issue #789, #1056).
///
/// A higher exponent (closer to linear) produces a less forgiving curve at
/// moderate ratios. With a ~0.1% success rate (3/1001), synapse candidates
/// need the most aggressive discounting of all candidate types. The
/// multi-weight search creates selection bias where the best weight for
/// the sample data does not generalise to the full evaluation.
///
/// Updated from 0.85 to 0.90 to match production data.
///
/// With exponent 0.90 and floor 0.03 (discount = 0.03 + 0.97 × ratio^0.90):
/// - ratio 0.1 → 0.1^0.90 ≈ 0.126 → discount ≈ 0.152
/// - ratio 0.4 → 0.4^0.90 ≈ 0.427 → discount ≈ 0.444
/// - ratio 0.6 → 0.6^0.90 ≈ 0.621 → discount ≈ 0.632
/// - ratio 0.7 → 0.7^0.90 ≈ 0.723 → discount ≈ 0.731
/// - ratio 1.0 → 1.0       → discount = 1.000
///
/// ## Valid Range
/// Must be in (`NEURON_PESSIMISM_CURVE_EXPONENT`, 1.0]. Values above 0.95 give
/// near-linear behaviour.
pub const SYNAPSE_PESSIMISM_CURVE_EXPONENT: f32 = 0.90;

// =============================================================================
// Metropolis-Hastings Temperature (Issue #1018)
// =============================================================================

/// Default temperature for Metropolis-Hastings probabilistic acceptance.
///
/// Controls the exploration-exploitation trade-off when evaluating marginal
/// synapse candidates (those with 0 < improvement ≤ threshold). Higher
/// temperatures accept more marginal candidates; lower temperatures are
/// more selective.
///
/// The acceptance probability for a marginal candidate is:
///
/// ```text
/// p = min(1, exp(improvement / temperature))
/// ```
///
/// At the default value of 0.01, a candidate with improvement = 0.005
/// (half the typical threshold) has acceptance probability ≈ 0.607.
///
/// This feature is gated behind the `NEAT_AI_DISCOVERY_MH_TEMPERATURE`
/// environment variable. When the variable is unset, deterministic
/// threshold-based acceptance is used (preserving existing behaviour).
///
/// ## Valid Range
/// Must be > 0.0. Values below 0.001 make acceptance near-deterministic.
/// Values above 0.1 accept nearly all marginal candidates.
pub const DEFAULT_MH_TEMPERATURE: f32 = 0.01;

// =============================================================================
// Adaptive Proposal Distribution (Issue #1019)
// =============================================================================

/// Default standard deviation (σ) for the Gaussian proposal distribution.
///
/// Controls the initial spread of proposed weight candidates around the
/// computed optimal weight. A larger σ explores more broadly; a smaller σ
/// focuses proposals near the optimum.
///
/// ## Valid Range
/// Must be > 0.0. Values below 0.1 may under-explore. Values above 2.0
/// may waste evaluations on extreme weights.
pub const ADAPTIVE_PROPOSAL_INITIAL_SIGMA: f32 = 0.5;

/// Number of weight candidates to sample from the adaptive proposal distribution.
///
/// Replaces the fixed 9-variant grid. More candidates improve coverage of the
/// weight space at the cost of additional computation per target.
///
/// ## Valid Range
/// Must be >= 3 to provide meaningful exploration. Values above 32 provide
/// diminishing returns.
pub const ADAPTIVE_PROPOSAL_CANDIDATE_COUNT: usize = 12;

/// Minimum number of historical outcomes before using adaptive σ.
///
/// Below this threshold, the acceptance rate estimate is too noisy to
/// adapt σ reliably. The system falls back to the fixed grid when
/// insufficient data is available.
///
/// ## Valid Range
/// Must be >= 5 to avoid noise and <= 100 to be responsive.
pub const ADAPTIVE_PROPOSAL_MIN_HISTORY: usize = 15;

/// Target acceptance rate for the adaptive σ controller.
///
/// When the observed acceptance rate exceeds this target, σ is decreased
/// (focus around the current optimum). When below, σ is increased (explore
/// more broadly). Derived from optimal acceptance rates for
/// Metropolis-Hastings on unimodal targets (~0.234 for high-dimensional,
/// ~0.44 for one-dimensional).
///
/// ## Valid Range
/// Must be in (0.1, 0.8). Values outside this range cause σ to diverge.
pub const ADAPTIVE_PROPOSAL_TARGET_ACCEPTANCE: f32 = 0.35;

/// Multiplicative adaptation rate for σ adjustment.
///
/// When acceptance is too high, σ is multiplied by `1.0 / rate` (shrink).
/// When acceptance is too low, σ is multiplied by `rate` (grow).
/// Applied once per evaluation batch.
///
/// ## Valid Range
/// Must be in (1.0, 2.0). Values near 1.0 adapt slowly; values near 2.0
/// cause oscillation.
pub const ADAPTIVE_PROPOSAL_ADAPTATION_RATE: f32 = 1.2;

/// Minimum σ to prevent the proposal distribution from collapsing.
///
/// ## Valid Range
/// Must be > 0.0 and < `ADAPTIVE_PROPOSAL_INITIAL_SIGMA`.
pub const ADAPTIVE_PROPOSAL_MIN_SIGMA: f32 = 0.05;

/// Maximum σ to prevent the proposal distribution from becoming too broad.
///
/// ## Valid Range
/// Must be > `ADAPTIVE_PROPOSAL_INITIAL_SIGMA`.
pub const ADAPTIVE_PROPOSAL_MAX_SIGMA: f32 = 3.0;

/// Probability of proposing a sign-flipped (negative) weight.
///
/// Maintains exploration of negative weights that might be missed by
/// a Gaussian centred on the positive optimal weight.
///
/// ## Valid Range
/// Must be in (0.0, 0.5). Values above 0.3 waste too many candidates
/// on unlikely negative weights.
pub const ADAPTIVE_PROPOSAL_SIGN_FLIP_PROBABILITY: f32 = 0.15;

// =============================================================================
// Absolute Expected-Gain Floor for Add-Neuron / Add-Synapse Candidates (Issue #1191)
// =============================================================================

/// Absolute minimum `expected_creature_score_gain` for add-neuron and
/// add-synapse candidates emitted from post-processing (Issue #1191).
///
/// The previous filter only required `> 0.0`, which kept candidates with
/// gains as small as ~6e-7 (see Issue #1189 failure cache). At those
/// magnitudes the predicted improvement is dominated by floating-point
/// round-off in the downstream evaluator, so testing them is wasted budget
/// regardless of the prediction model's accuracy.
///
/// This floor is applied in:
/// - `build_neuron_results()` before the per-target / per-squash diversity
///   filters, so noise-level candidates do not consume the cap budget.
/// - `apply_post_processing()` for synapse helpful/harmful candidates, in
///   place of the prior `> 0.0` retain.
///
/// Coordinated-structural candidates have their own floor
/// ([`COORDINATED_MIN_EXPECTED_GAIN`]) and post-discount noise floor
/// ([`COORDINATED_POST_DISCOUNT_NOISE_FLOOR`]) applied downstream — those
/// remain authoritative for that path.
///
/// Overridable via the `NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN` environment
/// variable.
///
/// ## Scale — this value is denominated in the *pre-calibration* prediction
/// scale (Issue #1778)
///
/// This is the noise screen on the **raw prediction**: the smallest creature
/// error reduction a candidate can predict before that prediction is dominated
/// by round-off in the downstream evaluator. It is *not* denominated in the
/// realised creature-score-delta scale that
/// `expected_creature_score_gain` carries after
/// [`NEURON_PREDICTION_CALIBRATION`] / [`SYNAPSE_PREDICTION_CALIBRATION`] have
/// been applied.
///
/// Comparing it directly against a post-calibration gain — as the filters did
/// before Issue #1778 — silently multiplied the screen's strictness by
/// `1 / calibration`, i.e. 333× for add-neuron and 3333× for add-synapse, so no
/// candidate the pipeline can produce could clear it. Call
/// [`min_expected_gain_floor_for_neurons`] /
/// [`min_expected_gain_floor_for_synapses`] at a post-calibration comparison
/// site instead; they convert this value into the calibrated scale.
///
/// ## Valid Range
/// Must be > 0.0. Values above 1e-3 may filter genuinely useful candidates.
/// Values below 1e-8 defeat the purpose of the floor.
pub const MIN_EXPECTED_CREATURE_SCORE_GAIN: f32 = 1e-5;

/// Lower clamp for the configurable `MIN_EXPECTED_CREATURE_SCORE_GAIN`
/// override (Issue #1191).
///
/// `0.0` is accepted as a valid "disable the floor" value for tests that
/// exercise the pre-Issue-#1191 contract at tiny magnitudes.
pub const MIN_EXPECTED_CREATURE_SCORE_GAIN_FLOOR: f32 = 0.0;

/// Upper clamp for the configurable `MIN_EXPECTED_CREATURE_SCORE_GAIN`
/// override (Issue #1191).
pub const MIN_EXPECTED_CREATURE_SCORE_GAIN_CEILING: f32 = 1e-2;

/// Returns the effective minimum-expected-gain floor for add-neuron and
/// add-synapse candidates (Issue #1191).
///
/// Reads `NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN` at call time so tests and
/// operators can override the default without recompiling. Values outside
/// `[MIN_EXPECTED_CREATURE_SCORE_GAIN_FLOOR,
/// MIN_EXPECTED_CREATURE_SCORE_GAIN_CEILING]` are clamped. Unparsable,
/// non-finite, or negative values fall back to
/// [`MIN_EXPECTED_CREATURE_SCORE_GAIN`].
#[must_use]
pub fn min_expected_creature_score_gain() -> f32 {
    std::env::var("NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(MIN_EXPECTED_CREATURE_SCORE_GAIN)
        .clamp(
            MIN_EXPECTED_CREATURE_SCORE_GAIN_FLOOR,
            MIN_EXPECTED_CREATURE_SCORE_GAIN_CEILING,
        )
}

/// Absolute backstop under the calibrated gain floor (Issue #1778).
///
/// [`calibrated_gain_floor`] converts the pre-calibration noise screen into the
/// calibrated scale by multiplying by the per-type calibration constant. This
/// backstop stops that conversion from ever opening the screen wider than the
/// band in which the estimate is known to be uncorrelated with the outcome.
///
/// Evidence (`docs/analysis/candidate-rate-diagnosis-1777.md`): the production
/// coordinated `change-squash` that estimated `4.17e-10` realised `-8.65e-4` —
/// an actively harmful change whose estimate carried no usable signal, and the
/// exact false acceptance the Issue #1740 guard tests exist to prevent. `1e-9`
/// sits above that whole collapsed band while remaining two orders below the
/// smallest realised *accepted* delta (`1.95e-7`), so it rejects noise without
/// touching the achievable band.
///
/// ## Valid Range
/// Must be > 0.0 and well below the smallest realised accepted delta
/// (`~1.95e-7`). Values above 1e-7 would re-reject the achievable band.
pub const GAIN_FLOOR_NOISE_BACKSTOP: f32 = 1e-9;

/// Convert the pre-calibration noise screen into the calibrated scale
/// (Issue #1778).
///
/// `expected_creature_score_gain` reaches the add-neuron / add-synapse floor
/// filters *after* the fixed per-type calibration constant has rescaled it from
/// the prediction scale into the realised creature-score-delta scale. The
/// screen itself ([`min_expected_creature_score_gain`]) is denominated in the
/// prediction scale, so it must be rescaled the same way before the comparison
/// is meaningful.
///
/// Only the **fixed, type-level** calibration constant is divided out here. The
/// per-creature calibration *correction* (`calibration_correction.rs`, Issue
/// #1131) stays on the candidate side deliberately: it is evidence that *this
/// creature's* predictions over-shoot, so tightening acceptance in response is
/// the intended behaviour. The base constant is not evidence about a candidate
/// at all — it differs 10× between add-neuron and add-synapse purely by
/// candidate *type*, and leaving it only on the candidate side made the floor
/// 10× stricter for synapses for no reason connected to their merit.
///
/// Returns `0.0` when the screen is configured off (`0.0` is the documented
/// "disable the floor" override). A `calibration` outside `(0.0, 1.0]` is not a
/// units conversion, so the unconverted screen is returned rather than a
/// silently widened one.
#[must_use]
pub fn calibrated_gain_floor(calibration: f32) -> f32 {
    let configured = min_expected_creature_score_gain();
    if configured <= 0.0 {
        return 0.0;
    }
    if !calibration.is_finite() || calibration <= 0.0 || calibration > 1.0 {
        return configured;
    }
    (configured * calibration).max(GAIN_FLOOR_NOISE_BACKSTOP)
}

/// Effective post-calibration expected-gain floor for add-neuron candidates
/// (Issue #1778).
///
/// [`calibrated_gain_floor`] applied to [`NEURON_PREDICTION_CALIBRATION`]:
/// `1e-5 × 0.003 = 3e-8` at the default screen.
#[must_use]
pub fn min_expected_gain_floor_for_neurons() -> f32 {
    calibrated_gain_floor(NEURON_PREDICTION_CALIBRATION)
}

/// Effective post-calibration expected-gain floor for add-synapse candidates
/// (Issue #1778).
///
/// [`calibrated_gain_floor`] applied to [`SYNAPSE_PREDICTION_CALIBRATION`]:
/// `1e-5 × 0.0003 = 3e-9` at the default screen.
#[must_use]
pub fn min_expected_gain_floor_for_synapses() -> f32 {
    calibrated_gain_floor(SYNAPSE_PREDICTION_CALIBRATION)
}

// =============================================================================
// Bypass-Weight Floor for Hidden-Neuron Collapse (Issue #1270)
// =============================================================================

/// Minimum absolute bypass-synapse weight required to emit a 1-in/1-out hidden
/// neuron collapse candidate (Issue #1270).
///
/// `detect_collapsible_hidden_neurons` proposes a 4-op coordinated structural
/// change (`removeSynapse(a→h)`, `removeSynapse(h→b)`, `removeNeuron(h)`,
/// `addSynapse(a→b, weight=w)`). When `|w|` is below this floor the chain
/// `a→h→b` was contributing essentially nothing through `h`, so the candidate
/// is functionally equivalent to a 1-op `remove-neuron` but still carries the
/// much higher implementation-risk profile of a 4-op coordinated change.
///
/// Production discovery-cache analysis captured 41 consecutive
/// such failures: a bypass weight of `0.0021` produced an actual error
/// reduction of `-0.0033` (~6,500× worse than predicted), and a bypass weight
/// of `-0.000028` produced `-0.00083` (~1,600× worse). The
/// `COORDINATED_MIN_EXPECTED_GAIN` (#1110) and per-op
/// `COORDINATED_POST_DISCOUNT_NOISE_FLOOR_*` (#1128, #1272) floors filter on
/// predicted gain; this floor targets the orthogonal failure mode where the
/// bypass weight itself signals the collapse is risky regardless of predicted
/// gain.
///
/// Overridable via the `NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE`
/// environment variable.
///
/// ## Valid Range
/// Must be >= 0.0. Values above 0.1 may filter genuinely useful collapses.
pub const MIN_BYPASS_WEIGHT_FOR_COLLAPSE: f32 = 0.01;

/// Lower clamp for the `MIN_BYPASS_WEIGHT_FOR_COLLAPSE` env-var override
/// (Issue #1270). `0.0` is accepted as a "disable the floor" value for tests
/// that exercise the pre-#1270 contract at tiny magnitudes.
pub const MIN_BYPASS_WEIGHT_FOR_COLLAPSE_FLOOR: f32 = 0.0;

/// Upper clamp for the `MIN_BYPASS_WEIGHT_FOR_COLLAPSE` env-var override
/// (Issue #1270).
pub const MIN_BYPASS_WEIGHT_FOR_COLLAPSE_CEILING: f32 = 0.1;

/// Returns the effective minimum absolute bypass weight required for the
/// 1-in/1-out hidden neuron collapse candidate (Issue #1270).
///
/// Reads `NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE` at call time so
/// tests and operators can override the default without recompiling. Values
/// outside `[MIN_BYPASS_WEIGHT_FOR_COLLAPSE_FLOOR,
/// MIN_BYPASS_WEIGHT_FOR_COLLAPSE_CEILING]` are clamped. Unparsable,
/// non-finite, or negative values fall back to
/// [`MIN_BYPASS_WEIGHT_FOR_COLLAPSE`].
#[must_use]
pub fn min_bypass_weight_for_collapse() -> f32 {
    std::env::var("NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(MIN_BYPASS_WEIGHT_FOR_COLLAPSE)
        .clamp(
            MIN_BYPASS_WEIGHT_FOR_COLLAPSE_FLOOR,
            MIN_BYPASS_WEIGHT_FOR_COLLAPSE_CEILING,
        )
}

// =============================================================================
// Coordinated Candidate Minimum Expected Gain (Issue #1110)
// =============================================================================

/// Minimum expected-gain floor for coordinated structural candidates (Issue #1110).
///
/// Production discovery-cache failure data shows coordinated
/// structural candidates with `expectedCreatureScoreGain` of ~8e-8 and ~4e-8
/// (after 0.2× weight scaling) that produced actual error changes of -0.0008
/// and -0.0004 respectively — harming the network rather than helping.
///
/// Expected gains at 1e-8 to 1e-7 are indistinguishable from numerical noise
/// and should never be proposed. This floor filters noise-level proposals while
/// preserving genuinely promising candidates.
///
/// ## Valid Range
/// Must be > 0.0. Values above 1e-3 may filter too aggressively.
pub const COORDINATED_MIN_EXPECTED_GAIN: f32 = 1e-5;

// =============================================================================
// Per-Op-Count Post-Discount Noise Floors (Issue #1128, #1272)
// =============================================================================
//
// The post-discount noise floor was originally a single value (5e-7) applied
// to every coordinated-structural candidate regardless of operation count. The
// floor sits at the FFI boundary: candidates reach `analyze_all`'s final sweep
// with gains that have passed the pre-merge `COORDINATED_MIN_EXPECTED_GAIN`
// (1e-5) filter but may have been legitimately discounted by downstream steps
// — module-boost (minimum 0.5×), ensemble disagreement penalty (0.7×), per-op
// empirical discounting (`coordinated_empirical_discount`, #1058),
// synapse-analysis calibration (`COORDINATED_PREDICTION_CALIBRATION`, 5e-5×)
// and hidden-neuron impact discount (0.1×).
//
// Issue #1272 found that a single floor under-weights the implementation
// risk of higher-op candidates. Production discovery-cache analysis (creature
// `bcbca347`) captured two 4-op coordinated-structural failures that **just
// barely** cleared the 5e-7 floor yet harmed the network by 1000–6000× the
// predicted magnitude:
//
// | Failure entry         | ops | predicted | actual    |
// |-----------------------|-----|-----------|-----------|
// | `cdff1014…`           | 4   | 5.06e-7   | -3.29e-3  |
// | `113ec32a…`           | 4   | 5.21e-7   | -8.28e-4  |
//
// The `coordinated_empirical_discount()` lookup (#1058) already encodes that
// 4-op candidates have near-zero production success — that same evidence
// supports a stricter noise floor for the same op-count tiers. The per-tier
// floors below pair with the empirical-discount tiers from #1058:
//
//   1-op : 5e-7  (existing value preserved)
//   2-op : 1e-6
//   3-op : 2e-6
//   4+-op: 5e-6

/// Post-discount noise floor for 1-operation coordinated candidates
/// (Issue #1128, #1272).
///
/// Preserved at the original 5e-7 value — above the 1.17e-7 noise case
/// captured in Issue #1127 with a ~4× safety margin, but below the floor of
/// legitimately discounted collapse/pruning candidates whose post-calibration
/// gains sit just below 1e-6 (e.g. 9.95e-7 for the
/// `coordinated_structural_can_collapse_hidden_neuron_to_single_synapse`
/// regression fixture).
pub const COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP: f32 = 5e-7;

/// Post-discount noise floor for 2-operation coordinated candidates
/// (Issue #1272).
///
/// 2-op candidates carry compounding implementation risk over 1-op variants
/// — the 0.5× empirical discount (`COORDINATED_EMPIRICAL_DISCOUNT_2OPS`)
/// already encodes a ~halved production-success rate. The 1e-6 floor is
/// double the 1-op tier so the post-discount noise screen tightens in step
/// with the empirical-discount table.
pub const COORDINATED_POST_DISCOUNT_NOISE_FLOOR_2OPS: f32 = 1e-6;

/// Post-discount noise floor for 3-operation coordinated candidates
/// (Issue #1272).
///
/// 3-op candidates have substantially lower production-success rates than
/// 2-op (`COORDINATED_EMPIRICAL_DISCOUNT_3OPS` = 0.2). The 2e-6 floor sits
/// 4× above the 1-op tier — reflecting both the additional risk and the
/// scarcity of legitimately discounted 3-op candidates that hover in the
/// 1e-6 range.
pub const COORDINATED_POST_DISCOUNT_NOISE_FLOOR_3OPS: f32 = 2e-6;

/// Post-discount noise floor for 4+-operation coordinated candidates
/// (Issue #1272).
///
/// Production discovery-cache evidence (creature
/// `bcbca347`) captured 4-op coordinated candidates with predicted gains of
/// 5.06e-7 and 5.21e-7 — both **just barely** clearing the previous 5e-7
/// floor — that produced actual error changes of -3.29e-3 and -8.28e-4
/// respectively. The 5e-6 floor is 10× the 1-op tier and matches the
/// `COORDINATED_MIN_EXPECTED_GAIN` magnitude (1e-5) more closely, rejecting
/// the bcbca347 entries by an order of magnitude while still allowing
/// genuinely promising 4+-op candidates through.
pub const COORDINATED_POST_DISCOUNT_NOISE_FLOOR_4PLUS_OPS: f32 = 5e-6;

/// Return the per-op-count post-discount noise floor (Issue #1272).
///
/// Mirrors the tiering convention of [`coordinated_empirical_discount`]: an
/// `op_count` of `0` is treated as the single-op tier so callers do not need
/// to special-case empty operation lists.
///
/// The returned floor is multiplied by the optional
/// `NEAT_AI_DISCOVERY_COORDINATED_NOISE_FLOOR_MULTIPLIER` env-var override —
/// a test-only escape hatch that lets legitimate-collapse fixtures whose
/// post-calibration gains sit just below the new per-tier floors continue
/// to exercise the feature. Production callers leave the env var unset and
/// receive the strict per-tier floors documented above.
#[inline]
#[must_use]
pub fn coordinated_post_discount_noise_floor(op_count: usize) -> f32 {
    let base = match op_count {
        0 | 1 => COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP,
        2 => COORDINATED_POST_DISCOUNT_NOISE_FLOOR_2OPS,
        3 => COORDINATED_POST_DISCOUNT_NOISE_FLOOR_3OPS,
        _ => COORDINATED_POST_DISCOUNT_NOISE_FLOOR_4PLUS_OPS,
    };
    base * coordinated_noise_floor_multiplier()
}

/// Lower clamp for the test-only noise-floor multiplier (Issue #1272).
///
/// Negative or non-finite values are rejected; the lowest sensible value is
/// just above zero (effectively disabling the floor).
pub const COORDINATED_NOISE_FLOOR_MULTIPLIER_FLOOR: f32 = 1e-3;

/// Upper clamp for the test-only noise-floor multiplier (Issue #1272).
///
/// Capped at 100 so a misconfigured env var cannot inadvertently filter
/// every coordinated-structural candidate.
pub const COORDINATED_NOISE_FLOOR_MULTIPLIER_CEILING: f32 = 100.0;

/// Returns the active multiplier applied to every per-op-count noise floor
/// (Issue #1272).
///
/// Reads `NEAT_AI_DISCOVERY_COORDINATED_NOISE_FLOOR_MULTIPLIER` at call time
/// so integration tests can relax (or further tighten) the floor without
/// recompiling. Defaults to `1.0`; unparsable, non-finite, or out-of-range
/// values fall back to `1.0`.
#[inline]
#[must_use]
pub fn coordinated_noise_floor_multiplier() -> f32 {
    std::env::var("NEAT_AI_DISCOVERY_COORDINATED_NOISE_FLOOR_MULTIPLIER")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map_or(1.0, |v| {
            v.clamp(
                COORDINATED_NOISE_FLOOR_MULTIPLIER_FLOOR,
                COORDINATED_NOISE_FLOOR_MULTIPLIER_CEILING,
            )
        })
}

/// Legacy alias for [`COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP`]
/// (Issue #1272).
///
/// Kept so external callers continue to build. New code should call
/// [`coordinated_post_discount_noise_floor`] with the candidate's op-count
/// instead of comparing against a single global floor.
#[deprecated(
    note = "Use coordinated_post_discount_noise_floor(op_count) for per-op-count tiering (Issue #1272)"
)]
pub const COORDINATED_POST_DISCOUNT_NOISE_FLOOR: f32 = COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP;

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
// Coordinated-Structural Empirical Discount (Issue #732, #790, #1058)
// =============================================================================

// Issue #1058: The previous three-layer compound discount (per-op exponential ×
// flat pessimism × calibration) was too aggressive, creating a paradox: candidates
// that survived filtering were still poorly calibrated, while potentially viable
// candidates were filtered out entirely.
//
// Production discovery-cache evidence (creature 0e18e62c: 6/389 successes, creature
// 066649c7: 0/519):
// - Successful coordinated-structural candidates are predominantly 2-op
// - 3+ op candidates have near-zero success rates in production
// - Predictions overestimate by ~10,000× (handled by calibration factor)
//
// The new model uses a single empirical discount per operation count, derived
// from production discovery-cache success rates. The calibration factor
// (`COORDINATED_PREDICTION_CALIBRATION`) remains separate to bridge the
// prediction-to-reality magnitude gap.
//
// Old compound (per-op × pessimism):
//   1-op: 1.0 × 0.15 = 0.15 | 2-op: 0.65 × 0.15 = 0.0975
//   3-op: 0.4225 × 0.15 = 0.0634 | 4-op: 0.274 × 0.15 = 0.0411
//
// New empirical factors (single lookup, less aggressive):
//   1-op: 1.0 | 2-op: 0.5 | 3-op: 0.2 | 4+-op: 0.1

/// Empirical discount for 2-operation coordinated candidates (Issue #1058).
///
/// Production discovery-cache data shows 2-op candidates account for the majority of successful
/// coordinated-structural candidates (creature 0e18e62c). The 0.5 factor replaces
/// the old compound of `0.65^1 × 0.15 = 0.0975`, allowing ~5× more candidates
/// through while relying on the calibration factor for magnitude correction.
///
/// ## Valid Range
/// Must be in (0.0, 1.0).
pub const COORDINATED_EMPIRICAL_DISCOUNT_2OPS: f32 = 0.5;

/// Empirical discount for 3-operation coordinated candidates (Issue #1058).
///
/// 3-op candidates have substantially lower success rates than 2-op in the
/// production discovery cache. The 0.2 factor replaces the old compound of
/// `0.65^2 × 0.15 = 0.0634`, still allowing ~3× more candidates through.
///
/// ## Valid Range
/// Must be in (0.0, 1.0).
pub const COORDINATED_EMPIRICAL_DISCOUNT_3OPS: f32 = 0.2;

/// Empirical discount for 4+ operation coordinated candidates (Issue #1058).
///
/// 4+ op candidates have near-zero success rates in production data.
/// The 0.1 factor replaces the old compound of `0.65^3 × 0.15 = 0.0411`, still
/// allowing ~2.4× more candidates through but remaining heavily discounted.
///
/// ## Valid Range
/// Must be in (0.0, 1.0).
pub const COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS: f32 = 0.1;

/// Legacy alias for backward compatibility with code referencing the old per-op
/// discount constant (Issue #1058).
///
/// New code should use the empirical per-op-count factors directly via
/// `coordinated_empirical_discount()`.
#[deprecated(note = "Use COORDINATED_EMPIRICAL_DISCOUNT_*OPS constants (Issue #1058)")]
pub const COORDINATED_OPERATION_DISCOUNT: f32 = 0.65;

/// Minimum absolute gain required for a multi-operation coordinated candidate.
///
/// Single-operation candidates are accepted with any positive gain, but
/// multi-operation candidates (>= 2 operations) must exceed this threshold
/// after discounting. This prevents near-zero predictions from generating
/// candidates that almost never succeed.
///
/// Issue #1058: Lowered from 1e-3 to 1e-5. The calibration factor
/// (`COORDINATED_PREDICTION_CALIBRATION`) already accounts for the ~10,000×
/// overestimation gap. The previous 1e-3 threshold was filtering out
/// viable candidates whose post-calibration gains were legitimately small.
///
/// ## Valid Range
/// Must be > 0.0. Values above 1e-3 may filter too aggressively.
pub const MIN_COORDINATED_MULTI_OP_GAIN: f32 = 1e-5;

/// Return the single empirical discount factor for a given operation count (Issue #1058).
///
/// Replaces the old three-layer compound discount (per-op exponential × flat
/// pessimism discount) with a single lookup by operation count, derived from
/// production discovery-cache success rates.
///
/// - 1 op: no discount (1.0)
/// - 2 ops: moderate discount (0.5)
/// - 3 ops: substantial discount (0.2)
/// - 4+ ops: heavy discount (0.1)
#[inline]
pub fn coordinated_empirical_discount(op_count: usize) -> f32 {
    match op_count {
        0 | 1 => 1.0,
        2 => COORDINATED_EMPIRICAL_DISCOUNT_2OPS,
        3 => COORDINATED_EMPIRICAL_DISCOUNT_3OPS,
        _ => COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS,
    }
}

// =============================================================================
// Coordinated-Structural Weight Estimation (Issue #897)
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

/// Legacy alias for backward compatibility (Issue #1058).
///
/// The flat pessimism discount has been folded into the per-op-count empirical
/// factors. New code should not use this constant.
#[deprecated(note = "Folded into empirical per-op-count factors (Issue #1058)")]
pub const COORDINATED_PESSIMISM_DISCOUNT: f32 = 0.15;

// =============================================================================
// Scoring Boost Multipliers
// =============================================================================

/// Scoring boost multiplier for Micro-Nudge variant candidates (Issue #888).
///
/// Production discovery-cache evidence shows the Micro-Nudge pattern (incoming=2,
/// outgoing=0.001–0.005) dominates successes at ~90% of successful samples.
/// This boost is applied to Micro-Nudge variant candidates to prioritise
/// them in the ranking.
///
/// ## Valid Range
/// Must be > 1.0 (boost) and <= 2.0 (avoid over-biasing).
pub const MICRO_NUDGE_VARIANT_BOOST: f32 = 1.5;

/// Scoring boost multiplier for `remove-low-impact` candidates (Issue #892).
///
/// The production discovery cache shows `remove-low-impact` has the highest success
/// rate at 21.5% (440/2,043) — roughly double the overall 10.7% rate. This boost
/// is applied to the `removal_savings` score to ensure removal candidates are
/// ranked higher relative to other candidate types.
///
/// The boost is derived from the ratio of `remove-low-impact` success rate to the
/// overall baseline: 21.5% / 10.7% ≈ 2.0, dampened with square-root to 1.41,
/// then rounded to 1.5 for conservatism.
///
/// Issue #1142 note: the boost is applied to raw complexity savings **before** the
/// noise-floor check (`REMOVE_LOW_IMPACT_NOISE_FLOOR`), so candidates whose raw
/// savings would not clear impact still survive the savings-vs-impact test thanks
/// to the boost. The noise-floor gate below ensures such marginal candidates do
/// not reach the FFI response.
///
/// ## Valid Range
/// Must be >= 1.0 (boost) and <= 3.0 (avoid over-biasing).
pub const REMOVAL_CANDIDATE_BOOST: f32 = 1.5;

/// Minimum net improvement required for a `remove-low-impact` candidate to
/// survive, **in units of `costOfGrowth`** (Issues #1142, #1814).
///
/// # Why the floor is denominated, not absolute (Issue #1814)
///
/// The screened quantity is
/// `boostedSavings − contribution`, and `boostedSavings` is
/// `REMOVAL_CANDIDATE_BOOST × costOfGrowth × (1 + degree/10)` — **linear in the
/// host-supplied `costOfGrowth`**. Screening it with an absolute `f32` compared
/// two quantities on different scales: at the shipped default
/// [`DEFAULT_COST_OF_GROWTH`](crate::focus::DEFAULT_COST_OF_GROWTH) of `1e-7` the
/// old absolute `1e-5` floor needed a **657-synapse** hidden neuron with zero
/// contribution to clear it, so it rejected every neuron the production
/// population actually contains. Worse, any host-side change to `costOfGrowth`
/// silently rescaled the gate's strictness by the same factor.
///
/// Expressing the floor as a multiple of `costOfGrowth` makes that coupling
/// explicit and scale-invariant: the same neuron gets the same verdict whatever
/// `costOfGrowth` the host sends. This mirrors Gate 1's
/// [`REMOVAL_NET_GAIN_FLOOR_UNITS`] on the analysis path — the shared
/// denomination decided in `docs/analysis/remove-neuron-gain-scale-1785.md`.
///
/// # Why `1.0` — one hidden neuron's complexity cost
///
/// A removal must be worth at least one whole hidden neuron of complexity, net
/// of the contribution it takes with it. The value is bracketed, not picked:
///
/// - **Above `0.664`**, so the #1142 numerical-noise class is still rejected.
///   Production discovery-cache entry
///   `v2_remove-low-impact_0ce92a87-a048-49d0-9b53-43487d123817.json` captured
///   `boosted_savings = 1.20e-7`, `activation_weighted_impact = 1.14e-7`,
///   `net_improvement = +6.64e-8` and `actualErrorReduction = -2.39e-7` (the
///   removal *harmed* the creature). At the shipped
///   [`DEFAULT_COST_OF_GROWTH`](crate::focus::DEFAULT_COST_OF_GROWTH) that net
///   is `0.664` units, so a floor of `1.0` unit rejects it with 1.5× margin —
///   the #1142 guarantee is preserved, re-denominated.
/// - **Below `1.5`**, so a zero-contribution orphan stays prunable: with no
///   synapses its net is exactly `REMOVAL_CANDIDATE_BOOST × costOfGrowth`
///   = `1.5` units. A rule that cannot prune a dead neuron has failed at its
///   only certain case.
///
/// # Break-even degree (replaces the old 657)
///
/// With zero contribution the net is `1.5 × (1 + degree/10)` units, which
/// exceeds `1.0` unit at **degree 0** — every zero-contribution hidden neuron
/// now clears the floor, at any degree and any `costOfGrowth`. The floor instead
/// bounds the *contribution* a neuron may carry: it survives while
/// `contribution ≤ costOfGrowth × (0.5 + 0.15 × degree)`, e.g. `2.3e-7` for a
/// 12-synapse neuron at the shipped default.
///
/// # Overrides
///
/// Both are read at call time by [`remove_low_impact_noise_floor`]:
/// `NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS` sets this multiplier,
/// and the pre-#1814 `NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR` still
/// pins an **absolute** floor verbatim (including `0.0` to disable it).
///
/// ## Valid Range
/// Must be in (0.664, 1.5). Below `0.664` the #1142 noise class leaks through;
/// at or above `1.5` a zero-contribution orphan can no longer be pruned.
pub const REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS: f32 = 1.0;

/// Return the effective remove-low-impact noise-floor for a given
/// `cost_of_growth` (Issues #1142, #1814).
///
/// Returns `max(REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS × cost_of_growth,
/// GAIN_FLOOR_NOISE_BACKSTOP)` — the same shape, and the same backstop, as
/// Gate 1's [`removal_net_gain_floor`]. The backstop stops a host sending a
/// vanishing `cost_of_growth` from opening the screen into the band where the
/// arithmetic is pure floating-point noise; it binds only below
/// `cost_of_growth = 1e-9`.
///
/// A non-finite or non-positive `cost_of_growth` is a caller bug (#1807 owns
/// validating it) and cannot produce a meaningful saving, so the backstop alone
/// is returned rather than a silently widened — or `NaN` — floor.
///
/// # Overrides
///
/// * `NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR` — an **absolute** floor,
///   returned verbatim and unscaled. Preserved from #1142 so existing operator
///   and test overrides keep their meaning; `0.0` still disables the floor.
///   Takes precedence over the units override.
/// * `NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS` — the multiplier on
///   `cost_of_growth`, replacing [`REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS`]. `0.0`
///   disables the floor.
///
/// A value that fails to parse, is non-finite, or is negative is ignored and the
/// compile-time default applies.
#[must_use]
pub fn remove_low_impact_noise_floor(cost_of_growth: f32) -> f32 {
    if let Some(absolute) = env_f32_non_negative("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR")
    {
        return absolute;
    }

    let units = env_f32_non_negative("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS")
        .unwrap_or(REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS);

    if units == 0.0 {
        return 0.0;
    }
    if !cost_of_growth.is_finite() || cost_of_growth <= 0.0 {
        return GAIN_FLOOR_NOISE_BACKSTOP;
    }
    (units * cost_of_growth).max(GAIN_FLOOR_NOISE_BACKSTOP)
}

/// Read a finite, non-negative `f32` from `name`, or `None` when unset or
/// invalid (Issue #1814).
fn env_f32_non_negative(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
}

/// Lock protecting every test that reads or writes either
/// `NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR` override (Issues #1142,
/// #1767, #1814).
///
/// The whole crate shares one lock, so a constants test mutating the env-var can
/// never race a focus-triage test that depends on the default floor.
///
/// Poisoning is tolerated deliberately: one failing test would otherwise
/// cascade into a `PoisonError` panic in every other test holding the lock,
/// hiding the single real failure behind a wall of unrelated ones.
#[cfg(test)]
pub(crate) fn noise_floor_env_lock() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::{Mutex, OnceLock, PoisonError};
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

// =============================================================================
// Prediction Calibration Scaling (Issue #891)
// =============================================================================

// The production discovery cache reveals that `expectedCreatureScoreGain` overestimates
// actual outcomes by 100–10,000×. The magnitude of overestimation varies by candidate
// type, making cross-type comparisons unreliable. These per-type calibration factors
// are applied post-pessimism-discount to correct the systematic magnitude gap.
//
// Cache Evidence (production discovery cache):
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

/// Prediction calibration factor for synapse candidates (Issue #891, #1056).
///
/// The production discovery cache (30+ creatures) shows add-synapses has a ~0.1%
/// actual success rate (3/1001), with massive overestimation of predicted gains.
/// Updated from 0.001 to 0.0003 to match production reality.
///
/// This is the base factor used by the logistic calibration function. The
/// effective calibration is further modulated by the `improved_ratio` via a
/// logistic curve (Issue #1056).
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values above 0.001 provide insufficient correction.
/// Values below 0.0001 risk suppressing all synapse candidates.
pub const SYNAPSE_PREDICTION_CALIBRATION: f32 = 0.0003;

/// Prediction calibration factor for neuron candidates (Issue #891, #1056).
///
/// The production discovery cache shows add-neurons has a ~2.7% actual success
/// rate (28/1028), with the predicted improved ratio (~50%) overestimating
/// actual success by ~18×. Updated from 0.01 to 0.003 to account for this
/// gap after measuring against production outcomes across 30+ creatures.
///
/// This is the base factor used by the logistic calibration function.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values above 0.01 provide insufficient correction.
/// Values below 0.001 risk suppressing all neuron candidates.
pub const NEURON_PREDICTION_CALIBRATION: f32 = 0.003;

/// Prediction calibration factor for coordinated-structural candidates (Issue #891, #1056).
///
/// The production discovery cache shows coordinated-structural has a ~1.1% actual
/// success rate (6/525), with predictions overestimating by orders of magnitude.
/// Updated from 0.0001 to 0.00005 to match production reality.
///
/// This is the base factor used by the logistic calibration function.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values above 0.0001 provide insufficient correction.
/// Values below 0.00001 risk suppressing all coordinated candidates.
pub const COORDINATED_PREDICTION_CALIBRATION: f32 = 0.00005;

// =============================================================================
// Logistic Prediction Calibration (Issue #1056)
// =============================================================================

// The linear calibration multiplier (gain × factor) was insufficient to bridge
// the neuron-level → creature-level prediction gap. The relationship between
// `improvedCount/totalCount` and actual creature-level success probability is
// non-linear — moderate improved ratios (0.3–0.6) are far more overestimated
// than high ratios (>0.8).
//
// The logistic calibration modulates the base calibration factor using a sigmoid
// of the improved_ratio:
//
//   effective = base_factor × (floor + (1 - floor) × sigmoid(steepness × (ratio - midpoint)))
//
// Production discovery-cache empirical evidence (30+ creatures):
//
// | Improved Ratio | Approximate Actual Success | Logistic Modulator |
// |----------------|---------------------------|--------------------|
// | 0.1            | Very unlikely             | ~0.12              |
// | 0.3            | Rare                      | ~0.18              |
// | 0.5            | Uncommon (~2.7% neurons)  | ~0.37              |
// | 0.7            | Moderate                  | ~0.72              |
// | 0.9            | More likely               | ~0.94              |

/// Floor for the logistic calibration modulator (Issue #1056).
///
/// The minimum modulator value, applied when the improved ratio is very low.
/// Prevents complete suppression of candidates that may still succeed despite
/// few samples showing improvement.
///
/// ## Valid Range
/// Must be in (0.0, 0.5). Values below 0.05 risk zeroing-out all candidates
/// at low ratios.
pub const LOGISTIC_CALIBRATION_FLOOR: f32 = 0.1;

/// Steepness of the logistic calibration curve (Issue #1056).
///
/// Controls how sharply the modulator transitions from floor to 1.0 around
/// the midpoint. Higher values produce a sharper transition. Derived from
/// fitting against production discovery-cache success rates: a steepness of 8.0 gives
/// the best fit to the observed ratio → success relationship.
///
/// ## Valid Range
/// Must be > 0.0. Values below 4.0 produce too gradual a transition.
/// Values above 15.0 produce near-step-function behaviour.
pub const LOGISTIC_CALIBRATION_STEEPNESS: f32 = 8.0;

/// Midpoint of the logistic calibration curve (Issue #1056).
///
/// The improved ratio at which the modulator is at 50% between floor and 1.0.
/// Set to 0.6 because production discovery-cache data shows improved ratios below 60% are
/// substantially overestimated (add-neurons has ~50% predicted vs ~2.7% actual),
/// while ratios above 70% are relatively more reliable.
///
/// ## Valid Range
/// Must be in (0.2, 0.9). Values below 0.3 do not sufficiently discount
/// moderate ratios. Values above 0.8 discount too aggressively.
pub const LOGISTIC_CALIBRATION_MIDPOINT: f32 = 0.6;

// =============================================================================
// Sole-op remove-neuron net-gain screen (Issue #1812, decided by Issue #1811)
// =============================================================================

/// Converts a neuron's unitless propagation-aware influence fraction into a
/// creature-score delta, for the sole-op `RemoveNeuron` net-gain rule
/// (Issue #1812).
///
/// `estimate_remove_neuron_gain` emits `−influence`, a dimensionless fraction of
/// output sensitivity in `[0, 1]` — not a creature-score quantity. This constant
/// is the missing unit conversion, mirroring what
/// [`NEURON_PREDICTION_CALIBRATION`] does for the add-neuron path, and it is set
/// to the same value: it is the only measured neuron-level → creature-level
/// conversion in the repository, derived from the same production population
/// (#891/#1056), and it is the larger of the two measured calibrations, which
/// makes the removal screen *stricter* — the conservative default while removal
/// outcomes remain unmeasured.
///
/// **Interim value.** No removal has ever survived either gate (#1810), so this
/// cannot yet be fitted from realised removal outcomes; #1815's end-to-end guard
/// supplies the first ones. The fixture verdicts are invariant to it for every
/// value above `1.03e-6`, so no verdict rests on the exact number — see
/// `docs/analysis/remove-neuron-gain-scale-1785.md`.
///
/// **Applied bare.** Unlike the add path, the per-creature
/// `calibration_correction` is deliberately *not* applied on top. That
/// correction is clamped to `[0.001, 1.0]` so it only ever discounts, and the
/// calibrated quantity here is a **cost**: discounting it would shrink the
/// penalty and make removals easier to accept, inverting the correction's safety
/// direction.
///
/// ## Valid Range
/// Must be in (0.0, 1.0]. Values below `1.03e-6` start admitting neurons that
/// carry real downstream influence.
pub const REMOVE_INFLUENCE_CALIBRATION: f32 = NEURON_PREDICTION_CALIBRATION;

/// Acceptance margin for a sole-op `RemoveNeuron`, in units of `costOfGrowth`
/// (Issue #1812).
///
/// `0.5` is half of one hidden neuron's complexity cost, bracketed rather than
/// picked:
///
/// - **Below `1.0`**, because a zero-influence orphan nets exactly
///   `1.0 × costOfGrowth` — the smallest unambiguously free removal that exists.
///   A floor at or above one neuron's cost rejects it, and a rule that cannot
///   prune a dead neuron has failed at its only certain case.
/// - **Above `0.0`**, because at zero a removal whose estimated loss exactly
///   cancels its saving is accepted, spending the host's ablation budget on a
///   break-even change with no margin for estimator error.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). At `1.0` the orphan case becomes an exact-equality
/// comparison in `f32`.
pub const REMOVAL_NET_GAIN_FLOOR_UNITS: f32 = 0.5;

/// The acceptance floor for a sole-op `RemoveNeuron` candidate's
/// **net** `expected_creature_score_gain` (Issue #1812).
///
/// Sole-op removals are screened here instead of against
/// [`coordinated_post_discount_noise_floor`]: their gain is
/// `saving − calibrated influence loss`, denominated in units of
/// `cost_of_growth` rather than on the add-path prediction scale, so the shared
/// floor is not a comparison between like quantities. The shared floor keeps its
/// value and still governs every other candidate type and every multi-op group
/// — this replaces it for one candidate type, it does not drop it.
///
/// Returns `max(REMOVAL_NET_GAIN_FLOOR_UNITS × cost_of_growth,
/// GAIN_FLOOR_NOISE_BACKSTOP)`. The backstop stops a host sending a tiny
/// `cost_of_growth` from opening the screen into the band where the estimate is
/// uncorrelated with the outcome; it binds only below `cost_of_growth = 2e-9`.
/// A non-finite or non-positive `cost_of_growth` is a caller bug and cannot
/// produce a meaningful saving, so the backstop alone is returned rather than a
/// silently widened (or `NaN`) floor.
#[must_use]
pub fn removal_net_gain_floor(cost_of_growth: f32) -> f32 {
    if !cost_of_growth.is_finite() || cost_of_growth <= 0.0 {
        return GAIN_FLOOR_NOISE_BACKSTOP;
    }
    (REMOVAL_NET_GAIN_FLOOR_UNITS * cost_of_growth).max(GAIN_FLOOR_NOISE_BACKSTOP)
}

// =============================================================================
// Unit tests — remove-low-impact noise floor denomination (Issue #1814)
// =============================================================================

#[cfg(test)]
mod remove_low_impact_noise_floor_tests {
    use super::*;

    const ABSOLUTE_ENV: &str = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR";
    const UNITS_ENV: &str = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS";

    /// Clear both overrides so a test starts from the compile-time default.
    fn clear_overrides() {
        // SAFETY: every caller holds `noise_floor_env_lock()`, so no other test
        // touches the environment concurrently.
        unsafe {
            std::env::remove_var(ABSOLUTE_ENV);
            std::env::remove_var(UNITS_ENV);
        }
    }

    /// The floor is a multiple of `costOfGrowth`, so a 10× change in
    /// `costOfGrowth` moves the threshold by exactly 10× — the property that
    /// makes the screen scale-invariant (Issue #1814).
    #[test]
    fn floor_is_linear_in_cost_of_growth() {
        let _guard = noise_floor_env_lock();
        clear_overrides();

        assert!((remove_low_impact_noise_floor(1e-7) - 1e-7).abs() < 1e-12);
        assert!((remove_low_impact_noise_floor(1e-6) - 1e-6).abs() < 1e-11);
        assert!((remove_low_impact_noise_floor(1e-4) - 1e-4).abs() < 1e-9);
    }

    /// A vanishing, non-finite or non-positive `costOfGrowth` cannot widen the
    /// screen into pure floating-point noise: the shared backstop applies.
    #[test]
    fn backstop_clamps_degenerate_cost_of_growth() {
        let _guard = noise_floor_env_lock();
        clear_overrides();

        for degenerate in [1e-12_f32, 0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                (remove_low_impact_noise_floor(degenerate) - GAIN_FLOOR_NOISE_BACKSTOP).abs()
                    < 1e-14,
                "costOfGrowth {degenerate} must fall back to the noise backstop"
            );
        }
    }

    /// The pre-#1814 absolute override still pins the threshold verbatim, at
    /// any `costOfGrowth` — the #1142 operator escape hatch is preserved.
    #[test]
    fn absolute_env_override_is_preserved() {
        let _guard = noise_floor_env_lock();
        clear_overrides();
        // SAFETY: env access is serialised via `noise_floor_env_lock()`.
        unsafe {
            std::env::set_var(ABSOLUTE_ENV, "3e-4");
        }
        let pinned_low = remove_low_impact_noise_floor(1e-7);
        let pinned_high = remove_low_impact_noise_floor(1e-3);

        // `0.0` still disables the floor entirely (the pre-#1814 test path).
        // SAFETY: env access is serialised via `noise_floor_env_lock()`.
        unsafe {
            std::env::set_var(ABSOLUTE_ENV, "0.0");
        }
        let disabled = remove_low_impact_noise_floor(1e-7);
        clear_overrides();

        assert!((pinned_low - 3e-4).abs() < 1e-9, "got {pinned_low:e}");
        assert!((pinned_high - 3e-4).abs() < 1e-9, "got {pinned_high:e}");
        assert!(
            (disabled - 0.0).abs() < f32::EPSILON,
            "0.0 must disable the floor, got {disabled:e}"
        );
    }

    /// The new units override replaces the multiplier and keeps scaling with
    /// `costOfGrowth`; the absolute override wins when both are set.
    #[test]
    fn units_env_override_scales_and_yields_to_absolute() {
        let _guard = noise_floor_env_lock();
        clear_overrides();
        // SAFETY: env access is serialised via `noise_floor_env_lock()`.
        unsafe {
            std::env::set_var(UNITS_ENV, "2.0");
        }
        let scaled_low = remove_low_impact_noise_floor(1e-7);
        let scaled_high = remove_low_impact_noise_floor(1e-6);

        // SAFETY: env access is serialised via `noise_floor_env_lock()`.
        unsafe {
            std::env::set_var(ABSOLUTE_ENV, "5e-6");
        }
        let absolute_wins = remove_low_impact_noise_floor(1e-7);
        clear_overrides();

        assert!((scaled_low - 2e-7).abs() < 1e-12, "got {scaled_low:e}");
        assert!((scaled_high - 2e-6).abs() < 1e-11, "got {scaled_high:e}");
        assert!(
            (absolute_wins - 5e-6).abs() < 1e-11,
            "got {absolute_wins:e}"
        );
    }

    /// An unparseable, negative or non-finite override is ignored rather than
    /// silently disabling the screen.
    #[test]
    fn invalid_overrides_fall_back_to_the_default() {
        let _guard = noise_floor_env_lock();
        for invalid in ["not-a-number", "-1e-6", "NaN", ""] {
            clear_overrides();
            // SAFETY: env access is serialised via `noise_floor_env_lock()`.
            unsafe {
                std::env::set_var(ABSOLUTE_ENV, invalid);
                std::env::set_var(UNITS_ENV, invalid);
            }
            let floor = remove_low_impact_noise_floor(1e-7);
            assert!(
                (floor - 1e-7).abs() < 1e-12,
                "override {invalid:?} must be ignored, got {floor:e}"
            );
        }
        clear_overrides();
    }

    /// The shipped multiplier stays inside the bracket its doc comment claims:
    /// strict enough to reject the #1142 `6.64e-8`-class net improvement at the
    /// default `costOfGrowth`, loose enough to prune a zero-contribution orphan.
    #[test]
    fn shipped_units_sit_inside_the_documented_bracket() {
        const { assert!(REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS > 0.664) };
        const { assert!(REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS < REMOVAL_CANDIDATE_BOOST) };
    }
}
