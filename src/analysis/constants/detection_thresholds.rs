//! Detection module thresholds for pattern filtering.
//!
//! Thresholds used by detection modules (saturation, dead neuron, bottleneck,
//! removal, weight constraints) to filter and validate candidates before
//! scoring.

// =============================================================================
// Pessimism Discount Ratio Thresholds (Issue #506)
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
/// Issue #1109: Raised from 0.4 to 0.55. Production failure data from GRQ-sampler
/// (commit 50a2909) showed neuron candidates with improved ratios of 52-54%
/// (267-274 out of 510) consistently producing negative actual error reductions
/// despite positive predictions. The previous 0.4 threshold was too permissive —
/// candidates barely above 50/50 are indistinguishable from random chance and
/// waste evaluation budget. Raising the neuron threshold to 0.55 aligns with the
/// observed failure patterns while still allowing moderately confident candidates
/// through (it remains below `MIN_IMPROVED_RATIO = 0.6` because neuron candidates
/// are noisier than synapse candidates).
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.3 provide insufficient filtering.
/// Values above `MIN_IMPROVED_RATIO` may be too strict for neuron candidates.
pub const NEURON_MIN_IMPROVED_RATIO: f32 = 0.55;

// =============================================================================
// Remove-Low-Impact Candidate Thresholds (Issue #892)
// =============================================================================

/// Maximum mean activation for a removal candidate to be considered high-quality.
///
/// GRQ-sampler discovery cache shows that successful `remove-low-impact` candidates
/// (21.5% success rate, 440/2,043) consistently have mean activation near zero
/// (~0 to 0.04). Failed removals often have much higher mean activation (up to 57.8),
/// indicating the neuron was actually contributing to the network.
///
/// Candidates with `mean_activation` above this threshold are filtered out to
/// focus removal efforts on neurons that are genuinely inactive.
///
/// ## Valid Range
/// Must be > 0.0. Values above 0.1 risk including neurons that are contributing.
/// Values below 0.01 may be too restrictive and miss valid removal candidates.
pub const REMOVAL_MEAN_ACTIVATION_THRESHOLD: f32 = 0.04;

/// Maximum structural impact for a removal candidate to be considered high-quality.
///
/// GRQ-sampler discovery cache shows successful `remove-low-impact` removals have
/// impact magnitudes ≤ 6e-5. Neurons with higher structural impact are more likely
/// to be contributing to the network output even if their activation is low.
///
/// ## Valid Range
/// Must be > 0.0. Values above 1e-3 risk including neurons with meaningful impact.
/// Values below 1e-6 may be too restrictive.
pub const REMOVAL_IMPACT_THRESHOLD: f32 = 6e-5;

// =============================================================================
// Add-Neuron Weight Constraints (Issue #888)
// =============================================================================

// GRQ-sampler discovery cache shows that successful add-neuron candidates have
// dramatically different weight/bias magnitudes than failures:
//
// | Parameter       | Successful Range  | Failed Range      |
// |-----------------|-------------------|-------------------|
// | Outgoing weight | 0.001–0.005 (e-3) | 0.01–0.1 (e-2/1) |
// | Incoming weight | ~2                | 5, 10, 20         |
// | Bias            | 0 to 1            | -10, -5, 5, 10    |
//
// The "Micro-Nudge" variant (incoming=2, outgoing=0.001–0.005) dominates
// successes (~90% of successful samples). "Extreme" variants (incoming=10–20,
// outgoing=0.02–0.1) almost always fail, sometimes catastrophically.

/// Maximum absolute incoming weight for add-neuron candidates (Issue #888).
///
/// GRQ-sampler cache evidence shows successful candidates consistently have
/// incoming weight ~2. Candidates with incoming weights of 5, 10, or 20
/// almost always fail. A threshold of 5.0 provides margin while filtering
/// the clearly extreme values.
///
/// ## Valid Range
/// Must be > 1.0. Values above 10.0 allow too many doomed candidates through.
/// Values below 2.0 may filter the dominant success pattern.
pub const MAX_INCOMING_WEIGHT: f32 = 5.0;

/// Maximum absolute bias for add-neuron candidates (Issue #888).
///
/// GRQ-sampler cache evidence shows successful candidates have bias in
/// the range 0 to 1. Failed candidates have extreme bias values (-10, -5,
/// 5, 10). A threshold of 2.0 provides margin while filtering the clearly
/// extreme values.
///
/// ## Valid Range
/// Must be > 0.0. Values above 5.0 allow too many doomed candidates through.
/// Values below 1.0 may filter some valid candidates.
pub const MAX_BIAS_MAGNITUDE: f32 = 2.0;

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
