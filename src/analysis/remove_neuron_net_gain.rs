//! The sole-op `RemoveNeuron` net-gain rule (Issue #1812, decided by #1811).
//!
//! [`estimate_remove_neuron_gain`] emits `−influence`: a **unitless** fraction
//! of the output's sensitivity, negated. It is a cost term with no units, and
//! until this module existed it was written straight into
//! `expectedCreatureScoreGain` — a field the shared coordinated floor screens as
//! a strictly-positive **creature-score benefit**. No non-positive value clears a
//! positive floor, so every sole-op removal was dropped by construction.
//!
//! The fix is neither to flip the estimator's sign nor to drop the floor. It is
//! to give the gain its missing benefit term and its missing unit conversion:
//!
//! ```text
//! saving(u) = calculate_removal_savings(incoming, outgoing, cost_of_growth)
//!           = cost_of_growth × (1 + degree(u) / 10)
//! loss(u)   = |estimate_remove_neuron_gain(u)| × REMOVE_INFLUENCE_CALIBRATION
//!
//! expected_creature_score_gain(u) = saving(u) − loss(u)
//! accept(u) ⟺ gain(u) ≥ removal_net_gain_floor(cost_of_growth)
//! ```
//!
//! `saving(u)` needs no calibration: `costOfGrowth` **is** NEAT-AI's own
//! `Score.ts` complexity penalty per hidden neuron, so the saving is already
//! exact and already in creature-score units. Only the influence term is
//! estimated, so only it is converted.
//!
//! Substituted, the rule is a sentence rather than a threshold: **a removal must
//! pay for its influence loss out of the synapses it takes with it.** A neuron
//! carrying real downstream influence is rejected outright — `loss` grows
//! linearly with influence while `saving` is bounded by degree times a `1e-7`
//! constant, so on the #1810 fixture the two are three to four orders apart for
//! every connected neuron. Only a neuron attenuated to near-nothing is prunable,
//! which is exactly the population the pruning path exists for.
//!
//! Scope: **sole-op `RemoveNeuron` only**, matching how
//! [`apply_honest_remove_neuron_gain`](super::discovery_dispatch::apply_honest_remove_neuron_gain)
//! already scopes itself. A multi-op coordinated candidate's gain reflects the
//! whole atomic group, so it keeps
//! [`coordinated_post_discount_noise_floor`](super::constants::coordinated_post_discount_noise_floor)
//! unchanged.
//!
//! The full derivation, the tolerance bracket and the worked fixture example are
//! in `docs/analysis/remove-neuron-gain-scale-1785.md`.

use crate::CreatureJson;
use crate::focus::{DEFAULT_COST_OF_GROWTH, SynapseCounts, calculate_removal_savings};

use super::constants::{REMOVE_INFLUENCE_CALIBRATION, removal_net_gain_floor};
use super::remove_neuron_gain::estimate_remove_neuron_gain;

/// The `cost_of_growth` the analysis path screens sole-op removals against
/// (Issue #1812).
///
/// The host's `costOfGrowth` reaches the focus path on `RankFocusNeuronsInput`
/// but **not** `AnalyzeParallelInput`, so the analysis path has no host value to
/// use. Rather than restate a literal, this resolves through the same
/// [`DEFAULT_COST_OF_GROWTH`] definition the focus triage falls back to, so the
/// two gates cannot drift apart (the Issue #1807 single-definition rule).
/// Threading the host value onto `AnalyzeParallelInput` is follow-on plumbing:
/// at the default the rule is already reachable.
#[must_use]
pub fn analysis_cost_of_growth() -> f32 {
    DEFAULT_COST_OF_GROWTH
}

/// The creature-score cost of losing a neuron's downstream influence
/// (Issue #1812).
///
/// `influence` is the unitless propagation-aware fraction from
/// [`estimate_remove_neuron_gain`]; its sign is irrelevant here (the estimator
/// emits it negated), so the magnitude is taken and converted by
/// [`REMOVE_INFLUENCE_CALIBRATION`]. A non-finite influence cannot be converted
/// into a meaningful cost, so it yields an infinite loss — the removal is
/// rejected rather than silently treated as free.
#[must_use]
// The conversion is done in f64 and narrowed once at the end; the result is a
// small value well within f32 range (Issue #873).
#[allow(clippy::cast_possible_truncation)]
pub fn removal_influence_loss(influence: f64) -> f32 {
    if !influence.is_finite() {
        return f32::INFINITY;
    }
    (influence.abs() * f64::from(REMOVE_INFLUENCE_CALIBRATION)) as f32
}

/// The net creature-score benefit of removing a neuron: its exact complexity
/// saving minus its calibrated influence loss (Issue #1812).
///
/// Both terms are on the realised creature-score-delta scale, so the difference
/// is the value `expectedCreatureScoreGain` is documented to carry and the
/// shared gain-descending ranking sort is documented to consume.
#[must_use]
pub fn removal_net_gain(saving: f32, influence: f64) -> f32 {
    saving - removal_influence_loss(influence)
}

/// The net gain of removing `neuron_uuid` from `creature`, or `None` when the
/// neuron is absent or is an output (Issue #1812).
///
/// `counts` is the caller's pre-built [`SynapseCounts`] so a whole candidate set
/// can be scored without rebuilding the degree index per neuron.
#[must_use]
pub fn estimate_remove_neuron_net_gain(
    creature: &CreatureJson,
    counts: &SynapseCounts<'_>,
    neuron_uuid: &str,
    cost_of_growth: f32,
) -> Option<f32> {
    let influence = estimate_remove_neuron_gain(creature, neuron_uuid)?;
    let (incoming, outgoing) = counts.get(neuron_uuid);
    let saving = calculate_removal_savings(incoming, outgoing, cost_of_growth);
    Some(removal_net_gain(saving, influence))
}

/// Whether a sole-op `RemoveNeuron` candidate's net gain clears the removal
/// acceptance floor (Issue #1812).
///
/// `floor_multiplier` is the conservative-discovery tightening factor the shared
/// coordinated floor already applies; values below `1.0` are clamped to `1.0` so
/// the screen can never become looser than the shipped default.
#[must_use]
pub fn removal_net_gain_accepted(
    net_gain: f32,
    cost_of_growth: f32,
    floor_multiplier: f32,
) -> bool {
    let multiplier = if floor_multiplier.is_finite() {
        floor_multiplier.max(1.0)
    } else {
        1.0
    };
    net_gain >= removal_net_gain_floor(cost_of_growth) * multiplier
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::constants::GAIN_FLOOR_NOISE_BACKSTOP;
    use crate::analysis::remove_neuron_constant_promotion::CONSTANT_NEURON_PRIORITY_GAIN;

    const COST_OF_GROWTH: f32 = 1e-7;

    /// A zero-influence orphan nets exactly one neuron's complexity cost times
    /// its degree factor, and clears the floor by the documented 2.4×.
    #[test]
    fn zero_influence_orphan_clears_the_floor() {
        // degree 2 → saving = 1e-7 × 1.2.
        let net = removal_net_gain(calculate_removal_savings(2, 0, COST_OF_GROWTH), 0.0);
        assert!(
            (net - 1.2e-7).abs() < 1e-12,
            "a zero-influence degree-2 orphan nets 1.2e-7, got {net:e}"
        );
        assert!(removal_net_gain_accepted(net, COST_OF_GROWTH, 1.0));
    }

    /// The fixture's least-influential *connected* neuron (`h-b-08`, degree 5,
    /// influence 9.72e-2) is still rejected — by ~3.5 orders of magnitude.
    #[test]
    fn least_influential_connected_neuron_is_rejected() {
        let net = removal_net_gain(
            calculate_removal_savings(5, 0, COST_OF_GROWTH),
            -9.722_222e-2,
        );
        assert!(net < 0.0, "the influence loss must dominate, got {net:e}");
        assert!(!removal_net_gain_accepted(net, COST_OF_GROWTH, 1.0));
    }

    /// The worst case (influence 1.0) is rejected by four orders of magnitude,
    /// no matter how many synapses it takes with it.
    #[test]
    fn maximum_influence_neuron_is_rejected_at_any_degree() {
        for degree in [0_usize, 4, 12, 600] {
            let net = removal_net_gain(calculate_removal_savings(degree, 0, COST_OF_GROWTH), -1.0);
            assert!(
                !removal_net_gain_accepted(net, COST_OF_GROWTH, 1.0),
                "a full-influence neuron of degree {degree} must be rejected, net {net:e}"
            );
        }
    }

    /// The promotion marker clears the floor unchanged — #1622's escape hatch
    /// keeps working under the new screen.
    #[test]
    fn promotion_priority_gain_clears_the_floor() {
        assert!(removal_net_gain_accepted(
            CONSTANT_NEURON_PRIORITY_GAIN,
            COST_OF_GROWTH,
            1.0
        ));
    }

    /// The floor tracks `cost_of_growth`, clamped from below by the shared
    /// noise backstop.
    #[test]
    fn floor_tracks_cost_of_growth_and_clamps_at_the_backstop() {
        assert!((removal_net_gain_floor(1e-7) - 5e-8).abs() < 1e-14);
        assert!((removal_net_gain_floor(1e-12) - GAIN_FLOOR_NOISE_BACKSTOP).abs() < 1e-14);
        assert!((removal_net_gain_floor(f32::NAN) - GAIN_FLOOR_NOISE_BACKSTOP).abs() < 1e-14);
        assert!((removal_net_gain_floor(-1.0) - GAIN_FLOOR_NOISE_BACKSTOP).abs() < 1e-14);
    }

    /// A conservative multiplier tightens the screen; a nonsense multiplier
    /// never loosens it.
    #[test]
    fn floor_multiplier_only_ever_tightens() {
        let net = 6e-8_f32; // clears 5e-8, but not 5e-8 × 2.
        assert!(removal_net_gain_accepted(net, COST_OF_GROWTH, 1.0));
        assert!(!removal_net_gain_accepted(net, COST_OF_GROWTH, 2.0));
        assert!(removal_net_gain_accepted(net, COST_OF_GROWTH, 0.1));
        assert!(removal_net_gain_accepted(net, COST_OF_GROWTH, f32::NAN));
    }

    /// A non-finite influence is a fault, not a free removal.
    #[test]
    fn non_finite_influence_is_rejected() {
        let net = removal_net_gain(calculate_removal_savings(600, 0, COST_OF_GROWTH), f64::NAN);
        assert!(!removal_net_gain_accepted(net, COST_OF_GROWTH, 1.0));
    }

    /// The estimator seam: an output neuron yields no net gain at all.
    #[test]
    fn output_neuron_yields_no_net_gain() {
        let creature: CreatureJson = serde_json::from_str(
            r#"{
                "input": 1, "output": 1,
                "neurons": [
                    {"uuid": "input-0", "type": "input"},
                    {"uuid": "hidden-0", "type": "hidden", "squash": "IDENTITY"},
                    {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
                ],
                "synapses": [
                    {"fromUUID": "input-0", "toUUID": "hidden-0", "weight": 1.0},
                    {"fromUUID": "hidden-0", "toUUID": "out-0", "weight": 1.0}
                ]
            }"#,
        )
        .expect("valid creature JSON");
        let counts = SynapseCounts::new(&creature);
        assert!(
            estimate_remove_neuron_net_gain(&creature, &counts, "out-0", COST_OF_GROWTH).is_none()
        );
        assert!(
            estimate_remove_neuron_net_gain(&creature, &counts, "absent", COST_OF_GROWTH).is_none()
        );
        assert!(
            estimate_remove_neuron_net_gain(&creature, &counts, "hidden-0", COST_OF_GROWTH)
                .is_some()
        );
    }
}
