//! Bias perturbation for activation regime shifts (Issue #551).
//!
//! A neuron's bias determines its operating regime on the activation function
//! curve. When a neuron operates in a saturated or flat region, small bias
//! changes keep it in the same regime. A large bias shift can move the neuron
//! to a qualitatively different operating regime (e.g., from the saturated tail
//! of TANH to the linear centre), effectively escaping the local minimum.
//!
//! ## Relationship to Other Detection Modules
//!
//! - **Operating point** (Issue #401): measures dynamic range utilisation and
//!   recommends incremental bias/weight adjustments.
//! - **Saturation** (Issue #342): detects neurons stuck at activation bounds.
//! - **This module** (Issue #551): identifies neurons in suboptimal activation
//!   regimes and generates exploratory `setBias` candidates that target
//!   specific regime shifts rather than incremental improvements.
//!
//! ## Detection Criteria
//!
//! A hidden neuron is a regime-shift candidate when:
//! 1. It uses a bounded squash with a known active zone.
//! 2. Its mean pre-activation falls outside the active zone (saturated tail)
//!    **or** is confined to a narrow sub-region (linear-only).
//! 3. It has non-negligible error (mean |error| ≥ `MIN_ERROR_THRESHOLD`).
//! 4. Sufficient samples are available (≥ `MIN_SAMPLES`).
//!
//! ## Recommended Actions
//!
//! - `setBias` — shift the operating point to the centre of the active zone
//!   for a full regime change.

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Minimum mean absolute error before a neuron is considered for regime shift.
/// Below this threshold, the neuron is performing well enough.
const MIN_ERROR_THRESHOLD: f32 = 0.05;

/// Maximum fraction of the active zone utilised before a neuron is flagged.
/// Neurons using less than this fraction of the zone's dynamic range are
/// operating in a suboptimal regime.
const MAX_UTILISATION_FOR_REGIME_SHIFT: f32 = 0.25;

/// A detected bias perturbation candidate for activation regime shift.
#[derive(Debug, Clone)]
pub struct BiasPerturbationCandidate {
    /// UUID of the neuron requiring a regime shift.
    pub neuron_uuid: String,
    /// Current activation function.
    pub squash: String,
    /// Current bias value.
    pub current_bias: f32,
    /// Recommended bias value targeting a different operating regime.
    pub recommended_bias: f32,
    /// Active zone lower bound for the squash function.
    pub active_zone_min: f32,
    /// Active zone upper bound for the squash function.
    pub active_zone_max: f32,
    /// Mean pre-activation value across observations.
    pub mean_pre_activation: f32,
    /// Fraction of the active zone's dynamic range actually utilised (0.0–1.0).
    pub dynamic_range_utilisation: f32,
    /// Mean absolute error across observations.
    pub mean_error: f32,
    /// Estimated improvement from the regime shift.
    pub estimated_improvement: f32,
    /// Human-readable explanation.
    pub reason: String,
}

/// Returns the active zone `(min, max)` for a squash function — the
/// pre-activation range over which the function transitions most of its
/// output dynamic range.
///
/// Returns `None` for unbounded or discrete activations.
fn active_zone(squash: &str) -> Option<(f32, f32)> {
    match squash {
        "TANH" | "HARD_TANH" | "CLIPPED" => Some((-2.0, 2.0)),
        "LOGISTIC" => Some((-4.0, 4.0)),
        "SOFTSIGN" => Some((-4.0, 4.0)),
        "ARCTAN" => Some((-3.0, 3.0)),
        "RELU6" => Some((0.0, 6.0)),
        _ => None,
    }
}

/// Compute what fraction of a squash function's dynamic range is utilised
/// by the given pre-activation range.
fn compute_dynamic_range_utilisation(
    squash: &str,
    value_min: f32,
    value_max: f32,
    zone_min: f32,
    zone_max: f32,
) -> f32 {
    let activation_fn = match crate::activations::target_simulation_fn(squash) {
        Some(f) => f,
        None => return 0.0,
    };

    let zone_out_min = activation_fn(zone_min);
    let zone_out_max = activation_fn(zone_max);
    let theoretical_range = (zone_out_max - zone_out_min).abs();

    if theoretical_range < 1e-9 {
        return 0.0;
    }

    let obs_out_min = activation_fn(value_min);
    let obs_out_max = activation_fn(value_max);
    let observed_range = (obs_out_max - obs_out_min).abs();

    (observed_range / theoretical_range).clamp(0.0, 1.0)
}

/// Compute a bias that would shift the neuron's operating point to the
/// centre of the active zone, effecting a regime change.
///
/// The key insight: `pre_activation = weighted_sum + bias`, so
/// `new_bias = old_bias + (zone_centre - mean_pre_activation)`.
fn compute_regime_shift_bias(
    current_bias: f32,
    mean_pre_activation: f32,
    zone_min: f32,
    zone_max: f32,
) -> f32 {
    let zone_centre = (zone_min + zone_max) / 2.0;
    let delta = zone_centre - mean_pre_activation;
    current_bias + delta
}

/// Detect hidden neurons in suboptimal activation regimes that would
/// benefit from bias perturbation to shift to a different operating regime.
///
/// # Arguments
/// * `hidden_neurons` - Slice of `(uuid, squash, bias)` tuples for hidden neurons.
/// * `neuron_records` - Slice of `(uuid, records)` pairs with observation data.
///
/// # Returns
/// Vector of detected candidates, sorted by estimated improvement (best first).
pub fn detect_bias_perturbation_candidates(
    hidden_neurons: &[(String, String, f32)],
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<BiasPerturbationCandidate> {
    let mut candidates = Vec::with_capacity(hidden_neurons.len());

    for (uuid, squash, bias) in hidden_neurons {
        let Some((_id, records)) = neuron_records.iter().find(|(u, _)| u == uuid) else {
            continue;
        };

        if records.len() < MIN_SAMPLES {
            continue;
        }

        // Get the active zone for this squash function
        let Some((zone_min, zone_max)) = active_zone(squash) else {
            continue;
        };

        // Collect pre-activation values
        let values: Vec<f32> = records
            .iter()
            .filter_map(|r| r.value)
            .filter(|v| v.is_finite())
            .collect();

        if values.len() < MIN_SAMPLES {
            continue;
        }

        // Compute error statistics
        let mean_error: f32 = records
            .iter()
            .map(|r| r.errors.first().copied().unwrap_or(0.0).abs())
            .sum::<f32>()
            / records.len() as f32;

        // Skip neurons with negligible error — they're performing fine
        if mean_error < MIN_ERROR_THRESHOLD {
            continue;
        }

        // Compute pre-activation statistics
        let n = values.len() as f32;
        let mean_pre_activation: f32 = values.iter().sum::<f32>() / n;
        let value_min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let value_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute dynamic range utilisation
        let utilisation =
            compute_dynamic_range_utilisation(squash, value_min, value_max, zone_min, zone_max);

        // Skip neurons that are already using a healthy portion of the active zone
        if utilisation >= MAX_UTILISATION_FOR_REGIME_SHIFT {
            continue;
        }

        // Compute the regime-shifting bias
        let recommended_bias =
            compute_regime_shift_bias(*bias, mean_pre_activation, zone_min, zone_max);

        // Skip if the bias change would be negligible
        if (recommended_bias - bias).abs() < 0.1 {
            continue;
        }

        // Estimated improvement: proportional to error magnitude and how far
        // outside the active zone the neuron is operating
        let regime_severity = 1.0 - utilisation;
        let estimated_improvement = mean_error * regime_severity * 0.2;

        candidates.push(BiasPerturbationCandidate {
            neuron_uuid: uuid.clone(),
            squash: squash.clone(),
            current_bias: *bias,
            recommended_bias,
            active_zone_min: zone_min,
            active_zone_max: zone_max,
            mean_pre_activation,
            dynamic_range_utilisation: utilisation,
            mean_error,
            estimated_improvement,
            reason: format!(
                "Hidden neuron {uuid} operating in suboptimal regime: {squash} with \
                 mean pre-activation {mean_pre_activation:.2} using {:.0}% of dynamic \
                 range (active zone [{zone_min:.1}, {zone_max:.1}]). Bias perturbation \
                 {bias:.3} → {recommended_bias:.3} targets regime shift to active zone \
                 centre. (Issue #551)",
                utilisation * 100.0,
            ),
        });
    }

    // Sort by estimated improvement descending (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Convert bias perturbation candidates to coordinated structural candidates.
///
/// Each candidate generates a `setBias` operation targeting a specific
/// activation regime shift. The coordinated format allows NEAT-AI to
/// evaluate the regime shift as an atomic operation.
pub fn bias_perturbation_to_coordinated_candidates(
    candidates: &[BiasPerturbationCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    candidates
        .iter()
        .map(|c| CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: c.neuron_uuid.clone(),
                bias: c.recommended_bias,
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(c.reason.clone()),
        })
        .collect()
}
