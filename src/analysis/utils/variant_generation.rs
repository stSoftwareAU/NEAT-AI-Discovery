//! Parameterised variant generation for add-neuron and synapse candidates (Issue #806).
//!
//! Previously, three near-identical functions existed for each candidate type
//! (conservative, gentle nudge, micro-nudge). This module consolidates them into
//! single parameterised functions driven by config structs.
//!
//! ## Sub-module Structure
//!
//! - `NeuronVariantConfig` — configuration for add-neuron candidate variants
//! - `SynapseVariantConfig` — configuration for synapse candidate variants
//! - Pairing functions that combine originals with their variants

use crate::CandidateNeuronJson;
use crate::CandidateSynapseJson;
use crate::CoordinatedStructuralCandidateJson;
use crate::CoordinatedStructuralOpJson;

// ============================================================================
// Neuron variant configuration and generation
// ============================================================================

/// Configuration for generating a clamped/scaled variant of an add-neuron candidate.
///
/// Each variant strategy (conservative, gentle nudge, micro-nudge) is expressed
/// as a static `NeuronVariantConfig` instance rather than a separate function.
pub struct NeuronVariantConfig {
    /// Maximum absolute incoming weight (clamped symmetrically).
    pub incoming_abs_max: f32,
    /// Maximum absolute bias (clamped symmetrically).
    pub bias_abs_max: f32,
    /// Maximum absolute outgoing weight after scaling.
    pub outgoing_abs_max: f32,
    /// Scale factor applied to outgoing weight before clamping.
    pub outgoing_scale: f32,
    /// Multiplier for expected improvement (lower = less displacement in ranking).
    pub expected_multiplier: f32,
    /// Minimum non-zero outgoing weight fallback when scaling produces near-zero.
    pub min_outgoing_fallback: f32,
    /// Comment text applied to the variant.
    pub comment: &'static str,
}

/// Conservative: tight clamps on all parameters.
pub static CONSERVATIVE_CONFIG: NeuronVariantConfig = NeuronVariantConfig {
    incoming_abs_max: 2.0,
    bias_abs_max: 1.0,
    outgoing_abs_max: 0.05,
    outgoing_scale: 0.2,
    expected_multiplier: 0.5,
    min_outgoing_fallback: 0.01,
    comment: "Conservative variant (clamped incoming/bias, outgoing scaled)",
};

/// Gentle Nudge: moderate incoming/bias range, very small outgoing.
pub static GENTLE_NUDGE_CONFIG: NeuronVariantConfig = NeuronVariantConfig {
    incoming_abs_max: 20.0,
    bias_abs_max: 10.0,
    outgoing_abs_max: 0.02,
    outgoing_scale: 0.1,
    expected_multiplier: 0.75,
    min_outgoing_fallback: 0.005,
    comment: "Gentle Nudge variant (tight outgoing, bias tamed)",
};

/// Micro-Nudge: ultra-conservative, targeting ±0.002–0.005 outgoing range.
pub static MICRO_NUDGE_CONFIG: NeuronVariantConfig = NeuronVariantConfig {
    incoming_abs_max: 2.0,
    bias_abs_max: 1.0,
    outgoing_abs_max: 0.005,
    outgoing_scale: 0.05,
    expected_multiplier: 0.25,
    min_outgoing_fallback: 0.002,
    comment: "Micro-Nudge variant (ultra-conservative outgoing, tight incoming/bias)",
};

/// Create a variant of an add-neuron candidate using the given configuration.
///
/// Clamps incoming weight, bias, and outgoing weight according to the config,
/// scales expected improvement, and sets the comment.
pub fn make_neuron_variant(
    candidate: &CandidateNeuronJson,
    config: &NeuronVariantConfig,
) -> CandidateNeuronJson {
    let incoming_sign = if candidate.incoming_weight >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let outgoing_sign = if candidate.outgoing_weight >= 0.0 {
        1.0
    } else {
        -1.0
    };

    let incoming_weight =
        incoming_sign * candidate.incoming_weight.abs().min(config.incoming_abs_max);
    let bias = candidate
        .bias
        .clamp(-config.bias_abs_max, config.bias_abs_max);

    let mut outgoing_weight = (candidate.outgoing_weight * config.outgoing_scale)
        .clamp(-config.outgoing_abs_max, config.outgoing_abs_max);
    // Keep a small non-zero weight so the candidate actually does something.
    if outgoing_weight.abs() <= 1e-6 {
        outgoing_weight = outgoing_sign * config.min_outgoing_fallback;
    }

    let mut variant = candidate.clone();
    variant.incoming_weight = incoming_weight;
    variant.bias = bias;
    variant.outgoing_weight = outgoing_weight;
    variant.expected_creature_error_reduction *= config.expected_multiplier;
    variant.expected_creature_score_gain = variant.expected_creature_error_reduction;
    variant.comment = Some(config.comment.to_string());
    variant
}

// ============================================================================
// Synapse variant configuration and generation
// ============================================================================

/// Configuration for generating a scaled variant of a synapse candidate.
pub struct SynapseVariantConfig {
    /// Scale factor applied to the synapse weight.
    pub weight_scale: f32,
    /// Multiplier for expected improvement.
    pub expected_multiplier: f32,
    /// Comment text applied to the variant.
    pub comment: &'static str,
}

/// Conservative: halve the weight.
pub static SYNAPSE_CONSERVATIVE_CONFIG: SynapseVariantConfig = SynapseVariantConfig {
    weight_scale: 0.5,
    expected_multiplier: 0.5,
    comment: "Conservative variant (weight scaled to 0.5\u{00d7})",
};

/// Gentle Nudge: quarter the weight.
pub static SYNAPSE_GENTLE_NUDGE_CONFIG: SynapseVariantConfig = SynapseVariantConfig {
    weight_scale: 0.25,
    expected_multiplier: 0.75,
    comment: "Gentle Nudge variant (weight scaled to 0.25\u{00d7})",
};

/// Micro-Nudge: tenth of the weight.
pub static SYNAPSE_MICRO_NUDGE_CONFIG: SynapseVariantConfig = SynapseVariantConfig {
    weight_scale: 0.1,
    expected_multiplier: 0.25,
    comment: "Micro-Nudge variant (weight scaled to 0.1\u{00d7})",
};

/// Create a variant of a synapse candidate using the given configuration.
pub fn make_synapse_variant(
    candidate: &CandidateSynapseJson,
    config: &SynapseVariantConfig,
) -> CandidateSynapseJson {
    let mut variant = candidate.clone();
    variant.weight = candidate.weight * config.weight_scale;
    variant.expected_creature_error_reduction *= config.expected_multiplier;
    variant.expected_creature_score_gain *= config.expected_multiplier;
    variant.comment = Some(config.comment.to_string());
    variant
}

// ============================================================================
// Sensible-range filtering (production guard rails)
// ============================================================================

/// Maximum absolute incoming weight we consider "sensible" for add-neuron candidates.
const SENSIBLE_INCOMING_ABS_MAX: f32 = 20.0;

/// Maximum absolute bias we consider "sensible" for add-neuron candidates.
const SENSIBLE_BIAS_ABS_MAX: f32 = 10.0;

/// Maximum absolute outgoing weight we consider "sensible" for add-neuron candidates.
const SENSIBLE_OUTGOING_ABS_MAX: f32 = 0.1;

pub(crate) fn sensible_bias_abs_max_for_squash(_squash: &str) -> f32 {
    SENSIBLE_BIAS_ABS_MAX
}

/// Drop add-neuron candidates outside our "sensible" parameter ranges.
#[doc(hidden)]
pub fn filter_candidates_to_sensible_ranges(
    candidates: Vec<CandidateNeuronJson>,
) -> Vec<CandidateNeuronJson> {
    candidates
        .into_iter()
        .filter(|c| {
            c.incoming_weight.is_finite()
                && c.outgoing_weight.is_finite()
                && c.bias.is_finite()
                && c.incoming_weight.abs() <= SENSIBLE_INCOMING_ABS_MAX
                && c.outgoing_weight.abs() <= SENSIBLE_OUTGOING_ABS_MAX
                && c.bias.abs() <= sensible_bias_abs_max_for_squash(&c.squash)
        })
        .collect()
}

// ============================================================================
// Neuron candidate pairing
// ============================================================================

/// Returns true if this add-neuron candidate is "extreme" enough to warrant variants.
fn is_extreme_add_neuron_candidate(candidate: &CandidateNeuronJson) -> bool {
    candidate.incoming_weight.abs() > CONSERVATIVE_CONFIG.incoming_abs_max
        || candidate.bias.abs() > CONSERVATIVE_CONFIG.bias_abs_max
}

fn candidates_meaningfully_differ(a: &CandidateNeuronJson, b: &CandidateNeuronJson) -> bool {
    if a.source_neuron_uuid != b.source_neuron_uuid
        || a.target_neuron_uuid != b.target_neuron_uuid
        || a.source_neuron_index != b.source_neuron_index
        || a.target_neuron_index != b.target_neuron_index
        || a.squash != b.squash
    {
        return true;
    }

    (a.incoming_weight - b.incoming_weight).abs() > 1e-6
        || (a.bias - b.bias).abs() > 1e-6
        || (a.outgoing_weight - b.outgoing_weight).abs() > 1e-6
}

/// Check whether the micro-nudge variant would meaningfully differ from the conservative variant.
fn should_generate_micro_nudge(candidate: &CandidateNeuronJson) -> bool {
    let conservative_outgoing = (candidate.outgoing_weight * CONSERVATIVE_CONFIG.outgoing_scale)
        .clamp(
            -CONSERVATIVE_CONFIG.outgoing_abs_max,
            CONSERVATIVE_CONFIG.outgoing_abs_max,
        );
    conservative_outgoing.abs() > MICRO_NUDGE_CONFIG.outgoing_abs_max
}

/// Pair extreme candidates with variant strategies, respecting max_candidates.
///
/// Input is expected to be pre-sorted by expected_creature_score_gain (highest first).
#[doc(hidden)]
pub fn pair_extreme_candidates_with_conservative_variants(
    sorted_candidates: Vec<CandidateNeuronJson>,
    max_candidates: Option<usize>,
) -> Vec<CandidateNeuronJson> {
    let limit = max_candidates.unwrap_or(usize::MAX);
    if limit == 0 {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(sorted_candidates.len().min(limit));

    for candidate in sorted_candidates.into_iter() {
        if output.len() >= limit {
            break;
        }

        let should_pair = is_extreme_add_neuron_candidate(&candidate);
        let original_index = output.len();
        output.push(candidate.clone());

        let mut added_conservative = false;
        let mut added_gentle_nudge = false;
        let mut added_micro_nudge = false;
        if should_pair && output.len() < limit {
            let conservative = make_neuron_variant(&candidate, &CONSERVATIVE_CONFIG);
            if candidates_meaningfully_differ(&conservative, &candidate) {
                output.push(conservative);
                added_conservative = true;
            }
        }

        if should_pair && output.len() < limit {
            let gentle = make_neuron_variant(&candidate, &GENTLE_NUDGE_CONFIG);
            if candidates_meaningfully_differ(&gentle, &candidate)
                && output
                    .iter()
                    .all(|existing| candidates_meaningfully_differ(existing, &gentle))
            {
                output.push(gentle);
                added_gentle_nudge = true;
            }
        }

        if should_pair && output.len() < limit && should_generate_micro_nudge(&candidate) {
            let micro = make_neuron_variant(&candidate, &MICRO_NUDGE_CONFIG);
            if candidates_meaningfully_differ(&micro, &candidate)
                && output
                    .iter()
                    .all(|existing| candidates_meaningfully_differ(existing, &micro))
            {
                output.push(micro);
                added_micro_nudge = true;
            }
        }

        if should_pair && output[original_index].comment.is_none() {
            let mut parts = Vec::new();
            if added_conservative {
                parts.push("Conservative");
            }
            if added_gentle_nudge {
                parts.push("Gentle Nudge");
            }
            if added_micro_nudge {
                parts.push("Micro-Nudge");
            }
            output[original_index].comment = Some(if parts.is_empty() {
                "Extreme candidate (no safety variants included)".to_string()
            } else if parts.len() == 1 {
                format!("Extreme candidate (paired with {} variant only)", parts[0])
            } else {
                format!(
                    "Extreme candidate (paired with {} variants)",
                    parts.join(" + ")
                )
            });
        }
    }

    output
}

// ============================================================================
// Synapse candidate pairing
// ============================================================================

/// Minimum absolute weight for a synapse variant to be considered meaningful.
const SYNAPSE_VARIANT_MIN_WEIGHT: f32 = 1e-6;

fn synapse_candidates_meaningfully_differ(
    a: &CandidateSynapseJson,
    b: &CandidateSynapseJson,
) -> bool {
    if a.from_neuron_uuid != b.from_neuron_uuid || a.to_neuron_uuid != b.to_neuron_uuid {
        return true;
    }
    (a.weight - b.weight).abs() > SYNAPSE_VARIANT_MIN_WEIGHT
}

/// Pair each synapse candidate with weight variants (Issue #513).
#[doc(hidden)]
pub fn pair_synapse_candidates_with_weight_variants(
    sorted_candidates: Vec<CandidateSynapseJson>,
    max_candidates: Option<usize>,
) -> Vec<CandidateSynapseJson> {
    let limit = max_candidates.unwrap_or(usize::MAX);
    if limit == 0 {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(sorted_candidates.len().min(limit));

    let configs = [
        &SYNAPSE_CONSERVATIVE_CONFIG,
        &SYNAPSE_GENTLE_NUDGE_CONFIG,
        &SYNAPSE_MICRO_NUDGE_CONFIG,
    ];

    for candidate in sorted_candidates.into_iter() {
        if output.len() >= limit {
            break;
        }

        let original_index = output.len();
        output.push(candidate.clone());

        let mut added_names = Vec::new();

        for config in &configs {
            if output.len() >= limit {
                break;
            }
            let variant = make_synapse_variant(&candidate, config);
            if variant.weight.abs() > SYNAPSE_VARIANT_MIN_WEIGHT
                && synapse_candidates_meaningfully_differ(&variant, &candidate)
                && output
                    .iter()
                    .all(|existing| synapse_candidates_meaningfully_differ(existing, &variant))
            {
                // Extract the variant name from the comment for labelling the original.
                let name = if config.comment.starts_with("Conservative") {
                    "Conservative"
                } else if config.comment.starts_with("Gentle") {
                    "Gentle Nudge"
                } else {
                    "Micro-Nudge"
                };
                added_names.push(name);
                output.push(variant);
            }
        }

        if output[original_index].comment.is_none() && !added_names.is_empty() {
            output[original_index].comment = Some(if added_names.len() == 1 {
                format!("Original (paired with {} variant only)", added_names[0])
            } else {
                format!(
                    "Original (paired with {} variants)",
                    added_names.join(" + ")
                )
            });
        }
    }

    output
}

// ============================================================================
// Coordinated-structural weight variant generation (Issue #510)
// ============================================================================

/// Weight scaling factors for coordinated-structural candidate variants.
const COORDINATED_CONSERVATIVE_WEIGHT_SCALE: f32 = 0.2;
const COORDINATED_CONSERVATIVE_EXPECTED_MULTIPLIER: f32 = 0.5;

const COORDINATED_GENTLE_NUDGE_WEIGHT_SCALE: f32 = 0.1;
const COORDINATED_GENTLE_NUDGE_EXPECTED_MULTIPLIER: f32 = 0.75;

const COORDINATED_MICRO_NUDGE_WEIGHT_SCALE: f32 = 0.05;
const COORDINATED_MICRO_NUDGE_EXPECTED_MULTIPLIER: f32 = 0.25;

/// Minimum absolute weight for a coordinated-structural AddSynapse variant.
const COORDINATED_VARIANT_MIN_WEIGHT: f32 = 1e-6;

/// Scale AddSynapse weights in a coordinated-structural candidate by the given factor.
fn scale_coordinated_add_synapse_weights(
    original: &CoordinatedStructuralCandidateJson,
    scale: f32,
    expected_multiplier: f32,
    comment: &str,
) -> Option<CoordinatedStructuralCandidateJson> {
    let mut has_add_synapse = false;
    let mut differs = false;

    let scaled_ops: Vec<CoordinatedStructuralOpJson> = original
        .operations
        .iter()
        .map(|op| match op {
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                weight,
            } => {
                has_add_synapse = true;
                let scaled_weight = weight * scale;
                if (scaled_weight - weight).abs() > COORDINATED_VARIANT_MIN_WEIGHT {
                    differs = true;
                }
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: from_neuron_uuid.clone(),
                    to_neuron_uuid: to_neuron_uuid.clone(),
                    weight: scaled_weight,
                }
            }
            other => other.clone(),
        })
        .collect();

    if !has_add_synapse || !differs {
        return None;
    }

    let all_above_min = scaled_ops.iter().all(|op| match op {
        CoordinatedStructuralOpJson::AddSynapse { weight, .. } => {
            weight.abs() > COORDINATED_VARIANT_MIN_WEIGHT
        }
        _ => true,
    });

    if !all_above_min {
        return None;
    }

    Some(CoordinatedStructuralCandidateJson {
        operations: scaled_ops,
        expected_creature_score_gain: original.expected_creature_score_gain * expected_multiplier,
        comment: Some(comment.to_string()),
    })
}

/// Coordinated-structural variant specs: (scale, expected_multiplier, comment).
const COORDINATED_VARIANT_SPECS: [(f32, f32, &str); 3] = [
    (
        COORDINATED_CONSERVATIVE_WEIGHT_SCALE,
        COORDINATED_CONSERVATIVE_EXPECTED_MULTIPLIER,
        "Conservative variant (AddSynapse weights scaled to 0.2\u{00d7})",
    ),
    (
        COORDINATED_GENTLE_NUDGE_WEIGHT_SCALE,
        COORDINATED_GENTLE_NUDGE_EXPECTED_MULTIPLIER,
        "Gentle Nudge variant (AddSynapse weights scaled to 0.1\u{00d7})",
    ),
    (
        COORDINATED_MICRO_NUDGE_WEIGHT_SCALE,
        COORDINATED_MICRO_NUDGE_EXPECTED_MULTIPLIER,
        "Micro-Nudge variant (AddSynapse weights scaled to 0.05\u{00d7})",
    ),
];

/// Pair each coordinated-structural candidate with weight variants (Issue #510).
#[doc(hidden)]
pub fn pair_coordinated_structural_with_weight_variants(
    sorted_candidates: Vec<CoordinatedStructuralCandidateJson>,
    max_candidates: Option<usize>,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let limit = max_candidates.unwrap_or(usize::MAX);
    if limit == 0 {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(sorted_candidates.len().min(limit));

    for candidate in sorted_candidates.into_iter() {
        if output.len() >= limit {
            break;
        }

        let original_index = output.len();
        output.push(candidate.clone());

        let mut added_names: Vec<&str> = Vec::new();

        for &(scale, expected_multiplier, comment) in &COORDINATED_VARIANT_SPECS {
            if output.len() >= limit {
                break;
            }
            if let Some(variant) = scale_coordinated_add_synapse_weights(
                &candidate,
                scale,
                expected_multiplier,
                comment,
            ) {
                let name = if comment.starts_with("Conservative") {
                    "Conservative"
                } else if comment.starts_with("Gentle") {
                    "Gentle Nudge"
                } else {
                    "Micro-Nudge"
                };
                added_names.push(name);
                output.push(variant);
            }
        }

        if !added_names.is_empty() {
            let variant_suffix = if added_names.len() == 1 {
                format!(" (paired with {} variant only)", added_names[0])
            } else {
                format!(" (paired with {} variants)", added_names.join(" + "))
            };
            let existing = output[original_index].comment.take().unwrap_or_default();
            output[original_index].comment = Some(format!("{existing}{variant_suffix}"));
        }
    }

    output
}
