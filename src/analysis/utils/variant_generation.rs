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
///
/// Issue #888: Tightened outgoing from 0.05→0.005 to match cache evidence
/// where successes cluster at 0.001–0.005.
pub static CONSERVATIVE_CONFIG: NeuronVariantConfig = NeuronVariantConfig {
    incoming_abs_max: 2.0,
    bias_abs_max: 1.0,
    outgoing_abs_max: 0.005,
    outgoing_scale: 0.2,
    expected_multiplier: 0.5,
    min_outgoing_fallback: 0.001,
    comment: "Conservative variant (clamped incoming/bias, outgoing scaled)",
};

/// Gentle Nudge: moderate incoming/bias range, very small outgoing.
///
/// Issue #888: Tightened incoming from 20→5, bias from 10→2, outgoing from
/// 0.02→0.01 to match GRQ-sampler cache evidence. The previous ranges
/// allowed values in the "Extreme" pattern that almost always fails.
pub static GENTLE_NUDGE_CONFIG: NeuronVariantConfig = NeuronVariantConfig {
    incoming_abs_max: 5.0,
    bias_abs_max: 2.0,
    outgoing_abs_max: 0.01,
    outgoing_scale: 0.1,
    expected_multiplier: 0.75,
    min_outgoing_fallback: 0.002,
    comment: "Gentle Nudge variant (tight outgoing, bias tamed)",
};

/// Micro-Nudge: ultra-conservative, targeting ±0.002–0.005 outgoing range.
///
/// Issue #888: Boosted `expected_multiplier` from 0.25 to 0.5 because
/// the Micro-Nudge pattern dominates successes (~90% of successful samples
/// in GRQ-sampler cache). The higher multiplier ensures these candidates
/// are prioritised in the ranking over other variants.
pub static MICRO_NUDGE_CONFIG: NeuronVariantConfig = NeuronVariantConfig {
    incoming_abs_max: 2.0,
    bias_abs_max: 1.0,
    outgoing_abs_max: 0.005,
    outgoing_scale: 0.05,
    expected_multiplier: 0.5,
    min_outgoing_fallback: 0.002,
    comment: "Micro-Nudge variant (ultra-conservative outgoing, tight incoming/bias)",
};

/// Feather-Touch: finer-grained variant for near-equilibrium networks (Issue #962).
///
/// Targets outgoing weights around ±0.001, below Micro-Nudge. Only generated
/// when the base weight is above `ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD`.
pub static FEATHER_TOUCH_CONFIG: NeuronVariantConfig = NeuronVariantConfig {
    incoming_abs_max: 2.0,
    bias_abs_max: 1.0,
    outgoing_abs_max: 0.002,
    outgoing_scale: 0.01,
    expected_multiplier: 0.25,
    min_outgoing_fallback: 0.0005,
    comment: "Feather-Touch variant (near-equilibrium outgoing, minimal perturbation)",
};

/// Whisper: the most conservative variant tier (Issue #962).
///
/// Targets outgoing weights around ±0.0005, the smallest perturbation we generate.
/// Only generated when the base weight is above `ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD`.
pub static WHISPER_CONFIG: NeuronVariantConfig = NeuronVariantConfig {
    incoming_abs_max: 2.0,
    bias_abs_max: 1.0,
    outgoing_abs_max: 0.001,
    outgoing_scale: 0.005,
    expected_multiplier: 0.1,
    min_outgoing_fallback: 0.0002,
    comment: "Whisper variant (minimal outgoing, near-zero perturbation)",
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

/// Feather-Touch: 5% of the weight (Issue #962).
///
/// Only generated when the base weight is above `ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD`.
pub static SYNAPSE_FEATHER_TOUCH_CONFIG: SynapseVariantConfig = SynapseVariantConfig {
    weight_scale: 0.05,
    expected_multiplier: 0.25,
    comment: "Feather-Touch variant (weight scaled to 0.05\u{00d7})",
};

/// Whisper: 2% of the weight (Issue #962).
///
/// Only generated when the base weight is above `ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD`.
pub static SYNAPSE_WHISPER_CONFIG: SynapseVariantConfig = SynapseVariantConfig {
    weight_scale: 0.02,
    expected_multiplier: 0.1,
    comment: "Whisper variant (weight scaled to 0.02\u{00d7})",
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
// Ultra-conservative threshold (Issue #962)
// ============================================================================

/// Minimum absolute base weight required before ultra-conservative variants
/// (Feather-Touch, Whisper) are generated. Prevents creating near-zero
/// variants of already-small weights.
pub const ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD: f32 = 0.05;

// ============================================================================
// Sensible-range filtering (production guard rails)
// ============================================================================

/// Maximum absolute incoming weight we consider "sensible" for add-neuron candidates.
///
/// Issue #888: Tightened from 20.0 to match `MAX_INCOMING_WEIGHT` (5.0).
/// GRQ-sampler cache shows incoming weights of 10+ almost always fail.
const SENSIBLE_INCOMING_ABS_MAX: f32 = 5.0;

/// Maximum absolute bias we consider "sensible" for add-neuron candidates.
///
/// Issue #888: Tightened from 10.0 to match `MAX_BIAS_MAGNITUDE` (2.0).
/// GRQ-sampler cache shows bias values outside [-1, 1] almost always fail.
const SENSIBLE_BIAS_ABS_MAX: f32 = 2.0;

/// Maximum absolute outgoing weight we consider "sensible" for add-neuron candidates.
///
/// Issue #888: Tightened from 0.1 to match `MAX_OUTGOING_WEIGHT` (0.01).
const SENSIBLE_OUTGOING_ABS_MAX: f32 = 0.01;

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

        // Ultra-conservative variants (Issue #962): only when base weight is large enough.
        let mut added_feather_touch = false;
        let mut added_whisper = false;
        let base_weight_above_threshold =
            candidate.outgoing_weight.abs() >= ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD;

        if should_pair && output.len() < limit && base_weight_above_threshold {
            let feather = make_neuron_variant(&candidate, &FEATHER_TOUCH_CONFIG);
            if candidates_meaningfully_differ(&feather, &candidate)
                && output
                    .iter()
                    .all(|existing| candidates_meaningfully_differ(existing, &feather))
            {
                output.push(feather);
                added_feather_touch = true;
            }
        }

        if should_pair && output.len() < limit && base_weight_above_threshold {
            let whisper = make_neuron_variant(&candidate, &WHISPER_CONFIG);
            if candidates_meaningfully_differ(&whisper, &candidate)
                && output
                    .iter()
                    .all(|existing| candidates_meaningfully_differ(existing, &whisper))
            {
                output.push(whisper);
                added_whisper = true;
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
            if added_feather_touch {
                parts.push("Feather-Touch");
            }
            if added_whisper {
                parts.push("Whisper");
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

/// Extract a human-readable variant name from a config comment string.
fn variant_name_from_comment(comment: &str) -> &'static str {
    if comment.starts_with("Conservative") {
        "Conservative"
    } else if comment.starts_with("Gentle") {
        "Gentle Nudge"
    } else if comment.starts_with("Micro") {
        "Micro-Nudge"
    } else if comment.starts_with("Feather") {
        "Feather-Touch"
    } else if comment.starts_with("Whisper") {
        "Whisper"
    } else {
        "Unknown"
    }
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

    let base_configs = [
        &SYNAPSE_CONSERVATIVE_CONFIG,
        &SYNAPSE_GENTLE_NUDGE_CONFIG,
        &SYNAPSE_MICRO_NUDGE_CONFIG,
    ];

    // Ultra-conservative configs gated by base weight threshold (Issue #962).
    let ultra_configs = [&SYNAPSE_FEATHER_TOUCH_CONFIG, &SYNAPSE_WHISPER_CONFIG];

    for candidate in sorted_candidates.into_iter() {
        if output.len() >= limit {
            break;
        }

        let original_index = output.len();
        output.push(candidate.clone());

        let mut added_names = Vec::new();

        for config in &base_configs {
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
                let name = variant_name_from_comment(config.comment);
                added_names.push(name);
                output.push(variant);
            }
        }

        // Only generate ultra-conservative variants when the base weight is large enough.
        if candidate.weight.abs() >= ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD {
            for config in &ultra_configs {
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
                    let name = variant_name_from_comment(config.comment);
                    added_names.push(name);
                    output.push(variant);
                }
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

/// Issue #962: Feather-Touch scaling for coordinated-structural candidates.
const COORDINATED_FEATHER_TOUCH_WEIGHT_SCALE: f32 = 0.01;
const COORDINATED_FEATHER_TOUCH_EXPECTED_MULTIPLIER: f32 = 0.25;

/// Issue #962: Whisper scaling for coordinated-structural candidates.
const COORDINATED_WHISPER_WEIGHT_SCALE: f32 = 0.005;
const COORDINATED_WHISPER_EXPECTED_MULTIPLIER: f32 = 0.1;

/// Minimum absolute weight for a coordinated-structural `AddSynapse` variant.
const COORDINATED_VARIANT_MIN_WEIGHT: f32 = 1e-6;

/// Scale `AddSynapse` weights in a coordinated-structural candidate by the given factor.
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

/// Coordinated-structural variant specs: (scale, `expected_multiplier`, comment).
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

/// Ultra-conservative coordinated-structural variant specs (Issue #962).
/// Only applied when the maximum `AddSynapse` weight in the candidate exceeds
/// `ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD`.
const COORDINATED_ULTRA_VARIANT_SPECS: [(f32, f32, &str); 2] = [
    (
        COORDINATED_FEATHER_TOUCH_WEIGHT_SCALE,
        COORDINATED_FEATHER_TOUCH_EXPECTED_MULTIPLIER,
        "Feather-Touch variant (AddSynapse weights scaled to 0.01\u{00d7})",
    ),
    (
        COORDINATED_WHISPER_WEIGHT_SCALE,
        COORDINATED_WHISPER_EXPECTED_MULTIPLIER,
        "Whisper variant (AddSynapse weights scaled to 0.005\u{00d7})",
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
                let name = variant_name_from_comment(comment);
                added_names.push(name);
                output.push(variant);
            }
        }

        // Ultra-conservative variants (Issue #962): only when the largest AddSynapse
        // weight in the candidate exceeds the threshold.
        let max_add_synapse_weight = candidate
            .operations
            .iter()
            .filter_map(|op| match op {
                CoordinatedStructuralOpJson::AddSynapse { weight, .. } => Some(weight.abs()),
                _ => None,
            })
            .fold(0.0_f32, f32::max);

        if max_add_synapse_weight >= ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD {
            for &(scale, expected_multiplier, comment) in &COORDINATED_ULTRA_VARIANT_SPECS {
                if output.len() >= limit {
                    break;
                }
                if let Some(variant) = scale_coordinated_add_synapse_weights(
                    &candidate,
                    scale,
                    expected_multiplier,
                    comment,
                ) {
                    let name = variant_name_from_comment(comment);
                    added_names.push(name);
                    output.push(variant);
                }
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
