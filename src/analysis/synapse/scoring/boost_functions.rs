//! Type-based scoring boosts for synapse and neuron candidates
//!
//! This module applies source-type, target-type, and activation-function-aware
//! boosts to candidate expected score gains (Issues #467, #468, #887).

#![allow(clippy::cast_possible_truncation)] // Intentional f64→f32 casts for scoring constants (Issue #873)
use crate::analysis::constants::{
    EXISTING_HIDDEN_TARGET_BOOST, HIDDEN_SOURCE_BOOST, INPUT_SOURCE_BOOST,
};
use crate::analysis::utils::parse_input_index;
use std::collections::HashMap;

// =============================================================================
// Source-Type Prioritisation (Issue #467)
// =============================================================================

/// Applies source-type boost to a candidate's expected score gain.
///
/// Input neurons as synapse sources have a 36.2% success rate compared to
/// 2.8–3.3% for hidden neurons (GRQ-sampler data). This function applies
/// [`INPUT_SOURCE_BOOST`] as a multiplier when the source neuron is an input
/// neuron (UUID matches `input-N` pattern).
///
/// Hidden neuron sources receive [`HIDDEN_SOURCE_BOOST`] (Issue #910) to
/// ensure hidden-to-hidden synapse candidates can compete more fairly against
/// input-sourced candidates, enabling the network to build deeper structures.
///
/// Output neurons receive no boost (multiplier = 1.0).
pub fn apply_source_type_boost(gain: f32, source_uuid: &str) -> f32 {
    if parse_input_index(source_uuid).is_some() {
        gain * INPUT_SOURCE_BOOST as f32
    } else if !source_uuid.starts_with("output") {
        // Hidden neuron source — apply modest boost (Issue #910)
        gain * HIDDEN_SOURCE_BOOST as f32
    } else {
        gain
    }
}

// =============================================================================
// Target-Type Prioritisation (Issue #468)
// =============================================================================

/// Applies target-type boost to a candidate's expected score gain.
///
/// Existing hidden neurons as targets have a 31.4% success rate compared to
/// 5.3–5.4% for output or discovery-hidden neurons (GRQ-sampler data). This
/// function applies [`EXISTING_HIDDEN_TARGET_BOOST`] as a multiplier when the
/// target neuron is an existing hidden neuron.
///
/// Output, input, constant, and unknown neurons receive no boost (multiplier = 1.0).
pub fn apply_target_type_boost(
    gain: f32,
    target_uuid: &str,
    neuron_type_map: &HashMap<&str, &str>,
) -> f32 {
    if neuron_type_map
        .get(target_uuid)
        .is_some_and(|t| *t == "hidden")
    {
        gain * EXISTING_HIDDEN_TARGET_BOOST as f32
    } else {
        gain
    }
}

// =============================================================================
// Activation-Function-Aware Neuron Scoring (Issue #887)
// =============================================================================

/// Applies activation-function-aware boost/penalty to a neuron candidate's
/// expected score gain (Issue #887).
///
/// GRQ-sampler cache analysis shows dramatic differences in success rates by
/// activation function (e.g., GELU at 60% vs `HARD_TANH` at 7.2%). This function
/// applies Bayesian-smoothed boost multipliers to prioritise candidates using
/// historically more successful activation functions.
///
/// The boost is applied as a direct multiplier on `expected_creature_score_gain`,
/// similar to [`apply_source_type_boost`] and [`apply_target_type_boost`].
pub fn apply_activation_neuron_boost(gain: f32, squash_name: &str) -> f32 {
    let boost = crate::analysis::constants::activation_neuron_boost(squash_name);
    gain * boost as f32
}
