//! Utility functions for analysis module
//!
//! This module contains helper functions for memory checks, deadline handling,
//! activation functions, and other utilities used across analysis modules.

/// Check if verbose logging is enabled. Result is cached for performance.
/// Set `NEAT_AI_DISCOVERY_VERBOSE=1` to enable verbose logging.
/// This is public so it can be used by focus.rs and other modules.
pub fn verbose_enabled() -> bool {
    use std::sync::OnceLock;
    static VERBOSE: OnceLock<bool> = OnceLock::new();
    *VERBOSE.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok())
}

// ============================================================================
// Production experiment helpers
// ============================================================================

use crate::CandidateNeuronJson;

/// Conservative parameter clamps for add-neuron candidates.
///
/// Production evidence (Dec 2025): many failed add-neuron candidates are generated with very
/// large incoming weights and biases (eg incoming=200, bias=50). These can create a near-constant
/// or overly-aggressive correction, which tends to fail when evaluated on the full training set.
///
/// A simple production experiment is to return a *paired* candidate:
/// - the original ("extreme") candidate discovered by the current search
/// - a "conservative" variant with clamped incoming/bias and smaller outgoing weight
///
/// TypeScript still does the real validation by scoring both candidates.
const CONSERVATIVE_INCOMING_ABS_MAX: f32 = 2.0;
const CONSERVATIVE_BIAS_ABS_MAX: f32 = 1.0;
const CONSERVATIVE_OUTGOING_ABS_MAX: f32 = 0.05;
const CONSERVATIVE_OUTGOING_SCALE: f32 = 0.2;
const CONSERVATIVE_EXPECTED_MULTIPLIER: f32 = 0.5;

/// Returns true if this add-neuron candidate is "extreme" enough to warrant a conservative pair.
///
/// We intentionally base this on incoming weight and bias (not outgoing), because outgoing
/// weight is already clamped for add-neuron candidates and ReLU candidates commonly use
/// outgoing=0.1 by design.
fn is_extreme_add_neuron_candidate(candidate: &CandidateNeuronJson) -> bool {
    candidate.incoming_weight.abs() > CONSERVATIVE_INCOMING_ABS_MAX
        || candidate.bias.abs() > CONSERVATIVE_BIAS_ABS_MAX
}

/// Create a conservative variant of an add-neuron candidate.
///
/// Notes:
/// - We *do not* attempt to re-run the full optimisation here. This is intentionally cheap.
/// - We scale expected improvement down so the conservative variant doesn't displace the
///   original in ranking, while still being returned for evaluation by TypeScript.
fn make_conservative_add_neuron_variant(candidate: &CandidateNeuronJson) -> CandidateNeuronJson {
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

    let incoming_weight = incoming_sign
        * candidate
            .incoming_weight
            .abs()
            .min(CONSERVATIVE_INCOMING_ABS_MAX);
    let bias = candidate
        .bias
        .clamp(-CONSERVATIVE_BIAS_ABS_MAX, CONSERVATIVE_BIAS_ABS_MAX);

    let mut outgoing_weight = (candidate.outgoing_weight * CONSERVATIVE_OUTGOING_SCALE).clamp(
        -CONSERVATIVE_OUTGOING_ABS_MAX,
        CONSERVATIVE_OUTGOING_ABS_MAX,
    );
    // Keep a small non-zero weight so the candidate actually does something.
    if outgoing_weight.abs() <= 1e-6 {
        outgoing_weight = outgoing_sign * 0.01;
    }

    let mut conservative = candidate.clone();
    conservative.incoming_weight = incoming_weight;
    conservative.bias = bias;
    conservative.outgoing_weight = outgoing_weight;
    conservative.expected_creature_error_reduction *= CONSERVATIVE_EXPECTED_MULTIPLIER;
    conservative.expected_creature_score_gain = conservative.expected_creature_error_reduction;
    conservative.comment =
        Some("Conservative variant (clamped incoming/bias, outgoing scaled)".to_string());
    conservative
}

/// Pair extreme candidates with a conservative variant, respecting max_candidates.
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

    for mut candidate in sorted_candidates.into_iter() {
        if output.len() >= limit {
            break;
        }

        let should_pair = is_extreme_add_neuron_candidate(&candidate);
        if should_pair && candidate.comment.is_none() {
            candidate.comment =
                Some("Extreme candidate (paired with conservative variant)".to_string());
        }
        output.push(candidate.clone());

        if should_pair && output.len() < limit {
            let conservative = make_conservative_add_neuron_variant(&candidate);
            // Only add if it meaningfully differs (avoid duplicates).
            if (conservative.incoming_weight - candidate.incoming_weight).abs() > 1e-6
                || (conservative.bias - candidate.bias).abs() > 1e-6
                || (conservative.outgoing_weight - candidate.outgoing_weight).abs() > 1e-6
            {
                output.push(conservative);
            }
        }
    }

    output
}

// TODO: Move other utility functions from impl.rs here:
// - check_memory_for_parquet
// - get_memory_info (platform-specific)
// - parse_vm_stat_line, parse_vm_stat_page_size (macOS)
// - parse_meminfo_line (Linux)
// - detect_system_resources
// - build_deadline, deadline_passed, calculate_effective_timeout_ms
// - log_analysis_start, log_analysis_timeout
// - wait_for_buffer_map, wait_for_buffer_maps_batch
// - calculate_gpu_batch_timeout
// - All activation functions (identity_activation, tanh_activation, etc.)
// - is_threshold_activation, has_sufficient_output_variance
// - activation_name_to_gpu_id, ACTIVATION_SPECS
