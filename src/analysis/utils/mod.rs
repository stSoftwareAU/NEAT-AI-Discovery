//! Utility functions for analysis module
//!
//! This module contains helper functions for memory checks, deadline handling,
//! activation functions, and other utilities used across analysis modules.
//!
//! ## Module Structure
//!
//! - `memory` - Memory detection and system requirements checking (Issue #267)
//! - `platform` - Platform-specific setup (Linux XDG, Mesa warnings) (Issue #267)
//! - `deadline` - Deadline handling and logging utilities (Issue #268)

pub mod deadline;
pub mod memory;
pub mod platform;

// ============================================================================
// Mutex helpers — graceful handling of poisoned mutexes (Issue #525)
// ============================================================================

use std::sync::Mutex;

/// Lock a mutex, returning an `anyhow::Error` instead of panicking if poisoned.
///
/// In an FFI library a panic unwinds into the calling process (Deno / NEAT-AI)
/// and causes an unexpected abort. This helper converts a `PoisonError` into a
/// recoverable `anyhow::Error` that propagates to the JSON `success: false`
/// boundary.
///
/// The `context` parameter is included in the error message for diagnostics.
pub fn lock_or_bail<'a, T>(
    mutex: &'a Mutex<T>,
    context: &str,
) -> anyhow::Result<std::sync::MutexGuard<'a, T>> {
    mutex.lock().map_err(|_| {
        anyhow::anyhow!("Mutex poisoned ({context}): a thread panicked while holding this lock")
    })
}

/// Consume a mutex and return its inner value, or an error if poisoned.
///
/// Equivalent to `Mutex::into_inner().unwrap()` but returns an error instead
/// of panicking.
pub fn into_inner_or_bail<T>(mutex: Mutex<T>, context: &str) -> anyhow::Result<T> {
    mutex.into_inner().map_err(|_| {
        anyhow::anyhow!(
            "Mutex poisoned on into_inner ({context}): a thread panicked while holding this lock"
        )
    })
}

// Re-export key memory functions for convenience
pub use memory::{
    DEFAULT_GPU_BATCH_SIZE, HIGH_PERF_GPU_BATCH_SIZE, LOW_MEMORY_GPU_BATCH_SIZE, MemoryPressure,
    MemoryTier, cap_gpu_batch_size_by_bytes, categorise_memory_pressure, categorise_memory_tier,
    check_memory_for_parquet, check_system_memory_requirements, detect_memory_pressure,
    detect_memory_tier, get_memory_info, get_work_queue_capacity, get_work_queue_capacity_for_tier,
    validate_parquet_memory_requirements,
};

// Re-export platform setup functions
pub use platform::{ensure_xdg_runtime_dir, suppress_mesa_warnings_if_requested};

// Re-export deadline handling functions (Issue #268)
pub use deadline::{
    DEFAULT_DURATION_MS, GPU_QUEUE_TIMEOUT_MAX_SECS, GPU_QUEUE_TIMEOUT_MIN_SECS, MAX_DURATION_MS,
    MIN_DURATION_MS, OrderedNeuron, YEAR_2000_MS, build_deadline, calculate_effective_timeout_ms,
    calculate_gpu_batch_timeout, deadline_passed, derive_seed, focus_unused_observations_from_env,
    log_analysis_start, log_analysis_timeout, order_eligible_sources, order_focus_targets,
    parse_input_index, shuffle_slice, shuffle_within_top_k, source_input_index_bias_from_env,
};

// Re-export deadline override for tests
#[cfg(test)]
pub use deadline::deadline_override;

/// Check if verbose logging is enabled. Result is cached for performance.
/// Set `NEAT_AI_DISCOVERY_VERBOSE=1` to enable verbose logging.
/// This is public so it can be used by focus.rs and other modules.
pub fn verbose_enabled() -> bool {
    use std::sync::OnceLock;
    static VERBOSE: OnceLock<bool> = OnceLock::new();
    *VERBOSE.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok())
}

/// Check if GPU timing is enabled. Result is cached for performance.
/// Set `NEAT_AI_DISCOVERY_GPU_TIMING=1` to enable GPU timing collection.
/// This adds ~5% overhead when enabled but provides detailed timing breakdown.
pub fn gpu_timing_enabled() -> bool {
    use std::sync::OnceLock;
    static GPU_TIMING: OnceLock<bool> = OnceLock::new();
    *GPU_TIMING.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_GPU_TIMING").is_ok())
}

// ============================================================================
// Production experiment helpers
// ============================================================================

use crate::CandidateNeuronJson;
use crate::CandidateSynapseJson;
use crate::CoordinatedStructuralCandidateJson;
use crate::CoordinatedStructuralOpJson;

// ============================================================================
// Sensible-range filtering (production guard rails)
// ============================================================================

/// Maximum absolute incoming weight we consider "sensible" for add-neuron candidates.
///
/// Rationale (Dec 2025): very large incoming weights (50, 100, 200) frequently produce
/// brittle candidates that look positive under the linear prediction but fail when
/// applied to the full training set.
const SENSIBLE_INCOMING_ABS_MAX: f32 = 20.0;

/// Maximum absolute bias we consider "sensible" for add-neuron candidates.
///
/// Rationale (Dec 2025): large |bias| often turns the new neuron into a near-constant
/// offset (or forces saturation), which tends to crater performance in ablation tests.
const SENSIBLE_BIAS_ABS_MAX: f32 = 10.0;

/// Maximum absolute outgoing weight we consider "sensible" for add-neuron candidates.
///
/// This matches `MAX_OUTGOING_WEIGHT` in the analysis implementation. Keeping this
/// aligned avoids surprising "why is this candidate rejected?" behaviour.
const SENSIBLE_OUTGOING_ABS_MAX: f32 = 0.1;

pub(crate) fn sensible_bias_abs_max_for_squash(_squash: &str) -> f32 {
    // For now we keep this uniform across squashes. If we find a function that
    // legitimately needs a wider bias range, we can special-case it here.
    SENSIBLE_BIAS_ABS_MAX
}

/// Drop add-neuron candidates that are outside our "sensible" parameter ranges.
///
/// We do not mutate candidates here (no clamping). If a candidate is out of range,
/// it's simply not worth returning because TypeScript will cache the failure and
/// never re-try it.
///
/// Note: This is applied AFTER the "Extreme -> Conservative/Gentle Nudge" pairing, so
/// unsafe originals can be dropped while still keeping safe variants.
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

/// Guard rails for the "Gentle Nudge" safety variant.
///
/// The intent is to preserve a meaningful incoming/bias range (so the neuron can still
/// represent a useful feature), while keeping the outgoing effect small and stable.
///
/// This is a third candidate returned alongside the original and the conservative clamp.
const GENTLE_NUDGE_INCOMING_ABS_MAX: f32 = 20.0;
const GENTLE_NUDGE_BIAS_ABS_MAX: f32 = 10.0;
const GENTLE_NUDGE_OUTGOING_ABS_MAX: f32 = 0.02;
const GENTLE_NUDGE_OUTGOING_SCALE: f32 = 0.1;
const GENTLE_NUDGE_EXPECTED_MULTIPLIER: f32 = 0.75;

/// Guard rails for the "Micro-Nudge" safety variant (Issue #507).
///
/// Ultra-conservative variant targeting the ±0.002–0.005 outgoing weight range.
/// Production evidence (Feb 2026): several conservative and gentle-nudge candidates
/// narrowly missed success with score deltas within 0.0003 of zero. The micro-nudge
/// variant explores even smaller outgoing weights where the sweet spot likely lies.
///
/// Only generated when the conservative variant's outgoing weight exceeds
/// `MICRO_NUDGE_OUTGOING_ABS_MAX` (i.e. when it would meaningfully differ).
const MICRO_NUDGE_INCOMING_ABS_MAX: f32 = 2.0;
const MICRO_NUDGE_BIAS_ABS_MAX: f32 = 1.0;
const MICRO_NUDGE_OUTGOING_ABS_MAX: f32 = 0.005;
const MICRO_NUDGE_OUTGOING_SCALE: f32 = 0.05;
const MICRO_NUDGE_EXPECTED_MULTIPLIER: f32 = 0.25;

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

/// Create a "Gentle Nudge" variant of an add-neuron candidate.
///
/// This is intentionally more permissive than the conservative clamp:
/// - incoming is clamped to a moderate range (not forced all the way down to ~2.0)
/// - bias is clamped to a range that still allows threshold crossing in many squashes
/// - outgoing weight is kept very small (a gentle correction rather than a shove)
fn make_gentle_nudge_add_neuron_variant(candidate: &CandidateNeuronJson) -> CandidateNeuronJson {
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
            .min(GENTLE_NUDGE_INCOMING_ABS_MAX);
    let bias = candidate
        .bias
        .clamp(-GENTLE_NUDGE_BIAS_ABS_MAX, GENTLE_NUDGE_BIAS_ABS_MAX);

    let mut outgoing_weight = (candidate.outgoing_weight * GENTLE_NUDGE_OUTGOING_SCALE).clamp(
        -GENTLE_NUDGE_OUTGOING_ABS_MAX,
        GENTLE_NUDGE_OUTGOING_ABS_MAX,
    );
    // Keep a small non-zero weight so the candidate actually does something.
    if outgoing_weight.abs() <= 1e-6 {
        outgoing_weight = outgoing_sign * 0.005;
    }

    let mut gentle = candidate.clone();
    gentle.incoming_weight = incoming_weight;
    gentle.bias = bias;
    gentle.outgoing_weight = outgoing_weight;
    gentle.expected_creature_error_reduction *= GENTLE_NUDGE_EXPECTED_MULTIPLIER;
    gentle.expected_creature_score_gain = gentle.expected_creature_error_reduction;
    gentle.comment = Some("Gentle Nudge variant (tight outgoing, bias tamed)".to_string());
    gentle
}

/// Create a "Micro-Nudge" variant of an add-neuron candidate (Issue #507).
///
/// Ultra-conservative: the outgoing weight is halved again from the conservative variant,
/// targeting the ±0.002–0.005 range where near-miss production candidates clustered.
fn make_micro_nudge_add_neuron_variant(candidate: &CandidateNeuronJson) -> CandidateNeuronJson {
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
            .min(MICRO_NUDGE_INCOMING_ABS_MAX);
    let bias = candidate
        .bias
        .clamp(-MICRO_NUDGE_BIAS_ABS_MAX, MICRO_NUDGE_BIAS_ABS_MAX);

    let mut outgoing_weight = (candidate.outgoing_weight * MICRO_NUDGE_OUTGOING_SCALE)
        .clamp(-MICRO_NUDGE_OUTGOING_ABS_MAX, MICRO_NUDGE_OUTGOING_ABS_MAX);
    // Keep a small non-zero weight so the candidate actually does something.
    if outgoing_weight.abs() <= 1e-6 {
        outgoing_weight = outgoing_sign * 0.002;
    }

    let mut micro = candidate.clone();
    micro.incoming_weight = incoming_weight;
    micro.bias = bias;
    micro.outgoing_weight = outgoing_weight;
    micro.expected_creature_error_reduction *= MICRO_NUDGE_EXPECTED_MULTIPLIER;
    micro.expected_creature_score_gain = micro.expected_creature_error_reduction;
    micro.comment =
        Some("Micro-Nudge variant (ultra-conservative outgoing, tight incoming/bias)".to_string());
    micro
}

/// Check whether the micro-nudge variant would meaningfully differ from the conservative variant.
///
/// Only generate micro-nudge when the conservative outgoing weight exceeds the micro-nudge max.
/// This avoids wasting the candidate budget on near-identical variants.
fn should_generate_micro_nudge(candidate: &CandidateNeuronJson) -> bool {
    let conservative_outgoing = (candidate.outgoing_weight * CONSERVATIVE_OUTGOING_SCALE).clamp(
        -CONSERVATIVE_OUTGOING_ABS_MAX,
        CONSERVATIVE_OUTGOING_ABS_MAX,
    );
    conservative_outgoing.abs() > MICRO_NUDGE_OUTGOING_ABS_MAX
}

fn candidates_meaningfully_differ(a: &CandidateNeuronJson, b: &CandidateNeuronJson) -> bool {
    // Two candidates connecting different neuron pairs (or using different neuron squash)
    // are fundamentally different operations, even if clamping produces identical weights.
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
            let conservative = make_conservative_add_neuron_variant(&candidate);
            // Only add if it meaningfully differs (avoid duplicates).
            if candidates_meaningfully_differ(&conservative, &candidate) {
                output.push(conservative);
                added_conservative = true;
            }
        }

        if should_pair && output.len() < limit {
            let gentle = make_gentle_nudge_add_neuron_variant(&candidate);
            // Only add if it meaningfully differs (avoid duplicates).
            if candidates_meaningfully_differ(&gentle, &candidate)
                && output
                    .iter()
                    .all(|existing| candidates_meaningfully_differ(existing, &gentle))
            {
                output.push(gentle);
                added_gentle_nudge = true;
            }
        }

        // Issue #507: Micro-Nudge variant — only when conservative outgoing exceeds
        // the micro-nudge max (otherwise the two variants would be near-identical).
        if should_pair && output.len() < limit && should_generate_micro_nudge(&candidate) {
            let micro = make_micro_nudge_add_neuron_variant(&candidate);
            if candidates_meaningfully_differ(&micro, &candidate)
                && output
                    .iter()
                    .all(|existing| candidates_meaningfully_differ(existing, &micro))
            {
                output.push(micro);
                added_micro_nudge = true;
            }
        }

        // Avoid misleading diagnostics: only claim pairing once we have actually returned
        // variants (and we had room under max_candidates).
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
// Synapse weight variant generation (Issue #513)
// ============================================================================

/// Weight scaling factors for synapse candidate variants.
///
/// Synapse candidates have a single `weight` field (already clamped to ±0.1).
/// We generate scaled-down variants so TypeScript can test multiple weight
/// magnitudes for the same from→to pair. The discovery process is expensive
/// (~1 hour), but testing each candidate is cheap (~1 minute), so generating
/// multiple variants maximises the return on discovery investment.
const SYNAPSE_CONSERVATIVE_WEIGHT_SCALE: f32 = 0.5;
const SYNAPSE_CONSERVATIVE_EXPECTED_MULTIPLIER: f32 = 0.5;

const SYNAPSE_GENTLE_NUDGE_WEIGHT_SCALE: f32 = 0.25;
const SYNAPSE_GENTLE_NUDGE_EXPECTED_MULTIPLIER: f32 = 0.75;

const SYNAPSE_MICRO_NUDGE_WEIGHT_SCALE: f32 = 0.1;
const SYNAPSE_MICRO_NUDGE_EXPECTED_MULTIPLIER: f32 = 0.25;

/// Minimum absolute weight for a synapse variant to be considered meaningful.
/// Below this, the variant is indistinguishable from zero and would be wasted.
const SYNAPSE_VARIANT_MIN_WEIGHT: f32 = 1e-6;

/// Check whether two synapse candidates are meaningfully different.
fn synapse_candidates_meaningfully_differ(
    a: &CandidateSynapseJson,
    b: &CandidateSynapseJson,
) -> bool {
    if a.from_neuron_uuid != b.from_neuron_uuid || a.to_neuron_uuid != b.to_neuron_uuid {
        return true;
    }
    (a.weight - b.weight).abs() > SYNAPSE_VARIANT_MIN_WEIGHT
}

/// Create a conservative weight variant of a synapse candidate.
fn make_conservative_synapse_variant(candidate: &CandidateSynapseJson) -> CandidateSynapseJson {
    let mut variant = candidate.clone();
    variant.weight = candidate.weight * SYNAPSE_CONSERVATIVE_WEIGHT_SCALE;
    variant.expected_creature_error_reduction *= SYNAPSE_CONSERVATIVE_EXPECTED_MULTIPLIER;
    variant.expected_creature_score_gain *= SYNAPSE_CONSERVATIVE_EXPECTED_MULTIPLIER;
    variant.comment = Some("Conservative variant (weight scaled to 0.5×)".to_string());
    variant
}

/// Create a gentle-nudge weight variant of a synapse candidate.
fn make_gentle_nudge_synapse_variant(candidate: &CandidateSynapseJson) -> CandidateSynapseJson {
    let mut variant = candidate.clone();
    variant.weight = candidate.weight * SYNAPSE_GENTLE_NUDGE_WEIGHT_SCALE;
    variant.expected_creature_error_reduction *= SYNAPSE_GENTLE_NUDGE_EXPECTED_MULTIPLIER;
    variant.expected_creature_score_gain *= SYNAPSE_GENTLE_NUDGE_EXPECTED_MULTIPLIER;
    variant.comment = Some("Gentle Nudge variant (weight scaled to 0.25×)".to_string());
    variant
}

/// Create a micro-nudge weight variant of a synapse candidate.
fn make_micro_nudge_synapse_variant(candidate: &CandidateSynapseJson) -> CandidateSynapseJson {
    let mut variant = candidate.clone();
    variant.weight = candidate.weight * SYNAPSE_MICRO_NUDGE_WEIGHT_SCALE;
    variant.expected_creature_error_reduction *= SYNAPSE_MICRO_NUDGE_EXPECTED_MULTIPLIER;
    variant.expected_creature_score_gain *= SYNAPSE_MICRO_NUDGE_EXPECTED_MULTIPLIER;
    variant.comment = Some("Micro-Nudge variant (weight scaled to 0.1×)".to_string());
    variant
}

/// Pair each helpful synapse candidate with weight variants (Issue #513).
///
/// For every synapse candidate, generates up to three additional variants with
/// progressively smaller weights. This gives the TypeScript controller multiple
/// options to test for the same from→to synapse pair — the discovery process is
/// expensive but testing each variant is cheap.
///
/// Variants are only added when they meaningfully differ from the original and
/// from each other (weight difference > 1e-6).
///
/// Input is expected to be pre-sorted by expected_creature_score_gain (highest first).
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

    for candidate in sorted_candidates.into_iter() {
        if output.len() >= limit {
            break;
        }

        let original_index = output.len();
        output.push(candidate.clone());

        let mut added_conservative = false;
        let mut added_gentle_nudge = false;
        let mut added_micro_nudge = false;

        // Conservative variant (0.5× weight)
        if output.len() < limit {
            let conservative = make_conservative_synapse_variant(&candidate);
            if conservative.weight.abs() > SYNAPSE_VARIANT_MIN_WEIGHT
                && synapse_candidates_meaningfully_differ(&conservative, &candidate)
            {
                output.push(conservative);
                added_conservative = true;
            }
        }

        // Gentle Nudge variant (0.25× weight)
        if output.len() < limit {
            let gentle = make_gentle_nudge_synapse_variant(&candidate);
            if gentle.weight.abs() > SYNAPSE_VARIANT_MIN_WEIGHT
                && synapse_candidates_meaningfully_differ(&gentle, &candidate)
                && output
                    .iter()
                    .all(|existing| synapse_candidates_meaningfully_differ(existing, &gentle))
            {
                output.push(gentle);
                added_gentle_nudge = true;
            }
        }

        // Micro-Nudge variant (0.1× weight)
        if output.len() < limit {
            let micro = make_micro_nudge_synapse_variant(&candidate);
            if micro.weight.abs() > SYNAPSE_VARIANT_MIN_WEIGHT
                && synapse_candidates_meaningfully_differ(&micro, &candidate)
                && output
                    .iter()
                    .all(|existing| synapse_candidates_meaningfully_differ(existing, &micro))
            {
                output.push(micro);
                added_micro_nudge = true;
            }
        }

        // Label the original with the variants that were included.
        if output[original_index].comment.is_none() {
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
            if !parts.is_empty() {
                output[original_index].comment = Some(if parts.len() == 1 {
                    format!("Original (paired with {} variant only)", parts[0])
                } else {
                    format!("Original (paired with {} variants)", parts.join(" + "))
                });
            }
        }
    }

    output
}

// ============================================================================
// Coordinated-structural weight variant generation (Issue #510)
// ============================================================================

/// Weight scaling factors for coordinated-structural candidate variants.
///
/// Coordinated-structural candidates contain AddSynapse operations whose weights
/// are derived from `source.optimal_weight` (often 0.1). Production evidence
/// (creature b2ff6e45, GRQ-sampler commit a1340f8d) showed that this magnitude
/// consistently fails — the same pattern seen with add-neuron extreme candidates.
///
/// We reuse the same conservative scaling strategy as synapse candidates (#513),
/// generating scaled-down variants so the TypeScript controller can test multiple
/// weight magnitudes for the same coordinated pair.
const COORDINATED_CONSERVATIVE_WEIGHT_SCALE: f32 = 0.2;
const COORDINATED_CONSERVATIVE_EXPECTED_MULTIPLIER: f32 = 0.5;

const COORDINATED_GENTLE_NUDGE_WEIGHT_SCALE: f32 = 0.1;
const COORDINATED_GENTLE_NUDGE_EXPECTED_MULTIPLIER: f32 = 0.75;

const COORDINATED_MICRO_NUDGE_WEIGHT_SCALE: f32 = 0.05;
const COORDINATED_MICRO_NUDGE_EXPECTED_MULTIPLIER: f32 = 0.25;

/// Minimum absolute weight for a coordinated-structural AddSynapse variant.
/// Below this, the variant is indistinguishable from zero and would be wasted.
const COORDINATED_VARIANT_MIN_WEIGHT: f32 = 1e-6;

/// Scale AddSynapse weights in a coordinated-structural candidate by the given factor.
///
/// Non-AddSynapse operations (RemoveSynapse, RemoveNeuron, etc.) are preserved
/// unchanged. Returns `None` if the candidate has no AddSynapse operations or
/// the scaled weights are too close to the originals to be meaningful.
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

    // Check all AddSynapse weights are above minimum threshold
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

/// Pair each coordinated-structural candidate with weight variants (Issue #510).
///
/// For every coordinated-structural candidate that contains AddSynapse operations,
/// generates up to three additional variants with progressively smaller weights.
/// The scaling factors match the conservative strategy proven successful for
/// add-neuron candidates: 0.2× (conservative), 0.1× (gentle nudge), 0.05× (micro-nudge).
///
/// Non-AddSynapse operations (RemoveSynapse, RemoveNeuron, etc.) are preserved
/// unchanged in all variants. Candidates without any AddSynapse operations are
/// returned as-is with no variants.
///
/// Input is expected to be pre-sorted by expected_creature_score_gain (highest first).
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

        let mut added_conservative = false;
        let mut added_gentle_nudge = false;
        let mut added_micro_nudge = false;

        // Conservative variant (0.2× weight — targeting ~0.02 for 0.1 originals)
        if output.len() < limit
            && let Some(conservative) = scale_coordinated_add_synapse_weights(
                &candidate,
                COORDINATED_CONSERVATIVE_WEIGHT_SCALE,
                COORDINATED_CONSERVATIVE_EXPECTED_MULTIPLIER,
                "Conservative variant (AddSynapse weights scaled to 0.2×)",
            )
        {
            output.push(conservative);
            added_conservative = true;
        }

        // Gentle Nudge variant (0.1× weight — targeting ~0.01 for 0.1 originals)
        if output.len() < limit
            && let Some(gentle) = scale_coordinated_add_synapse_weights(
                &candidate,
                COORDINATED_GENTLE_NUDGE_WEIGHT_SCALE,
                COORDINATED_GENTLE_NUDGE_EXPECTED_MULTIPLIER,
                "Gentle Nudge variant (AddSynapse weights scaled to 0.1×)",
            )
        {
            output.push(gentle);
            added_gentle_nudge = true;
        }

        // Micro-Nudge variant (0.05× weight — targeting ~0.005 for 0.1 originals)
        if output.len() < limit
            && let Some(micro) = scale_coordinated_add_synapse_weights(
                &candidate,
                COORDINATED_MICRO_NUDGE_WEIGHT_SCALE,
                COORDINATED_MICRO_NUDGE_EXPECTED_MULTIPLIER,
                "Micro-Nudge variant (AddSynapse weights scaled to 0.05×)",
            )
        {
            output.push(micro);
            added_micro_nudge = true;
        }

        // Append variant info to the original's comment (preserve existing context).
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
        if !parts.is_empty() {
            let variant_suffix = if parts.len() == 1 {
                format!(" (paired with {} variant only)", parts[0])
            } else {
                format!(" (paired with {} variants)", parts.join(" + "))
            };
            let existing = output[original_index].comment.take().unwrap_or_default();
            output[original_index].comment = Some(format!("{existing}{variant_suffix}"));
        }
    }

    output
}
