//! Fan-in candidate generation module (Issue #908).
//!
//! Generates "fan-in" candidates where multiple inputs converge to a single
//! hidden neuron. When correlated input pairs are detected — i.e. inputs
//! whose activations jointly predict a target's error — a fan-in candidate
//! is emitted:
//!
//! ```text
//! input-A --+
//!           +--> [hidden (non-linear)] --> target
//! input-B --+
//! ```
//!
//! Non-linear activations (TANH, GELU) are preferred because IDENTITY would
//! reduce the fan-in to a linear combination, missing interaction effects.
//!
//! ## Detection Method
//!
//! 1. Identify target neurons (output, hidden) with error records.
//! 2. For each target, find input neurons whose activation correlates with
//!    the target error.
//! 3. For each pair of correlated inputs, check whether they have
//!    complementary activation patterns (low mutual correlation) — this
//!    indicates an interaction effect.
//! 4. Estimate combined improvement from the input pair via least-squares
//!    regression on the target error.
//! 5. Emit a `CoordinatedStructuralCandidateJson` with `AddNeuron` + two
//!    `AddSynapse` (inputs → hidden) + one `AddSynapse` (hidden → target).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)
#![allow(clippy::cast_possible_truncation)] // f64→f32 regression results are intentionally truncated

use std::collections::{HashMap, HashSet};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;
use crate::analysis::detection::stats::pearson_correlation;
use crate::analysis::quantised_error::is_quantised_zero_one;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum absolute correlation between an input's activation and a target's
/// error to consider that input as a fan-in contributor.
const INPUT_ERROR_CORRELATION_THRESHOLD: f32 = 0.3;

/// Maximum absolute correlation between two inputs for them to be considered
/// complementary (low redundancy).
const MAX_INPUT_MUTUAL_CORRELATION: f32 = 0.8;

/// Minimum combined improvement ratio vs the best single-input improvement
/// to justify a fan-in candidate over a simple single-input path.
const MIN_COMBINED_BENEFIT_RATIO: f32 = 1.05;

/// Maximum fan-in candidates returned per analysis run.
const MAX_FAN_IN_CANDIDATES: usize = 30;

/// Maximum number of input contributors to evaluate per target.
const MAX_INPUTS_PER_TARGET: usize = 15;

/// Non-linear activations preferred for fan-in neurons (interaction capture).
const FAN_IN_ACTIVATIONS: &[&str] = &["TANH", "GELU"];

/// A detected fan-in opportunity where multiple inputs converge to one target.
#[derive(Debug, Clone)]
pub struct FanInCandidate {
    /// UUIDs of the input neurons that converge.
    pub input_uuids: Vec<String>,
    /// UUID of the target neuron (output or hidden).
    pub target_uuid: String,
    /// Optimal weights from each input to the fan-in hidden neuron.
    pub input_weights: Vec<f32>,
    /// Weight from the fan-in hidden neuron to the target.
    pub output_weight: f32,
    /// Activation function for the fan-in hidden neuron.
    pub activation: String,
    /// Estimated creature score improvement.
    pub estimated_improvement: f32,
    /// Mean correlation between each input and the target error.
    pub mean_input_error_correlation: f32,
    /// Mutual correlation between the inputs (lower = more complementary).
    pub input_mutual_correlation: f32,
    /// Number of shared samples used for estimation.
    pub sample_count: usize,
    /// Descriptive reason for this candidate.
    pub reason: String,
}

/// Detect fan-in candidates by finding pairs of inputs whose activations
/// jointly predict a target neuron's error.
///
/// # Arguments
/// * `creature` - Network topology.
/// * `neuron_records` - Recorded activations and errors per neuron.
///
/// # Returns
/// Fan-in candidates sorted by estimated improvement (best first).
pub fn detect_fan_in_candidates(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<FanInCandidate> {
    if neuron_records.is_empty() {
        return Vec::new();
    }

    // Build record lookup by neuron UUID.
    let record_map: HashMap<&str, &[DiscoverRecord]> = neuron_records
        .iter()
        .map(|(uuid, recs)| (uuid.as_str(), recs.as_ref()))
        .collect();

    // Classify neurons.
    let neuron_types: HashMap<&str, &str> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.neuron_type.as_str()))
        .collect();

    // Build existing synapse set for deduplication.
    let existing_synapses: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    // Identify input neurons.
    let input_uuids: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();

    // Identify target neurons (output and hidden) with sufficient error records.
    let target_uuids: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output" || n.neuron_type == "hidden")
        .filter(|n| {
            record_map.get(n.uuid.as_str()).is_some_and(|recs| {
                recs.len() >= MIN_DISCOVERY_SAMPLE_COUNT
                    && recs.iter().any(|r| !r.errors.is_empty())
            })
        })
        .map(|n| n.uuid.as_str())
        .collect();

    let mut candidates = Vec::new();

    for &target_uuid in &target_uuids {
        let target_records = match record_map.get(target_uuid) {
            Some(r) => *r,
            None => continue,
        };

        // Build target error map: obs_index → first error value.
        let target_errors: HashMap<u32, f32> = target_records
            .iter()
            .filter(|r| !r.errors.is_empty())
            .map(|r| (r.obs_index, r.errors[0]))
            .collect();

        if target_errors.len() < MIN_DISCOVERY_SAMPLE_COUNT {
            continue;
        }

        // Score each input by correlation with target error.
        let mut input_scores: Vec<(&str, f32, HashMap<u32, f32>)> = Vec::new();

        for &input_uuid in &input_uuids {
            let input_records = match record_map.get(input_uuid) {
                Some(r) => *r,
                None => continue,
            };

            // Build input activation map.
            let input_activations: HashMap<u32, f32> = input_records
                .iter()
                .map(|r| (r.obs_index, r.activation))
                .collect();

            // Find shared observation indices.
            let shared_indices: Vec<u32> = target_errors
                .keys()
                .copied()
                .filter(|idx| input_activations.contains_key(idx))
                .collect();

            if shared_indices.len() < MIN_DISCOVERY_SAMPLE_COUNT {
                continue;
            }

            // Compute correlation between input activation and target error.
            let acts: Vec<f32> = shared_indices
                .iter()
                .map(|idx| input_activations[idx])
                .collect();
            let errs: Vec<f32> = shared_indices
                .iter()
                .map(|idx| target_errors[idx])
                .collect();

            let corr = pearson_correlation(&acts, &errs);
            if corr.abs() < INPUT_ERROR_CORRELATION_THRESHOLD {
                continue;
            }

            input_scores.push((input_uuid, corr, input_activations));
        }

        // Sort by absolute correlation strength and take top candidates.
        input_scores.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        input_scores.truncate(MAX_INPUTS_PER_TARGET);

        // Evaluate all pairs for fan-in opportunity.
        for i in 0..input_scores.len() {
            for j in (i + 1)..input_scores.len() {
                let (input_a_uuid, corr_a, ref acts_a) = input_scores[i];
                let (input_b_uuid, corr_b, ref acts_b) = input_scores[j];

                if let Some(candidate) = evaluate_fan_in_pair(
                    input_a_uuid,
                    input_b_uuid,
                    corr_a,
                    corr_b,
                    acts_a,
                    acts_b,
                    target_uuid,
                    &target_errors,
                    &existing_synapses,
                    &neuron_types,
                ) {
                    candidates.push(candidate);
                }
            }
        }
    }

    // Sort by estimated improvement (best first) and truncate.
    candidates.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(MAX_FAN_IN_CANDIDATES);

    candidates
}

/// Evaluate a pair of inputs for a fan-in candidate targeting a specific neuron.
#[allow(clippy::too_many_arguments)]
fn evaluate_fan_in_pair(
    input_a_uuid: &str,
    input_b_uuid: &str,
    corr_a: f32,
    corr_b: f32,
    acts_a: &HashMap<u32, f32>,
    acts_b: &HashMap<u32, f32>,
    target_uuid: &str,
    target_errors: &HashMap<u32, f32>,
    existing_synapses: &HashSet<(&str, &str)>,
    neuron_types: &HashMap<&str, &str>,
) -> Option<FanInCandidate> {
    // Find shared observation indices across all three neurons.
    let shared_indices: Vec<u32> = target_errors
        .keys()
        .copied()
        .filter(|idx| acts_a.contains_key(idx) && acts_b.contains_key(idx))
        .collect();

    if shared_indices.len() < MIN_DISCOVERY_SAMPLE_COUNT {
        return None;
    }

    // Check mutual correlation between the two inputs — we want complementary patterns.
    let a_vals: Vec<f32> = shared_indices.iter().map(|idx| acts_a[idx]).collect();
    let b_vals: Vec<f32> = shared_indices.iter().map(|idx| acts_b[idx]).collect();
    let mutual_corr = pearson_correlation(&a_vals, &b_vals);

    if mutual_corr.abs() > MAX_INPUT_MUTUAL_CORRELATION {
        return None; // Too similar; a single-input path would suffice.
    }

    let err_vals: Vec<f32> = shared_indices
        .iter()
        .map(|idx| target_errors[idx])
        .collect();

    // Compute individual improvements via least-squares: w = Σ(act × err) / Σ(act²).
    let individual_a = compute_least_squares_improvement(&a_vals, &err_vals);
    let individual_b = compute_least_squares_improvement(&b_vals, &err_vals);

    // Compute combined improvement: two-variable regression.
    let (weight_a, weight_b, combined_improvement) =
        compute_two_input_regression(&a_vals, &b_vals, &err_vals)?;

    // The combined improvement should exceed the best individual by a ratio.
    let best_individual = individual_a.max(individual_b);
    if best_individual <= 0.0 {
        return None;
    }
    if combined_improvement < best_individual * MIN_COMBINED_BENEFIT_RATIO {
        return None;
    }

    // Compute the target impact discount based on neuron type.
    let target_impact = match neuron_types.get(target_uuid) {
        Some(&"output") => 1.0_f32,
        _ => 0.5_f32, // Hidden neurons have discounted impact.
    };

    // Apply conservative scaling: fan-in adds multiple operations.
    let scaled_improvement = combined_improvement * target_impact * 0.01;

    if scaled_improvement <= 0.0 {
        return None;
    }

    // Choose activation: prefer TANH as default non-linear activation.
    let activation =
        select_fan_in_activation(existing_synapses, input_a_uuid, input_b_uuid, target_uuid);

    let mean_corr = (corr_a.abs() + corr_b.abs()) / 2.0;

    Some(FanInCandidate {
        input_uuids: vec![input_a_uuid.to_string(), input_b_uuid.to_string()],
        target_uuid: target_uuid.to_string(),
        input_weights: vec![weight_a, weight_b],
        output_weight: 0.1,
        activation: activation.to_string(),
        estimated_improvement: scaled_improvement,
        mean_input_error_correlation: mean_corr,
        input_mutual_correlation: mutual_corr.abs(),
        sample_count: shared_indices.len(),
        reason: format!(
            "Fan-in: {} and {} converge to {} via {} (corr {:.2}/{:.2}, mutual {:.2}, combined improvement {:.4})",
            input_a_uuid,
            input_b_uuid,
            target_uuid,
            activation,
            corr_a,
            corr_b,
            mutual_corr.abs(),
            scaled_improvement,
        ),
    })
}

/// Compute least-squares improvement for a single-input predictor:
/// minimise Σ(error - w × activation)² → w = Σ(act × err) / Σ(act²),
/// improvement = Σ(err²) - Σ(err - w × act)².
///
/// ## Quantised `{0, 1}` error regime (Issue #1249)
///
/// When errors are `CATEGORICAL_ERROR`-style misclassification flags the
/// SSE collapses to `Σ e = error_count` (because `e² = e`). The
/// least-squares slope `w` becomes a regression of the misclassification
/// flag onto the activation, and the SSE-improvement number is bounded
/// by the misclassification count rather than NEAT-AI's loss reduction.
/// Emitting such a value would mislead the downstream candidate ranker,
/// so this helper returns `0.0` for the quantised regime — the
/// `best_individual <= 0.0` guard in [`evaluate_fan_in_pair`] then drops
/// every fan-in pair targeting the affected neuron. See
/// `docs/COST_FUNCTION_NOTES.md` §4.7 for the per-cost catalogue.
fn compute_least_squares_improvement(activations: &[f32], errors: &[f32]) -> f32 {
    let n = activations.len().min(errors.len());
    if n < 2 {
        return 0.0;
    }

    // Issue #1249: quantised `{0, 1}` errors break the "improvement = SSE
    // reduction" identity. Gate the fan-in path off rather than emit a
    // misleading magnitude.
    if is_quantised_zero_one(&errors[..n]) {
        return 0.0;
    }

    let sum_act_sq: f32 = activations[..n].iter().map(|a| a * a).sum();
    if sum_act_sq < 1e-10 {
        return 0.0;
    }

    let sum_act_err: f32 = activations[..n]
        .iter()
        .zip(errors[..n].iter())
        .map(|(a, e)| a * e)
        .sum();

    let w = sum_act_err / sum_act_sq;

    let original_sse: f32 = errors[..n].iter().map(|e| e * e).sum();
    let residual_sse: f32 = activations[..n]
        .iter()
        .zip(errors[..n].iter())
        .map(|(a, e)| {
            let residual = e - w * a;
            residual * residual
        })
        .sum();

    (original_sse - residual_sse).max(0.0)
}

/// Two-input least-squares regression: minimise Σ(err - `w_a` × `act_a` - `w_b` × `act_b`)².
/// Returns (`weight_a`, `weight_b`, improvement) or `None` if underdetermined.
fn compute_two_input_regression(
    acts_a: &[f32],
    acts_b: &[f32],
    errors: &[f32],
) -> Option<(f32, f32, f32)> {
    let n = acts_a.len().min(acts_b.len()).min(errors.len());
    if n < 3 {
        return None; // Need at least 3 samples for 2-variable regression.
    }

    // Issue #1249: the SSE-improvement identity collapses for quantised
    // `{0, 1}` errors. Gate the pair regression off in the same way as
    // the single-input helper above; the per-input
    // `compute_least_squares_improvement` already rejects each input,
    // but this is defence-in-depth for callers that bypass the
    // per-input check.
    if is_quantised_zero_one(&errors[..n]) {
        return None;
    }

    // Normal equations: [a'a, a'b; b'a, b'b] [wa; wb] = [a'e; b'e]
    let mut aa = 0.0_f64;
    let mut ab = 0.0_f64;
    let mut bb = 0.0_f64;
    let mut ae = 0.0_f64;
    let mut be = 0.0_f64;
    let mut ee = 0.0_f64;

    for i in 0..n {
        let a = f64::from(acts_a[i]);
        let b = f64::from(acts_b[i]);
        let e = f64::from(errors[i]);
        aa += a * a;
        ab += a * b;
        bb += b * b;
        ae += a * e;
        be += b * e;
        ee += e * e;
    }

    // Solve 2x2 system via Cramer's rule.
    let det = aa * bb - ab * ab;
    if det.abs() < 1e-12 {
        return None; // Singular or near-singular (inputs perfectly correlated).
    }

    let wa = (bb * ae - ab * be) / det;
    let wb = (aa * be - ab * ae) / det;

    // Compute residual SSE.
    let mut residual_sse = 0.0_f64;
    for i in 0..n {
        let a = f64::from(acts_a[i]);
        let b = f64::from(acts_b[i]);
        let e = f64::from(errors[i]);
        let residual = e - wa * a - wb * b;
        residual_sse += residual * residual;
    }

    let improvement = (ee - residual_sse).max(0.0);

    Some((wa as f32, wb as f32, improvement as f32))
}

/// Select the activation function for a fan-in hidden neuron.
/// Prefers non-linear activations; IDENTITY is never used for fan-in.
fn select_fan_in_activation(
    _existing_synapses: &HashSet<(&str, &str)>,
    _input_a: &str,
    _input_b: &str,
    _target: &str,
) -> &'static str {
    // Default to TANH — the most common non-linear activation for
    // capturing interaction effects between inputs.
    FAN_IN_ACTIVATIONS[0]
}

/// Convert detected fan-in candidates into coordinated structural candidates
/// that can be applied atomically to the creature.
///
/// Each fan-in candidate generates:
/// - `AddNeuron` — a new hidden neuron with non-linear activation
/// - `AddSynapse` × N — one synapse per input to the hidden neuron
/// - `AddSynapse` × 1 — hidden neuron to target
pub fn fan_in_to_coordinated_candidates(
    candidates: &[FanInCandidate],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    candidates
        .iter()
        .filter_map(|c| fan_in_to_single_coordinated(c, creature))
        .collect()
}

/// Convert a single fan-in candidate to a coordinated structural candidate.
fn fan_in_to_single_coordinated(
    candidate: &FanInCandidate,
    creature: &CreatureJson,
) -> Option<CoordinatedStructuralCandidateJson> {
    if candidate.input_uuids.len() < 2 {
        return None;
    }

    // Generate deterministic UUID for the new hidden neuron from the
    // input UUIDs and target UUID (FNV-1a hash).
    let neuron_uuid = generate_fan_in_uuid(&candidate.input_uuids, &candidate.target_uuid);

    // Determine insertion point: place hidden neuron before the target.
    let insert_before = find_insert_position(creature, &candidate.target_uuid);

    let mut operations = Vec::new();

    // 1. Add the fan-in hidden neuron.
    operations.push(CoordinatedStructuralOpJson::AddNeuron {
        neuron_uuid: neuron_uuid.clone(),
        neuron_type: "hidden".to_string(),
        squash: candidate.activation.clone(),
        bias: 0.0,
        insert_before_neuron_uuid: insert_before,
    });

    // 2. Add synapses from each input to the hidden neuron.
    for (i, input_uuid) in candidate.input_uuids.iter().enumerate() {
        let weight = candidate.input_weights.get(i).copied().unwrap_or(0.5);
        operations.push(CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: input_uuid.clone(),
            to_neuron_uuid: neuron_uuid.clone(),
            weight,
        });
    }

    // 3. Add synapse from hidden neuron to target.
    operations.push(CoordinatedStructuralOpJson::AddSynapse {
        from_neuron_uuid: neuron_uuid,
        to_neuron_uuid: candidate.target_uuid.clone(),
        weight: candidate.output_weight,
    });

    Some(CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations,
        expected_creature_score_gain: candidate.estimated_improvement,
        comment: Some(candidate.reason.clone()),
    })
}

/// Generate a deterministic UUID for a fan-in hidden neuron using FNV-1a
/// hash of the sorted input UUIDs and target UUID.
fn generate_fan_in_uuid(input_uuids: &[String], target_uuid: &str) -> String {
    let mut sorted_inputs: Vec<&str> = input_uuids.iter().map(String::as_str).collect();
    sorted_inputs.sort();

    // FNV-1a 64-bit hash.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let fnv_prime: u64 = 0x0100_0000_01b3;

    for byte in b"fan-in:" {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(fnv_prime);
    }
    for input in &sorted_inputs {
        for byte in input.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(fnv_prime);
        }
        // Separator between inputs.
        hash ^= u64::from(b'+');
        hash = hash.wrapping_mul(fnv_prime);
    }
    for byte in b"->" {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(fnv_prime);
    }
    for byte in target_uuid.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(fnv_prime);
    }

    format!("fan-in-{hash:016x}")
}

/// Find the neuron UUID before which the new hidden neuron should be inserted
/// to maintain forward-only evaluation order.
fn find_insert_position(creature: &CreatureJson, target_uuid: &str) -> Option<String> {
    // Insert before the target neuron.
    creature
        .neurons
        .iter()
        .find(|n| n.uuid == target_uuid)
        .map(|n| n.uuid.clone())
}

/// Compute the mean absolute error from a set of error values.
fn _compute_mean_abs_error(errors: &[f32]) -> f32 {
    if errors.is_empty() {
        return 0.0;
    }
    let sum: f32 = errors.iter().map(|e| e.abs()).sum();
    sum / errors.len() as f32
}
