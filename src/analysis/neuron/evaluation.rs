//! Neuron candidate GPU evaluation — `ReLU` split evaluation and activation
//! spec batched evaluation for neuron candidates.
//!
//! Extracted from neuron.rs as part of issue #598.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::CandidateNeuronJson;
use anyhow::Result;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use crate::analysis::diagnostics::NeuronDiagnostics;
use crate::analysis::gpu::GpuWorkQueue;
use crate::analysis::samples::{EPSILON, HelpfulSample, compute_source_variance_discount};
use crate::analysis::scoring::cross_validation::{
    CrossValidationConfig, compute_cross_validation_score,
};
use crate::analysis::shared::TimingScope;
use crate::analysis::utils::{deadline_passed, lock_or_bail, verbose_enabled};

use crate::analysis::synapse::{
    apply_activation_neuron_boost, evaluate_all_activation_specs_batched,
    evaluate_relu_candidates_split, upsert_candidate,
};

/// Work result from sample building phase, containing samples for a single
/// source neuron.
///
/// Uses `&str` for `source_uuid` to avoid cloning UUIDs in the hot path
/// (Issue #808). The reference points to `OrderedNeuron.uuid` from the
/// locality group's borrowed source neurons.
pub(crate) struct NeuronWorkResult<'a> {
    pub source_uuid: &'a str,
    pub samples: Vec<HelpfulSample>,
}

/// Shared context for neuron candidate evaluation, grouping parameters that
/// are passed through evaluate → `relu_split` / `activation_specs`.
pub(crate) struct NeuronEvalContext<'a> {
    pub gpu: &'a GpuWorkQueue,
    pub neuron_squash_map: &'a Arc<HashMap<super::preparation::SharedUuid, String>>,
    pub timing_collector: &'a Arc<super::super::shared::TimingCollector>,
    pub diagnostics: &'a Arc<NeuronDiagnostics>,
    pub helpful_map: &'a Arc<Mutex<HashMap<u64, CandidateNeuronJson>>>,
    pub threshold: f32,
    /// Target saturation info from the pre-check (Issue #1111).
    pub target_saturation: super::preparation::TargetSaturationInfo,
}

/// Evaluate neuron candidates for all sources with samples against a single
/// target neuron using GPU shaders.
///
/// This handles both `ReLU` split evaluation and batched activation spec
/// evaluation, applying source variance discounting.
pub(crate) fn evaluate_neuron_candidates(
    work_results: &[NeuronWorkResult<'_>],
    target_uuid: &str,
    ctx: &NeuronEvalContext<'_>,
    deadline: &Option<SystemTime>,
    analysis_timed_out: &Arc<AtomicBool>,
) -> Result<()> {
    for result in work_results {
        // Check deadline before each evaluation batch
        if deadline_passed(deadline) {
            analysis_timed_out.store(true, Ordering::Relaxed);
            break;
        }

        if result.samples.is_empty() {
            continue;
        }

        // Get target_squash for accurate HARD_TANH modelling
        let target_squash = ctx
            .neuron_squash_map
            .get(target_uuid)
            .map(std::string::String::as_str);

        // Issue #130 (v0.2.2): Compute source variance discount.
        // If source activation has low variance, predictions are unreliable.
        let source_variance_discount = compute_source_variance_discount(&result.samples);
        if source_variance_discount <= EPSILON {
            // Source is constant - skip evaluation entirely
            continue;
        }

        // ReLU evaluation: split by TARGET neuron's error sign.
        evaluate_relu_split(
            result.source_uuid,
            target_uuid,
            &result.samples,
            target_squash,
            source_variance_discount,
            ctx,
        )?;

        // Issue #201: Evaluate all activation specs in a single batched GPU call
        evaluate_activation_specs(
            result.source_uuid,
            target_uuid,
            &result.samples,
            target_squash,
            source_variance_discount,
            ctx,
        )?;
    }
    Ok(())
}

/// Evaluate `ReLU` candidates with positive/negative error split.
fn evaluate_relu_split(
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    target_squash: Option<&str>,
    source_variance_discount: f32,
    ctx: &NeuronEvalContext<'_>,
) -> Result<()> {
    let split_result = {
        let _timing = TimingScope::shader(ctx.timing_collector, "relu");
        evaluate_relu_candidates_split(
            ctx.gpu,
            source_uuid,
            target_uuid,
            samples,
            ctx.threshold,
            target_squash,
        )?
    };

    // Issue #1143: Hard-reject ReLU-split candidates when the target is
    // saturated. Independent of the intermediate squash: a saturated
    // HARD_TANH output cannot absorb new gradient from a `ReLU` intermediate
    // any more than from `ArcTan` or `BENT_IDENTITY`.
    if ctx.target_saturation.rejects_candidates() {
        let dropped = u32::from(split_result.positive_error_candidate.is_some())
            + u32::from(split_result.negative_error_candidate.is_some());
        ctx.diagnostics.record_target_saturated_drops(dropped);
        return Ok(());
    }

    if let Some(mut candidate) = split_result.positive_error_candidate {
        // Issue #733: Filter neuron candidates where insufficient samples improve.
        if !passes_neuron_improved_ratio(&candidate) {
            if verbose_enabled() {
                tracing::trace!(
                    direction = "push UP",
                    source_uuid = %source_uuid,
                    target_uuid = %target_uuid,
                    improved = candidate.improved_count,
                    total = candidate.total_count,
                    "ReLU candidate filtered by NEURON_MIN_IMPROVED_RATIO"
                );
            }
        } else {
            // Issue #130: Apply source variance discount
            candidate.expected_creature_error_reduction *= source_variance_discount;
            candidate.expected_creature_score_gain *= source_variance_discount;

            // Issue #1111: Apply target saturation discount
            apply_target_saturation_discount(&mut candidate, &ctx.target_saturation);

            // Issue #791: Apply cross-validation brittleness penalty
            apply_cross_validation_penalty(&mut candidate, samples);

            if verbose_enabled() {
                tracing::trace!(
                    direction = "push UP",
                    source_uuid = %source_uuid,
                    target_uuid = %target_uuid,
                    improvement_pct = format_args!("{:.2}", candidate.expected_creature_score_gain * 100.0),
                    variance_discount = format_args!("{:.2}", source_variance_discount),
                    "ReLU candidate identified"
                );
            }
            ctx.diagnostics.mark_candidate_selected(target_uuid);
            let mut map = lock_or_bail(ctx.helpful_map, "helpful_map")?;
            upsert_candidate(&mut map, candidate);
        }
    }

    if let Some(mut candidate) = split_result.negative_error_candidate {
        // Issue #733: Filter neuron candidates where insufficient samples improve.
        if !passes_neuron_improved_ratio(&candidate) {
            if verbose_enabled() {
                tracing::trace!(
                    direction = "push DOWN",
                    source_uuid = %source_uuid,
                    target_uuid = %target_uuid,
                    improved = candidate.improved_count,
                    total = candidate.total_count,
                    "ReLU candidate filtered by NEURON_MIN_IMPROVED_RATIO"
                );
            }
        } else {
            // Issue #130: Apply source variance discount
            candidate.expected_creature_error_reduction *= source_variance_discount;
            candidate.expected_creature_score_gain *= source_variance_discount;

            // Issue #1111: Apply target saturation discount
            apply_target_saturation_discount(&mut candidate, &ctx.target_saturation);

            // Issue #791: Apply cross-validation brittleness penalty
            apply_cross_validation_penalty(&mut candidate, samples);

            if verbose_enabled() {
                tracing::trace!(
                    direction = "push DOWN",
                    source_uuid = %source_uuid,
                    target_uuid = %target_uuid,
                    improvement_pct = format_args!("{:.2}", candidate.expected_creature_score_gain * 100.0),
                    variance_discount = format_args!("{:.2}", source_variance_discount),
                    "ReLU candidate identified"
                );
            }
            ctx.diagnostics.mark_candidate_selected(target_uuid);
            let mut map = lock_or_bail(ctx.helpful_map, "helpful_map")?;
            upsert_candidate(&mut map, candidate);
        }
    }

    Ok(())
}

/// Evaluate all activation function specs in a single batched GPU call.
fn evaluate_activation_specs(
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    target_squash: Option<&str>,
    source_variance_discount: f32,
    ctx: &NeuronEvalContext<'_>,
) -> Result<()> {
    let batched_candidates = {
        let _timing = TimingScope::shader(ctx.timing_collector, "activation");
        evaluate_all_activation_specs_batched(
            ctx.gpu,
            source_uuid,
            target_uuid,
            samples,
            ctx.threshold,
            target_squash,
        )?
    };

    // Issue #1143: Hard-reject every activation-spec candidate when the
    // target neuron is saturated. The gate is deliberately independent of
    // the candidate/intermediate squash — a saturated HARD_TANH output
    // cannot produce a usable gradient from any new intermediate neuron,
    // including ArcTan and BENT_IDENTITY (the GRQ-sampler `744ac60d`
    // evidence). Count the drops so they surface in the rejection
    // breakdown under `REJECTION_TARGET_SATURATED`.
    if ctx.target_saturation.rejects_candidates() {
        ctx.diagnostics.record_target_saturated_drops(
            u32::try_from(batched_candidates.len()).unwrap_or(u32::MAX),
        );
        return Ok(());
    }

    for mut candidate in batched_candidates {
        // Issue #733: Filter neuron candidates where insufficient samples improve.
        if !passes_neuron_improved_ratio(&candidate) {
            continue;
        }

        // Issue #1111: Skip activation specs that compound clipping with a
        // near-saturated target (e.g., ABSOLUTE feeding into HARD_TANH).
        // Kept for targets that are close to but not over the saturation
        // threshold — the hard reject above handles the fully-saturated
        // case.
        if ctx.target_saturation.is_near_saturated
            && target_squash.is_some_and(|ts| {
                super::preparation::compounds_target_clipping(&candidate.squash, ts)
            })
        {
            continue;
        }

        // Issue #130: Apply source variance discount
        candidate.expected_creature_error_reduction *= source_variance_discount;
        candidate.expected_creature_score_gain *= source_variance_discount;

        // Issue #1111: Apply target saturation discount
        apply_target_saturation_discount(&mut candidate, &ctx.target_saturation);

        // Issue #887: Apply activation-function-aware boost/penalty
        candidate.expected_creature_score_gain = apply_activation_neuron_boost(
            candidate.expected_creature_score_gain,
            &candidate.squash,
        );

        // Issue #1113: Apply activation compatibility scoring — penalises candidate
        // squash functions that compound clipping with bounded target activations.
        if let Some(ts) = target_squash {
            let compat =
                crate::analysis::activation::activation_compatibility_score(&candidate.squash, ts);
            candidate.expected_creature_error_reduction *= compat;
            candidate.expected_creature_score_gain *= compat;
        }

        // Issue #791: Apply cross-validation brittleness penalty
        apply_cross_validation_penalty(&mut candidate, samples);

        ctx.diagnostics.mark_candidate_selected(target_uuid);
        let mut map = lock_or_bail(ctx.helpful_map, "helpful_map")?;
        upsert_candidate(&mut map, candidate);
    }

    Ok(())
}

/// Issue #1111: Apply target saturation adjustments to a neuron candidate.
///
/// When the target neuron is near saturation, we:
/// 1. Set `target_saturation_factor` on the candidate for downstream scoring
/// 2. Discount expected gains proportionally to how saturated the target is
fn apply_target_saturation_discount(
    candidate: &mut CandidateNeuronJson,
    saturation: &super::preparation::TargetSaturationInfo,
) {
    if !saturation.is_near_saturated {
        return;
    }

    candidate.target_saturation_factor = Some(saturation.saturation_factor);

    // Discount: a target at saturation_factor=1.0 gets a 50% discount;
    // at 0.9 (threshold) the discount is small (~5%).
    let discount = 1.0 - (saturation.saturation_factor * 0.5);
    candidate.expected_creature_error_reduction *= discount;
    candidate.expected_creature_score_gain *= discount;
}

/// Issue #733: Check whether a neuron candidate passes the minimum improved ratio threshold.
///
/// Filters out neuron candidates where insufficient samples show improvement,
/// reducing candidate volume while retaining higher-quality candidates.
fn passes_neuron_improved_ratio(candidate: &CandidateNeuronJson) -> bool {
    use crate::analysis::constants::NEURON_MIN_IMPROVED_RATIO;

    if candidate.total_count == 0 {
        return false;
    }
    let improved_ratio = candidate.improved_count as f32 / candidate.total_count as f32;
    improved_ratio >= NEURON_MIN_IMPROVED_RATIO
}

/// Issue #791: Apply cross-validation brittleness penalty to a neuron candidate.
///
/// Splits the samples into k folds and checks whether the candidate's improvement
/// is consistent across all subsets. Candidates that only improve on specific data
/// subsets are penalised, reducing their expected score gain.
///
/// This addresses the 15% success rate for add-neurons by filtering candidates
/// that overfit to noise in the discovery samples.
fn apply_cross_validation_penalty(candidate: &mut CandidateNeuronJson, samples: &[HelpfulSample]) {
    let config = CrossValidationConfig::default();
    if let Some(cv_result) = compute_cross_validation_score(samples, &config)
        && cv_result.brittleness_penalty > 0.0
    {
        let penalty_factor = 1.0 - cv_result.brittleness_penalty;
        candidate.expected_creature_error_reduction *= penalty_factor;
        candidate.expected_creature_score_gain *= penalty_factor;

        if verbose_enabled() {
            tracing::trace!(
                source_uuid = %candidate.source_neuron_uuid,
                target_uuid = %candidate.target_neuron_uuid,
                penalty = format_args!("{:.4}", cv_result.brittleness_penalty),
                variance = format_args!("{:.6}", cv_result.variance.variance),
                mean_ratio = format_args!("{:.3}", cv_result.variance.mean_improvement_ratio),
                "Neuron candidate cross-validation brittleness penalty applied"
            );
        }
    }
}
