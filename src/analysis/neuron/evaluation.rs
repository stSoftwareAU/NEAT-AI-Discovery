//! Neuron candidate GPU evaluation — ReLU split evaluation and activation
//! spec batched evaluation for neuron candidates.
//!
//! Extracted from neuron.rs as part of issue #598.

use crate::CandidateNeuronJson;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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
    evaluate_all_activation_specs_batched, evaluate_relu_candidates_split, upsert_candidate,
};

/// Work result from sample building phase, containing samples for a single
/// source neuron.
pub(crate) struct NeuronWorkResult {
    pub source_uuid: String,
    pub samples: Vec<HelpfulSample>,
}

/// Shared context for neuron candidate evaluation, grouping parameters that
/// are passed through evaluate → relu_split / activation_specs.
pub(crate) struct NeuronEvalContext<'a> {
    pub gpu: &'a GpuWorkQueue,
    pub neuron_squash_map: &'a Arc<HashMap<String, String>>,
    pub timing_collector: &'a Arc<super::super::shared::TimingCollector>,
    pub diagnostics: &'a Arc<NeuronDiagnostics>,
    pub helpful_map: &'a Arc<Mutex<HashMap<u64, CandidateNeuronJson>>>,
    pub threshold: f32,
}

/// Evaluate neuron candidates for all sources with samples against a single
/// target neuron using GPU shaders.
///
/// This handles both ReLU split evaluation and batched activation spec
/// evaluation, applying source variance discounting.
pub(crate) fn evaluate_neuron_candidates(
    work_results: &[NeuronWorkResult],
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
            &result.source_uuid,
            target_uuid,
            &result.samples,
            target_squash,
            source_variance_discount,
            ctx,
        )?;

        // Issue #201: Evaluate all activation specs in a single batched GPU call
        evaluate_activation_specs(
            &result.source_uuid,
            target_uuid,
            &result.samples,
            target_squash,
            source_variance_discount,
            ctx,
        )?;
    }
    Ok(())
}

/// Evaluate ReLU candidates with positive/negative error split.
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

    for mut candidate in batched_candidates {
        // Issue #733: Filter neuron candidates where insufficient samples improve.
        if !passes_neuron_improved_ratio(&candidate) {
            continue;
        }

        // Issue #130: Apply source variance discount
        candidate.expected_creature_error_reduction *= source_variance_discount;
        candidate.expected_creature_score_gain *= source_variance_discount;

        // Issue #791: Apply cross-validation brittleness penalty
        apply_cross_validation_penalty(&mut candidate, samples);

        ctx.diagnostics.mark_candidate_selected(target_uuid);
        let mut map = lock_or_bail(ctx.helpful_map, "helpful_map")?;
        upsert_candidate(&mut map, candidate);
    }

    Ok(())
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
