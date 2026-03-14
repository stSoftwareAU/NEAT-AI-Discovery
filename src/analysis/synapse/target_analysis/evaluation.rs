//! Per-target evaluation logic
//!
//! This module handles GPU work submission, result collection, and processing
//! of helpful and harmful synapse candidates.
//!
//! Extracted from target_analysis.rs as part of Issue #599.

use crate::CandidateSynapseJson;
use crate::analysis::activation::get_target_simulation_fn;
use crate::analysis::cache::RecordCache;
use crate::analysis::detection::redundant_path::ExistingPathContribution;
use crate::analysis::diagnostics::ThresholdContext;
use crate::analysis::gpu::GpuWorkQueue;
use crate::analysis::recommendation::epistatic::{SourceContribution, build_source_contribution};
use crate::analysis::samples::{EPSILON, HelpfulSample, HelpfulStats, NeuronStats};
use crate::analysis::scoring::confidence::compute_confidence_metrics;
use crate::analysis::scoring::weights::{
    MAX_OUTGOING_WEIGHT, calculate_optimal_outgoing_weight, clamp_weight_update_delta,
};
use crate::analysis::shared::TimingScope;
use anyhow::Result;

use super::statistics::PreparedHarmfulWork;
use super::{HelpfulWork, TargetAnalysisContext, TargetAnalysisResults};

use crate::analysis::synapse::scoring::compute_synapse_improvement_and_count;

/// Issue #568: Submit helpful GPU work non-blocking.
///
/// Clones sample data for the GPU, tracks metadata, and submits the batch.
/// Returns a `GpuFuture` that can be collected after overlapping CPU work.
pub(crate) fn submit_helpful_gpu_work(
    helpful_work_batch: &[HelpfulWork],
    gpu: &GpuWorkQueue,
    ctx: &TargetAnalysisContext,
    results: &mut TargetAnalysisResults,
) -> Result<crate::analysis::gpu::queue::GpuFuture<Vec<HelpfulStats>>> {
    let helpful_samples: Vec<Vec<HelpfulSample>> = helpful_work_batch
        .iter()
        .map(|w| w.samples.clone()) // Clone required: GPU queue takes ownership of sample data
        .collect();

    // Track metadata
    for samples in &helpful_samples {
        if samples.iter().any(|s| s.target_value.is_some()) {
            results.target_value_seen = true;
            break;
        }
    }

    let _timing = TimingScope::shader(&ctx.timing_collector, "helpful");
    gpu.submit_helpful_batch(helpful_samples, &ctx.deadline)
}

/// Issue #568: Collect helpful GPU results and process them into candidates.
pub(crate) fn collect_and_process_helpful_results(
    future: crate::analysis::gpu::queue::GpuFuture<Vec<HelpfulStats>>,
    helpful_work_batch: &[HelpfulWork],
    target_uuid: &str,
    cache: &RecordCache,
    ctx: &TargetAnalysisContext,
    existing_path_contributions: &[ExistingPathContribution],
    results: &mut TargetAnalysisResults,
) -> Result<()> {
    let helpful_stats_batch = future.collect()?;

    let mut candidates_to_add = Vec::new();
    let mut coordinated_to_add = Vec::new();
    let mut diagnostics_zero_improvements: Vec<(&str, &str, usize, u32, u32)> = Vec::new();
    let mut diagnostics_below_threshold: Vec<(&str, &str, ThresholdContext)> = Vec::new();
    let mut diagnostics_selected: Vec<&str> = Vec::new();
    let mut source_contributions: Vec<SourceContribution> = Vec::new();

    {
        let _timing = TimingScope::result_processing(&ctx.timing_collector);
        for (work, stats) in helpful_work_batch.iter().zip(helpful_stats_batch.iter()) {
            let positive_is_better = stats.positive_count >= stats.negative_count;
            let gpu_improved_count = if positive_is_better {
                stats.positive_count
            } else {
                stats.negative_count
            };
            if gpu_improved_count == 0 {
                diagnostics_zero_improvements.push((
                    work.target_uuid.as_str(),
                    work.source_uuid.as_str(),
                    work.samples.len(),
                    stats.positive_count,
                    stats.negative_count,
                ));
                continue;
            }

            let total_count = work.samples.len() as u32;
            if total_count == 0 {
                continue;
            }

            let weight = match calculate_optimal_outgoing_weight(
                stats.error_activation_sum,
                stats.activation_sq_sum,
                1.0,
            ) {
                Some(w) => w,
                None => continue,
            };

            let target_squash = ctx
                .neuron_squash_map
                .get(work.target_uuid.as_str())
                .copied();

            if get_target_simulation_fn(&work.samples, target_squash).is_some() {
                results.saturation_aware_used = true;
            }

            let baseline_error_sq = stats.error_sq_sum;

            let (applied_weight, neuron_error_improvement, improved_count, worsened_count) =
                if let Some(old_weight) = work.existing_weight {
                    let Some((_new_weight, delta_weight)) =
                        clamp_weight_update_delta(old_weight, weight)
                    else {
                        continue;
                    };
                    let (improvement, improved, worsened, _) =
                        compute_synapse_improvement_and_count(
                            &work.samples,
                            delta_weight,
                            baseline_error_sq,
                            target_squash,
                        );
                    (delta_weight, improvement, improved, worsened)
                } else {
                    // Issue #730: Multi-weight search for ALL new synapse candidates.
                    // Previously only saturating targets used weight search (Issue #413).
                    // Production data showed 0% success rate because a single computed
                    // weight often overshoots, especially with noisy samples.
                    let weight_candidates: [f32; 9] = [
                        weight * 0.1,
                        weight * 0.25,
                        weight * 0.5,
                        weight * 0.75,
                        weight,
                        weight * 1.5,
                        weight * 2.0,
                        -weight * 0.5,
                        -weight,
                    ];

                    let mut best_weight = weight;
                    let mut best_improvement = f32::NEG_INFINITY;
                    let mut best_improved = 0u32;
                    let mut best_worsened = 0u32;

                    for &w in &weight_candidates {
                        let clamped = w.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);
                        if clamped.abs() <= EPSILON {
                            continue;
                        }
                        let (imp, improved, worsened, _) = compute_synapse_improvement_and_count(
                            &work.samples,
                            clamped,
                            baseline_error_sq,
                            target_squash,
                        );
                        if imp > best_improvement {
                            best_improvement = imp;
                            best_weight = clamped;
                            best_improved = improved;
                            best_worsened = worsened;
                        }
                    }
                    (best_weight, best_improvement, best_improved, best_worsened)
                };

            // Issue #202: Track source contribution for epistatic pair detection
            if work.existing_weight.is_none() {
                source_contributions.push(build_source_contribution(
                    &work.source_uuid,
                    work.samples.clone(), // Clone required: SourceContribution takes ownership
                    *stats,
                    applied_weight,
                    neuron_error_improvement,
                ));
            }

            if neuron_error_improvement <= 0.0 {
                continue;
            }

            // Issue #730: Filter candidates where insufficient samples improve.
            // Candidates where worsened > improved have 0% success rate in production.
            {
                use crate::analysis::constants::MIN_IMPROVED_RATIO;
                let improved_ratio = if total_count > 0 {
                    improved_count as f32 / total_count as f32
                } else {
                    0.0
                };
                if improved_ratio < MIN_IMPROVED_RATIO {
                    continue;
                }
            }

            if neuron_error_improvement <= ctx.threshold {
                diagnostics_below_threshold.push((
                    work.target_uuid.as_str(),
                    work.source_uuid.as_str(),
                    ThresholdContext {
                        sample_count: work.samples.len(),
                        expected_improvement: neuron_error_improvement,
                        threshold: ctx.threshold,
                        improved_count,
                        worsened_count,
                        weight: applied_weight,
                    },
                ));
            }

            let target_stats = cache
                .get(&work.target_uuid)
                .ok()
                .and_then(|records| NeuronStats::from_records(records.as_ref()))
                .map(|s| s.to_json());
            if let Some(old_weight) = work.existing_weight {
                let Some((new_weight, delta_weight)) =
                    clamp_weight_update_delta(old_weight, weight)
                else {
                    continue;
                };
                coordinated_to_add.push(crate::CoordinatedStructuralCandidateJson {
                    operations: vec![crate::CoordinatedStructuralOpJson::SetWeight {
                        from_neuron_uuid: work.source_uuid.clone(),
                        to_neuron_uuid: work.target_uuid.clone(),
                        weight: new_weight,
                    }],
                    expected_creature_score_gain: neuron_error_improvement,
                    comment: Some(format!(
                        "Adjust synapse weight: old={old_weight:.6}, new={new_weight:.6}, delta={delta_weight:.6}"
                    )),
                });
            } else {
                diagnostics_selected.push(work.target_uuid.as_str());
                // Issue #178: constant source folding into setBias
                if let Some(threshold) = ctx.constant_source_effect_threshold {
                    let mut act_min = f32::INFINITY;
                    let mut act_max = f32::NEG_INFINITY;
                    let mut act_sum = 0.0f64;
                    let mut act_count: u32 = 0;
                    for s in &work.samples {
                        if s.activation.is_finite() {
                            act_min = act_min.min(s.activation);
                            act_max = act_max.max(s.activation);
                            act_sum += s.activation as f64;
                            act_count += 1;
                        }
                    }

                    if act_count > 0 {
                        let mean_activation = (act_sum / act_count as f64) as f32;
                        let activation_range = (act_max - act_min).abs();
                        let effect_range = applied_weight.abs() * activation_range;

                        if mean_activation.is_finite()
                            && activation_range.is_finite()
                            && effect_range.is_finite()
                            && effect_range <= threshold
                        {
                            let old_bias = ctx
                                .neuron_bias_map
                                .get(work.target_uuid.as_str())
                                .copied()
                                .unwrap_or(0.0);
                            let new_bias = old_bias + (applied_weight * mean_activation);
                            if new_bias.is_finite() {
                                coordinated_to_add.push(
                                    crate::CoordinatedStructuralCandidateJson {
                                        operations: vec![
                                            crate::CoordinatedStructuralOpJson::SetBias {
                                                neuron_uuid: work.target_uuid.clone(),
                                                bias: new_bias,
                                            },
                                        ],
                                        expected_creature_score_gain: neuron_error_improvement,
                                        comment: Some(format!(
                                            "Fold constant source into setBias: old_bias={old_bias:.6}, new_bias={new_bias:.6}, weight={applied_weight:.6}, mean_act={mean_activation:.6}, act_range={activation_range:.6e}, effect_range={effect_range:.6e}"
                                        )),
                                    },
                                );
                                continue;
                            }
                        }
                    }
                }

                let confidence_metrics =
                    compute_confidence_metrics(&work.samples, neuron_error_improvement, None);
                candidates_to_add.push(CandidateSynapseJson {
                    from_neuron_uuid: work.source_uuid.clone(),
                    to_neuron_uuid: work.target_uuid.clone(),
                    from_neuron_index: None,
                    to_neuron_index: None,
                    weight: applied_weight,
                    target_neuron_impact: 1.0,
                    expected_creature_error_reduction: neuron_error_improvement,
                    expected_creature_score_gain: neuron_error_improvement,
                    improved_count,
                    total_count,
                    target_neuron_stats: target_stats,
                    outlier_reduction_info: None,
                    prediction_confidence: confidence_metrics.prediction_confidence,
                    expected_score_gain_confidence_interval: confidence_metrics
                        .expected_score_gain_confidence_interval,
                    comment: None,
                });
            }
        } // End timing scope for result processing
    }

    // Apply diagnostics updates
    for (target, source, sample_count, pos, neg) in diagnostics_zero_improvements {
        ctx.diagnostics
            .record_zero_improvement(target, source, sample_count, pos, neg);
    }
    for (target, source, context) in diagnostics_below_threshold {
        ctx.diagnostics
            .record_below_threshold(target, source, context);
    }
    for target in diagnostics_selected {
        ctx.diagnostics.mark_candidate_selected(target);
    }

    results.helpful.extend(candidates_to_add);
    results.coordinated.extend(coordinated_to_add);

    // Detect epistatic/synergistic candidates and redundant paths
    super::candidate_selection::detect_epistatic_and_synergistic(
        target_uuid,
        &source_contributions,
        ctx,
        results,
    );
    super::candidate_selection::detect_redundant_path_candidates(
        target_uuid,
        existing_path_contributions,
        ctx,
        results,
    );

    Ok(())
}

/// Issue #568: Process pre-built harmful samples via GPU evaluation.
pub(crate) fn process_harmful_batch_from_prepared(
    target_uuid: &str,
    harmful_work: &[PreparedHarmfulWork<'_>],
    gpu: &GpuWorkQueue,
    ctx: &TargetAnalysisContext,
    cache: &RecordCache,
    results: &mut TargetAnalysisResults,
) -> Result<()> {
    let batch_input: Vec<(Vec<HelpfulSample>, f32)> = harmful_work
        .iter()
        .map(|w| (w.samples.clone(), w.weight)) // Clone required: GPU queue takes ownership
        .collect();

    let batch_stats = {
        let _timing = TimingScope::shader(&ctx.timing_collector, "harmful");
        gpu.evaluate_harmful_batch(batch_input, &ctx.deadline)?
    };

    let mut harmful_candidates = Vec::with_capacity(batch_stats.len());
    let target_stats = cache
        .get(target_uuid)
        .ok()
        .and_then(|records| NeuronStats::from_records(records.as_ref()))
        .map(|s| s.to_json());

    for (work, stats) in harmful_work.iter().zip(batch_stats.iter()) {
        let total_count = work.samples.len() as u32;
        if total_count == 0 {
            continue;
        }

        let neuron_error_improvement =
            (stats.harmful_count as f32 - stats.helpful_count as f32) / total_count as f32;

        if neuron_error_improvement <= 0.0 {
            continue;
        }

        let confidence_metrics =
            compute_confidence_metrics(&work.samples, neuron_error_improvement, None);
        harmful_candidates.push(CandidateSynapseJson {
            from_neuron_uuid: work.from_uuid.to_string(),
            to_neuron_uuid: work.to_uuid.to_string(),
            from_neuron_index: None,
            to_neuron_index: None,
            weight: work.weight,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: neuron_error_improvement,
            expected_creature_score_gain: neuron_error_improvement,
            improved_count: stats.harmful_count,
            total_count,
            target_neuron_stats: target_stats,
            outlier_reduction_info: None,
            prediction_confidence: confidence_metrics.prediction_confidence,
            expected_score_gain_confidence_interval: confidence_metrics
                .expected_score_gain_confidence_interval,
            comment: None,
        });
    }

    results.harmful.extend(harmful_candidates);
    Ok(())
}
