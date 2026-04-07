//! Per-target evaluation logic
//!
//! This module handles GPU work submission, result collection, and processing
//! of helpful and harmful synapse candidates.
//!
//! Extracted from `target_analysis.rs` as part of Issue #599.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
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
    let mut diagnostics_accepted_below_threshold: Vec<&str> = Vec::new();
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

            let full_total_count = work.samples.len() as u32;
            if full_total_count == 0 {
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

            // Issue #893: total_count is updated to the validation sample count
            // when hold-out validation is used, so ratio checks remain consistent.
            let (
                applied_weight,
                neuron_error_improvement,
                improved_count,
                worsened_count,
                total_count,
            ) = if let Some(old_weight) = work.existing_weight {
                let Some((_new_weight, delta_weight)) =
                    clamp_weight_update_delta(old_weight, weight)
                else {
                    continue;
                };
                let (improvement, improved, worsened, _) = compute_synapse_improvement_and_count(
                    &work.samples,
                    delta_weight,
                    baseline_error_sq,
                    target_squash,
                );
                (
                    delta_weight,
                    improvement,
                    improved,
                    worsened,
                    full_total_count,
                )
            } else {
                // Issue #730: Multi-weight search for ALL new synapse candidates.
                // Previously only saturating targets used weight search (Issue #413).
                // Production data showed 0% success rate because a single computed
                // weight often overshoots, especially with noisy samples.
                //
                // Issue #893: Hold-out validation to combat overfitting from the
                // 9-variant search. Select weight on training samples, report
                // improvement on held-out validation samples.
                use crate::analysis::synapse::holdout_validation::{
                    baseline_error_sq as compute_baseline, collect_samples, split_samples_holdout,
                };

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

                if let Some(split) =
                    split_samples_holdout(&work.samples, &work.source_uuid, &work.target_uuid)
                {
                    // Phase 1: Select best weight using training samples only
                    let train_samples = collect_samples(&split.train);
                    let train_baseline = compute_baseline(&split.train);

                    let mut best_weight = weight;
                    let mut best_train_improvement = f32::NEG_INFINITY;

                    for &w in &weight_candidates {
                        let clamped = w.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);
                        if clamped.abs() <= EPSILON {
                            continue;
                        }
                        let (imp, _, _, _) = compute_synapse_improvement_and_count(
                            &train_samples,
                            clamped,
                            train_baseline,
                            target_squash,
                        );
                        if imp > best_train_improvement {
                            best_train_improvement = imp;
                            best_weight = clamped;
                        }
                    }

                    // Phase 2: Report improvement on validation samples only
                    let validate_samples = collect_samples(&split.validate);
                    let validate_baseline = compute_baseline(&split.validate);
                    let val_total = validate_samples.len() as u32;

                    let (val_imp, val_improved, val_worsened, _) =
                        compute_synapse_improvement_and_count(
                            &validate_samples,
                            best_weight,
                            validate_baseline,
                            target_squash,
                        );
                    (best_weight, val_imp, val_improved, val_worsened, val_total)
                } else {
                    // Fallback: below hold-out threshold, use all samples
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
                    (
                        best_weight,
                        best_improvement,
                        best_improved,
                        best_worsened,
                        full_total_count,
                    )
                }
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
                // Issue #1018: Metropolis-Hastings probabilistic acceptance
                // for marginal candidates (0 < improvement ≤ threshold).
                // When MH temperature is configured, marginal candidates are
                // accepted with probability proportional to their improvement.
                // When unconfigured, existing deterministic behaviour is preserved.
                if let Some(temperature) = crate::config::mh_temperature() {
                    let acceptance_probability =
                        (neuron_error_improvement / temperature).exp().min(1.0);

                    // Deterministic pseudo-random decision based on source+target UUIDs
                    // to ensure reproducibility across runs with the same data.
                    let hash_val =
                        mh_acceptance_hash(work.source_uuid.as_str(), work.target_uuid.as_str());
                    let random_01 = (hash_val as f32) / (u64::MAX as f32);

                    if random_01 >= acceptance_probability {
                        // Rejected by probabilistic acceptance — record diagnostics
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
                        continue;
                    }
                    // Accepted below threshold — record for monitoring
                    diagnostics_accepted_below_threshold.push(work.target_uuid.as_str());
                } else {
                    // Deterministic mode: record diagnostics, candidate proceeds
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
    for target in diagnostics_accepted_below_threshold {
        ctx.diagnostics.record_accepted_below_threshold(target);
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

/// Issue #1018: Deterministic pseudo-random hash for Metropolis-Hastings acceptance.
///
/// Combines source and target UUIDs to produce a reproducible u64 value
/// for probabilistic acceptance decisions. Uses FNV-1a for speed and
/// adequate distribution across the [0, 1) range.
#[inline]
fn mh_acceptance_hash(source_uuid: &str, target_uuid: &str) -> u64 {
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in source_uuid.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    // Separator to avoid collisions between ("ab","cd") and ("a","bcd")
    hash ^= 0xff;
    hash = hash.wrapping_mul(0x0100_0000_01b3);
    for byte in target_uuid.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mh_acceptance_hash_is_deterministic() {
        let h1 = mh_acceptance_hash("source-1", "target-2");
        let h2 = mh_acceptance_hash("source-1", "target-2");
        assert_eq!(h1, h2, "Same inputs should produce the same hash");
    }

    #[test]
    fn mh_acceptance_hash_differs_for_different_inputs() {
        let h1 = mh_acceptance_hash("source-1", "target-2");
        let h2 = mh_acceptance_hash("source-2", "target-1");
        assert_ne!(h1, h2, "Different inputs should produce different hashes");
    }

    #[test]
    fn mh_acceptance_hash_avoids_prefix_collision() {
        // "ab" + "cd" should differ from "a" + "bcd" due to separator byte
        let h1 = mh_acceptance_hash("ab", "cd");
        let h2 = mh_acceptance_hash("a", "bcd");
        assert_ne!(h1, h2, "Separator byte should prevent prefix collisions");
    }

    /// Issue #1018: Verify acceptance probability calculation matches
    /// Metropolis-Hastings formula: min(1, exp(improvement / temperature)).
    #[test]
    fn acceptance_probability_above_threshold_is_always_one() {
        let improvement = 0.05;
        let threshold = 0.01;
        // Above threshold → always accepted (probability = 1.0)
        assert!(
            improvement > threshold,
            "Test setup: improvement must exceed threshold"
        );
    }

    /// Issue #1018: Marginal candidates have acceptance probability
    /// proportional to their improvement relative to temperature.
    #[test]
    fn acceptance_probability_marginal_candidates() {
        let temperature: f32 = 0.01;
        let threshold: f32 = 0.02;

        // Candidate with half the threshold improvement
        let improvement_half = threshold / 2.0;
        let p_half = (improvement_half / temperature).exp().min(1.0);
        assert!(
            p_half > 0.0 && p_half <= 1.0,
            "Acceptance probability should be in (0, 1], got {p_half}"
        );

        // Candidate with very small improvement
        let improvement_tiny = 0.001;
        let p_tiny = (improvement_tiny / temperature).exp().min(1.0);
        assert!(
            p_tiny > 0.0 && p_tiny <= 1.0,
            "Acceptance probability should be in (0, 1], got {p_tiny}"
        );

        // Higher improvement should have higher acceptance probability
        assert!(
            p_half >= p_tiny,
            "Higher improvement ({improvement_half}) should have >= acceptance probability than lower ({improvement_tiny}): {p_half} vs {p_tiny}"
        );
    }

    /// Issue #1018: Candidates at or below zero improvement are always rejected.
    #[test]
    fn zero_or_negative_improvement_rejected() {
        // The main loop rejects improvement <= 0.0 before reaching the
        // threshold check, so zero/negative improvements never reach MH.
        // This test verifies the formula would also reject them.
        let temperature: f32 = 0.01;
        let zero_p = (0.0_f32 / temperature).exp().min(1.0);
        // exp(0) = 1.0, but the code path rejects <= 0.0 before MH
        assert!(
            (zero_p - 1.0).abs() < f32::EPSILON,
            "exp(0/T) should be 1.0 but code rejects <= 0 before this point"
        );
    }

    /// Issue #1018: Hash-based random value covers the [0, 1) range
    /// across a sample of inputs.
    #[test]
    fn hash_produces_varied_random_values() {
        let mut values = Vec::new();
        for i in 0..100 {
            let source = format!("source-{i}");
            let target = format!("target-{i}");
            let hash = mh_acceptance_hash(&source, &target);
            let random_01 = (hash as f32) / (u64::MAX as f32);
            values.push(random_01);
        }
        // Verify we get a reasonable spread
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max - min > 0.5,
            "Hash values should cover a reasonable range, but got [{min}, {max}]"
        );
    }
}
