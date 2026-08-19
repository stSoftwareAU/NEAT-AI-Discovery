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

use crate::analysis::diagnostics::{NeuronDiagnostics, target_saturated_should_abort_pass};
use crate::analysis::gpu::GpuWorkQueue;
use crate::analysis::samples::{HelpfulSample, compute_source_variance_discount};
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
    /// Squash-aware activation scan plan for hidden add-neuron targets
    /// (Issue #1545). Computed once per phase; replaces the full-cross-product
    /// [`crate::analysis::activation::ACTIVATION_SPECS`] scan.
    pub scan_plan: &'a crate::analysis::activation::SquashScanPlan,
    /// Target saturation info from the pre-check (Issue #1111).
    pub target_saturation: super::preparation::TargetSaturationInfo,
    /// Issue #1164: Within-batch target-failure short-circuit tracker.
    pub within_batch_failures:
        &'a Arc<crate::analysis::within_batch_failures::WithinBatchFailureTracker>,
    /// Issue #1798: per-batch counters for the pre-evaluation drop sites
    /// (no samples / constant source), folded into the metadata rejection
    /// breakdown once per surface.
    pub evaluation_drops: &'a Arc<crate::analysis::evaluation_drops::EvaluationDropCounters>,
    /// Issue #1802: per-pass reconciliation ledger for this surface.
    ///
    /// Candidates are counted as *considered* where the GPU evaluators form
    /// them, and as *accounted* at each disposition. A future bare `continue`
    /// leaves the two out of balance, which fails the invariant instead of
    /// losing the candidate silently.
    pub ledger: &'a Arc<crate::analysis::candidate_reconciliation::CandidateLedger>,
    /// Issue #4140: set when `target_saturated` dominates so remaining
    /// sources and later targets are skipped.
    pub saturation_aborted: &'a Arc<AtomicBool>,
    /// Issue #4140: set the moment a candidate is admitted, so a productive
    /// pass never aborts. Mirrors the synapse surface's flag of the same name;
    /// an atomic (not a probe of `helpful_map`) keeps the signal correct while
    /// a parallel target holds the map lock.
    pub any_candidates: &'a Arc<AtomicBool>,
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
        if deadline_passed(deadline) || ctx.saturation_aborted.load(Ordering::Relaxed) {
            if deadline_passed(deadline) {
                analysis_timed_out.store(true, Ordering::Relaxed);
            }
            break;
        }

        // Issue #1798: count the drop (`no_samples`) instead of dropping
        // silently — the candidate never reaches the accept gate, so the
        // starvation classifier could not otherwise see it.
        if ctx.evaluation_drops.drop_for_empty_samples(&result.samples) {
            continue;
        }

        // Issue #1164: short-circuit subsequent same-target candidates if
        // an earlier candidate for this target failed within this batch.
        if ctx.within_batch_failures.should_skip(target_uuid) {
            ctx.within_batch_failures.record_skip();
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
        // Source is constant — skip evaluation entirely. Issue #1798: counted
        // as `zero_source_variance`, a distinct cause from `no_samples` (the
        // source was sampled, it just carries no signal).
        if ctx
            .evaluation_drops
            .drop_for_zero_source_variance(source_variance_discount)
        {
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

    // Issue #1802: the split shader has now formed this batch, so count it once
    // here — the single batch-formation site for the ReLU-split path.
    let formed = usize::from(split_result.positive_error_candidate.is_some())
        + usize::from(split_result.negative_error_candidate.is_some());
    ctx.ledger.record_considered(formed);

    // Issue #1143: Hard-reject ReLU-split candidates when the target is
    // saturated. Independent of the intermediate squash: a saturated
    // HARD_TANH output cannot absorb new gradient from a `ReLU` intermediate
    // any more than from `ArcTan` or `BENT_IDENTITY`.
    if ctx.target_saturation.rejects_candidates() {
        ctx.diagnostics
            .record_target_saturated_drops(u32::try_from(formed).unwrap_or(u32::MAX));
        ctx.ledger.record_accounted(formed);
        maybe_abort_saturated_pass(ctx);
        return Ok(());
    }

    if let Some(mut candidate) = split_result.positive_error_candidate {
        // Issue #733: Filter neuron candidates where insufficient samples improve.
        if !passes_neuron_improved_ratio(&candidate) {
            // Issue #1164: a rejected candidate counts as a within-batch failure
            // for the target so subsequent same-target candidates can be skipped.
            ctx.within_batch_failures.record_failure(target_uuid);
            // Issue #1802: count the drop instead of only recording the
            // within-batch failure — the candidate was formed and rejected at
            // the improved-ratio gate, so that verdict belongs in the breakdown.
            ctx.evaluation_drops.drop_below_improved_ratio();
            ctx.ledger.record_accounted(1);
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
            // Issue #1802: admitted to the surface's candidate map — the
            // accounted side of the reconciliation identity.
            ctx.ledger.record_accounted(1);
            accept_candidate(ctx.helpful_map, ctx.any_candidates, candidate)?;
        }
    }

    if let Some(mut candidate) = split_result.negative_error_candidate {
        // Issue #733: Filter neuron candidates where insufficient samples improve.
        if !passes_neuron_improved_ratio(&candidate) {
            // Issue #1164: rejected candidate → within-batch failure for target.
            ctx.within_batch_failures.record_failure(target_uuid);
            // Issue #1802: as above — the improved-ratio verdict is recorded.
            ctx.evaluation_drops.drop_below_improved_ratio();
            ctx.ledger.record_accounted(1);
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
            // Issue #1802: admitted to the surface's candidate map — the
            // accounted side of the reconciliation identity.
            ctx.ledger.record_accounted(1);
            accept_candidate(ctx.helpful_map, ctx.any_candidates, candidate)?;
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
            ctx.scan_plan,
        )?
    };

    // Issue #1143: Hard-reject every activation-spec candidate when the
    // target neuron is saturated. The gate is deliberately independent of
    // the candidate/intermediate squash — a saturated HARD_TANH output
    // cannot produce a usable gradient from any new intermediate neuron,
    // including ArcTan and BENT_IDENTITY (the production discovery-cache
    // evidence). Count the drops so they surface in the rejection
    // breakdown under `REJECTION_TARGET_SATURATED`.
    // Issue #1802: the batched shader has now formed this batch — the single
    // batch-formation site for the activation-spec path.
    ctx.ledger.record_considered(batched_candidates.len());

    if ctx.target_saturation.rejects_candidates() {
        ctx.diagnostics.record_target_saturated_drops(
            u32::try_from(batched_candidates.len()).unwrap_or(u32::MAX),
        );
        ctx.ledger.record_accounted(batched_candidates.len());
        maybe_abort_saturated_pass(ctx);
        return Ok(());
    }

    for mut candidate in batched_candidates {
        // Issue #733: Filter neuron candidates where insufficient samples improve.
        if !passes_neuron_improved_ratio(&candidate) {
            // Issue #1164: rejected candidate → within-batch failure for target.
            ctx.within_batch_failures.record_failure(target_uuid);
            // Issue #1802: record the improved-ratio verdict rather than
            // dropping the formed candidate silently.
            ctx.evaluation_drops.drop_below_improved_ratio();
            ctx.ledger.record_accounted(1);
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
            // Issue #1802: the near-saturated target is the cause, so this drop
            // joins the fully-saturated hard-reject under the same reason
            // instead of vanishing.
            ctx.diagnostics.record_target_saturated_drops(1);
            ctx.ledger.record_accounted(1);
            maybe_abort_saturated_pass(ctx);
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
        // Issue #1802: admitted to the surface's candidate map.
        ctx.ledger.record_accounted(1);
        accept_candidate(ctx.helpful_map, ctx.any_candidates, candidate)?;
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

/// Admit a candidate to this surface's candidate map (Issue #4140).
///
/// The productivity signal is raised here, on the accept path, rather than
/// inferred later from the map's contents: targets are evaluated in parallel,
/// so a probe of `helpful_map` can find the lock held by another target and
/// read a productive pass as empty.
fn accept_candidate(
    helpful_map: &Arc<Mutex<HashMap<u64, CandidateNeuronJson>>>,
    any_candidates: &Arc<AtomicBool>,
    candidate: CandidateNeuronJson,
) -> Result<()> {
    let mut map = lock_or_bail(helpful_map, "helpful_map")?;
    upsert_candidate(&mut map, candidate);
    any_candidates.store(true, Ordering::Relaxed);
    Ok(())
}

/// Whether the neuron pass should stop because `target_saturated` dominates
/// (Issue #4140).
///
/// `within_batch_skips` are excluded from the sample — a within-batch
/// short-circuit ends a batch, not the pass. `saturated` may exceed the ledger
/// (saturated source-drops need not be formed first), so the larger of the two
/// is the rejection sample rather than their sum, which would double-count
/// formed-then-saturated proposals.
fn neuron_pass_should_abort(
    saturated: u32,
    considered: u32,
    within_batch_skips: u32,
    any_candidates: bool,
) -> bool {
    let total_rejections = rejection_sample(saturated, considered, within_batch_skips);
    target_saturated_should_abort_pass(saturated, total_rejections, any_candidates)
}

/// Size of the rejection sample the saturation ratio is measured against.
fn rejection_sample(saturated: u32, considered: u32, within_batch_skips: u32) -> u32 {
    saturated.max(considered.saturating_sub(within_batch_skips))
}

/// Issue #4140: abort the rest of the pass when `target_saturated` dominates.
fn maybe_abort_saturated_pass(ctx: &NeuronEvalContext<'_>) {
    if ctx.saturation_aborted.load(Ordering::Relaxed) {
        return;
    }
    let saturated = ctx.diagnostics.target_saturated_drop_count();
    let considered = ctx.ledger.considered();
    let within_batch = ctx.within_batch_failures.skip_count();
    let any_candidates = ctx.any_candidates.load(Ordering::Relaxed);
    if neuron_pass_should_abort(saturated, considered, within_batch, any_candidates) {
        ctx.saturation_aborted.store(true, Ordering::Relaxed);
        tracing::warn!(
            saturated_drops = saturated,
            total_rejections = rejection_sample(saturated, considered, within_batch),
            proposals_formed = considered,
            within_batch_skips = within_batch,
            "pass aborted: saturation-dominant (target_saturated), remaining \
             targets skipped"
        );
    }
}

#[cfg(test)]
mod saturation_early_exit_tests {
    use super::{accept_candidate, neuron_pass_should_abort};
    use crate::CandidateNeuronJson;
    use crate::analysis::diagnostics::{
        NeuronDiagnostics, TARGET_SATURATED_EARLY_EXIT_MIN_SAMPLE,
        target_saturated_should_abort_pass,
    };
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Recorded production shape: a 40-minute zero-candidate pass whose
    /// rejections were 3080 saturated drops.
    const RECORDED_SATURATED_DROPS: u32 = 3080;

    fn test_candidate() -> CandidateNeuronJson {
        CandidateNeuronJson {
            source_neuron_uuid: "source-0".to_string(),
            target_neuron_uuid: "output-0".to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: 1.0,
            outgoing_weight: 1.0,
            squash: "RELU".to_string(),
            bias: 0.0,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.1,
            expected_creature_score_gain: 0.1,
            improved_count: 10,
            total_count: 10,
            improvement_magnitude_ratio: None,
            target_neuron_stats: None,
            prediction_confidence: 0.5,
            expected_score_gain_confidence_interval: [0.1, 0.1],
            target_saturation_factor: None,
            variant_key: None,
        }
    }

    /// Admitting a candidate is what marks the pass productive — the signal
    /// comes from the accept path itself, not from probing the candidate map.
    #[test]
    fn accepting_a_candidate_marks_the_pass_productive() {
        let helpful_map = Arc::new(Mutex::new(HashMap::new()));
        let any_candidates = Arc::new(AtomicBool::new(false));

        accept_candidate(&helpful_map, &any_candidates, test_candidate())
            .expect("candidate must be admitted");

        assert!(
            !helpful_map.lock().is_empty(),
            "the candidate must reach the surface's candidate map"
        );
        assert!(
            any_candidates.load(Ordering::Relaxed),
            "admitting a candidate must mark the pass productive"
        );
    }

    /// Issue #4140 regression: neuron targets are analysed in parallel, so the
    /// shared `helpful_map` lock is routinely held by another thread. Deriving
    /// "did this pass keep anything?" from a `try_lock` probe read a contended
    /// map as *empty* and aborted a productive pass, losing candidates. The
    /// productivity signal must survive contention.
    #[test]
    fn productive_pass_does_not_abort_while_helpful_map_is_locked() {
        let helpful_map = Arc::new(Mutex::new(HashMap::new()));
        let any_candidates = Arc::new(AtomicBool::new(false));
        accept_candidate(&helpful_map, &any_candidates, test_candidate())
            .expect("candidate must be admitted");

        // Another target holds the map lock while this thread evaluates the
        // abort — exactly the state a `try_lock` probe mis-read.
        let _contending_guard = helpful_map.lock();
        assert!(
            helpful_map.try_lock().is_none(),
            "precondition: a contended probe cannot see the kept candidate"
        );

        assert!(
            !neuron_pass_should_abort(
                RECORDED_SATURATED_DROPS,
                RECORDED_SATURATED_DROPS,
                0,
                any_candidates.load(Ordering::Relaxed),
            ),
            "a pass that kept a candidate must never abort, contended or not"
        );
    }

    /// The saturated pass still trips: no candidate was ever admitted, so the
    /// pass stops instead of forming thousands more doomed proposals.
    #[test]
    fn saturation_dominant_pass_with_no_candidates_aborts() {
        let any_candidates = Arc::new(AtomicBool::new(false));
        assert!(
            neuron_pass_should_abort(
                RECORDED_SATURATED_DROPS,
                RECORDED_SATURATED_DROPS,
                0,
                any_candidates.load(Ordering::Relaxed),
            ),
            "a saturation-dominated pass that kept nothing must abort"
        );
    }

    /// `within_batch_target_short_circuit` only ends a batch, so its skips are
    /// excluded from the rejection sample rather than inflating it.
    #[test]
    fn within_batch_skips_are_excluded_from_the_sample() {
        // 100 formed proposals, 100 of them within-batch skips, no saturation:
        // nothing terminal happened, so the pass runs on.
        assert!(!neuron_pass_should_abort(0, 100, 100, false));
    }

    #[test]
    fn saturated_pass_exits_early_with_zero_candidate_summary() {
        let diagnostics = NeuronDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.record_target_saturated_drops(TARGET_SATURATED_EARLY_EXIT_MIN_SAMPLE);
        let saturated = diagnostics.target_saturated_drop_count();
        assert!(
            target_saturated_should_abort_pass(saturated, saturated, false),
            "a saturation-dominated sample must trip the early exit"
        );
        assert_eq!(
            saturated, TARGET_SATURATED_EARLY_EXIT_MIN_SAMPLE,
            "trip uses target_saturated_drop_count(), not a parallel counter"
        );
    }

    #[test]
    fn non_saturated_pass_candidate_count_unchanged() {
        // Minority saturation (or any kept candidate) must not abort — the
        // productive fixture's candidate count is unchanged.
        assert!(!target_saturated_should_abort_pass(10, 100, false));
        assert!(!target_saturated_should_abort_pass(
            TARGET_SATURATED_EARLY_EXIT_MIN_SAMPLE,
            TARGET_SATURATED_EARLY_EXIT_MIN_SAMPLE,
            true,
        ));
    }
}
