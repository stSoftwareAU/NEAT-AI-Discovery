//! Neuron analysis post-processing — impact discounting, sorting, filtering,
//! and result assembly.
//!
//! Extracted from neuron.rs as part of issue #598.

use crate::{AnalyzeNeuronsInput, CandidateNeuronJson};
use anyhow::Result;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::analysis::cache::RecordCache;
use crate::analysis::diagnostics::{NeuronDiagnostics, compute_impact_scores_for_discounting};
use crate::analysis::gpu::GpuAnalyzer;
use crate::analysis::scoring::calibration_correction::{
    CHANGE_TYPE_ADD_NEURONS, CalibrationCorrection,
};
use crate::analysis::shared::AnalyzeNeuronsResult;
use crate::analysis::synapse::{
    apply_logistic_prediction_calibration, apply_neuron_pessimism_discount,
    apply_saturation_prediction_discount,
};
use crate::analysis::utils::{
    lock_or_bail, log_analysis_timeout, shuffle_within_top_k, verbose_enabled,
};

/// Parameters for building the final neuron analysis result.
pub(crate) struct NeuronResultParams<'a> {
    pub analysis_timed_out: &'a Arc<AtomicBool>,
    pub helpful_map: &'a Arc<Mutex<HashMap<u64, CandidateNeuronJson>>>,
    pub completed_count: &'a Arc<std::sync::atomic::AtomicUsize>,
    pub total_focus_count: usize,
    pub original_focus_count: usize,
    pub order_map: &'a Arc<HashMap<super::preparation::SharedUuid, usize>>,
    pub neuron_type_map: &'a HashMap<super::preparation::SharedUuid, String>,
    pub input: &'a AnalyzeNeuronsInput,
    pub cache: &'a Arc<RecordCache>,
    pub error_values_for_distribution: &'a [f32],
    pub timing_collector: &'a Arc<crate::analysis::shared::TimingCollector>,
    pub diagnostics: &'a Arc<NeuronDiagnostics>,
    pub gpu_used: bool,
}

/// Build the final neuron analysis result from the collected candidates.
///
/// This applies impact-based discounting, pessimism discount, filtering,
/// sorting, and assembles the result metadata.
pub(crate) fn build_neuron_results(
    params: &NeuronResultParams<'_>,
) -> Result<AnalyzeNeuronsResult> {
    let analysis_timed_out = params.analysis_timed_out.load(Ordering::Relaxed);
    let helpful_map = lock_or_bail(params.helpful_map, "helpful_map")?.clone();

    // Log timeout with completion stats (always visible, not just verbose)
    if analysis_timed_out {
        let completed = params
            .completed_count
            .load(std::sync::atomic::Ordering::Relaxed);
        log_analysis_timeout("neuron", completed, params.total_focus_count);
    }

    let mut helpful_results: Vec<CandidateNeuronJson> = helpful_map.into_values().collect();

    // Issue #1131: Derive per-creature calibration correction from the failure cache.
    let calibration_correction = CalibrationCorrection::from_failure_cache(
        params.input.failure_cache.as_deref().unwrap_or(&[]),
    );

    // Issue #1162: build a uuid -> target squash map so per-candidate
    // calibration can prefer the (change_type, target_squash) specific
    // correction when enough failure-cache evidence is available.
    let target_squash_map: HashMap<&str, &str> = params
        .input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.squash.as_str()))
        .collect();

    // Issue #128: Apply impact-based discounting and set creature-level metrics.
    apply_impact_discounting(
        &mut helpful_results,
        params.order_map,
        params.neuron_type_map,
        &target_squash_map,
        params.input,
        params.cache,
        &calibration_correction,
    );

    // Issue #557, #1191: Drop candidates whose expected gain is below the
    // absolute noise floor. Predictions in the 1e-7 range are dominated by
    // floating-point round-off in the downstream evaluator (see Issue #1189
    // failure cache evidence). The floor runs before the per-target / per-
    // squash diversity filters so noise-level candidates do not consume the
    // cap budget. The helper records its drops on the global counter
    // (`candidates_below_gain_floor_total`).
    let _floor_dropped = apply_min_expected_gain_floor_for_neurons(&mut helpful_results);

    // Sort by expected creature score gain (highest first) - Issue #128
    helpful_results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    // Production experiment: pair "extreme" candidates with a conservative variant.
    // Issue #1163: pass the per-variant calibration so each variant's
    // expected-improvement multiplier is `min(static, calibrated)`.
    helpful_results =
        crate::analysis::utils::pair_extreme_candidates_with_conservative_variants_calibrated(
            helpful_results,
            None,
            Some(&calibration_correction),
        );

    // Production guard rail (Dec 2025): only return candidates within sensible parameter ranges.
    helpful_results = crate::analysis::utils::filter_candidates_to_sensible_ranges(helpful_results);

    // Issue #1141: Enforce squash diversity within each target before the
    // per-target cap. If 17 candidates all propose the same `ReLU6` squash
    // for the same target neuron and `ReLU6` is the wrong squash for that
    // target, that is 17 near-identical failing bets. Keep only the
    // highest-gain candidate per `(target_uuid, squash)` pair so the
    // remaining cap budget is spent on genuinely distinct proposals.
    let same_target_squash_drops = apply_same_target_squash_diversity(&mut helpful_results);

    // Issue #1319: Under a OneHot task descriptor (e.g. CATEGORICAL_ERROR)
    // bias the per-target cap's distinct-target spread toward output neurons
    // (classes) with the highest cumulative per-target failure counts. The
    // helper returns `None` for every non-OneHot descriptor (including OTHER /
    // Unknown / absent) so the legacy allocation path is preserved verbatim.
    let class_priority = build_one_hot_class_priority(
        params.input.task_descriptor.as_ref(),
        params.input.failure_cache.as_deref().unwrap_or(&[]),
        params.neuron_type_map,
    );

    // Issue #1140: Cap add-neuron candidates per target within a single
    // discovery batch. Without this cap, a single hopeless target can consume
    // most of the budget with minor variants (e.g. 17 of 19 failed add-neuron
    // candidates in GRQ-sampler commit 744ac60d targeted the same neuron).
    // The cross-batch cooldown (Issue #1130) does not help within a batch.
    let per_target_cap_drops =
        apply_per_target_cap_with_priority(&mut helpful_results, class_priority.as_ref());

    // Deadline coverage (Jan 2026): diversify within the top-K
    if params.input.analysis_deadline_ms.is_some() {
        use crate::analysis::constants::DIVERSIFY_TOP_K;
        shuffle_within_top_k(
            helpful_results.as_mut_slice(),
            params.input.random_seed,
            "neuron:candidates:top_k",
            DIVERSIFY_TOP_K,
        );
    }

    // Track candidates_found AFTER pairing but BEFORE truncation.
    let candidates_found = helpful_results.len();

    // Apply max_candidates limit (truncation)
    if let Some(limit) = params.input.max_candidates {
        helpful_results.truncate(limit);
    }

    let candidates_returned = helpful_results.len();

    let no_candidate_reasons = params.diagnostics.no_candidate_summaries();
    params.diagnostics.emit_logs();

    // Issue #486 / #192: Compute error distribution from collected target neuron error samples.
    // Issue #834: Error values are now collected lock-free via Rayon fold/reduce.
    let error_distribution =
        crate::analysis::scoring::error_distribution::ErrorDistribution::from_errors(
            params.error_values_for_distribution,
        );

    Ok(AnalyzeNeuronsResult {
        helpful_neurons: helpful_results,
        gpu_used: params.gpu_used,
        no_candidate_reasons,
        metadata: crate::analysis::shared::NeuronAnalysisMetadata {
            candidates_found,
            candidates_returned,
            timed_out: analysis_timed_out,
            completed_focus_neurons: params
                .completed_count
                .load(std::sync::atomic::Ordering::Relaxed),
            total_focus_neurons: params.original_focus_count,
            timing: params.timing_collector.finalize(),
            gpu_info: GpuAnalyzer::get_adapter_info(),
            error_distribution,
            // Issue #1129: populated by orchestration from neuron no-candidate
            // summaries after the result is built.
            rejection_breakdown: {
                let mut breakdown = crate::analysis::diagnostics::RejectionBreakdown::new();
                // Issue #1140: surface per-target cap drops in the structured
                // rejection breakdown so operators can see why candidates
                // were removed.
                breakdown.record_many_u32(
                    crate::analysis::diagnostics::rejection_reasons::REJECTION_PER_TARGET_CAP,
                    u32::try_from(per_target_cap_drops).unwrap_or(u32::MAX),
                );
                // Issue #1141: surface squash-diversity drops in the structured
                // rejection breakdown so operators can see how many duplicate
                // (target, squash) candidates were filtered.
                breakdown.record_many_u32(
                    crate::analysis::diagnostics::rejection_reasons::REJECTION_SAME_TARGET_SQUASH_DUPLICATE,
                    u32::try_from(same_target_squash_drops).unwrap_or(u32::MAX),
                );
                // Issue #1143: surface target-saturation drops so operators
                // can see when the pre-check gated proposals against a fully
                // saturated target neuron.
                breakdown.record_many_u32(
                    crate::analysis::diagnostics::rejection_reasons::REJECTION_TARGET_SATURATED,
                    params.diagnostics.target_saturated_drop_count(),
                );
                breakdown
            },
            top_level_summary: None,
            // Issue #1131: per-creature calibration corrections derived from failure cache.
            calibration_corrections: calibration_correction.as_map().clone(),
            // Issue #1132: populated by orchestration once the outcome log is decided.
            discovery_mode: crate::analysis::discovery_mode::DiscoveryMode::Normal,
            rolling_success_rate: 1.0,
            // Issue #1202: populated by orchestration when the trailing-failure
            // streak crosses the configured drought threshold.
            drought_diagnostic: None,
            // Issue #1424: populated by orchestration on the alarm crossing.
            creature_drought_alarm: None,
            // Issue #1444: populated by orchestration's fail-fast gate only.
            insufficient_recording: None,
        },
    })
}

/// Apply impact-based discounting to neuron candidates.
///
/// Output neurons have impact = 1.0 (no discount).
/// Hidden neurons have impact in [0, 1] based on their weighted paths to outputs.
fn apply_impact_discounting(
    helpful_results: &mut [CandidateNeuronJson],
    order_map_arc: &Arc<HashMap<super::preparation::SharedUuid, usize>>,
    neuron_type_map: &HashMap<super::preparation::SharedUuid, String>,
    target_squash_map: &HashMap<&str, &str>,
    input: &AnalyzeNeuronsInput,
    cache: &Arc<RecordCache>,
    calibration_correction: &CalibrationCorrection,
) {
    let impact_scores = compute_impact_scores_for_discounting(&input.creature, cache.as_ref());
    for candidate in helpful_results.iter_mut() {
        candidate.source_neuron_index = order_map_arc
            .get(candidate.source_neuron_uuid.as_str())
            .copied();
        candidate.target_neuron_index = order_map_arc
            .get(candidate.target_neuron_uuid.as_str())
            .copied();

        let is_hidden = neuron_type_map
            .get(candidate.target_neuron_uuid.as_str())
            .is_none_or(|t| t != "output");

        let impact = if is_hidden {
            if let Some(&impact) = impact_scores.get(&candidate.target_neuron_uuid) {
                impact.clamp(0.0, 1.0)
            } else {
                // No impact score means disconnected from outputs - heavy discount
                0.1
            }
        } else {
            // Output neuron - full impact
            1.0
        };

        // Update creature-level metrics
        candidate.target_neuron_impact = impact;
        let original = candidate.expected_creature_error_reduction;
        candidate.expected_creature_error_reduction *= impact;
        candidate.expected_creature_score_gain = candidate.expected_creature_error_reduction;

        // Issue #791: Apply neuron-specific pessimism discount based on improved sample ratio.
        // Neuron candidates have a 15% success rate (vs higher synapse rates), so they
        // use more aggressive discounting parameters (lower floor, higher exponent).
        candidate.expected_creature_score_gain = apply_neuron_pessimism_discount(
            candidate.expected_creature_score_gain,
            candidate.improved_count,
            candidate.total_count,
            // Issue #1161: combine the binary improved-ratio with the magnitude-weighted ratio.
            candidate.improvement_magnitude_ratio,
        );

        // Issue #1112: Apply saturation-aware prediction discount.
        // Targets near activation saturation bounds cannot respond to perturbations,
        // so predictions are heavily over-estimated and need proportional discounting.
        candidate.expected_creature_score_gain = apply_saturation_prediction_discount(
            candidate.expected_creature_score_gain,
            candidate.target_saturation_factor,
        );

        // Issue #1056: Apply logistic prediction calibration to correct ~18× overestimation.
        // The non-linear calibration uses the improved ratio to modulate the base
        // factor, matching GRQ-sampler data showing ~2.7% actual success rate (28/1028).
        //
        // Issue #1131: Multiplied by the per-creature calibration correction derived
        // from the failure cache so creatures with poor recent prediction accuracy
        // receive additional discounting.
        //
        // Issue #1162: prefer the per-(change_type, target_squash) specific
        // correction when enough failure-cache evidence has accumulated for
        // this candidate's target squash, otherwise fall back to the
        // per-change_type correction.
        let target_squash = target_squash_map
            .get(candidate.target_neuron_uuid.as_str())
            .copied();
        let neuron_calibration = crate::analysis::constants::NEURON_PREDICTION_CALIBRATION
            * calibration_correction.correction_for(CHANGE_TYPE_ADD_NEURONS, target_squash);
        candidate.expected_creature_score_gain = apply_logistic_prediction_calibration(
            candidate.expected_creature_score_gain,
            candidate.improved_count,
            candidate.total_count,
            neuron_calibration,
        );

        if verbose_enabled() && is_hidden {
            tracing::trace!(
                target_uuid = %&candidate.target_neuron_uuid[..12.min(candidate.target_neuron_uuid.len())],
                impact = format_args!("{impact:.3}"),
                original_pct = format_args!("{:.4}", original * 100.0),
                discounted_pct = format_args!("{:.4}", candidate.expected_creature_score_gain * 100.0),
                "Neuron candidate impact discount applied"
            );
        }
    }
}

/// Drop add-neuron candidates whose `expected_creature_score_gain` is below
/// the absolute noise floor (Issue #1191).
///
/// Reads the configured floor via [`min_expected_creature_score_gain`]
/// (overridable through `NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN`). Each drop is
/// recorded against the global
/// [`candidates_below_gain_floor_total`](crate::observability::GainFloorMetrics)
/// counter so operators can see how many candidates the floor removes per
/// batch. Returns the number of candidates dropped so callers can record it
/// in the rejection breakdown.
///
/// Applied before the per-target / per-squash diversity filters so
/// noise-level proposals do not consume the cap budget.
pub fn apply_min_expected_gain_floor_for_neurons(
    candidates: &mut Vec<CandidateNeuronJson>,
) -> usize {
    let floor = crate::analysis::constants::min_expected_creature_score_gain();
    let before = candidates.len();
    candidates.retain(|c| c.expected_creature_score_gain >= floor);
    let dropped = before - candidates.len();
    crate::observability::global_gain_floor_metrics().record_dropped(dropped);
    dropped
}

/// Enforce squash diversity within each target neuron (Issue #1141).
///
/// Sorts `candidates` by `expected_creature_score_gain` descending (NaN-safe
/// via `total_cmp`), then retains only the highest-gain candidate for each
/// distinct `(target_neuron_uuid, squash)` pair. Lower-gain candidates whose
/// squash matches that of an already-retained candidate for the same target
/// are dropped. Returns the number of candidates dropped so callers can
/// record it in the rejection breakdown.
///
/// This filter runs before [`apply_per_target_cap`] so the remaining
/// per-target budget is spent on genuinely distinct squash proposals rather
/// than near-duplicate variants of the same squash.
pub(crate) fn apply_same_target_squash_diversity(
    candidates: &mut Vec<CandidateNeuronJson>,
) -> usize {
    if candidates.is_empty() {
        return 0;
    }

    // Sort by gain descending so retained candidates per (target, squash) are
    // the highest-gain ones. `total_cmp` provides a total order including NaN
    // and ties break deterministically.
    candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    let original_len = candidates.len();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    candidates.retain(|candidate| {
        let key = (
            candidate.target_neuron_uuid.clone(),
            candidate.squash.clone(),
        );
        seen.insert(key)
    });
    original_len - candidates.len()
}

/// Cap add-neuron candidates per target neuron within a single discovery
/// batch (Issue #1140).
///
/// Sorts `candidates` by `expected_creature_score_gain` descending (NaN-safe
/// via `total_cmp`), applies the cross-target diversity spread (Issue #1193),
/// then retains only the top-K candidates per `target_neuron_uuid`, where K
/// is [`max_add_neuron_candidates_per_target`]. The retained candidates start
/// with up to [`min_distinct_targets_per_batch`] distinct targets (in
/// gain-descending order) followed by the remaining candidates in gain-
/// descending order. Returns the number of candidates dropped by the cap so
/// callers can record it in the rejection breakdown.
#[cfg(test)]
pub(crate) fn apply_per_target_cap(candidates: &mut Vec<CandidateNeuronJson>) -> usize {
    apply_per_target_cap_with_priority(candidates, None)
}

/// Issue #1319: Like [`apply_per_target_cap`] but accepts an optional
/// per-target priority map. When `Some`, the cross-target spread step is
/// replaced by
/// [`crate::analysis::one_hot_class_allocation::apply_class_priority_spread`],
/// which pulls the highest-priority targets (e.g. worst-performing output
/// classes under a `OneHot` descriptor) to the front of the list before the
/// per-target cap is applied. Passing `None` reproduces the legacy
/// allocation verbatim — regression guard.
pub(crate) fn apply_per_target_cap_with_priority(
    candidates: &mut Vec<CandidateNeuronJson>,
    class_priority: Option<&HashMap<String, u32>>,
) -> usize {
    let cap = crate::analysis::constants::max_add_neuron_candidates_per_target();
    if candidates.is_empty() || cap == 0 {
        return 0;
    }

    // Sort by gain descending so that retained candidates per target are the
    // highest-gain ones. `total_cmp` provides a total order including NaN,
    // breaking ties deterministically.
    candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    // Issue #1193 / #1319: reorder so the top of the list covers at least
    // `MIN_DISTINCT_TARGETS_PER_BATCH` distinct targets when the pool supports
    // it. Under a OneHot descriptor with class-failure priority the worst-
    // performing classes are pulled to the front first.
    let min_distinct = crate::analysis::constants::min_distinct_targets_per_batch();
    match class_priority {
        Some(priority) if !priority.is_empty() => {
            crate::analysis::one_hot_class_allocation::apply_class_priority_spread(
                candidates,
                priority,
                min_distinct,
            );
        }
        _ => apply_distinct_target_spread(candidates),
    }

    let original_len = candidates.len();
    let mut per_target: HashMap<String, usize> = HashMap::new();
    candidates.retain(|candidate| {
        let count = per_target
            .entry(candidate.target_neuron_uuid.clone())
            .or_insert(0);
        if *count < cap {
            *count += 1;
            true
        } else {
            false
        }
    });
    original_len - candidates.len()
}

/// Issue #1319: Build the per-output-class priority map used by
/// [`apply_per_target_cap_with_priority`].
///
/// Returns `None` when the supplied `task_descriptor` is not `OneHot` (or is
/// absent / neutral / `OTHER`), preserving the legacy allocation path
/// verbatim — regression guard. When `OneHot`, returns a map from each output
/// neuron UUID to its cumulative within-batch / cross-batch failure count
/// derived from `failure_cache` (Issue #1131, #1194). An empty map (no
/// recorded class failures yet) is also returned as `None` so the legacy
/// spread keeps the gain-order signal when there is no per-class evidence.
fn build_one_hot_class_priority(
    task_descriptor: Option<&crate::analysis::task_descriptor::TaskDescriptor>,
    failure_cache: &[crate::analysis::scoring::calibration_correction::FailureCacheEntry],
    neuron_type_map: &HashMap<super::preparation::SharedUuid, String>,
) -> Option<HashMap<String, u32>> {
    let descriptor = task_descriptor?;
    let counts = crate::analysis::one_hot_class_allocation::compute_class_failure_counts(
        descriptor,
        failure_cache,
        |uuid| {
            neuron_type_map
                .get(uuid)
                .map(String::as_str)
                .is_some_and(|t| t == "output")
        },
    )?;
    if counts.is_empty() {
        None
    } else {
        Some(counts)
    }
}

/// Cross-target diversity spread (Issue #1193).
///
/// Assumes `candidates` is already sorted by `expected_creature_score_gain`
/// descending. When the pool contains at least
/// [`min_distinct_targets_per_batch`] distinct target neurons, reorders so
/// the top of the list contains the highest-gain candidate from each of the
/// first `min_distinct_targets_per_batch` distinct targets, followed by the
/// remaining candidates in their original gain-descending order. When the
/// pool has fewer distinct targets than the spread requires, falls through
/// without reordering.
///
/// The reorder is a stable partition: candidates kept "in front" appear in
/// the gain-rank order at which their target was first encountered, and
/// candidates pushed to the rear remain in gain-rank order relative to each
/// other.
pub(crate) fn apply_distinct_target_spread(candidates: &mut Vec<CandidateNeuronJson>) {
    let min_distinct = crate::analysis::constants::min_distinct_targets_per_batch();
    if candidates.len() <= 1 || min_distinct <= 1 {
        return;
    }

    // Count distinct targets in the pool. If the pool cannot support the
    // requested spread, fall through and let the existing gain-rank ordering
    // stand — Issue #1193 acceptance criteria 2.
    let distinct_count: usize = {
        let mut seen: HashSet<&str> = HashSet::new();
        for c in candidates.iter() {
            seen.insert(c.target_neuron_uuid.as_str());
        }
        seen.len()
    };
    if distinct_count < min_distinct {
        return;
    }

    // Walk the gain-sorted list once. The first occurrence of each new target
    // (until `min_distinct` distinct targets have been collected) goes to the
    // spread bucket; everything else falls to the rest bucket in gain order.
    let mut seen_targets: HashSet<String> = HashSet::new();
    let mut spread: Vec<CandidateNeuronJson> = Vec::with_capacity(min_distinct);
    let mut rest: Vec<CandidateNeuronJson> = Vec::with_capacity(candidates.len());
    for candidate in candidates.drain(..) {
        if spread.len() < min_distinct
            && !seen_targets.contains(candidate.target_neuron_uuid.as_str())
        {
            seen_targets.insert(candidate.target_neuron_uuid.clone());
            spread.push(candidate);
        } else {
            rest.push(candidate);
        }
    }

    candidates.extend(spread);
    candidates.extend(rest);
}

// =============================================================================
// Tests (Issue #1140)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::{
        apply_distinct_target_spread, apply_per_target_cap, apply_per_target_cap_with_priority,
        apply_same_target_squash_diversity,
    };
    use crate::CandidateNeuronJson;
    use serial_test::serial;
    use std::collections::HashSet;

    fn test_candidate_with_squash(
        target_uuid: &str,
        gain: f32,
        squash: &str,
    ) -> CandidateNeuronJson {
        CandidateNeuronJson {
            source_neuron_uuid: format!("source-{gain}-{squash}"),
            target_neuron_uuid: target_uuid.to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: 1.0,
            outgoing_weight: 1.0,
            squash: squash.to_string(),
            bias: 0.0,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: gain,
            expected_creature_score_gain: gain,
            improved_count: 10,
            total_count: 10,
            improvement_magnitude_ratio: None,
            target_neuron_stats: None,
            prediction_confidence: 0.5,
            expected_score_gain_confidence_interval: [gain, gain],
            target_saturation_factor: None,
            variant_key: None,
        }
    }

    /// Build a candidate using a squash that varies with the gain so cap
    /// tests are not unintentionally affected by squash dedup. Each test
    /// candidate ends up with a unique squash within its target.
    fn test_candidate(target_uuid: &str, gain: f32) -> CandidateNeuronJson {
        let squash = format!("SQUASH-{gain}");
        test_candidate_with_squash(target_uuid, gain, &squash)
    }

    #[test]
    #[serial]
    fn per_target_cap_truncates_target_over_limit() {
        // Seven candidates for one target; cap defaults to 3.
        let mut candidates = vec![
            test_candidate("target-A", 0.1),
            test_candidate("target-A", 0.5),
            test_candidate("target-A", 0.2),
            test_candidate("target-A", 0.9),
            test_candidate("target-A", 0.3),
            test_candidate("target-A", 0.7),
            test_candidate("target-A", 0.4),
        ];

        let dropped = apply_per_target_cap(&mut candidates);

        assert_eq!(candidates.len(), 3, "target-A should be capped to 3");
        assert_eq!(dropped, 4, "four candidates should be dropped");

        // Retained candidates must be the highest-gain ones (0.9, 0.7, 0.5).
        let retained_gains: Vec<f32> = candidates
            .iter()
            .map(|c| c.expected_creature_score_gain)
            .collect();
        assert_eq!(retained_gains, vec![0.9, 0.7, 0.5]);
    }

    #[test]
    #[serial]
    fn per_target_cap_leaves_targets_under_limit_unchanged() {
        // Two candidates for target-A, one for target-B; both under the cap of 3.
        let mut candidates = vec![
            test_candidate("target-A", 0.5),
            test_candidate("target-A", 0.2),
            test_candidate("target-B", 0.8),
        ];

        let dropped = apply_per_target_cap(&mut candidates);

        assert_eq!(candidates.len(), 3, "no candidate should be dropped");
        assert_eq!(dropped, 0, "zero candidates should be dropped");

        // Per-target counts must be preserved.
        let target_a: Vec<f32> = candidates
            .iter()
            .filter(|c| c.target_neuron_uuid == "target-A")
            .map(|c| c.expected_creature_score_gain)
            .collect();
        assert_eq!(target_a.len(), 2);
        let target_b: Vec<f32> = candidates
            .iter()
            .filter(|c| c.target_neuron_uuid == "target-B")
            .map(|c| c.expected_creature_score_gain)
            .collect();
        assert_eq!(target_b.len(), 1);
    }

    #[test]
    #[serial]
    fn per_target_cap_breakdown_reports_drop() {
        // Build a mix: target-A exceeds the cap, target-B is under it.
        let mut candidates = vec![
            test_candidate("target-A", 0.9),
            test_candidate("target-A", 0.8),
            test_candidate("target-A", 0.7),
            test_candidate("target-A", 0.6),
            test_candidate("target-A", 0.5),
            test_candidate("target-B", 0.4),
        ];

        let dropped = apply_per_target_cap(&mut candidates);
        assert_eq!(dropped, 2, "two target-A candidates should be dropped");

        // Feed into a RejectionBreakdown exactly as build_neuron_results does.
        let mut breakdown = crate::analysis::diagnostics::RejectionBreakdown::new();
        breakdown.record_many_u32(
            crate::analysis::diagnostics::rejection_reasons::REJECTION_PER_TARGET_CAP,
            u32::try_from(dropped).expect("fits in u32"),
        );

        assert_eq!(
            breakdown
                .counts()
                .get(crate::analysis::diagnostics::rejection_reasons::REJECTION_PER_TARGET_CAP,),
            Some(&2),
            "rejection breakdown should report two per-target-cap drops"
        );
        assert_eq!(breakdown.total(), 2);
    }

    #[test]
    fn squash_diversity_keeps_one_candidate_per_distinct_squash() {
        // Five candidates targeting the same neuron with squashes
        // [ReLU6, ReLU6, SOFTSIGN, ReLU6, ArcTan]. Highest-gain ReLU6 should
        // survive plus SOFTSIGN and ArcTan — three distinct squashes.
        let mut candidates = vec![
            test_candidate_with_squash("target-A", 0.9, "ReLU6"),
            test_candidate_with_squash("target-A", 0.5, "ReLU6"),
            test_candidate_with_squash("target-A", 0.7, "SOFTSIGN"),
            test_candidate_with_squash("target-A", 0.3, "ReLU6"),
            test_candidate_with_squash("target-A", 0.6, "ArcTan"),
        ];

        let dropped = apply_same_target_squash_diversity(&mut candidates);

        assert_eq!(
            dropped, 2,
            "two duplicate-squash ReLU6 candidates should be dropped"
        );
        assert_eq!(candidates.len(), 3);

        // Order is gain-descending after sort: ReLU6(0.9), SOFTSIGN(0.7), ArcTan(0.6).
        let kept: Vec<(&str, f32)> = candidates
            .iter()
            .map(|c| (c.squash.as_str(), c.expected_creature_score_gain))
            .collect();
        assert_eq!(
            kept,
            vec![("ReLU6", 0.9), ("SOFTSIGN", 0.7), ("ArcTan", 0.6)],
            "highest-gain ReLU6 plus SOFTSIGN plus ArcTan should remain"
        );
    }

    #[test]
    fn squash_diversity_independent_targets_not_collapsed() {
        // Same squash on different targets must not be deduplicated.
        let mut candidates = vec![
            test_candidate_with_squash("target-A", 0.5, "ReLU6"),
            test_candidate_with_squash("target-B", 0.4, "ReLU6"),
            test_candidate_with_squash("target-C", 0.3, "ReLU6"),
        ];

        let dropped = apply_same_target_squash_diversity(&mut candidates);

        assert_eq!(dropped, 0, "different targets must not collide");
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn squash_diversity_empty_input_is_safe() {
        let mut candidates: Vec<CandidateNeuronJson> = Vec::new();
        let dropped = apply_same_target_squash_diversity(&mut candidates);
        assert_eq!(dropped, 0);
        assert!(candidates.is_empty());
    }

    #[test]
    fn squash_diversity_breakdown_reports_drop() {
        // Three duplicate-squash candidates targeting the same neuron.
        let mut candidates = vec![
            test_candidate_with_squash("target-A", 0.9, "ReLU6"),
            test_candidate_with_squash("target-A", 0.8, "ReLU6"),
            test_candidate_with_squash("target-A", 0.7, "ReLU6"),
            test_candidate_with_squash("target-A", 0.6, "TANH"),
        ];

        let dropped = apply_same_target_squash_diversity(&mut candidates);
        assert_eq!(
            dropped, 2,
            "two duplicate ReLU6 candidates should be dropped"
        );

        // Feed into a RejectionBreakdown exactly as build_neuron_results does.
        let mut breakdown = crate::analysis::diagnostics::RejectionBreakdown::new();
        breakdown.record_many_u32(
            crate::analysis::diagnostics::rejection_reasons::REJECTION_SAME_TARGET_SQUASH_DUPLICATE,
            u32::try_from(dropped).expect("fits in u32"),
        );

        assert_eq!(
            breakdown.counts().get(
                crate::analysis::diagnostics::rejection_reasons::REJECTION_SAME_TARGET_SQUASH_DUPLICATE,
            ),
            Some(&2),
            "rejection breakdown should report two squash-duplicate drops"
        );
        assert_eq!(breakdown.total(), 2);
    }

    // =========================================================================
    // Issue #1193 — cross-target diversity spread
    // =========================================================================

    #[test]
    #[serial]
    fn distinct_target_spread_reorders_when_pool_supports_min() {
        // Top six candidates all hit target-A; three other targets have one
        // candidate each lower down the list. With MIN_DISTINCT=3, the front
        // of the reordered list must contain three distinct targets.
        let mut candidates = vec![
            test_candidate("target-A", 0.99),
            test_candidate("target-A", 0.95),
            test_candidate("target-A", 0.92),
            test_candidate("target-A", 0.90),
            test_candidate("target-A", 0.87),
            test_candidate("target-A", 0.85),
            test_candidate("target-B", 0.50),
            test_candidate("target-C", 0.40),
            test_candidate("target-D", 0.30),
        ];

        // Pre-sort to mirror the precondition documented on the function.
        candidates.sort_by(|a, b| {
            b.expected_creature_score_gain
                .total_cmp(&a.expected_creature_score_gain)
        });

        apply_distinct_target_spread(&mut candidates);

        // Front three positions cover three distinct targets: A, B, C
        // (highest-gain occurrence of each in gain order).
        let front_targets: Vec<&str> = candidates[0..3]
            .iter()
            .map(|c| c.target_neuron_uuid.as_str())
            .collect();
        assert_eq!(front_targets, vec!["target-A", "target-B", "target-C"]);

        // Front candidates are the highest-gain entry per distinct target.
        assert!((candidates[0].expected_creature_score_gain - 0.99).abs() < f32::EPSILON);
        assert!((candidates[1].expected_creature_score_gain - 0.50).abs() < f32::EPSILON);
        assert!((candidates[2].expected_creature_score_gain - 0.40).abs() < f32::EPSILON);

        // Tail preserves gain order for the remaining candidates.
        let tail_gains: Vec<f32> = candidates[3..]
            .iter()
            .map(|c| c.expected_creature_score_gain)
            .collect();
        let mut sorted_tail = tail_gains.clone();
        sorted_tail.sort_by(|a, b| b.total_cmp(a));
        assert_eq!(
            tail_gains, sorted_tail,
            "tail must remain in gain-descending order"
        );
        // No candidates are dropped.
        assert_eq!(candidates.len(), 9);
    }

    #[test]
    #[serial]
    fn distinct_target_spread_falls_through_when_pool_too_narrow() {
        // Only two distinct targets — below MIN_DISTINCT (3) — so the spread
        // must leave the gain-sorted ordering untouched.
        let mut candidates = vec![
            test_candidate("target-A", 0.9),
            test_candidate("target-A", 0.8),
            test_candidate("target-A", 0.7),
            test_candidate("target-B", 0.6),
        ];
        candidates.sort_by(|a, b| {
            b.expected_creature_score_gain
                .total_cmp(&a.expected_creature_score_gain)
        });
        let before: Vec<(String, f32)> = candidates
            .iter()
            .map(|c| (c.target_neuron_uuid.clone(), c.expected_creature_score_gain))
            .collect();

        apply_distinct_target_spread(&mut candidates);

        let after: Vec<(String, f32)> = candidates
            .iter()
            .map(|c| (c.target_neuron_uuid.clone(), c.expected_creature_score_gain))
            .collect();
        assert_eq!(
            before, after,
            "fall-through path must not reorder when distinct < MIN"
        );
    }

    #[test]
    #[serial]
    fn per_target_cap_emits_distinct_targets_when_top_dominated_by_one_target() {
        // Acceptance: top six candidates all hit target-A; three other targets
        // exist. After cap, the emitted batch must contain at least
        // `MIN_DISTINCT_TARGETS_PER_BATCH` distinct targets.
        let mut candidates = vec![
            test_candidate("target-A", 0.99),
            test_candidate("target-A", 0.95),
            test_candidate("target-A", 0.92),
            test_candidate("target-A", 0.90),
            test_candidate("target-A", 0.87),
            test_candidate("target-A", 0.85),
            test_candidate("target-B", 0.50),
            test_candidate("target-C", 0.40),
            test_candidate("target-D", 0.30),
        ];

        let _dropped = apply_per_target_cap(&mut candidates);

        let distinct: HashSet<&str> = candidates
            .iter()
            .map(|c| c.target_neuron_uuid.as_str())
            .collect();
        let min_distinct = crate::analysis::constants::min_distinct_targets_per_batch();
        assert!(
            distinct.len() >= min_distinct,
            "emitted batch must include at least {} distinct targets, got {}",
            min_distinct,
            distinct.len()
        );

        // The front three positions of the cap output must each hit a distinct
        // target — that is the visible signature of the spread.
        let front_targets: HashSet<&str> = candidates[0..3]
            .iter()
            .map(|c| c.target_neuron_uuid.as_str())
            .collect();
        assert_eq!(
            front_targets.len(),
            3,
            "first three slots of the emitted batch should each hit a distinct target"
        );
    }

    #[test]
    #[serial]
    fn per_target_cap_preserves_full_quota_when_no_alternatives() {
        // Only one distinct target exists. The per-target cap of three must
        // still admit that target's full quota — the spread must not reduce
        // the cap below `max_add_neuron_candidates_per_target`.
        let mut candidates = vec![
            test_candidate("target-A", 0.9),
            test_candidate("target-A", 0.8),
            test_candidate("target-A", 0.7),
            test_candidate("target-A", 0.6),
            test_candidate("target-A", 0.5),
        ];

        let dropped = apply_per_target_cap(&mut candidates);

        assert_eq!(candidates.len(), 3, "cap should retain the full quota of 3");
        assert_eq!(dropped, 2);
        let retained_gains: Vec<f32> = candidates
            .iter()
            .map(|c| c.expected_creature_score_gain)
            .collect();
        assert_eq!(retained_gains, vec![0.9, 0.8, 0.7]);
    }

    #[test]
    #[serial]
    fn distinct_target_spread_env_override_controls_min() {
        // SAFETY: env access is serialised via `#[serial]`.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_MIN_DISTINCT_TARGETS_PER_BATCH", "1");
        }

        let mut candidates = vec![
            test_candidate("target-A", 0.9),
            test_candidate("target-A", 0.8),
            test_candidate("target-B", 0.5),
        ];
        candidates.sort_by(|a, b| {
            b.expected_creature_score_gain
                .total_cmp(&a.expected_creature_score_gain)
        });
        let before: Vec<(String, f32)> = candidates
            .iter()
            .map(|c| (c.target_neuron_uuid.clone(), c.expected_creature_score_gain))
            .collect();

        apply_distinct_target_spread(&mut candidates);

        let after: Vec<(String, f32)> = candidates
            .iter()
            .map(|c| (c.target_neuron_uuid.clone(), c.expected_creature_score_gain))
            .collect();

        // SAFETY: env access is serialised via `#[serial]`.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_MIN_DISTINCT_TARGETS_PER_BATCH");
        }

        // With min_distinct = 1 the spread becomes a no-op (early return) and
        // the gain-sorted order is preserved verbatim.
        assert_eq!(before, after);
    }

    #[test]
    #[serial]
    fn per_target_cap_env_override_controls_limit() {
        // SAFETY: env access is serialised via `#[serial]`.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_MAX_ADD_NEURON_PER_TARGET", "1");
        }

        let mut candidates = vec![
            test_candidate("target-A", 0.9),
            test_candidate("target-A", 0.8),
            test_candidate("target-A", 0.7),
        ];
        let dropped = apply_per_target_cap(&mut candidates);

        // SAFETY: env access is serialised via `#[serial]`.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_MAX_ADD_NEURON_PER_TARGET");
        }

        assert_eq!(candidates.len(), 1, "cap=1 should leave only one candidate");
        assert_eq!(dropped, 2);
        assert!((candidates[0].expected_creature_score_gain - 0.9).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Issue #1319 — per-class capacity allocation under OneHot
    // =========================================================================

    /// When the per-target cap is invoked with a non-empty priority map, the
    /// spread step pulls the highest-priority targets to the front of the
    /// list before the cap is applied. Mirrors the `OneHot` acceptance criterion
    /// from issue #1319.
    #[test]
    #[serial]
    fn per_target_cap_with_priority_pulls_high_priority_targets_to_front() {
        let mut candidates = vec![
            test_candidate("target-A", 0.99),
            test_candidate("target-A", 0.95),
            test_candidate("target-A", 0.92),
            test_candidate("target-A", 0.90),
            test_candidate("target-A", 0.87),
            test_candidate("target-A", 0.85),
            test_candidate("target-B", 0.50),
            test_candidate("target-C", 0.40),
            test_candidate("target-D", 0.30),
        ];
        let mut priority: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        priority.insert("target-D".to_string(), 7);
        priority.insert("target-C".to_string(), 5);

        let _dropped = apply_per_target_cap_with_priority(&mut candidates, Some(&priority));

        // Cap defaults to 3; the spread fronts target-D and target-C ahead of
        // the cap, so they must be present in the emitted batch even though
        // target-A had the gain-dominant candidates.
        let distinct: HashSet<&str> = candidates
            .iter()
            .map(|c| c.target_neuron_uuid.as_str())
            .collect();
        assert!(distinct.contains("target-D"));
        assert!(distinct.contains("target-C"));

        // Front of the emitted list must lead with the priority targets.
        assert_eq!(candidates[0].target_neuron_uuid, "target-D");
        assert_eq!(candidates[1].target_neuron_uuid, "target-C");
    }

    /// Regression guard: `None` priority preserves the existing distinct-
    /// target spread behaviour byte-for-byte.
    #[test]
    #[serial]
    fn per_target_cap_with_no_priority_matches_legacy_path() {
        let mut legacy = vec![
            test_candidate("target-A", 0.99),
            test_candidate("target-A", 0.95),
            test_candidate("target-A", 0.92),
            test_candidate("target-A", 0.90),
            test_candidate("target-A", 0.87),
            test_candidate("target-A", 0.85),
            test_candidate("target-B", 0.50),
            test_candidate("target-C", 0.40),
            test_candidate("target-D", 0.30),
        ];
        let mut with_none = legacy.clone();

        let dropped_legacy = apply_per_target_cap(&mut legacy);
        let dropped_with_none = apply_per_target_cap_with_priority(&mut with_none, None);
        assert_eq!(dropped_legacy, dropped_with_none);
        let legacy_view: Vec<(String, f32)> = legacy
            .iter()
            .map(|c| (c.target_neuron_uuid.clone(), c.expected_creature_score_gain))
            .collect();
        let new_view: Vec<(String, f32)> = with_none
            .iter()
            .map(|c| (c.target_neuron_uuid.clone(), c.expected_creature_score_gain))
            .collect();
        assert_eq!(legacy_view, new_view);
    }

    /// Acceptance: an empty priority map is treated as no signal, so the
    /// legacy distinct-target spread runs (regression guard).
    #[test]
    #[serial]
    fn per_target_cap_with_empty_priority_uses_legacy_spread() {
        let empty: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut legacy = vec![
            test_candidate("target-A", 0.99),
            test_candidate("target-A", 0.95),
            test_candidate("target-B", 0.50),
            test_candidate("target-C", 0.40),
        ];
        let mut with_empty = legacy.clone();

        apply_per_target_cap(&mut legacy);
        apply_per_target_cap_with_priority(&mut with_empty, Some(&empty));

        let legacy_view: Vec<&str> = legacy
            .iter()
            .map(|c| c.target_neuron_uuid.as_str())
            .collect();
        let new_view: Vec<&str> = with_empty
            .iter()
            .map(|c| c.target_neuron_uuid.as_str())
            .collect();
        assert_eq!(legacy_view, new_view);
    }
}
