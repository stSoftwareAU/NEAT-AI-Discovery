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

    // Issue #557: Filter out candidates with non-positive expected_creature_score_gain.
    helpful_results.retain(|c| c.expected_creature_score_gain > 0.0);

    // Sort by expected creature score gain (highest first) - Issue #128
    helpful_results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    // Production experiment: pair "extreme" candidates with a conservative variant.
    helpful_results = crate::analysis::utils::pair_extreme_candidates_with_conservative_variants(
        helpful_results,
        None,
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

    // Issue #1140: Cap add-neuron candidates per target within a single
    // discovery batch. Without this cap, a single hopeless target can consume
    // most of the budget with minor variants (e.g. 17 of 19 failed add-neuron
    // candidates in GRQ-sampler commit 744ac60d targeted the same neuron).
    // The cross-batch cooldown (Issue #1130) does not help within a batch.
    let per_target_cap_drops = apply_per_target_cap(&mut helpful_results);

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
/// via `total_cmp`), then retains only the top-K candidates per
/// `target_neuron_uuid`, where K is
/// [`max_add_neuron_candidates_per_target`]. The retained candidates remain
/// in gain-descending order. Returns the number of candidates dropped by the
/// cap so callers can record it in the rejection breakdown.
pub(crate) fn apply_per_target_cap(candidates: &mut Vec<CandidateNeuronJson>) -> usize {
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

// =============================================================================
// Tests (Issue #1140)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::{apply_per_target_cap, apply_same_target_squash_diversity};
    use crate::CandidateNeuronJson;
    use serial_test::serial;

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
}
