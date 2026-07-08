//! Generic discovery module dispatch pattern (Issue #375).
//!
//! Extracts the repeated detect → convert → log → merge boilerplate from
//! `analyze_all()` into a single reusable function. Each discovery module
//! supplies its detection and conversion logic via closures, while this
//! module handles watchdog beats, phase timing, verbose logging, and
//! merging into the synapse result.
//!
//! ## Parallel dispatch (Issue #419)
//!
//! `run_discovery_modules_parallel()` runs all detection phases concurrently
//! via `rayon::into_par_iter()`, then merges results sequentially. This
//! preserves deterministic ordering while utilising multiple CPU cores.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use crate::CoordinatedStructuralCandidateJson;
use crate::CoordinatedStructuralOpJson;
use crate::CreatureJson;
use crate::observability::PhaseTimer;
use rayon::prelude::*;

use super::remove_neuron_gain::estimate_remove_neuron_gain;

use super::constants::{
    COORDINATED_MIN_EXPECTED_GAIN, MODULE_GATE_THRESHOLD, QUALITY_SKIP_GAIN_THRESHOLD,
    QUALITY_SKIP_MIN_CANDIDATES, SOFT_FAILURE_WEIGHT,
};
use super::diagnostics::rejection_reasons::{
    REJECTION_BELOW_EXPECTED_GAIN_FLOOR, REJECTION_BUDGET_TRUNCATED,
    REJECTION_COORDINATED_TARGET_CAP_EXCEEDED, REJECTION_MODULE_STARVED,
};
use super::module_starvation_tracker::ModuleStarvationTracker;
use super::module_weights::{DiscoveryModuleStatsJson, ModuleOutcomeTracker};
use super::shared;
use super::utils;

/// Return the target neuron UUID of the supplied operation, used as the
/// grouping key for the per-final-target coordinated-structural cap
/// (Issue #1271).
fn op_target_uuid(op: &CoordinatedStructuralOpJson) -> &str {
    match op {
        CoordinatedStructuralOpJson::RemoveSynapse { to_neuron_uuid, .. }
        | CoordinatedStructuralOpJson::AddSynapse { to_neuron_uuid, .. }
        | CoordinatedStructuralOpJson::SetWeight { to_neuron_uuid, .. } => to_neuron_uuid,
        CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. }
        | CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid }
        | CoordinatedStructuralOpJson::ChangeSquash { neuron_uuid, .. }
        | CoordinatedStructuralOpJson::SetBias { neuron_uuid, .. } => neuron_uuid,
    }
}

/// Cap coordinated-structural candidates per **final-operation** target
/// neuron within a single batch (Issue #1271).
///
/// Sorts `candidates` by `expected_creature_score_gain` descending (NaN-safe
/// via `total_cmp`), then retains only the top-K candidates per last-operation
/// target UUID, where K is
/// [`super::constants::max_coordinated_per_target_output`]. Returns the number
/// of candidates dropped by the cap so callers can record it in the rejection
/// breakdown under [`REJECTION_COORDINATED_TARGET_CAP_EXCEEDED`].
///
/// This cap mirrors the per-target add-neuron cap (Issue #1140) for the
/// coordinated-structural pipeline. It runs after the post-discount expected-
/// gain floor and **before** any downstream cross-target diversity reordering,
/// so it constrains the candidate pool rather than the emitted batch order.
///
/// Candidates whose last operation cannot supply a target UUID (an empty
/// operations vec, which should not occur in practice) are retained unchanged
/// — the cap is intentionally conservative and only drops candidates with an
/// identifiable target.
fn apply_coordinated_per_target_cap(
    candidates: &mut Vec<CoordinatedStructuralCandidateJson>,
) -> usize {
    let cap = super::constants::max_coordinated_per_target_output();
    if candidates.is_empty() || cap == 0 {
        return 0;
    }

    // Sort gain-descending so the retained candidates per target are the
    // highest-gain ones. `total_cmp` provides a total order including NaN,
    // breaking ties deterministically.
    candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    let original_len = candidates.len();
    let mut per_target: HashMap<String, usize> = HashMap::new();
    candidates.retain(|candidate| {
        let Some(last_op) = candidate.operations.last() else {
            // Candidate has no operations — keep it (the cap can only group
            // by a known final target).
            return true;
        };
        let target = op_target_uuid(last_op).to_string();
        let count = per_target.entry(target).or_insert(0);
        if *count < cap {
            *count += 1;
            true
        } else {
            false
        }
    });
    original_len - candidates.len()
}

/// Override the reported gain of every single-op `RemoveNeuron` coordinated
/// candidate with the honest, propagation-aware estimate (Issue #1530).
///
/// Milestone #1516 merged [`estimate_remove_neuron_gain`] (PR #1523) but nothing
/// in the live pipeline invoked it, so the emitted remove-neuron gain stayed the
/// fabricated NEAT-AI `#2483` placeholder (`+0.17879` on creature `45a04ef1`,
/// versus a measured `−0.00032`). This makes Discovery the source of truth: for
/// each candidate whose **sole** operation is a `RemoveNeuron`, the reported
/// `expected_creature_score_gain` is replaced with the propagation-aware
/// estimate for that neuron, which attenuates a deep neuron's influence all the
/// way to the output(s) and is signed (non-positive) — no fabricated large
/// positive gain survives to crowd out realistic candidates.
///
/// Multi-operation coordinated candidates are left untouched: their gain
/// reflects the combined effect of the whole atomic group, not a bare neuron
/// removal. Candidates whose neuron is absent or is an output (the estimator
/// returns `None`) are also left untouched.
///
/// Returns the number of candidates whose gain was overridden, for diagnostics.
// The estimator works in f64; the candidate carries f32. The gain is a small
// value in `[-1, 0]`, well within f32 range, so the narrowing is intentional
// precision loss, not overflow (Issue #873).
#[allow(clippy::cast_possible_truncation)]
pub fn apply_honest_remove_neuron_gain(
    creature: &CreatureJson,
    candidates: &mut [CoordinatedStructuralCandidateJson],
) -> usize {
    let mut overridden = 0;
    for candidate in candidates.iter_mut() {
        // A lone RemoveNeuron op is the only bare neuron removal; anything else
        // (multi-op group, or a different single op) is not this estimator's
        // concern.
        let [CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid }] =
            candidate.operations.as_slice()
        else {
            continue;
        };

        if let Some(gain) = estimate_remove_neuron_gain(creature, neuron_uuid) {
            candidate.expected_creature_score_gain = gain as f32;
            overridden += 1;
        }
    }
    overridden
}

/// Result of a discovery module's detection phase.
///
/// The `detected_count` is used for verbose logging (how many raw detections),
/// while `candidates` are the coordinated structural candidates to merge.
pub struct DiscoveryDetectionResult {
    pub detected_count: usize,
    pub candidates: Vec<CoordinatedStructuralCandidateJson>,
}

/// Run a single discovery detection module with the standard dispatch pattern:
///
/// 1. Watchdog beat (starting)
/// 2. `PhaseTimer` creation
/// 3. Call `detect_fn` which runs module-specific detection and conversion
/// 4. Verbose logging if results are non-empty
/// 5. Merge into synapse result via `merge_coordinated_structural_replacements`
/// 6. Watchdog beat (finished)
///
/// The `detect_fn` closure encapsulates all module-specific logic (record
/// collection, detection, and conversion to coordinated candidates).
#[tracing::instrument(skip_all, fields(module = module_name, phase = phase_name))]
pub fn run_discovery_module(
    syn: &mut shared::AnalyzeSynapsesResult,
    module_name: &str,
    phase_name: &'static str,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
    detect_fn: impl FnOnce() -> Option<DiscoveryDetectionResult>,
) {
    let starting = format!("analysis::analyze_all → {module_name} starting");
    let finished = format!("analysis::analyze_all → {module_name} finished");

    crate::watchdog::beat(&starting);
    let _timer = PhaseTimer::new(phase_name);

    if let Some(mut result) = detect_fn()
        && !result.candidates.is_empty()
    {
        // Issue #1110: Filter coordinated candidates below the minimum
        // expected-gain floor. Gains at 1e-8 to 1e-7 are indistinguishable
        // from numerical noise and harm the network in production.
        let before_floor = result.candidates.len();
        result
            .candidates
            .retain(|c| c.expected_creature_score_gain >= COORDINATED_MIN_EXPECTED_GAIN);
        let dropped_floor =
            u32::try_from(before_floor.saturating_sub(result.candidates.len())).unwrap_or(u32::MAX);
        syn.metadata
            .rejection_breakdown
            .record_many_u32(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, dropped_floor);

        // Issue #1271: After the post-discount floor, cap coordinated
        // candidates by their final-operation target neuron so a single
        // problematic target cannot consume the whole batch budget.
        let dropped_cap_usize = apply_coordinated_per_target_cap(&mut result.candidates);
        let dropped_cap = u32::try_from(dropped_cap_usize).unwrap_or(u32::MAX);
        syn.metadata
            .rejection_breakdown
            .record_many_u32(REJECTION_COORDINATED_TARGET_CAP_EXCEEDED, dropped_cap);

        if utils::verbose_enabled() {
            tracing::debug!(
                module = module_name,
                detections = result.detected_count,
                candidates = result.candidates.len(),
                "discovery module results"
            );
        }

        if !result.candidates.is_empty() {
            super::merge_coordinated_structural_replacements(
                syn,
                result.candidates,
                max_synapse_candidates,
                diversify,
            );
        }
    }

    crate::watchdog::beat(&finished);
}

/// Specification for a single discovery module to be dispatched in parallel.
///
/// Each module provides a name (for logging/watchdog), a phase name (for timing),
/// a maximum candidate budget (Issue #967), and a detection closure that returns
/// candidates independently.
pub struct DiscoveryModuleSpec {
    pub module_name: String,
    pub phase_name: &'static str,
    /// Maximum number of candidates this module should generate (Issue #967).
    /// Allocated by `allocate_candidate_budgets` based on historical success rate.
    pub max_candidates: usize,
    pub detect_fn: Box<dyn FnOnce() -> Option<DiscoveryDetectionResult> + Send>,
}

/// A single module's detection output, paired with its metadata for the merge phase.
pub struct DiscoveryModuleDetectionEntry {
    pub module_name: String,
    pub phase_name: &'static str,
    pub max_candidates: usize,
    pub result: Option<DiscoveryDetectionResult>,
    /// `true` when this module was skipped because the per-(creature, module)
    /// starvation tracker has it in active cooldown (Issue #1273). The merge
    /// phase uses this flag to record a `module_starved` rejection.
    pub starved: bool,
}

/// Collected detection results from all discovery modules (Issue #1004).
///
/// Returned by [`detect_discovery_modules_parallel`] for deferred merging,
/// allowing the detection phase to overlap with other concurrent work
/// (e.g., candidate compression).
pub struct DiscoveryModuleDetectionResults {
    pub entries: Vec<DiscoveryModuleDetectionEntry>,
}

/// Run all discovery module detection phases in parallel, returning results
/// without merging into the synapse result (Issue #1004).
///
/// Detection closures execute concurrently via `rayon::into_par_iter()`. Results
/// are collected in original module order (rayon preserves indexed iterator order),
/// ensuring deterministic output regardless of thread scheduling.
///
/// ## Module gating (Issue #1060)
///
/// When a tracker is provided, modules whose historical success rate is below
/// [`MODULE_GATE_THRESHOLD`] are skipped entirely, saving compute on consistently
/// failing modules.
///
/// ## Deadline enforcement (Issue #1029)
///
/// When a deadline is provided, each module checks `deadline_passed()` before
/// executing its detection closure. Modules that start after the deadline are
/// skipped (returning `None`), preventing the parallel detection phase from
/// running indefinitely when the analysis time budget is exhausted.
///
/// Call [`merge_discovery_module_results`] afterwards to merge into `syn`.
#[tracing::instrument(skip_all, fields(module_count = modules.len()))]
pub fn detect_discovery_modules_parallel(
    modules: Vec<DiscoveryModuleSpec>,
    deadline: Option<SystemTime>,
    tracker: Option<&ModuleOutcomeTracker>,
) -> DiscoveryModuleDetectionResults {
    detect_discovery_modules_parallel_with_starvation(modules, deadline, tracker, None, 0)
}

/// Run all discovery module detection phases in parallel, applying both the
/// population-wide module gate (Issue #1060) and the per-(creature, module)
/// starvation cooldown (Issue #1273).
///
/// Identical contract to [`detect_discovery_modules_parallel`] except modules
/// whose per-creature streak has tripped
/// [`ModuleStarvationTracker::is_starved`] for `current_epoch` are skipped and
/// flagged with `starved = true` on the returned entry so the merge phase
/// records a `module_starved` rejection.
#[tracing::instrument(skip_all, fields(module_count = modules.len()))]
pub fn detect_discovery_modules_parallel_with_starvation(
    modules: Vec<DiscoveryModuleSpec>,
    deadline: Option<SystemTime>,
    tracker: Option<&ModuleOutcomeTracker>,
    starvation_tracker: Option<&ModuleStarvationTracker>,
    current_epoch: u64,
) -> DiscoveryModuleDetectionResults {
    if modules.is_empty() {
        return DiscoveryModuleDetectionResults {
            entries: Vec::new(),
        };
    }

    crate::watchdog::beat("analysis::analyze_all → parallel discovery detection starting");
    let _timer = PhaseTimer::new("parallel_discovery_detection");

    // Issue #1029: Shared flag so that once the deadline is observed by any thread,
    // all remaining modules skip execution promptly without repeated syscalls.
    let timed_out = AtomicBool::new(false);

    // Parallel detection phase: run all closures concurrently.
    // `into_par_iter().map().collect()` preserves input order for indexed iterators.
    let entries: Vec<DiscoveryModuleDetectionEntry> = modules
        .into_par_iter()
        .map(|spec| {
            // Issue #1273: Skip detection if the per-(creature, module) starvation
            // tracker has the module in active cooldown.
            if let Some(s) = starvation_tracker
                && s.is_starved(&spec.module_name, current_epoch)
            {
                tracing::debug!(
                    module = %spec.module_name,
                    streak_threshold = s.failure_streak_threshold(),
                    cooldown_epochs = s.cooldown_epochs(),
                    "Skipping discovery module — per-creature starvation cooldown active \
                     (Issue #1273)"
                );
                return DiscoveryModuleDetectionEntry {
                    module_name: spec.module_name,
                    phase_name: spec.phase_name,
                    max_candidates: spec.max_candidates,
                    result: None,
                    starved: true,
                };
            }

            // Issue #1060: Skip detection if the module is gated (low success rate).
            if let Some(t) = tracker
                && t.is_gated(&spec.module_name, MODULE_GATE_THRESHOLD)
            {
                tracing::debug!(
                    module = %spec.module_name,
                    threshold = MODULE_GATE_THRESHOLD,
                    "Skipping discovery module — gated by low success rate (Issue #1060)"
                );
                return DiscoveryModuleDetectionEntry {
                    module_name: spec.module_name,
                    phase_name: spec.phase_name,
                    max_candidates: spec.max_candidates,
                    result: None,
                    starved: false,
                };
            }

            // Issue #1029: Skip detection if the analysis deadline has passed.
            if timed_out.load(Ordering::Relaxed) || utils::deadline_passed(&deadline) {
                timed_out.store(true, Ordering::Relaxed);
                tracing::debug!(
                    module = %spec.module_name,
                    "Skipping discovery module — deadline passed (Issue #1029)"
                );
                return DiscoveryModuleDetectionEntry {
                    module_name: spec.module_name,
                    phase_name: spec.phase_name,
                    max_candidates: spec.max_candidates,
                    result: None,
                    starved: false,
                };
            }

            // Issue #1087: Wrap detection closure with catch_unwind so a panic
            // in one module does not corrupt results from sibling modules.
            let result = match std::panic::catch_unwind(AssertUnwindSafe(|| (spec.detect_fn)())) {
                Ok(r) => r,
                Err(panic_payload) => {
                    let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        format!("{panic_payload:?}")
                    };
                    tracing::warn!(
                        module = %spec.module_name,
                        panic_message = %panic_msg,
                        "Discovery module panicked — caught and converted to empty result \
                         (Issue #1087)"
                    );
                    None
                }
            };
            DiscoveryModuleDetectionEntry {
                module_name: spec.module_name,
                phase_name: spec.phase_name,
                max_candidates: spec.max_candidates,
                result,
                starved: false,
            }
        })
        .collect();

    let skipped = timed_out.load(Ordering::Relaxed);
    if skipped {
        let skipped_count = entries.iter().filter(|e| e.result.is_none()).count();
        tracing::info!(
            skipped_count,
            total = entries.len(),
            "Discovery detection reached deadline — some modules were skipped (Issue #1029)"
        );
    }

    // Issue #1060: Log gated modules for observability.
    if let Some(t) = tracker {
        let gated_count = entries
            .iter()
            .filter(|e| t.is_gated(&e.module_name, MODULE_GATE_THRESHOLD))
            .count();
        if gated_count > 0 {
            tracing::info!(
                gated_count,
                total = entries.len(),
                threshold = MODULE_GATE_THRESHOLD,
                "Discovery detection: module(s) gated by low success rate (Issue #1060)"
            );
        }
    }

    // Issue #1273: Log starved modules for observability.
    let starved_count = entries.iter().filter(|e| e.starved).count();
    if starved_count > 0 {
        tracing::info!(
            starved_count,
            total = entries.len(),
            current_epoch,
            "Discovery detection: module(s) skipped by per-creature starvation cooldown \
             (Issue #1273)"
        );
    }

    crate::watchdog::beat("analysis::analyze_all → parallel discovery detection finished");

    DiscoveryModuleDetectionResults { entries }
}

/// Count how many accumulated coordinated structural candidates exceed the
/// quality gain threshold (Issue #1074).
fn count_high_quality_candidates(syn: &shared::AnalyzeSynapsesResult, threshold: f32) -> usize {
    syn.coordinated_structural_candidates
        .iter()
        .filter(|c| c.expected_creature_score_gain >= threshold)
        .count()
}

/// Merge previously-detected discovery module results into the synapse result (Issue #1004).
///
/// Iterates entries in original order and merges non-empty results sequentially.
/// Also collects per-module stats for metadata (Issue #485, #792) and records
/// pre-filtering soft failures in the tracker (Issue #1060).
///
/// ## Quality-based module skipping (Issue #1074)
///
/// After merging each module's results, checks whether the accumulated
/// candidates already contain enough high-quality entries (at least
/// [`QUALITY_SKIP_MIN_CANDIDATES`] candidates with gain above
/// [`QUALITY_SKIP_GAIN_THRESHOLD`]). If so, remaining modules are skipped
/// during the merge phase, saving post-processing time.
pub fn merge_discovery_module_results(
    syn: &mut shared::AnalyzeSynapsesResult,
    detection_results: DiscoveryModuleDetectionResults,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
    tracker: &mut ModuleOutcomeTracker,
) {
    let mut quality_skip_active = false;
    let mut modules_skipped_by_quality: usize = 0;

    for entry in detection_results.entries {
        let candidates_produced = entry.result.as_ref().map_or(0, |r| r.candidates.len());

        // Issue #1273: Record skip in the rejection breakdown when the module
        // was gated out by the per-creature starvation cooldown.
        if entry.starved {
            syn.metadata
                .rejection_breakdown
                .record(REJECTION_MODULE_STARVED);
        }

        // Record per-module stats in metadata from historical tracker (Issue #792).
        let historical = tracker.stats(&entry.module_name);
        let gated = tracker.is_gated_default(&entry.module_name);
        syn.metadata
            .discovery_module_stats
            .push(DiscoveryModuleStatsJson {
                module_name: entry.module_name.clone(),
                candidates_produced,
                attempts: historical.attempts,
                successes: historical.successes,
                success_rate: historical.success_rate(),
                soft_failures: historical.soft_failures,
                gated,
            });

        // Issue #1074: Quality-based module skipping — once enough high-quality
        // candidates have been accumulated, skip merging results from remaining
        // modules. Stats are still recorded for observability.
        if quality_skip_active {
            modules_skipped_by_quality += 1;
            let finished = format!("analysis::analyze_all → {} finished", entry.module_name);
            crate::watchdog::beat(&finished);
            continue;
        }

        if let Some(mut result) = entry.result
            && !result.candidates.is_empty()
        {
            let initial_count = result.candidates.len();

            // Issue #967: Truncate to per-module candidate budget if set.
            if entry.max_candidates > 0 && result.candidates.len() > entry.max_candidates {
                if utils::verbose_enabled() {
                    tracing::debug!(
                        module = %entry.module_name,
                        before = result.candidates.len(),
                        budget = entry.max_candidates,
                        "Truncating candidates to module budget (Issue #967)"
                    );
                }
                let truncated =
                    u32::try_from(result.candidates.len().saturating_sub(entry.max_candidates))
                        .unwrap_or(u32::MAX);
                syn.metadata
                    .rejection_breakdown
                    .record_many_u32(REJECTION_BUDGET_TRUNCATED, truncated);
                result.candidates.truncate(entry.max_candidates);
            }

            // Issue #1060, #1110: Filter coordinated candidates below the minimum
            // expected-gain floor. Gains at 1e-8 to 1e-7 are indistinguishable
            // from numerical noise and harm the network in production.
            let pre_filter_count = result.candidates.len();
            result
                .candidates
                .retain(|c| c.expected_creature_score_gain >= COORDINATED_MIN_EXPECTED_GAIN);
            let post_filter_count = result.candidates.len();
            let dropped_floor = u32::try_from(pre_filter_count.saturating_sub(post_filter_count))
                .unwrap_or(u32::MAX);
            syn.metadata
                .rejection_breakdown
                .record_many_u32(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, dropped_floor);

            // Issue #1271: After the post-discount floor, cap coordinated
            // candidates by their final-operation target neuron so a single
            // problematic target cannot consume the whole batch budget.
            let dropped_cap_usize = apply_coordinated_per_target_cap(&mut result.candidates);
            let dropped_cap = u32::try_from(dropped_cap_usize).unwrap_or(u32::MAX);
            syn.metadata
                .rejection_breakdown
                .record_many_u32(REJECTION_COORDINATED_TARGET_CAP_EXCEEDED, dropped_cap);

            // Issue #1060, #1271: Record pre-filtering soft failures for
            // candidates that were truncated (budget exceeded), filtered
            // (non-positive gain), or capped (per-final-target).
            let filtered_count = initial_count - result.candidates.len();
            if filtered_count > 0 {
                tracker.record_soft_failures(
                    &entry.module_name,
                    filtered_count,
                    SOFT_FAILURE_WEIGHT,
                );
                if utils::verbose_enabled() {
                    tracing::debug!(
                        module = %entry.module_name,
                        filtered = filtered_count,
                        truncated = initial_count - pre_filter_count,
                        negative_gain = pre_filter_count - post_filter_count,
                        weight = SOFT_FAILURE_WEIGHT,
                        "Recorded pre-filtering soft failures (Issue #1060)"
                    );
                }
            }

            if !result.candidates.is_empty() {
                if utils::verbose_enabled() {
                    tracing::debug!(
                        module = %entry.module_name,
                        detections = result.detected_count,
                        candidates = result.candidates.len(),
                        budget = entry.max_candidates,
                        "discovery module results"
                    );
                }

                super::merge_coordinated_structural_replacements(
                    syn,
                    result.candidates,
                    max_synapse_candidates,
                    diversify,
                );
            }
        }

        // Issue #1074: After merging this module's results, check if we have
        // enough high-quality candidates to skip remaining modules.
        let high_quality_count = count_high_quality_candidates(syn, QUALITY_SKIP_GAIN_THRESHOLD);
        if high_quality_count >= QUALITY_SKIP_MIN_CANDIDATES {
            quality_skip_active = true;
            tracing::debug!(
                high_quality_count,
                threshold = QUALITY_SKIP_GAIN_THRESHOLD,
                min_required = QUALITY_SKIP_MIN_CANDIDATES,
                "Quality-based module skipping activated (Issue #1074)"
            );
        }

        let finished = format!("analysis::analyze_all → {} finished", entry.module_name);
        crate::watchdog::beat(&finished);
    }

    if modules_skipped_by_quality > 0 {
        tracing::info!(
            skipped = modules_skipped_by_quality,
            "Discovery merge: skipped module(s) due to quality-based early exit (Issue #1074)"
        );
    }
}

/// Run all discovery module detection phases in parallel, then merge results
/// sequentially into the synapse result (Issue #419).
///
/// Detection closures execute concurrently via `rayon::into_par_iter()`. Results
/// are collected in original module order (rayon preserves indexed iterator order),
/// ensuring deterministic output regardless of thread scheduling.
///
/// The merge phase runs sequentially because `merge_coordinated_structural_replacements`
/// mutates the synapse result and may re-sort/truncate the combined candidate set.
#[tracing::instrument(skip_all, fields(module_count = modules.len()))]
pub fn run_discovery_modules_parallel(
    syn: &mut shared::AnalyzeSynapsesResult,
    modules: Vec<DiscoveryModuleSpec>,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
    tracker: &mut ModuleOutcomeTracker,
) {
    let detection_results = detect_discovery_modules_parallel(modules, None, Some(tracker));
    merge_discovery_module_results(
        syn,
        detection_results,
        max_synapse_candidates,
        diversify,
        tracker,
    );
}

#[cfg(test)]
#[path = "discovery_dispatch_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "discovery_dispatch_parallel_tests.rs"]
mod parallel_tests;
