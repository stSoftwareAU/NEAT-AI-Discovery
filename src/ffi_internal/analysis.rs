//! Rust-native counterparts of the analysis entry points exposed over FFI:
//! `rank_focus_neurons` and `analyze_parallel` (`src/ffi/analysis.rs`), plus
//! `get_calibration_summary` (`src/ffi/utilities.rs`), call into the
//! `*_internal` functions here rather than inlining the logic, so the same
//! code path is reachable from Rust integration tests without crossing the C
//! boundary.
//!
//! **Contract** — JSON in, JSON out. A caller-input failure (unparseable JSON,
//! a creature rejected by `validate_creature`, an unreadable Parquet file) is
//! returned as a `success: false` payload carrying `error` and `error_kind`,
//! never as a Rust `Err` handed back to the caller; `Err` is reserved for a
//! failure to serialise the response itself. The C-boundary concerns —
//! null/invalid-UTF-8 pointer rejection, `catch_unwind` panic containment and
//! `CString` conversion — stay one layer up in `src/ffi/`.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use anyhow::Result;

use crate::analysis::utils::build_deadline;
use crate::ffi_types::*;
use crate::{analysis, focus};

/// Runs parallel discovery analysis over recorded data.
///
/// Takes JSON input and returns JSON output for easy integration with TypeScript/DenoJS.
pub fn analyze_parallel_internal(input_json: &str) -> Result<String> {
    let input: AnalyzeParallelInput = match serde_json::from_str::<AnalyzeParallelInput>(input_json)
    {
        Ok(value) => value,
        Err(e) => {
            let typed = DiscoveryError::InvalidInput {
                detail: format!("Failed to parse input JSON: {e}"),
            };
            let kind = typed.error_kind();
            let output = AnalyzeParallelOutput {
                success: false,
                schema_version: SCHEMA_VERSION.to_string(),
                helpful_synapses: None,
                harmful_synapses: None,
                synapse_diagnostics: None,
                synapse_gpu_used: None,
                synapse_metadata: None,
                helpful_neurons: None,
                synapse_weight_updates: None,
                coordinated_structural_candidates: None,
                candidate_clusters: None,
                neuron_diagnostics: None,
                neuron_gpu_used: None,
                neuron_metadata: None,
                neuron_fingerprints: None,
                fingerprint_cache_hits: None,
                fingerprint_cache_misses: None,
                module_outcome_tracker: None,
                memory_budget_exceeded: None,
                cancelled: None,
                memory_pressure_cancelled: None,
                environmentally_disabled: None,
                zero_candidate_summary: None,
                error: Some(typed.to_string()),
                error_kind: Some(kind),
                retryable: Some(kind.is_retryable()),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Issue #1184: Reject corrupt creatures carrying recurrent or
    // unresolved synapses before they reach the analysis pipeline. The
    // downstream `target_analysis` filter silently drops back-edges, so
    // failing fast here surfaces upstream corruption (e.g. NEAT-AI
    // `loadFrom` strip warnings) instead of letting it taint discovery.
    // Issue #1867: the same gate bounds the creature's input-neuron count,
    // which sizes per-input allocations downstream.
    if let Err(typed) = validate_creature(&input.creature) {
        let kind = typed.error_kind();
        let output = AnalyzeParallelOutput {
            success: false,
            schema_version: SCHEMA_VERSION.to_string(),
            helpful_synapses: None,
            harmful_synapses: None,
            synapse_diagnostics: None,
            synapse_gpu_used: None,
            synapse_metadata: None,
            helpful_neurons: None,
            synapse_weight_updates: None,
            coordinated_structural_candidates: None,
            candidate_clusters: None,
            neuron_diagnostics: None,
            neuron_gpu_used: None,
            neuron_metadata: None,
            neuron_fingerprints: None,
            fingerprint_cache_hits: None,
            fingerprint_cache_misses: None,
            module_outcome_tracker: None,
            memory_budget_exceeded: None,
            cancelled: None,
            memory_pressure_cancelled: None,
            environmentally_disabled: None,
            zero_candidate_summary: None,
            error: Some(typed.to_string()),
            error_kind: Some(kind),
            retryable: Some(kind.is_retryable()),
        };
        return Ok(serde_json::to_string(&output)?);
    }

    let combined_input = build_analyze_all_input_from_parallel(input);

    // Issue #1048 / #1077: Track that an analysis is active so the host
    // knows not to delete the parquet temp directory until we finish.
    // Uses an RAII guard so the counter is decremented even if the
    // analysis panics (e.g., rayon thread panic), preventing the host
    // from waiting forever ("discovery locked up").
    let _active_guard = crate::cancellation::AnalysisActiveGuard::new();

    let analysis_result = analysis::analyze_all(&combined_input);

    match analysis_result {
        Ok(result) => {
            // Issue #1421: classify the pass before destructuring so an
            // environmentally-disabled pass (memory/GPU gated) is surfaced as a
            // distinct, countable category rather than conflated with genuine
            // search exhaustion ("0 candidates").
            let outcome = analysis::AnalysisOutcome::from_result(&result);
            let environmentally_disabled = outcome.disable_reason();
            if let Some(reason) = environmentally_disabled {
                tracing::warn!(
                    reason = reason.as_str(),
                    "Issue #1421: discovery pass environmentally disabled — \
                     creature not evaluated; excluded from drought/cooldown/starvation accounting"
                );
            }

            let synapse = result.synapse;
            let neuron = result.neuron;
            let synapse_weight_updates = synapse.as_ref().and_then(|s| {
                if s.synapse_weight_updates.is_empty() {
                    None
                } else {
                    Some(s.synapse_weight_updates.clone())
                }
            });

            let coordinated_structural_candidates = synapse.as_ref().and_then(|s| {
                if s.coordinated_structural_candidates.is_empty() {
                    None
                } else {
                    Some(s.coordinated_structural_candidates.clone())
                }
            });

            let candidate_clusters = synapse.as_ref().and_then(|s| {
                if s.candidate_clusters.is_empty() {
                    None
                } else {
                    Some(s.candidate_clusters.clone())
                }
            });

            let memory_budget_exceeded = if result.memory_budget_exceeded {
                Some(true)
            } else {
                None
            };

            // Issue #1447: cross-stack failure-cache handshake. Count how many
            // returned candidates match the per-creature failure cache (the set
            // NEAT-AI's `CandidateFiltering.ts` will drop) and decide whether
            // novelty escalation should fire so the host can bypass that filter
            // for the top-K candidates. Computed at the FFI boundary because
            // matching is against the same failure-cache identities NEAT-AI
            // filters on.
            let failure_cache = combined_input.failure_cache.as_deref().unwrap_or(&[]);
            let syn_identities = synapse
                .as_ref()
                .map(synapse_candidate_identities)
                .unwrap_or_default();
            let neu_identities = neuron
                .as_ref()
                .map(neuron_candidate_identities)
                .unwrap_or_default();
            let syn_suppressed =
                analysis::failure_cache_handshake::count_suppressed(&syn_identities, failure_cache);
            let neu_suppressed =
                analysis::failure_cache_handshake::count_suppressed(&neu_identities, failure_cache);
            // Creature-level escalation across all returned candidates; the
            // rolling success rate is creature-scoped (identical on both
            // surfaces), defaulting to neutral when neither surface is present.
            let rolling_success_rate = synapse
                .as_ref()
                .map(|s| s.metadata.rolling_success_rate)
                .or_else(|| neuron.as_ref().map(|n| n.metadata.rolling_success_rate))
                .unwrap_or(1.0);
            let mut all_identities = syn_identities;
            all_identities.extend(neu_identities);
            let handshake = analysis::failure_cache_handshake::evaluate(
                &all_identities,
                failure_cache,
                rolling_success_rate,
                crate::config::low_success_rate_threshold(),
                crate::config::novelty_suppression_ratio(),
            );
            // Issue #1739: gate the generation-widening hint by the
            // candidate-starvation classification. The #1737 diagnosis showed
            // the large converged production profile is proposal-rich but
            // over-rejected (candidates reach the accept gate and are rejected
            // there), so widening
            // generation cannot lift the accepted rate and would only add
            // noise. Escalation is therefore allowed to fire only when the run
            // is genuinely candidate-starved (few proposals ever reach the
            // gate), computed from the same RejectionBreakdown the diagnosis
            // reads. This changes nothing on the accept path (#1623): it only
            // suppresses a widening hint that would otherwise waste host effort.
            // Issue #1781: pass-level drops (currently the whole-pass
            // fingerprint skip) never reach either surface's metadata, and
            // Issue #1800: failure-cache suppression is counted only at this
            // boundary. Both are folded in before the starvation classifier
            // reads the breakdown.
            let combined_breakdown = starvation_classifier_breakdown(
                synapse.as_ref().map(|s| &s.metadata.rejection_breakdown),
                neuron.as_ref().map(|n| &n.metadata.rejection_breakdown),
                &result.pass_rejection_breakdown,
                syn_suppressed.saturating_add(neu_suppressed),
            );
            let surviving_candidates = synapse
                .as_ref()
                .map_or(0, |s| s.metadata.candidates_returned)
                .saturating_add(
                    neuron
                        .as_ref()
                        .map_or(0, |n| n.metadata.candidates_returned),
                );
            let starvation_signals = analysis::candidate_starvation::signals_from_breakdown(
                &combined_breakdown,
                u32::try_from(surviving_candidates).unwrap_or(u32::MAX),
            );
            let starvation_class = analysis::candidate_starvation::classify(
                &starvation_signals,
                &analysis::candidate_starvation::StarvationConfig::default(),
            );
            let novelty_escalation_active = analysis::candidate_starvation::gate_escalation(
                handshake.novelty_escalation_active,
                starvation_class,
            );

            // Issue #1446: when a pass produces no candidates of any kind,
            // attach a consolidated `zeroCandidateSummary` so operators can see
            // the dominant rejection reason without opening JSON sidecars.
            let has_candidates = synapse.as_ref().is_some_and(|s| {
                !s.helpful_synapses.is_empty()
                    || !s.harmful_synapses.is_empty()
                    || !s.synapse_weight_updates.is_empty()
                    || !s.coordinated_structural_candidates.is_empty()
            }) || neuron
                .as_ref()
                .is_some_and(|n| !n.helpful_neurons.is_empty());

            let zero_candidate_summary = if has_candidates {
                None
            } else {
                let environmental_gates = EnvironmentalGatesJson {
                    memory_budget_exceeded: result.memory_budget_exceeded,
                    memory_pressure_cancelled: result.memory_pressure_cancelled,
                    cancelled: result.cancelled,
                    environmentally_disabled,
                };
                let summary = build_zero_candidate_summary(
                    synapse.as_ref().map(|s| &s.metadata),
                    neuron.as_ref().map(|n| &n.metadata),
                    &result.pass_rejection_breakdown,
                    environmental_gates,
                    // Issue #1925: the classification and its counts were
                    // already computed above to gate novelty escalation; a
                    // barren pass now reports them instead of discarding them.
                    starvation_signals,
                    starvation_class,
                );
                // Emit a single structured WARN naming the dominant reason for
                // genuinely-empty passes. Environmentally-gated passes already
                // warned above (Issue #1421), so we don't double-log them.
                if environmentally_disabled.is_none() {
                    tracing::warn!(
                        dominant_rejection_reason = summary
                            .dominant_rejection_reason
                            .as_deref()
                            .unwrap_or("unknown"),
                        drought_consecutive_failures = summary
                            .drought_diagnostic
                            .as_ref()
                            .map(|d| d.consecutive_failures),
                        // Issue #1925: which failure mode, and how many
                        // proposals were ever formed — the two facts that say
                        // whether the pass was starved or over-rejected.
                        starvation_class = summary.starvation_class,
                        proposals_formed = summary.generation_signals.proposals_formed,
                        upstream_rejections = summary.generation_signals.upstream_rejections,
                        "Issue #1446: discovery pass produced 0 candidates — \
                         zeroCandidateSummary attached"
                    );
                }
                Some(summary)
            };

            let output = AnalyzeParallelOutput {
                success: true,
                schema_version: SCHEMA_VERSION.to_string(),
                helpful_synapses: synapse.as_ref().map(|s| s.helpful_synapses.clone()),
                harmful_synapses: synapse.as_ref().map(|s| s.harmful_synapses.clone()),
                synapse_diagnostics: synapse
                    .as_ref()
                    .and_then(|s| synapse_diagnostics_json(s.no_candidate_reasons.as_slice())),
                synapse_gpu_used: synapse.as_ref().map(|s| s.gpu_used),
                synapse_metadata: synapse.as_ref().map(|s| SynapseAnalysisMetadataJson {
                    target_value_available: s.metadata.target_value_available,
                    saturation_aware_simulation_used: s.metadata.saturation_aware_simulation_used,
                    candidates_found: s.metadata.candidates_found,
                    candidates_returned: s.metadata.candidates_returned,
                    timed_out: s.metadata.timed_out,
                    completed_focus_neurons: s.metadata.completed_focus_neurons,
                    total_focus_neurons: s.metadata.total_focus_neurons,
                    // Issue #1409: explicit starvation flag for the calling host layer.
                    starved: s.metadata.timed_out
                        && s.metadata.completed_focus_neurons < s.metadata.total_focus_neurons,
                    input_index_min_seen_with_records: s.metadata.input_index_min_seen_with_records,
                    input_index_max_seen_with_records: s.metadata.input_index_max_seen_with_records,
                    timing: s.metadata.timing.as_ref().map(timing_to_json),
                    gpu_info: s.metadata.gpu_info.as_ref().map(gpu_info_to_json),
                    discovery_module_stats: s.metadata.discovery_module_stats.clone(),
                    mcmc_diagnostics: s.metadata.mcmc_diagnostics.as_ref().map(mcmc_to_json),
                    rejection_breakdown: breakdown_with_failure_cache(
                        &s.metadata.rejection_breakdown,
                        syn_suppressed,
                    ),
                    top_level_summary: s.metadata.top_level_summary.clone(),
                    calibration_corrections: s.metadata.calibration_corrections.clone(),
                    discovery_mode: s.metadata.discovery_mode,
                    rolling_success_rate: s.metadata.rolling_success_rate,
                    failure_cache_suppressed_count: syn_suppressed,
                    novelty_escalation_active,
                    drought_diagnostic: s.metadata.drought_diagnostic.clone(),
                    creature_drought_alarm: s.metadata.creature_drought_alarm.clone(),
                    insufficient_recording: s.metadata.insufficient_recording.clone(),
                }),
                helpful_neurons: neuron.as_ref().map(|n| n.helpful_neurons.clone()),
                synapse_weight_updates,
                coordinated_structural_candidates,
                candidate_clusters,
                neuron_diagnostics: neuron
                    .as_ref()
                    .and_then(|n| neuron_diagnostics_json(n.no_candidate_reasons.as_slice())),
                neuron_gpu_used: neuron.as_ref().map(|n| n.gpu_used),
                neuron_metadata: neuron.as_ref().map(|n| NeuronAnalysisMetadataJson {
                    candidates_found: n.metadata.candidates_found,
                    candidates_returned: n.metadata.candidates_returned,
                    timed_out: n.metadata.timed_out,
                    completed_focus_neurons: n.metadata.completed_focus_neurons,
                    total_focus_neurons: n.metadata.total_focus_neurons,
                    // Issue #1409: explicit starvation flag for the calling host layer.
                    starved: n.metadata.timed_out
                        && n.metadata.completed_focus_neurons < n.metadata.total_focus_neurons,
                    timing: n.metadata.timing.as_ref().map(timing_to_json),
                    gpu_info: n.metadata.gpu_info.as_ref().map(gpu_info_to_json),
                    rejection_breakdown: breakdown_with_failure_cache(
                        &n.metadata.rejection_breakdown,
                        neu_suppressed,
                    ),
                    top_level_summary: n.metadata.top_level_summary.clone(),
                    calibration_corrections: n.metadata.calibration_corrections.clone(),
                    discovery_mode: n.metadata.discovery_mode,
                    rolling_success_rate: n.metadata.rolling_success_rate,
                    failure_cache_suppressed_count: neu_suppressed,
                    novelty_escalation_active,
                    drought_diagnostic: n.metadata.drought_diagnostic.clone(),
                    creature_drought_alarm: n.metadata.creature_drought_alarm.clone(),
                    insufficient_recording: n.metadata.insufficient_recording.clone(),
                }),
                neuron_fingerprints: result.neuron_fingerprints,
                fingerprint_cache_hits: if result.fingerprint_cache_hits > 0 {
                    Some(result.fingerprint_cache_hits)
                } else {
                    None
                },
                fingerprint_cache_misses: if result.fingerprint_cache_misses > 0 {
                    Some(result.fingerprint_cache_misses)
                } else {
                    None
                },
                module_outcome_tracker: if result.module_outcome_tracker.is_empty() {
                    None
                } else {
                    Some(result.module_outcome_tracker)
                },
                memory_budget_exceeded,
                cancelled: if result.cancelled { Some(true) } else { None },
                memory_pressure_cancelled: if result.memory_pressure_cancelled {
                    Some(true)
                } else {
                    None
                },
                environmentally_disabled,
                zero_candidate_summary,
                error: None,
                error_kind: None,
                retryable: None,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            // Issue #1421: the GPU-unavailable early return is an environmental
            // gate, not search exhaustion — surface it as such so the host
            // excludes it from drought / cooldown / starvation accounting.
            let environmentally_disabled = match e.downcast_ref::<DiscoveryError>() {
                Some(DiscoveryError::GpuUnavailable { .. }) => {
                    let reason = analysis::AnalysisOutcome::gpu_unavailable()
                        .disable_reason()
                        .expect("gpu_unavailable carries a reason");
                    tracing::warn!(
                        reason = reason.as_str(),
                        "Issue #1421: discovery pass environmentally disabled — \
                         creature not evaluated; excluded from drought/cooldown/starvation accounting"
                    );
                    Some(reason)
                }
                _ => None,
            };
            // Issue #1932: one place builds the failure shape, so a wedged GPU
            // reaches the host as `errorKind: "gpu_wedged"`, `retryable: false`.
            let output = AnalyzeParallelOutput {
                environmentally_disabled,
                ..AnalyzeParallelOutput::failure(&e)
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
}

/// Build failure-cache match identities for the returned synapse candidates
/// (Issue #1447). Covers add-synapse candidates (keyed by target neuron) and
/// coordinated-structural candidates (target-agnostic — they have no single
/// target neuron).
fn synapse_candidate_identities(
    syn: &analysis::shared::AnalyzeSynapsesResult,
) -> Vec<analysis::failure_cache_handshake::CandidateIdentity> {
    use analysis::failure_cache_handshake::{
        CHANGE_TYPE_ADD_SYNAPSES, CHANGE_TYPE_COORDINATED_STRUCTURAL, CandidateIdentity,
    };
    let mut ids = Vec::with_capacity(
        syn.helpful_synapses.len() + syn.coordinated_structural_candidates.len(),
    );
    for c in &syn.helpful_synapses {
        ids.push(CandidateIdentity::new(
            CHANGE_TYPE_ADD_SYNAPSES,
            Some(c.to_neuron_uuid.clone()),
            None,
        ));
    }
    for _ in &syn.coordinated_structural_candidates {
        ids.push(CandidateIdentity::new(
            CHANGE_TYPE_COORDINATED_STRUCTURAL,
            None,
            None,
        ));
    }
    ids
}

/// Build failure-cache match identities for the returned neuron candidates
/// (Issue #1447). Add-neuron candidates are keyed by target neuron and squash.
fn neuron_candidate_identities(
    neu: &analysis::shared::AnalyzeNeuronsResult,
) -> Vec<analysis::failure_cache_handshake::CandidateIdentity> {
    use analysis::failure_cache_handshake::{CHANGE_TYPE_ADD_NEURONS, CandidateIdentity};
    neu.helpful_neurons
        .iter()
        .map(|c| {
            CandidateIdentity::new(
                CHANGE_TYPE_ADD_NEURONS,
                Some(c.target_neuron_uuid.clone()),
                Some(c.squash.clone()),
            )
        })
        .collect()
}

/// Assemble the rejection breakdown the starvation classifier reads for a pass
/// (Issue #1800).
///
/// Folds, in one place: both surfaces' metadata breakdowns, the pass-level
/// breakdown (#1781), and the cross-stack failure-cache suppression count
/// (#1447, `syn_suppressed + neu_suppressed`) under
/// `REJECTION_DUPLICATE_OF_FAILURE_CACHE`. That suppression is the very
/// evidence of starvation, so it must be present *before*
/// `candidate_starvation::signals_from_breakdown` reads the breakdown —
/// otherwise the classifier can never recommend bypassing it.
///
/// The count lands here exactly once: the surfaced per-surface wire maps are
/// built separately by `breakdown_with_failure_cache` and never feed back
/// into this breakdown.
#[must_use]
pub fn starvation_classifier_breakdown(
    synapse_breakdown: Option<&analysis::diagnostics::RejectionBreakdown>,
    neuron_breakdown: Option<&analysis::diagnostics::RejectionBreakdown>,
    pass_breakdown: &analysis::diagnostics::RejectionBreakdown,
    failure_cache_suppressed: usize,
) -> analysis::diagnostics::RejectionBreakdown {
    let mut combined = synapse_breakdown.cloned().unwrap_or_default();
    if let Some(neuron) = neuron_breakdown {
        combined.merge_from(neuron.counts());
    }
    combined.merge_from(pass_breakdown.counts());
    // `record_many` is a no-op at zero, so an unsuppressed pass keeps the
    // reason absent rather than present-and-zero.
    combined.record_many(
        analysis::diagnostics::rejection_reasons::REJECTION_DUPLICATE_OF_FAILURE_CACHE,
        u32::try_from(failure_cache_suppressed).unwrap_or(u32::MAX),
    );
    combined
}

/// Clone a rejection breakdown into its wire map, wiring the previously-dead
/// `REJECTION_DUPLICATE_OF_FAILURE_CACHE` reason with the cross-stack
/// failure-cache suppression count (Issue #1447). Keeps duplicate suppression
/// visible in the surfaced rejection stats.
fn breakdown_with_failure_cache(
    breakdown: &analysis::diagnostics::RejectionBreakdown,
    suppressed: usize,
) -> std::collections::HashMap<String, u32> {
    let mut map = breakdown.counts().clone();
    if suppressed > 0 {
        let count = u32::try_from(suppressed).unwrap_or(u32::MAX);
        *map.entry(
            analysis::diagnostics::rejection_reasons::REJECTION_DUPLICATE_OF_FAILURE_CACHE
                .to_string(),
        )
        .or_insert(0) += count;
    }
    map
}

pub(crate) fn build_analyze_all_input_from_parallel(
    input: AnalyzeParallelInput,
) -> AnalyzeAllInput {
    AnalyzeAllInput {
        parquet_file: input.parquet_file,
        creature: input.creature,
        focus_neurons: input.focus_neurons,
        max_synapse_candidates: input.max_synapse_candidates,
        max_neuron_candidates: input.max_neuron_candidates,
        analysis_deadline_ms: input.analysis_deadline_ms,
        // Issue #1937: honour the caller's phase gating instead of forcing
        // both phases on. `None` still means "run it" downstream.
        include_synapse_analysis: input.include_synapse_analysis,
        include_neuron_analysis: input.include_neuron_analysis,
        random_seed: input.random_seed,
        previous_neuron_fingerprints: input.previous_neuron_fingerprints,
        module_outcome_tracker: input.module_outcome_tracker,
        temperature: input.temperature,
        max_analysis_memory_mb: input.max_analysis_memory_mb,
        max_discovery_wall_clock_minutes: input.max_discovery_wall_clock_minutes,
        failure_cache: input.failure_cache,
        discovery_outcome_log: input.discovery_outcome_log,
        // Issue #1317: forward the cost-function identity into analysis.
        cost_name: input.cost_name,
    }
}

/// Ranks focus neurons for the next discovery pass.
///
/// Takes JSON input and returns JSON output for easy integration with TypeScript/DenoJS.
pub fn rank_focus_neurons_internal(input_json: &str) -> Result<String> {
    let input: RankFocusNeuronsInput = match serde_json::from_str(input_json) {
        Ok(value) => value,
        Err(e) => {
            let typed = DiscoveryError::InvalidInput {
                detail: format!("Failed to parse input JSON: {e}"),
            };
            let kind = typed.error_kind();
            let output = RankFocusNeuronsOutput {
                success: false,
                schema_version: SCHEMA_VERSION.to_string(),
                neurons: None,
                focus_selection: None,
                removal_candidates: None,
                constant_neuron_removals: None,
                max_output_error: None,
                processed_neurons: None,
                total_neurons: None,
                duration_ms: None,
                rejection_breakdown: None,
                loading_mode: None,
                lazy_reason: None,
                // Parse failed before the caller budget was available (Issue #4138).
                budget_mb: None,
                projected_mb: None,
                error: Some(typed.to_string()),
                error_kind: Some(kind),
                retryable: Some(kind.is_retryable()),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Issue #1184: Reject corrupt creatures carrying recurrent or
    // unresolved synapses before ranking touches the topology.
    // Issue #1867: the same gate bounds the creature's input-neuron count,
    // which sizes per-input allocations downstream.
    if let Err(typed) = validate_creature(&input.creature) {
        let kind = typed.error_kind();
        let output = RankFocusNeuronsOutput {
            success: false,
            schema_version: SCHEMA_VERSION.to_string(),
            neurons: None,
            focus_selection: None,
            removal_candidates: None,
            constant_neuron_removals: None,
            max_output_error: None,
            processed_neurons: None,
            total_neurons: None,
            duration_ms: None,
            rejection_breakdown: None,
            loading_mode: None,
            lazy_reason: None,
            budget_mb: input.max_analysis_memory_mb,
            projected_mb: None,
            error: Some(typed.to_string()),
            error_kind: Some(kind),
            retryable: Some(kind.is_retryable()),
        };
        return Ok(serde_json::to_string(&output)?);
    }

    // Issue #1766: focus SELECTION is now structure-only. It never opens the
    // discovery parquet — impact is derived from creature topology (path weights;
    // output neurons seed at 1.0) and the focus set is a weighted-random draw over
    // that impact. Coupling focus choice to multi-GB record I/O was the root cause
    // of the ~2h focus-stall incident: the analysis budget was burned reading records
    // *before* any focus neuron was picked. Parquet stays for the analysis phase
    // that runs *after* the focus set is chosen (see `analyze_parallel`), and
    // record-derived removal-candidate detection moves off this path (companion
    // issue) — so `removalCandidates` / `constantNeuronRemovals` are omitted here.
    let start = std::time::Instant::now();

    let focus_set_size = input.focus_set_size.unwrap_or(DEFAULT_FOCUS_SET_SIZE);
    // The monotonic per-creature cursor (falling back to the drought epoch
    // counter, then 0) seeds the weighted-random draw so successive passes
    // explore fresh neurons while still landing mostly on high impact.
    let seed = input
        .focus_selection_cursor
        .or(input.epochs_since_last_accepted_candidate)
        .unwrap_or(0);

    let structural =
        focus::select_focus_by_structural_impact(&input.creature, focus_set_size, seed);
    let focus_selection = structural.selection;

    // WARN when the raw structural-impact weight is pathologically single-target
    // so the collapse the weighted draw spread stays visible to operators.
    if focus_selection.raw_weight_concentration_ratio > focus::CONCENTRATION_WARN_THRESHOLD {
        tracing::warn!(
            raw_weight_concentration_ratio = focus_selection.raw_weight_concentration_ratio,
            effective_weight_concentration_ratio = focus_selection.weight_concentration_ratio,
            eligible_pool_size = focus_selection.eligible_pool_size,
            focus_set_size,
            "focus_selection_weight_concentration_high: one neuron dominated the \
             structural-impact weights; the weighted-random draw spread the focus \
             set while keeping the high-impact majority"
        );
    }

    let focus_selection_json = FocusSelectionJson {
        selected: focus_selection.selected,
        raw_weight_concentration_ratio: focus_selection.raw_weight_concentration_ratio,
        weight_concentration_ratio: focus_selection.weight_concentration_ratio,
        exploitation_count: focus_selection.exploitation_count,
        exploration_count: focus_selection.exploration_count,
        exploration_cursor: focus_selection.exploration_cursor,
        eligible_pool_size: focus_selection.eligible_pool_size,
        cumulative_coverage: focus_selection.cumulative_coverage,
        drought_active: focus_selection.drought_active,
        pool_size: focus_selection.pool_size,
    };

    // Structure-only ranked pool (impact-ordered). `totalError` / `meanActivation`
    // are record-derived and intentionally 0.0 on the focus path; `impact` and
    // `weightedScore` both carry the structural impact used as the draw weight.
    let total_neurons = structural.ranked.len();
    let mut ranked = structural.ranked;
    if let Some(limit) = input.max_results
        && ranked.len() > limit
    {
        ranked.truncate(limit);
    }
    let neurons: Vec<RankedNeuronJson> = ranked
        .into_iter()
        .map(|c| RankedNeuronJson {
            neuron_uuid: c.neuron_uuid,
            total_error: 0.0,
            impact: c.weight,
            mean_activation: 0.0,
            activation_weighted_impact: 0.0,
            weighted_score: c.weight,
        })
        .collect();
    let processed_neurons = neurons.len();

    // Issue #1767: removal triage runs at focus time on the **near-opposite** axis
    // to focus — focus draws HIGH structural impact, removal flags hidden neurons
    // whose LOW structural contribution is outweighed by the complexity savings of
    // pruning them. It is structure-only: no discovery parquet is opened, so it
    // never reintroduces the focus-time parquet dependency #1766 removed. The
    // activation-weighted gates that need records stay in the analysis phase.
    // `costOfGrowth` defaults to `DEFAULT_COST_OF_GROWTH` (NEAT-AI's Score.ts
    // value). Issue #1783: the single criterion validates it, so a non-finite
    // or non-positive value falls back to that default with a WARN instead of
    // being taken raw — Issue #1807 pins that on this entry point.
    //
    // Issue #1923: the activation-weighted gate the structural triage defers is
    // then resolved here, from one two-column streaming pass over the discovery
    // parquet that materialises no records. `remove-low-impact` previously
    // ranked on `meanActivation: 0.0` — a dead field — because nothing
    // downstream ever revisited the deferral. The pass runs strictly *after*
    // focus selection (already complete above) and is bounded by the shared
    // discovery deadline, so it cannot recreate the #1766 focus stall.
    // Issue #4139: a sub-minimum deadline skips parquet-backed removal triage
    // rather than inflating the remainder to 10 minutes. Focus selection itself
    // is already complete above and is structure-only.
    let skip_removal = crate::analysis::utils::deadline_too_short_to_analyse(
        input.analysis_deadline_ms,
        "rank_focus_neurons",
    );
    let (rejection_breakdown, removal_candidates) = if skip_removal {
        (std::collections::HashMap::new(), Vec::new())
    } else {
        let removal_outcome = focus::identify_removal_candidates_for_focus(
            &input.creature,
            input.cost_of_growth,
            &input.parquet_file,
            build_deadline(input.analysis_deadline_ms),
        );
        let breakdown = removal_outcome.rejection_breakdown();
        let candidates: Vec<RemovalCandidateJson> = removal_outcome
            .candidates
            .into_iter()
            .map(|c| RemovalCandidateJson {
                neuron_uuid: c.neuron_uuid,
                total_error: c.total_error,
                impact: c.impact,
                mean_activation: c.mean_activation,
                activation_weighted_impact: c.activation_weighted_impact,
                incoming_synapses: c.incoming_synapses,
                outgoing_synapses: c.outgoing_synapses,
                removal_savings: c.removal_savings,
                expected_error_reduction: c.expected_error_reduction,
                reason: c.reason,
            })
            .collect();
        (breakdown, candidates)
    };

    let (error_kind, retryable) = no_error_fields();
    let output = RankFocusNeuronsOutput {
        success: true,
        schema_version: SCHEMA_VERSION.to_string(),
        neurons: Some(neurons),
        focus_selection: Some(focus_selection_json),
        // Issue #1767: structure-only removal triage (near-opposite of focus).
        removal_candidates: if removal_candidates.is_empty() {
            None
        } else {
            Some(removal_candidates)
        },
        // Constant-neuron removal folds recorded activation variance into biases,
        // so it needs records — it stays in the analysis phase, off the focus path.
        constant_neuron_removals: None,
        max_output_error: None,
        processed_neurons: Some(processed_neurons),
        total_neurons: Some(total_neurons),
        duration_ms: Some(start.elapsed().as_millis().min(u64::MAX as u128) as u64),
        // Issue #1142: surface removal candidates dropped by the noise-floor gate
        // so operators can root-cause "no removal candidates" without re-running.
        rejection_breakdown: if rejection_breakdown.is_empty() {
            None
        } else {
            Some(rejection_breakdown)
        },
        // No parquet is loaded on the focus path (Issue #1766), so the loading-mode
        // observability fields are omitted rather than reporting a decode that
        // never happened. `budget_mb` still reports the caller-supplied analysis
        // budget so the four FFI sites no longer hard-code `None` when a value
        // was provided (Issue #4138).
        loading_mode: None,
        lazy_reason: None,
        budget_mb: input.max_analysis_memory_mb,
        projected_mb: None,
        error: None,
        error_kind,
        retryable,
    };
    Ok(serde_json::to_string(&output)?)
}

/// Returns a calibration summary from discovery history (Issue #605).
///
/// Takes JSON input containing a serialised `DiscoveryHistory` and returns
/// calibration metrics (MAE, bias, calibration factor) per module/candidate type.
pub fn get_calibration_summary_internal(input_json: &str) -> Result<String> {
    let input: CalibrationSummaryInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let typed = DiscoveryError::InvalidInput {
                detail: format!("Failed to parse input JSON: {e}"),
            };
            let kind = typed.error_kind();
            let output = CalibrationSummaryOutput {
                success: false,
                calibration_summary: vec![],
                error: Some(typed.to_string()),
                error_kind: Some(kind),
                retryable: Some(kind.is_retryable()),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    let history: crate::discovery_history::DiscoveryHistory =
        match serde_json::from_str(&input.discovery_history) {
            Ok(h) => h,
            Err(e) => {
                let typed = DiscoveryError::InvalidInput {
                    detail: format!("Failed to parse discovery history: {e}"),
                };
                let kind = typed.error_kind();
                let output = CalibrationSummaryOutput {
                    success: false,
                    calibration_summary: vec![],
                    error: Some(typed.to_string()),
                    error_kind: Some(kind),
                    retryable: Some(kind.is_retryable()),
                };
                return Ok(serde_json::to_string(&output)?);
            }
        };

    let summary = history.calibration_summary();
    let (error_kind, retryable) = no_error_fields();
    let output = CalibrationSummaryOutput {
        success: true,
        calibration_summary: summary,
        error: None,
        error_kind,
        retryable,
    };
    Ok(serde_json::to_string(&output)?)
}

#[cfg(test)]
mod failure_cache_handshake_wiring_tests {
    //! FFI-layer wiring for the cross-stack failure-cache handshake (Issue #1447).

    use super::*;
    use crate::analysis::diagnostics::RejectionBreakdown;
    use crate::analysis::diagnostics::rejection_reasons::{
        REJECTION_BELOW_THRESHOLD, REJECTION_DUPLICATE_OF_FAILURE_CACHE,
    };
    use crate::analysis::failure_cache_handshake::{
        CHANGE_TYPE_ADD_NEURONS, CHANGE_TYPE_ADD_SYNAPSES, CHANGE_TYPE_COORDINATED_STRUCTURAL,
    };
    use crate::analysis::shared::{AnalyzeNeuronsResult, AnalyzeSynapsesResult};
    use crate::ffi_types::{
        CandidateNeuronJson, CandidateSynapseJson, CoordinatedStructuralCandidateJson,
    };

    fn synapse(to: &str) -> CandidateSynapseJson {
        CandidateSynapseJson {
            from_neuron_uuid: "src".to_string(),
            to_neuron_uuid: to.to_string(),
            from_neuron_index: None,
            to_neuron_index: None,
            weight: 0.1,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.01,
            expected_creature_score_gain: 0.01,
            improved_count: 8,
            total_count: 10,
            improvement_magnitude_ratio: None,
            target_neuron_stats: None,
            outlier_reduction_info: None,
            prediction_confidence: 0.8,
            expected_score_gain_confidence_interval: [0.0, 0.02],
            comment: None,
            variant_key: None,
        }
    }

    fn neuron(target: &str, squash: &str) -> CandidateNeuronJson {
        CandidateNeuronJson {
            source_neuron_uuid: "src".to_string(),
            target_neuron_uuid: target.to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: 0.1,
            outgoing_weight: 0.1,
            squash: squash.to_string(),
            bias: 0.0,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.01,
            expected_creature_score_gain: 0.01,
            improved_count: 8,
            total_count: 10,
            improvement_magnitude_ratio: None,
            target_neuron_stats: None,
            prediction_confidence: 0.8,
            expected_score_gain_confidence_interval: [0.0, 0.02],
            target_saturation_factor: None,
            variant_key: None,
        }
    }

    fn coordinated() -> CoordinatedStructuralCandidateJson {
        CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: Vec::new(),
            expected_creature_score_gain: 0.01,
            comment: None,
        }
    }

    fn synapse_result(
        helpful: Vec<CandidateSynapseJson>,
        coordinated_cands: Vec<CoordinatedStructuralCandidateJson>,
    ) -> AnalyzeSynapsesResult {
        AnalyzeSynapsesResult {
            helpful_synapses: helpful,
            harmful_synapses: Vec::new(),
            synapse_weight_updates: Vec::new(),
            coordinated_structural_candidates: coordinated_cands,
            candidate_clusters: Vec::new(),
            gpu_used: false,
            no_candidate_reasons: Vec::new(),
            metadata: crate::analysis::shared::SynapseAnalysisMetadata::default(),
        }
    }

    fn neuron_result(helpful: Vec<CandidateNeuronJson>) -> AnalyzeNeuronsResult {
        AnalyzeNeuronsResult {
            helpful_neurons: helpful,
            gpu_used: false,
            no_candidate_reasons: Vec::new(),
            metadata: crate::analysis::shared::NeuronAnalysisMetadata::default(),
        }
    }

    #[test]
    fn synapse_identities_cover_add_synapse_and_coordinated() {
        let result = synapse_result(vec![synapse("n1"), synapse("n2")], vec![coordinated()]);
        let ids = synapse_candidate_identities(&result);
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0].change_type, CHANGE_TYPE_ADD_SYNAPSES);
        assert_eq!(ids[0].target_uuid.as_deref(), Some("n1"));
        // Coordinated candidates are target-agnostic.
        assert_eq!(ids[2].change_type, CHANGE_TYPE_COORDINATED_STRUCTURAL);
        assert!(ids[2].target_uuid.is_none());
    }

    #[test]
    fn neuron_identities_carry_target_and_squash() {
        let result = neuron_result(vec![neuron("n1", "RELU")]);
        let ids = neuron_candidate_identities(&result);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].change_type, CHANGE_TYPE_ADD_NEURONS);
        assert_eq!(ids[0].target_uuid.as_deref(), Some("n1"));
        assert_eq!(ids[0].target_squash.as_deref(), Some("RELU"));
    }

    #[test]
    fn breakdown_wires_failure_cache_reason_when_suppressed() {
        let mut breakdown = RejectionBreakdown::new();
        breakdown.record_many(REJECTION_BELOW_THRESHOLD, 3);
        let map = breakdown_with_failure_cache(&breakdown, 2);
        assert_eq!(map.get(REJECTION_DUPLICATE_OF_FAILURE_CACHE), Some(&2));
        assert_eq!(map.get(REJECTION_BELOW_THRESHOLD), Some(&3));
    }

    #[test]
    fn breakdown_omits_failure_cache_reason_when_none_suppressed() {
        let breakdown = RejectionBreakdown::new();
        let map = breakdown_with_failure_cache(&breakdown, 0);
        assert!(!map.contains_key(REJECTION_DUPLICATE_OF_FAILURE_CACHE));
    }

    /// Issue #1800: the classifier input carries the suppression count once,
    /// and building the surfaced wire maps from the same surface breakdowns
    /// does not add a second copy to it.
    #[test]
    fn classifier_breakdown_folds_suppression_without_double_counting() {
        let mut synapse = RejectionBreakdown::new();
        synapse.record_many(REJECTION_BELOW_THRESHOLD, 3);
        let mut neuron = RejectionBreakdown::new();
        neuron.record_many(REJECTION_BELOW_THRESHOLD, 1);

        let combined = starvation_classifier_breakdown(
            Some(&synapse),
            Some(&neuron),
            &RejectionBreakdown::new(),
            5,
        );

        assert_eq!(
            combined.counts().get(REJECTION_DUPLICATE_OF_FAILURE_CACHE),
            Some(&5)
        );
        assert_eq!(combined.counts().get(REJECTION_BELOW_THRESHOLD), Some(&4));
        assert_eq!(combined.total(), 9);

        // The surfaced wire maps are built independently and leave the
        // classifier input untouched.
        let _syn_map = breakdown_with_failure_cache(&synapse, 3);
        let _neu_map = breakdown_with_failure_cache(&neuron, 2);
        assert_eq!(combined.total(), 9);
    }
}

#[cfg(test)]
mod phase_gating_wiring_tests {
    //! `analyze_parallel` phase gating reaches the orchestrator (Issue #1937).
    //!
    //! `docs/FFI_API.md` documents `includeSynapseAnalysis` /
    //! `includeNeuronAnalysis` on the `analyze_parallel` request. The FFI
    //! payload previously lacked both fields and the conversion hard-coded
    //! `Some(true)`, so serde silently dropped the caller's choice.

    use super::*;

    /// Minimal well-formed `analyze_parallel` payload with the caller's
    /// phase-gating keys spliced in verbatim.
    fn payload(gating: &str) -> String {
        format!(
            r#"{{
                "parquetFile": "/tmp/does-not-need-to-exist.parquet",
                "creature": {{
                    "neurons": [
                        {{ "uuid": "hidden-0", "type": "hidden", "squash": "IDENTITY" }}
                    ],
                    "synapses": [],
                    "input": 1,
                    "output": 1
                }},
                "focusNeurons": ["hidden-0"]{gating}
            }}"#
        )
    }

    fn convert(gating: &str) -> AnalyzeAllInput {
        let input: AnalyzeParallelInput = serde_json::from_str(&payload(gating))
            .expect("payload must deserialise as AnalyzeParallelInput");
        build_analyze_all_input_from_parallel(input)
    }

    #[test]
    fn neuron_phase_can_be_disabled_by_the_caller() {
        let converted = convert(r#", "includeNeuronAnalysis": false"#);
        assert_eq!(converted.include_neuron_analysis, Some(false));
        assert_eq!(converted.include_synapse_analysis, None);
    }

    #[test]
    fn synapse_phase_can_be_disabled_by_the_caller() {
        let converted = convert(r#", "includeSynapseAnalysis": false"#);
        assert_eq!(converted.include_synapse_analysis, Some(false));
        assert_eq!(converted.include_neuron_analysis, None);
    }

    #[test]
    fn both_phases_can_be_disabled_together() {
        let converted =
            convert(r#", "includeSynapseAnalysis": false, "includeNeuronAnalysis": false"#);
        assert_eq!(converted.include_synapse_analysis, Some(false));
        assert_eq!(converted.include_neuron_analysis, Some(false));
    }

    #[test]
    fn explicit_true_is_preserved() {
        let converted =
            convert(r#", "includeSynapseAnalysis": true, "includeNeuronAnalysis": true"#);
        assert_eq!(converted.include_synapse_analysis, Some(true));
        assert_eq!(converted.include_neuron_analysis, Some(true));
    }

    /// Omitting both keys leaves them `None`, which the orchestrator reads as
    /// `true` — the documented default keeps existing callers unchanged.
    #[test]
    fn absent_fields_default_to_running_both_phases() {
        let converted = convert("");
        assert_eq!(converted.include_synapse_analysis, None);
        assert_eq!(converted.include_neuron_analysis, None);
    }

    /// Issue #4138: `analyze_parallel` must forward the caller budget into
    /// `AnalyzeAllInput.max_analysis_memory_mb`, which `analyze_all` then
    /// passes to `RecordCache::new_adaptive_with_deadline_and_budget` rather
    /// than hard-coding `None`.
    #[test]
    fn analyze_parallel_forwards_caller_memory_budget() {
        let converted = convert(r#", "maxAnalysisMemoryMb": 3457"#);
        assert_eq!(converted.max_analysis_memory_mb, Some(3457));
    }
}

#[cfg(test)]
mod issue_4139_deadline_skip_tests {
    //! Sub-minimum analysis deadlines skip as a non-fatal outcome (Issue #4139).
    //!
    //! The caller asked for 0.89 s and previously received a 10-minute grant. The
    //! skip must happen *before* GPU availability is checked so hosts without
    //! a GPU still see `success: true` rather than
    //! `Rust neuron analysis unavailable (failed during analysis dispatch)`.

    use super::*;

    fn skip_payload(deadline_ms: u64) -> String {
        format!(
            r#"{{
                "parquetFile": "/tmp/does-not-need-to-exist.parquet",
                "creature": {{
                    "neurons": [
                        {{ "uuid": "input-1", "type": "input", "squash": "IDENTITY", "bias": 0.0 }},
                        {{ "uuid": "output-1", "type": "output", "squash": "LOGISTIC", "bias": 0.0 }}
                    ],
                    "synapses": [
                        {{ "from_uuid": "input-1", "to_uuid": "output-1", "weight": 1.0 }}
                    ],
                    "input": 1,
                    "output": 1
                }},
                "focusNeurons": ["output-1"],
                "analysisDeadlineMs": {deadline_ms},
                "randomSeed": 42
            }}"#
        )
    }

    #[test]
    fn sub_minimum_deadline_is_non_fatal_skip_not_dispatch_error() {
        let json = analyze_parallel_internal(&skip_payload(890))
            .expect("skip path must return Ok JSON, not Err");
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("skip path must be valid JSON");

        assert_eq!(
            value["success"], true,
            "budget skip is a normal outcome, got: {value}"
        );
        assert!(
            value.get("error").is_none() || value["error"].is_null(),
            "skip must not surface an error field, got: {value}"
        );
        let dump = value.to_string();
        assert!(
            !dump.contains("failed during analysis dispatch"),
            "skip must not be classified as a dispatch failure: {dump}"
        );
        assert!(
            !dump.contains("Rust neuron analysis unavailable"),
            "skip must not be the generic neuron-analysis-unavailable error: {dump}"
        );
        assert_eq!(
            value["cancelled"], true,
            "skip is signalled as cancelled so the host can distinguish it from \
             search exhaustion, got: {value}"
        );
    }
}

#[cfg(test)]
mod issue_4138_budget_ffi_tests {
    //! Caller memory budget reaches the analysis FFI types (Issue #4138).

    use super::*;

    #[test]
    fn rank_focus_reports_supplied_budget_mb() {
        let input = r#"{
            "parquetFile": "/tmp/does-not-need-to-exist.parquet",
            "creature": {
                "neurons": [
                    { "uuid": "hidden-0", "type": "hidden", "squash": "IDENTITY", "bias": 0.0 },
                    { "uuid": "output-0", "type": "output", "squash": "IDENTITY", "bias": 0.0 }
                ],
                "synapses": [
                    { "from_uuid": "hidden-0", "to_uuid": "output-0", "weight": 1.0 }
                ],
                "input": 1,
                "output": 1
            },
            "maxAnalysisMemoryMb": 3457,
            "analysisDeadlineMs": 890
        }"#;
        let json = rank_focus_neurons_internal(input).expect("rank_focus must return JSON");
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("rank_focus JSON must parse");
        assert_eq!(value["success"], true, "got: {value}");
        assert_eq!(
            value["budgetMb"], 3457,
            "supplied budget must be reported, not hard-coded null: {value}"
        );
    }
}
