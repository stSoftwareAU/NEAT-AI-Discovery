//! Internal business-logic functions for analysis FFI entry points.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use anyhow::Result;

use crate::ffi_types::*;
use crate::{analysis, focus};

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
    if let Err(typed) = validate_forward_only_synapses(&input.creature) {
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
            let novelty_escalation_active = handshake.novelty_escalation_active;

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
                    environmental_gates,
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
                    // Issue #1409: explicit starvation flag for the GRQ layer.
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
                    // Issue #1409: explicit starvation flag for the GRQ layer.
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
            let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
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
                environmentally_disabled,
                zero_candidate_summary: None,
                error: Some(err_msg),
                error_kind,
                retryable,
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
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
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
    if let Err(typed) = validate_forward_only_synapses(&input.creature) {
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
            budget_mb: None,
            projected_mb: None,
            error: Some(typed.to_string()),
            error_kind: Some(kind),
            retryable: Some(kind.is_retryable()),
        };
        return Ok(serde_json::to_string(&output)?);
    }

    // Issue #1048 / #1077: Track that an analysis is active so the host
    // knows not to delete the parquet temp directory until we finish.
    // Uses an RAII guard so the counter is decremented even on panic.
    let _active_guard = crate::cancellation::AnalysisActiveGuard::new();

    // Issue #1318: forward the optional task descriptor so OneHot / Margin
    // topologies activate margin-aware focus ranking. Other topologies
    // (Independent / Simplex / Unknown / OTHER) and `None` get the existing
    // unweighted ranking.
    // Issue #1407: thread the shared absolute discovery deadline so focus
    // selection bills against the same budget as the analysis phase rather
    // than opening a fresh independent window.
    let rank_result = focus::rank_focus_neurons_with_descriptor_and_deadline(
        &input.parquet_file,
        &input.creature,
        input.max_results,
        input.cost_of_growth,
        input.task_descriptor.as_ref(),
        input.analysis_deadline_ms,
    );

    match rank_result {
        Ok(stats) => {
            // Issue #1445: Build the diversity-aware focus selection over the
            // ranked pool BEFORE the neurons are consumed into JSON. The
            // weighted_score stored on each ranked neuron is the roulette
            // weight; selection enforces a diversity floor (or drought
            // rotation) so a single dominant neuron cannot collapse the focus
            // set to one target.
            let focus_candidates: Vec<focus::FocusCandidate> = stats
                .neurons
                .iter()
                .map(|n| focus::FocusCandidate {
                    neuron_uuid: n.neuron_uuid.clone(),
                    weight: n.weighted_score,
                })
                .collect();
            let focus_set_size = input.focus_set_size.unwrap_or(DEFAULT_FOCUS_SET_SIZE);
            let drought_threshold = u64::from(crate::config::drought_log_threshold());
            let epochs = input.epochs_since_last_accepted_candidate.unwrap_or(0);
            let drought_active = drought_threshold > 0 && epochs >= drought_threshold;
            let focus_selection = focus::select_focus_neurons(
                &focus_candidates,
                focus_set_size,
                drought_active,
                epochs,
            );

            // Issue #1445: WARN when the raw roulette is pathologically
            // single-target so operators can see the collapse the diversity
            // floor / rotation just corrected.
            if focus_selection.raw_weight_concentration_ratio > focus::CONCENTRATION_WARN_THRESHOLD
            {
                tracing::warn!(
                    raw_weight_concentration_ratio = focus_selection.raw_weight_concentration_ratio,
                    effective_weight_concentration_ratio =
                        focus_selection.weight_concentration_ratio,
                    diversity_floor_applied = focus_selection.diversity_floor_applied,
                    rotation_applied = focus_selection.rotation_applied,
                    pool_size = focus_selection.pool_size,
                    focus_set_size,
                    "focus_selection_weight_concentration_high: a single neuron \
                     dominated the focus-selection roulette; diversity floor / \
                     drought rotation applied to spread the focus set"
                );
            }

            let focus_selection_json = FocusSelectionJson {
                selected: focus_selection.selected,
                raw_weight_concentration_ratio: focus_selection.raw_weight_concentration_ratio,
                weight_concentration_ratio: focus_selection.weight_concentration_ratio,
                diversity_floor_applied: focus_selection.diversity_floor_applied,
                rotation_applied: focus_selection.rotation_applied,
                pool_size: focus_selection.pool_size,
            };

            let neurons: Vec<RankedNeuronJson> = stats
                .neurons
                .into_iter()
                .map(|neuron| RankedNeuronJson {
                    neuron_uuid: neuron.neuron_uuid,
                    total_error: neuron.total_error,
                    impact: neuron.impact,
                    mean_activation: neuron.mean_activation,
                    activation_weighted_impact: neuron.activation_weighted_impact,
                    weighted_score: neuron.weighted_score,
                })
                .collect();
            let removal_candidates: Vec<RemovalCandidateJson> = stats
                .removal_candidates
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
            let (error_kind, retryable) = no_error_fields();
            let output = RankFocusNeuronsOutput {
                success: true,
                schema_version: SCHEMA_VERSION.to_string(),
                neurons: Some(neurons),
                focus_selection: Some(focus_selection_json),
                removal_candidates: if removal_candidates.is_empty() {
                    None
                } else {
                    Some(removal_candidates)
                },
                // Issue #306: Return constant neuron removal candidates
                constant_neuron_removals: if stats.constant_neuron_removals.is_empty() {
                    None
                } else {
                    Some(stats.constant_neuron_removals)
                },
                max_output_error: Some(stats.max_output_error),
                processed_neurons: Some(stats.processed_neurons),
                total_neurons: Some(stats.total_neurons),
                duration_ms: Some(stats.duration_ms.min(u64::MAX as u128) as u64),
                // Issue #1142: Surface rejection counts (e.g. removal candidates
                // dropped by the noise-floor gate) so operators can root-cause
                // "no candidates found" failures without re-running analysis.
                rejection_breakdown: if stats.rejection_breakdown.is_empty() {
                    None
                } else {
                    Some(stats.rejection_breakdown)
                },
                // Issue #1172: Surface the chosen record loading mode and
                // memory budget projection so callers can tune low-memory
                // hosts without scraping log lines.
                loading_mode: Some(stats.loading_mode.as_str().to_string()),
                lazy_reason: Some(stats.lazy_reason.as_str().to_string()),
                budget_mb: stats.budget_mb,
                projected_mb: Some(stats.projected_mb),
                error: None,
                error_kind,
                retryable,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
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
                budget_mb: None,
                projected_mb: None,
                error: Some(err_msg),
                error_kind,
                retryable,
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
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
}
