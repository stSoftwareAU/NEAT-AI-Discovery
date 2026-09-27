//! Candidate selection from target analysis
//!
//! This module handles epistatic pair detection, synergistic candidate detection,
//! and redundant path detection for coordinated structural candidates.
//!
//! Extracted from `target_analysis.rs` as part of Issue #599.

use crate::analysis::detection::redundant_path::{
    ExistingPathContribution, detect_redundant_paths, redundant_paths_to_coordinated_candidates,
};
use crate::analysis::recommendation::epistatic::{
    ScanTruncation, SourceContribution, deduplicate_by_dominant_neuron,
    deduplicate_synergistic_by_dominant_neuron, detect_epistatic_pairs_with_deadline,
    detect_synergistic_candidates_with_deadline, epistatic_pairs_to_coordinated_candidates,
    filter_interfering_epistatic_pairs, filter_interfering_synergistic_candidates,
    synergistic_to_coordinated_candidates,
};
use crate::analysis::utils::verbose_enabled;

use super::TargetAnalysisContext;
use super::TargetAnalysisResults;

/// Detect epistatic neuron pairs and synergistic candidates from source contributions.
///
/// Appends any discovered coordinated structural candidates to results.
/// (Issue #202, Issue #189, Issue #509)
pub(crate) fn detect_epistatic_and_synergistic(
    target_uuid: &str,
    source_contributions: &[SourceContribution],
    ctx: &TargetAnalysisContext,
    results: &mut TargetAnalysisResults,
) {
    if verbose_enabled() {
        tracing::debug!(
            target_uuid = target_uuid,
            source_contribution_count = source_contributions.len(),
            "Collected source contributions for epistatic detection."
        );
    }
    if source_contributions.len() < 2 {
        return;
    }

    let target_is_output = ctx
        .neuron_type_map
        .get(target_uuid)
        .is_some_and(|t| *t == "output");
    let target_impact = if target_is_output { 1.0 } else { 0.5 };

    // Issue #897: Pass target squash function for saturation-aware estimation
    let target_squash = ctx.neuron_squash_map.get(target_uuid).copied();

    // Issue #202: Detect epistatic neuron pairs
    // Issue #2190: bounded by the analysis deadline, cancellation and a ceiling.
    let epistatic_scan = detect_epistatic_pairs_with_deadline(
        target_uuid,
        source_contributions,
        target_impact,
        target_squash,
        &ctx.deadline,
    );
    report_truncation(target_uuid, "epistatic", epistatic_scan.truncation);
    let epistatic_pairs = epistatic_scan.candidates;

    if !epistatic_pairs.is_empty() {
        let filtered_pairs =
            filter_interfering_epistatic_pairs(epistatic_pairs, source_contributions);

        // Issue #509: Deduplicate pairs sharing a dominant neuron
        let deduped_pairs = deduplicate_by_dominant_neuron(filtered_pairs);

        if !deduped_pairs.is_empty() {
            let epistatic_candidates = epistatic_pairs_to_coordinated_candidates(&deduped_pairs);
            if !epistatic_candidates.is_empty() {
                results.coordinated.extend(epistatic_candidates);

                if verbose_enabled() {
                    tracing::trace!(
                        target_uuid = target_uuid,
                        epistatic_pair_count = deduped_pairs.len(),
                        "Found epistatic pair(s) for target."
                    );
                }
            }
        }
    }

    // Issue #189: Detect synergistic candidates via residual analysis
    let synergistic_scan = detect_synergistic_candidates_with_deadline(
        target_uuid,
        source_contributions,
        target_impact,
        target_squash,
        &ctx.deadline,
    );
    report_truncation(target_uuid, "synergistic", synergistic_scan.truncation);
    let synergistic_candidates = synergistic_scan.candidates;

    if !synergistic_candidates.is_empty() {
        let filtered_synergistic =
            filter_interfering_synergistic_candidates(synergistic_candidates, source_contributions);

        // Issue #509: Deduplicate candidates sharing a dominant (primary) neuron
        let deduped_synergistic = deduplicate_synergistic_by_dominant_neuron(filtered_synergistic);

        if !deduped_synergistic.is_empty() {
            let synergistic_coordinated =
                synergistic_to_coordinated_candidates(&deduped_synergistic);
            if !synergistic_coordinated.is_empty() {
                results.coordinated.extend(synergistic_coordinated);

                if verbose_enabled() {
                    tracing::trace!(
                        target_uuid = target_uuid,
                        synergistic_candidate_count = deduped_synergistic.len(),
                        "Found synergistic candidate(s) for target."
                    );
                }
            }
        }
    }
}

/// Log a scan that stopped early so partial results are never mistaken for a
/// complete scan (Issue #2190).
fn report_truncation(target_uuid: &str, scan: &str, truncation: Option<ScanTruncation>) {
    if let Some(reason) = truncation {
        tracing::warn!(
            target_uuid = target_uuid,
            scan = scan,
            reason = ?reason,
            "Candidate scan stopped early; returning partial candidates."
        );
    }
}

/// Detect redundant paths among existing synapse edges.
///
/// Appends any discovered coordinated structural candidates to results.
/// (Issue #164)
pub(crate) fn detect_redundant_path_candidates(
    target_uuid: &str,
    existing_path_contributions: &[ExistingPathContribution],
    ctx: &TargetAnalysisContext,
    results: &mut TargetAnalysisResults,
) {
    if existing_path_contributions.len() < 2 {
        return;
    }

    let target_is_output = ctx
        .neuron_type_map
        .get(target_uuid)
        .is_some_and(|t| *t == "output");
    let target_impact = if target_is_output { 1.0 } else { 0.5 };

    let redundant_paths =
        detect_redundant_paths(target_uuid, existing_path_contributions, target_impact);

    if !redundant_paths.is_empty() {
        let redundant_coordinated = redundant_paths_to_coordinated_candidates(&redundant_paths);
        if !redundant_coordinated.is_empty() {
            results.coordinated.extend(redundant_coordinated);

            if verbose_enabled() {
                tracing::trace!(
                    target_uuid = target_uuid,
                    redundant_path_count = redundant_paths.len(),
                    "Found redundant path(s) for pruning on target."
                );
            }
        }
    }
}
