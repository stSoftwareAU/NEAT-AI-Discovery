//! Diagnostic tracking and rejection reason structures for analysis.
//!
//! This module provides visibility into why candidates were rejected during discovery:
//! - `NEAT_AI_DISCOVERY_VERBOSE=1` enables detailed logging
//! - JSON responses include `diagnostics` arrays
//! - Critical for debugging "no candidates found" situations
//!
//! The rejection tracking is essential for understanding discovery behaviour without
//! having to dig through logs.
//!
//! **Extracted from implementation.rs as part of Issue #271**

use crate::focus::{compute_impacts_public, compute_impacts_with_activations, RecordProvider};
use crate::types::DiscoverRecord;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

// Import shared types
use crate::analysis::shared::{
    NeuronNoCandidateDetail, NeuronNoCandidateReason, NeuronNoCandidateSummary,
    SynapseNoCandidateDetail, SynapseNoCandidateReason, SynapseNoCandidateSummary,
};

// Import activation functions for threshold detection
use crate::analysis::activation::is_threshold_activation;

// Import sample data structures
use crate::analysis::samples::HelpfulSample;

// Import utility functions
use crate::analysis::utils::verbose_enabled;

// =============================================================================
// RecordCacheProvider - Adapter for focus impact calculation
// =============================================================================

/// Adapter to allow `RecordCache` to be used where focus impact code expects a `RecordProvider`.
///
/// This lets us compute squash-aware impacts using *recorded activations* for selection squashes
/// (MINIMUM/MAXIMUM/IF) during candidate discounting, rather than falling back to the conservative
/// 1/N probability model.
pub(crate) struct RecordCacheProvider<'a> {
    pub(crate) cache: &'a super::cache::RecordCache,
}

impl RecordProvider for RecordCacheProvider<'_> {
    fn get(&self, neuron_uuid: &str) -> Result<Option<Arc<Vec<DiscoverRecord>>>> {
        let records = self.cache.get(neuron_uuid)?;
        if records.is_empty() {
            Ok(None)
        } else {
            Ok(Some(records))
        }
    }

    fn len(&self) -> usize {
        let cache = self
            .cache
            .cache
            .lock()
            .expect("record cache mutex poisoned");
        cache.len()
    }
}

/// Compute neuron impact scores for candidate discounting.
///
/// We prefer activation-based selection statistics when available so MINIMUM/MAXIMUM/IF neurons
/// don't get incorrectly diluted via the 1/N fallback.
pub(crate) fn compute_impact_scores_for_discounting(
    creature: &crate::CreatureJson,
    cache: &super::cache::RecordCache,
) -> HashMap<String, f32> {
    let provider = RecordCacheProvider { cache };
    match compute_impacts_with_activations(creature, &provider) {
        Ok(scores) => scores,
        Err(err) => {
            if verbose_enabled() {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Falling back to conservative impact calculation \
                    (no activation-based selection stats). Reason: {err}"
                );
            }
            compute_impacts_public(creature)
        }
    }
}

// =============================================================================
// Synapse Rejection Tracking
// =============================================================================

/// Why a synapse candidate was rejected during analysis.
#[derive(Clone, Copy)]
pub(crate) enum RejectionReason {
    /// No overlapping discovery samples between source and target.
    NoSamples,
    /// No consistent improvement in GPU stats.
    ZeroImprovement,
    /// Expected improvement below threshold.
    BelowThreshold,
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RejectionReason::NoSamples => write!(f, "no overlapping discovery samples"),
            RejectionReason::ZeroImprovement => write!(f, "no consistent improvement in GPU stats"),
            RejectionReason::BelowThreshold => write!(f, "expected improvement below threshold"),
        }
    }
}

/// Detailed rejection information for a synapse candidate.
#[derive(Clone)]
pub(crate) struct RejectionDetail {
    pub(crate) source_uuid: String,
    pub(crate) reason: RejectionReason,
    pub(crate) sample_count: usize,
    pub(crate) source_record_count: usize,
    pub(crate) improved_count: u32,
    pub(crate) worsened_count: u32,
    pub(crate) expected_improvement: f32,
    pub(crate) threshold: f32,
    pub(crate) weight: Option<f32>,
}

impl RejectionDetail {
    pub(crate) fn score(&self) -> f32 {
        self.expected_improvement
    }
}

/// Threshold calculation context for recording below-threshold rejections.
pub(crate) struct ThresholdContext {
    pub(crate) sample_count: usize,
    pub(crate) expected_improvement: f32,
    pub(crate) threshold: f32,
    pub(crate) improved_count: u32,
    pub(crate) worsened_count: u32,
    pub(crate) weight: f32,
}

/// Per-target diagnostic entry for synapse analysis.
pub(crate) struct TargetDiagnosticEntry {
    pub(crate) target_uuid: String,
    pub(crate) target_record_count: usize,
    pub(crate) evaluated_candidates: u32,
    pub(crate) candidates_with_samples: u32,
    pub(crate) total_eligible_sources: u32,
    pub(crate) input_neuron_count: u32,
    pub(crate) already_connected_count: u32,
    pub(crate) record_load_failures: u32,
    pub(crate) had_candidate: bool,
    pub(crate) best_rejection: Option<RejectionDetail>,
}

impl TargetDiagnosticEntry {
    pub(crate) fn new(target_uuid: &str) -> Self {
        Self {
            target_uuid: target_uuid.to_string(),
            target_record_count: 0,
            evaluated_candidates: 0,
            candidates_with_samples: 0,
            total_eligible_sources: 0,
            input_neuron_count: 0,
            already_connected_count: 0,
            record_load_failures: 0,
            had_candidate: false,
            best_rejection: None,
        }
    }

    pub(crate) fn update_best(&mut self, detail: RejectionDetail) {
        match &self.best_rejection {
            Some(current) => {
                if detail.score() > current.score()
                    || (detail.score() == current.score()
                        && detail.sample_count > current.sample_count)
                {
                    self.best_rejection = Some(detail);
                }
            }
            None => self.best_rejection = Some(detail),
        }
    }
}

/// Collection of target diagnostics for synapse analysis.
pub(crate) struct TargetDiagnostics {
    log_enabled: bool,
    pub(crate) entries: HashMap<String, TargetDiagnosticEntry>,
}

impl TargetDiagnostics {
    pub(crate) fn new(targets: &[&String]) -> Self {
        let log_enabled = verbose_enabled();
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert(target.to_string(), TargetDiagnosticEntry::new(target));
        }
        Self {
            log_enabled,
            entries,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_tests(targets: &[&str]) -> Self {
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert((*target).to_string(), TargetDiagnosticEntry::new(target));
        }
        Self {
            log_enabled: true,
            entries,
        }
    }

    pub(crate) fn set_target_record_count(&mut self, target_uuid: &str, count: usize) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.target_record_count = count;
        }
    }

    pub(crate) fn set_total_eligible_sources(&mut self, target_uuid: &str, count: u32) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.total_eligible_sources = count;
        }
    }

    pub(crate) fn set_input_neuron_count(&mut self, target_uuid: &str, count: u32) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.input_neuron_count = count;
        }
    }

    pub(crate) fn record_already_connected(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.already_connected_count += 1;
        }
    }

    pub(crate) fn record_load_failure(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.record_load_failures += 1;
        }
    }

    pub(crate) fn record_candidate_attempt(&mut self, target_uuid: &str, had_samples: bool) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.evaluated_candidates += 1;
            if had_samples {
                entry.candidates_with_samples += 1;
            }
        }
    }

    pub(crate) fn record_no_samples(
        &mut self,
        target_uuid: &str,
        source_uuid: &str,
        source_record_count: usize,
    ) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::NoSamples,
                sample_count: 0,
                source_record_count,
                improved_count: 0,
                worsened_count: 0,
                expected_improvement: f32::NEG_INFINITY,
                threshold: 0.0,
                weight: None,
            });
        }
    }

    pub(crate) fn record_zero_improvement(
        &mut self,
        target_uuid: &str,
        source_uuid: &str,
        sample_count: usize,
        positive_count: u32,
        negative_count: u32,
    ) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::ZeroImprovement,
                sample_count,
                source_record_count: sample_count,
                improved_count: positive_count.max(negative_count),
                worsened_count: positive_count.min(negative_count),
                expected_improvement: 0.0,
                threshold: 0.0,
                weight: None,
            });
        }
    }

    pub(crate) fn record_below_threshold(
        &mut self,
        target_uuid: &str,
        source_uuid: &str,
        context: ThresholdContext,
    ) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::BelowThreshold,
                sample_count: context.sample_count,
                source_record_count: context.sample_count,
                improved_count: context.improved_count,
                worsened_count: context.worsened_count,
                expected_improvement: context.expected_improvement,
                threshold: context.threshold,
                weight: Some(context.weight),
            });
        }
    }

    pub(crate) fn mark_candidate_selected(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.had_candidate = true;
        }
    }

    pub(crate) fn emit_logs(&self) {
        if !self.log_enabled {
            return;
        }

        for entry in self.entries.values() {
            if entry.had_candidate {
                continue;
            }

            if entry.total_eligible_sources == 0 {
                // This should never happen - input/constant neurons are skipped early
                // and hidden/output neurons should always have at least input neurons as eligible sources
                // Skip logging to avoid cluttering logs with impossible conditions
                continue;
            }

            // Check if neuron is fully connected (all eligible sources already have synapses)
            // Eligible sources include: ALL input neurons (input-0 through input-(creature.input-1))
            // AND ALL prior hidden/output neurons (with index < target_index, excluding constants)
            // This condition is rare - only occurs when neuron is connected to all possible sources
            if entry.already_connected_count == entry.total_eligible_sources {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} is fully connected: all {} eligible upstream sources already have synapses (all {} input neurons and all prior hidden/output neurons). This is a rare condition.",
                    entry.target_uuid, entry.total_eligible_sources, entry.input_neuron_count
                );
                continue;
            }

            // Check for record loading failures (this indicates a bug)
            if entry.record_load_failures > 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} had {} record loading failures (this may indicate a bug - records exist but couldn't be loaded).",
                    entry.target_uuid, entry.record_load_failures
                );
            }

            if entry.evaluated_candidates == 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream neurons but none were evaluated ({} already connected, {} record load failures).",
                    entry.target_uuid, entry.total_eligible_sources, entry.already_connected_count, entry.record_load_failures
                );
                continue;
            }

            let best = match &entry.best_rejection {
                Some(detail) => detail,
                None => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} evaluated {} potential synapses but recorded no diagnostics.",
                        entry.target_uuid, entry.evaluated_candidates
                    );
                    continue;
                }
            };

            match best.reason {
                RejectionReason::NoSamples => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} skipped candidate from {} because no aligned samples were available (source records {}, target records {}).",
                        entry.target_uuid,
                        best.source_uuid,
                        best.source_record_count,
                        entry.target_record_count
                    );
                }
                RejectionReason::ZeroImprovement => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} saw {} aligned samples from {} but GPU stats reported zero consistent improvements (positive {}, negative {}).",
                        entry.target_uuid,
                        best.sample_count,
                        best.source_uuid,
                        best.improved_count,
                        best.worsened_count
                    );
                }
                RejectionReason::BelowThreshold => {
                    if let Some(weight) = best.weight {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {} best candidate from {} improved {:.4} but remained below threshold {:.4} (improved {}, worsened {}, suggested weight {:.4}).",
                            entry.target_uuid,
                            best.source_uuid,
                            best.expected_improvement,
                            best.threshold,
                            best.improved_count,
                            best.worsened_count,
                            weight
                        );
                    } else {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {} best candidate from {} improved {:.4} but remained below threshold {:.4} (improved {}, worsened {}).",
                            entry.target_uuid,
                            best.source_uuid,
                            best.expected_improvement,
                            best.threshold,
                            best.improved_count,
                            best.worsened_count
                        );
                    }
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn entry_for(&self, target_uuid: &str) -> Option<&TargetDiagnosticEntry> {
        self.entries.get(target_uuid)
    }

    pub(crate) fn no_candidate_summaries(&self) -> Vec<SynapseNoCandidateSummary> {
        self.entries
            .values()
            .filter(|entry| !entry.had_candidate)
            .map(|entry| {
                // Only report "no eligible sources" if both total_eligible_sources and evaluated_candidates are 0
                // This handles the case where total_eligible_sources might be 0 in tests but evaluated_candidates > 0
                if entry.total_eligible_sources == 0 && entry.evaluated_candidates == 0 {
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: SynapseNoCandidateReason::NoEligibleSources,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                // Check if neuron is fully connected (all eligible sources already have synapses)
                // Eligible sources include: ALL input neurons (input-0 through input-(creature.input-1))
                // AND ALL prior hidden/output neurons (with index < target_index, excluding constants)
                // This condition is rare - only occurs when neuron is connected to all possible sources
                if entry.total_eligible_sources > 0
                    && entry.already_connected_count == entry.total_eligible_sources
                    && entry.evaluated_candidates == 0
                {
                    // Neuron is fully connected - all eligible sources (all inputs + all prior hidden neurons) already have synapses
                    // This is legitimate but rare, and we report it as "no eligible sources"
                    // since there are no NEW sources to evaluate
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: SynapseNoCandidateReason::NoEligibleSources,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                if let Some(best) = &entry.best_rejection {
                    let reason = match best.reason {
                        RejectionReason::NoSamples => SynapseNoCandidateReason::NoSamples,
                        RejectionReason::ZeroImprovement => {
                            SynapseNoCandidateReason::ZeroImprovement
                        }
                        RejectionReason::BelowThreshold => SynapseNoCandidateReason::BelowThreshold,
                    };
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: Some(SynapseNoCandidateDetail {
                            source_uuid: Some(best.source_uuid.clone()),
                            sample_count: Some(best.sample_count),
                            source_record_count: Some(best.source_record_count),
                            improved_count: Some(best.improved_count),
                            worsened_count: Some(best.worsened_count),
                            expected_improvement: Some(best.expected_improvement),
                            threshold: Some(best.threshold),
                            suggested_weight: best.weight,
                        }),
                    };
                }

                SynapseNoCandidateSummary {
                    target_uuid: entry.target_uuid.clone(),
                    reason: SynapseNoCandidateReason::NoDiagnostics,
                    evaluated_candidates: entry.evaluated_candidates,
                    candidates_with_samples: entry.candidates_with_samples,
                    target_record_count: entry.target_record_count,
                    detail: None,
                }
            })
            .collect()
    }
}

// =============================================================================
// Neuron Rejection Tracking
// =============================================================================

/// Detail about why a source was rejected (currently only used for NoSamples).
#[derive(Clone)]
pub(crate) struct NeuronRejectionDetail {
    pub(crate) source_uuid: String,
    pub(crate) orientation: Option<&'static str>,
    pub(crate) sample_count: usize,
    pub(crate) expected_improvement: f32,
}

impl NeuronRejectionDetail {
    pub(crate) fn score(&self) -> f32 {
        self.expected_improvement
    }
}

/// Per-neuron diagnostic entry for neuron analysis.
pub(crate) struct NeuronDiagnosticEntry {
    pub(crate) target_uuid: String,
    pub(crate) target_record_count: usize,
    pub(crate) total_eligible_sources: u32,
    pub(crate) record_load_failures: u32,
    pub(crate) evaluated_sources: u32,
    pub(crate) sources_with_samples: u32,
    pub(crate) had_candidate: bool,
    pub(crate) best_rejection: Option<NeuronRejectionDetail>,
    /// Set to true when this neuron was filtered out because it's a hidden neuron
    /// (only output neurons are valid targets for add-neuron analysis).
    pub(crate) hidden_filtered: bool,
    /// Set to true when this neuron was filtered out because it's an input neuron
    /// (input neurons are observation sources, not computation nodes).
    pub(crate) input_filtered: bool,
    /// Set to true when this neuron was filtered out because it's a constant neuron
    /// (constant neurons don't receive inputs - they always output a fixed value).
    pub(crate) constant_filtered: bool,
}

impl NeuronDiagnosticEntry {
    pub(crate) fn new(target_uuid: &str) -> Self {
        Self {
            target_uuid: target_uuid.to_string(),
            target_record_count: 0,
            total_eligible_sources: 0,
            record_load_failures: 0,
            evaluated_sources: 0,
            sources_with_samples: 0,
            had_candidate: false,
            best_rejection: None,
            hidden_filtered: false,
            input_filtered: false,
            constant_filtered: false,
        }
    }

    pub(crate) fn update_best(&mut self, detail: NeuronRejectionDetail) {
        match &self.best_rejection {
            Some(current) => {
                if detail.score() > current.score()
                    || (detail.score() == current.score()
                        && detail.sample_count > current.sample_count)
                {
                    self.best_rejection = Some(detail);
                }
            }
            None => self.best_rejection = Some(detail),
        }
    }
}

/// Collection of neuron diagnostics for neuron analysis.
pub(crate) struct NeuronDiagnostics {
    log_enabled: bool,
    pub(crate) entries: HashMap<String, NeuronDiagnosticEntry>,
}

impl NeuronDiagnostics {
    pub(crate) fn new(targets: &[&String]) -> Self {
        let log_enabled = verbose_enabled();
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert(target.to_string(), NeuronDiagnosticEntry::new(target));
        }
        Self {
            log_enabled,
            entries,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_tests(targets: &[&str]) -> Self {
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert((*target).to_string(), NeuronDiagnosticEntry::new(target));
        }
        Self {
            log_enabled: true,
            entries,
        }
    }

    pub(crate) fn set_target_record_count(&mut self, target_uuid: &str, count: usize) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.target_record_count = count;
        }
    }

    pub(crate) fn set_total_eligible_sources(&mut self, target_uuid: &str, count: u32) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.total_eligible_sources = count;
        }
    }

    pub(crate) fn record_load_failure(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.record_load_failures += 1;
        }
    }

    pub(crate) fn record_candidate_attempt(&mut self, target_uuid: &str, had_samples: bool) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.evaluated_sources += 1;
            if had_samples {
                entry.sources_with_samples += 1;
            }
        }
    }

    pub(crate) fn record_no_samples(&mut self, target_uuid: &str, source_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(NeuronRejectionDetail {
                source_uuid: source_uuid.to_string(),
                orientation: None,
                sample_count: 0,
                expected_improvement: f32::NEG_INFINITY,
            });
        }
    }

    pub(crate) fn mark_candidate_selected(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.had_candidate = true;
        }
    }

    /// Mark a neuron as filtered out because it's a hidden neuron.
    /// Hidden neurons are not valid targets for add-neuron analysis because their
    /// backpropagated errors don't reliably translate to output error reduction.
    pub(crate) fn mark_hidden_filtered(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.hidden_filtered = true;
        }
    }

    /// Mark a neuron as filtered out because it's an input neuron.
    /// Input neurons are observation sources, not computation nodes - they have
    /// no activation function or error to reduce.
    pub(crate) fn mark_input_filtered(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.input_filtered = true;
        }
    }

    /// Mark a neuron as filtered out because it's a constant neuron.
    /// Constant neurons don't receive inputs - they always output a fixed value
    /// regardless of network state, so adding a connection to them has no effect.
    pub(crate) fn mark_constant_filtered(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.constant_filtered = true;
        }
    }

    pub(crate) fn emit_logs(&self) {
        if !self.log_enabled {
            return;
        }

        for entry in self.entries.values() {
            if entry.had_candidate {
                continue;
            }

            // Check pre-analysis filters FIRST - these take precedence over all other reasons.
            // These neurons are filtered out before analysis even begins, so they won't
            // have any other diagnostic data (eligible sources, samples, etc.).

            // Input neurons are observation sources, not computation nodes
            if entry.input_filtered {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} was filtered out (input neuron). \
                    Input neurons are observation sources, not computation nodes - they have no \
                    activation function or error to reduce.",
                    entry.target_uuid
                );
                continue;
            }

            // Hidden neurons have backpropagated errors that don't reliably predict output error
            if entry.hidden_filtered {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} was filtered out (hidden neuron). \
                    Add-neuron analysis only targets output neurons because hidden neuron error \
                    reduction doesn't reliably translate to creature score improvement.",
                    entry.target_uuid
                );
                continue;
            }

            // Constant neurons don't receive inputs - they always output a fixed value
            if entry.constant_filtered {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} was filtered out (constant neuron). \
                    Constant neurons don't receive inputs - they always output a fixed value \
                    regardless of network state, so adding a connection to them has no effect.",
                    entry.target_uuid
                );
                continue;
            }

            // Check for record loading failures (this indicates a bug or data issue)
            if entry.record_load_failures > 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} had {} record loading failures out of {} eligible sources (this may indicate a data integrity issue - records exist but couldn't be loaded).",
                    entry.target_uuid, entry.record_load_failures, entry.total_eligible_sources
                );
            }

            if entry.evaluated_sources == 0 {
                if entry.total_eligible_sources > 0
                    && entry.record_load_failures == entry.total_eligible_sources
                {
                    // All eligible sources failed to load - this is a data/bug issue, not "no sources"
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream sources but all {} failed to load from parquet file.",
                        entry.target_uuid, entry.total_eligible_sources, entry.record_load_failures
                    );
                } else if entry.total_eligible_sources > 0 && entry.record_load_failures == 0 {
                    // Sources exist, no load failures, but none evaluated - likely timeout before sources could be checked
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream sources but none were evaluated (0 load failures). Likely analysis TIMEOUT before source loading could start.",
                        entry.target_uuid, entry.total_eligible_sources
                    );
                } else if entry.total_eligible_sources > 0 {
                    // Some sources exist, some failures, none evaluated
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream sources but none were evaluated ({} load failures).",
                        entry.target_uuid, entry.total_eligible_sources, entry.record_load_failures
                    );
                } else {
                    // Genuinely no eligible sources (e.g., target is first neuron)
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had no upstream neurons to analyse.",
                        entry.target_uuid
                    );
                }
                continue;
            }

            let best = match &entry.best_rejection {
                Some(detail) => detail,
                None => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} evaluated {} upstream neurons but recorded no diagnostics.",
                        entry.target_uuid, entry.evaluated_sources
                    );
                    continue;
                }
            };

            // Currently only NoSamples is used
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Target {} skipped candidate from {} because no overlapping samples were found.",
                entry.target_uuid, best.source_uuid
            );
        }
    }

    #[cfg(test)]
    pub(crate) fn entry_for(&self, target_uuid: &str) -> Option<&NeuronDiagnosticEntry> {
        self.entries.get(target_uuid)
    }

    pub(crate) fn no_candidate_summaries(&self) -> Vec<NeuronNoCandidateSummary> {
        self.entries
            .values()
            // Include entries that never had a candidate
            .filter(|entry| !entry.had_candidate)
            .map(|entry| {
                // Check pre-analysis filters FIRST - these take precedence over other reasons.
                // These neurons are filtered out before analysis even begins, so they won't
                // have any other diagnostic data (eligible sources, samples, etc.).

                // Input neurons are observation sources, not computation nodes
                if entry.input_filtered {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::InputNeuronFiltered,
                        evaluated_sources: 0,
                        sources_with_samples: 0,
                        target_record_count: 0,
                        detail: None,
                    };
                }

                // Hidden neurons have backpropagated errors that don't reliably predict output error
                if entry.hidden_filtered {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::HiddenNeuronFiltered,
                        evaluated_sources: 0,
                        sources_with_samples: 0,
                        target_record_count: 0,
                        detail: None,
                    };
                }

                // Constant neurons don't receive inputs - they always output a fixed value
                if entry.constant_filtered {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::ConstantNeuronFiltered,
                        evaluated_sources: 0,
                        sources_with_samples: 0,
                        target_record_count: 0,
                        detail: None,
                    };
                }

                // Only report "no eligible sources" if there were genuinely no eligible sources
                // AND no evaluated candidates. If there were eligible sources but they all failed
                // to load or had empty records, report that as NoSamples with context.
                if entry.evaluated_sources == 0 && entry.total_eligible_sources == 0 {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::NoEligibleSources,
                        evaluated_sources: entry.evaluated_sources,
                        sources_with_samples: entry.sources_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                // If evaluated_sources is 0 but total_eligible_sources > 0, sources existed
                // but all failed to load or had empty records - report as NoSamples
                if entry.evaluated_sources == 0 && entry.total_eligible_sources > 0 {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::NoSamples,
                        evaluated_sources: entry.evaluated_sources,
                        sources_with_samples: entry.sources_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                if let Some(best) = &entry.best_rejection {
                    // Currently only NoSamples is ever set as rejection reason
                    let reason = NeuronNoCandidateReason::NoSamples;
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason,
                        evaluated_sources: entry.evaluated_sources,
                        sources_with_samples: entry.sources_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: Some(NeuronNoCandidateDetail {
                            source_uuid: Some(best.source_uuid.clone()),
                            orientation: best.orientation.map(|name| name.to_string()),
                            sample_count: Some(best.sample_count),
                            improved_count: None,
                            worsened_count: None,
                            expected_improvement: Some(best.expected_improvement),
                            threshold: None,
                            outgoing_weight: None,
                        }),
                    };
                }

                NeuronNoCandidateSummary {
                    target_uuid: entry.target_uuid.clone(),
                    reason: NeuronNoCandidateReason::NoDiagnostics,
                    evaluated_sources: entry.evaluated_sources,
                    sources_with_samples: entry.sources_with_samples,
                    target_record_count: entry.target_record_count,
                    detail: None,
                }
            })
            .collect()
    }
}

// =============================================================================
// Target Data Structures for Sample Building
// =============================================================================

/// Target data for a single observation (used for matching with source records).
pub(crate) struct TargetData {
    pub(crate) avg_error: f32,
    pub(crate) value: Option<f32>,
    pub(crate) activation: f32,
}

/// Pre-built target map for efficient sample building across multiple sources.
/// This avoids rebuilding the HashMap for each source when analysing a single target.
pub(crate) struct TargetMap {
    pub(crate) map: HashMap<u32, TargetData>,
}

impl TargetMap {
    /// Build a target map from target records.
    /// This should be done ONCE per focus neuron, then reused for all sources.
    pub(crate) fn from_records(target_records: &[DiscoverRecord]) -> Self {
        let mut map: HashMap<u32, TargetData> = HashMap::with_capacity(target_records.len());
        for record in target_records {
            if record.errors.is_empty() || !record.activation.is_finite() {
                continue;
            }
            let mut sum = 0.0;
            let mut count = 0;
            for error in &record.errors {
                if error.is_finite() {
                    sum += *error;
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            map.insert(
                record.obs_index,
                TargetData {
                    avg_error: sum / count as f32,
                    value: record.value,
                    activation: record.activation,
                },
            );
        }
        Self { map }
    }

    /// Build samples by matching source records against this pre-built target map.
    /// This is much faster than build_samples() when processing multiple sources
    /// against the same target.
    pub(crate) fn build_samples_from(&self, from_records: &[DiscoverRecord]) -> Vec<HelpfulSample> {
        if self.map.is_empty() || from_records.is_empty() {
            return Vec::new();
        }

        let mut samples = Vec::with_capacity(from_records.len().min(self.map.len()));
        for record in from_records {
            if let Some(target) = self.map.get(&record.obs_index) {
                if record.activation.is_finite() && target.avg_error.is_finite() {
                    samples.push(HelpfulSample {
                        activation: record.activation,
                        avg_error: target.avg_error,
                        target_value: target.value,
                        target_activation: Some(target.activation),
                    });
                }
            }
        }

        samples
    }

    /// Build samples for multiple sources against this target map in a single pass.
    /// This is an optimisation for sources that share the same obs_indices.
    ///
    /// Instead of calling `build_samples_from` N times for N sources with identical
    /// obs_indices, we iterate through the target data once and build all sample
    /// vectors simultaneously.
    ///
    /// Returns a Vec of sample vectors, one per source, in the same order as
    /// `source_records`.
    ///
    /// # Issue #221: Sample Locality Optimisation
    ///
    /// When multiple sources share the same obs_indices (common for input neurons
    /// recorded together), this method reduces redundant target map lookups from
    /// O(sources × samples) to O(samples).
    pub(crate) fn build_samples_for_group(
        &self,
        source_records: &[(&str, &[DiscoverRecord])],
    ) -> Vec<Vec<HelpfulSample>> {
        if self.map.is_empty() || source_records.is_empty() {
            return vec![Vec::new(); source_records.len()];
        }

        // Build activation maps for each source: obs_index -> activation
        let activation_maps: Vec<HashMap<u32, f32>> = source_records
            .iter()
            .map(|(_, records)| {
                let mut map = HashMap::with_capacity(records.len());
                for record in *records {
                    if record.activation.is_finite() {
                        map.insert(record.obs_index, record.activation);
                    }
                }
                map
            })
            .collect();

        // Pre-allocate result vectors
        let mut results: Vec<Vec<HelpfulSample>> = source_records
            .iter()
            .map(|(_, records)| Vec::with_capacity(records.len().min(self.map.len())))
            .collect();

        // Single pass through target data, building samples for all sources
        for (&obs_index, target) in &self.map {
            if !target.avg_error.is_finite() {
                continue;
            }

            for (source_idx, activation_map) in activation_maps.iter().enumerate() {
                if let Some(&activation) = activation_map.get(&obs_index) {
                    results[source_idx].push(HelpfulSample {
                        activation,
                        avg_error: target.avg_error,
                        target_value: target.value,
                        target_activation: Some(target.activation),
                    });
                }
            }
        }

        results
    }
}

// =============================================================================
// Focus Target Filtering
// =============================================================================

/// Result of filtering focus targets for add-neuron analysis.
///
/// This is split out to keep the rules testable without needing a GPU (the main analysis path
/// asserts GPU availability).
#[derive(Debug, Default)]
pub(crate) struct FocusTargetFilterResult {
    pub(crate) focus_order: Vec<String>,
    pub(crate) skipped_hidden: Vec<String>,
    pub(crate) skipped_input: Vec<String>,
    pub(crate) skipped_constant: Vec<String>,
    pub(crate) threshold_targets: Vec<String>,
}

/// Filter and classify focus targets for add-neuron analysis.
///
/// Notes (Dec 2025):
/// - By default we allow both output and hidden focus targets (hidden will be impact-discounted later).
/// - When `output_only_targets` is enabled, hidden (and unknown treated-as-hidden) targets are filtered out.
/// - STEP/BIPOLAR targets are tracked in `threshold_targets` for visibility (and potential future branching).
pub(crate) fn filter_focus_targets_for_neuron_analysis(
    unique_focus: &[&String],
    neuron_type_map: &HashMap<String, String>,
    neuron_squash_map: &HashMap<String, String>,
    output_only_targets: bool,
) -> FocusTargetFilterResult {
    let mut result = FocusTargetFilterResult::default();

    // Helper: record STEP/BIPOLAR targets consistently across output/hidden/unknown.
    let mut record_threshold_target = |uuid: &String| {
        if let Some(squash) = neuron_squash_map.get(uuid) {
            if is_threshold_activation(squash) {
                result.threshold_targets.push(uuid.clone());
            }
        }
    };

    result.focus_order = unique_focus
        .iter()
        .filter_map(|uuid| {
            // By default we analyse both output and hidden focus targets (hidden will be discounted).
            // If `output_only_targets` is set, hidden targets are filtered.
            let neuron_type = neuron_type_map.get(*uuid).map(|s| s.as_str());
            match neuron_type {
                Some("output") => {
                    record_threshold_target(uuid);
                    Some((*uuid).clone())
                }
                Some("hidden") => {
                    if output_only_targets {
                        result.skipped_hidden.push((*uuid).clone());
                        None
                    } else {
                        record_threshold_target(uuid);
                        Some((*uuid).clone())
                    }
                }
                Some("input") => {
                    // Input neurons are observation sources, not computation nodes.
                    result.skipped_input.push((*uuid).clone());
                    None
                }
                Some("constant") => {
                    // Constant neurons don't receive inputs - filter them out.
                    result.skipped_constant.push((*uuid).clone());
                    None
                }
                Some(unknown_type) => {
                    // Unknown type - treat as hidden.
                    eprintln!(
                        "[NEAT-AI-Discovery] Warning: Unknown neuron type '{unknown_type}' for UUID '{uuid}'. \
                        Treating as hidden neuron."
                    );
                    if output_only_targets {
                        result.skipped_hidden.push((*uuid).clone());
                        None
                    } else {
                        record_threshold_target(uuid);
                        Some((*uuid).clone())
                    }
                }
                None => {
                    // Unknown UUID - this is likely a bug, skip it.
                    eprintln!(
                        "[NEAT-AI-Discovery] Warning: Unknown neuron UUID '{uuid}' in focus list \
                        (not found in creature). Skipping."
                    );
                    None
                }
            }
        })
        .collect();

    result
}

// =============================================================================
// Focus Validation
// =============================================================================

/// Validate that focus neurons list is not empty and contains no duplicates.
///
/// Rust side refuses to run if `focus_neurons` is empty or contains duplicates.
/// Controllers **must** validate and de-duplicate targets before calling into FFI
/// so any upstream issues are surfaced promptly.
pub(crate) fn require_unique_focus<'a>(
    focus_neurons: &'a [String],
    context: &str,
) -> anyhow::Result<Vec<&'a String>> {
    if focus_neurons.is_empty() {
        return Err(anyhow::anyhow!(
            "{context} needs at least one focus neuron. The Deno controller supplied an empty `focus_neurons` array, so there is nothing to analyse. Please fix the upstream request and retry after setting `NEAT_AI_DISCOVERY_VERBOSE=1` if you need extra logging."
        ));
    }

    let mut seen_targets: HashSet<&str> = HashSet::new();
    let mut unique_focus: Vec<&String> = Vec::new();
    let mut duplicates: Vec<String> = Vec::new();

    for target_uuid in focus_neurons {
        if seen_targets.insert(target_uuid.as_str()) {
            unique_focus.push(target_uuid);
        } else {
            duplicates.push(target_uuid.clone());
        }
    }

    if !duplicates.is_empty() {
        duplicates.sort();
        duplicates.dedup();
        let joined = duplicates.join(", ");
        return Err(anyhow::anyhow!(
            "{context} received duplicate focus neurons ({joined}). Each target must be unique so we can map diagnostics back to the Deno request. We are refusing to continue so the upstream behaviour can be corrected."
        ));
    }

    Ok(unique_focus)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rejection_reason_display() {
        assert_eq!(
            format!("{}", RejectionReason::NoSamples),
            "no overlapping discovery samples"
        );
        assert_eq!(
            format!("{}", RejectionReason::ZeroImprovement),
            "no consistent improvement in GPU stats"
        );
        assert_eq!(
            format!("{}", RejectionReason::BelowThreshold),
            "expected improvement below threshold"
        );
    }

    #[test]
    fn test_target_diagnostics_tracks_candidate() {
        let targets = ["target-1".to_string()];
        let target_refs: Vec<&String> = targets.iter().collect();
        let mut diagnostics = TargetDiagnostics::new_for_tests(
            &target_refs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        );

        diagnostics.set_target_record_count("target-1", 100);
        diagnostics.set_total_eligible_sources("target-1", 50);
        diagnostics.record_candidate_attempt("target-1", true);
        diagnostics.mark_candidate_selected("target-1");

        let entry = diagnostics.entry_for("target-1").unwrap();
        assert_eq!(entry.target_record_count, 100);
        assert_eq!(entry.total_eligible_sources, 50);
        assert_eq!(entry.evaluated_candidates, 1);
        assert!(entry.had_candidate);
    }

    #[test]
    fn test_neuron_diagnostics_tracks_filtered() {
        let targets = ["hidden-1".to_string(), "output-1".to_string()];
        let target_refs: Vec<&String> = targets.iter().collect();
        let mut diagnostics = NeuronDiagnostics::new_for_tests(
            &target_refs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        );

        diagnostics.mark_hidden_filtered("hidden-1");
        diagnostics.mark_candidate_selected("output-1");

        let hidden_entry = diagnostics.entry_for("hidden-1").unwrap();
        assert!(hidden_entry.hidden_filtered);
        assert!(!hidden_entry.had_candidate);

        let output_entry = diagnostics.entry_for("output-1").unwrap();
        assert!(!output_entry.hidden_filtered);
        assert!(output_entry.had_candidate);
    }

    #[test]
    fn test_require_unique_focus_empty() {
        let result = require_unique_focus(&[], "test");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at least one focus neuron"));
    }

    #[test]
    fn test_require_unique_focus_duplicates() {
        let focus = vec![
            "uuid-1".to_string(),
            "uuid-2".to_string(),
            "uuid-1".to_string(),
        ];
        let result = require_unique_focus(&focus, "test");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("duplicate"));
    }

    #[test]
    fn test_require_unique_focus_valid() {
        let focus = vec!["uuid-1".to_string(), "uuid-2".to_string()];
        let result = require_unique_focus(&focus, "test").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(*result[0], "uuid-1");
        assert_eq!(*result[1], "uuid-2");
    }

    #[test]
    fn test_filter_focus_targets_output_only() {
        let focus = [
            "output-1".to_string(),
            "hidden-1".to_string(),
            "input-0".to_string(),
        ];
        let focus_refs: Vec<&String> = focus.iter().collect();

        let mut type_map = HashMap::new();
        type_map.insert("output-1".to_string(), "output".to_string());
        type_map.insert("hidden-1".to_string(), "hidden".to_string());
        type_map.insert("input-0".to_string(), "input".to_string());

        let squash_map = HashMap::new();

        let result =
            filter_focus_targets_for_neuron_analysis(&focus_refs, &type_map, &squash_map, true);

        assert_eq!(result.focus_order, vec!["output-1"]);
        assert_eq!(result.skipped_hidden, vec!["hidden-1"]);
        assert_eq!(result.skipped_input, vec!["input-0"]);
    }

    #[test]
    fn test_filter_focus_targets_allow_hidden() {
        let focus = ["output-1".to_string(), "hidden-1".to_string()];
        let focus_refs: Vec<&String> = focus.iter().collect();

        let mut type_map = HashMap::new();
        type_map.insert("output-1".to_string(), "output".to_string());
        type_map.insert("hidden-1".to_string(), "hidden".to_string());

        let squash_map = HashMap::new();

        let result =
            filter_focus_targets_for_neuron_analysis(&focus_refs, &type_map, &squash_map, false);

        assert_eq!(result.focus_order, vec!["output-1", "hidden-1"]);
        assert!(result.skipped_hidden.is_empty());
    }

    #[test]
    fn test_target_map_from_records() {
        let records = vec![
            DiscoverRecord {
                neuron_uuid: "target".to_string(),
                obs_index: 0,
                activation: 0.5,
                errors: vec![0.1, 0.2],
                value: Some(0.3),
            },
            DiscoverRecord {
                neuron_uuid: "target".to_string(),
                obs_index: 1,
                activation: 0.7,
                errors: vec![0.3],
                value: None,
            },
        ];

        let target_map = TargetMap::from_records(&records);

        assert_eq!(target_map.map.len(), 2);
        let data_0 = target_map.map.get(&0).unwrap();
        assert!((data_0.avg_error - 0.15).abs() < 0.01); // (0.1 + 0.2) / 2
        assert_eq!(data_0.value, Some(0.3));
    }

    #[test]
    fn test_target_map_build_samples() {
        let target_records = vec![DiscoverRecord {
            neuron_uuid: "target".to_string(),
            obs_index: 0,
            activation: 0.5,
            errors: vec![0.1],
            value: Some(0.3),
        }];

        let source_records = vec![DiscoverRecord {
            neuron_uuid: "source".to_string(),
            obs_index: 0,
            activation: 0.8,
            errors: vec![],
            value: None,
        }];

        let target_map = TargetMap::from_records(&target_records);
        let samples = target_map.build_samples_from(&source_records);

        assert_eq!(samples.len(), 1);
        assert!((samples[0].activation - 0.8).abs() < 0.01);
        assert!((samples[0].avg_error - 0.1).abs() < 0.01);
    }
}
