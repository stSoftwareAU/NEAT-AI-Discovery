//! Neuron rejection tracking and diagnostic reporting.
//!
//! Tracks why neuron candidates were rejected during add-neuron analysis, including:
//! - Pre-analysis filters (input, hidden, constant neurons)
//! - Record loading failures
//! - No overlapping samples between source and target
//!
//! Uses `DashMap` for lock-free concurrent access from parallel analysis threads.

use dashmap::DashMap;

use crate::analysis::shared::{
    NeuronNoCandidateDetail, NeuronNoCandidateReason, NeuronNoCandidateSummary,
};
use crate::analysis::utils::verbose_enabled;

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
#[derive(Clone)]
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
///
/// Uses `DashMap` internally for lock-free concurrent access (Issue #216).
/// All methods take `&self` instead of `&mut self` to allow concurrent updates
/// from multiple threads without external synchronisation.
pub(crate) struct NeuronDiagnostics {
    log_enabled: bool,
    /// Lock-free concurrent map for diagnostic entries.
    /// Each focus neuron is processed by a separate thread, and diagnostics
    /// are recorded without contention using DashMap's sharded internal structure.
    pub(crate) entries: DashMap<String, NeuronDiagnosticEntry>,
}

impl NeuronDiagnostics {
    pub(crate) fn new(targets: &[&String]) -> Self {
        let log_enabled = verbose_enabled();
        let entries = DashMap::new();
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
        let entries = DashMap::new();
        for target in targets {
            entries.insert((*target).to_string(), NeuronDiagnosticEntry::new(target));
        }
        Self {
            log_enabled: true,
            entries,
        }
    }

    pub(crate) fn set_target_record_count(&self, target_uuid: &str, count: usize) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.target_record_count = count;
        }
    }

    pub(crate) fn set_total_eligible_sources(&self, target_uuid: &str, count: u32) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.total_eligible_sources = count;
        }
    }

    pub(crate) fn record_load_failure(&self, target_uuid: &str) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.record_load_failures += 1;
        }
    }

    pub(crate) fn record_candidate_attempt(&self, target_uuid: &str, had_samples: bool) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.evaluated_sources += 1;
            if had_samples {
                entry.sources_with_samples += 1;
            }
        }
    }

    pub(crate) fn record_no_samples(&self, target_uuid: &str, source_uuid: &str) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(NeuronRejectionDetail {
                source_uuid: source_uuid.to_string(),
                orientation: None,
                sample_count: 0,
                expected_improvement: f32::NEG_INFINITY,
            });
        }
    }

    pub(crate) fn mark_candidate_selected(&self, target_uuid: &str) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.had_candidate = true;
        }
    }

    /// Mark a neuron as filtered out because it's a hidden neuron.
    /// Hidden neurons are not valid targets for add-neuron analysis because their
    /// backpropagated errors don't reliably translate to output error reduction.
    pub(crate) fn mark_hidden_filtered(&self, target_uuid: &str) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.hidden_filtered = true;
        }
    }

    /// Mark a neuron as filtered out because it's an input neuron.
    /// Input neurons are observation sources, not computation nodes - they have
    /// no activation function or error to reduce.
    pub(crate) fn mark_input_filtered(&self, target_uuid: &str) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.input_filtered = true;
        }
    }

    /// Mark a neuron as filtered out because it's a constant neuron.
    /// Constant neurons don't receive inputs - they always output a fixed value
    /// regardless of network state, so adding a connection to them has no effect.
    pub(crate) fn mark_constant_filtered(&self, target_uuid: &str) {
        if let Some(mut entry) = self.entries.get_mut(target_uuid) {
            entry.constant_filtered = true;
        }
    }

    pub(crate) fn emit_logs(&self) {
        if !self.log_enabled {
            return;
        }

        for entry_ref in self.entries.iter() {
            let entry = entry_ref.value();
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
    pub(crate) fn entry_for(&self, target_uuid: &str) -> Option<NeuronDiagnosticEntry> {
        self.entries.get(target_uuid).map(|r| r.value().clone())
    }

    pub(crate) fn no_candidate_summaries(&self) -> Vec<NeuronNoCandidateSummary> {
        self.entries
            .iter()
            // Include entries that never had a candidate
            .filter(|entry_ref| !entry_ref.value().had_candidate)
            .map(|entry_ref| {
                let entry = entry_ref.value();
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
