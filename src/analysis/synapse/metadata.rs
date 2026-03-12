//! Atomic metadata and result merging for parallel synapse analysis
//!
//! Contains lock-free atomic metadata collected during the parallel per-target
//! analysis phase (Issue #744), and the single-threaded merge pass that
//! combines per-target results into a unified collection.

use crate::CandidateSynapseJson;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::target_analysis;

/// Lock-free atomic metadata collected during parallel target analysis (Issue #744).
///
/// These fields use atomics so they can be updated from rayon threads without
/// mutex contention. The Vec-based collections (helpful, harmful, coordinated,
/// error values) are collected per-thread and merged afterwards.
pub(super) struct AtomicMetadata {
    pub(super) target_value_seen: AtomicBool,
    pub(super) saturation_aware_used: AtomicBool,
    pub(super) seen_any_input_with_records: AtomicBool,
    pub(super) input_min_with_records: AtomicUsize,
    pub(super) input_max_with_records: AtomicUsize,
}

impl AtomicMetadata {
    pub(super) fn new() -> Self {
        Self {
            target_value_seen: AtomicBool::new(false),
            saturation_aware_used: AtomicBool::new(false),
            seen_any_input_with_records: AtomicBool::new(false),
            input_min_with_records: AtomicUsize::new(usize::MAX),
            input_max_with_records: AtomicUsize::new(0),
        }
    }

    /// Update atomic metadata from a single target's results (lock-free).
    pub(super) fn merge_atomic(&self, target_results: &target_analysis::TargetAnalysisResults) {
        if target_results.target_value_seen {
            self.target_value_seen.store(true, Ordering::Relaxed);
        }
        if target_results.saturation_aware_used {
            self.saturation_aware_used.store(true, Ordering::Relaxed);
        }
        if target_results.input_metadata.seen_any {
            self.seen_any_input_with_records
                .store(true, Ordering::Relaxed);
            let _ = self
                .input_min_with_records
                .fetch_min(target_results.input_metadata.min_index, Ordering::Relaxed);
            let _ = self
                .input_max_with_records
                .fetch_max(target_results.input_metadata.max_index, Ordering::Relaxed);
        }
    }
}

/// Merged results from all per-target analyses (Issue #744).
///
/// Built in a single-threaded pass after the parallel section completes,
/// avoiding mutex contention entirely during the parallel phase.
pub(super) struct MergedResults {
    pub(super) helpful_results: Vec<CandidateSynapseJson>,
    pub(super) harmful_results: Vec<CandidateSynapseJson>,
    pub(super) coordinated_structural_results: Vec<crate::CoordinatedStructuralCandidateJson>,
    pub(super) error_values_for_distribution: Vec<f32>,
    pub(super) analysis_timed_out: bool,
    pub(super) metadata: Arc<AtomicMetadata>,
}

impl MergedResults {
    /// Merge per-target results into a single collection (single-threaded, no contention).
    pub(super) fn from_per_target(
        per_target: Vec<Option<target_analysis::TargetAnalysisResults>>,
        timed_out: bool,
        metadata: Arc<AtomicMetadata>,
    ) -> Self {
        let mut helpful_results = Vec::new();
        let mut harmful_results = Vec::new();
        let mut coordinated_structural_results = Vec::new();
        let mut error_values_for_distribution = Vec::new();

        for result in per_target.into_iter().flatten() {
            helpful_results.extend(result.helpful);
            harmful_results.extend(result.harmful);
            coordinated_structural_results.extend(result.coordinated);
            error_values_for_distribution.extend(result.error_values);
        }

        Self {
            helpful_results,
            harmful_results,
            coordinated_structural_results,
            error_values_for_distribution,
            analysis_timed_out: timed_out,
            metadata,
        }
    }
}
