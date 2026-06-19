//! Fail-fast gate for partial record phases with insufficient Parquet coverage
//! (Issue #1444).
//!
//! When the discovery record phase times out, many of the selected focus
//! neurons can end up with **zero** rows in the Parquet file. The downstream
//! synapse/neuron analysis then correctly returns nothing (every target reports
//! `no_target_records`, Issue #1101) — but only after spending the *entire*
//! analysis budget (~15 min of combined GPU analysis on a large dataset). The
//! operator pays full wall-clock cost and gets no actionable signal that
//! **recording**, not search, failed.
//!
//! This module detects that scenario cheaply, *before* any GPU work, with an
//! in-memory record-count scan of the already-loaded cache. When the fraction
//! of focus neurons with zero rows reaches the configured threshold the
//! orchestrator skips analysis entirely and surfaces `insufficient_recording`
//! as the dominant rejection reason, carrying a structured
//! [`InsufficientRecordingDiagnostic`] in the performance-summary metadata.

use serde::{Deserialize, Serialize};

use super::cache::RecordCache;
use super::diagnostics::rejection_reasons::{REJECTION_INSUFFICIENT_RECORDING, top_level_summary};
use super::shared::{
    AnalyzeNeuronsResult, AnalyzeSynapsesResult, NeuronAnalysisMetadata, SynapseAnalysisMetadata,
};

/// Parquet recording coverage across the selected focus neurons (Issue #1444).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FocusRecordingCoverage {
    /// Number of focus neurons assessed.
    pub focus_neurons_total: usize,
    /// Focus neurons that resolved to zero Parquet rows.
    pub focus_neurons_with_zero_rows: usize,
    /// Total Parquet rows summed across the assessed focus neurons.
    pub focus_neuron_records_total: usize,
}

/// Structured fail-fast diagnostic attached to the analysis performance summary
/// when recording coverage is insufficient (Issue #1444).
///
/// Serialised in camelCase on both `synapseMetadata` and `neuronMetadata` so
/// the NEAT-AI host can distinguish a *recording* failure (this) from genuine
/// search exhaustion without scraping logs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InsufficientRecordingDiagnostic {
    /// Selected focus neurons assessed for coverage.
    pub focus_neurons_total: usize,
    /// Focus neurons that had zero Parquet rows.
    pub focus_neurons_with_zero_rows: usize,
    /// Total Parquet rows across the assessed focus neurons.
    pub focus_neuron_records_total: usize,
    /// Total records materialised in the parquet cache for this pass — the
    /// records the record phase actually produced.
    pub records_processed: usize,
    /// The zero-row fraction threshold that triggered the gate.
    pub threshold_fraction: f64,
}

/// Assess how many of the selected focus neurons have zero Parquet rows.
///
/// Reads each focus neuron's records from the (already-loaded) cache. A focus
/// neuron is counted as missing only when the cache lookup succeeds and returns
/// an empty record set — a transient load error is treated as "unknown" and
/// never counted as missing, so the gate cannot false-trigger on an I/O error.
#[must_use]
pub fn assess_focus_recording_coverage(
    cache: &RecordCache,
    focus_neurons: &[String],
) -> FocusRecordingCoverage {
    let mut zero = 0usize;
    let mut total_records = 0usize;
    for uuid in focus_neurons {
        if let Ok(records) = cache.get(uuid) {
            let n = records.len();
            total_records += n;
            if n == 0 {
                zero += 1;
            }
        }
    }
    FocusRecordingCoverage {
        focus_neurons_total: focus_neurons.len(),
        focus_neurons_with_zero_rows: zero,
        focus_neuron_records_total: total_records,
    }
}

/// Whether the coverage is insufficient at the given zero-row fraction
/// threshold (Issue #1444).
///
/// Returns `true` only when at least one focus neuron is missing **and** the
/// missing fraction reaches `fraction_threshold`. Always `false` when there are
/// no focus neurons to assess.
#[must_use]
pub fn is_insufficient_recording(
    coverage: &FocusRecordingCoverage,
    fraction_threshold: f64,
) -> bool {
    if coverage.focus_neurons_total == 0 || coverage.focus_neurons_with_zero_rows == 0 {
        return false;
    }
    #[allow(clippy::cast_precision_loss)] // Counts are small; precision loss is irrelevant.
    let missing_fraction =
        coverage.focus_neurons_with_zero_rows as f64 / coverage.focus_neurons_total as f64;
    missing_fraction >= fraction_threshold
}

/// Build the `(rejection_breakdown, top_level_summary)` pair for a skipped pass.
fn skip_breakdown_and_summary(
    diagnostic: &InsufficientRecordingDiagnostic,
) -> (super::diagnostics::RejectionBreakdown, Option<String>) {
    let mut breakdown = super::diagnostics::RejectionBreakdown::new();
    let missing = u32::try_from(diagnostic.focus_neurons_with_zero_rows).unwrap_or(u32::MAX);
    breakdown.record_many(REJECTION_INSUFFICIENT_RECORDING, missing);
    let denom = u32::try_from(diagnostic.focus_neurons_total).unwrap_or(u32::MAX);
    let summary = top_level_summary(&breakdown, Some(denom));
    (breakdown, summary)
}

/// Construct an empty synapse result whose metadata flags the insufficient
/// recording as the dominant rejection reason (Issue #1444).
#[must_use]
pub fn synapse_skip_result(
    diagnostic: &InsufficientRecordingDiagnostic,
    focus_neurons: &[String],
) -> AnalyzeSynapsesResult {
    let (rejection_breakdown, top_level_summary) = skip_breakdown_and_summary(diagnostic);
    let metadata = SynapseAnalysisMetadata {
        // The pass "completed" every focus neuron in the sense that each was
        // assessed and found to have no usable records — so `starved`
        // (timed_out && completed < total) stays false; this was not a deadline
        // starvation but a recording failure.
        total_focus_neurons: focus_neurons.len(),
        completed_focus_neurons: focus_neurons.len(),
        rejection_breakdown,
        top_level_summary,
        insufficient_recording: Some(diagnostic.clone()),
        ..SynapseAnalysisMetadata::default()
    };
    AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata,
    }
}

/// Construct an empty neuron result whose metadata flags the insufficient
/// recording as the dominant rejection reason (Issue #1444).
#[must_use]
pub fn neuron_skip_result(
    diagnostic: &InsufficientRecordingDiagnostic,
    focus_neurons: &[String],
) -> AnalyzeNeuronsResult {
    let (rejection_breakdown, top_level_summary) = skip_breakdown_and_summary(diagnostic);
    let metadata = NeuronAnalysisMetadata {
        total_focus_neurons: focus_neurons.len(),
        completed_focus_neurons: focus_neurons.len(),
        rejection_breakdown,
        top_level_summary,
        insufficient_recording: Some(diagnostic.clone()),
        ..NeuronAnalysisMetadata::default()
    };
    AnalyzeNeuronsResult {
        helpful_neurons: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagnostic(total: usize, zero: usize) -> InsufficientRecordingDiagnostic {
        InsufficientRecordingDiagnostic {
            focus_neurons_total: total,
            focus_neurons_with_zero_rows: zero,
            focus_neuron_records_total: 0,
            records_processed: 1766,
            threshold_fraction: 1.0,
        }
    }

    #[test]
    fn no_focus_neurons_is_never_insufficient() {
        let coverage = FocusRecordingCoverage::default();
        assert!(!is_insufficient_recording(&coverage, 1.0));
    }

    #[test]
    fn all_missing_triggers_at_default_threshold() {
        let coverage = FocusRecordingCoverage {
            focus_neurons_total: 6,
            focus_neurons_with_zero_rows: 6,
            focus_neuron_records_total: 0,
        };
        assert!(is_insufficient_recording(&coverage, 1.0));
    }

    #[test]
    fn partial_missing_below_threshold_does_not_trigger() {
        let coverage = FocusRecordingCoverage {
            focus_neurons_total: 6,
            focus_neurons_with_zero_rows: 3,
            focus_neuron_records_total: 120,
        };
        // Default threshold (all must be missing) is not reached.
        assert!(!is_insufficient_recording(&coverage, 1.0));
        // A 0.5 threshold is reached at 3/6.
        assert!(is_insufficient_recording(&coverage, 0.5));
    }

    #[test]
    fn some_present_some_missing_respects_fraction() {
        let coverage = FocusRecordingCoverage {
            focus_neurons_total: 4,
            focus_neurons_with_zero_rows: 1,
            focus_neuron_records_total: 90,
        };
        assert!(!is_insufficient_recording(&coverage, 0.5));
        assert!(is_insufficient_recording(&coverage, 0.25));
    }

    #[test]
    fn skip_result_marks_insufficient_recording_as_dominant() {
        let diag = diagnostic(6, 6);
        let focus: Vec<String> = (0..6).map(|i| format!("output-{i}")).collect();
        let syn = synapse_skip_result(&diag, &focus);
        assert!(syn.helpful_synapses.is_empty());
        assert_eq!(syn.metadata.total_focus_neurons, 6);
        assert_eq!(
            syn.metadata.rejection_breakdown.dominant_reason(),
            Some((REJECTION_INSUFFICIENT_RECORDING, 6))
        );
        let summary = syn.metadata.top_level_summary.expect("summary present");
        assert!(
            summary.contains("insufficient Parquet recording"),
            "summary should name the reason, got: {summary}"
        );
        assert_eq!(
            syn.metadata.insufficient_recording.as_ref(),
            Some(&diag),
            "diagnostic should be attached to metadata"
        );

        let neu = neuron_skip_result(&diag, &focus);
        assert!(neu.helpful_neurons.is_empty());
        assert_eq!(
            neu.metadata.rejection_breakdown.dominant_reason(),
            Some((REJECTION_INSUFFICIENT_RECORDING, 6))
        );
        assert_eq!(neu.metadata.insufficient_recording.as_ref(), Some(&diag));
    }
}
