//! Signalled partial results for a wedged GPU (Issue #1931).
//!
//! The circuit breaker (Issue #1930) stops GPU work; this module decides what
//! the analyses return with the rest of the run's budget. Every GPU-queue
//! construction site used to treat "no GPU queue" as a hard error, so a tripped
//! breaker propagated a raw error out of `analyze_all`, discarded whatever
//! CPU-side state existed, and read to the host as one more failed attempt.
//!
//! Instead the analyses **skip** the GPU work, finish the CPU-side accounting
//! and exit normally with the wedged signal set — mirroring the cancellation
//! path, which already returns an `AnalyzeAllResult` with `cancelled: true`
//! rather than an error.
//!
//! There is **no CPU fallback** for these analyses (Issue #1419 removed that
//! false claim). A skipped pass is genuinely empty, and the signal is what
//! stops it being reported as a zero-candidate success.

use super::diagnostics::RejectionBreakdown;
use super::diagnostics::rejection_reasons::{REJECTION_GPU_WEDGED, top_level_summary};
use super::shared::{
    AnalyzeNeuronsResult, AnalyzeSynapsesResult, NeuronAnalysisMetadata, SynapseAnalysisMetadata,
};

/// Build the `(rejection_breakdown, top_level_summary)` pair for a skipped pass.
///
/// One count per focus neuron: none of them was evaluated.
fn skip_breakdown_and_summary(focus_neurons: usize) -> (RejectionBreakdown, Option<String>) {
    let skipped = u32::try_from(focus_neurons).unwrap_or(u32::MAX);
    let mut breakdown = RejectionBreakdown::new();
    breakdown.record_many(REJECTION_GPU_WEDGED, skipped);
    let summary = top_level_summary(&breakdown, Some(skipped));
    (breakdown, summary)
}

/// Empty synapse result flagged as skipped because the GPU is wedged.
#[must_use]
pub fn synapse_wedged_result(focus_neurons: &[String]) -> AnalyzeSynapsesResult {
    let (rejection_breakdown, top_level_summary) = skip_breakdown_and_summary(focus_neurons.len());
    AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: SynapseAnalysisMetadata {
            // Nothing was completed: the pass never reached a GPU submission.
            total_focus_neurons: focus_neurons.len(),
            completed_focus_neurons: 0,
            rejection_breakdown,
            top_level_summary,
            gpu_wedged: true,
            ..SynapseAnalysisMetadata::default()
        },
    }
}

/// Empty neuron result flagged as skipped because the GPU is wedged.
#[must_use]
pub fn neuron_wedged_result(focus_neurons: &[String]) -> AnalyzeNeuronsResult {
    let (rejection_breakdown, top_level_summary) = skip_breakdown_and_summary(focus_neurons.len());
    AnalyzeNeuronsResult {
        helpful_neurons: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: NeuronAnalysisMetadata {
            total_focus_neurons: focus_neurons.len(),
            completed_focus_neurons: 0,
            rejection_breakdown,
            top_level_summary,
            gpu_wedged: true,
            ..NeuronAnalysisMetadata::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn focus() -> Vec<String> {
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    }

    fn wedged_count(breakdown: &RejectionBreakdown) -> u32 {
        breakdown
            .counts()
            .get(REJECTION_GPU_WEDGED)
            .copied()
            .unwrap_or(0)
    }

    #[test]
    fn the_synapse_result_is_empty_and_flagged() {
        let result = synapse_wedged_result(&focus());

        assert!(result.helpful_synapses.is_empty());
        assert!(result.harmful_synapses.is_empty());
        assert!(result.synapse_weight_updates.is_empty());
        assert!(result.coordinated_structural_candidates.is_empty());
        assert!(!result.gpu_used, "no GPU work was attempted");
        assert!(result.metadata.gpu_wedged, "the wedged signal must be set");
        assert_eq!(result.metadata.candidates_returned, 0);
        assert_eq!(result.metadata.completed_focus_neurons, 0);
        assert_eq!(result.metadata.total_focus_neurons, 3);
        assert_eq!(
            wedged_count(&result.metadata.rejection_breakdown),
            3,
            "one count per focus neuron that was never evaluated"
        );
        assert!(
            result
                .metadata
                .top_level_summary
                .as_deref()
                .is_some_and(|s| s.contains("GPU wedged")),
            "the summary must name the wedged GPU: {:?}",
            result.metadata.top_level_summary
        );
    }

    #[test]
    fn the_neuron_result_is_empty_and_flagged() {
        let result = neuron_wedged_result(&focus());

        assert!(result.helpful_neurons.is_empty());
        assert!(!result.gpu_used);
        assert!(result.metadata.gpu_wedged, "the wedged signal must be set");
        assert_eq!(result.metadata.candidates_returned, 0);
        assert_eq!(result.metadata.completed_focus_neurons, 0);
        assert_eq!(result.metadata.total_focus_neurons, 3);
        assert_eq!(wedged_count(&result.metadata.rejection_breakdown), 3);
    }

    /// A healthy zero-candidate pass must stay distinguishable — the default
    /// metadata is what every real analysis starts from.
    #[test]
    fn a_default_result_is_not_flagged_as_wedged() {
        assert!(!SynapseAnalysisMetadata::default().gpu_wedged);
        assert!(!NeuronAnalysisMetadata::default().gpu_wedged);
    }

    /// An empty focus set must not fabricate a rejection count.
    #[test]
    fn an_empty_focus_set_records_no_rejections() {
        let result = neuron_wedged_result(&[]);
        assert!(result.metadata.gpu_wedged, "still flagged as wedged");
        assert_eq!(wedged_count(&result.metadata.rejection_breakdown), 0);
    }
}
