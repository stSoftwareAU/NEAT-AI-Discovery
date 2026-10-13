//! GPU-free harness for the neuron target-load diagnostics (Issue #2352).
//!
//! `analyze_neurons_with_cache_and_gpu_queue` needs a GPU before it loads any
//! target's records, so the "target records failed to load" path cannot be
//! exercised from a GPU-less test. This harness runs the same preparation,
//! filter-marking (`mark_skipped_filters`) and target-load
//! (`load_target_records`) steps as production, sequentially and without a GPU,
//! then reports the resulting diagnostics.
//!
//! Regression-test scaffolding only, gated behind the `regression-harness`
//! feature.

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::AnalyzeNeuronsInput;
use crate::analysis::cache::RecordCache;
use crate::analysis::diagnostics::NeuronDiagnostics;
use crate::analysis::neuron::preparation::prepare_neuron_analysis;
use crate::analysis::neuron::{load_target_records, mark_skipped_filters};
use crate::analysis::shared::NeuronNoCandidateSummary;
use crate::analysis::synapse::build_ordered_neurons;

/// Diagnostics observed after the GPU-free target-load pass.
#[derive(Debug)]
pub struct NeuronTargetLoadOutcome {
    /// The `no_candidate_summaries` for every focus target.
    pub summaries: Vec<NeuronNoCandidateSummary>,
    /// `(target_uuid, record_load_failures)` for every focus target.
    pub load_failures: Vec<(String, u32)>,
    /// Targets whose own records failed to load.
    pub target_load_failed: Vec<String>,
}

/// Run preparation, filter marking and the target load for each focus target,
/// without a GPU, and return the resulting diagnostics.
///
/// # Errors
///
/// Returns an error if neuron preparation fails.
pub fn run_target_load_pass(
    input: &AnalyzeNeuronsInput,
    cache: &Arc<RecordCache>,
) -> Result<NeuronTargetLoadOutcome> {
    let ordered_neurons = build_ordered_neurons(&input.creature);
    let prep = prepare_neuron_analysis(input, &ordered_neurons, cache)
        .context("failed to prepare neuron analysis")?;

    let diagnostics = NeuronDiagnostics::new(&prep.unique_focus);
    mark_skipped_filters(&prep, &diagnostics);

    for target_uuid in &prep.focus_order {
        if let Some(records) = load_target_records(cache, &diagnostics, target_uuid) {
            diagnostics.set_target_record_count(target_uuid, records.len());
        }
    }

    let mut load_failures = Vec::new();
    let mut target_load_failed = Vec::new();
    for entry_ref in &diagnostics.entries {
        let entry = entry_ref.value();
        load_failures.push((entry.target_uuid.clone(), entry.record_load_failures));
        if entry.target_load_failed {
            target_load_failed.push(entry.target_uuid.clone());
        }
    }

    Ok(NeuronTargetLoadOutcome {
        summaries: diagnostics.no_candidate_summaries(),
        load_failures,
        target_load_failed,
    })
}
