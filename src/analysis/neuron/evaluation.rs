//! Neuron candidate GPU evaluation — ReLU split evaluation and activation
//! spec batched evaluation for neuron candidates.
//!
//! Extracted from neuron.rs as part of issue #598.

use crate::CandidateNeuronJson;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::analysis::diagnostics::NeuronDiagnostics;
use crate::analysis::gpu::GpuWorkQueue;
use crate::analysis::samples::{EPSILON, HelpfulSample, compute_source_variance_discount};
use crate::analysis::shared::TimingScope;
use crate::analysis::utils::{deadline_passed, lock_or_bail, verbose_enabled};

use crate::analysis::synapse::{
    evaluate_all_activation_specs_batched, evaluate_relu_candidates_split, upsert_candidate,
};

/// Work result from sample building phase, containing samples for a single
/// source neuron.
pub(crate) struct NeuronWorkResult {
    pub source_uuid: String,
    pub samples: Vec<HelpfulSample>,
}

/// Evaluate neuron candidates for all sources with samples against a single
/// target neuron using GPU shaders.
///
/// This handles both ReLU split evaluation and batched activation spec
/// evaluation, applying source variance discounting.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_neuron_candidates(
    work_results: &[NeuronWorkResult],
    target_uuid: &str,
    gpu: &GpuWorkQueue,
    neuron_squash_map_arc: &Arc<HashMap<String, String>>,
    timing_collector: &Arc<super::super::shared::TimingCollector>,
    deadline: &Option<SystemTime>,
    analysis_timed_out: &Arc<Mutex<bool>>,
    diagnostics: &Arc<NeuronDiagnostics>,
    helpful_map: &Arc<Mutex<HashMap<u64, CandidateNeuronJson>>>,
    threshold: f32,
) -> Result<()> {
    for result in work_results {
        // Check deadline before each evaluation batch
        if deadline_passed(deadline) {
            *lock_or_bail(analysis_timed_out, "analysis_timed_out")? = true;
            break;
        }

        if result.samples.is_empty() {
            continue;
        }

        // Get target_squash for accurate HARD_TANH modelling
        let target_squash = neuron_squash_map_arc.get(target_uuid).map(|s| s.as_str());

        // Issue #130 (v0.2.2): Compute source variance discount.
        // If source activation has low variance, predictions are unreliable.
        let source_variance_discount = compute_source_variance_discount(&result.samples);
        if source_variance_discount <= EPSILON {
            // Source is constant - skip evaluation entirely
            continue;
        }

        // ReLU evaluation: split by TARGET neuron's error sign.
        evaluate_relu_split(
            gpu,
            &result.source_uuid,
            target_uuid,
            &result.samples,
            threshold,
            target_squash,
            source_variance_discount,
            timing_collector,
            diagnostics,
            helpful_map,
        )?;

        // Issue #201: Evaluate all activation specs in a single batched GPU call
        evaluate_activation_specs(
            gpu,
            &result.source_uuid,
            target_uuid,
            &result.samples,
            threshold,
            target_squash,
            source_variance_discount,
            timing_collector,
            diagnostics,
            helpful_map,
        )?;
    }
    Ok(())
}

/// Evaluate ReLU candidates with positive/negative error split.
#[allow(clippy::too_many_arguments)]
fn evaluate_relu_split(
    gpu: &GpuWorkQueue,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    target_squash: Option<&str>,
    source_variance_discount: f32,
    timing_collector: &Arc<super::super::shared::TimingCollector>,
    diagnostics: &Arc<NeuronDiagnostics>,
    helpful_map: &Arc<Mutex<HashMap<u64, CandidateNeuronJson>>>,
) -> Result<()> {
    let split_result = {
        let _timing = TimingScope::shader(timing_collector, "relu");
        evaluate_relu_candidates_split(
            gpu,
            source_uuid,
            target_uuid,
            samples,
            threshold,
            target_squash,
        )?
    };

    if let Some(mut candidate) = split_result.positive_error_candidate {
        // Issue #130: Apply source variance discount
        candidate.expected_creature_error_reduction *= source_variance_discount;
        candidate.expected_creature_score_gain *= source_variance_discount;

        if verbose_enabled() {
            tracing::trace!(
                direction = "push UP",
                source_uuid = %source_uuid,
                target_uuid = %target_uuid,
                improvement_pct = format_args!("{:.2}", candidate.expected_creature_score_gain * 100.0),
                variance_discount = format_args!("{:.2}", source_variance_discount),
                "ReLU candidate identified"
            );
        }
        diagnostics.mark_candidate_selected(target_uuid);
        let mut map = lock_or_bail(helpful_map, "helpful_map")?;
        upsert_candidate(&mut map, candidate);
    }

    if let Some(mut candidate) = split_result.negative_error_candidate {
        // Issue #130: Apply source variance discount
        candidate.expected_creature_error_reduction *= source_variance_discount;
        candidate.expected_creature_score_gain *= source_variance_discount;

        if verbose_enabled() {
            tracing::trace!(
                direction = "push DOWN",
                source_uuid = %source_uuid,
                target_uuid = %target_uuid,
                improvement_pct = format_args!("{:.2}", candidate.expected_creature_score_gain * 100.0),
                variance_discount = format_args!("{:.2}", source_variance_discount),
                "ReLU candidate identified"
            );
        }
        diagnostics.mark_candidate_selected(target_uuid);
        let mut map = lock_or_bail(helpful_map, "helpful_map")?;
        upsert_candidate(&mut map, candidate);
    }

    Ok(())
}

/// Evaluate all activation function specs in a single batched GPU call.
#[allow(clippy::too_many_arguments)]
fn evaluate_activation_specs(
    gpu: &GpuWorkQueue,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    target_squash: Option<&str>,
    source_variance_discount: f32,
    timing_collector: &Arc<super::super::shared::TimingCollector>,
    diagnostics: &Arc<NeuronDiagnostics>,
    helpful_map: &Arc<Mutex<HashMap<u64, CandidateNeuronJson>>>,
) -> Result<()> {
    let batched_candidates = {
        let _timing = TimingScope::shader(timing_collector, "activation");
        evaluate_all_activation_specs_batched(
            gpu,
            source_uuid,
            target_uuid,
            samples,
            threshold,
            target_squash,
        )?
    };

    for mut candidate in batched_candidates {
        // Issue #130: Apply source variance discount
        candidate.expected_creature_error_reduction *= source_variance_discount;
        candidate.expected_creature_score_gain *= source_variance_discount;

        diagnostics.mark_candidate_selected(target_uuid);
        let mut map = lock_or_bail(helpful_map, "helpful_map")?;
        upsert_candidate(&mut map, candidate);
    }

    Ok(())
}
