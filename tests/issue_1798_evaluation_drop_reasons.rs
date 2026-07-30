//! Integration tests for the evaluation drop-site accounting (Issue #1798).
//!
//! Three per-candidate `continue` sites dropped candidates without
//! incrementing anything, so the dropped candidates never reached the accept
//! gate *and* never appeared in the rejection breakdown — biasing
//! `candidate_starvation::classify` toward `ProposalRichOverRejected` exactly
//! when generation was the real bottleneck.
//!
//! These tests drive batches through the same guard predicates the drop sites
//! call (`drop_for_empty_samples` / `drop_for_zero_source_variance`), then fold
//! the per-batch counters into the surface metadata exactly as the neuron and
//! synapse orchestrators do. They cover the orchestration/accounting layer, not
//! the GPU evaluation pipeline (which needs a GPU adapter and a Parquet
//! fixture) — the same scope as `issue_1164_within_batch_failures.rs`.

use neat_ai_discovery::analysis::candidate_starvation::{
    StarvationClass, StarvationConfig, classify, signals_from_breakdown,
};
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    REJECTION_NO_SAMPLES, REJECTION_ZERO_SOURCE_VARIANCE,
};
use neat_ai_discovery::analysis::evaluation_drops::{
    EvaluationDropCounters, fold_evaluation_drops,
};
use neat_ai_discovery::analysis::samples::{HelpfulSample, compute_source_variance_discount};
use neat_ai_discovery::analysis::shared::{NeuronAnalysisMetadata, SynapseAnalysisMetadata};

/// A candidate as the evaluation loops see it: a source neuron's samples.
struct Candidate {
    samples: Vec<HelpfulSample>,
}

fn sample(activation: f32) -> HelpfulSample {
    HelpfulSample {
        activation,
        avg_error: 0.25,
        target_value: None,
        target_activation: None,
    }
}

/// A candidate whose sample building produced nothing.
fn empty_candidate() -> Candidate {
    Candidate {
        samples: Vec::new(),
    }
}

/// A candidate whose source activation never varies — sampled, but no signal.
fn constant_source_candidate() -> Candidate {
    Candidate {
        samples: vec![sample(0.75); 4],
    }
}

/// A candidate that should survive both guards.
fn usable_candidate(tag: f32) -> Candidate {
    Candidate {
        samples: vec![sample(-1.0 + tag), sample(0.0 + tag), sample(1.0 + tag)],
    }
}

/// Drive a batch through the neuron surface's two pre-evaluation guards in the
/// same order `analysis::neuron::evaluation::evaluate_neuron_candidates` does.
/// Returns the indices of the candidates that reached evaluation.
fn drive_neuron_batch(batch: &[Candidate], counters: &EvaluationDropCounters) -> Vec<usize> {
    let mut evaluated = Vec::new();
    for (index, candidate) in batch.iter().enumerate() {
        if counters.drop_for_empty_samples(&candidate.samples) {
            continue;
        }
        let discount = compute_source_variance_discount(&candidate.samples);
        if counters.drop_for_zero_source_variance(discount) {
            continue;
        }
        evaluated.push(index);
    }
    evaluated
}

/// The pre-#1798 drop behaviour, expressed with the bare predicates the guards
/// replaced. Used to prove the counters did not change which candidates drop.
fn drive_neuron_batch_uncounted(batch: &[Candidate]) -> Vec<usize> {
    const EPSILON: f32 = 1e-8;
    let mut evaluated = Vec::new();
    for (index, candidate) in batch.iter().enumerate() {
        if candidate.samples.is_empty() {
            continue;
        }
        if compute_source_variance_discount(&candidate.samples) <= EPSILON {
            continue;
        }
        evaluated.push(index);
    }
    evaluated
}

/// Drive a batch through the synapse surface's guard, mirroring
/// `synapse::target_analysis::evaluation::collect_and_process_helpful_results`.
fn drive_synapse_batch(batch: &[Candidate], counters: &EvaluationDropCounters) -> Vec<usize> {
    let mut evaluated = Vec::new();
    for (index, candidate) in batch.iter().enumerate() {
        if counters.drop_for_empty_samples(&candidate.samples) {
            continue;
        }
        evaluated.push(index);
    }
    evaluated
}

/// Acceptance: a neuron batch with E empty-sample candidates and V
/// constant-source candidates surfaces `no_samples: E` and
/// `zero_source_variance: V`.
#[test]
fn neuron_batch_surfaces_both_drop_reasons() {
    const E: u32 = 3;
    const V: u32 = 2;

    let mut batch = Vec::new();
    for _ in 0..E {
        batch.push(empty_candidate());
    }
    for _ in 0..V {
        batch.push(constant_source_candidate());
    }
    batch.push(usable_candidate(0.0));
    batch.push(usable_candidate(5.0));

    let counters = EvaluationDropCounters::new();
    let evaluated = drive_neuron_batch(&batch, &counters);
    assert_eq!(
        evaluated.len(),
        2,
        "only the two usable candidates should reach evaluation"
    );

    let mut metadata = NeuronAnalysisMetadata::default();
    let folded = fold_evaluation_drops(&counters, &mut metadata.rejection_breakdown);

    assert_eq!(folded, E + V);
    assert_eq!(
        metadata
            .rejection_breakdown
            .counts()
            .get(REJECTION_NO_SAMPLES),
        Some(&E),
        "every empty-sample candidate must be reported as no_samples"
    );
    assert_eq!(
        metadata
            .rejection_breakdown
            .counts()
            .get(REJECTION_ZERO_SOURCE_VARIANCE),
        Some(&V),
        "every constant-source candidate must be reported as zero_source_variance"
    );
}

/// Acceptance: the synapse surface reports its own empty-sample drops.
#[test]
fn synapse_batch_surfaces_no_samples() {
    const E: u32 = 4;

    let mut batch = vec![usable_candidate(0.0)];
    for _ in 0..E {
        batch.push(empty_candidate());
    }

    let counters = EvaluationDropCounters::new();
    let evaluated = drive_synapse_batch(&batch, &counters);
    assert_eq!(evaluated, vec![0], "only the populated work item survives");

    let mut metadata = SynapseAnalysisMetadata::default();
    let folded = fold_evaluation_drops(&counters, &mut metadata.rejection_breakdown);

    assert_eq!(folded, E);
    assert_eq!(
        metadata
            .rejection_breakdown
            .counts()
            .get(REJECTION_NO_SAMPLES),
        Some(&E)
    );
    assert_eq!(
        metadata
            .rejection_breakdown
            .counts()
            .get(REJECTION_ZERO_SOURCE_VARIANCE),
        None,
        "the synapse drop site has no source-variance guard, so the reason is absent"
    );
}

/// Acceptance: observability only — the counters must not change which
/// candidates are dropped.
#[test]
fn drop_behaviour_is_unchanged_by_the_counters() {
    let batch = vec![
        usable_candidate(0.0),
        empty_candidate(),
        constant_source_candidate(),
        usable_candidate(2.0),
        empty_candidate(),
        constant_source_candidate(),
    ];

    let counters = EvaluationDropCounters::new();
    let with_counters = drive_neuron_batch(&batch, &counters);
    let without_counters = drive_neuron_batch_uncounted(&batch);

    assert_eq!(
        with_counters, without_counters,
        "counting a drop must not change the accepted-candidate set"
    );
    assert_eq!(with_counters, vec![0, 3]);
}

/// Each surface owns a distinct counter set, so folding once per surface cannot
/// double count.
#[test]
fn per_surface_counters_do_not_double_count() {
    let neuron_counters = EvaluationDropCounters::new();
    let synapse_counters = EvaluationDropCounters::new();

    drive_neuron_batch(
        &[empty_candidate(), empty_candidate(), empty_candidate()],
        &neuron_counters,
    );
    drive_synapse_batch(&[empty_candidate()], &synapse_counters);

    let mut synapse_metadata = SynapseAnalysisMetadata::default();
    fold_evaluation_drops(&synapse_counters, &mut synapse_metadata.rejection_breakdown);

    assert_eq!(
        synapse_metadata
            .rejection_breakdown
            .counts()
            .get(REJECTION_NO_SAMPLES),
        Some(&1),
        "the synapse breakdown must count only the synapse surface's drops"
    );
}

/// A clean batch records neither reason — the keys are absent, not zero.
#[test]
fn clean_batch_records_no_reasons() {
    let counters = EvaluationDropCounters::new();
    let evaluated = drive_neuron_batch(&[usable_candidate(0.0), usable_candidate(3.0)], &counters);
    assert_eq!(evaluated.len(), 2);

    let mut metadata = NeuronAnalysisMetadata::default();
    assert_eq!(
        fold_evaluation_drops(&counters, &mut metadata.rejection_breakdown),
        0
    );
    assert!(metadata.rejection_breakdown.is_empty());
}

/// The whole point of #1798: these drops are *upstream* evidence, so a pass
/// that only ever drops candidates before evaluation is classified as
/// candidate-starved rather than proposal-rich-over-rejected.
#[test]
fn drops_count_as_upstream_starvation_evidence() {
    let counters = EvaluationDropCounters::new();
    let mut batch = Vec::new();
    for _ in 0..6 {
        batch.push(empty_candidate());
    }
    for _ in 0..6 {
        batch.push(constant_source_candidate());
    }
    drive_neuron_batch(&batch, &counters);

    let mut metadata = NeuronAnalysisMetadata::default();
    fold_evaluation_drops(&counters, &mut metadata.rejection_breakdown);

    let signals = signals_from_breakdown(&metadata.rejection_breakdown, 0);
    assert_eq!(signals.upstream_rejections, 12);
    assert_eq!(signals.gate_side_rejections, 0);
    assert_eq!(
        classify(&signals, &StarvationConfig::default()),
        StarvationClass::CandidateStarved,
        "pre-evaluation drops are generation-side evidence, not over-rejection"
    );
}
