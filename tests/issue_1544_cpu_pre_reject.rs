//! Integration tests for Issue #1544 — wire the CPU pre-reject screen before
//! the helpful GPU submit in the synapse target-analysis path.
//!
//! These tests exercise the public screen predicate
//! [`helpful_candidate_has_no_signal`] and mirror exactly what the production
//! path (`target_analysis::analyse_single_target`) does with its result: it
//! `retain`s only survivors in the helpful work batch (so no-signal sources
//! produce **zero** helpful GPU submits) and records the dropped count under
//! [`REJECTION_CPU_PRE_REJECT_NO_SIGNAL`] on the synapse metadata's
//! `rejection_breakdown`.
//!
//! No GPU is required: the screen is a pure CPU function, so these run under
//! `./quality.sh` / `cargo-quality.yml` on any machine.
//!
//! Two scenarios (per the issue's Failure Detection plan):
//! 1. `no_signal_source_is_rejected_before_gpu_submit` — a zero-variance /
//!    no-signal source is screened out (zero helpful submits) and its rejection
//!    is recorded under `cpu_pre_reject_no_signal`.
//! 2. `strong_signal_source_survives_pre_reject` — a source with a clear
//!    improvement signal is **not** screened out (guards the dangerous failure
//!    mode: silently dropping good candidates, which looks like a perf win but
//!    is a quality regression).

#![allow(clippy::cast_precision_loss)] // Test fixtures build sample values from small loop indices.
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    REJECTION_CPU_PRE_REJECT_NO_SIGNAL, RejectionBreakdown, top_level_summary,
};
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::synapse::cpu_pre_reject::helpful_candidate_has_no_signal;

fn sample(activation: f32, avg_error: f32) -> HelpfulSample {
    HelpfulSample {
        activation,
        avg_error,
        target_value: None,
        target_activation: None,
    }
}

/// A constant (zero-variance) source: activation never changes, so adding a
/// synapse from it cannot correlate with the varying target error. This is the
/// canonical "no signal" dud described in Issue #130 / #1544.
fn zero_variance_source() -> Vec<HelpfulSample> {
    (0..64)
        .map(|i| {
            let err = if i % 2 == 0 { 0.3 } else { -0.3 };
            sample(0.0, err)
        })
        .collect()
}

/// A strong-signal source: activation is strongly correlated with the target
/// error, so a usable outgoing weight exists and the candidate must survive.
fn strong_signal_source() -> Vec<HelpfulSample> {
    (0..64)
        .map(|i| {
            let a = (i as f32 % 7.0) - 3.0;
            sample(a, a * 0.8)
        })
        .collect()
}

/// Mirror the production filter+record step: partition a batch of helpful work
/// (represented as per-source sample vectors) into the survivors that would be
/// submitted to the GPU and a `RejectionBreakdown` crediting the drops.
fn screen_batch(batch: Vec<Vec<HelpfulSample>>) -> (Vec<Vec<HelpfulSample>>, RejectionBreakdown) {
    let before = batch.len();
    let survivors: Vec<Vec<HelpfulSample>> = batch
        .into_iter()
        .filter(|samples| !helpful_candidate_has_no_signal(samples))
        .collect();
    let dropped = u32::try_from(before - survivors.len()).unwrap_or(u32::MAX);

    let mut breakdown = RejectionBreakdown::new();
    breakdown.record_many_u32(REJECTION_CPU_PRE_REJECT_NO_SIGNAL, dropped);
    (survivors, breakdown)
}

#[test]
fn no_signal_source_is_rejected_before_gpu_submit() {
    // A single no-signal source in the batch.
    let batch = vec![zero_variance_source()];

    let (submitted, breakdown) = screen_batch(batch);

    // Zero helpful GPU submits: the no-signal source is screened out.
    assert!(
        submitted.is_empty(),
        "no-signal source must produce zero helpful GPU submits"
    );

    // The drop is recorded under the dedicated reason string so drought
    // diagnostics stay honest (distinguishable from a genuine drought).
    assert_eq!(
        breakdown
            .counts()
            .get(REJECTION_CPU_PRE_REJECT_NO_SIGNAL)
            .copied(),
        Some(1),
        "breakdown should credit one cpu_pre_reject_no_signal drop, got: {:?}",
        breakdown.counts()
    );

    let (dominant, _) = breakdown
        .dominant_reason()
        .expect("dominant reason present");
    assert_eq!(dominant, REJECTION_CPU_PRE_REJECT_NO_SIGNAL);

    let summary = top_level_summary(&breakdown, None).expect("summary populated");
    assert!(
        summary.contains("CPU pre-reject"),
        "summary should name the CPU pre-reject screen, got: {summary}"
    );
}

#[test]
fn strong_signal_source_survives_pre_reject() {
    // A batch mixing one strong-signal source with one no-signal source: only
    // the dud is dropped, the good candidate is submitted to the GPU.
    let batch = vec![strong_signal_source(), zero_variance_source()];

    let (submitted, breakdown) = screen_batch(batch);

    assert_eq!(
        submitted.len(),
        1,
        "exactly the strong-signal source should survive the screen"
    );
    // The survivor is genuinely the strong-signal one.
    assert!(
        !helpful_candidate_has_no_signal(&submitted[0]),
        "surviving candidate must carry signal"
    );

    // Only the dud was dropped.
    assert_eq!(
        breakdown
            .counts()
            .get(REJECTION_CPU_PRE_REJECT_NO_SIGNAL)
            .copied(),
        Some(1),
        "only the no-signal source should be counted as dropped"
    );
}

#[test]
fn strong_signal_only_batch_records_no_drops() {
    // Guards over-rejection: a batch of only strong-signal sources must submit
    // every candidate and record zero drops.
    let batch = vec![strong_signal_source(), strong_signal_source()];

    let (submitted, breakdown) = screen_batch(batch);

    assert_eq!(submitted.len(), 2, "no strong-signal source may be dropped");
    assert!(
        breakdown.is_empty(),
        "no drops expected, got: {:?}",
        breakdown.counts()
    );
    assert!(
        top_level_summary(&breakdown, None).is_none(),
        "no summary when nothing was rejected"
    );
}
