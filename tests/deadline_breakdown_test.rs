//! Tests for the consolidated per-cycle deadline-consumption breakdown
//! (Issue #1409 / GRQ-23).
//!
//! These verify the observable behaviour of the breakdown: the greppable
//! summary line, the explicit `STARVED` warning emitted when synapse/neuron
//! analysis is curtailed by the deadline, and the starvation predicate that the
//! structured analysis result surfaces to the GRQ layer.

use neat_ai_discovery::analysis::deadline_breakdown::{
    DeadlineConsumptionBreakdown, PhaseCompletion,
};

/// A phase that finished all its focus neurons is never starved, even if the
/// `timed_out` flag was set after the last neuron completed.
#[test]
fn phase_completion_not_starved_when_all_completed() {
    let phase = PhaseCompletion {
        timed_out: true,
        completed_focus_neurons: 8,
        total_focus_neurons: 8,
    };
    assert!(!phase.is_starved());
    assert_eq!(phase.skipped(), 0);
}

/// A phase that timed out before reaching every focus neuron is starved.
#[test]
fn phase_completion_starved_when_timed_out_with_remaining() {
    let phase = PhaseCompletion {
        timed_out: true,
        completed_focus_neurons: 3,
        total_focus_neurons: 10,
    };
    assert!(phase.is_starved());
    assert_eq!(phase.skipped(), 7);
}

/// Partial completion without a timeout is not starvation — the phase simply
/// found nothing more to do.
#[test]
fn phase_completion_not_starved_without_timeout() {
    let phase = PhaseCompletion {
        timed_out: false,
        completed_focus_neurons: 3,
        total_focus_neurons: 10,
    };
    assert!(!phase.is_starved());
}

/// On a simulated timeout the breakdown reports starvation and the warning
/// names the skipped/total counts for each curtailed phase.
#[test]
fn simulated_timeout_emits_starved_warning_with_counts() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 4200,
        synapse_analysis_ms: Some(1200),
        neuron_analysis_ms: Some(900),
        total_analysis_ms: 6300,
        synapse: Some(PhaseCompletion {
            timed_out: true,
            completed_focus_neurons: 2,
            total_focus_neurons: 12,
        }),
        neuron: Some(PhaseCompletion {
            timed_out: true,
            completed_focus_neurons: 1,
            total_focus_neurons: 12,
        }),
    };

    assert!(breakdown.is_starved());

    let warning = breakdown
        .starvation_warning()
        .expect("a curtailed run must produce a STARVED warning");
    assert!(warning.contains("STARVED"), "warning: {warning}");
    assert!(
        warning.contains("synapse analysis skipped 10/12 targets"),
        "warning: {warning}"
    );
    assert!(
        warning.contains("neuron analysis skipped 11/12 targets"),
        "warning: {warning}"
    );

    // Emission must not panic.
    breakdown.emit();
}

/// Only the curtailed phase is named when the other phase completed in time.
#[test]
fn starved_warning_only_names_curtailed_phase() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 100,
        synapse_analysis_ms: Some(50),
        neuron_analysis_ms: Some(50),
        total_analysis_ms: 200,
        synapse: Some(PhaseCompletion {
            timed_out: true,
            completed_focus_neurons: 4,
            total_focus_neurons: 9,
        }),
        neuron: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 9,
            total_focus_neurons: 9,
        }),
    };

    let warning = breakdown
        .starvation_warning()
        .expect("synapse was curtailed");
    assert!(warning.contains("synapse analysis skipped 5/9 targets"));
    assert!(
        !warning.contains("neuron analysis skipped"),
        "neuron completed in time: {warning}"
    );
}

/// A run that completes within the deadline produces no warning.
#[test]
fn completed_run_has_no_starvation_warning() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 300,
        synapse_analysis_ms: Some(800),
        neuron_analysis_ms: Some(700),
        total_analysis_ms: 1800,
        synapse: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 16,
            total_focus_neurons: 16,
        }),
        neuron: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 16,
            total_focus_neurons: 16,
        }),
    };

    assert!(!breakdown.is_starved());
    assert!(breakdown.starvation_warning().is_none());
    breakdown.emit();
}

/// The consolidated summary line spans every phase the analysis call owns and
/// stays greppable via the stable `GRQ-23` marker.
#[test]
fn summary_line_attributes_all_phases() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 4200,
        synapse_analysis_ms: Some(1200),
        neuron_analysis_ms: Some(900),
        total_analysis_ms: 6300,
        synapse: Some(PhaseCompletion {
            timed_out: true,
            completed_focus_neurons: 2,
            total_focus_neurons: 12,
        }),
        neuron: Some(PhaseCompletion {
            timed_out: true,
            completed_focus_neurons: 1,
            total_focus_neurons: 12,
        }),
    };

    let line = breakdown.summary_line();
    assert!(line.contains("GRQ-23"), "line: {line}");
    assert!(line.contains("parquet_reload=4200"), "line: {line}");
    assert!(line.contains("synapse_analysis=1200"), "line: {line}");
    assert!(line.contains("neuron_analysis=900"), "line: {line}");
    assert!(line.contains("total_analysis=6300"), "line: {line}");
    assert!(line.contains("synapse=2/12"), "line: {line}");
    assert!(line.contains("neuron=1/12"), "line: {line}");
}

/// Disabled phases render as `n/a` rather than fabricated zeroes, so the
/// breakdown does not imply a phase ran when it was switched off.
#[test]
fn summary_line_marks_disabled_phases_as_not_applicable() {
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms: 500,
        synapse_analysis_ms: Some(1000),
        neuron_analysis_ms: None,
        total_analysis_ms: 1600,
        synapse: Some(PhaseCompletion {
            timed_out: false,
            completed_focus_neurons: 5,
            total_focus_neurons: 5,
        }),
        neuron: None,
    };

    let line = breakdown.summary_line();
    assert!(line.contains("neuron_analysis=n/a"), "line: {line}");
    assert!(line.contains("neuron=n/a"), "line: {line}");
    assert!(!breakdown.is_starved());
    assert!(breakdown.starvation_warning().is_none());
}
