//! Consolidated per-cycle deadline-consumption breakdown (Issue #1409).
//!
//! The data needed to diagnose where the analysis deadline went is scattered:
//! per-phase timings live in `ProfileData`, while synapse/neuron starvation
//! signals (`timed_out`, `completed_focus_neurons`, `total_focus_neurons`) live
//! on the per-phase metadata. This module consolidates them into a single,
//! greppable summary emitted once per `analyze_all` invocation so operators can
//! attribute deadline consumption across phases and see, explicitly, when
//! synapse/neuron analysis was starved by the deadline.
//!
//! # Emission
//!
//! [`DeadlineConsumptionBreakdown::emit`] logs one structured `tracing::info!`
//! summary line. When either phase was curtailed by the deadline it also logs a
//! `tracing::warn!` carrying the `STARVED` marker and the skipped/total counts.
//!
//! Both events carry the [`DEADLINE_BREAKDOWN_MARKER`] token so operators can
//! grep a log for the whole breakdown in one pass.
//!
//! This is **observability only** — it never changes analysis math.

/// Stable, greppable token prefixing every deadline-breakdown log event.
///
/// Operators grep logs for this string, so it is a published contract: changing
/// it breaks existing log queries. It is deliberately named after what it
/// reports rather than after any particular deployment (Issue #1723).
pub const DEADLINE_BREAKDOWN_MARKER: &str = "DEADLINE-BREAKDOWN";

/// Completion accounting for one analysis phase (synapse or neuron).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseCompletion {
    /// Whether this phase hit its deadline and returned partial results.
    pub timed_out: bool,
    /// Focus neurons completed before the phase returned.
    pub completed_focus_neurons: usize,
    /// Focus neurons requested for the phase.
    pub total_focus_neurons: usize,
}

impl PhaseCompletion {
    /// Number of focus neurons the phase did not reach (saturating).
    #[must_use]
    pub fn skipped(&self) -> usize {
        self.total_focus_neurons
            .saturating_sub(self.completed_focus_neurons)
    }

    /// True when the deadline curtailed this phase: it timed out **and** left at
    /// least one focus neuron unanalysed.
    #[must_use]
    pub fn is_starved(&self) -> bool {
        self.timed_out && self.completed_focus_neurons < self.total_focus_neurons
    }
}

/// Per-phase millisecond breakdown of where the analysis deadline was consumed,
/// plus synapse/neuron completion ratios, for a single discovery cycle.
///
/// `synapse`/`neuron` are `None` when the corresponding analysis was disabled
/// for the run. The focus phase (parquet load + focus ranking) runs in a
/// separate FFI call (`rank_focus_neurons`); its timings are surfaced there
/// (Issue #1377) and combined by the calling host layer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeadlineConsumptionBreakdown {
    /// Milliseconds spent re-loading the Parquet record cache for the analysis
    /// phase.
    pub parquet_reload_ms: u64,
    /// Wall-clock milliseconds spent in synapse analysis (`None` when disabled).
    pub synapse_analysis_ms: Option<u64>,
    /// Wall-clock milliseconds spent in neuron analysis (`None` when disabled).
    pub neuron_analysis_ms: Option<u64>,
    /// Total wall-clock milliseconds for the whole analysis invocation.
    pub total_analysis_ms: u64,
    /// Synapse completion accounting (`None` when synapse analysis disabled).
    pub synapse: Option<PhaseCompletion>,
    /// Neuron completion accounting (`None` when neuron analysis disabled).
    pub neuron: Option<PhaseCompletion>,
}

impl DeadlineConsumptionBreakdown {
    /// True when either phase was curtailed by the deadline.
    #[must_use]
    pub fn is_starved(&self) -> bool {
        self.synapse.is_some_and(|s| s.is_starved()) || self.neuron.is_some_and(|n| n.is_starved())
    }

    /// One greppable line attributing deadline consumption across all phases
    /// owned by the analysis call.
    #[must_use]
    pub fn summary_line(&self) -> String {
        fn ms(value: Option<u64>) -> String {
            value.map_or_else(|| "n/a".to_string(), |v| v.to_string())
        }
        fn ratio(phase: Option<PhaseCompletion>) -> String {
            phase.map_or_else(
                || "n/a".to_string(),
                |p| format!("{}/{}", p.completed_focus_neurons, p.total_focus_neurons),
            )
        }

        format!(
            "{DEADLINE_BREAKDOWN_MARKER} deadline consumption (ms): parquet_reload={} \
             synapse_analysis={} neuron_analysis={} total_analysis={}; \
             focus completed synapse={} neuron={}",
            self.parquet_reload_ms,
            ms(self.synapse_analysis_ms),
            ms(self.neuron_analysis_ms),
            self.total_analysis_ms,
            ratio(self.synapse),
            ratio(self.neuron),
        )
    }

    /// The `STARVED` warning message when the deadline curtailed analysis,
    /// otherwise `None`.
    ///
    /// Names which phase(s) were skipped and by how much, so the warning is
    /// actionable without re-running analysis.
    #[must_use]
    pub fn starvation_warning(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(syn) = self.synapse.filter(PhaseCompletion::is_starved) {
            parts.push(format!(
                "synapse analysis skipped {}/{} targets",
                syn.skipped(),
                syn.total_focus_neurons
            ));
        }
        if let Some(neu) = self.neuron.filter(PhaseCompletion::is_starved) {
            parts.push(format!(
                "neuron analysis skipped {}/{} targets",
                neu.skipped(),
                neu.total_focus_neurons
            ));
        }

        if parts.is_empty() {
            return None;
        }
        Some(format!(
            "STARVED: {} — deadline exhausted by focus+parquet before analysis could finish",
            parts.join("; ")
        ))
    }

    /// Emit the consolidated breakdown: one structured `info` summary line, plus
    /// a `warn` carrying the `STARVED` marker when analysis was curtailed.
    pub fn emit(&self) {
        tracing::info!(
            marker = DEADLINE_BREAKDOWN_MARKER,
            parquet_reload_ms = self.parquet_reload_ms,
            synapse_analysis_ms = self.synapse_analysis_ms.unwrap_or(0),
            neuron_analysis_ms = self.neuron_analysis_ms.unwrap_or(0),
            total_analysis_ms = self.total_analysis_ms,
            synapse_completed = self.synapse.map_or(0, |s| s.completed_focus_neurons),
            synapse_total = self.synapse.map_or(0, |s| s.total_focus_neurons),
            neuron_completed = self.neuron.map_or(0, |n| n.completed_focus_neurons),
            neuron_total = self.neuron.map_or(0, |n| n.total_focus_neurons),
            "{}",
            self.summary_line()
        );

        if let Some(warning) = self.starvation_warning() {
            tracing::warn!(marker = DEADLINE_BREAKDOWN_MARKER, "{warning}");
        }
    }
}
