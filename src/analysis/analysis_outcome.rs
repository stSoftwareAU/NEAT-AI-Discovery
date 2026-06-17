//! Distinguish environmentally-disabled discovery passes from genuine search
//! exhaustion (Issue #1421).
//!
//! When a discovery pass is gated by a host-environment check (memory budget,
//! system memory pressure, or a missing GPU adapter) the analysis never
//! actually evaluated the creature. Historically such a pass returned the
//! *same* "0 candidates / no improvement" signal as a genuinely-exhausted
//! search, so drought counters, the drought diagnostic (Issue #1202), and the
//! per-target / per-module trackers treated *"this host cannot run discovery"*
//! as *"the creature has no improving move"* — corrupting every downstream
//! mitigation decision.
//!
//! [`AnalysisOutcome`] makes the two cases distinct. Drought / staleness /
//! cooldown / starvation accounting MUST ignore
//! [`AnalysisOutcome::EnvironmentallyDisabled`] passes; they are not evidence
//! of search exhaustion.

use serde::{Deserialize, Serialize};

use super::shared::AnalyzeAllResult;

/// Why a discovery pass could not actually evaluate the creature (Issue #1421).
///
/// Serialised in camelCase so the host can report the disable category
/// distinctly from a genuinely-empty pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EnvironmentalDisableReason {
    /// The Rust-side memory budget (`maxAnalysisMemoryMb`) was exceeded before
    /// GPU analysis began (Issue #1028).
    MemoryGated,
    /// Analysis was cancelled because the system was under CRITICAL memory
    /// pressure (Issue #1099).
    MemoryPressure,
    /// No compatible GPU adapter was found on this host (Issue #988).
    GpuUnavailable,
}

impl EnvironmentalDisableReason {
    /// Stable, greppable identifier for logs and metrics.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MemoryGated => "memory_gated",
            Self::MemoryPressure => "memory_pressure",
            Self::GpuUnavailable => "gpu_unavailable",
        }
    }
}

/// Outcome of a single discovery analysis pass (Issue #1421).
///
/// [`Self::Completed`] means the pass actually evaluated the creature and the
/// candidate count is genuine signal. [`Self::EnvironmentallyDisabled`] means
/// the pass was gated before evaluation and carries **no** search-exhaustion
/// signal — callers must exclude it from drought / cooldown / starvation
/// accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisOutcome {
    /// The pass evaluated the creature and produced `candidates` candidates.
    Completed {
        /// Total candidates returned across synapse and neuron analysis.
        candidates: usize,
    },
    /// The pass was gated by an environmental check and never evaluated the
    /// creature.
    EnvironmentallyDisabled {
        /// Why discovery could not run on this host.
        reason: EnvironmentalDisableReason,
    },
}

impl AnalysisOutcome {
    /// Classify a successful [`AnalyzeAllResult`].
    ///
    /// The environmental gates that produce an `Ok` result are the memory
    /// budget (Issue #1028) and CRITICAL memory pressure (Issue #1099). Any
    /// other `Ok` result is [`Self::Completed`] with its genuine candidate
    /// count — even when that count is zero (true search exhaustion).
    #[must_use]
    pub fn from_result(result: &AnalyzeAllResult) -> Self {
        if result.memory_budget_exceeded {
            return Self::EnvironmentallyDisabled {
                reason: EnvironmentalDisableReason::MemoryGated,
            };
        }
        if result.memory_pressure_cancelled {
            return Self::EnvironmentallyDisabled {
                reason: EnvironmentalDisableReason::MemoryPressure,
            };
        }
        Self::Completed {
            candidates: Self::count_candidates(result),
        }
    }

    /// Outcome for the GPU-unavailable early return, which surfaces as an
    /// `Err` from `analyze_all` rather than an `AnalyzeAllResult`.
    #[must_use]
    pub fn gpu_unavailable() -> Self {
        Self::EnvironmentallyDisabled {
            reason: EnvironmentalDisableReason::GpuUnavailable,
        }
    }

    /// `true` when the pass was gated before evaluation and therefore carries
    /// no search-exhaustion signal.
    #[must_use]
    pub fn is_environmentally_disabled(&self) -> bool {
        matches!(self, Self::EnvironmentallyDisabled { .. })
    }

    /// `true` when the pass actually evaluated the creature and returned no
    /// candidates — genuine search exhaustion that drought / cooldown
    /// accounting *should* count.
    #[must_use]
    pub fn is_genuinely_empty(&self) -> bool {
        matches!(self, Self::Completed { candidates: 0 })
    }

    /// `true` when the pass evaluated the creature and returned at least one
    /// candidate.
    #[must_use]
    pub fn is_productive(&self) -> bool {
        matches!(self, Self::Completed { candidates } if *candidates > 0)
    }

    /// The environmental disable reason, if any.
    #[must_use]
    pub fn disable_reason(&self) -> Option<EnvironmentalDisableReason> {
        match self {
            Self::EnvironmentallyDisabled { reason } => Some(*reason),
            Self::Completed { .. } => None,
        }
    }

    /// Total candidates returned across synapse and neuron analysis.
    fn count_candidates(result: &AnalyzeAllResult) -> usize {
        let synapse = result
            .synapse
            .as_ref()
            .map_or(0, |s| s.metadata.candidates_returned);
        let neuron = result
            .neuron
            .as_ref()
            .map_or(0, |n| n.metadata.candidates_returned);
        synapse.saturating_add(neuron)
    }
}

/// Running tally of pass outcomes for the discovery summary (Issue #1421).
///
/// Separates environmentally-disabled passes (host could not run discovery)
/// from genuinely-empty passes (search exhausted) so operators can tell the
/// two halves of a drought apart.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PassOutcomeCounts {
    /// Passes that returned at least one candidate.
    pub productive: u32,
    /// Passes that evaluated the creature but found no candidate — true search
    /// exhaustion.
    pub genuinely_empty: u32,
    /// Passes gated by an environmental check that never evaluated the
    /// creature.
    pub environmentally_disabled: u32,
}

impl PassOutcomeCounts {
    /// Record a single pass outcome.
    pub fn record(&mut self, outcome: &AnalysisOutcome) {
        match outcome {
            AnalysisOutcome::Completed { candidates } if *candidates > 0 => {
                self.productive = self.productive.saturating_add(1);
            }
            AnalysisOutcome::Completed { .. } => {
                self.genuinely_empty = self.genuinely_empty.saturating_add(1);
            }
            AnalysisOutcome::EnvironmentallyDisabled { .. } => {
                self.environmentally_disabled = self.environmentally_disabled.saturating_add(1);
            }
        }
    }

    /// Total passes recorded across all categories.
    #[must_use]
    pub fn total(&self) -> u32 {
        self.productive
            .saturating_add(self.genuinely_empty)
            .saturating_add(self.environmentally_disabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_unavailable_is_environmentally_disabled() {
        let outcome = AnalysisOutcome::gpu_unavailable();
        assert!(outcome.is_environmentally_disabled());
        assert!(!outcome.is_genuinely_empty());
        assert!(!outcome.is_productive());
        assert_eq!(
            outcome.disable_reason(),
            Some(EnvironmentalDisableReason::GpuUnavailable)
        );
    }

    #[test]
    fn completed_zero_is_genuinely_empty_not_disabled() {
        let outcome = AnalysisOutcome::Completed { candidates: 0 };
        assert!(outcome.is_genuinely_empty());
        assert!(!outcome.is_environmentally_disabled());
        assert_eq!(outcome.disable_reason(), None);
    }

    #[test]
    fn completed_with_candidates_is_productive() {
        let outcome = AnalysisOutcome::Completed { candidates: 3 };
        assert!(outcome.is_productive());
        assert!(!outcome.is_genuinely_empty());
        assert!(!outcome.is_environmentally_disabled());
    }

    #[test]
    fn reason_strings_are_stable() {
        assert_eq!(
            EnvironmentalDisableReason::MemoryGated.as_str(),
            "memory_gated"
        );
        assert_eq!(
            EnvironmentalDisableReason::MemoryPressure.as_str(),
            "memory_pressure"
        );
        assert_eq!(
            EnvironmentalDisableReason::GpuUnavailable.as_str(),
            "gpu_unavailable"
        );
    }

    #[test]
    fn pass_counts_separate_disabled_from_empty() {
        let mut counts = PassOutcomeCounts::default();
        counts.record(&AnalysisOutcome::Completed { candidates: 2 });
        counts.record(&AnalysisOutcome::Completed { candidates: 0 });
        counts.record(&AnalysisOutcome::Completed { candidates: 0 });
        counts.record(&AnalysisOutcome::gpu_unavailable());
        counts.record(&AnalysisOutcome::EnvironmentallyDisabled {
            reason: EnvironmentalDisableReason::MemoryGated,
        });

        assert_eq!(counts.productive, 1);
        assert_eq!(counts.genuinely_empty, 2);
        assert_eq!(counts.environmentally_disabled, 2);
        assert_eq!(counts.total(), 5);
    }
}
