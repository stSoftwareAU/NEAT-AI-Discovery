//! Focus neuron selection module.
//!
//! This module provides hierarchical focus selection and neuron ranking
//! for NEAT-AI discovery. It determines which neurons should be prioritised
//! for analysis based on error, structural impact, gradient flow, and
//! activation frequency.
//!
//! ## Module Structure (Issue #491, #564)
//!
//! - `layers` — Network layer computation via BFS from inputs
//! - `allocation` — Budget allocation strategies (Equal, Proportional, `OutputFirst`)
//! - `gradient` — Gradient flow analysis (saturation, dead neurons)
//! - `impact` — Structural impact calculation (squash-aware, selection stats)
//! - `ranking/` — Neuron ranking, record providers, removal candidates (Issue #564)
//!   - `record_providers` — Record provider trait and implementations (eager/lazy)
//!   - `score_calculation` — Individual neuron ranking score computation
//!   - `removal_candidates` — Removal candidate identification and constant neuron removal
//!   - `removal_triage` — Structure-only removal triage, no parquet (Issue #1767)

mod allocation;
mod gradient;
mod impact;
pub(crate) mod layers;
pub mod partial_dominance;
pub(crate) mod ranking;
pub mod selection;

#[cfg(test)]
mod tests;

// Re-export public API — all items remain accessible via `crate::focus::*`

// From layers
pub use layers::{NeuronInfo, NeuronLayer, compute_network_layers};

// From allocation
pub use allocation::{
    AllocationStrategy, HIERARCHICAL_SELECTION_THRESHOLD, hierarchical_focus_selection,
};

// From gradient
pub use gradient::{GradientFlowStats, compute_gradient_flow_stats};

// From impact
pub use impact::{
    ConsumerContract, MARGIN_WEIGHT_EPS, OutputGate, compute_impacts_public,
    compute_impacts_with_activations, compute_impacts_with_contract, compute_per_obs_margins,
    compute_selection_stats, derive_regime_threshold_from_records, margin_weights_from_margins,
};

// From partial_dominance (Issue #1712 — gap G2, partial-dominance safety gate)
pub use partial_dominance::{
    AggregateDominance, BranchDominance, BranchVerdict, DEFAULT_PARTIAL_WIN_FRACTION,
    DominanceThresholds, IfConditionRegime, analyse_partial_dominance, safe_collapse_branches,
};

// From selection (Issue #1445, superseded by exploit/explore under #1662)
pub use selection::{
    CONCENTRATION_WARN_THRESHOLD, DEFAULT_EXPLORATION_FRACTION, DROUGHT_EXPLORATION_FRACTION,
    FocusCandidate, FocusSelection, StructuralFocusSelection, select_focus_by_structural_impact,
    select_focus_neurons, weight_concentration_ratio,
};

// From ranking
pub use ranking::{
    FocusLazyReason, FocusLoadingMode, RankFocusStats, RankedNeuron, RecordProvider,
    RemovalCandidate, SelectionStats, StructuralRemovalCandidate, StructuralRemovalTriage,
    SynapseCounts, calculate_removal_savings, decide_loading_mode_for_available_memory,
    decide_loading_mode_for_budget, lazy_pass_exceeds_perf_cliff, rank_focus_neurons,
    rank_focus_neurons_with_descriptor, rank_focus_neurons_with_descriptor_and_deadline,
    rank_focus_neurons_with_history, rank_focus_neurons_with_history_and_descriptor,
    triage_removal_candidates,
};
// Issue #1767: structure-only removal triage (near-opposite axis to focus).
pub(crate) use ranking::identify_structural_removal_candidates;
