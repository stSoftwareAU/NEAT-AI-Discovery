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

mod allocation;
mod gradient;
mod impact;
pub(crate) mod layers;
pub(crate) mod ranking;

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

// From ranking
pub use ranking::{
    FocusLazyReason, FocusLoadingMode, RankFocusStats, RankedNeuron, RecordProvider,
    RemovalCandidate, SelectionStats, SynapseCounts, calculate_removal_savings,
    decide_loading_mode_for_available_memory, decide_loading_mode_for_budget, rank_focus_neurons,
    rank_focus_neurons_with_descriptor, rank_focus_neurons_with_history,
    rank_focus_neurons_with_history_and_descriptor,
};
