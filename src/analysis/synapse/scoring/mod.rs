//! Synapse scoring and improvement calculation
//!
//! This module contains functions for computing improvement scores, saturation-aware
//! simulation, and source/target type boosting (Issues #413, #467, #468).
//!
//! ## Sub-modules (Issue #982)
//!
//! - `boost_functions` — Source-type, target-type, and activation-function-aware boosts
//! - `improvement` — Core synapse improvement calculation algorithm and candidate dedup
//! - `discounting` — Pessimism discounting and prediction calibration
//! - `test_helpers` — Test-only wrapper functions (cfg(test))

mod boost_functions;
mod discounting;
pub mod improvement;
#[cfg(test)]
mod test_helpers;

// Re-export all public items for backward compatibility
pub use boost_functions::{
    apply_activation_neuron_boost, apply_source_type_boost, apply_target_type_boost,
};
pub use discounting::{
    apply_logistic_prediction_calibration, apply_neuron_pessimism_discount,
    apply_pessimism_discount, apply_prediction_calibration, apply_synapse_pessimism_discount,
};

pub(crate) use improvement::upsert_candidate;
pub use improvement::{
    compute_activation_improvement_and_count, compute_relu_improvement_and_count,
    compute_synapse_improvement_and_count,
};

#[cfg(test)]
pub(crate) use improvement::{compute_candidate_dedup_key, weight_sign};

#[cfg(test)]
pub(crate) use test_helpers::{
    compute_net_improvement_with_squash, compute_synapse_improvement_with_target_squash,
    count_improved_samples,
};

#[cfg(test)]
mod tests;
