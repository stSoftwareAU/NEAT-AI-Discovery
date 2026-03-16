//! Activation function integration tests.

#[path = "../common/mod.rs"]
mod common;

mod activation_based_impact;
mod activation_coverage_neat_ai_registry;
mod activation_overflow_protection_f32;
mod complement_via_identity;
mod discrete_activation_filtering;
mod issue_417_increase_change_squash_rate;
mod issue_768_activation_properties;
mod leaky_relu_filtered;
mod relu_adjacent_filtered;
mod squash_bounded_impact;
mod squash_consistency_with_neat_ai;
mod step_activation_based_impact;
mod step_hidden_neuron_prediction;
