//! Detection modules — pattern detection for network issues.
//!
//! This subdirectory groups modules that detect specific problematic patterns
//! in the neural network (saturated neurons, bottlenecks, dead neurons, etc.)
//! and convert them into coordinated structural candidates.

pub mod bottleneck;
pub mod bounded_range;
pub mod correlated_error;
pub mod dead_neuron;
pub mod dormant_synapse;
pub mod input_sensitivity;
pub mod noise_signal;
pub mod observation_range;
pub mod operating_point;
pub mod opposing_synapse;
pub mod oscillating_neuron;
pub mod redundant_path;
pub mod restricted_range;
pub mod saturation;
pub mod sentinel_gating;
pub mod topology;
pub mod unbounded_capping;
pub mod weight_coherence;
