//! Detection modules — pattern detection for network issues.
//!
//! This subdirectory groups modules that detect specific problematic patterns
//! in the neural network (saturated neurons, bottlenecks, dead neurons, etc.)
//! and convert them into coordinated structural candidates.

pub mod activation_mismatch;
pub mod activation_properties;
pub mod bias_perturbation;
pub mod bimodal_neuron;
pub mod bottleneck;
pub mod bounded_range;
pub mod co_adaptation;
pub mod compound_degradation;
pub mod correlated_error;
pub mod dead_neuron;
pub mod dormant_synapse;
pub mod error_plateau;
pub mod fanin_polarity_conflict;
pub mod hard_sample_cluster;
pub mod helpers;
pub mod high_error_squash_exploration;
pub mod input_sensitivity;
pub mod low_impact_neuron;
pub mod monotonicity;
pub mod noise_signal;
pub mod observation_range;
pub mod observation_utilisation;
pub mod operating_point;
pub mod opposing_synapse;
pub mod oscillating_neuron;
pub mod output_conflict;
pub mod output_range_compression;
pub mod output_squash_mismatch;
pub mod redundant_path;
pub mod restricted_range;
pub mod saturation;
pub mod sentinel_gating;
pub mod skip_connection;
pub mod squash_weight_rescale;
pub mod stats;
pub mod symmetry_breaking;
pub mod topology;
pub mod topology_cache;
pub mod topology_diversification;
pub mod unbounded_capping;
pub mod weight_coherence;
pub mod weight_magnitude_reset;
pub mod weight_polarity_flip;
