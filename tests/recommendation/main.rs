//! Candidate recommendation integration tests.

#[path = "../common/mod.rs"]
mod common;

mod issue_189_synergistic_discovery;
mod issue_202_epistatic_neuron_pairs;
mod issue_230_multi_hop_candidate_analysis;
mod issue_402_range_aware_weight_optimisation;
mod issue_421_gradient_based_discovery;
mod issue_422_topology_aware_discovery;
mod issue_423_sample_weighted_discovery;
mod issue_431_activation_recommendation;
mod issue_507_micro_nudge_variant;
mod issue_509_deduplicate_dominant_neuron;
mod issue_548_squash_weight_rescale;
mod issue_549_topology_diversification;
mod issue_550_weight_magnitude_reset;
mod issue_551_bias_perturbation_regime_shift;
mod issue_570_skip_connection_discovery;
mod issue_788_high_error_squash_exploration;
