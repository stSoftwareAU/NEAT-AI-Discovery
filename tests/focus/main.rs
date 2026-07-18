//! Focus and ranking system integration tests.

#[path = "../common/mod.rs"]
mod common;

mod focus;
mod focus_allocation;
mod focus_gradient;
mod focus_layers;
mod focus_ranking;
mod issue_1172_focus_ranking_memory_budget;
mod issue_1318_margin_aware_ranking;
mod issue_1370_arc_synapse_key;
mod issue_1374_lazy_focus_ranking_single_pass;
mod issue_1375_focus_ranking_budget;
mod issue_1376_focus_ranking_available_memory;
mod issue_1377_focus_ranking_perf_cliff;
mod issue_156_hidden_focus_neurons_filtered;
mod issue_1624_constant_neuron_focus_ineligible;
mod issue_1634_reconstruction_mismatch_focus;
mod issue_182_focus_unused_observations;
mod issue_222_hierarchical_focus;
mod issue_491_split_focus_submodules;
mod issue_564_split_ranking_submodules;
mod issue_835_impact_cache_contention;
mod issue_892_removal_candidate_boost;
