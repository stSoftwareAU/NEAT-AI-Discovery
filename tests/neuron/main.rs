//! Neuron analysis integration tests.

#[path = "../common/mod.rs"]
mod common;

mod issue_132_cost_of_growth;
mod issue_134_direction_flip;
mod issue_164_redundant_path_pruning;
mod issue_178_constant_source_folds_to_setbias;
mod issue_180_setweight_operation;
mod issue_210_neuron_interning;
mod issue_306_constant_neuron_removal_with_bias_adjustment;
mod issue_414_remove_neuron_high_error;
mod issue_415_combo_successful_interference;
mod issue_743_uuid_hashing;
mod issue_834_lock_free_error_collection;
mod neuron_metadata_candidates_found_includes_pairing;
