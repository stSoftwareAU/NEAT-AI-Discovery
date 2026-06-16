//! FFI boundary integration tests.

#[path = "../common/mod.rs"]
mod common;

mod issue_1027_memory_usage_ffi;
mod issue_1028_memory_budget;
mod issue_1184_recurrent_synapse_rejection;
mod issue_1188_grq3_strip_pattern_rejection;
mod issue_1314_task_descriptor_plumbing;
mod issue_1402_lowercase_task_descriptor_regression;
mod issue_574_ffi_json_fuzz_edge_cases;
mod issue_711_ffi_safety_unsafe_markers;
mod issue_950_numeric_neuron_ids;
mod issue_952_uuid_only_ffi_contract;
mod issue_975_ffi_boundary_error_handling;
