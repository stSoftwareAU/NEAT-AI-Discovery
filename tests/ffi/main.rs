//! FFI boundary integration tests.

#[path = "../common/mod.rs"]
mod common;

mod issue_574_ffi_json_fuzz_edge_cases;
mod issue_711_ffi_safety_unsafe_markers;
mod issue_950_numeric_neuron_ids;
mod issue_952_uuid_only_ffi_contract;
