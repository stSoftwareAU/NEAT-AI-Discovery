//! Infrastructure integration tests (caching, config, observability).

#[path = "../common/mod.rs"]
mod common;

mod issue_1002_concurrent_analysis;
mod issue_186_rwlock_cache;
mod issue_196_cache_locality_benchmark;
mod issue_228_zero_copy_buffer;
mod issue_477_rust_edition_2024;
mod issue_487_reduce_clone_allocations;
mod issue_525_graceful_mutex_handling;
mod issue_575_structured_logging;
mod issue_677_typed_error_enums;
mod issue_717_config_env_vars;
mod issue_718_bug_comment_error_handling;
mod issue_754_topology_cache;
mod issue_769_zero_alloc_synapse_lookup;
mod issue_776_hashset_hashmap_iteration;
mod issue_832_deadlock_stress;
mod issue_836_deadlock_abort;
mod issue_837_lock_contention_tracing;
mod issue_874_module_split_backward_compat;
mod issue_988_graceful_gpu_error;
mod issue_994_fix_deadlock;
mod observability;
