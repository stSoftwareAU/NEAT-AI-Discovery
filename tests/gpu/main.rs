//! GPU functionality integration tests.

#[path = "../common/mod.rs"]
mod common;

mod gpu_activation_shaders;
mod gpu_work_queue;
mod gpu_workgroup_reduction;
mod issue_1083_gpu_batch_size_reduction;
mod issue_1369_buffer_reuse_chunking;
mod issue_1873_gpu_env_setup_thread_guard;
mod issue_1929_stale_request_skip;
mod issue_1930_gpu_circuit_breaker;
mod issue_647_gpu_device_lost_recovery;
mod issue_713_deduplicate_gpu_env_setup;
mod issue_807_gpu_queue_tracing;
mod issue_953_gpu_queue_deadline;
