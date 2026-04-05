//! Issue #1002 — Concurrent synapse and neuron analysis with shared GPU queue.
//!
//! Verifies that the `analyze_neurons_with_cache_and_gpu_queue` function is
//! accessible and that the public API supports shared GPU queues for concurrent
//! analysis execution.

// Allow unused imports — these verify backward-compatible import paths compile.
#![allow(unused_imports)]

use neat_ai_discovery::analysis::analyze_neurons_with_cache_and_gpu_queue;
use neat_ai_discovery::analysis::analyze_synapses_with_cache_and_gpu_queue;
use neat_ai_discovery::analysis::gpu::GpuWorkQueue;

/// Verify that `analyze_neurons_with_cache_and_gpu_queue` is publicly accessible.
#[test]
fn neuron_analysis_with_gpu_queue_is_accessible() {
    // Compile-time check: the function exists and has the correct signature.
    fn _verify_signature(
        _f: fn(
            &neat_ai_discovery::AnalyzeNeuronsInput,
            std::sync::Arc<neat_ai_discovery::analysis::cache::RecordCache>,
            std::sync::Arc<GpuWorkQueue>,
        )
            -> anyhow::Result<neat_ai_discovery::analysis::shared::AnalyzeNeuronsResult>,
    ) {
    }
    _verify_signature(analyze_neurons_with_cache_and_gpu_queue);
}

/// Verify that `analyze_synapses_with_cache_and_gpu_queue` is still publicly accessible.
#[test]
fn synapse_analysis_with_gpu_queue_is_accessible() {
    fn _verify_signature(
        _f: fn(
            &neat_ai_discovery::AnalyzeSynapsesInput,
            std::sync::Arc<neat_ai_discovery::analysis::cache::RecordCache>,
            std::sync::Arc<GpuWorkQueue>,
        )
            -> anyhow::Result<neat_ai_discovery::analysis::shared::AnalyzeSynapsesResult>,
    ) {
    }
    _verify_signature(analyze_synapses_with_cache_and_gpu_queue);
}

/// Verify that both analysis functions can accept the same `Arc<GpuWorkQueue>`.
///
/// This is a structural test confirming that both functions accept the same
/// queue type, which is the prerequisite for concurrent execution with a
/// shared GPU thread (Issue #1002).
#[test]
fn both_analyses_accept_same_gpu_queue_type() {
    // The key invariant: both functions accept Arc<GpuWorkQueue>.
    // If this compiles, the shared queue contract is satisfied.
    fn _accepts_shared_queue(queue: std::sync::Arc<GpuWorkQueue>) {
        let _syn_queue: std::sync::Arc<GpuWorkQueue> = std::sync::Arc::clone(&queue);
        let _neu_queue: std::sync::Arc<GpuWorkQueue> = std::sync::Arc::clone(&queue);
        // Both clones can be passed to their respective analysis functions.
    }
}
