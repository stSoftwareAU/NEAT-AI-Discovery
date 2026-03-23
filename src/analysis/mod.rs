//! Analysis module for NEAT-AI Discovery
//!
//! This module provides functions for analysing recorded discovery data to identify
//! beneficial new synapses and neurons that would reduce error.
//!
//! The module is organised into thematic subdirectories (Issue #528):
//!
//! ## Thematic subdirectories
//! - `detection/` - Pattern detection modules (saturation, bottleneck, dead neuron, etc.)
//! - `recommendation/` - Candidate recommendation engines (epistatic, multi-hop, etc.)
//! - `scoring/` - Scoring, confidence, and validation (weights, confidence, etc.)
//!
//! ## Core modules (root level)
//! - `shared/` - Common types, result structures, diagnostics (timing, metadata, `gpu_info`)
//! - `synapse/` - Synapse analysis pipeline
//! - `neuron/` - Neuron analysis (evaluation, post-processing, preparation)
//! - `gpu/` - GPU infrastructure (`GpuAnalyzer`, `GpuWorkQueue`)
//! - `utils/` - Utility functions (memory checks, deadlines)
//! - `diagnostics/` - Diagnostic tracking and rejection reasons
//! - `system.rs` - System utilities facade (memory, GPU tier detection)
//! - `activation.rs` - Activation function related code
//! - `samples/` - Sample data structures and GPU formats
//! - `constants.rs` - Central discovery thresholds and constants
//! - `cache/` - Record caching for parquet files (Issue #565)
//! - `candidate_cache.rs` - Candidate outcome cache for success/failure tracking
//! - `streaming.rs` - Streaming parquet loading with block-based caching
//! - `discovery_dispatch.rs` - Generic discovery module dispatch pattern
//! - `candidate_clustering.rs` - Candidate clustering to reduce redundant ablation tests
//! - `module_weights.rs` - Per-module success rate tracking for adaptive weighting
//! - `neuron_fingerprint.rs` - Neuron structural fingerprinting for incremental analysis
//! - `early_termination.rs` - SPRT-based early termination for GPU evaluation

// Core modules (remain at root level)
pub mod activation;
pub mod cache;
pub mod candidate_cache;
pub mod candidate_clustering;
pub mod candidate_compression;
pub mod candidate_diversity;
pub mod constants;
pub mod diagnostics;
pub mod discovery_dispatch;
pub mod early_termination;
pub mod ensemble_scoring;
pub mod gpu;
pub mod module_weights;
pub mod neuron;
pub mod neuron_fingerprint;
pub mod samples;
pub mod shared;
pub mod streaming;
pub mod synapse;
pub mod system;
pub mod utils;

// Thematic subdirectories (Issue #528)
pub mod detection;
pub mod recommendation;
pub mod scoring;

// Orchestration sub-modules (Issue #562)
pub mod candidate_aggregation;
pub(crate) mod module_dispatch_specs;
mod orchestration;

// Re-export analyze_all from orchestration sub-module
pub use orchestration::analyze_all;

// Re-export candidate aggregation functions used by discovery_dispatch and sub-modules
pub(crate) use candidate_aggregation::merge_coordinated_structural_replacements;

// Re-export helper functions for unit tests (accessed via `super::*` in mod_tests)
#[cfg(test)]
pub(crate) use candidate_aggregation::apply_kept_neuron_candidates;
#[cfg(test)]
pub(crate) use orchestration::{choose_deadline_order_synapse_first, run_optional_analysis};

// Core public API entry points
pub use gpu::GpuAnalyzer;
pub use gpu::supports_unified_memory;
pub use neuron::analyze_neurons;
pub use synapse::analyze_synapses;
// Re-export benchmark helper functions for use in benches/
pub use synapse::analyze_synapses_with_cache_and_gpu_queue;
pub use synapse::deterministic_coordinated_neuron_uuid;

#[cfg(test)]
mod mod_tests;
