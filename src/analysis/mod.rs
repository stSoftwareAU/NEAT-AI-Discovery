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
//! - `shared.rs` - Common types, result structures, diagnostics
//! - `synapse/` - Synapse analysis pipeline
//! - `neuron.rs` - Neuron analysis functions
//! - `gpu/` - GPU infrastructure (GpuAnalyzer, GpuWorkQueue)
//! - `utils/` - Utility functions (memory checks, deadlines)
//! - `diagnostics/` - Diagnostic tracking and rejection reasons
//! - `system.rs` - System utilities facade (memory, GPU tier detection)
//! - `activation.rs` - Activation function related code
//! - `samples.rs` - Sample data structures and GPU formats
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

// Re-export detection modules at analysis level for backward compatibility
pub use detection::activation_mismatch;
pub use detection::bias_perturbation;
pub use detection::bottleneck;
pub use detection::bounded_range;
pub use detection::correlated_error;
pub use detection::dead_neuron;
pub use detection::dormant_synapse;
pub use detection::error_plateau;
pub use detection::input_sensitivity;
pub use detection::noise_signal;
pub use detection::observation_range;
pub use detection::observation_utilisation;
pub use detection::operating_point;
pub use detection::opposing_synapse;
pub use detection::oscillating_neuron;
pub use detection::output_squash_mismatch;
pub use detection::redundant_path;
pub use detection::restricted_range;
pub use detection::saturation;
pub use detection::sentinel_gating;
pub use detection::skip_connection;
pub use detection::squash_weight_rescale;
pub use detection::symmetry_breaking;
pub use detection::topology;
pub use detection::topology_diversification;
pub use detection::unbounded_capping;
pub use detection::weight_coherence;
pub use detection::weight_magnitude_reset;

// Re-export recommendation modules at analysis level for backward compatibility
pub use recommendation::activation_recommendation;
pub use recommendation::epistatic;
pub use recommendation::gradient_discovery;
pub use recommendation::multi_hop;
pub use recommendation::output_bias_drift;
pub use recommendation::sample_weighted;

// Re-export scoring modules at analysis level for backward compatibility
pub use scoring::confidence;
pub use scoring::cross_validation;
pub use scoring::error_distribution;
pub use scoring::weights;

// Re-export shared types
pub use shared::{
    // GPU timing types (Issue #195)
    AnalysisTiming,
    AnalyzeAllResult,
    AnalyzeNeuronsResult,
    AnalyzeSynapsesResult,
    CpuTimingBreakdown,
    // GPU info and zero-copy types (Issue #228)
    GpuAdapterInfo,
    GpuDeviceType,
    GpuTimingBreakdown,
    NeuronAnalysisMetadata,
    NeuronNoCandidateReason,
    NeuronNoCandidateSummary,
    ShaderTiming,
    SynapseAnalysisMetadata,
    SynapseNoCandidateReason,
    SynapseNoCandidateSummary,
    TimingCollector,
    TimingScope,
    ZeroCopyBufferConfig,
};

// Re-export confidence interval types (Issue #194)
pub use confidence::{PredictionConfidenceMetrics, compute_confidence_metrics};

// Re-export from utils
pub use utils::{gpu_timing_enabled, verbose_enabled};

// Re-export memory functions from utils (Issue #267)
pub use utils::check_memory_for_parquet;

// Re-export system utilities for backward compatibility (Issue #239)
// These are the primary types and functions for memory detection and GPU performance
pub use system::{
    // GPU batch size constants
    DEFAULT_GPU_BATCH_SIZE,
    GpuPerformanceTier,
    HIGH_PERF_GPU_BATCH_SIZE,
    LOW_MEMORY_GPU_BATCH_SIZE,
    MemoryTier,
    // GPU batch size utilities
    cap_gpu_batch_size_by_bytes,
    // Memory tier classification
    categorise_memory_tier,
    // System requirements
    check_system_memory_requirements,
    // GPU performance tier
    detect_gpu_tier,
    detect_memory_tier,
    detect_unified_memory,
    // Memory detection
    get_memory_info,
    get_work_queue_capacity,
    get_work_queue_capacity_for_tier,
    // Parquet memory validation
    validate_parquet_memory_requirements,
};

// Re-export Detail types from shared
pub use shared::{NeuronNoCandidateDetail, SynapseNoCandidateDetail};

// Re-export activation-related items from activation module (Issue #266, #238)
pub use activation::{
    ACTIVATION_SPECS,
    // Activation candidate spec
    ActivationCandidateSpec,
    ORIENTATIONS_BIDIRECTIONAL,
    SCALES_SMOOTH,
    SCALES_WIDE,
    TargetSimulationMode,
    // Activation functions
    absolute_activation,
    // GPU ID mapping
    activation_name_to_gpu_id,
    arctan_activation,
    bent_identity_activation,
    bipolar_activation,
    // Target simulation functions (Issue #238)
    can_use_hard_tanh,
    clipped_activation,
    elu_activation,
    gelu_activation,
    // Bias helpers
    get_bias_range,
    get_bias_values,
    get_target_activation_fn,
    get_target_simulation_fn,
    get_target_simulation_mode,
    hard_tanh_activation,
    has_sufficient_output_variance,
    identity_activation,
    // Activation predicates
    is_threshold_activation,
    logistic_activation,
    mish_activation,
    relu6_activation,
    softplus_activation,
    softsign_activation,
    tanh_activation,
};

// Re-export sample data structures from samples module (Issue #269)
pub use samples::{
    ActivationOutput, ActivationUniforms, BiasResult, BiasUniforms,
    DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD, EPSILON, GpuHelpfulSample, HarmfulContribution,
    HarmfulStats, HarmfulUniforms, HelpfulContribution, HelpfulSample, HelpfulStats,
    HelpfulUniforms, NeuronStats, ReluContribution, ReluOrientation, ReluStats, ReluUniforms,
    compute_dynamic_constant_source_threshold, compute_source_std_dev,
    compute_source_variance_discount, constant_source_effect_threshold_from_env,
    get_constant_source_threshold,
};

// Re-export focus_unused_observations_from_env for tests (Issue #182)
// Moved to utils/deadline module in Issue #268
pub use utils::focus_unused_observations_from_env;

// Re-export weight calculation functions from weights module (Issue #270, #402)
pub use weights::{
    DEFAULT_SENTINEL_TOLERANCE, MAX_OUTGOING_WEIGHT, calculate_optimal_bias,
    calculate_optimal_identity_outgoing_and_bias, calculate_optimal_outgoing_weight,
    calculate_range_aware_weight, clamp_weight_update_delta, compute_range_aware_sums,
    coordinated_structural_activation_delta,
};

// Re-export error distribution types (Issue #192)
pub use error_distribution::{
    ErrorDistribution, ErrorMode, OutlierReductionInfo, detect_error_modes,
    outlier_analysis_enabled, outlier_percentile_from_env,
};

// Orchestration sub-modules (Issue #562)
pub(crate) mod candidate_aggregation;
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

// Re-export from modules
pub use gpu::GpuAnalyzer;
pub use gpu::GpuAvailabilityResult;
pub use gpu::supports_unified_memory;
pub use neuron::analyze_neurons;
pub use synapse::analyze_synapses;
// Re-export benchmark helper function for use in benches/
pub use synapse::analyze_synapses_with_cache_and_gpu_queue;

// Re-export early termination types (Issue #219)
pub use early_termination::{
    EarlyTerminationConfig, EarlyTerminationDecision, EarlyTerminationResult, SequentialEvaluator,
    check_batch_early_termination,
};

// Re-export cross-validation types (Issue #436)
pub use cross_validation::{
    CrossValidationConfig, CrossValidationResult, FoldResult, PerformanceVariance,
    apply_brittleness_penalty, compute_cross_validation_score,
};

#[cfg(test)]
mod mod_tests;
