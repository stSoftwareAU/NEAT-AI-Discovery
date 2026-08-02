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
//! - `constants/` - Central discovery thresholds and constants (thematic sub-modules)
//! - `cache/` - Record caching for parquet files (Issue #565)
//! - `cache_study/` - Study of the production discovery candidates cache (Issue #1920)
//! - `streaming.rs` - Streaming parquet loading with block-based caching
//! - `discovery_dispatch.rs` - Generic discovery module dispatch pattern
//! - `candidate_clustering.rs` - Candidate clustering to reduce redundant ablation tests
//! - `module_weights.rs` - Per-module success rate tracking for adaptive weighting
//! - `neuron_fingerprint.rs` - Neuron structural fingerprinting for incremental analysis
//! - `early_termination.rs` - SPRT-based early termination for GPU evaluation

// Core modules (remain at root level)
pub mod activation;
pub mod analysis_outcome;
pub mod cache;
pub mod cache_study;
pub mod candidate_clustering;
pub mod candidate_compression;
pub mod candidate_diversity;
pub mod candidate_reconciliation;
pub mod candidate_starvation;
pub mod change_squash_gain;
pub mod constants;
pub mod cost_function_hint;
pub mod creature_drought_alarm;
pub mod deadline_breakdown;
pub mod diagnostics;
pub mod discovery_dispatch;
pub mod discovery_mode;
pub mod dominated_branch_collapse;
pub mod drought_diagnostic;
pub mod drought_reset;
pub mod early_termination;
pub mod ensemble_scoring;
pub mod evaluation_drops;
pub mod failure_cache_handshake;
pub mod fingerprint_skip_escape;
pub mod gpu;
pub mod insufficient_recording;
pub mod merge_redundant_neuron;
pub mod module_tiering;
pub mod module_weights;
pub mod neuron;
pub mod neuron_fingerprint;
pub mod novelty_escalation;
pub mod one_hot_class_allocation;
/// Regression-test scaffolding, not shipped API — gated behind the
/// off-by-default `regression-harness` feature (Issue #1877).
#[cfg(feature = "regression-harness")]
pub mod production_discovery_regression;
pub mod quantised_error;
pub mod recent_failure_window;
pub mod remove_neuron_bias_fold;
pub mod remove_neuron_compensation;
pub mod remove_neuron_constant_promotion;
pub mod remove_neuron_drought;
pub mod remove_neuron_gain;
pub mod remove_neuron_net_gain;
#[cfg(test)]
mod remove_neuron_regression_test;
pub mod samples;
pub mod scale_outcomes;
pub mod shared;
pub mod streaming;
pub mod synapse;
pub mod system;
pub mod target_failure_tracker;
pub mod target_pass_outcomes;
pub mod task_descriptor;
pub mod utils;
pub mod within_batch_failures;

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
// Issue #1128: The coordinated gain-floor helper is public so that the final
// orchestration step and integration tests can apply it after downstream
// discount passes (module boost, diversity reranking, calibration).
pub use candidate_aggregation::apply_coordinated_gain_floor;

// Re-export helper functions for unit tests (accessed via `super::*` in mod_tests)
#[cfg(test)]
pub(crate) use candidate_aggregation::apply_kept_neuron_candidates;
#[cfg(test)]
pub(crate) use orchestration::run_optional_analysis;

// Issue #1421: outcome classification distinguishing environmentally-disabled
// passes from genuine search exhaustion.
pub use analysis_outcome::{AnalysisOutcome, EnvironmentalDisableReason, PassOutcomeCounts};

// Issue #1518: propagation-aware remove-neuron gain estimator (replaces the
// fabricated floor-at-0.1 placeholder).
pub use remove_neuron_gain::estimate_remove_neuron_gain;

// Issue #1812: the sole-op remove-neuron net-gain rule — the estimator's
// unitless cost converted onto the creature-score scale and netted against the
// exact complexity saving, plus the floor that screens it.
pub use remove_neuron_net_gain::{
    analysis_cost_of_growth, estimate_remove_neuron_net_gain, removal_influence_loss,
    removal_net_gain, removal_net_gain_accepted,
};

// Issue #1532: propagation-aware change-squash gain estimator (extends the
// #1518 approach to the change-squash estimate path).
pub use change_squash_gain::estimate_change_squash_gain;

// Issue #1519: decoupled hygiene removal-eligibility (squash-error threshold)
// from the honest ranking gain.
pub use remove_neuron_gain::{
    MAX_REASONABLE_SQUASH_ERROR, RemoveNeuronAssessment, assess_remove_neuron,
};

// Issue #1622: promote flagged functionally-constant hidden neurons past the
// #1518 gain ranking and #1448 drought demotion as priority remove-neuron
// candidates.
pub use remove_neuron_constant_promotion::{
    CONSTANT_NEURON_PRIORITY_GAIN, bias_folded_constant_neuron_uuids,
    functionally_constant_neuron_uuids, promote_constant_remove_neuron_candidates,
};

// Issue #1623: bias-fold removal for functionally-constant hidden neurons —
// fold each neuron's constant downstream contribution into its targets' biases
// behind the evaluate-before-accept gate.
pub use remove_neuron_bias_fold::{
    BIAS_FOLD_GATE_TOLERANCE, BiasFoldOutcome, FoldedTarget, evaluate_constant_neuron_bias_fold,
    fold_and_remove_constant_neuron,
};

// Issue #1711: analytical dominated-branch collapse for MAX/MIN selection
// aggregates — prove a branch's `weight × squash(range)` can never win the
// aggregate, then remove it and fold the single-survivor aggregate to a
// pass-through, behind the evaluate-before-accept gate.
pub use dominated_branch_collapse::{
    AggregateKind, COLLAPSE_GATE_TOLERANCE, CollapseOutcome, DominatedBranch, SquashSign,
    analytically_dominated_branch_uuids, collapse_dominated_branch, detect_dominated_branches,
    evaluate_dominated_branch_collapse, scalar_squash_sign,
};

// Issue #1633: merge/fold redundant (highly-correlated) hidden neurons — fold
// the lower-impact neuron's fan-out into its twin (scaled by the fitted linear
// relationship) and remove it, behind the evaluate-before-accept ablation gate.
pub use merge_redundant_neuron::{
    MERGE_CORRELATION_THRESHOLD, RedundantNeuronPair, detect_redundant_neuron_pairs,
    detect_redundant_neuron_pairs_with_threshold, redundant_pairs_to_coordinated_candidates,
};

// Issue #1559: weight-redistribution compensation for remove-neuron candidates
// — persist a compact covariance sufficient statistic and evaluate
// counterfactual (d) from the #1558 study.
pub use remove_neuron_compensation::{
    ActivationCovariance, SharedTarget, WeightRedistribution, aligned_activations,
    best_weight_redistribution, evaluate_weight_redistribution, shared_downstream_targets,
};

// Core public API entry points
pub use gpu::GpuAnalyzer;
pub use gpu::supports_unified_memory;
pub use neuron::analyze_neurons;
pub use neuron::analyze_neurons_with_cache_and_gpu_queue;
pub use synapse::analyze_synapses;
// Re-export benchmark helper functions for use in benches/
pub use synapse::analyze_synapses_with_cache_and_gpu_queue;
pub use synapse::deterministic_coordinated_neuron_uuid;

#[cfg(test)]
mod mod_tests;
