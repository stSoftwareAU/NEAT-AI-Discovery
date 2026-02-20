//! Tests for synapse analysis core implementation (Issue #278, #425).
//!
//! These tests cover the core synapse analysis pipeline including GPU batch
//! evaluation, diagnostics, and prediction accuracy. Originally in implementation.rs,
//! moved to synapse.rs as part of Issue #425.
//!
//! Test modules:
//! - `cache_tests` — Record cache contention handling
//! - `gpu_batch_tests` — GPU batch evaluation edge cases
//! - `sample_matching_tests` — Sample matching and filtering
//! - `bias_calculation_tests` — Bias calculation for various activation functions
//! - `diagnostics_tests` — Diagnostics and rejection tracking
//! - `relu_evaluation_tests` — ReLU candidate evaluation and splitting
//! - `improvement_model_tests` — Linear and HARD_TANH improvement models
//! - `synapse_analysis_tests` — Synapse and neuron analysis integration
//! - `optimal_weight_tests` — Optimal outgoing weight calculation
//! - `prediction_accuracy_tests` — Prediction accuracy vs manual simulation
//! - `clone_reduction_tests` — Hash-based deduplication key correctness (Issue #526)

// Shared imports for all test modules
#[allow(unused_imports)]
use super::*;

// Common test utilities - all submodules import from this via super::common
// Using pub(super) to restrict visibility to this test module
#[allow(unused_imports)]
pub(super) mod common {
    pub(crate) use crate::analysis::activation::{
        get_target_simulation_fn, identity_activation, is_threshold_activation,
        logistic_activation, tanh_activation,
    };
    pub(crate) use crate::analysis::analyze_all;
    pub(crate) use crate::analysis::analyze_neurons;
    pub(crate) use crate::analysis::analyze_synapses;
    pub(crate) use crate::analysis::cache::RecordCache;
    pub(crate) use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT;
    pub(crate) use crate::analysis::diagnostics::{
        NeuronDiagnostics, RejectionReason, TargetDiagnostics, ThresholdContext,
        filter_focus_targets_for_neuron_analysis,
    };
    pub(crate) use crate::analysis::gpu::GpuAnalyzer;
    pub(crate) use crate::analysis::samples::{EPSILON, HelpfulSample, HelpfulStats};
    pub(crate) use crate::analysis::scoring::weights::{
        MAX_OUTGOING_WEIGHT, calculate_optimal_bias, calculate_optimal_outgoing_weight,
    };
    pub(crate) use crate::analysis::shared::{NeuronNoCandidateReason, SynapseNoCandidateReason};
    pub(crate) use crate::analysis::synapse::{
        build_samples, compute_candidate_dedup_key, compute_net_improvement_with_squash,
        count_improved_samples, evaluate_relu_candidates_split, upsert_candidate,
    };
    pub(crate) use crate::analysis::utils::deadline_override;
    pub(crate) use crate::parquet_format::write_records_to_parquet;
    pub(crate) use crate::types::DiscoverRecord;
    pub(crate) use crate::{
        AnalyzeAllInput, AnalyzeNeuronsInput, AnalyzeSynapsesInput, CandidateNeuronJson,
        CreatureJson, NeuronJson, SynapseJson,
    };
    pub(crate) use anyhow::Result;
    pub(crate) use std::collections::HashMap;
    pub(crate) use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    pub(crate) use std::sync::{Arc, Barrier};
    pub(crate) use std::thread;
    pub(crate) use std::time::Duration;
    pub(crate) use tempfile::tempdir;

    /// Helper macro to skip tests that require GPU when no GPU is available.
    /// This allows tests to pass gracefully in CI environments without GPUs.
    macro_rules! skip_if_no_gpu {
        () => {
            if !$crate::analysis::gpu::GpuAnalyzer::gpu_is_available() {
                eprintln!("⚠️  Skipping test: GPU not available");
                return;
            }
        };
    }

    pub(crate) use skip_if_no_gpu;
}

mod bias_calculation_tests;
mod cache_tests;
mod clone_reduction_tests;
mod diagnostics_tests;
mod gpu_batch_tests;
mod improvement_model_tests;
mod optimal_weight_tests;
mod prediction_accuracy_tests;
mod relu_evaluation_tests;
mod safe_unwrap_tests;
mod sample_matching_tests;
mod synapse_analysis_tests;
