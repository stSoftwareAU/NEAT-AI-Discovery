//! Shared types and structures used across analysis modules

use crate::{CandidateNeuronJson, CandidateSynapseJson};

/// Result of synapse analysis
#[derive(Debug)]
pub struct AnalyzeSynapsesResult {
    pub helpful_synapses: Vec<CandidateSynapseJson>,
    pub harmful_synapses: Vec<CandidateSynapseJson>,
    pub gpu_used: bool,
    pub no_candidate_reasons: Vec<SynapseNoCandidateSummary>,
}

/// Result of neuron analysis
#[derive(Debug)]
pub struct AnalyzeNeuronsResult {
    pub helpful_neurons: Vec<CandidateNeuronJson>,
    pub gpu_used: bool,
    pub no_candidate_reasons: Vec<NeuronNoCandidateSummary>,
}

/// Combined result of both synapse and neuron analysis
#[derive(Debug)]
pub struct AnalyzeAllResult {
    pub synapse: Option<AnalyzeSynapsesResult>,
    pub neuron: Option<AnalyzeNeuronsResult>,
}

/// Reason why no synapse candidate was found for a target neuron
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynapseNoCandidateReason {
    NoEligibleSources,
    NoDiagnostics,
    NoSamples,
    ZeroImprovement,
    BelowThreshold,
}

/// Detailed information about why a synapse candidate was rejected
#[derive(Debug, Clone)]
pub struct SynapseNoCandidateDetail {
    pub source_uuid: Option<String>,
    pub sample_count: Option<usize>,
    pub source_record_count: Option<usize>,
    pub improved_count: Option<u32>,
    pub worsened_count: Option<u32>,
    pub expected_improvement: Option<f32>,
    pub threshold: Option<f32>,
    pub suggested_weight: Option<f32>,
}

/// Summary of why no synapse candidate was found for a target neuron
#[derive(Debug, Clone)]
pub struct SynapseNoCandidateSummary {
    pub target_uuid: String,
    pub reason: SynapseNoCandidateReason,
    pub evaluated_candidates: u32,
    pub candidates_with_samples: u32,
    pub target_record_count: usize,
    pub detail: Option<SynapseNoCandidateDetail>,
}

/// Reason why no neuron candidate was found for a target neuron
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeuronNoCandidateReason {
    NoEligibleSources,
    NoDiagnostics,
    NoSamples,
    NotEnoughActivations,
    WeightDegenerate,
    BelowThreshold,
    /// Hidden neurons are filtered out from add-neuron analysis because their
    /// backpropagated errors don't reliably translate to output error reduction.
    HiddenNeuronFiltered,
    /// Input neurons are filtered out from add-neuron analysis because they're
    /// observation sources, not computation nodes - they have no activation function
    /// or error to reduce.
    InputNeuronFiltered,
    /// Constant neurons are filtered out from add-neuron analysis because they
    /// don't receive inputs - they always output a fixed value regardless of
    /// network state, so adding a connection to them has no effect.
    ConstantNeuronFiltered,
}

/// Detailed information about why a neuron candidate was rejected
#[derive(Debug, Clone)]
pub struct NeuronNoCandidateDetail {
    pub source_uuid: Option<String>,
    pub orientation: Option<String>,
    pub sample_count: Option<usize>,
    pub improved_count: Option<u32>,
    pub worsened_count: Option<u32>,
    pub expected_improvement: Option<f32>,
    pub threshold: Option<f32>,
    pub outgoing_weight: Option<f32>,
}

/// Summary of why no neuron candidate was found for a target neuron
#[derive(Debug, Clone)]
pub struct NeuronNoCandidateSummary {
    pub target_uuid: String,
    pub reason: NeuronNoCandidateReason,
    pub evaluated_sources: u32,
    pub sources_with_samples: u32,
    pub target_record_count: usize,
    pub detail: Option<NeuronNoCandidateDetail>,
}
