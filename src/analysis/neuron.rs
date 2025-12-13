//! Neuron analysis module
//!
//! This module contains functions for analysing neuron candidates - identifying
//! beneficial new neurons that would reduce error.

use crate::analysis::AnalyzeNeuronsResult;
use crate::AnalyzeNeuronsInput;
use anyhow::Result;

// TODO: Move neuron analysis functions from impl.rs here:
// - analyze_neurons
// - analyze_neurons_with_cache
// - evaluate_neuron_candidates
// - evaluate_relu_candidates_split
// - evaluate_activation_candidate
// - calculate_optimal_bias
// - calculate_optimal_outgoing_weight
// - Related helper functions

/// Analyze neurons for a given input.
/// This is the public entry point for neuron analysis.
pub fn analyze_neurons(input: &AnalyzeNeuronsInput) -> Result<AnalyzeNeuronsResult> {
    // Temporarily delegate to implementation until refactoring is complete
    // Access via parent module to avoid circular dependency
    super::implementation::analyze_neurons(input)
}
