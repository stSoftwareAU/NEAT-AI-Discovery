//! Synapse analysis module
//!
//! This module contains functions for analysing synapse candidates - identifying
//! beneficial new synapses that would reduce error.

use crate::analysis::AnalyzeSynapsesResult;
use crate::AnalyzeSynapsesInput;
use anyhow::Result;

// TODO: Move synapse analysis functions from impl.rs here:
// - analyze_synapses
// - analyze_synapses_with_cache
// - evaluate_synapse_candidates
// - build_samples
// - compute_synapse_improvement_and_count
// - evaluate_discrete_candidate
// - Related helper functions

/// Analyze synapses for a given input.
/// This is the public entry point for synapse analysis.
pub fn analyze_synapses(input: &AnalyzeSynapsesInput) -> Result<AnalyzeSynapsesResult> {
    // Temporarily delegate to implementation until refactoring is complete
    // Access via parent module to avoid circular dependency
    super::implementation::analyze_synapses(input)
}
