//! Epistatic neuron pair detection module (Issue #202) and synergistic discovery (Issue #189).
//!
//! Epistatic changes are structural modifications where no single operation improves
//! the score, but a group of operations does. This module pre-detects such relationships
//! during analysis, rather than discovering them post-hoc.
//!
//! ## Detection Strategies
//!
//! 1. **Correlation analysis**: For neuron pairs (A, B) targeting the same output,
//!    compute correlation between their predicted improvements. Flag pairs where
//!    individual improvements are low but combined might be high.
//!
//! 2. **Shared error pattern detection**: Find neurons that improve on complementary
//!    sample subsets. These are candidates for combined structural changes.
//!
//! 3. **Residual analysis** (Issue #189): Find the best single-source candidate, compute
//!    residual error after applying it, then search for a second source that reduces the
//!    residual. This is O(2n) instead of O(n²) and detects XOR-like patterns.
//!
//! ## Module Structure (Issue #563)
//!
//! - `candidate_generation` — Candidate pair generation, complementarity analysis, conversion
//! - `pre_screening` — Individual operation pre-screening via residual analysis
//! - `deduplication` — Dominant-neuron deduplication (Issue #509)
//! - `scoring` — Interference detection and candidate filtering (Issue #415)
//!
//! ## Key Functions
//!
//! - `detect_epistatic_pairs` - Main entry point for epistatic detection (Issue #202)
//! - `detect_synergistic_candidates` - Residual-based synergistic discovery (Issue #189)
//! - `detect_interfering_pairs` - Combo-successful interference detection (Issue #415)
//! - `deduplicate_by_dominant_neuron` - Dominant neuron deduplication (Issue #509)

mod candidate_generation;
mod deduplication;
mod pre_screening;
mod scoring;

use crate::analysis::samples::{HelpfulSample, HelpfulStats};

use std::collections::HashSet;

// Re-export all public items for backward compatibility
pub use candidate_generation::{
    build_source_contribution, compute_firing_indices, detect_epistatic_pairs,
    epistatic_pairs_to_coordinated_candidates,
};
pub use deduplication::{
    deduplicate_by_dominant_neuron, deduplicate_synergistic_by_dominant_neuron,
};
pub use pre_screening::{detect_synergistic_candidates, synergistic_to_coordinated_candidates};
pub use scoring::{
    detect_interfering_pairs, filter_interfering_epistatic_pairs,
    filter_interfering_synergistic_candidates,
};

/// Result of evaluating a potential epistatic pair.
#[derive(Debug, Clone)]
pub struct EpistaticPairCandidate {
    /// First source neuron UUID
    pub source_a_uuid: String,
    /// Second source neuron UUID
    pub source_b_uuid: String,
    /// Target neuron UUID
    pub target_uuid: String,
    /// Optimal weight for source A's synapse
    pub weight_a: f32,
    /// Optimal weight for source B's synapse
    pub weight_b: f32,
    /// Expected combined improvement
    pub combined_improvement: f32,
    /// Individual improvement for source A alone
    pub individual_improvement_a: f32,
    /// Individual improvement for source B alone
    pub individual_improvement_b: f32,
    /// Complementarity score (0 to 1, higher = more complementary)
    pub complementarity_score: f32,
    /// Description of why this pair is epistatic
    pub reason: String,
}

/// Result of synergistic candidate detection via residual analysis (Issue #189).
///
/// A synergistic candidate is a pair of sources where:
/// - Neither source alone provides strong improvement
/// - Together they reduce error better than either alone
/// - This is detected via residual analysis: apply best source, find second source for residual
#[derive(Debug, Clone)]
pub struct SynergisticCandidate {
    /// Primary source neuron UUID (best single-source candidate)
    pub primary_source_uuid: String,
    /// Complementary source neuron UUID (reduces residual error)
    pub complement_source_uuid: String,
    /// Target neuron UUID
    pub target_uuid: String,
    /// Optimal weight for primary source's synapse
    pub primary_weight: f32,
    /// Optimal weight for complement source's synapse
    pub complement_weight: f32,
    /// Expected combined improvement (fraction of error reduced)
    pub combined_improvement: f32,
    /// Primary source individual improvement
    pub primary_improvement: f32,
    /// Complement source individual improvement
    pub complement_improvement: f32,
    /// Residual reduction achieved by complement source (fraction)
    pub residual_reduction: f32,
    /// Synergy ratio: combined / max(individual)
    pub synergy_ratio: f32,
    /// Description of why this pair is synergistic
    pub reason: String,
}

/// Represents a source neuron's contribution to a target.
#[derive(Debug, Clone)]
pub struct SourceContribution {
    /// Source neuron UUID
    pub source_uuid: String,
    /// Samples where this source contributes to the target
    pub samples: Vec<HelpfulSample>,
    /// Computed optimal weight for this source
    pub optimal_weight: f32,
    /// Individual improvement prediction
    pub individual_improvement: f32,
    /// Set of obs_indices where source fires (activation > threshold)
    pub firing_indices: HashSet<u32>,
    /// GPU evaluation stats
    pub stats: HelpfulStats,
}

/// Type of interference detected between candidate pairs (Issue #415).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterferenceType {
    /// Two candidates target the same synapse with conflicting (opposite sign) weights.
    ConflictingWeights,
    /// Combined contributions would push target neuron into saturation.
    SaturationRisk,
    /// Two candidates have highly correlated activations (redundant).
    RedundantContribution,
}

/// Result of interference analysis for a candidate pair (Issue #415).
#[derive(Debug, Clone)]
pub struct InterferencePairResult {
    /// Source UUID of the first candidate.
    pub source_a_uuid: String,
    /// Source UUID of the second candidate.
    pub source_b_uuid: String,
    /// Type of interference detected.
    pub interference_type: InterferenceType,
    /// Severity score (0.0 to 1.0, higher = more severe interference).
    pub severity: f32,
    /// Description of the interference.
    pub reason: String,
}
