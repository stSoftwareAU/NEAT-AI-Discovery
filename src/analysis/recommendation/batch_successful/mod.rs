//! Batch-successful candidate grouping module (Issue #965).
//!
//! Groups multiple individually high-confidence candidates into combined
//! operations for batch application. Unlike epistatic pair detection (which
//! finds synergistic pairs that individually fail), this module batches
//! proven winners for combined testing.
//!
//! ## Rationale
//!
//! If adding synapse A improves the score and adding synapse B improves the
//! score, adding both A and B together may produce an even better result.
//! This module identifies such candidates and emits coordinated structural
//! candidates for batch testing.
//!
//! ## Detection Strategy
//!
//! 1. Evaluate individual source → target pairs from recorded data
//! 2. Identify candidates with high predicted improvement (individually successful)
//! 3. Check for structural conflicts (no duplicate source→target pairs)
//! 4. Group non-conflicting candidates into batches of 2–4 operations
//! 5. Apply per-op-count empirical discount via the merge pipeline
//!
//! ## Module Structure
//!
//! - `detection` — Individual candidate detection from recorded data
//! - `grouping` — Conflict detection, batch formation, conversion

mod detection;
mod grouping;

pub use detection::detect_individually_successful;
pub use grouping::{
    batch_successful_to_coordinated_candidates, detect_batch_successful_groups, group_into_batches,
    has_structural_conflict,
};

/// A single individually successful candidate identified from recorded data.
///
/// Represents a source → target synapse addition that individually reduces
/// the target neuron's prediction error.
#[derive(Debug, Clone)]
pub struct IndividualCandidate {
    /// Source neuron UUID.
    pub source_uuid: String,
    /// Target neuron UUID.
    pub target_uuid: String,
    /// Optimal weight for the source → target synapse.
    pub weight: f32,
    /// Fraction of target error variance explained (0.0–1.0).
    pub improvement: f32,
    /// Number of shared observation samples used for estimation.
    pub sample_count: usize,
}

/// A batch of non-conflicting individually successful candidates.
///
/// All candidates in a batch can be applied together without structural
/// conflicts (no duplicate source→target pairs).
#[derive(Debug, Clone)]
pub struct BatchSuccessfulGroup {
    /// The individually successful candidates in this batch.
    pub candidates: Vec<IndividualCandidate>,
    /// Sum of individual improvements (before coordinated discount).
    pub combined_improvement: f32,
    /// Descriptive reason for this batch.
    pub reason: String,
}
