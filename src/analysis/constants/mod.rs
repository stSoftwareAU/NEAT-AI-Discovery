//! Central constants module for discovery thresholds (Issue #424, #938).
//!
//! This module is the single source of truth for all discovery detection
//! constants and thresholds. Previously these were duplicated across
//! individual analysis modules (consolidated in Issue #424), then split
//! into thematic sub-modules for maintainability (Issue #938).
//!
//! ## Sub-modules
//!
//! - `sample_thresholds` — Sample count thresholds and hold-out validation
//! - `sentinel_detection` — Sentinel values and clustering thresholds
//! - `source_variance` — Source variance filtering thresholds
//! - `candidate_scoring` — Scoring boosts, pessimism discounts, calibration, comparisons
//! - `compression` — Candidate compression thresholds
//! - `detection_thresholds` — Detection filtering thresholds (removal, weight constraints)
//!
//! All constants are re-exported from this module for backward compatibility.

mod candidate_scoring;
mod compression;
mod detection_thresholds;
mod sample_thresholds;
mod sentinel_detection;
mod source_variance;

// Re-export all constants and functions for backward compatibility.
// Consumers can continue to use `crate::analysis::constants::CONSTANT_NAME`.
pub use candidate_scoring::*;
pub use compression::*;
pub use detection_thresholds::*;
pub use sample_thresholds::*;
pub use sentinel_detection::*;
pub use source_variance::*;
