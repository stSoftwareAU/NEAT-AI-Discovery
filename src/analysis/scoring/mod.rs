//! Scoring modules — scoring, confidence, and validation.
//!
//! This subdirectory groups modules responsible for computing confidence
//! metrics, weight calculations, error distribution analysis, and
//! cross-validation scoring.

pub mod calibration_correction;
pub mod confidence;
pub mod cross_validation;
pub mod error_distribution;
pub mod weights;
