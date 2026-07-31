//! Visualisation snapshot export module
//!
//! Exports a debug-friendly JSON snapshot of a creature and its recorded
//! activations/errors for use with the NEAT-AI-Explore visualiser.
//!
//! This is an optional debug tool that does not affect existing analysis behaviour.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

pub mod dense_bound;
mod snapshot;
mod stats;
mod timestamp;
pub mod types;

// Re-export all public items for backward compatibility
pub use snapshot::export_visualisation_snapshot;
pub use types::*;
