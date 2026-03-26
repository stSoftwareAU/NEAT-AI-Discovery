//! Source variance filtering thresholds.
//!
//! Thresholds for filtering source activations based on variance,
//! preventing over-prediction from constant or near-constant sources.

// =============================================================================
// Source Variance Thresholds
// =============================================================================

/// Minimum source activation standard deviation for full credit.
///
/// Sources with std dev below this are progressively discounted to avoid
/// over-prediction from constant-ish sources. Also used as the reference
/// value for dynamic constant-source threshold scaling.
///
/// ## Context
/// Based on production analysis: input-1064 had std dev 0.01 and caused
/// massive over-prediction. Sources should have at least 0.05 std dev for
/// reliable correlation.
///
/// ## Valid Range
/// Must be > 0.0. Values above 0.1 may discard useful low-variance sources.
pub const MIN_SOURCE_STD_DEV: f32 = 0.05;
