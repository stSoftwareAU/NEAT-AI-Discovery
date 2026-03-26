//! Sentinel value detection and clustering thresholds.
//!
//! Constants for identifying sentinel/null values in normalised data
//! and determining whether value clusters are meaningful.

// =============================================================================
// Sentinel Detection Constants
// =============================================================================

/// Candidate sentinel values to check for clusters.
///
/// These are the most common sentinel values in normalised data.
/// Used by observation range, bounded range, and sentinel gating modules.
pub const CANDIDATE_SENTINELS: [f32; 3] = [-1.0, 0.0, 1.0];

/// Minimum fraction of samples at a sentinel value to consider it a cluster.
///
/// If fewer than this fraction of samples cluster at a candidate sentinel
/// value, it is not considered a meaningful sentinel.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.10 may flag noise as sentinels.
pub const MIN_SENTINEL_FRACTION: f32 = 0.15;

/// Tolerance for grouping values into a sentinel cluster.
///
/// Values within this distance of a candidate sentinel are considered part
/// of the cluster. Also used as the default sentinel tolerance for
/// range-aware weight calculations.
///
/// ## Valid Range
/// Must be > 0.0. Values above 0.1 may merge distinct value groups.
pub const SENTINEL_TOLERANCE: f32 = 0.02;

/// Minimum gap between a sentinel cluster and the useful value range.
///
/// If the gap between the sentinel cluster and the nearest useful value is
/// smaller than this threshold, the values are too interleaved to separate
/// reliably.
///
/// ## Valid Range
/// Must be > `SENTINEL_TOLERANCE`. Values above 0.2 may miss valid sentinels.
pub const MIN_SENTINEL_GAP: f32 = 0.05;
