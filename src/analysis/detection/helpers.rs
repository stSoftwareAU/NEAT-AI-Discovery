//! Shared helper utilities for detection modules (Issues #804, #941).
//!
//! Extracts common boilerplate patterns used across detection modules into
//! reusable functions, reducing duplication and improving maintainability.
//!
//! ## Helpers provided
//!
//! | Helper | Pattern | Modules using it |
//! |--------|---------|------------------|
//! | [`build_record_map`] | UUID → records lookup | 15+ |
//! | [`compute_activation_stats`] | Mean, variance, std dev | 16+ |
//! | [`compute_activation_range`] | Min/max activation | 8+ |
//! | [`compute_mean_abs_activation`] | Mean absolute activation | 6+ |
//! | [`sort_candidates_by_score_gain`] | Descending sort on score gain | 20+ |
//! | [`weighted_confidence`] | Multi-factor confidence scoring | 14+ |

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)

use std::collections::HashMap;

use crate::CoordinatedStructuralCandidateJson;
use crate::types::DiscoverRecord;

/// Build a lookup map from neuron UUID strings to their discovery records.
///
/// Many detection modules need to look up records by neuron UUID. This
/// helper extracts the common pattern of converting `&[(String, Vec<DiscoverRecord>)]`
/// into a `HashMap<&str, &Vec<DiscoverRecord>>` for O(1) lookups.
///
/// # Arguments
/// * `neuron_records` - Slice of `(neuron_uuid, records)` tuples.
///
/// # Returns
/// A `HashMap` mapping borrowed UUID strings to borrowed record vectors.
pub fn build_record_map(
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> HashMap<&str, &Vec<DiscoverRecord>> {
    neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect()
}

// =============================================================================
// Activation statistics (Issue #941)
// =============================================================================

/// Summary statistics for neuron activations.
///
/// Returned by [`compute_activation_stats`]. Contains the mean, population
/// variance, and standard deviation of a set of activation values.
#[derive(Debug, Clone, Copy)]
pub struct ActivationStats {
    /// Arithmetic mean of the activation values.
    pub mean: f32,
    /// Population variance of the activation values.
    pub variance: f32,
    /// Population standard deviation (square root of variance).
    pub std_dev: f32,
}

/// Compute mean, variance, and standard deviation of activation values
/// from discovery records.
///
/// This is the single most duplicated computation across detection modules
/// (16+). Extracting it reduces 8–12 lines of boilerplate per module to a
/// single function call.
///
/// Returns zeroed [`ActivationStats`] for empty input.
///
/// # Examples
/// ```
/// # use neat_ai_discovery::analysis::detection::helpers::compute_activation_stats;
/// # use neat_ai_discovery::types::DiscoverRecord;
/// let records = vec![
///     DiscoverRecord::new(0, "n1".into(), Some(1.0), 1.0, vec![]),
///     DiscoverRecord::new(1, "n1".into(), Some(3.0), 3.0, vec![]),
/// ];
/// let stats = compute_activation_stats(&records);
/// assert!((stats.mean - 2.0).abs() < 1e-6);
/// ```
pub fn compute_activation_stats(records: &[DiscoverRecord]) -> ActivationStats {
    if records.is_empty() {
        return ActivationStats {
            mean: 0.0,
            variance: 0.0,
            std_dev: 0.0,
        };
    }

    let n = records.len() as f32;
    let mean: f32 = records.iter().map(|r| r.activation).sum::<f32>() / n;
    let variance: f32 = records
        .iter()
        .map(|r| {
            let diff = r.activation - mean;
            diff * diff
        })
        .sum::<f32>()
        / n;

    ActivationStats {
        mean,
        variance,
        std_dev: variance.sqrt(),
    }
}

// =============================================================================
// Activation range (Issue #941)
// =============================================================================

/// Observed activation range (min, max, span).
///
/// Returned by [`compute_activation_range`].
#[derive(Debug, Clone, Copy)]
pub struct ActivationRange {
    /// Minimum observed activation.
    pub min: f32,
    /// Maximum observed activation.
    pub max: f32,
    /// Span of the range (`max - min`).
    pub span: f32,
}

/// Compute the minimum, maximum, and span of activation values from records.
///
/// Used by restricted-range, output-range-compression, and other detection
/// modules that compare observed activation range to theoretical bounds.
///
/// For empty input, returns `min = INFINITY`, `max = NEG_INFINITY`, `span`
/// will be negative (callers should check `records.is_empty()` or `span > 0`).
pub fn compute_activation_range(records: &[DiscoverRecord]) -> ActivationRange {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for r in records {
        min = min.min(r.activation);
        max = max.max(r.activation);
    }
    ActivationRange {
        min,
        max,
        span: max - min,
    }
}

// =============================================================================
// Mean absolute activation (Issue #941)
// =============================================================================

/// Compute the mean absolute activation from discovery records.
///
/// Used by dead-neuron, low-impact-neuron, oscillating-neuron and other
/// modules that check whether a neuron's output magnitude is negligible.
///
/// Returns `0.0` for empty input.
pub fn compute_mean_abs_activation(records: &[DiscoverRecord]) -> f32 {
    if records.is_empty() {
        return 0.0;
    }
    let n = records.len() as f32;
    records.iter().map(|r| r.activation.abs()).sum::<f32>() / n
}

// =============================================================================
// Candidate sorting (Issue #941)
// =============================================================================

/// Sort coordinated structural candidates by `expected_creature_score_gain`
/// in descending order (best improvement first).
///
/// Every detection module's `*_to_coordinated_candidates` function ends with
/// this identical sort. Extracting it into a shared helper ensures consistent
/// ordering and reduces one-line boilerplate across 20+ modules.
pub fn sort_candidates_by_score_gain(candidates: &mut [CoordinatedStructuralCandidateJson]) {
    candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });
}

// =============================================================================
// Weighted confidence scoring (Issue #941)
// =============================================================================

/// A single factor contributing to a confidence score.
///
/// Used with [`weighted_confidence`] to compute multi-factor confidence
/// scores in a consistent way across detection modules.
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceFactor {
    /// Normalised factor value in `[0.0, 1.0]`. Values outside this range
    /// are clamped before use.
    pub value: f32,
    /// Relative weight of this factor (e.g., `0.4` for 40%).
    pub weight: f32,
}

/// Compute a weighted confidence score from multiple factors.
///
/// Many detection modules compute confidence as:
/// ```text
/// raw = Σ(factor_i × weight_i)
/// confidence = floor + raw × (ceiling - floor)
/// ```
///
/// This helper standardises that pattern.
///
/// # Arguments
/// * `factors` — slice of [`ConfidenceFactor`] values (weights should sum to 1.0
///   for intuitive results, but this is not enforced).
/// * `floor` — minimum confidence value (returned when all factors are zero).
/// * `ceiling` — maximum confidence value (returned when all factors are 1.0).
///
/// # Returns
/// A confidence score in `[floor, ceiling]`.
pub fn weighted_confidence(factors: &[ConfidenceFactor], floor: f32, ceiling: f32) -> f32 {
    let raw: f32 = factors
        .iter()
        .map(|f| f.value.clamp(0.0, 1.0) * f.weight)
        .sum();
    floor + raw * (ceiling - floor)
}
