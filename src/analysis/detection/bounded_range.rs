//! Bounded range discovery module (Issue #395).
//!
//! Observations are finite numbers, often normalised to -1…1. Since observations lack
//! a concept of null, sentinel values like -1 or 0 are used instead. For example, a
//! low "Debt-to-Equity" value might meaningfully influence the output, but -1 should
//! not (it may indicate missing data entirely).
//!
//! This module detects neurons (input observations and hidden neurons) where a
//! significant cluster of activation values sits at a boundary (e.g., -1, 0, or +1),
//! separate from the "useful" range. It then recommends adding a gating neuron so that
//! the sentinel region does not negatively impact the creature's score.
//!
//! ## Detection Criteria
//!
//! A neuron has a "bounded range issue" if:
//! 1. **Boundary cluster**: A significant fraction (≥ 20%) of samples have activation
//!    at or very near a single value (the sentinel).
//! 2. **Gap**: There is a measurable gap between the sentinel cluster and the useful
//!    range of values.
//! 3. **Sufficient samples**: At least 20 samples are available.
//! 4. **Not output neurons**: Only input and hidden neurons are considered.
//!
//! ## Recommended Actions
//!
//! When a bounded range issue is detected, the module recommends adding a gating
//! neuron (using `addNeuron` + `addSynapse` coordinated operations) that can learn
//! to suppress the sentinel region while passing through useful values.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashSet;

use super::helpers::{ConfidenceFactor, sort_candidates_by_score_gain, weighted_confidence};
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

// Constants moved to constants.rs (Issue #424)
use crate::analysis::constants::{
    CANDIDATE_SENTINELS, MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_BOUNDED_RANGE,
    MIN_SENTINEL_GAP as MIN_GAP, SENTINEL_TOLERANCE as BOUNDARY_TOLERANCE,
};

/// Minimum fraction of samples at a boundary value to consider it a sentinel cluster.
/// Bounded range uses a higher threshold (0.20) than sentinel detection (0.15)
/// because boundary clustering requires stronger evidence.
const MIN_BOUNDARY_FRACTION: f32 = 0.20;

/// Result of detecting a bounded range issue on a neuron.
#[derive(Debug, Clone)]
pub struct BoundedRangeCandidate {
    /// UUID of the neuron with the bounded range issue.
    pub neuron_uuid: String,
    /// The sentinel/boundary value where the cluster sits.
    pub boundary_value: f32,
    /// Fraction of samples at the boundary value (0.0 to 1.0).
    pub boundary_fraction: f32,
    /// Minimum of the useful (non-sentinel) range.
    pub useful_range_min: f32,
    /// Maximum of the useful (non-sentinel) range.
    pub useful_range_max: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Confidence in the detection (0.0 to 1.0).
    pub detection_confidence: f32,
    /// Estimated creature score improvement from gating the sentinel region.
    pub estimated_improvement: f32,
}

/// Detect neurons with bounded range issues from recorded activations.
///
/// Analyses both input (observation) and hidden neurons to find those where a
/// significant cluster of values sits at a sentinel boundary, separate from the
/// useful range.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `BoundedRangeCandidate` sorted by detection confidence (highest first).
pub fn detect_bounded_range_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<BoundedRangeCandidate> {
    // Only consider input and hidden neurons (exclude output)
    let eligible_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input" || n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut candidates = Vec::with_capacity(neuron_records.len());

    for (uuid, records) in neuron_records {
        let records = records.as_ref();
        if !eligible_uuids.contains(uuid.as_str()) {
            continue;
        }

        if records.len() < MIN_SAMPLES_FOR_BOUNDED_RANGE {
            continue;
        }

        if let Some(candidate) = analyse_neuron_for_boundary_cluster(uuid, records) {
            candidates.push(candidate);
        }
    }

    // Sort by detection confidence (highest first)
    candidates.sort_by(|a, b| b.detection_confidence.total_cmp(&a.detection_confidence));

    candidates
}

/// Analyse a single neuron's activation records for boundary clustering.
///
/// Checks each candidate sentinel value (-1, 0, +1) and returns the strongest
/// detection if any boundary cluster is found.
fn analyse_neuron_for_boundary_cluster(
    uuid: &str,
    records: &[DiscoverRecord],
) -> Option<BoundedRangeCandidate> {
    let n = records.len() as f32;
    let activations: Vec<f32> = records.iter().map(|r| r.activation).collect();

    let mut best_candidate: Option<BoundedRangeCandidate> = None;

    for &sentinel in &CANDIDATE_SENTINELS {
        // Count samples at or near the sentinel value
        let boundary_count = activations
            .iter()
            .filter(|&&a| (a - sentinel).abs() <= BOUNDARY_TOLERANCE)
            .count();
        let boundary_fraction = boundary_count as f32 / n;

        if boundary_fraction < MIN_BOUNDARY_FRACTION {
            continue;
        }

        // Collect the non-sentinel (useful) values
        let useful_values: Vec<f32> = activations
            .iter()
            .copied()
            .filter(|&a| (a - sentinel).abs() > BOUNDARY_TOLERANCE)
            .collect();

        if useful_values.is_empty() {
            // All values are at the sentinel — no useful range to separate
            continue;
        }

        let useful_min = useful_values.iter().copied().fold(f32::INFINITY, f32::min);
        let useful_max = useful_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // Check there is a gap between the sentinel and the useful range
        let gap = if sentinel <= useful_min {
            useful_min - (sentinel + BOUNDARY_TOLERANCE)
        } else if sentinel >= useful_max {
            (sentinel - BOUNDARY_TOLERANCE) - useful_max
        } else {
            // Sentinel is inside the useful range — not a clear boundary
            0.0
        };

        if gap < MIN_GAP {
            continue;
        }

        let confidence = compute_detection_confidence(boundary_fraction, gap, n);
        let improvement = confidence * 0.005;

        let is_better = best_candidate
            .as_ref()
            .is_none_or(|prev| confidence > prev.detection_confidence);

        if is_better {
            best_candidate = Some(BoundedRangeCandidate {
                neuron_uuid: uuid.to_string(),
                boundary_value: sentinel,
                boundary_fraction,
                useful_range_min: useful_min,
                useful_range_max: useful_max,
                sample_count: records.len(),
                detection_confidence: confidence,
                estimated_improvement: improvement,
            });
        }
    }

    best_candidate
}

/// Compute detection confidence based on boundary statistics.
///
/// Higher confidence when:
/// - Boundary fraction is larger (more samples at sentinel)
/// - Gap between sentinel and useful range is wider
/// - More samples were analysed
///
/// Issue #941: refactored to use `weighted_confidence` shared helper.
fn compute_detection_confidence(boundary_fraction: f32, gap: f32, sample_count: f32) -> f32 {
    let fraction_factor = ((boundary_fraction - MIN_BOUNDARY_FRACTION)
        / (1.0 - MIN_BOUNDARY_FRACTION))
        .clamp(0.0, 1.0);
    let gap_factor = (gap / 0.5).clamp(0.0, 1.0);
    let sample_factor = (sample_count / 1000.0).min(1.0);

    weighted_confidence(
        &[
            ConfidenceFactor {
                value: fraction_factor,
                weight: 0.4,
            },
            ConfidenceFactor {
                value: gap_factor,
                weight: 0.4,
            },
            ConfidenceFactor {
                value: sample_factor,
                weight: 0.2,
            },
        ],
        0.5,
        1.0,
    )
}

/// Convert bounded range candidates into coordinated structural candidates.
///
/// Each candidate produces a coordinated operation that adds a gating neuron
/// between the source and its downstream targets. The gating neuron can learn
/// to suppress sentinel values while passing through useful-range values.
///
/// The NEAT-AI controller will validate each candidate through ablation testing
/// before applying it.
pub fn bounded_range_to_coordinated_candidates(
    candidates: &[BoundedRangeCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        // Recommend a bias that centres the gating neuron on the useful range.
        // The bias shifts the activation so that sentinel values fall below the
        // gating threshold while useful values pass through.
        let useful_centre = (c.useful_range_min + c.useful_range_max) / 2.0;
        let gate_bias = -useful_centre;

        // Create a deterministic UUID for the new gating neuron
        let gate_uuid = format!("gate-bounded-{}", c.neuron_uuid);

        results.push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            operations: vec![
                CoordinatedStructuralOpJson::AddNeuron {
                    neuron_uuid: gate_uuid.clone(),
                    neuron_type: "hidden".to_string(),
                    squash: "RELU".to_string(),
                    bias: gate_bias,
                    insert_before_neuron_uuid: None,
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: c.neuron_uuid.clone(),
                    to_neuron_uuid: gate_uuid,
                    weight: 1.0,
                },
            ],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Bounded range on {}: {:.0}% of samples at sentinel {:.2}, useful range [{:.2}, {:.2}] → add gating neuron to suppress sentinel region",
                c.neuron_uuid,
                c.boundary_fraction * 100.0,
                c.boundary_value,
                c.useful_range_min,
                c.useful_range_max,
            )),
        });
    }

    // Sort by expected improvement (best first) (Issue #941: shared helper)
    sort_candidates_by_score_gain(&mut results);

    results
}
