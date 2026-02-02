//! Bounded range neuron detection module (Issue #395).
//!
//! Identifies hidden neurons that operate in a restricted sub-range of their
//! activation function's output domain. For example, a TANH neuron consistently
//! outputting values in [0.1, 0.3] is only using 10% of its [-1, 1] range.
//! This wastes representational capacity and suggests the neuron's activation
//! function, bias, or incoming weights are misconfigured.
//!
//! ## Detection Criteria
//!
//! A hidden neuron has a "bounded range" issue if:
//! 1. **Bounded activation function**: The squash function has a known theoretical
//!    range (e.g., TANH → [-1, 1], LOGISTIC → [0, 1]).
//! 2. **Low utilisation**: The observed activation range is a small fraction of the
//!    theoretical range (below `MAX_UTILISATION_THRESHOLD`).
//! 3. **Not dead**: The neuron has meaningful activation (not near-zero).
//! 4. **Sufficient samples**: At least `MIN_SAMPLES` records to be statistically
//!    reliable.
//!
//! ## Recommended Actions
//!
//! When a bounded range neuron is detected, we recommend:
//! 1. **Change activation** (`ChangeSquash`) — switch to a function better suited
//!    to the observed operating range.
//! 2. **Adjust bias** (`SetBias`) — shift the operating point to use more of the
//!    activation function's range.

use std::collections::HashMap;

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum samples required for reliable bounded range detection.
const MIN_SAMPLES: usize = 20;

/// Maximum utilisation ratio to flag a neuron as restricted.
/// A neuron using less than 30% of its activation range is flagged.
const MAX_UTILISATION_THRESHOLD: f32 = 0.30;

/// Minimum activation range (max - min) to exclude dead/near-constant neurons.
/// Neurons with a range below this are likely dead or constant and handled
/// by other detection modules.
const MIN_ACTIVATION_RANGE: f32 = 0.01;

/// Get the theoretical output range for a bounded activation function.
///
/// Returns `Some((min, max))` for bounded squash functions, `None` for unbounded.
fn theoretical_range(squash: &str) -> Option<(f32, f32)> {
    let upper = squash.to_ascii_uppercase();
    match upper.as_str() {
        "TANH" | "HARD_TANH" | "CLIPPED" | "BIPOLAR" | "BIPOLAR_SIGMOID" => Some((-1.0, 1.0)),
        "LOGISTIC" => Some((0.0, 1.0)),
        "STEP" => Some((0.0, 1.0)),
        "SOFTSIGN" | "ISRU" => Some((-1.0, 1.0)),
        "ARCTAN" => {
            // arctan output is in (-π/2, π/2), practically ≈ (-1.57, 1.57)
            let half_pi = std::f32::consts::FRAC_PI_2;
            Some((-half_pi, half_pi))
        }
        "RELU6" => Some((0.0, 6.0)),
        _ => None, // IDENTITY, RELU, ELU, GELU, MISH, etc. are unbounded
    }
}

/// Result of detecting a bounded range neuron.
#[derive(Debug, Clone)]
pub struct BoundedRangeCandidate {
    /// UUID of the neuron with restricted range.
    pub neuron_uuid: String,
    /// Current activation function of the neuron.
    pub current_squash: String,
    /// Observed activation minimum.
    pub observed_min: f32,
    /// Observed activation maximum.
    pub observed_max: f32,
    /// Theoretical activation range of the squash function.
    pub theoretical_min: f32,
    /// Theoretical activation range of the squash function.
    pub theoretical_max: f32,
    /// Fraction of theoretical range actually used (0.0 to 1.0).
    pub utilisation_ratio: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated creature score improvement from fixing this neuron.
    pub estimated_improvement: f32,
}

/// Detect hidden neurons with restricted activation ranges.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `BoundedRangeCandidate` for neurons with restricted ranges,
/// sorted by estimated improvement (best first).
pub fn detect_bounded_range_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<BoundedRangeCandidate> {
    // Build records lookup
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    let mut candidates = Vec::new();

    for neuron in &creature.neurons {
        // Only analyse hidden neurons
        if neuron.neuron_type != "hidden" {
            continue;
        }

        // Must be a bounded activation function
        let Some((theo_min, theo_max)) = theoretical_range(&neuron.squash) else {
            continue;
        };
        let theoretical_span = theo_max - theo_min;
        if theoretical_span <= 0.0 {
            continue;
        }

        // Must have recorded data
        let Some(records) = records_map.get(neuron.uuid.as_str()) else {
            continue;
        };

        if records.len() < MIN_SAMPLES {
            continue;
        }

        // Compute observed activation range
        let mut obs_min = f32::INFINITY;
        let mut obs_max = f32::NEG_INFINITY;
        let mut valid_count = 0usize;

        for record in records.iter() {
            if record.activation.is_finite() {
                if record.activation < obs_min {
                    obs_min = record.activation;
                }
                if record.activation > obs_max {
                    obs_max = record.activation;
                }
                valid_count += 1;
            }
        }

        if valid_count < MIN_SAMPLES {
            continue;
        }

        let observed_range = obs_max - obs_min;

        // Exclude dead/near-constant neurons (handled by dead_neuron detection)
        if observed_range < MIN_ACTIVATION_RANGE {
            continue;
        }

        let utilisation_ratio = observed_range / theoretical_span;

        if utilisation_ratio >= MAX_UTILISATION_THRESHOLD {
            continue;
        }

        // Estimated improvement: higher for lower utilisation (more wasted capacity)
        // Scale so that 0% utilisation → 0.005, 30% → ~0.0
        let estimated_improvement = 0.005 * (1.0 - utilisation_ratio / MAX_UTILISATION_THRESHOLD);

        candidates.push(BoundedRangeCandidate {
            neuron_uuid: neuron.uuid.clone(),
            current_squash: neuron.squash.clone(),
            observed_min: obs_min,
            observed_max: obs_max,
            theoretical_min: theo_min,
            theoretical_max: theo_max,
            utilisation_ratio,
            sample_count: valid_count,
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Convert bounded range candidates into coordinated structural candidates.
///
/// Each bounded range neuron produces a `ChangeSquash` coordinated candidate
/// to switch to an activation function that better utilises the observed range.
/// NEAT-AI validates the change through ablation testing.
pub fn bounded_range_to_coordinated_candidates(
    candidates: &[BoundedRangeCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        // Suggest a squash function that better matches the observed range.
        // If the neuron uses a small positive range, RELU might be appropriate.
        // If it uses a small range near zero, IDENTITY could work.
        let suggested_squash = suggest_squash_for_range(c);

        let mut operations = Vec::new();

        // Primary recommendation: change squash function
        if suggested_squash != c.current_squash {
            operations.push(CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: c.neuron_uuid.clone(),
                squash: suggested_squash.clone(),
            });
        }

        // Secondary recommendation: adjust bias to centre in active region
        let observed_midpoint = (c.observed_min + c.observed_max) / 2.0;
        let theoretical_midpoint = (c.theoretical_min + c.theoretical_max) / 2.0;
        let bias_offset = theoretical_midpoint - observed_midpoint;
        if bias_offset.abs() > 0.01 {
            operations.push(CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: c.neuron_uuid.clone(),
                bias: bias_offset,
            });
        }

        if operations.is_empty() {
            continue;
        }

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Bounded range {}: {} uses [{:.3}, {:.3}] of [{:.1}, {:.1}] ({:.0}% utilisation, {} samples)",
                c.neuron_uuid,
                c.current_squash,
                c.observed_min,
                c.observed_max,
                c.theoretical_min,
                c.theoretical_max,
                c.utilisation_ratio * 100.0,
                c.sample_count,
            )),
        });
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Suggest a better squash function based on the observed activation range.
fn suggest_squash_for_range(candidate: &BoundedRangeCandidate) -> String {
    let mid = (candidate.observed_min + candidate.observed_max) / 2.0;
    let range = candidate.observed_max - candidate.observed_min;

    // If the neuron operates in a small positive range, RELU may be appropriate
    if candidate.observed_min >= -0.01 && range > 0.0 {
        return "RELU".to_string();
    }

    // If the neuron operates near zero in a symmetric range, IDENTITY avoids squashing
    if mid.abs() < 0.1 && range < 0.5 {
        return "IDENTITY".to_string();
    }

    // Default: suggest IDENTITY to avoid wasting dynamic range
    "IDENTITY".to_string()
}
