//! Sentinel value gating discovery module (Issue #400).
//!
//! When observations use sentinel values (e.g., -1 meaning "no data"), simply
//! multiplying by a weight propagates meaningless information through the network.
//! This module detects observations where sentinel values degrade performance and
//! proposes gated neuron structures using scalar, GPU-compatible squash functions.
//!
//! ## Detection Criteria
//!
//! An observation needs sentinel gating if:
//! 1. **Sentinel cluster**: A significant fraction of samples cluster at a sentinel
//!    value (e.g., -1, 0, +1), separated from the useful range by a gap.
//! 2. **Error decorrelation**: The sentinel cluster has lower error variance than the
//!    useful range, indicating the sentinel does not meaningfully influence the output.
//! 3. **Sufficient samples**: At least 20 samples are available for reliable detection.
//! 4. **Input neurons only**: Only observations (input neurons) are considered.
//!
//! ## Proposed Gated Structure
//!
//! For each detected sentinel observation, the module proposes a `coordinatedStructural`
//! candidate with:
//! - **AddNeuron**: A hidden gate neuron with `STEP` activation that outputs ~0 for
//!   sentinel values and ~1 for useful values.
//! - **AddSynapse** (obs → gate): Connects the observation to the gate neuron.
//! - **AddSynapse** (gate → target): Connects the gate to each downstream target,
//!   so the gate can modulate the observation's influence.
//!
//! All squash functions used are scalar and GPU-compatible (no `IF` or other
//! aggregate functions).
//!
//! ## Dependency
//!
//! Uses error-correlation analysis similar to observation_range (Issue #398)
//! for sentinel identification.

use std::collections::HashSet;

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

// Constants moved to constants.rs (Issue #424)
use super::constants::{
    CANDIDATE_SENTINELS, MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES, MIN_SENTINEL_FRACTION,
    MIN_SENTINEL_GAP as MIN_GAP, SENTINEL_TOLERANCE,
};

/// Result of detecting a sentinel gating candidate on an observation.
#[derive(Debug, Clone)]
pub struct SentinelGatingCandidate {
    /// UUID of the observation (input neuron) with the sentinel issue.
    pub neuron_uuid: String,
    /// The detected sentinel value (e.g., -1.0, 0.0).
    pub sentinel_value: f32,
    /// Fraction of samples at the sentinel value (0.0 to 1.0).
    pub sentinel_fraction: f32,
    /// Minimum of the useful (non-sentinel) range.
    pub useful_range_min: f32,
    /// Maximum of the useful (non-sentinel) range.
    pub useful_range_max: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated creature score improvement from gating.
    pub estimated_improvement: f32,
}

/// Detect observations where sentinel values degrade performance.
///
/// Uses error-correlation analysis: a sentinel cluster is only flagged when its
/// error variance is lower than the useful range's error variance, indicating
/// the sentinel does not carry meaningful information.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded data.
///
/// # Returns
/// A list of `SentinelGatingCandidate` sorted by estimated improvement (highest first).
pub fn detect_sentinel_gating_candidates(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<SentinelGatingCandidate> {
    // Only consider input neurons (observations)
    let input_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut candidates = Vec::new();

    for (uuid, records) in neuron_records {
        if !input_uuids.contains(uuid.as_str()) {
            continue;
        }

        if records.len() < MIN_SAMPLES {
            continue;
        }

        if let Some(candidate) = analyse_observation_for_sentinel(uuid, records) {
            candidates.push(candidate);
        }
    }

    // Sort by estimated improvement (highest first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Analyse a single observation for sentinel gating using error correlation.
fn analyse_observation_for_sentinel(
    uuid: &str,
    records: &[DiscoverRecord],
) -> Option<SentinelGatingCandidate> {
    let activations: Vec<f32> = records.iter().map(|r| r.activation).collect();
    let errors: Vec<f32> = records
        .iter()
        .map(|r| {
            if r.errors.is_empty() {
                0.0
            } else {
                r.errors[0]
            }
        })
        .collect();
    let n = activations.len() as f32;

    let mut best: Option<SentinelGatingCandidate> = None;

    for &sentinel in &CANDIDATE_SENTINELS {
        // Collect indices of samples at/near the sentinel
        let sentinel_indices: Vec<usize> = activations
            .iter()
            .enumerate()
            .filter(|&(_, &a)| (a - sentinel).abs() <= SENTINEL_TOLERANCE)
            .map(|(i, _)| i)
            .collect();

        let sentinel_fraction = sentinel_indices.len() as f32 / n;
        if sentinel_fraction < MIN_SENTINEL_FRACTION {
            continue;
        }

        // Non-sentinel indices
        let non_sentinel_indices: Vec<usize> = (0..activations.len())
            .filter(|i| !sentinel_indices.contains(i))
            .collect();

        if non_sentinel_indices.is_empty() {
            continue;
        }

        // Compute useful range
        let useful_values: Vec<f32> = non_sentinel_indices
            .iter()
            .map(|&i| activations[i])
            .collect();
        let useful_min = useful_values.iter().copied().fold(f32::INFINITY, f32::min);
        let useful_max = useful_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // Check gap between sentinel and useful range
        let gap = if sentinel <= useful_min {
            useful_min - (sentinel + SENTINEL_TOLERANCE)
        } else if sentinel >= useful_max {
            (sentinel - SENTINEL_TOLERANCE) - useful_max
        } else {
            0.0
        };

        if gap < MIN_GAP {
            continue;
        }

        // Error correlation analysis: sentinel should have LOWER error variance
        let sentinel_error_var = compute_error_variance(&errors, &sentinel_indices);
        let non_sentinel_error_var = compute_error_variance(&errors, &non_sentinel_indices);

        if sentinel_error_var >= non_sentinel_error_var {
            // Sentinel has higher or equal error variance — it may be informative,
            // so do not gate it
            continue;
        }

        let improvement = estimate_gating_improvement(
            sentinel_fraction,
            gap,
            sentinel_error_var,
            non_sentinel_error_var,
            n,
        );

        let is_better = best
            .as_ref()
            .is_none_or(|prev| improvement > prev.estimated_improvement);

        if is_better {
            best = Some(SentinelGatingCandidate {
                neuron_uuid: uuid.to_string(),
                sentinel_value: sentinel,
                sentinel_fraction,
                useful_range_min: useful_min,
                useful_range_max: useful_max,
                sample_count: records.len(),
                estimated_improvement: improvement,
            });
        }
    }

    best
}

/// Compute the variance of error values at the given indices.
fn compute_error_variance(errors: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }

    let n = indices.len() as f32;
    let sum: f32 = indices.iter().map(|&i| errors[i]).sum();
    let mean = sum / n;

    indices
        .iter()
        .map(|&i| (errors[i] - mean).powi(2))
        .sum::<f32>()
        / n
}

/// Estimate the score improvement from gating a sentinel observation.
///
/// Higher improvement when:
/// - More samples are at the sentinel (larger fraction being filtered)
/// - Wider gap between sentinel and useful range (clearer separation)
/// - Greater disparity between sentinel and non-sentinel error variance
/// - More samples analysed (higher confidence)
fn estimate_gating_improvement(
    sentinel_fraction: f32,
    gap: f32,
    sentinel_error_var: f32,
    non_sentinel_error_var: f32,
    sample_count: f32,
) -> f32 {
    // Fraction factor: more samples at sentinel = larger benefit from gating
    let fraction_factor = sentinel_fraction.clamp(0.0, 1.0);

    // Gap factor: wider gap = cleaner gating (saturates at 0.5)
    let gap_factor = (gap / 0.5).clamp(0.0, 1.0);

    // Error disparity: how much less informative the sentinel is
    let error_ratio = if non_sentinel_error_var > 1e-8 {
        (1.0 - sentinel_error_var / non_sentinel_error_var).clamp(0.0, 1.0)
    } else {
        0.5
    };

    // Sample confidence (plateaus at 1000)
    let sample_factor = (sample_count / 1000.0).min(1.0);

    // Combine factors and scale to a reasonable improvement range
    let raw = fraction_factor * 0.3 + gap_factor * 0.2 + error_ratio * 0.3 + sample_factor * 0.2;
    raw * 0.01
}

/// Convert sentinel gating candidates into coordinated structural candidates.
///
/// Each candidate produces a coordinated operation set that adds a gate neuron
/// between the observation and its downstream targets. The gate neuron uses
/// `STEP` activation (GPU-compatible scalar function) to produce ~0 for sentinel
/// values and ~1 for useful values.
///
/// # Structure
///
/// For an observation `obs` with sentinel at `s` connected to targets `[t1, t2, ...]`:
///
/// 1. **AddNeuron** — gate neuron with `STEP` squash and bias computed to place the
///    step threshold between the sentinel and useful range.
/// 2. **AddSynapse** (obs → gate) — weight 1.0 feeding the observation into the gate.
/// 3. **AddSynapse** (gate → t_i) — for each downstream target, a synapse carrying
///    the gated signal with the original connection weight.
pub fn sentinel_gating_to_coordinated_candidates(
    candidates: &[SentinelGatingCandidate],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        // Find downstream targets: neurons that receive a synapse from this observation
        let downstream_targets: Vec<(&str, f32)> = creature
            .synapses
            .iter()
            .filter(|s| s.from_uuid == c.neuron_uuid)
            .map(|s| (s.to_uuid.as_str(), s.weight))
            .collect();

        if downstream_targets.is_empty() {
            continue;
        }

        // Compute the bias for the STEP gate neuron.
        // The STEP function outputs 1 when input >= 0, 0 otherwise.
        // We want: gate(sentinel) ≈ 0, gate(useful) ≈ 1
        // input to gate = activation * weight + bias
        // For sentinel: sentinel * 1.0 + bias < 0 → bias < -sentinel
        // For useful_min: useful_min * 1.0 + bias >= 0 → bias >= -useful_min
        // So bias should be between -useful_min and -sentinel.
        // We pick the midpoint of the gap.
        let threshold = (c.sentinel_value + SENTINEL_TOLERANCE + c.useful_range_min) / 2.0;
        let gate_bias = -threshold;

        let gate_uuid = format!("gate-sentinel-{}", c.neuron_uuid);

        let mut operations = Vec::new();

        // 1. Add the gate neuron with STEP activation
        operations.push(CoordinatedStructuralOpJson::AddNeuron {
            neuron_uuid: gate_uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash: "STEP".to_string(),
            bias: gate_bias,
            insert_before_neuron_uuid: None,
        });

        // 2. Connect observation → gate
        operations.push(CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: c.neuron_uuid.clone(),
            to_neuron_uuid: gate_uuid.clone(),
            weight: 1.0,
        });

        // 3. Connect gate → each downstream target
        for (target_uuid, original_weight) in &downstream_targets {
            operations.push(CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: gate_uuid.clone(),
                to_neuron_uuid: target_uuid.to_string(),
                weight: *original_weight,
            });
        }

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Sentinel gating on {}: {:.0}% of samples at sentinel {:.2}, \
                 useful range [{:.2}, {:.2}] → STEP gate to suppress sentinel region",
                c.neuron_uuid,
                c.sentinel_fraction * 100.0,
                c.sentinel_value,
                c.useful_range_min,
                c.useful_range_max,
            )),
        });
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    results
}
