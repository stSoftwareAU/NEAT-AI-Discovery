//! Bounded range discovery module (Issue #395).
//!
//! Identifies neurons (observations and hidden) where activation values split into
//! a meaningful bounded range and a sentinel/null cluster. For example, an observation
//! like "Debt-to-Equity" might use -1 as a sentinel for "no data", while values in
//! 0..1 carry actual meaning. Simply multiplying by a weight treats the sentinel as
//! a real value, which degrades the creature's score.
//!
//! ## Detection Criteria
//!
//! A neuron has a bounded range with sentinel if:
//! 1. **Bimodal clustering**: A significant fraction of activations cluster tightly
//!    around a single value (the sentinel), while the remainder spread across a
//!    different sub-range (the meaningful range).
//! 2. **Separation gap**: There is a clear gap between the sentinel cluster and the
//!    meaningful range.
//! 3. **Only input and hidden neurons**: Output neurons are excluded.
//!
//! ## Recommended Actions
//!
//! When a bounded range is detected, we recommend inserting an IF-gated hidden neuron
//! that passes the meaningful signal when above/below the sentinel threshold, and
//! outputs zero otherwise. This uses the existing "IF" squash function which implements
//! `if condition > 0 then positive_input else negative_input`.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Bounded Range Detection" for full documentation.

use std::collections::{HashMap, HashSet};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum samples required for reliable bounded range detection.
const MIN_SAMPLES_FOR_DETECTION: usize = 30;

/// Minimum fraction of samples that must cluster at the sentinel to trigger detection.
/// Below this, the "sentinel" is too rare to justify structural changes.
const MIN_SENTINEL_FRACTION: f32 = 0.10;

/// Maximum fraction of samples at the sentinel. If nearly all samples are at the
/// sentinel, there is no meaningful range to exploit.
const MAX_SENTINEL_FRACTION: f32 = 0.80;

/// Maximum standard deviation within the sentinel cluster (relative to overall range).
/// The sentinel cluster should be tightly concentrated.
const MAX_SENTINEL_CLUSTER_RELATIVE_STD: f32 = 0.05;

/// Minimum gap between sentinel cluster and meaningful range, as a fraction of the
/// overall activation range. Ensures the two groups are well-separated.
const MIN_SEPARATION_GAP_FRACTION: f32 = 0.10;

/// Number of histogram bins used for detecting bimodal clustering.
const HISTOGRAM_BINS: usize = 20;

/// Result of detecting a bounded range neuron.
#[derive(Debug, Clone)]
pub struct BoundedRangeCandidate {
    /// UUID of the neuron with a bounded range.
    pub neuron_uuid: String,
    /// The detected sentinel value (centre of the sentinel cluster).
    pub sentinel_value: f32,
    /// Fraction of samples at the sentinel.
    pub sentinel_fraction: f32,
    /// Lower bound of the meaningful activation range.
    pub meaningful_range_lo: f32,
    /// Upper bound of the meaningful activation range.
    pub meaningful_range_hi: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Confidence that the bounded range detection is correct (0.0 to 1.0).
    pub detection_confidence: f32,
    /// Estimated creature score improvement from gating this input.
    pub estimated_improvement: f32,
    /// UUIDs of neurons directly downstream from this neuron.
    pub downstream_uuids: Vec<String>,
}

/// Detect neurons with bounded activation ranges and sentinel values.
///
/// Examines activation distributions to find bimodal patterns where one mode
/// is a tight sentinel cluster and the other is a spread of meaningful values.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `BoundedRangeCandidate` sorted by detection confidence (highest first).
pub fn detect_bounded_ranges(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<BoundedRangeCandidate> {
    // Identify eligible neurons (input and hidden only)
    let eligible_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input" || n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    // Build fan-out map for finding downstream neurons
    let mut fan_out_map: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in &creature.synapses {
        fan_out_map
            .entry(s.from_uuid.as_str())
            .or_default()
            .push(s.to_uuid.as_str());
    }

    // Build records lookup
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    let mut candidates = Vec::new();

    for uuid in &eligible_uuids {
        let Some(records) = records_map.get(uuid) else {
            continue;
        };

        if records.len() < MIN_SAMPLES_FOR_DETECTION {
            continue;
        }

        if let Some(candidate) = analyse_activation_distribution(uuid, records, &fan_out_map) {
            candidates.push(candidate);
        }
    }

    // Sort by detection confidence (highest first)
    candidates.sort_by(|a, b| {
        b.detection_confidence
            .partial_cmp(&a.detection_confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Analyse the activation distribution of a single neuron to detect bimodal
/// sentinel + meaningful-range patterns.
fn analyse_activation_distribution(
    uuid: &str,
    records: &[DiscoverRecord],
    fan_out_map: &HashMap<&str, Vec<&str>>,
) -> Option<BoundedRangeCandidate> {
    let activations: Vec<f32> = records.iter().map(|r| r.activation).collect();
    let n = activations.len() as f32;

    // Compute overall range
    let min_val = activations.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = activations
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    // Need a meaningful range to analyse
    if range < 1e-6 {
        return None;
    }

    // Build histogram to find clusters
    let bin_width = range / HISTOGRAM_BINS as f32;
    let mut histogram = [0u32; HISTOGRAM_BINS];

    for &val in &activations {
        let bin = ((val - min_val) / bin_width).floor() as usize;
        let bin = bin.min(HISTOGRAM_BINS - 1);
        histogram[bin] += 1;
    }

    // Find the peak bin (potential sentinel cluster)
    let (peak_bin, &peak_count) = histogram
        .iter()
        .enumerate()
        .max_by_key(|(_, &count)| count)
        .unwrap();

    let peak_fraction = peak_count as f32 / n;

    // The peak must represent a sentinel-like cluster
    if peak_fraction < MIN_SENTINEL_FRACTION {
        return None;
    }

    if peak_fraction > MAX_SENTINEL_FRACTION {
        return None;
    }

    // Compute the sentinel cluster centre and spread
    let peak_lo = min_val + peak_bin as f32 * bin_width;
    let peak_hi = peak_lo + bin_width;

    // Collect activations in the peak bin
    let sentinel_activations: Vec<f32> = activations
        .iter()
        .filter(|&&v| v >= peak_lo && v <= peak_hi)
        .cloned()
        .collect();

    let sentinel_mean =
        sentinel_activations.iter().sum::<f32>() / sentinel_activations.len() as f32;
    let sentinel_var = sentinel_activations
        .iter()
        .map(|&v| {
            let d = v - sentinel_mean;
            d * d
        })
        .sum::<f32>()
        / sentinel_activations.len() as f32;
    let sentinel_std = sentinel_var.sqrt();

    // Sentinel cluster must be tight relative to overall range
    if sentinel_std / range > MAX_SENTINEL_CLUSTER_RELATIVE_STD {
        return None;
    }

    // Collect non-sentinel activations (meaningful range)
    // Use a tolerance of 2 std devs around sentinel mean, or bin width, whichever is larger
    let sentinel_tolerance = (sentinel_std * 2.0).max(bin_width);
    let meaningful_activations: Vec<f32> = activations
        .iter()
        .filter(|&&v| (v - sentinel_mean).abs() > sentinel_tolerance)
        .cloned()
        .collect();

    if meaningful_activations.is_empty() {
        return None;
    }

    let sentinel_fraction = 1.0 - (meaningful_activations.len() as f32 / n);

    // Re-check sentinel fraction with the refined count
    if !(MIN_SENTINEL_FRACTION..=MAX_SENTINEL_FRACTION).contains(&sentinel_fraction) {
        return None;
    }

    // Compute meaningful range bounds
    let meaningful_lo = meaningful_activations
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let meaningful_hi = meaningful_activations
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Check separation gap between sentinel and meaningful range
    let gap = if sentinel_mean < meaningful_lo {
        meaningful_lo - (sentinel_mean + sentinel_tolerance)
    } else if sentinel_mean > meaningful_hi {
        (sentinel_mean - sentinel_tolerance) - meaningful_hi
    } else {
        // Sentinel is in the middle of the meaningful range — not a clear bimodal pattern
        0.0
    };

    if gap / range < MIN_SEPARATION_GAP_FRACTION {
        return None;
    }

    // Find downstream neurons
    let downstream_uuids: Vec<String> = fan_out_map
        .get(uuid)
        .map(|targets| targets.iter().map(|&t| t.to_string()).collect())
        .unwrap_or_default();

    // Compute detection confidence
    let confidence = compute_detection_confidence(sentinel_fraction, gap / range, n);

    // Estimated improvement: gating sentinel values prevents them from corrupting the signal.
    // The more samples are sentinel, the more damage is being done.
    let estimated_improvement = confidence * sentinel_fraction * 0.01;

    Some(BoundedRangeCandidate {
        neuron_uuid: uuid.to_string(),
        sentinel_value: sentinel_mean,
        sentinel_fraction,
        meaningful_range_lo: meaningful_lo,
        meaningful_range_hi: meaningful_hi,
        sample_count: records.len(),
        detection_confidence: confidence,
        estimated_improvement,
        downstream_uuids,
    })
}

/// Compute detection confidence from distribution characteristics.
///
/// Higher confidence when:
/// - Sentinel fraction is clearly in the valid range (not borderline)
/// - Gap between sentinel and meaningful range is large
/// - More samples were analysed
fn compute_detection_confidence(
    sentinel_fraction: f32,
    gap_fraction: f32,
    sample_count: f32,
) -> f32 {
    // Sentinel fraction factor: peaks at ~0.3, lower near boundaries
    let mid_sentinel = 0.5 * (MIN_SENTINEL_FRACTION + MAX_SENTINEL_FRACTION);
    let sentinel_range = 0.5 * (MAX_SENTINEL_FRACTION - MIN_SENTINEL_FRACTION);
    let sentinel_factor =
        1.0 - ((sentinel_fraction - mid_sentinel).abs() / sentinel_range).min(1.0);

    // Gap factor: larger gap = higher confidence
    let gap_factor = (gap_fraction / MIN_SEPARATION_GAP_FRACTION).min(2.0) / 2.0;

    // Sample size factor: more samples = higher confidence, plateaus at 1000
    let sample_factor = (sample_count / 1000.0).min(1.0);

    // Combine factors
    let raw = sentinel_factor * 0.3 + gap_factor * 0.4 + sample_factor * 0.3;

    // Scale to [0.3, 1.0] since we already passed threshold checks
    0.3 + raw * 0.7
}

/// Convert bounded range candidates into coordinated structural candidates.
///
/// For each bounded range detection, produces a coordinated candidate that inserts
/// an IF-gated hidden neuron. The IF squash function implements:
///   `if condition > 0 then positive_input else negative_input`
///
/// This allows the network to use the meaningful signal when the observation is in
/// the valid range and output zero (or a learned default) when at the sentinel value.
///
/// The IF neuron requires three incoming synapses:
/// - `condition`: the source neuron, with bias set so `condition > 0` when in meaningful range
/// - `positive`: the source neuron (passes through when condition is met)
/// - `negative`: effectively zero (sentinel case)
pub fn bounded_ranges_to_coordinated_candidates(
    candidates: &[BoundedRangeCandidate],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    // Find existing synapses from each candidate neuron
    let mut downstream_synapses: HashMap<&str, Vec<&crate::SynapseJson>> = HashMap::new();
    for s in &creature.synapses {
        downstream_synapses
            .entry(s.from_uuid.as_str())
            .or_default()
            .push(s);
    }

    for c in candidates {
        // For each downstream synapse from the bounded-range neuron, create an
        // IF-gated replacement path.
        let synapses = downstream_synapses
            .get(c.neuron_uuid.as_str())
            .cloned()
            .unwrap_or_default();

        if synapses.is_empty() {
            continue;
        }

        // Pick the first downstream target for the IF-gated path.
        // Each downstream synapse could get its own IF gate, but we start with one
        // to minimise structural complexity.
        let target_synapse = synapses[0];

        // Generate a deterministic UUID for the new IF neuron
        let if_neuron_uuid = format!("if-gate-{}-{}", c.neuron_uuid, target_synapse.to_uuid);

        // Compute the threshold bias: the IF condition fires when
        // (activation * weight + bias) > 0. We want it to fire when the activation
        // is in the meaningful range, not at the sentinel.
        //
        // If sentinel < meaningful range: condition = activation + bias > 0
        //   → bias = -(midpoint between sentinel and meaningful_lo)
        //   → so activation > midpoint triggers the positive branch
        //
        // If sentinel > meaningful range: condition = -activation + bias > 0
        //   → bias = midpoint between meaningful_hi and sentinel
        let (condition_weight, condition_bias) = if c.sentinel_value < c.meaningful_range_lo {
            let threshold = (c.sentinel_value + c.meaningful_range_lo) / 2.0;
            (1.0, -threshold)
        } else {
            let threshold = (c.meaningful_range_hi + c.sentinel_value) / 2.0;
            (-1.0, threshold)
        };

        let mut operations = vec![
            // Add the IF-gated hidden neuron
            CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: if_neuron_uuid.clone(),
                neuron_type: "hidden".to_string(),
                squash: "IF".to_string(),
                bias: condition_bias,
                insert_before_neuron_uuid: Some(target_synapse.to_uuid.clone()),
            },
            // Condition synapse: determines if we are in the meaningful range
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: c.neuron_uuid.clone(),
                to_neuron_uuid: if_neuron_uuid.clone(),
                weight: condition_weight,
            },
            // Positive synapse: passes through the original signal when in range
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: c.neuron_uuid.clone(),
                to_neuron_uuid: if_neuron_uuid.clone(),
                weight: target_synapse.weight,
            },
            // Connect the IF neuron to the downstream target
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: if_neuron_uuid,
                to_neuron_uuid: target_synapse.to_uuid.clone(),
                weight: 1.0,
            },
            // Remove the original direct synapse (replaced by the IF-gated path)
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: c.neuron_uuid.clone(),
                to_neuron_uuid: target_synapse.to_uuid.clone(),
            },
        ];

        // Annotate IF synapse types for clarity
        if let CoordinatedStructuralOpJson::AddSynapse { .. } = &operations[1] {
            // The first AddSynapse is the condition
        }
        if let CoordinatedStructuralOpJson::AddSynapse { .. } = &operations[2] {
            // The second AddSynapse is the positive branch
        }

        // Sort operations: removes first, then adds (standard ordering)
        operations.sort_by_key(|op| match op {
            CoordinatedStructuralOpJson::RemoveSynapse { .. } => 0,
            CoordinatedStructuralOpJson::AddNeuron { .. } => 1,
            CoordinatedStructuralOpJson::AddSynapse { .. } => 2,
            _ => 3,
        });

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Bounded range {}: sentinel {:.3} ({:.0}% of samples), meaningful range [{:.3}, {:.3}] → IF-gated path to {}",
                c.neuron_uuid,
                c.sentinel_value,
                c.sentinel_fraction * 100.0,
                c.meaningful_range_lo,
                c.meaningful_range_hi,
                target_synapse.to_uuid
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
