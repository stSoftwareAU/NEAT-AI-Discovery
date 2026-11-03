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
//! - **`AddNeuron`**: A hidden gate neuron with `STEP` activation that outputs ~0 for
//!   sentinel values and ~1 for useful values.
//! - **`AddSynapse`** (obs → gate): Connects the observation to the gate neuron.
//! - **`AddSynapse`** (gate → target): Connects the gate to each downstream target,
//!   so the gate can modulate the observation's influence.
//!
//! All squash functions used are scalar and GPU-compatible (no `IF` or other
//! aggregate functions).
//!
//! ## Dependency
//!
//! The "is this cluster a sentinel?" decision is shared with `observation_range`
//! (Issue #398) and defined once in
//! [`sentinel_cluster::assess_sentinel_cluster`](super::sentinel_cluster::assess_sentinel_cluster)
//! (Issue #2042).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::{HashMap, HashSet};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

// Constants moved to constants.rs (Issue #424)
use crate::analysis::constants::{
    CANDIDATE_SENTINELS, MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES, SENTINEL_TOLERANCE,
};
// Shared sentinel-cluster rule (Issue #2042)
use crate::analysis::detection::helpers::{existing_neuron_uuids, first_free_neuron_uuid};
use crate::analysis::detection::sentinel_cluster::assess_sentinel_cluster;
use crate::analysis::utils::deadline_passed;

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
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<SentinelGatingCandidate> {
    // Only consider input neurons (observations)
    let input_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut candidates = Vec::with_capacity(neuron_records.len());

    for (uuid, records) in neuron_records {
        let records = records.as_ref();
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
        // The accept/reject rule lives in one place (Issue #2042): density,
        // separation from the useful range, and lower error variance inside the
        // cluster. `None` means this value is not a sentinel.
        let Some(cluster) = assess_sentinel_cluster(&activations, &errors, sentinel) else {
            continue;
        };

        let improvement = estimate_gating_improvement(
            cluster.sentinel_fraction,
            cluster.gap,
            cluster.sentinel_error_var,
            cluster.non_sentinel_error_var,
            n,
        );

        let is_better = best
            .as_ref()
            .is_none_or(|prev| improvement > prev.estimated_improvement);

        if is_better {
            best = Some(SentinelGatingCandidate {
                neuron_uuid: uuid.to_string(),
                sentinel_value: cluster.sentinel_value,
                sentinel_fraction: cluster.sentinel_fraction,
                useful_range_min: cluster.useful_min,
                useful_range_max: cluster.useful_max,
                sample_count: records.len(),
                estimated_improvement: improvement,
            });
        }
    }

    best
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
/// values and ~1 for useful values. Its deterministic UUID keeps the legacy
/// spelling when free and is salted when the creature already uses it (Issue #2412).
///
/// # Structure
///
/// For an observation `obs` with sentinel at `s` connected to targets `[t1, t2, ...]`:
///
/// 1. **`AddNeuron`** — gate neuron with `STEP` squash and bias computed to place the
///    step threshold between the sentinel and useful range.
/// 2. **`AddSynapse`** (obs → gate) — weight 1.0 feeding the observation into the gate.
/// 3. **`AddSynapse`** (gate → `t_i`) — for each downstream target, a synapse carrying
///    the gated signal with the original connection weight.
///
/// Issue #2479: no deadline — equivalent to calling
/// [`sentinel_gating_to_coordinated_candidates_with_deadline`] with a `None`
/// deadline.
pub fn sentinel_gating_to_coordinated_candidates(
    candidates: &[SentinelGatingCandidate],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    sentinel_gating_to_coordinated_candidates_with_deadline(candidates, creature, &None)
}

/// Deadline-aware variant of [`sentinel_gating_to_coordinated_candidates`]
/// (Issue #2479).
///
/// Builds the from-uuid → downstream adjacency once up front (rather than
/// rescanning `creature.synapses` per candidate, which was O(C·S)) and checks
/// `deadline` once per candidate, so a hung or oversized conversion can be
/// stopped mid-flight, returning the coordinated candidates built so far.
pub fn sentinel_gating_to_coordinated_candidates_with_deadline(
    candidates: &[SentinelGatingCandidate],
    creature: &CreatureJson,
    deadline: &Option<std::time::SystemTime>,
) -> Vec<CoordinatedStructuralCandidateJson> {
    sentinel_gating_to_coordinated_candidates_with_observer(
        candidates,
        creature,
        deadline,
        &mut || {},
    )
}

fn sentinel_gating_to_coordinated_candidates_with_observer(
    candidates: &[SentinelGatingCandidate],
    creature: &CreatureJson,
    deadline: &Option<std::time::SystemTime>,
    work_observer: &mut impl FnMut(),
) -> Vec<CoordinatedStructuralCandidateJson> {
    let existing = existing_neuron_uuids(creature);

    // Issue #2479: from_uuid -> [(to_uuid, weight)], in synapse order, built once.
    let mut downstream_by_source: HashMap<&str, Vec<(&str, f32)>> = HashMap::new();
    for s in &creature.synapses {
        work_observer();
        downstream_by_source
            .entry(s.from_uuid.as_str())
            .or_default()
            .push((s.to_uuid.as_str(), s.weight));
    }

    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        if deadline_passed(deadline) {
            break;
        }

        // Find downstream targets: neurons that receive a synapse from this observation
        work_observer();
        let downstream_targets: Vec<(&str, f32)> = downstream_by_source
            .get(c.neuron_uuid.as_str())
            .cloned()
            .unwrap_or_default();

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

        let gate_uuid = first_free_neuron_uuid(&existing, |salt| {
            if salt == 0 {
                format!("gate-sentinel-{}", c.neuron_uuid)
            } else {
                format!("gate-sentinel-{}-{salt}", c.neuron_uuid)
            }
        });

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
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
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

#[cfg(test)]
mod issue_2479_tests {
    use super::*;

    fn candidate(uuid: &str) -> SentinelGatingCandidate {
        SentinelGatingCandidate {
            neuron_uuid: uuid.to_string(),
            sentinel_value: -1.0,
            sentinel_fraction: 0.3,
            useful_range_min: 0.0,
            useful_range_max: 1.0,
            sample_count: 30,
            estimated_improvement: 0.01,
        }
    }

    /// `count` observations `in-0..`, each feeding two targets, so every candidate
    /// has downstream synapses and converts to exactly one coordinated candidate.
    fn fixture(count: usize) -> (CreatureJson, Vec<SentinelGatingCandidate>) {
        let mut neurons = Vec::new();
        let mut synapses = Vec::new();
        let mut candidates = Vec::new();
        for i in 0..count {
            let uuid = format!("in-{i}");
            neurons.push(crate::NeuronJson {
                uuid: uuid.clone(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            });
            for target in ["out-a", "out-b"] {
                synapses.push(crate::SynapseJson {
                    from_uuid: uuid.clone(),
                    to_uuid: target.to_string(),
                    weight: 0.5,
                    synapse_type: None,
                });
            }
            candidates.push(candidate(&uuid));
        }
        for target in ["out-a", "out-b"] {
            neurons.push(crate::NeuronJson {
                uuid: target.to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            });
        }
        let creature = CreatureJson {
            neurons,
            synapses,
            input: count,
            output: 2,
        };
        (creature, candidates)
    }

    fn gate_uuid(c: &CoordinatedStructuralCandidateJson) -> Option<&str> {
        c.operations.iter().find_map(|op| match op {
            CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. } => {
                Some(neuron_uuid.as_str())
            }
            _ => None,
        })
    }

    /// Work is one step per synapse (adjacency build) plus one per candidate
    /// (adjacency lookup), so it must scale exactly linearly: 3N at N and 12N at 4N.
    #[test]
    fn adjacency_work_scales_linearly_at_n_and_4n() {
        const N: usize = 128;

        let measured_work = |count: usize| {
            let (creature, candidates) = fixture(count);
            let mut work = 0;
            let result = sentinel_gating_to_coordinated_candidates_with_observer(
                &candidates,
                &creature,
                &None,
                &mut || work += 1,
            );
            assert_eq!(result.len(), count);
            work
        };

        let small_work = measured_work(N);
        let large_work = measured_work(4 * N);
        assert_eq!(small_work, 3 * N);
        assert_eq!(large_work, 3 * 4 * N);
        assert_eq!(large_work, small_work * 4);
    }

    /// The deadline is checked once per candidate, and on expiry the candidates
    /// already built are returned rather than discarded. The scripted sequence lets
    /// the first candidate through and stops at the second, so hoisting the check
    /// out of the loop (all three built) or replacing `break` with an empty return
    /// (none built) both fail this test.
    #[test]
    #[serial_test::serial]
    fn deadline_mid_conversion_returns_the_candidates_built_so_far() {
        use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;

        let (creature, candidates) = fixture(3);

        // Positive control: with no override every candidate converts.
        let control =
            sentinel_gating_to_coordinated_candidates_with_deadline(&candidates, &creature, &None);
        let mut control_gates: Vec<&str> = control.iter().filter_map(gate_uuid).collect();
        control_gates.sort_unstable();
        assert_eq!(
            control_gates,
            [
                "gate-sentinel-in-0",
                "gate-sentinel-in-1",
                "gate-sentinel-in-2"
            ],
            "positive control: fixture must yield one candidate per observation"
        );

        let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);
        let partial =
            sentinel_gating_to_coordinated_candidates_with_deadline(&candidates, &creature, &None);

        assert_eq!(
            partial.len(),
            1,
            "the deadline must stop the conversion at the second candidate and keep the first"
        );
        assert_eq!(
            gate_uuid(&partial[0]),
            Some("gate-sentinel-in-0"),
            "the kept candidate must be the first observation's"
        );
    }
}
