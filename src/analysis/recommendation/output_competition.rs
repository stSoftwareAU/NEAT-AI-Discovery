//! Output competition / lateral-inhibition recommender (Issue #1321).
//!
//! Under `OneHot` / `Simplex` target topologies, exactly one output class
//! "wins" per sample. When two output neurons co-fire strongly on the same
//! samples they are competing for the same evidence — a structural pattern
//! lateral inhibition is designed to resolve. This recommender proposes
//! mutual-exclusion structure between such output neurons by emitting
//! inhibitory output→output `AddSynapse` operations (forward-only ordered
//! so the resulting edge is valid).
//!
//! ## Gating
//!
//! - `target_topology = OneHot | Simplex` ⇒ run the detector.
//! - Every other topology (`Independent`, `Margin`, `Unknown`, including
//!   the neutral / `OTHER` fallbacks) ⇒ emit **nothing** (regression
//!   guard, per the acceptance criteria of Issue #1321).
//!
//! ## Detection criteria
//!
//! 1. The creature has at least two output neurons.
//! 2. For an ordered output pair `(a, b)` where `index_of(a) < index_of(b)`
//!    in `creature.neurons[]` (so `AddSynapse(a → b)` respects forward-only
//!    evaluation order):
//!    a. Their records cover the same observation indices.
//!    b. On at least `MIN_DISCOVERY_SAMPLE_COUNT` aligned samples both
//!    activations exceed `CO_ACTIVATION_THRESHOLD` (they are "co-firing").
//! 3. The pair has no existing synapse between them — we never duplicate.
//!
//! When all checks pass, an [`OutputCompetitionCandidate`] is emitted with
//! a small negative `recommended_weight` (`LATERAL_INHIBITION_WEIGHT`).
//! `output_competition_to_coordinated_candidates` packages each candidate
//! as a single `AddSynapse` operation inside a coordinated structural
//! candidate.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural-network statistics (Issue #873).

use std::collections::{HashMap, HashSet};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;
use crate::analysis::task_descriptor::{TargetTopology, TaskDescriptor};
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Activation threshold above which an output neuron is considered to be
/// firing strongly on a given sample. Chosen to sit comfortably inside the
/// upper half of the unit interval so that genuine co-activation patterns
/// (both outputs trying to "win" the same sample) are picked up while
/// near-zero noise is excluded.
const CO_ACTIVATION_THRESHOLD: f32 = 0.5;

/// Initial inhibitory weight for proposed lateral connections. Small
/// magnitude — the recommendation is structural; the optimiser is expected
/// to tune it. Negative so the receiving neuron is *suppressed* when the
/// sending neuron fires.
const LATERAL_INHIBITION_WEIGHT: f32 = -0.1;

/// Scale factor applied to the co-activation score when estimating the
/// creature-level score gain. Kept small so output-competition candidates
/// rank against the rest of the candidate set without dominating it.
const COMPETITION_GAIN_SCALE: f32 = 0.01;

/// A proposed inhibitory connection between two output neurons.
#[derive(Debug, Clone)]
pub struct OutputCompetitionCandidate {
    /// Source output neuron (the inhibitor — fires first in evaluation order).
    pub from_output_uuid: String,
    /// Target output neuron (the suppressed neuron).
    pub to_output_uuid: String,
    /// Recommended initial weight for the inhibitory synapse. Always
    /// negative; see the private `LATERAL_INHIBITION_WEIGHT` constant.
    pub recommended_weight: f32,
    /// Co-activation score in `[0, 1]` — the mean of `min(a, b)` over
    /// samples where both outputs cross the co-activation threshold.
    pub co_activation_score: f32,
    /// Number of aligned co-activated samples that supported this candidate.
    pub sample_count: usize,
    /// Estimated creature-level score gain from applying the inhibition.
    pub estimated_improvement: f32,
}

/// Whether a descriptor's topology gates the output-competition detector.
fn role_aware_topology(descriptor: &TaskDescriptor) -> bool {
    matches!(
        descriptor.target_topology,
        TargetTopology::OneHot | TargetTopology::Simplex
    )
}

/// Detect output competition under `OneHot` / `Simplex` topologies.
///
/// Returns an empty vector for any other descriptor (regression guard for
/// `OTHER` / `Unknown` / `Independent` / `Margin`).
///
/// # Arguments
/// * `creature` — the creature whose output neurons are inspected.
/// * `neuron_records` — recorded `(uuid, records)` pairs. Only output-neuron
///   entries are consulted; entries for non-outputs are ignored.
/// * `descriptor` — the task descriptor; only `OneHot` / `Simplex`
///   topologies activate detection.
#[must_use]
pub fn detect_output_competition(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
    descriptor: &TaskDescriptor,
) -> Vec<OutputCompetitionCandidate> {
    if !role_aware_topology(descriptor) {
        return Vec::new();
    }

    // Collect output neurons in their `creature.neurons[]` order so that
    // any emitted `AddSynapse(a -> b)` respects forward-only evaluation.
    let output_neurons: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    if output_neurons.len() < 2 {
        return Vec::new();
    }

    // Existing synapse set — we never propose a duplicate.
    let existing: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    // Records lookup by uuid.
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    let mut candidates = Vec::new();

    for i in 0..output_neurons.len() {
        for j in (i + 1)..output_neurons.len() {
            let a = output_neurons[i];
            let b = output_neurons[j];

            if existing.contains(&(a, b)) {
                continue;
            }

            let (Some(a_records), Some(b_records)) = (records_map.get(a), records_map.get(b))
            else {
                continue;
            };

            let Some((co_activation_score, sample_count)) = co_activation(a_records, b_records)
            else {
                continue;
            };

            if sample_count < MIN_DISCOVERY_SAMPLE_COUNT {
                continue;
            }

            candidates.push(OutputCompetitionCandidate {
                from_output_uuid: a.to_string(),
                to_output_uuid: b.to_string(),
                recommended_weight: LATERAL_INHIBITION_WEIGHT,
                co_activation_score,
                sample_count,
                estimated_improvement: co_activation_score * COMPETITION_GAIN_SCALE,
            });
        }
    }

    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));
    candidates
}

/// Compute a co-activation score between two output neurons' records.
///
/// Returns `Some((score, count))` where:
/// - `count` is the number of *aligned* samples (matched by `obs_index`) on
///   which both neurons cross the co-activation threshold.
/// - `score` is the mean of `min(a.activation, b.activation)` over those
///   co-activated samples — bounded in `[threshold, 1]` for bounded-unipolar
///   outputs.
///
/// Returns `None` when no aligned co-activated samples exist.
fn co_activation(
    a_records: &[DiscoverRecord],
    b_records: &[DiscoverRecord],
) -> Option<(f32, usize)> {
    let b_by_idx: HashMap<u32, f32> = b_records
        .iter()
        .map(|r| (r.obs_index, r.activation))
        .collect();

    let mut count = 0_usize;
    let mut sum_min = 0.0_f32;

    for r in a_records {
        let Some(&b_act) = b_by_idx.get(&r.obs_index) else {
            continue;
        };
        if r.activation > CO_ACTIVATION_THRESHOLD && b_act > CO_ACTIVATION_THRESHOLD {
            count += 1;
            sum_min += r.activation.min(b_act);
        }
    }

    if count == 0 {
        return None;
    }

    Some((sum_min / count as f32, count))
}

/// Package output-competition candidates as coordinated structural
/// candidates with a single `AddSynapse` operation each.
#[must_use]
pub fn output_competition_to_coordinated_candidates(
    candidates: &[OutputCompetitionCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: c.from_output_uuid.clone(),
                to_neuron_uuid: c.to_output_uuid.clone(),
                weight: c.recommended_weight,
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Output competition: {} ↔ {} co-fire on {} samples (score {:.3}) → propose inhibitory synapse (weight {:.3})",
                c.from_output_uuid,
                c.to_output_uuid,
                c.sample_count,
                c.co_activation_score,
                c.recommended_weight,
            )),
        });
    }

    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NeuronJson, SynapseJson};

    fn rec(uuid: &str, idx: u32, activation: f32) -> DiscoverRecord {
        DiscoverRecord {
            obs_index: idx,
            neuron_uuid: uuid.to_string(),
            value: Some(0.0),
            activation,
            errors: vec![0.0],
        }
    }

    fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: neuron_type.to_string(),
            squash: "LOGISTIC".to_string(),
            bias: 0.0,
        }
    }

    fn creature_with_two_outputs() -> CreatureJson {
        CreatureJson {
            neurons: vec![
                neuron("input-1", "input"),
                neuron("output-a", "output"),
                neuron("output-b", "output"),
            ],
            synapses: vec![SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-a".to_string(),
                weight: 0.1,
                synapse_type: None,
            }],
            input: 1,
            output: 2,
        }
    }

    #[test]
    fn co_activation_returns_none_for_disjoint_records() {
        let a: Vec<_> = (0..30).map(|i| rec("a", i, 0.9)).collect();
        let b: Vec<_> = (0..30).map(|i| rec("b", i, 0.1)).collect();
        assert!(co_activation(&a, &b).is_none());
    }

    #[test]
    fn co_activation_counts_only_aligned_pairs_above_threshold() {
        let a: Vec<_> = (0..30).map(|i| rec("a", i, 0.8)).collect();
        let b: Vec<_> = (0..30).map(|i| rec("b", i, 0.9)).collect();
        let (score, count) = co_activation(&a, &b).expect("co-activated");
        assert_eq!(count, 30);
        assert!((score - 0.8).abs() < 1e-5);
    }

    #[test]
    fn neutral_descriptor_skips_detection() {
        let creature = creature_with_two_outputs();
        let records = vec![
            (
                "output-a".to_string(),
                (0..30).map(|i| rec("output-a", i, 0.9)).collect(),
            ),
            (
                "output-b".to_string(),
                (0..30).map(|i| rec("output-b", i, 0.9)).collect(),
            ),
        ];
        let candidates = detect_output_competition(&creature, &records, &TaskDescriptor::neutral());
        assert!(candidates.is_empty());
    }
}
