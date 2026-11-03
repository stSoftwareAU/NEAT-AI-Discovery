//! Restricted activation range detection module (Issue #399).
//!
//! Hidden neurons may operate in a restricted sub-range of their activation
//! function's output domain. For example, a TANH neuron consistently outputting
//! values in [0.1, 0.3] is only using 10% of its [-1, 1] range. This wastes
//! representational capacity and suggests the neuron's activation function,
//! bias, or incoming weights are misconfigured.
//!
//! ## Relationship to Other Detection Modules
//!
//! - **Saturation detection** (Issue #342): neurons stuck at the *bounds* of
//!   their activation (e.g., TANH at ±1).
//! - **Dead neuron detection** (Issue #341): neurons with near-zero activation.
//! - **Bounded range / sentinel detection** (Issue #395): neurons with sentinel
//!   clusters at boundary values (e.g., -1 meaning "null").
//! - **This module** (Issue #399): neurons *active* but confined to a narrow
//!   band *within* the theoretical bounds.
//!
//! ## Detection Criteria
//!
//! A hidden neuron has a "restricted range" if:
//! 1. It uses a bounded squash function (TANH, LOGISTIC, etc.).
//! 2. `range_utilisation = (activation_max - activation_min) / theoretical_range`
//!    is below a configurable threshold (default: 20%).
//! 3. The neuron is not dead (observed range is not near zero).
//! 4. The neuron is not saturated (not at the activation bounds).
//! 5. Sufficient samples are available (≥ 20).
//!
//! ## Recommended Actions
//!
//! - `changeSquash` — switch to an activation function better suited to the
//!   observed range (e.g., IDENTITY for a TANH neuron in a narrow band).
//! - `setBias` — adjust bias to centre the neuron in its active region.
//! - `setWeight` — scale incoming weights to expand the operating range.

use super::helpers::{compute_activation_range, sort_candidates_by_score_gain};
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

// MIN_SAMPLES moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;
use crate::analysis::utils::deadline_passed;

/// Minimum observed range (`activation_max` - `activation_min`) to distinguish
/// from dead neurons. Below this, the neuron is considered dead, not restricted.
const MIN_OBSERVED_RANGE: f32 = 0.01;

/// Distance from the theoretical bounds within which a neuron is considered
/// saturated rather than restricted. If both `activation_min` and `activation_max`
/// fall within this margin of the bounds, it is saturation.
const SATURATION_MARGIN: f32 = 0.10;

/// Configuration for restricted range detection.
#[derive(Debug, Clone)]
pub struct RestrictedRangeConfig {
    /// Maximum range utilisation below which a neuron is flagged (default: 0.20 = 20%).
    pub utilisation_threshold: f32,
    /// Minimum samples required for detection.
    pub min_samples: usize,
}

impl Default for RestrictedRangeConfig {
    fn default() -> Self {
        Self {
            utilisation_threshold: 0.20,
            min_samples: MIN_SAMPLES,
        }
    }
}

/// Result of detecting a restricted-range neuron.
#[derive(Debug, Clone)]
pub struct RestrictedRangeNeuron {
    /// UUID of the neuron with the restricted range.
    pub neuron_uuid: String,
    /// Current squash (activation) function.
    pub squash: String,
    /// Current bias of the neuron.
    pub bias: f32,
    /// Minimum observed activation across samples.
    pub activation_min: f32,
    /// Maximum observed activation across samples.
    pub activation_max: f32,
    /// Theoretical range of the squash function (e.g., 2.0 for TANH).
    pub theoretical_range: f32,
    /// Ratio of observed range to theoretical range (0.0 to 1.0).
    pub range_utilisation: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
}

/// Returns the theoretical output range `(min, max)` for bounded squash functions.
///
/// Returns `None` for unbounded activations (IDENTITY, RELU, etc.) since they
/// have no fixed output bounds and restricted range detection does not apply.
fn theoretical_bounds(squash: &str) -> Option<(f32, f32)> {
    match squash {
        "TANH" | "HARD_TANH" | "CLIPPED" | "BIPOLAR_SIGMOID" | "SOFTSIGN" | "ISRU" => {
            Some((-1.0, 1.0))
        }
        "LOGISTIC" => Some((0.0, 1.0)),
        "BIPOLAR" | "STEP" => Some((-1.0, 1.0)),
        "ARCTAN" => {
            // arctan output range is approximately (-π/2, π/2) ≈ (-1.5708, 1.5708)
            Some((-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2))
        }
        "RELU6" => Some((0.0, 6.0)),
        _ => None, // Unbounded: IDENTITY, RELU, LEAKYRELU, ELU, SELU, GELU, MISH, etc.
    }
}

/// Detect hidden neurons with restricted activation ranges.
///
/// Analyses hidden neurons to find those where the observed activation range
/// is a small fraction of the theoretical range of their squash function.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
/// * `config` - Detection configuration (thresholds).
///
/// # Returns
/// A list of `RestrictedRangeNeuron` sorted by range utilisation (lowest first).
pub fn detect_restricted_range_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    config: &RestrictedRangeConfig,
) -> Vec<RestrictedRangeNeuron> {
    // Build map from UUID to neuron info (only hidden neurons)
    let hidden_neurons: Vec<(&str, &str, f32)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.as_str(), n.squash.as_str(), n.bias))
        .collect();

    let neuron_map: std::collections::HashMap<&str, (&str, f32)> = hidden_neurons
        .iter()
        .map(|&(uuid, squash, bias)| (uuid, (squash, bias)))
        .collect();

    let mut results = Vec::with_capacity(neuron_records.len());

    for (uuid, records) in neuron_records {
        let records = records.as_ref();
        // Only consider hidden neurons
        let Some(&(squash, bias)) = neuron_map.get(uuid.as_str()) else {
            continue;
        };

        // Skip if insufficient samples
        if records.len() < config.min_samples {
            continue;
        }

        // Skip unbounded activations
        let Some((theoretical_min, theoretical_max)) = theoretical_bounds(squash) else {
            continue;
        };
        let theoretical_range = theoretical_max - theoretical_min;

        // Compute observed activation range (Issue #941: shared helper)
        let act_range = compute_activation_range(records);
        let act_min = act_range.min;
        let act_max = act_range.max;
        let observed_range = act_range.span;

        // Exclude dead neurons (near-zero range)
        if observed_range < MIN_OBSERVED_RANGE {
            continue;
        }

        // Exclude saturated neurons (at bounds)
        // A neuron is "at bounds" if its activation range touches the theoretical
        // boundary within SATURATION_MARGIN
        let near_lower = act_min < (theoretical_min + SATURATION_MARGIN);
        let near_upper = act_max > (theoretical_max - SATURATION_MARGIN);
        if near_lower || near_upper {
            continue;
        }

        // Compute range utilisation
        let range_utilisation = observed_range / theoretical_range;

        if range_utilisation >= config.utilisation_threshold {
            continue;
        }

        results.push(RestrictedRangeNeuron {
            neuron_uuid: uuid.clone(),
            squash: squash.to_string(),
            bias,
            activation_min: act_min,
            activation_max: act_max,
            theoretical_range,
            range_utilisation,
            sample_count: records.len(),
        });
    }

    // Sort by range utilisation (lowest first — worst offenders first)
    results.sort_by(|a, b| a.range_utilisation.total_cmp(&b.range_utilisation));

    results
}

/// Convert restricted range neurons into coordinated structural candidates.
///
/// Each restricted-range neuron may produce multiple candidates:
/// 1. **changeSquash** — switch to IDENTITY to remove the bounding constraint.
/// 2. **setBias** — adjust bias to centre the operating range.
/// 3. **setWeight** — scale incoming weights to expand the range.
///
/// Bias and weight adjustments are combined into a `coordinatedStructural`
/// candidate for atomic application.
///
/// Issue #2479: no deadline — equivalent to calling
/// [`restricted_range_to_coordinated_candidates_with_deadline`] with a `None`
/// deadline.
pub fn restricted_range_to_coordinated_candidates(
    detected: &[RestrictedRangeNeuron],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    restricted_range_to_coordinated_candidates_with_deadline(detected, creature, &None)
}

/// Deadline-aware variant of [`restricted_range_to_coordinated_candidates`]
/// (Issue #2479).
///
/// Builds the target-uuid → incoming-synapses lookup once up front (rather than
/// scanning `creature.synapses` per neuron, which was O(C·S)) and checks
/// `deadline` once per candidate neuron so a hung or oversized conversion can be
/// stopped mid-flight, returning the coordinated candidates built so far.
pub fn restricted_range_to_coordinated_candidates_with_deadline(
    detected: &[RestrictedRangeNeuron],
    creature: &CreatureJson,
    deadline: &Option<std::time::SystemTime>,
) -> Vec<CoordinatedStructuralCandidateJson> {
    restricted_range_to_coordinated_candidates_with_observer(
        detected,
        creature,
        deadline,
        &mut || {},
    )
}

fn restricted_range_to_coordinated_candidates_with_observer(
    detected: &[RestrictedRangeNeuron],
    creature: &CreatureJson,
    deadline: &Option<std::time::SystemTime>,
    work_observer: &mut impl FnMut(),
) -> Vec<CoordinatedStructuralCandidateJson> {
    // Issue #2479: build the to_uuid -> incoming synapses lookup once, in synapse
    // order, so each candidate's list keeps the order of the former per-candidate filter.
    let mut incoming_by_target: std::collections::HashMap<&str, Vec<&crate::SynapseJson>> =
        std::collections::HashMap::new();
    for synapse in &creature.synapses {
        work_observer();
        incoming_by_target
            .entry(synapse.to_uuid.as_str())
            .or_default()
            .push(synapse);
    }

    let mut results = Vec::new();

    for n in detected {
        if deadline_passed(deadline) {
            break;
        }

        let base_improvement = (1.0 - n.range_utilisation) * 0.005;

        // Candidate 1: Change squash to IDENTITY (removes bounding)
        results.push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
            operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: n.neuron_uuid.clone(),
                squash: "IDENTITY".to_string(),
            }],
            expected_creature_score_gain: base_improvement,
            comment: Some(format!(
                "Restricted range on {}: {} using {:.0}% of range [{:.3}, {:.3}] → change to IDENTITY",
                n.neuron_uuid,
                n.squash,
                n.range_utilisation * 100.0,
                n.activation_min,
                n.activation_max,
            )),
        });

        // Candidate 2: Adjust bias to centre the operating region
        // The ideal centre of the activation function is the midpoint of the theoretical range.
        // Move bias so the neuron's mean activation shifts toward that centre.
        let observed_centre = (n.activation_min + n.activation_max) / 2.0;
        let (theoretical_min, theoretical_max) =
            theoretical_bounds(&n.squash).unwrap_or((-1.0, 1.0));
        let theoretical_centre = (theoretical_min + theoretical_max) / 2.0;
        let bias_delta = theoretical_centre - observed_centre;
        let new_bias = n.bias + bias_delta;

        results.push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
            operations: vec![CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: n.neuron_uuid.clone(),
                bias: new_bias,
            }],
            expected_creature_score_gain: base_improvement * 0.7,
            comment: Some(format!(
                "Restricted range on {}: {} using {:.0}% → adjust bias from {:.3} to {:.3} to centre activation",
                n.neuron_uuid,
                n.squash,
                n.range_utilisation * 100.0,
                n.bias,
                new_bias,
            )),
        });

        // Candidate 3: Scale incoming weights to expand the operating range
        // Find all synapses targeting this neuron and compute a scale factor
        work_observer();
        let incoming_synapses: Vec<&crate::SynapseJson> = incoming_by_target
            .get(n.neuron_uuid.as_str())
            .cloned()
            .unwrap_or_default();

        if !incoming_synapses.is_empty() {
            // Scale factor: we want the observed range to expand to fill more of the
            // theoretical range. Target ~80% utilisation.
            let target_utilisation = 0.80;
            let scale_factor = target_utilisation / n.range_utilisation.max(0.01);

            let mut weight_ops: Vec<CoordinatedStructuralOpJson> = Vec::new();
            for s in &incoming_synapses {
                weight_ops.push(CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: s.from_uuid.clone(),
                    to_neuron_uuid: s.to_uuid.clone(),
                    weight: s.weight * scale_factor,
                });
            }

            // Also adjust bias proportionally
            weight_ops.push(CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: n.neuron_uuid.clone(),
                bias: n.bias * scale_factor,
            });

            results.push(CoordinatedStructuralCandidateJson {
                remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
                operations: weight_ops,
                expected_creature_score_gain: base_improvement * 0.5,
                comment: Some(format!(
                    "Restricted range on {}: {} using {:.0}% → scale weights by {:.2}× to expand range",
                    n.neuron_uuid,
                    n.squash,
                    n.range_utilisation * 100.0,
                    scale_factor,
                )),
            });
        }
    }

    // Sort by expected improvement (best first) (Issue #941: shared helper)
    sort_candidates_by_score_gain(&mut results);

    results
}

#[cfg(test)]
mod issue_2479_tests {
    use super::*;

    fn neuron(uuid: &str) -> RestrictedRangeNeuron {
        RestrictedRangeNeuron {
            neuron_uuid: uuid.to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
            activation_min: -0.2,
            activation_max: 0.2,
            theoretical_range: 2.0,
            range_utilisation: 0.1,
            sample_count: 30,
        }
    }

    /// One hidden neuron per uuid, each fed by two incoming synapses.
    fn creature_with_hidden(uuids: &[String]) -> CreatureJson {
        let mut neurons = vec![
            crate::NeuronJson {
                uuid: "input-a".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            crate::NeuronJson {
                uuid: "input-b".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ];
        let mut synapses = Vec::new();
        for uuid in uuids {
            neurons.push(crate::NeuronJson {
                uuid: uuid.clone(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            });
            for from in ["input-a", "input-b"] {
                synapses.push(crate::SynapseJson {
                    from_uuid: from.to_string(),
                    to_uuid: uuid.clone(),
                    weight: 1.0,
                    synapse_type: None,
                });
            }
        }
        CreatureJson {
            neurons,
            synapses,
            input: 2,
            output: 1,
        }
    }

    #[test]
    fn incoming_synapse_work_scales_linearly_at_n_and_4n() {
        const N: usize = 128;

        let measured_work = |count: usize| {
            let uuids: Vec<String> = (0..count).map(|i| format!("h{i}")).collect();
            let creature = creature_with_hidden(&uuids);
            let detected: Vec<RestrictedRangeNeuron> =
                uuids.iter().map(|u| neuron(u.as_str())).collect();
            let mut work = 0;
            let result = restricted_range_to_coordinated_candidates_with_observer(
                &detected,
                &creature,
                &None,
                &mut || work += 1,
            );
            assert_eq!(result.len(), 3 * count);
            work
        };

        // Index build: one step per synapse (two per hidden neuron), then one
        // lookup per neuron.
        let small_work = measured_work(N);
        let large_work = measured_work(N * 4);
        assert_eq!(small_work, 2 * N + N);
        assert_eq!(large_work, 2 * (N * 4) + N * 4);
        assert_eq!(large_work, small_work * 4);
    }

    /// The deadline is checked once per neuron, and on expiry the candidates
    /// already built are returned rather than discarded. The scripted sequence lets
    /// the first neuron through and stops at the second, so hoisting the check out of
    /// the loop (all nine built) or replacing `break` with an empty return (none
    /// built) both fail this test.
    #[test]
    #[serial_test::serial]
    fn deadline_mid_conversion_returns_the_candidates_built_so_far() {
        use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;

        let uuids = [
            "hidden-alpha".to_string(),
            "hidden-beta".to_string(),
            "hidden-gamma".to_string(),
        ];
        let creature = creature_with_hidden(&uuids);
        let detected: Vec<RestrictedRangeNeuron> =
            uuids.iter().map(|u| neuron(u.as_str())).collect();

        // Positive control: with no override every neuron converts to three candidates.
        let control =
            restricted_range_to_coordinated_candidates_with_deadline(&detected, &creature, &None);
        assert_eq!(
            control.len(),
            9,
            "fixture must yield three candidates per neuron"
        );

        let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);
        let partial =
            restricted_range_to_coordinated_candidates_with_deadline(&detected, &creature, &None);

        assert_eq!(
            partial.len(),
            3,
            "the deadline must stop the conversion at the second neuron and keep the first's three candidates"
        );
        for candidate in &partial {
            let comment = candidate.comment.as_deref().unwrap_or_default();
            assert!(
                comment.contains("hidden-alpha"),
                "every kept candidate must belong to the first neuron, got: {comment}"
            );
        }
    }
}
