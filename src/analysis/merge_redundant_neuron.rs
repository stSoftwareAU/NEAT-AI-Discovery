//! Merge/fold redundant (highly-correlated) hidden neurons (Issue #1633).
//!
//! ## Why redundant hidden neurons are removable structure
//!
//! When two hidden neurons carry essentially the same signal — their recorded
//! activation vectors are correlated at `|r| > threshold` (default ≈ 0.999) —
//! one of them is pure width. It costs forward-pass and discovery-analysis
//! budget without adding representational capacity. The
//! [`co_adaptation`](super::detection::co_adaptation) detector *observes* this
//! correlation but only proposes a bias-preserving removal or a weight
//! perturbation; neither **folds** the redundant neuron's per-sample signal into
//! its twin.
//!
//! ## How the fold works
//!
//! Fit the linear relationship between the removed neuron `r` and the kept twin
//! `k` by least squares over the recorded window: `a_r ≈ α·a_k + β`, where
//! `α = cov(a_r, a_k) / var(a_k)` and `β = mean(a_r) − α·mean(a_k)`.
//!
//! For each outgoing connection `r → t` with weight `w`, the removed neuron
//! contributes `w · a_r ≈ w·α·a_k + w·β` to target `t`'s pre-activation.
//! Redirecting that contribution onto the twin is therefore exact when the
//! relationship is exact:
//!
//! - the twin edge `k → t` gains weight `α·w` (a [`SetWeight`] on an existing
//!   edge, or an [`AddSynapse`] when the twin has no edge to `t` yet), and
//! - target `t`'s bias gains the constant `β·w` (a [`SetBias`]).
//!
//! The removed neuron and every synapse touching it then disappear via a single
//! [`RemoveNeuron`]. On a genuine duplicate (`α ≈ ±1`, `β ≈ 0`, `r ≈ 1`) every
//! target's pre-activation — `bias_t + Σ_s w_{s→t}·a_{s,i}` — is preserved for
//! every observation, so the network's output is preserved.
//!
//! [`SetWeight`]: crate::CoordinatedStructuralOpJson::SetWeight
//! [`AddSynapse`]: crate::CoordinatedStructuralOpJson::AddSynapse
//! [`SetBias`]: crate::CoordinatedStructuralOpJson::SetBias
//! [`RemoveNeuron`]: crate::CoordinatedStructuralOpJson::RemoveNeuron
//!
//! ## Which neuron is removed
//!
//! The neuron with the **lower downstream impact** — `mean|activation| ×
//! Σ|outgoing weight|` — is folded into its higher-impact twin, so the more
//! influential neuron and its calibrated fan-out survive untouched.
//!
//! ## Anti-correlated and independent pairs are never merged
//!
//! Only **positively** correlated pairs (`r ≥ threshold`) are folded. An
//! anti-correlated pair (`a_r ≈ −a_k`) is deliberately excluded — production
//! snapshot mining (Issue #1631) found genuine duplicates, not opposing pairs,
//! and excluding negative correlation keeps the candidate conservative. An
//! independent pair falls below the threshold and is likewise skipped.
//!
//! ## Evaluate-before-accept
//!
//! Each emitted candidate carries the maximum per-sample residual the fold would
//! introduce, `max_i |w·(a_{r,i} − (α·a_{k,i} + β))|`, so the NEAT-AI controller
//! applies it behind the same evaluate-before-accept ablation gate as the #1623
//! bias-fold work: the fold is validated on the recorded window before the
//! neuron is deleted, and a pair that only *looks* redundant is rejected rather
//! than deleted blind.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)

use std::collections::HashMap;

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;
use crate::analysis::remove_neuron_compensation::{ActivationCovariance, VARIANCE_EPSILON};
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Default minimum Pearson correlation for two hidden neurons to be treated as
/// redundant duplicates. Matches the production snapshot-mining threshold
/// (Issue #1631): `|r| > 0.999` identified genuine duplicates.
pub const MERGE_CORRELATION_THRESHOLD: f64 = 0.999;

/// Base expected creature-score gain attributed to folding one redundant neuron.
/// Small and conservative — the controller's ablation gate is the real arbiter.
const MERGE_BASE_IMPROVEMENT: f32 = 0.003;

/// A detected redundant hidden-neuron pair and the fitted fold that consolidates
/// the lower-impact neuron into its twin (Issue #1633).
#[derive(Debug, Clone, PartialEq)]
pub struct RedundantNeuronPair {
    /// The lower-impact neuron that will be folded away and removed.
    pub remove_uuid: String,
    /// The higher-impact twin that absorbs the removed neuron's fan-out.
    pub keep_uuid: String,
    /// Pearson correlation between the two neurons' activations (positive).
    pub correlation: f64,
    /// Fitted scale `α` in `a_remove ≈ α·a_keep + β`.
    pub scale: f64,
    /// Fitted offset `β` in `a_remove ≈ α·a_keep + β`.
    pub offset: f64,
    /// Number of aligned observations used to fit the relationship.
    pub sample_count: usize,
    /// Maximum per-sample residual the fold would introduce across every
    /// outgoing connection: `max_i |w·(a_remove_i − (α·a_keep_i + β))|`.
    /// (Near-)zero for a genuine duplicate.
    pub max_residual: f64,
    /// Downstream impact of the removed neuron (`mean|act| × Σ|w_out|`).
    pub remove_impact: f64,
    /// Downstream impact of the kept twin.
    pub keep_impact: f64,
}

/// Per-observation activation vector for one hidden neuron, keyed by obs index.
type ActivationMap = HashMap<u32, f32>;

/// Build the eligible hidden-neuron activation maps, keeping only hidden neurons
/// with at least [`MIN_DISCOVERY_SAMPLE_COUNT`] recorded samples.
fn eligible_activation_maps<'a>(
    creature: &'a CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<(&'a str, ActivationMap)> {
    let records_map: HashMap<&str, &[DiscoverRecord]> = neuron_records
        .iter()
        .map(|(uuid, recs)| (uuid.as_str(), recs.as_ref()))
        .collect();

    let mut maps: Vec<(&str, ActivationMap)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .filter_map(|n| {
            let recs = records_map.get(n.uuid.as_str())?;
            if recs.len() < MIN_DISCOVERY_SAMPLE_COUNT {
                return None;
            }
            let map: ActivationMap = recs.iter().map(|r| (r.obs_index, r.activation)).collect();
            Some((n.uuid.as_str(), map))
        })
        .collect();

    // Deterministic order so candidate generation is replayable.
    maps.sort_by(|a, b| a.0.cmp(b.0));
    maps
}

/// Sum of absolute outgoing synapse weights for a neuron.
fn outgoing_weight_magnitude(creature: &CreatureJson, uuid: &str) -> f64 {
    creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid == uuid)
        .map(|s| f64::from(s.weight).abs())
        .sum()
}

/// Downstream impact estimate: `mean|activation| × Σ|outgoing weight|`.
fn downstream_impact(creature: &CreatureJson, uuid: &str, activations: &ActivationMap) -> f64 {
    let mean_abs = if activations.is_empty() {
        0.0
    } else {
        activations
            .values()
            .map(|v| f64::from(v.abs()))
            .sum::<f64>()
            / activations.len() as f64
    };
    mean_abs * outgoing_weight_magnitude(creature, uuid)
}

/// `true` when a synapse connects the two neurons in either direction — folding
/// such a pair would create a self-loop, so it is skipped.
fn directly_connected(creature: &CreatureJson, a: &str, b: &str) -> bool {
    creature
        .synapses
        .iter()
        .any(|s| (s.from_uuid == a && s.to_uuid == b) || (s.from_uuid == b && s.to_uuid == a))
}

/// Detect redundant hidden-neuron pairs using the default correlation threshold.
///
/// See [`detect_redundant_neuron_pairs_with_threshold`].
#[must_use]
pub fn detect_redundant_neuron_pairs(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<RedundantNeuronPair> {
    detect_redundant_neuron_pairs_with_threshold(
        creature,
        neuron_records,
        MERGE_CORRELATION_THRESHOLD,
    )
}

/// Detect hidden-neuron pairs whose activations are positively correlated at or
/// above `threshold`, fitting the fold that consolidates the lower-impact neuron
/// into its twin (Issue #1633).
///
/// Anti-correlated and sub-threshold (independent) pairs are excluded. A pair
/// whose two neurons are directly connected is skipped to avoid a self-loop.
/// Each neuron participates in at most one emitted pair (the highest-correlation
/// one), so the returned pairs are mutually independent and can be applied in
/// any order.
///
/// Returns pairs sorted by correlation, strongest first.
#[must_use]
pub fn detect_redundant_neuron_pairs_with_threshold(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    threshold: f64,
) -> Vec<RedundantNeuronPair> {
    let maps = eligible_activation_maps(creature, neuron_records);
    if maps.len() < 2 {
        return Vec::new();
    }

    // Gather every qualifying pair first, then greedily keep the strongest
    // non-overlapping ones so each neuron is folded at most once.
    let mut scored: Vec<RedundantNeuronPair> = Vec::new();

    for i in 0..maps.len() {
        for j in (i + 1)..maps.len() {
            let (uuid_a, map_a) = (maps[i].0, &maps[i].1);
            let (uuid_b, map_b) = (maps[j].0, &maps[j].1);

            // Align on shared observation indices.
            let pairs: Vec<(f64, f64)> = map_a
                .iter()
                .filter_map(|(obs, &act_a)| {
                    map_b
                        .get(obs)
                        .map(|&act_b| (f64::from(act_a), f64::from(act_b)))
                })
                .collect();

            if pairs.len() < MIN_DISCOVERY_SAMPLE_COUNT {
                continue;
            }

            let Some(stats) = ActivationCovariance::from_pairs(pairs.iter().copied()) else {
                continue;
            };

            let correlation = stats.correlation();
            // Only positively correlated genuine duplicates are folded.
            if correlation < threshold {
                continue;
            }

            if directly_connected(creature, uuid_a, uuid_b) {
                continue;
            }

            // Fold the lower-impact neuron into the higher-impact twin.
            let impact_a = downstream_impact(creature, uuid_a, map_a);
            let impact_b = downstream_impact(creature, uuid_b, map_b);
            let (remove_uuid, keep_uuid, remove_impact, keep_impact) = if impact_a <= impact_b {
                (uuid_a, uuid_b, impact_a, impact_b)
            } else {
                (uuid_b, uuid_a, impact_b, impact_a)
            };

            // Refit in remove→keep orientation: a_remove ≈ α·a_keep + β.
            let oriented: Vec<(f64, f64)> = if remove_uuid == uuid_a {
                pairs.clone()
            } else {
                pairs.iter().map(|&(a, b)| (b, a)).collect()
            };
            let Some(fit) = ActivationCovariance::from_pairs(oriented.iter().copied()) else {
                continue;
            };
            if fit.survivor_variance <= VARIANCE_EPSILON {
                // Kept neuron is constant — nothing to regress onto.
                continue;
            }
            let scale = fit.covariance / fit.survivor_variance;
            let offset = fit.candidate_mean - scale * fit.survivor_mean;

            // Residual the fold would leave, scaled by the largest outgoing weight.
            let max_dev = oriented
                .iter()
                .map(|&(a_remove, a_keep)| (a_remove - (scale * a_keep + offset)).abs())
                .fold(0.0_f64, f64::max);
            let max_weight = creature
                .synapses
                .iter()
                .filter(|s| s.from_uuid == remove_uuid)
                .map(|s| f64::from(s.weight).abs())
                .fold(0.0_f64, f64::max);
            let max_residual = max_dev * max_weight;

            scored.push(RedundantNeuronPair {
                remove_uuid: remove_uuid.to_string(),
                keep_uuid: keep_uuid.to_string(),
                correlation,
                scale,
                offset,
                sample_count: oriented.len(),
                max_residual,
                remove_impact,
                keep_impact,
            });
        }
    }

    // Strongest correlation first, then greedily drop overlapping pairs.
    scored.sort_by(|a, b| b.correlation.total_cmp(&a.correlation));
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut result = Vec::new();
    for pair in scored {
        if used.contains(&pair.remove_uuid) || used.contains(&pair.keep_uuid) {
            continue;
        }
        used.insert(pair.remove_uuid.clone());
        used.insert(pair.keep_uuid.clone());
        result.push(pair);
    }
    result
}

/// Convert detected redundant pairs into coordinated structural candidates that
/// fold the redundant neuron into its twin and remove it (Issue #1633).
///
/// Each candidate redirects the removed neuron's outgoing weights onto the twin
/// (scaled by the fitted `α`), folds the constant `β` term into each target's
/// bias, and removes the redundant neuron with a single `RemoveNeuron` (which
/// also drops every synapse touching it). The NEAT-AI controller validates each
/// candidate through its evaluate-before-accept ablation gate.
#[must_use]
pub fn redundant_pairs_to_coordinated_candidates(
    pairs: &[RedundantNeuronPair],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    // Existing synapse weights and neuron biases for absolute-value ops.
    let mut weight_of: HashMap<(&str, &str), f32> = HashMap::new();
    for s in &creature.synapses {
        weight_of.insert((s.from_uuid.as_str(), s.to_uuid.as_str()), s.weight);
    }
    let mut bias_of: HashMap<&str, f32> = HashMap::new();
    for n in &creature.neurons {
        bias_of.insert(n.uuid.as_str(), n.bias);
    }

    let mut results = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let remove = pair.remove_uuid.as_str();
        let keep = pair.keep_uuid.as_str();

        // Aggregate the removed neuron's fan-out per target.
        let mut redirect: HashMap<&str, f64> = HashMap::new();
        let mut bias_delta: HashMap<&str, f64> = HashMap::new();
        for syn in creature.synapses.iter().filter(|s| s.from_uuid == remove) {
            let target = syn.to_uuid.as_str();
            let w = f64::from(syn.weight);
            *redirect.entry(target).or_insert(0.0) += pair.scale * w;
            *bias_delta.entry(target).or_insert(0.0) += pair.offset * w;
        }

        // Deterministic op ordering.
        let mut targets: Vec<&str> = redirect.keys().copied().collect();
        targets.sort_unstable();

        let mut operations: Vec<CoordinatedStructuralOpJson> = Vec::new();
        for target in &targets {
            let delta_w = redirect[target];
            #[allow(clippy::cast_possible_truncation)]
            let new_weight = match weight_of.get(&(keep, *target)) {
                Some(existing) => {
                    let updated = f64::from(*existing) + delta_w;
                    operations.push(CoordinatedStructuralOpJson::SetWeight {
                        from_neuron_uuid: keep.to_string(),
                        to_neuron_uuid: (*target).to_string(),
                        weight: updated as f32,
                    });
                    continue;
                }
                None => delta_w as f32,
            };
            operations.push(CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: keep.to_string(),
                to_neuron_uuid: (*target).to_string(),
                weight: new_weight,
            });
        }

        // Fold the constant β·w term into each target's bias.
        for target in &targets {
            let delta_b = bias_delta[target];
            if delta_b.abs() <= f64::EPSILON {
                continue;
            }
            let base = f64::from(bias_of.get(*target).copied().unwrap_or(0.0));
            #[allow(clippy::cast_possible_truncation)]
            operations.push(CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: (*target).to_string(),
                bias: (base + delta_b) as f32,
            });
        }

        // Finally remove the redundant neuron (drops its remaining synapses).
        operations.push(CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: remove.to_string(),
        });

        // Higher correlation → cleaner fold → slightly higher expected gain.
        let severity = ((pair.correlation - MERGE_CORRELATION_THRESHOLD)
            / (1.0 - MERGE_CORRELATION_THRESHOLD))
            .clamp(0.0, 1.0);
        #[allow(clippy::cast_possible_truncation)]
        let expected_gain = MERGE_BASE_IMPROVEMENT * (0.5 + 0.5 * severity as f32);

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: expected_gain,
            comment: Some(format!(
                "[#1633] Merge redundant neurons: {remove} folds into {keep} \
                 (correlation {:.4}, scale {:.3}, offset {:.3}, {} samples, \
                 max residual {:.3e})",
                pair.correlation, pair.scale, pair.offset, pair.sample_count, pair.max_residual,
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
    use crate::ffi_types::{NeuronJson, SynapseJson};

    fn neuron(uuid: &str, neuron_type: &str, bias: f32) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: neuron_type.to_string(),
            squash: "IDENTITY".to_string(),
            bias,
        }
    }

    fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
        SynapseJson {
            from_uuid: from.to_string(),
            to_uuid: to.to_string(),
            weight,
            synapse_type: None,
        }
    }

    fn records_for(uuid: &str, activations: &[f32]) -> (String, Vec<DiscoverRecord>) {
        let recs = activations
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                DiscoverRecord::new(
                    u32::try_from(i).expect("obs index fits u32"),
                    uuid.to_string(),
                    None,
                    a,
                    vec![],
                )
            })
            .collect();
        (uuid.to_string(), recs)
    }

    /// 24 non-trivial samples so pairs clear `MIN_DISCOVERY_SAMPLE_COUNT` (20).
    fn ramp(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
        (0..n).map(f).collect()
    }

    #[test]
    fn identical_activations_emit_exactly_one_merge_candidate() {
        // dup1 and dup2 share identical activations; indep is independent.
        let creature = CreatureJson {
            neurons: vec![
                neuron("dup1", "hidden", 0.0),
                neuron("dup2", "hidden", 0.0),
                neuron("indep", "hidden", 0.0),
                neuron("out", "output", 0.1),
            ],
            synapses: vec![
                synapse("dup1", "out", 1.0),
                synapse("dup2", "out", 1.5),
                synapse("indep", "out", 0.7),
            ],
            input: 1,
            output: 1,
        };
        let dup = ramp(24, |i| (i as f32 * 0.13).sin());
        let indep = ramp(24, |i| (i as f32 * 0.71 + 2.0).cos());
        let records = vec![
            records_for("dup1", &dup),
            records_for("dup2", &dup),
            records_for("indep", &indep),
        ];

        let pairs = detect_redundant_neuron_pairs(&creature, &records);
        assert_eq!(pairs.len(), 1, "exactly one duplicate pair expected");
        let p = &pairs[0];
        let merged = [p.remove_uuid.as_str(), p.keep_uuid.as_str()];
        assert!(merged.contains(&"dup1") && merged.contains(&"dup2"));
        assert!(
            !merged.contains(&"indep"),
            "independent neuron must not merge"
        );
        assert!(
            (p.scale - 1.0).abs() < 1e-4,
            "scale ~1 for identical: {}",
            p.scale
        );
        assert!(
            p.offset.abs() < 1e-4,
            "offset ~0 for identical: {}",
            p.offset
        );
        assert!(
            p.max_residual < 1e-4,
            "identical duplicate leaves ~0 residual"
        );

        let candidates = redundant_pairs_to_coordinated_candidates(&pairs, &creature);
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn scale_shifted_duplicate_uses_fitted_scale() {
        // a_dup2 = 2 * a_dup1 → detected, fold scale reflects the 2x relationship.
        let creature = CreatureJson {
            neurons: vec![
                neuron("dup1", "hidden", 0.0),
                neuron("dup2", "hidden", 0.0),
                neuron("out", "output", 0.0),
            ],
            synapses: vec![synapse("dup1", "out", 1.0), synapse("dup2", "out", 2.0)],
            input: 1,
            output: 1,
        };
        let base = ramp(24, |i| (i as f32 * 0.17).sin() + 0.5);
        let doubled: Vec<f32> = base.iter().map(|v| v * 2.0).collect();
        let records = vec![records_for("dup1", &base), records_for("dup2", &doubled)];

        let pairs = detect_redundant_neuron_pairs(&creature, &records);
        assert_eq!(pairs.len(), 1);
        let p = &pairs[0];
        // dup1 has lower downstream impact (smaller mean|act| and weight) → removed.
        assert_eq!(p.remove_uuid, "dup1");
        assert_eq!(p.keep_uuid, "dup2");
        // a_dup1 ≈ 0.5 * a_dup2 → scale ~0.5.
        assert!(
            (p.scale - 0.5).abs() < 1e-4,
            "expected scale ~0.5, got {}",
            p.scale
        );
        assert!(p.offset.abs() < 1e-4, "offset ~0, got {}", p.offset);
        assert!(p.max_residual < 1e-4);
    }

    #[test]
    fn anti_correlated_pair_is_not_merged() {
        let creature = CreatureJson {
            neurons: vec![
                neuron("a", "hidden", 0.0),
                neuron("b", "hidden", 0.0),
                neuron("out", "output", 0.0),
            ],
            synapses: vec![synapse("a", "out", 1.0), synapse("b", "out", 1.0)],
            input: 1,
            output: 1,
        };
        let base = ramp(24, |i| (i as f32 * 0.23).sin());
        let negated: Vec<f32> = base.iter().map(|v| -v).collect();
        let records = vec![records_for("a", &base), records_for("b", &negated)];

        let pairs = detect_redundant_neuron_pairs(&creature, &records);
        assert!(pairs.is_empty(), "anti-correlated pair must not merge");
    }

    #[test]
    fn independent_pair_is_not_merged() {
        let creature = CreatureJson {
            neurons: vec![
                neuron("a", "hidden", 0.0),
                neuron("b", "hidden", 0.0),
                neuron("out", "output", 0.0),
            ],
            synapses: vec![synapse("a", "out", 1.0), synapse("b", "out", 1.0)],
            input: 1,
            output: 1,
        };
        let a = ramp(24, |i| (i as f32 * 0.29).sin());
        let b = ramp(24, |i| (i as f32 * 0.91 + 1.3).cos());
        let records = vec![records_for("a", &a), records_for("b", &b)];

        let pairs = detect_redundant_neuron_pairs(&creature, &records);
        assert!(pairs.is_empty(), "independent pair must not merge");
    }

    #[test]
    fn candidate_ops_redirect_weight_and_remove_neuron() {
        let creature = CreatureJson {
            neurons: vec![
                neuron("dup1", "hidden", 0.0),
                neuron("dup2", "hidden", 0.0),
                neuron("out", "output", 0.1),
            ],
            synapses: vec![synapse("dup1", "out", 1.0), synapse("dup2", "out", 1.5)],
            input: 1,
            output: 1,
        };
        let dup = ramp(24, |i| (i as f32 * 0.13).sin() + 0.3);
        let records = vec![records_for("dup1", &dup), records_for("dup2", &dup)];

        let pairs = detect_redundant_neuron_pairs(&creature, &records);
        assert_eq!(pairs.len(), 1);
        let candidates = redundant_pairs_to_coordinated_candidates(&pairs, &creature);
        assert_eq!(candidates.len(), 1);
        let ops = &candidates[0].operations;

        // Must remove the redundant neuron.
        let removed: Vec<&str> = ops
            .iter()
            .filter_map(|op| match op {
                CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } => {
                    Some(neuron_uuid.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(removed.len(), 1);
        let remove_uuid = removed[0];
        let keep_uuid = if remove_uuid == "dup1" {
            "dup2"
        } else {
            "dup1"
        };

        // Must redirect the removed neuron's out-weight onto the twin's edge.
        let set_weight = ops.iter().find_map(|op| match op {
            CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid,
                to_neuron_uuid,
                weight,
            } if from_neuron_uuid == keep_uuid && to_neuron_uuid == "out" => Some(*weight),
            _ => None,
        });
        let new_weight = set_weight.expect("twin edge weight redirect expected");

        // The fold preserves the summed contribution into `out` per observation.
        // Before: w_r·a_r + w_k·a_k. After: new_weight·a_k (+ bias, ~0 here).
        let w_remove = if remove_uuid == "dup1" { 1.0_f32 } else { 1.5 };
        let w_keep = if keep_uuid == "dup1" { 1.0_f32 } else { 1.5 };
        for &a in &dup {
            let before = w_remove * a + w_keep * a;
            let after = new_weight * a; // identical duplicate, offset ~0
            assert!(
                (before - after).abs() < 1e-3,
                "fold must preserve output: before={before}, after={after}"
            );
        }
    }
}
