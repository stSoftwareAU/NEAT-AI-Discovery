//! Reconstruction-mismatch focus signal (Issue #1634).
//!
//! Focus selection prioritises neurons the current model of the creature fails
//! to explain. The per-neuron **reconstruction mismatch** measures exactly
//! that: for each selectable neuron we reconstruct its activation from its
//! inbound synapses — `squash(bias + Σ from_activation × weight)` — and compare
//! it to the recorded activation. A large mean delta means a squash/bias/
//! structural change on that neuron is high-leverage, so it should rise in the
//! focus budget.
//!
//! This mirrors the reconstruction check computed at export time
//! (`src/export/snapshot.rs`) but returns only the aggregate mean activation
//! delta per neuron, computed on demand from the ranking record provider so the
//! signal is available without an export pass.

#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use anyhow::Result;

use crate::activations::apply_scalar_squash;
use crate::{CreatureJson, NeuronJson};

use super::record_providers::RecordProvider;

/// Per-neuron recorded activation indexed by observation index.
type ActivationByObs = HashMap<u32, f32>;

/// Apply a neuron's squash to a reconstructed value (Issue #1634).
///
/// Mirrors the export reconstruction: aggregate squashes (MIN/MAX/IF) that
/// cannot be represented as `f(value)` pass the value through unchanged.
fn apply_squash(squash: &str, value: f32) -> f32 {
    apply_scalar_squash(squash, value).unwrap_or(value)
}

/// Reconstruct one neuron's mean absolute activation delta from its inbound
/// synapses (Issue #1634).
///
/// Pure over its inputs (no I/O). `inbound` lists `(from_uuid, weight)` for
/// every synapse feeding this neuron; `activations` maps each neuron UUID to
/// its recorded activation-by-observation. Only observations present in the
/// target neuron's own recording are scored; a contributing source missing an
/// observation contributes `0.0` for that observation (matching the export
/// reconstruction, which sums only over sources it has a recording for).
///
/// Returns `0.0` when the neuron has no recorded observations.
#[must_use]
pub(super) fn mean_activation_delta_for_neuron(
    neuron: &NeuronJson,
    inbound: &[(&str, f32)],
    activations: &HashMap<String, ActivationByObs>,
    self_activation: &ActivationByObs,
) -> f32 {
    if self_activation.is_empty() {
        return 0.0;
    }

    let mut sum_delta = 0.0f32;
    let mut count = 0u32;

    for (&obs_index, &recorded_activation) in self_activation {
        if !recorded_activation.is_finite() {
            continue;
        }

        let mut reconstructed_value = neuron.bias;
        for &(from_uuid, weight) in inbound {
            if let Some(from_acts) = activations.get(from_uuid)
                && let Some(&from_activation) = from_acts.get(&obs_index)
                && from_activation.is_finite()
                && weight.is_finite()
            {
                reconstructed_value += from_activation * weight;
            }
        }

        if !reconstructed_value.is_finite() {
            continue;
        }

        let reconstructed_activation = apply_squash(&neuron.squash, reconstructed_value);
        let delta = (recorded_activation - reconstructed_activation).abs();
        if delta.is_finite() {
            sum_delta += delta;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum_delta / count as f32
    }
}

/// Compute the mean reconstruction activation delta for every selectable neuron
/// (Issue #1634).
///
/// Loads each neuron's recorded activations from `provider` (including inbound
/// source neurons — inputs/constants — so their activations feed the
/// reconstruction) and returns a map from neuron UUID to mean activation delta.
/// Neurons with no recording are simply absent from the map (their focus score
/// gains no reconstruction bonus). Fails loud only if the provider itself errors
/// — a missing record is treated as "no reconstruction available", not success
/// masking a fault.
pub(super) fn compute_reconstruction_mismatch_map(
    creature: &CreatureJson,
    selectable: &[&NeuronJson],
    provider: &dyn RecordProvider,
) -> Result<HashMap<String, f32>> {
    // Inbound synapses keyed by target neuron UUID.
    let mut inbound_by_neuron: HashMap<&str, Vec<(&str, f32)>> = HashMap::new();
    for syn in &creature.synapses {
        inbound_by_neuron
            .entry(syn.to_uuid.as_str())
            .or_default()
            .push((syn.from_uuid.as_str(), syn.weight));
    }

    // Collect the set of neuron UUIDs whose activations we need: every
    // selectable neuron plus every inbound source feeding one.
    let mut needed: Vec<&str> = Vec::new();
    for neuron in selectable {
        needed.push(neuron.uuid.as_str());
        if let Some(inbound) = inbound_by_neuron.get(neuron.uuid.as_str()) {
            for &(from_uuid, _) in inbound {
                needed.push(from_uuid);
            }
        }
    }
    needed.sort_unstable();
    needed.dedup();

    // Load recorded activation-by-observation for each needed neuron once.
    let mut activations: HashMap<String, ActivationByObs> = HashMap::new();
    for uuid in needed {
        // A missing neuron (no recording) is a legitimate "no reconstruction
        // available" state, not a fault — skip it. Only a provider error
        // propagates.
        if let Some(records) = provider.get(uuid)? {
            let mut by_obs = ActivationByObs::with_capacity(records.len());
            for record in records.iter() {
                if record.activation.is_finite() {
                    by_obs.insert(record.obs_index, record.activation);
                }
            }
            activations.insert(uuid.to_string(), by_obs);
        }
    }

    let empty_inbound: Vec<(&str, f32)> = Vec::new();
    let mut mismatch: HashMap<String, f32> = HashMap::with_capacity(selectable.len());
    for neuron in selectable {
        let Some(self_activation) = activations.get(neuron.uuid.as_str()) else {
            continue;
        };
        let inbound = inbound_by_neuron
            .get(neuron.uuid.as_str())
            .unwrap_or(&empty_inbound);
        let delta =
            mean_activation_delta_for_neuron(neuron, inbound, &activations, self_activation);
        mismatch.insert(neuron.uuid.clone(), delta);
    }

    Ok(mismatch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neuron(uuid: &str, ntype: &str, squash: &str, bias: f32) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: ntype.to_string(),
            squash: squash.to_string(),
            bias,
        }
    }

    fn acts(pairs: &[(u32, f32)]) -> ActivationByObs {
        pairs.iter().copied().collect()
    }

    #[test]
    fn zero_delta_when_reconstruction_matches_recording() {
        // Input fires 0.5 across two observations; hidden = IDENTITY(0 + 0.5×1).
        let hidden = neuron("h", "hidden", "IDENTITY", 0.0);
        let mut activations = HashMap::new();
        activations.insert("in".to_string(), acts(&[(0, 0.5), (1, 0.5)]));
        let self_act = acts(&[(0, 0.5), (1, 0.5)]);
        activations.insert("h".to_string(), self_act.clone());

        let delta =
            mean_activation_delta_for_neuron(&hidden, &[("in", 1.0)], &activations, &self_act);
        assert!(
            delta.abs() < 1e-6,
            "perfect reconstruction must yield ~0 delta, got {delta}"
        );
    }

    #[test]
    fn nonzero_delta_when_bias_shifts_reconstruction() {
        // Same recording, but bias 0.4 makes the reconstruction 0.9 ≠ 0.5.
        let hidden = neuron("h", "hidden", "IDENTITY", 0.4);
        let mut activations = HashMap::new();
        activations.insert("in".to_string(), acts(&[(0, 0.5), (1, 0.5)]));
        let self_act = acts(&[(0, 0.5), (1, 0.5)]);
        activations.insert("h".to_string(), self_act.clone());

        let delta =
            mean_activation_delta_for_neuron(&hidden, &[("in", 1.0)], &activations, &self_act);
        assert!(
            (delta - 0.4).abs() < 1e-6,
            "expected mean delta ~0.4, got {delta}"
        );
    }

    #[test]
    fn empty_recording_yields_zero() {
        let hidden = neuron("h", "hidden", "TANH", 0.0);
        let activations = HashMap::new();
        let self_act = ActivationByObs::new();
        let delta = mean_activation_delta_for_neuron(&hidden, &[], &activations, &self_act);
        assert_eq!(delta, 0.0);
    }
}
