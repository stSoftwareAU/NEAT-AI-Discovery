//! Merge/fold redundant (highly-correlated) hidden neurons (Issue #1633).
//!
//! Discovery had no candidate that **consolidates** redundant hidden neurons.
//! When two hidden neurons carry the same signal (activations correlated at
//! `|r| > 0.999`), one is pure width — it costs forward-pass and analysis budget
//! without adding capacity. The `co_adaptation` detector observed the
//! correlation but produced no merge/fold candidate.
//!
//! These end-to-end tests model the production snapshot-mining finding
//! (Issue #1631): a synthetic creature with a redundant hidden pair plus an
//! independent neuron. They assert the generator emits a coordinated candidate
//! that folds the lower-impact neuron into its twin — redirecting the twin's
//! outgoing weight by the fitted scale and removing the redundant neuron — and
//! that applying that candidate **preserves the output** on the recorded window.

#![allow(clippy::cast_precision_loss)] // synthetic-activation index casts in tests

use neat_ai_discovery::CoordinatedStructuralOpJson;
use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::{
    detect_redundant_neuron_pairs, redundant_pairs_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Build per-observation records for one neuron.
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
                Vec::new(),
            )
        })
        .collect();
    (uuid.to_string(), recs)
}

/// A small mutable creature the coordinated ops are applied to, so we can
/// recompute the output pre-activation from the recorded hidden activations.
#[derive(Clone)]
struct Net {
    biases: std::collections::HashMap<String, f32>,
    /// (from, to) -> weight
    weights: std::collections::HashMap<(String, String), f32>,
    neurons: std::collections::HashSet<String>,
}

impl Net {
    fn from(creature: &CreatureJson) -> Self {
        Net {
            biases: creature
                .neurons
                .iter()
                .map(|n| (n.uuid.clone(), n.bias))
                .collect(),
            weights: creature
                .synapses
                .iter()
                .map(|s| ((s.from_uuid.clone(), s.to_uuid.clone()), s.weight))
                .collect(),
            neurons: creature.neurons.iter().map(|n| n.uuid.clone()).collect(),
        }
    }

    /// Apply one coordinated structural op.
    fn apply(&mut self, op: &CoordinatedStructuralOpJson) {
        match op {
            CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid,
                to_neuron_uuid,
                weight,
            }
            | CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                weight,
            } => {
                self.weights
                    .insert((from_neuron_uuid.clone(), to_neuron_uuid.clone()), *weight);
            }
            CoordinatedStructuralOpJson::SetBias { neuron_uuid, bias } => {
                self.biases.insert(neuron_uuid.clone(), *bias);
            }
            CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } => {
                self.neurons.remove(neuron_uuid);
                self.weights
                    .retain(|(f, t), _| f != neuron_uuid && t != neuron_uuid);
            }
            other => panic!("unexpected op in merge candidate: {other:?}"),
        }
    }

    /// Output pre-activation for observation `i`: `bias_out + Σ w_{h→out}·a_h`,
    /// summed over surviving hidden neurons whose recorded activation is known.
    fn out_preact(
        &self,
        hidden_acts: &std::collections::HashMap<&str, &Vec<f32>>,
        i: usize,
    ) -> f32 {
        let mut sum = *self.biases.get("out").unwrap_or(&0.0);
        for ((from, to), w) in &self.weights {
            if to != "out" || !self.neurons.contains(from) {
                continue;
            }
            if let Some(acts) = hidden_acts.get(from.as_str()) {
                sum += w * acts[i];
            }
        }
        sum
    }
}

#[test]
fn merge_candidate_preserves_output_on_recorded_window() {
    // Synthetic creature modelled on production: dup1 and dup2 are duplicates
    // (identical recorded activations), indep is independent. All feed `out`.
    let creature: CreatureJson = serde_json::from_str(
        r#"{
            "neurons": [
                {"uuid": "dup1", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "dup2", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "indep", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "out", "type": "output", "squash": "IDENTITY", "bias": 0.25}
            ],
            "synapses": [
                {"from_uuid": "dup1", "to_uuid": "out", "weight": 1.0},
                {"from_uuid": "dup2", "to_uuid": "out", "weight": 1.5},
                {"from_uuid": "indep", "to_uuid": "out", "weight": 0.7}
            ],
            "input": 1,
            "output": 1
        }"#,
    )
    .expect("valid creature JSON");

    let n = 30usize;
    let dup: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() + 0.4).collect();
    let indep: Vec<f32> = (0..n).map(|i| (i as f32 * 0.83 + 1.1).cos()).collect();
    let records = vec![
        records_for("dup1", &dup),
        records_for("dup2", &dup),
        records_for("indep", &indep),
    ];

    // Detect exactly one duplicate pair (not involving the independent neuron).
    let pairs = detect_redundant_neuron_pairs(&creature, &records);
    assert_eq!(pairs.len(), 1, "one duplicate pair expected");
    let merged = [pairs[0].remove_uuid.as_str(), pairs[0].keep_uuid.as_str()];
    assert!(merged.contains(&"dup1") && merged.contains(&"dup2"));
    assert!(!merged.contains(&"indep"));

    // The candidate JSON must contain a redirect and a RemoveNeuron op.
    let candidates = redundant_pairs_to_coordinated_candidates(&pairs, &creature);
    assert_eq!(candidates.len(), 1);
    let ops = &candidates[0].operations;
    assert!(
        ops.iter().any(|op| matches!(
            op,
            CoordinatedStructuralOpJson::SetWeight { .. }
                | CoordinatedStructuralOpJson::AddSynapse { .. }
        )),
        "candidate must redirect the twin's outgoing weight"
    );
    assert!(
        ops.iter()
            .any(|op| matches!(op, CoordinatedStructuralOpJson::RemoveNeuron { .. })),
        "candidate must remove the redundant neuron"
    );

    // Serialised JSON carries the expected op tags.
    let json = serde_json::to_string(&candidates[0]).expect("serialise candidate");
    assert!(
        json.contains("removeNeuron"),
        "JSON must contain removeNeuron op"
    );

    // Apply the ops and confirm the output pre-activation is preserved per obs.
    let mut hidden_acts: std::collections::HashMap<&str, &Vec<f32>> =
        std::collections::HashMap::new();
    hidden_acts.insert("dup1", &dup);
    hidden_acts.insert("dup2", &dup);
    hidden_acts.insert("indep", &indep);

    let before = Net::from(&creature);
    let mut after = before.clone();
    for op in ops {
        after.apply(op);
    }
    // The removed neuron must be gone.
    let remove_uuid = &pairs[0].remove_uuid;
    assert!(!after.neurons.contains(remove_uuid));

    for i in 0..n {
        let b = before.out_preact(&hidden_acts, i);
        let a = after.out_preact(&hidden_acts, i);
        assert!(
            (a - b).abs() < 1e-3,
            "output must be preserved at obs {i}: before={b}, after={a}"
        );
    }
}

#[test]
fn scale_shifted_duplicate_is_folded_with_fitted_scale() {
    // a_dup2 = 2 * a_dup1 — a scale-shifted duplicate still folds; the fold uses
    // the fitted scale so the output is preserved.
    let creature: CreatureJson = serde_json::from_str(
        r#"{
            "neurons": [
                {"uuid": "dup1", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "dup2", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "out", "type": "output", "squash": "IDENTITY", "bias": 0.0}
            ],
            "synapses": [
                {"from_uuid": "dup1", "to_uuid": "out", "weight": 0.9},
                {"from_uuid": "dup2", "to_uuid": "out", "weight": 1.3}
            ],
            "input": 1,
            "output": 1
        }"#,
    )
    .expect("valid creature JSON");

    let n = 30usize;
    let base: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).sin() + 0.6).collect();
    let doubled: Vec<f32> = base.iter().map(|v| v * 2.0).collect();
    let records = vec![records_for("dup1", &base), records_for("dup2", &doubled)];

    let pairs = detect_redundant_neuron_pairs(&creature, &records);
    assert_eq!(pairs.len(), 1);
    let candidates = redundant_pairs_to_coordinated_candidates(&pairs, &creature);
    assert_eq!(candidates.len(), 1);

    let mut hidden_acts: std::collections::HashMap<&str, &Vec<f32>> =
        std::collections::HashMap::new();
    hidden_acts.insert("dup1", &base);
    hidden_acts.insert("dup2", &doubled);

    let before = Net::from(&creature);
    let mut after = before.clone();
    for op in &candidates[0].operations {
        after.apply(op);
    }
    for i in 0..n {
        let b = before.out_preact(&hidden_acts, i);
        let a = after.out_preact(&hidden_acts, i);
        assert!(
            (a - b).abs() < 1e-3,
            "scale-shifted fold must preserve output at obs {i}: before={b}, after={a}"
        );
    }
}
