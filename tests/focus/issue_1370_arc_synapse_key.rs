//! Regression tests for Issue #1370: sharing the per-record synapse key via
//! `Arc` in the MIN/MAX selection-stats path must not change the computed
//! impact attribution.
//!
//! The change replaced a per-record `(String, String)` clone in
//! `compute_min_stats` / `compute_max_stats` with an `Arc`-shared key. The key
//! is only hashed/compared (never mutated) downstream, so the win-probability
//! outputs must be byte-for-byte identical. These tests assert on the observable
//! outcome (win-proportional impact), not on how the key is stored.

use neat_ai_discovery::focus::RecordProvider;
use neat_ai_discovery::focus::compute_impacts_with_activations;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::collections::HashMap;
use std::sync::Arc;

/// Minimal in-memory record provider keyed by neuron UUID.
struct InMemoryProvider {
    records: HashMap<String, Arc<Vec<DiscoverRecord>>>,
}

impl RecordProvider for InMemoryProvider {
    fn get(&self, neuron_uuid: &str) -> anyhow::Result<Option<Arc<Vec<DiscoverRecord>>>> {
        Ok(self.records.get(neuron_uuid).cloned())
    }
    fn len(&self) -> usize {
        self.records.len()
    }
}

fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
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

fn records_for(uuid: &str, activations: &[f32]) -> Arc<Vec<DiscoverRecord>> {
    Arc::new(
        activations
            .iter()
            .enumerate()
            .map(|(obs, &a)| {
                let obs = u32::try_from(obs).expect("obs index fits in u32");
                DiscoverRecord::new(obs, uuid.to_string(), Some(a), a, Vec::new())
            })
            .collect(),
    )
}

/// A MINIMUM neuron where `a` wins 3/4 observations and `b` wins 1/4 must give
/// `a` ~3× the impact of `b`. Exercises the per-record key push that now shares
/// an `Arc`.
#[test]
fn minimum_win_proportional_impact_is_unchanged() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("a", "hidden", "IDENTITY"),
            neuron("b", "hidden", "IDENTITY"),
            neuron("min-out", "output", "MINIMUM"),
        ],
        synapses: vec![synapse("a", "min-out", 1.0), synapse("b", "min-out", 1.0)],
        input: 0,
        output: 1,
    };

    // obs0..2: a smaller (a wins). obs3: b smaller (b wins).
    let mut records = HashMap::new();
    records.insert("a".to_string(), records_for("a", &[0.1, 0.1, 0.1, 0.9]));
    records.insert("b".to_string(), records_for("b", &[0.9, 0.9, 0.9, 0.05]));
    let provider = InMemoryProvider { records };

    let impacts = compute_impacts_with_activations(&creature, &provider).unwrap();
    let a = impacts["a"];
    let b = impacts["b"];

    // a wins 3/4 → impact 0.75; b wins 1/4 → impact 0.25.
    assert!((a - 0.75).abs() < 1e-6, "a impact {a} expected ~0.75");
    assert!((b - 0.25).abs() < 1e-6, "b impact {b} expected ~0.25");
}

/// A MAXIMUM neuron where every observation is a tie must split the impact
/// equally. Ties drive the multi-winner branch of the win-counting loop, where
/// the shared key is inserted into `win_counts`.
#[test]
fn maximum_ties_split_impact_equally() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("a", "hidden", "IDENTITY"),
            neuron("b", "hidden", "IDENTITY"),
            neuron("max-out", "output", "MAXIMUM"),
        ],
        synapses: vec![synapse("a", "max-out", 1.0), synapse("b", "max-out", 1.0)],
        input: 0,
        output: 1,
    };

    // Identical activations every observation → both win every time (tie).
    let mut records = HashMap::new();
    records.insert("a".to_string(), records_for("a", &[0.5, 0.5, 0.5]));
    records.insert("b".to_string(), records_for("b", &[0.5, 0.5, 0.5]));
    let provider = InMemoryProvider { records };

    let impacts = compute_impacts_with_activations(&creature, &provider).unwrap();
    let a = impacts["a"];
    let b = impacts["b"];

    // Both win 3/3 observations → both get full win probability 1.0.
    assert!((a - 1.0).abs() < 1e-6, "a impact {a} expected ~1.0");
    assert!((b - 1.0).abs() < 1e-6, "b impact {b} expected ~1.0");
}
