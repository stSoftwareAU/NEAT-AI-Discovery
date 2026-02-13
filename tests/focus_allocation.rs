//! Issue #523: Targeted tests for the focus `allocation` sub-module.
//!
//! Tests budget allocation strategies with different neuron counts and budget sizes:
//! - Equal allocation across layers
//! - Proportional allocation (larger layers get more)
//! - OutputFirst allocation (deeper layers prioritised)
//! - Edge cases: zero budget, single layer, budget exceeding neuron count

mod common;

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::focus::{
    AllocationStrategy, NeuronInfo, NeuronLayer, hierarchical_focus_selection,
};
use std::collections::{HashMap, HashSet};

/// Build layers directly for allocation-focused tests.
fn make_layers(sizes: &[usize]) -> Vec<NeuronLayer> {
    sizes
        .iter()
        .enumerate()
        .map(|(depth, &count)| NeuronLayer {
            depth: depth + 1,
            neurons: (0..count)
                .map(|i| NeuronInfo {
                    uuid: format!("L{depth}-n{i}"),
                    neuron_type: "hidden".to_string(),
                })
                .collect(),
        })
        .collect()
}

/// Build a minimal creature (allocation tests do not depend on creature topology).
fn dummy_creature() -> CreatureJson {
    CreatureJson {
        input: 1,
        output: 0,
        neurons: vec![],
        synapses: vec![],
    }
}

/// Helper: run selection and return per-layer selected counts.
fn per_layer_counts(
    layers: &[NeuronLayer],
    max_focus: usize,
    strategy: AllocationStrategy,
    scores: &HashMap<String, f32>,
) -> Vec<usize> {
    let creature = dummy_creature();
    let selected = hierarchical_focus_selection(&creature, layers, max_focus, strategy, scores);

    layers
        .iter()
        .map(|layer| {
            layer
                .neurons
                .iter()
                .filter(|n| selected.contains(&n.uuid))
                .count()
        })
        .collect()
}

/// Build uniform scores for all neurons across layers.
fn uniform_scores(layers: &[NeuronLayer], score: f32) -> HashMap<String, f32> {
    layers
        .iter()
        .flat_map(|l| l.neurons.iter())
        .map(|n| (n.uuid.clone(), score))
        .collect()
}

// =============================================================================
// Equal allocation
// =============================================================================

#[test]
fn equal_distributes_budget_evenly() {
    // 3 layers, 10 neurons each, budget = 9 → 3 per layer
    let layers = make_layers(&[10, 10, 10]);
    let scores = uniform_scores(&layers, 1.0);

    let counts = per_layer_counts(&layers, 9, AllocationStrategy::Equal, &scores);

    assert_eq!(counts, vec![3, 3, 3], "Equal should give 3 per layer");
}

#[test]
fn equal_distributes_remainder_to_deeper_layers() {
    // 3 layers, 10 neurons each, budget = 10 → 3+3+4 or 3+3+4 (remainder to last)
    let layers = make_layers(&[10, 10, 10]);
    let scores = uniform_scores(&layers, 1.0);

    let counts = per_layer_counts(&layers, 10, AllocationStrategy::Equal, &scores);

    let total: usize = counts.iter().sum();
    assert_eq!(total, 10, "Total selected should equal budget");

    // Last layer (deepest, index 2) should get the extra slot(s)
    assert!(
        counts[2] >= counts[0],
        "Deeper layer should get remainder: {counts:?}"
    );
}

// =============================================================================
// Proportional allocation
// =============================================================================

#[test]
fn proportional_gives_more_to_larger_layers() {
    // Layer 0: 2 neurons, Layer 1: 8 neurons. Budget = 10
    // Proportional: layer 0 gets 2/10 * 10 = 2, layer 1 gets 8/10 * 10 = 8
    let layers = make_layers(&[2, 8]);
    let scores = uniform_scores(&layers, 1.0);

    let counts = per_layer_counts(&layers, 10, AllocationStrategy::Proportional, &scores);

    assert_eq!(counts[0], 2, "Small layer should get ~2");
    assert_eq!(counts[1], 8, "Large layer should get ~8");
}

#[test]
fn proportional_handles_uneven_division() {
    // 3 layers with sizes 3, 3, 4. Budget = 5.
    // Proportional: 3/10*5=1.5→1, 3/10*5=1.5→1, 4/10*5=2.0→2 => allocated 4, remainder 1
    let layers = make_layers(&[3, 3, 4]);
    let scores = uniform_scores(&layers, 1.0);

    let counts = per_layer_counts(&layers, 5, AllocationStrategy::Proportional, &scores);

    let total: usize = counts.iter().sum();
    assert_eq!(total, 5, "Total should match budget");
}

// =============================================================================
// OutputFirst allocation
// =============================================================================

#[test]
fn output_first_prioritises_deeper_layers() {
    // 3 layers, 5 neurons each, budget = 5
    // OutputFirst allocates from deepest to shallowest
    let layers = make_layers(&[5, 5, 5]);
    let scores = uniform_scores(&layers, 1.0);

    let counts = per_layer_counts(&layers, 5, AllocationStrategy::OutputFirst, &scores);

    let total: usize = counts.iter().sum();
    assert_eq!(total, 5, "Total should match budget");

    // Deepest layer should get the most (or equal) allocation
    assert!(
        counts[2] >= counts[0],
        "Deepest layer ({}) should get >= shallowest ({})",
        counts[2],
        counts[0]
    );
}

#[test]
fn output_first_fills_deep_before_shallow() {
    // 2 layers: shallow has 10, deep has 3. Budget = 4.
    // OutputFirst: deep layer gets up to its size first (3), then shallow gets 1.
    let layers = make_layers(&[10, 3]);
    let scores = uniform_scores(&layers, 1.0);

    let counts = per_layer_counts(&layers, 4, AllocationStrategy::OutputFirst, &scores);

    let total: usize = counts.iter().sum();
    assert_eq!(total, 4, "Total should match budget");

    // Deep layer (index 1) should get all 3 of its neurons
    assert!(
        counts[1] >= 2,
        "Deep layer should get at least 2, got {}",
        counts[1]
    );
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn zero_budget_returns_empty() {
    let layers = make_layers(&[5, 5]);
    let scores = uniform_scores(&layers, 1.0);
    let creature = dummy_creature();

    let selected =
        hierarchical_focus_selection(&creature, &layers, 0, AllocationStrategy::Equal, &scores);

    assert!(selected.is_empty(), "Zero budget should select nothing");
}

#[test]
fn empty_layers_returns_empty() {
    let layers: Vec<NeuronLayer> = vec![];
    let creature = dummy_creature();

    let selected = hierarchical_focus_selection(
        &creature,
        &layers,
        10,
        AllocationStrategy::Equal,
        &HashMap::new(),
    );

    assert!(
        selected.is_empty(),
        "Empty layers should produce empty selection"
    );
}

#[test]
fn budget_exceeding_total_neurons_selects_all() {
    // 2 layers with 3 neurons each (6 total), budget = 20
    let layers = make_layers(&[3, 3]);
    let scores = uniform_scores(&layers, 1.0);
    let creature = dummy_creature();

    let selected =
        hierarchical_focus_selection(&creature, &layers, 20, AllocationStrategy::Equal, &scores);

    // Should select all 6 neurons (can't exceed what exists)
    assert_eq!(
        selected.len(),
        6,
        "Should select all 6 neurons when budget > total"
    );
}

#[test]
fn single_layer_gets_entire_budget() {
    // One layer with 10 neurons, budget = 5
    let layers = make_layers(&[10]);
    let scores = uniform_scores(&layers, 1.0);

    let counts = per_layer_counts(&layers, 5, AllocationStrategy::Equal, &scores);

    assert_eq!(counts[0], 5, "Single layer should get entire budget");
}

// =============================================================================
// Score-based selection within layers
// =============================================================================

#[test]
fn higher_scored_neurons_selected_first_within_layer() {
    // Single layer with 5 neurons, budget = 2
    let layers = vec![NeuronLayer {
        depth: 1,
        neurons: vec![
            NeuronInfo {
                uuid: "low-1".to_string(),
                neuron_type: "hidden".to_string(),
            },
            NeuronInfo {
                uuid: "high-1".to_string(),
                neuron_type: "hidden".to_string(),
            },
            NeuronInfo {
                uuid: "low-2".to_string(),
                neuron_type: "hidden".to_string(),
            },
            NeuronInfo {
                uuid: "high-2".to_string(),
                neuron_type: "hidden".to_string(),
            },
            NeuronInfo {
                uuid: "medium".to_string(),
                neuron_type: "hidden".to_string(),
            },
        ],
    }];

    let mut scores = HashMap::new();
    scores.insert("low-1".to_string(), 1.0_f32);
    scores.insert("high-1".to_string(), 10.0);
    scores.insert("low-2".to_string(), 2.0);
    scores.insert("high-2".to_string(), 9.0);
    scores.insert("medium".to_string(), 5.0);

    let creature = dummy_creature();
    let selected =
        hierarchical_focus_selection(&creature, &layers, 2, AllocationStrategy::Equal, &scores);

    assert!(
        selected.contains(&"high-1".to_string()),
        "Highest-scored neuron should be selected"
    );
    assert!(
        selected.contains(&"high-2".to_string()),
        "Second highest-scored neuron should be selected"
    );
}

#[test]
fn neurons_with_no_score_treated_as_zero() {
    // Layer with scored and unscored neurons
    let layers = vec![NeuronLayer {
        depth: 1,
        neurons: vec![
            NeuronInfo {
                uuid: "scored".to_string(),
                neuron_type: "hidden".to_string(),
            },
            NeuronInfo {
                uuid: "unscored".to_string(),
                neuron_type: "hidden".to_string(),
            },
        ],
    }];

    let mut scores = HashMap::new();
    scores.insert("scored".to_string(), 5.0_f32);
    // "unscored" has no entry → defaults to 0.0

    let creature = dummy_creature();
    let selected =
        hierarchical_focus_selection(&creature, &layers, 1, AllocationStrategy::Equal, &scores);

    assert_eq!(
        selected,
        vec!["scored".to_string()],
        "Scored neuron should be preferred over unscored"
    );
}

// =============================================================================
// All strategies respect max_focus
// =============================================================================

#[test]
fn all_strategies_respect_max_focus() {
    let layers = make_layers(&[20, 20, 20]);
    let scores = uniform_scores(&layers, 1.0);
    let max_focus = 10;

    for strategy in [
        AllocationStrategy::Equal,
        AllocationStrategy::Proportional,
        AllocationStrategy::OutputFirst,
    ] {
        let creature = dummy_creature();
        let selected =
            hierarchical_focus_selection(&creature, &layers, max_focus, strategy, &scores);

        assert!(
            selected.len() <= max_focus,
            "{strategy:?} exceeded max_focus: {} > {max_focus}",
            selected.len()
        );
    }
}

#[test]
fn all_strategies_return_unique_neurons() {
    let layers = make_layers(&[10, 10, 10]);
    let scores = uniform_scores(&layers, 1.0);
    let max_focus = 15;

    for strategy in [
        AllocationStrategy::Equal,
        AllocationStrategy::Proportional,
        AllocationStrategy::OutputFirst,
    ] {
        let creature = dummy_creature();
        let selected =
            hierarchical_focus_selection(&creature, &layers, max_focus, strategy, &scores);

        let unique: HashSet<_> = selected.iter().collect();
        assert_eq!(
            unique.len(),
            selected.len(),
            "{strategy:?} returned duplicate neurons"
        );
    }
}
