//! Issue #2351 (CWE-682) sibling sites, batch A.
//!
//! The Parquet record loader does not filter non-finite activations, values or
//! errors, and f32 sums of squares overflow to `+inf` from finite inputs. The
//! resulting NaN or infinite intermediates pass `<= 0.0` / `< threshold` gates
//! (NaN comparisons are false), so candidates with a non-finite
//! `estimated_improvement` were emitted. Each detector below must fail closed.

use neat_ai_discovery::analysis::detection::activation_mismatch::detect_activation_mismatches;
use neat_ai_discovery::analysis::detection::bias_perturbation::detect_bias_perturbation_candidates;
use neat_ai_discovery::analysis::detection::bottleneck::detect_bottleneck_neurons;
use neat_ai_discovery::analysis::detection::compound_degradation::detect_compound_bias_weight_degradations;
use neat_ai_discovery::analysis::detection::low_impact_neuron::detect_low_impact_neurons;
use neat_ai_discovery::analysis::detection::topology::detect_topology_issues;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

const SAMPLES: u32 = 50;

fn neuron(uuid: &str, kind: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: kind.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

fn syn(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

fn records(
    uuid: &str,
    activation: impl Fn(u32) -> f32,
    value: Option<f32>,
    error: f32,
) -> Vec<DiscoverRecord> {
    (0..SAMPLES)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: uuid.to_string(),
            value,
            activation: activation(i),
            errors: vec![error],
        })
        .collect()
}

type NeuronRecords = Vec<(String, Vec<DiscoverRecord>)>;

// ---------------------------------------------------------------------------
// compound_degradation.rs
// ---------------------------------------------------------------------------

fn compound_inputs(error: f32) -> (CreatureJson, NeuronRecords) {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("h0", "hidden", "IDENTITY"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        synapses: vec![syn("input-0", "h0", 0.5), syn("h0", "output-0", 0.5)],
        input: 1,
        output: 1,
    };
    let recs = vec![
        ("h0".to_string(), records("h0", |_| 1.0, Some(0.0), error)),
        (
            "output-0".to_string(),
            records("output-0", |_| 1.0, Some(0.0), error),
        ),
    ];
    (creature, recs)
}

#[test]
fn overflowing_errors_emit_no_non_finite_compound_correction() {
    // Positive control: ordinary finite errors still produce a candidate.
    let (creature, recs) = compound_inputs(0.5);
    let ok = detect_compound_bias_weight_degradations(&creature, &recs);
    assert!(!ok.is_empty(), "finite input should emit a candidate");

    // Squared 1e30 errors overflow f32 to +inf.
    let (creature, recs) = compound_inputs(1.0e30);
    for c in detect_compound_bias_weight_degradations(&creature, &recs) {
        assert!(
            c.estimated_improvement.is_finite(),
            "non-finite compound gain: {c:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// topology.rs
// ---------------------------------------------------------------------------

#[test]
fn non_finite_errors_emit_no_non_finite_long_path_gain() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("h0", "hidden", "TANH"),
            neuron("h1", "hidden", "TANH"),
            neuron("h2", "hidden", "TANH"),
            neuron("h3", "hidden", "TANH"),
            neuron("output-0", "output", "TANH"),
        ],
        synapses: vec![
            syn("input-0", "h0", 0.5),
            syn("h0", "h1", 0.5),
            syn("h1", "h2", 0.5),
            syn("h2", "h3", 0.5),
            syn("h3", "output-0", 0.5),
        ],
        input: 1,
        output: 1,
    };
    let build = |h0_error: f32| -> NeuronRecords {
        ["h0", "h1", "h2", "h3"]
            .iter()
            .map(|u| {
                let e = if *u == "h0" { h0_error } else { 0.2 };
                (u.to_string(), records(u, |i| i as f32 * 0.01, None, e))
            })
            .collect()
    };

    let ok = detect_topology_issues(&creature, &build(0.3), None);
    assert!(
        ok.iter().any(|c| c.issue_type == "long_path"),
        "finite input should emit a long_path candidate"
    );

    for bad in [f32::NAN, f32::INFINITY] {
        for c in detect_topology_issues(&creature, &build(bad), None) {
            assert!(
                c.estimated_improvement.is_finite(),
                "non-finite topology gain for error {bad}: {c:?}"
            );
        }
    }
}

#[test]
fn non_finite_errors_emit_no_non_finite_imbalance_gain() {
    // h0 has fan-in 1 (starved), h1 has fan-in 3.
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("input-3", "input", "IDENTITY"),
            neuron("h0", "hidden", "TANH"),
            neuron("h1", "hidden", "TANH"),
            neuron("output-0", "output", "TANH"),
        ],
        synapses: vec![
            syn("input-0", "h0", 0.5),
            syn("h0", "output-0", 0.5),
            syn("input-1", "h1", 0.4),
            syn("input-2", "h1", 0.3),
            syn("input-3", "h1", 0.6),
            syn("h1", "output-0", 0.7),
        ],
        input: 4,
        output: 1,
    };
    let build = |h0_error: f32| -> NeuronRecords {
        vec![
            (
                "h0".to_string(),
                records("h0", |i| i as f32 * 0.01, None, h0_error),
            ),
            (
                "h1".to_string(),
                records("h1", |i| i as f32 * 0.01, None, 0.25),
            ),
        ]
    };

    let ok = detect_topology_issues(&creature, &build(0.1), None);
    assert!(
        ok.iter().any(|c| c.issue_type == "connectivity_imbalance"),
        "finite input should emit an imbalance candidate"
    );

    for bad in [f32::NAN, f32::INFINITY] {
        for c in detect_topology_issues(&creature, &build(bad), None) {
            assert!(
                c.estimated_improvement.is_finite(),
                "non-finite imbalance gain for error {bad}: {c:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// bottleneck.rs
// ---------------------------------------------------------------------------

#[test]
fn overflowing_errors_emit_no_non_finite_bottleneck_gain() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("input-3", "input", "IDENTITY"),
            neuron("input-4", "input", "IDENTITY"),
            neuron("hidden-b", "hidden", "TANH"),
            neuron("output-0", "output", "TANH"),
            neuron("output-1", "output", "TANH"),
        ],
        synapses: vec![
            syn("input-0", "hidden-b", 0.5),
            syn("input-1", "hidden-b", 0.3),
            syn("input-2", "hidden-b", -0.4),
            syn("input-3", "hidden-b", 0.6),
            syn("input-4", "hidden-b", -0.2),
            syn("hidden-b", "output-0", 0.8),
            syn("hidden-b", "output-1", -0.5),
        ],
        input: 5,
        output: 2,
    };
    let build = |error: f32| -> NeuronRecords {
        vec![(
            "hidden-b".to_string(),
            records("hidden-b", |i| i as f32 * 0.01, None, error),
        )]
    };

    let ok = detect_bottleneck_neurons(&creature, &build(0.3), None);
    assert!(!ok.is_empty(), "finite input should emit a candidate");

    // Summed absolute errors overflow f32 (50 * 3e38).
    for c in detect_bottleneck_neurons(&creature, &build(3.0e38), None) {
        assert!(
            c.estimated_improvement.is_finite() && c.error_contribution_ratio.is_finite(),
            "non-finite bottleneck payload: {c:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// low_impact_neuron.rs
// ---------------------------------------------------------------------------

#[test]
fn nan_activations_emit_no_non_finite_low_impact_candidate() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("h0", "hidden", "TANH"),
            neuron("output-0", "output", "TANH"),
        ],
        synapses: vec![syn("input-0", "h0", 0.5), syn("h0", "output-0", 0.5)],
        input: 1,
        output: 1,
    };
    let build = |nan: bool| -> NeuronRecords {
        vec![(
            "h0".to_string(),
            records(
                "h0",
                |i| if nan && i == 0 { f32::NAN } else { 0.001 },
                None,
                0.1,
            ),
        )]
    };

    let ok = detect_low_impact_neurons(&creature, &build(false), None);
    assert!(!ok.is_empty(), "finite input should emit a candidate");

    for c in detect_low_impact_neurons(&creature, &build(true), None) {
        assert!(
            c.mean_abs_activation.is_finite()
                && c.activation_std_dev.is_finite()
                && c.removal_confidence.is_finite()
                && c.estimated_improvement.is_finite(),
            "non-finite low-impact payload: {c:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// activation_mismatch.rs
// ---------------------------------------------------------------------------

#[test]
fn non_finite_activations_emit_no_non_finite_squash_gain() {
    let neurons = vec![("h0".to_string(), "TANH".to_string(), 0.0f32)];
    let build =
        |a: f32| -> NeuronRecords { vec![("h0".to_string(), records("h0", |_| a, None, 0.1))] };

    // Positive control: a narrow, finite activation band is underutilised.
    let ok = detect_activation_mismatches(&neurons, &build(0.01));
    assert!(!ok.is_empty(), "finite input should emit a candidate");

    // All-NaN leaves the min/max folds at +/-inf (utilisation -inf), all +inf
    // gives a NaN range.
    for bad in [f32::NAN, f32::INFINITY] {
        for c in detect_activation_mismatches(&neurons, &build(bad)) {
            assert!(
                c.estimated_improvement.is_finite() && c.clipped_fraction.is_finite(),
                "non-finite mismatch payload for activation {bad}: {c:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// bias_perturbation.rs
// ---------------------------------------------------------------------------

#[test]
fn non_finite_errors_emit_no_non_finite_bias_perturbation_gain() {
    let neurons = vec![("h0".to_string(), "TANH".to_string(), 0.0f32)];
    let build = |e: f32| -> NeuronRecords {
        vec![("h0".to_string(), records("h0", |_| 0.0, Some(5.0), e))]
    };

    let ok = detect_bias_perturbation_candidates(&neurons, &build(0.5));
    assert!(!ok.is_empty(), "finite input should emit a candidate");

    for bad in [f32::NAN, f32::INFINITY] {
        for c in detect_bias_perturbation_candidates(&neurons, &build(bad)) {
            assert!(
                c.estimated_improvement.is_finite() && c.mean_error.is_finite(),
                "non-finite bias perturbation payload for error {bad}: {c:?}"
            );
        }
    }
}
