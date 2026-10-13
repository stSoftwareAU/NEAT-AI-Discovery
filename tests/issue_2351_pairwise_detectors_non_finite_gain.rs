//! Regression tests for Issue #2351 (CWE-682): pairwise / fan-in detectors must
//! not emit candidates with a non-finite `estimated_improvement`.
//!
//! Finite `f32` synapse weights and errors overflow when squared or summed,
//! producing `NaN` or `+inf` intermediates. `NaN` comparisons are always false,
//! so a gate of the form `x < threshold => continue` fails open and the
//! candidate escapes with a non-finite gain.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts in test data (Issue #873)

use neat_ai_discovery::analysis::detection::fanin_polarity_conflict::detect_fanin_polarity_conflicts;
use neat_ai_discovery::analysis::detection::output_conflict::{
    detect_output_conflict_neurons, output_conflicts_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::symmetry_breaking::detect_symmetric_neurons;
use neat_ai_discovery::analysis::detection::weight_coherence::{
    WeightCoherenceConfig, detect_incoherent_weight_ratios,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CoordinatedStructuralOpJson, CreatureJson, NeuronJson, SynapseJson};

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
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

fn creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>, output: usize) -> CreatureJson {
    CreatureJson {
        neurons,
        synapses,
        input: 4,
        output,
    }
}

type NeuronRecords = Vec<(String, Vec<DiscoverRecord>)>;

fn records(uuid: &str, errors: &[f32]) -> NeuronRecords {
    records_with(uuid, |_| errors.to_vec())
}

fn records_with(
    uuid: &str,
    errors_for: impl Fn(u32) -> Vec<f32>,
) -> Vec<(String, Vec<DiscoverRecord>)> {
    let recs = (0..50u32)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: uuid.to_string(),
            value: Some((i as f32 + 1.0) / 10.0),
            activation: (i as f32 + 1.0) / 10.0,
            errors: errors_for(i),
        })
        .collect();
    vec![(uuid.to_string(), recs)]
}

fn records_for(uuids: &[&str]) -> Vec<(String, Vec<DiscoverRecord>)> {
    uuids.iter().flat_map(|u| records(u, &[0.01])).collect()
}

// -----------------------------------------------------------------------------
// symmetry_breaking
// -----------------------------------------------------------------------------

fn symmetric_pair(weight: f32) -> CreatureJson {
    creature(
        vec![
            neuron("in-0", "input"),
            neuron("in-1", "input"),
            neuron("h-a", "hidden"),
            neuron("h-b", "hidden"),
            neuron("out-0", "output"),
        ],
        vec![
            synapse("in-0", "h-a", weight),
            synapse("in-1", "h-a", weight),
            synapse("in-0", "h-b", weight),
            synapse("in-1", "h-b", weight),
            synapse("h-a", "out-0", 1.0),
            synapse("h-b", "out-0", 1.0),
        ],
        1,
    )
}

#[test]
fn overflowing_weights_emit_no_non_finite_candidate() {
    // --- symmetry_breaking: positive control (finite, normal weights) ---
    let recs = records_for(&["h-a", "h-b"]);
    let normal = detect_symmetric_neurons(&symmetric_pair(1.0), &recs);
    assert_eq!(normal.len(), 1, "symmetry control must still emit a pair");
    assert!(normal[0].estimated_improvement.is_finite());

    // x * x overflows to inf, similarity becomes NaN and used to pass the gate.
    for w in [1e20_f32, 3e38_f32] {
        let found = detect_symmetric_neurons(&symmetric_pair(w), &recs);
        for c in &found {
            assert!(
                c.estimated_improvement.is_finite() && c.cosine_similarity.is_finite(),
                "symmetry: weight {w} produced non-finite candidate {c:?}"
            );
        }
        assert!(
            found.is_empty(),
            "symmetry: overflowing weight {w} must be rejected"
        );
    }

    // --- fanin_polarity_conflict ---
    let fanin = |w: f32| {
        let c = creature(
            vec![
                neuron("in-0", "input"),
                neuron("in-1", "input"),
                neuron("in-2", "input"),
                neuron("in-3", "input"),
                neuron("h-0", "hidden"),
                neuron("out-0", "output"),
            ],
            vec![
                synapse("in-0", "h-0", w),
                synapse("in-1", "h-0", w),
                synapse("in-2", "h-0", -w),
                synapse("in-3", "h-0", -w),
                synapse("h-0", "out-0", 1.0),
            ],
            1,
        );
        detect_fanin_polarity_conflicts(&c, &records_for(&["h-0"]))
    };
    let normal = fanin(2.0);
    assert_eq!(
        normal.len(),
        1,
        "fan-in control must still emit a candidate"
    );
    assert!(normal[0].estimated_improvement.is_finite());
    for w in [3e38_f32, 2e38_f32] {
        let found = fanin(w);
        for c in &found {
            assert!(
                c.estimated_improvement.is_finite() && c.conflict_score.is_finite(),
                "fan-in: weight {w} produced non-finite candidate {c:?}"
            );
        }
        assert!(
            found.is_empty(),
            "fan-in: overflowing weight {w} must be rejected"
        );
    }

    // --- weight_coherence::detect_incoherent_weight_ratios ---
    let coherence = |incoming: f32, outgoing: f32| {
        let c = creature(
            vec![
                neuron("in-0", "input"),
                neuron("in-1", "input"),
                neuron("h-0", "hidden"),
                neuron("out-0", "output"),
            ],
            vec![
                synapse("in-0", "h-0", incoming),
                synapse("in-1", "h-0", incoming),
                synapse("h-0", "out-0", outgoing),
            ],
            1,
        );
        detect_incoherent_weight_ratios(
            &c,
            &records_for(&["h-0"]),
            &WeightCoherenceConfig::default(),
            None,
        )
    };
    let normal = coherence(50.0, 0.1);
    assert_eq!(
        normal.len(),
        1,
        "coherence control must still emit a candidate"
    );
    assert!(normal[0].estimated_improvement.is_finite());
    let found = coherence(3e38, 1.0);
    for c in &found {
        assert!(
            c.estimated_improvement.is_finite() && c.incoming_outgoing_ratio.is_finite(),
            "coherence: overflowing incoming sum produced non-finite candidate {c:?}"
        );
    }
    assert!(
        found.is_empty(),
        "coherence: overflowing incoming sum must be rejected"
    );
}

// -----------------------------------------------------------------------------
// output_conflict
// -----------------------------------------------------------------------------

fn output_creature(outputs: usize) -> CreatureJson {
    let mut neurons = vec![neuron("in-0", "input"), neuron("h-0", "hidden")];
    let mut synapses = vec![synapse("in-0", "h-0", 0.5)];
    for i in 0..outputs {
        let id = format!("out-{i}");
        neurons.push(neuron(&id, "output"));
        // No direct h-0 -> out synapse so the converter takes the AddSynapse path.
    }
    synapses.push(synapse("in-0", "out-0", 0.1));
    creature(neurons, synapses, outputs)
}

#[test]
fn overflowing_output_errors_emit_no_non_finite_output_conflict() {
    let c = output_creature(3);

    // Positive control: finite, normal errors still produce a conflict.
    let normal = detect_output_conflict_neurons(&c, &records("h-0", &[-0.5, 0.3, 0.0]));
    assert_eq!(normal.len(), 1, "output-conflict control must still emit");
    assert!(normal[0].estimated_improvement.is_finite());

    // Cases that used to leak: a single +inf error, f32 accumulation overflow
    // (50 x 1e37 > f32::MAX), and finite means whose harmful sum overflows.
    let wide = output_creature(70);
    let mut wide_errs = vec![6e36_f32; 70];
    wide_errs[0] = -0.5;
    let cases: Vec<(&str, &CreatureJson, NeuronRecords)> = vec![
        ("inf error", &c, records("h-0", &[-0.5, f32::INFINITY, 0.0])),
        (
            "accumulation overflow",
            &c,
            records("h-0", &[-0.5, 1e37, 0.0]),
        ),
        // +inf and -inf in one channel average to NaN, which the sign tests ignore.
        (
            "NaN mean",
            &c,
            records_with("h-0", |i| {
                let e = if i % 2 == 0 {
                    f32::INFINITY
                } else {
                    f32::NEG_INFINITY
                };
                vec![-0.5, e, 0.3]
            }),
        ),
        // 69 finite means of 6e36 sum past f32::MAX.
        ("harmful sum overflow", &wide, records("h-0", &wide_errs)),
    ];
    for (label, c, recs) in cases {
        let detected = detect_output_conflict_neurons(c, &recs);
        for d in &detected {
            assert!(
                d.estimated_improvement.is_finite()
                    && d.conflict_severity.is_finite()
                    && d.per_output_mean_error.iter().all(|e| e.is_finite()),
                "output-conflict {label}: non-finite detection {d:?}"
            );
        }
        let converted = output_conflicts_to_coordinated_candidates(&detected, c);
        for cand in &converted {
            assert!(cand.expected_creature_score_gain.is_finite(), "{label}");
            for op in &cand.operations {
                if let CoordinatedStructuralOpJson::AddSynapse { weight, .. }
                | CoordinatedStructuralOpJson::SetWeight { weight, .. } = op
                {
                    assert!(weight.is_finite(), "{label}: non-finite op weight");
                }
            }
        }
        assert!(
            detected.is_empty(),
            "output-conflict {label} must be rejected"
        );
    }
}
