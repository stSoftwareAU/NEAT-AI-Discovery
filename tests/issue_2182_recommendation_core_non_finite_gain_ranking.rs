//! Issue #2182: a non-finite `estimated_improvement` must never rank first in
//! the recommendation-core detectors.
//!
//! Every recorded value in these tests is finite, so the FFI finitude gates of
//! Issues #2132 / #2134 / #2135 accept them. The infinity (or NaN) is
//! manufactured inside the detector by `f32` accumulator overflow, and before
//! the fix the descending `total_cmp` sort put it at rank 0.
//!
//! Documented placement of a non-finite value: it is **dropped** — neither the
//! detector output nor its coordinated conversion carries a non-finite gain,
//! bias or weight, and the honest candidate is at rank 0.

use neat_ai_discovery::analysis::recommendation::fan_in::{
    detect_fan_in_candidates, fan_in_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::recommendation::gradient_discovery::{
    detect_gradient_candidates, gradient_candidates_to_coordinated,
};
use neat_ai_discovery::analysis::recommendation::multi_hop::{
    detect_multi_hop_candidates, multi_hop_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::recommendation::output_bias_drift::{
    detect_output_bias_drift, detect_output_bias_drift_with_descriptor,
    output_bias_drift_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::task_descriptor::TaskDescriptor;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CreatureJson, NeuronJson, SynapseJson,
};

fn rec(uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
    }
}

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

/// Issue #1799: the claim is "reachable from finite records" — a trigger that
/// smuggles a non-finite value in proves nothing.
fn assert_all_records_finite(records: &[(String, Vec<DiscoverRecord>)]) {
    for (_, recs) in records {
        for r in recs {
            assert!(
                r.activation.is_finite(),
                "trigger activation must be finite"
            );
            assert!(
                r.errors.iter().all(|e| e.is_finite()),
                "trigger errors must be finite"
            );
        }
    }
}

/// No coordinated candidate may carry a non-finite gain or parameter.
fn assert_payload_finite(coordinated: &[CoordinatedStructuralCandidateJson]) {
    for c in coordinated {
        assert!(
            c.expected_creature_score_gain.is_finite(),
            "non-finite expected_creature_score_gain {}",
            c.expected_creature_score_gain
        );
        let ops = serde_json::to_value(&c.operations).expect("operations serialise");
        for op in ops.as_array().expect("operations are an array") {
            for key in ["bias", "weight"] {
                if let Some(v) = op.get(key) {
                    let f = v.as_f64().unwrap_or_else(|| {
                        panic!(
                            "{key} is not a finite number in {op} (serde maps non-finite to null)"
                        )
                    });
                    assert!(f.is_finite() && f.abs() <= f64::from(f32::MAX));
                }
            }
        }
    }
}

// =============================================================================
// output_bias_drift.rs
// =============================================================================

fn two_output_creature(poison_bias: f32) -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("output-poison", "output", poison_bias),
            neuron("output-honest", "output", 0.0),
        ],
        synapses: vec![
            synapse("input-1", "output-poison", 0.5),
            synapse("input-1", "output-honest", 0.5),
        ],
        input: 1,
        output: 2,
    }
}

fn honest_bias_records() -> Vec<DiscoverRecord> {
    (0..100)
        .map(|i| {
            let error = if i < 80 { 0.3 } else { -0.1 };
            rec("output-honest", i, 0.5, vec![error])
        })
        .collect()
}

#[test]
fn output_bias_drift_drops_an_overflowed_mean_error_and_ranks_the_honest_candidate_first() {
    let records = vec![
        (
            "output-poison".to_string(),
            (0..30)
                .map(|i| rec("output-poison", i, 0.5, vec![1.0e38]))
                .collect(),
        ),
        ("output-honest".to_string(), honest_bias_records()),
    ];
    assert_all_records_finite(&records);

    let candidates = detect_output_bias_drift(&two_output_creature(0.0), &records);

    assert!(
        !candidates.is_empty(),
        "the honest output must still be detected"
    );
    assert_eq!(candidates[0].neuron_uuid, "output-honest");
    for c in &candidates {
        assert!(c.estimated_improvement.is_finite());
        assert!(c.recommended_bias_delta.is_finite());
        assert!(c.mean_error.is_finite());
    }
    assert!(candidates.iter().all(|c| c.neuron_uuid != "output-poison"));

    let coordinated = output_bias_drift_to_coordinated_candidates(&candidates);
    assert_payload_finite(&coordinated);
}

#[test]
fn output_bias_drift_never_emits_a_set_bias_that_overflows_the_current_bias() {
    // mean error 1.6e37 is finite, as is the gain it yields; only the sum
    // `current_bias + recommended_bias_delta` overflows to -inf.
    let records = vec![
        (
            "output-poison".to_string(),
            (0..20)
                .map(|i| rec("output-poison", i, 0.5, vec![1.6e37]))
                .collect(),
        ),
        ("output-honest".to_string(), honest_bias_records()),
    ];
    assert_all_records_finite(&records);

    let candidates = detect_output_bias_drift(&two_output_creature(-3.4e38), &records);
    assert_eq!(candidates[0].neuron_uuid, "output-honest");
    assert!(
        candidates
            .iter()
            .all(|c| (c.current_bias + c.recommended_bias_delta).is_finite())
    );

    let coordinated = output_bias_drift_to_coordinated_candidates(&candidates);
    assert!(!coordinated.is_empty());
    assert_payload_finite(&coordinated);
}

#[test]
fn output_bias_drift_with_descriptor_drops_a_non_finite_synthesised_bias() {
    // Capacity-starved records whose huge negative activations overflow the
    // mean activation; the synthesised delta `SAT - mean_activation` is +inf.
    let records = vec![
        (
            "output-poison".to_string(),
            (0..20)
                .map(|i| DiscoverRecord {
                    obs_index: i,
                    neuron_uuid: "output-poison".to_string(),
                    value: Some(1.0),
                    activation: -3.0e38,
                    errors: Vec::new(),
                })
                .collect(),
        ),
        ("output-honest".to_string(), honest_bias_records()),
    ];
    assert_all_records_finite(&records);
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 2);

    let candidates =
        detect_output_bias_drift_with_descriptor(&two_output_creature(0.0), &records, &descriptor);

    assert!(!candidates.is_empty());
    for c in &candidates {
        assert!(c.estimated_improvement.is_finite());
        assert!((c.current_bias + c.recommended_bias_delta).is_finite());
    }
    assert!(candidates.iter().all(|c| c.neuron_uuid != "output-poison"));
    assert_payload_finite(&output_bias_drift_to_coordinated_candidates(&candidates));
}

// =============================================================================
// multi_hop.rs
// =============================================================================

#[test]
fn multi_hop_drops_an_overflowed_mean_abs_error_and_ranks_the_honest_candidate_first() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("output-poison", "output", 0.0),
            neuron("output-honest", "output", 0.0),
        ],
        synapses: Vec::new(),
        input: 1,
        output: 2,
    };
    let act = |i: u32| f32::from(u16::try_from(i % 7).expect("fits")) * 0.1;
    let records = vec![
        (
            "input-1".to_string(),
            (0..30).map(|i| rec("input-1", i, act(i), vec![])).collect(),
        ),
        (
            "output-poison".to_string(),
            (0..60)
                .map(|i| {
                    // Shared window: ordinary, correlated errors. Target-only
                    // window: 1e38 errors that overflow the mean-abs sum.
                    let e = if i < 30 { act(i) * 0.5 } else { 1.0e38 };
                    rec("output-poison", i, 0.5, vec![e])
                })
                .collect(),
        ),
        (
            "output-honest".to_string(),
            (0..30)
                .map(|i| rec("output-honest", i, 0.5, vec![act(i) * 0.5]))
                .collect(),
        ),
    ];
    assert_all_records_finite(&records);

    let candidates = detect_multi_hop_candidates(&creature, &records, &None);

    assert!(
        !candidates.is_empty(),
        "the honest target must still be found"
    );
    assert_eq!(
        candidates[0].path.last().map(String::as_str),
        Some("output-honest")
    );
    assert!(
        candidates
            .iter()
            .all(|c| c.estimated_improvement.is_finite())
    );
    assert_payload_finite(&multi_hop_to_coordinated_candidates(&candidates, &creature));
}

#[test]
fn multi_hop_three_hop_drops_a_nan_source_intermediate_correlation() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-mid", "input", 0.0),
            neuron("input-poison", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        synapses: Vec::new(),
        input: 2,
        output: 1,
    };
    let act = |i: u32| f32::from(u16::try_from(i % 7).expect("fits")) * 0.1;
    let swing = |i: u32| if i.is_multiple_of(2) { 2.0e30 } else { -2.0e30 };
    let records = vec![
        (
            // obs 0..30 correlate with the target error; obs 30..60 carry a
            // ±2e30 swing the target never sees.
            "input-mid".to_string(),
            (0..60)
                .map(|i| {
                    rec(
                        "input-mid",
                        i,
                        if i < 30 { act(i) } else { swing(i) },
                        vec![],
                    )
                })
                .collect(),
        ),
        (
            // Shares only obs 30..60 with `input-mid`, so the covariance and
            // variance accumulators both overflow and the correlation is NaN.
            "input-poison".to_string(),
            (30..60)
                .map(|i| rec("input-poison", i, swing(i), vec![]))
                .collect(),
        ),
        (
            "output-1".to_string(),
            (0..30)
                .map(|i| rec("output-1", i, 0.5, vec![act(i) * 0.5]))
                .collect(),
        ),
    ];
    assert_all_records_finite(&records);

    let candidates = detect_multi_hop_candidates(&creature, &records, &None);

    assert!(
        !candidates.is_empty(),
        "the honest two-hop path must survive"
    );
    assert_eq!(
        candidates[0].path,
        vec!["input-mid".to_string(), "output-1".to_string()]
    );
    for c in &candidates {
        assert!(c.estimated_improvement.is_finite());
        assert!(c.correlation_strength.is_finite());
    }
    assert!(
        candidates
            .iter()
            .all(|c| !c.path.contains(&"input-poison".to_string()))
    );
    assert_payload_finite(&multi_hop_to_coordinated_candidates(&candidates, &creature));
}

// =============================================================================
// gradient_discovery.rs
// =============================================================================

#[test]
fn gradient_drops_an_overflowed_improvement_and_ranks_the_honest_candidate_first() {
    // Powers of two keep the f32 mean exact (2^123), so the variance is zero
    // and the consistency gate passes; only `mean × |delta ≈ 1e38|` overflows.
    const POISON_ACTIVATION: f32 = 9_223_372_036_854_775_808.0; // 2^63
    const POISON_ERROR: f32 = 1_152_921_504_606_846_976.0; // 2^60
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-poison", "input", 0.0),
            neuron("input-honest", "input", 0.0),
            neuron("output-poison", "output", 0.0),
            neuron("output-honest", "output", 0.0),
        ],
        synapses: vec![
            synapse("input-poison", "output-poison", 1.0e38),
            synapse("input-honest", "output-honest", 0.5),
        ],
        input: 2,
        output: 2,
    };
    let records = vec![
        (
            "input-poison".to_string(),
            (0..16)
                .map(|i| rec("input-poison", i, POISON_ACTIVATION, vec![]))
                .collect(),
        ),
        (
            "output-poison".to_string(),
            (0..16)
                .map(|i| rec("output-poison", i, 0.5, vec![POISON_ERROR]))
                .collect(),
        ),
        (
            "input-honest".to_string(),
            (0..30)
                .map(|i| rec("input-honest", i, 1.0, vec![]))
                .collect(),
        ),
        (
            "output-honest".to_string(),
            (0..30)
                .map(|i| rec("output-honest", i, 0.5, vec![0.5]))
                .collect(),
        ),
    ];
    assert_all_records_finite(&records);

    let candidates = detect_gradient_candidates(&creature, &records, &None);

    assert!(
        !candidates.is_empty(),
        "the honest synapse must still be found"
    );
    assert_eq!(candidates[0].from_neuron_uuid, "input-honest");
    assert!(
        candidates
            .iter()
            .all(|c| c.estimated_improvement.is_finite())
    );
    assert_payload_finite(&gradient_candidates_to_coordinated(&candidates));
}

// =============================================================================
// fan_in.rs
// =============================================================================

#[test]
fn fan_in_drops_an_overflowed_improvement_and_ranks_the_honest_candidate_first() {
    // Poisoned pair (obs 0..60): the error is exactly proportional to input
    // `a`, so the uncentred Σ error² overflows the f32 SSE while every
    // correlation stays finite and above threshold (see the Issue #2108 sweep).
    const ACTIVATION_BASE: f32 = 3.3e8;
    const ERROR_PER_ACTIVATION: f32 = 3.03e10;
    const ACTIVATION_SPREAD: f32 = 1.0e3;
    let spread_a = |j: u32| f32::from(i16::try_from(j % 5).expect("fits")) - 2.0;
    let spread_b = |j: u32| f32::from(i16::try_from(j % 7).expect("fits")) - 3.0;
    let act_a = |j: u32| ACTIVATION_BASE + ACTIVATION_SPREAD * spread_a(j);
    let act_b = |j: u32| ACTIVATION_BASE + ACTIVATION_SPREAD * (spread_a(j) + spread_b(j));

    // Honest pair (obs 100..160, disjoint from the poisoned window): the error
    // needs both inputs, so the pair beats either input alone.
    let honest = 100..160_u32;

    let creature = CreatureJson {
        neurons: vec![
            neuron("input-a", "input", 0.0),
            neuron("input-b", "input", 0.0),
            neuron("input-c", "input", 0.0),
            neuron("input-d", "input", 0.0),
            neuron("output-poison", "output", 0.0),
            neuron("output-honest", "output", 0.0),
        ],
        synapses: Vec::new(),
        input: 4,
        output: 2,
    };
    let records = vec![
        (
            "input-a".to_string(),
            (0..60)
                .map(|j| rec("input-a", j, act_a(j), vec![]))
                .collect(),
        ),
        (
            "input-b".to_string(),
            (0..60)
                .map(|j| rec("input-b", j, act_b(j), vec![]))
                .collect(),
        ),
        (
            "output-poison".to_string(),
            (0..60)
                .map(|j| {
                    rec(
                        "output-poison",
                        j,
                        0.5,
                        vec![act_a(j) * ERROR_PER_ACTIVATION],
                    )
                })
                .collect(),
        ),
        (
            "input-c".to_string(),
            honest
                .clone()
                .map(|j| rec("input-c", j, spread_a(j), vec![]))
                .collect(),
        ),
        (
            "input-d".to_string(),
            honest
                .clone()
                .map(|j| rec("input-d", j, spread_b(j), vec![]))
                .collect(),
        ),
        (
            "output-honest".to_string(),
            honest
                .map(|j| {
                    rec(
                        "output-honest",
                        j,
                        0.5,
                        vec![0.1 * (spread_a(j) + spread_b(j))],
                    )
                })
                .collect(),
        ),
    ];
    assert_all_records_finite(&records);

    let candidates = detect_fan_in_candidates(&creature, &records, &None);

    assert!(
        !candidates.is_empty(),
        "the honest pair must still be found"
    );
    assert_eq!(candidates[0].target_uuid, "output-honest");
    for c in &candidates {
        assert!(c.estimated_improvement.is_finite());
        assert!(c.input_weights.iter().all(|w| w.is_finite()));
    }
    assert!(candidates.iter().all(|c| c.target_uuid != "output-poison"));
    assert_payload_finite(&fan_in_to_coordinated_candidates(&candidates, &creature));
}
