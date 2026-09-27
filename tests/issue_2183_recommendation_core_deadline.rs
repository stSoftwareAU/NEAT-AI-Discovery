//! Issue #2183: the recommendation-core detectors must honour the discovery
//! deadline (and, through `deadline_passed`, global cancellation — #1047).
//!
//! Each detector is given a creature on which a full scan finds candidates.
//! An already-elapsed deadline must stop the scan before the first outer
//! iteration (so no candidate is produced), while a far-future deadline must
//! leave the result identical to an unbounded (`None`) scan.

use std::time::{Duration, SystemTime};

use neat_ai_discovery::analysis::recommendation::fan_in::{
    FanInCandidate, detect_fan_in_candidates,
};
use neat_ai_discovery::analysis::recommendation::gradient_discovery::{
    GradientCandidate, detect_gradient_candidates,
};
use neat_ai_discovery::analysis::recommendation::multi_hop::{
    MultiHopCandidate, detect_multi_hop_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn rec(uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
    }
}

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

fn elapsed_deadline() -> Option<SystemTime> {
    Some(SystemTime::now() - Duration::from_secs(10))
}

fn far_future_deadline() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(3600))
}

/// Order-insensitive fingerprint of a candidate list built from each
/// candidate's structural identity (UUIDs, path, sample count). Floating-point
/// scores are left out: the detectors sum in hash-map iteration order, so the
/// last digit can differ between two otherwise identical scans.
fn fingerprint<T>(candidates: &[T], key: impl Fn(&T) -> String) -> Vec<String> {
    let mut out: Vec<String> = candidates.iter().map(key).collect();
    out.sort();
    out
}

// =============================================================================
// Fixtures — each yields at least one candidate on an unbounded scan.
// =============================================================================

fn multi_hop_fixture() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let creature = CreatureJson {
        neurons: vec![neuron("input-1", "input"), neuron("output-1", "output")],
        synapses: Vec::new(),
        input: 1,
        output: 1,
    };
    let act = |i: u32| f32::from(u16::try_from(i % 7).expect("fits")) * 0.1;
    let records = vec![
        (
            "input-1".to_string(),
            (0..30).map(|i| rec("input-1", i, act(i), vec![])).collect(),
        ),
        (
            "output-1".to_string(),
            (0..30)
                .map(|i| rec("output-1", i, 0.5, vec![act(i) * 0.5]))
                .collect(),
        ),
    ];
    (creature, records)
}

fn gradient_fixture() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let creature = CreatureJson {
        neurons: vec![neuron("input-1", "input"), neuron("output-1", "output")],
        synapses: vec![synapse("input-1", "output-1", 0.5)],
        input: 1,
        output: 1,
    };
    let records = vec![
        (
            "input-1".to_string(),
            (0..30).map(|i| rec("input-1", i, 1.0, vec![])).collect(),
        ),
        (
            "output-1".to_string(),
            (0..30)
                .map(|i| rec("output-1", i, 0.5, vec![0.5]))
                .collect(),
        ),
    ];
    (creature, records)
}

fn fan_in_fixture() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    // The error needs both inputs, so the pair beats either input alone.
    let spread_a = |j: u32| f32::from(i16::try_from(j % 5).expect("fits")) - 2.0;
    let spread_b = |j: u32| f32::from(i16::try_from(j % 7).expect("fits")) - 3.0;
    let window = 100..160_u32;
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-c", "input"),
            neuron("input-d", "input"),
            neuron("output-1", "output"),
        ],
        synapses: Vec::new(),
        input: 2,
        output: 1,
    };
    let records = vec![
        (
            "input-c".to_string(),
            window
                .clone()
                .map(|j| rec("input-c", j, spread_a(j), vec![]))
                .collect(),
        ),
        (
            "input-d".to_string(),
            window
                .clone()
                .map(|j| rec("input-d", j, spread_b(j), vec![]))
                .collect(),
        ),
        (
            "output-1".to_string(),
            window
                .map(|j| rec("output-1", j, 0.5, vec![0.1 * (spread_a(j) + spread_b(j))]))
                .collect(),
        ),
    ];
    (creature, records)
}

// =============================================================================
// multi_hop.rs
// =============================================================================

#[test]
fn multi_hop_elapsed_deadline_returns_no_candidates() {
    let (creature, records) = multi_hop_fixture();
    assert!(
        !detect_multi_hop_candidates(&creature, &records, &None).is_empty(),
        "fixture must yield candidates on an unbounded scan"
    );

    let candidates = detect_multi_hop_candidates(&creature, &records, &elapsed_deadline());

    assert!(
        candidates.is_empty(),
        "an elapsed deadline must stop the scan before the first target, got {candidates:?}"
    );
}

#[test]
fn multi_hop_far_future_deadline_matches_no_deadline() {
    let (creature, records) = multi_hop_fixture();
    let unbounded = detect_multi_hop_candidates(&creature, &records, &None);
    assert!(!unbounded.is_empty(), "fixture must yield candidates");

    let bounded = detect_multi_hop_candidates(&creature, &records, &far_future_deadline());

    let key = |c: &MultiHopCandidate| c.path.join(">");
    assert_eq!(fingerprint(&bounded, key), fingerprint(&unbounded, key));
}

// =============================================================================
// gradient_discovery.rs
// =============================================================================

#[test]
fn gradient_elapsed_deadline_returns_no_candidates() {
    let (creature, records) = gradient_fixture();
    assert!(
        !detect_gradient_candidates(&creature, &records, &None).is_empty(),
        "fixture must yield candidates on an unbounded scan"
    );

    let candidates = detect_gradient_candidates(&creature, &records, &elapsed_deadline());

    assert!(
        candidates.is_empty(),
        "an elapsed deadline must stop the scan before the first synapse, got {candidates:?}"
    );
}

#[test]
fn gradient_far_future_deadline_matches_no_deadline() {
    let (creature, records) = gradient_fixture();
    let unbounded = detect_gradient_candidates(&creature, &records, &None);
    assert!(!unbounded.is_empty(), "fixture must yield candidates");

    let bounded = detect_gradient_candidates(&creature, &records, &far_future_deadline());

    let key = |c: &GradientCandidate| {
        format!(
            "{}>{}/{}",
            c.from_neuron_uuid, c.to_neuron_uuid, c.sample_count
        )
    };
    assert_eq!(fingerprint(&bounded, key), fingerprint(&unbounded, key));
}

// =============================================================================
// fan_in.rs
// =============================================================================

#[test]
fn fan_in_elapsed_deadline_returns_no_candidates() {
    let (creature, records) = fan_in_fixture();
    assert!(
        !detect_fan_in_candidates(&creature, &records, &None).is_empty(),
        "fixture must yield candidates on an unbounded scan"
    );

    let candidates = detect_fan_in_candidates(&creature, &records, &elapsed_deadline());

    assert!(
        candidates.is_empty(),
        "an elapsed deadline must stop the scan before the first target, got {candidates:?}"
    );
}

#[test]
fn fan_in_far_future_deadline_matches_no_deadline() {
    let (creature, records) = fan_in_fixture();
    let unbounded = detect_fan_in_candidates(&creature, &records, &None);
    assert!(!unbounded.is_empty(), "fixture must yield candidates");

    let bounded = detect_fan_in_candidates(&creature, &records, &far_future_deadline());

    let key = |c: &FanInCandidate| {
        let mut inputs = c.input_uuids.clone();
        inputs.sort();
        format!("{}>{}/{}", inputs.join("+"), c.target_uuid, c.sample_count)
    };
    assert_eq!(fingerprint(&bounded, key), fingerprint(&unbounded, key));
}
