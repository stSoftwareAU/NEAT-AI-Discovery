//! Review finding on PR #2400 (Issue #2349 follow-up): no test failed if the
//! mid-scan deadline checks (the build loop at L255 and the outer loop at
//! L284) were deleted — the only elapsed-deadline test in
//! `tests/issue_2349_symmetry_breaking_growth.rs` passes `UNIX_EPOCH`, which
//! trips the up-front check at L210 on the very first call and never reaches
//! either mid-scan check.
//!
//! These two in-crate tests use the `deadline_override` test seam
//! (`#[cfg(test)]`-only, so they cannot live in the `tests/` integration
//! crate) to script `deadline_passed` to return `false` for the up-front
//! check and the first N mid-scan calls, then `true` — proving each mid-scan
//! check is the one that actually stops a scan that started before the
//! deadline.
//!
//! `#[serial]` keeps these tests from interleaving with any other test that
//! installs a deadline override or flips the global cancellation flag
//! `deadline_passed` also consults.

use super::*;
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use std::time::Duration;

const SAMPLES: u32 = 30;

fn records_for(uuid: &str) -> (String, Vec<DiscoverRecord>) {
    let recs = (0..SAMPLES)
        .map(|idx| DiscoverRecord {
            obs_index: idx,
            neuron_uuid: uuid.to_string(),
            value: Some(0.0),
            activation: 0.9,
            errors: vec![0.0],
        })
        .collect();
    (uuid.to_string(), recs)
}

/// `n` identical hidden neurons (same squash, bias, and incoming weights from
/// every input), so every pair is a symmetric-pair candidate. Mirrors
/// `identical_creature` in `tests/issue_2349_symmetry_breaking_growth.rs`.
fn identical_creature(n: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let inputs = ["input-0", "input-1", "input-2"];
    let weights = [0.5_f32, -0.3, 0.8];

    let mut neurons: Vec<NeuronJson> = inputs
        .iter()
        .map(|u| NeuronJson {
            uuid: u.to_string(),
            neuron_type: "input".to_string(),
            squash: "LOGISTIC".to_string(),
            bias: 0.0,
        })
        .collect();
    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "LOGISTIC".to_string(),
        bias: 0.0,
    });

    let mut synapses = Vec::new();
    let mut records = Vec::with_capacity(n);

    for h in 0..n {
        let uuid = format!("hidden-{h}");
        neurons.push(NeuronJson {
            uuid: uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash: "LOGISTIC".to_string(),
            bias: 0.0,
        });
        for (input, weight) in inputs.iter().zip(weights.iter()) {
            synapses.push(SynapseJson {
                from_uuid: input.to_string(),
                to_uuid: uuid.clone(),
                weight: *weight,
                synapse_type: None,
            });
        }
        records.push(records_for(&uuid));
    }

    let creature = CreatureJson {
        neurons,
        synapses,
        input: inputs.len(),
        output: 1,
    };
    (creature, records)
}

/// A deadline value is required by the signature, but every call this test
/// drives is intercepted by the override queue — the real value is only a
/// fallback that must never be consulted while the queue still has entries.
fn unused_live_deadline() -> Option<std::time::SystemTime> {
    Some(std::time::SystemTime::now() + Duration::from_secs(86_400))
}

/// The build-loop deadline check (L255) must stop the scan after exactly one
/// weight-vector build, before the eligible-neuron cap or the outer-loop
/// check ever run.
#[test]
#[serial]
fn build_loop_deadline_check_stops_a_scan_that_started_before_the_deadline() {
    let (creature, records) = identical_creature(4);

    // [false /*L210 pre-build check*/, false /*build 0 proceeds*/,
    //  true /*build 1 — deadline now reported passed*/]
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, false, true]);

    let mut build_count = 0;
    let scan =
        detect_symmetric_neurons_observed(&creature, &records, &unused_live_deadline(), || {
            build_count += 1;
        });

    assert_eq!(
        scan.truncation,
        Some(ScanTruncation::DeadlinePassed),
        "the build-loop deadline check must report DeadlinePassed"
    );
    assert!(
        scan.candidates.is_empty(),
        "no pair can be compared before the weight vectors finish building"
    );
    assert_eq!(
        build_count, 1,
        "the build-loop deadline check must stop after exactly one weight-vector build"
    );
}

/// The outer-loop deadline check (L284) must stop the scan after the row for
/// `i = 0` is fully compared, before `i = 1` starts — proving it is this
/// check, not the up-front or build-loop checks, that cancels a scan that
/// started before the deadline.
#[test]
#[serial]
fn outer_loop_deadline_check_stops_a_scan_that_started_before_the_deadline() {
    let (creature, records) = identical_creature(4);

    // [false /*L210 pre-build check*/, false, false, false, false /*4 builds*/,
    //  false /*outer i=0 proceeds*/, true /*outer i=1 — deadline now reported passed*/]
    let _guard =
        DeadlineOverrideGuard::with_sequence(vec![false, false, false, false, false, false, true]);

    let scan = detect_symmetric_neurons_with_deadline(&creature, &records, &unused_live_deadline());

    assert_eq!(
        scan.truncation,
        Some(ScanTruncation::DeadlinePassed),
        "the outer-loop deadline check must report DeadlinePassed"
    );
    assert_eq!(
        scan.candidates.len(),
        3,
        "row i=0's 3 pairs (with hidden-1, hidden-2, hidden-3) must be found \
         before the outer-loop deadline check stops the scan at i=1"
    );
}
