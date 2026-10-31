//! Issue #2381 (CWE-697): `detect_output_squash_mismatches` fails open on a
//! non-finite mean error.
//!
//! A NaN recorded error makes the mean NaN. NaN loses the `< threshold` gate,
//! so the neuron was kept and emitted a NaN `estimated_improvement`. A finite
//! but very large error set overflows the f32 error sum to `+inf`, which gave
//! an infinite gain. Each such neuron must be skipped, not emitted.

use neat_ai_discovery::analysis::detection::output_squash_mismatch::{
    OutputSquashMismatchCandidate, detect_output_squash_mismatches,
};
use neat_ai_discovery::types::DiscoverRecord;

const SAMPLES: u32 = 40;

/// Builds `SAMPLES` records whose activations alternate +1.0 / -1.0, so every
/// activation sits on a `HARD_TANH` bound and the clipping strategy applies.
fn records(uuid: &str, error_at: impl Fn(u32) -> f32) -> Vec<DiscoverRecord> {
    (0..SAMPLES)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: if i % 2 == 0 { 1.0 } else { -1.0 },
            errors: vec![error_at(i)],
        })
        .collect()
}

fn output(uuid: &str) -> (String, String, f32) {
    (uuid.to_string(), "HARD_TANH".to_string(), 0.0)
}

fn assert_finite(c: &OutputSquashMismatchCandidate, label: &str) {
    assert!(
        c.estimated_improvement.is_finite(),
        "non-finite estimated_improvement for {label}: {c:?}"
    );
    assert!(
        c.mean_error.is_finite(),
        "non-finite mean_error for {label}: {c:?}"
    );
}

#[test]
fn non_finite_error_emits_no_non_finite_candidate() {
    let neurons = vec![output("out-0")];

    // Positive control: ordinary finite errors still produce a candidate.
    let data = vec![("out-0".to_string(), records("out-0", |_| 0.5))];
    let ok = detect_output_squash_mismatches(&neurons, &data);
    assert!(!ok.is_empty(), "finite input should emit a candidate");
    for c in &ok {
        assert_finite(c, "finite control");
    }

    // Case A: one NaN error makes the mean error NaN.
    let data = vec![(
        "out-0".to_string(),
        records("out-0", |i| if i == 0 { f32::NAN } else { 0.5 }),
    )];
    let got = detect_output_squash_mismatches(&neurons, &data);
    for c in &got {
        assert_finite(c, "NaN error");
    }
    assert!(got.is_empty(), "NaN mean error should skip the neuron");

    // Case B: each error is finite, but the 40-sample sum overflows f32 to +inf.
    let data = vec![("out-0".to_string(), records("out-0", |_| 3.0e38))];
    assert!(
        data[0].1.iter().all(|r| r.errors[0].is_finite()),
        "precondition: every error must be finite"
    );
    let got = detect_output_squash_mismatches(&neurons, &data);
    for c in &got {
        assert_finite(c, "overflowing error sum");
    }
    assert!(
        got.is_empty(),
        "overflowing mean error should skip the neuron"
    );
}

#[test]
fn finite_candidates_rank_by_descending_improvement() {
    // Lower-error neuron is listed first in the input so the sort is observable.
    let neurons = vec![output("out-lo"), output("out-nan"), output("out-hi")];
    let data = vec![
        ("out-lo".to_string(), records("out-lo", |_| 0.2)),
        (
            "out-nan".to_string(),
            records("out-nan", |i| if i == 0 { f32::NAN } else { 0.5 }),
        ),
        ("out-hi".to_string(), records("out-hi", |_| 0.5)),
    ];

    let result = detect_output_squash_mismatches(&neurons, &data);

    let uuids: Vec<&str> = result.iter().map(|c| c.neuron_uuid.as_str()).collect();
    assert_eq!(
        uuids,
        ["out-hi", "out-lo"],
        "only the two finite neurons should be emitted, highest gain first"
    );
    for c in &result {
        assert_finite(c, "ranking");
    }
    for pair in result.windows(2) {
        assert!(
            pair[0].estimated_improvement >= pair[1].estimated_improvement,
            "candidates not in descending order: {pair:?}"
        );
    }
}

#[test]
fn neg_infinity_and_infinity_errors_are_skipped() {
    let neurons = vec![output("out-0")];

    for bad in [f32::INFINITY, f32::NEG_INFINITY] {
        let data = vec![(
            "out-0".to_string(),
            records("out-0", |i| if i == 0 { bad } else { 0.5 }),
        )];
        let got = detect_output_squash_mismatches(&neurons, &data);
        for c in &got {
            assert_finite(c, "infinite error");
        }
        assert!(
            got.is_empty(),
            "infinite error {bad} should skip the neuron"
        );
    }
}
