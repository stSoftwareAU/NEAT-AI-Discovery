//! Issue #2047 — one overflow-checked `records_per_sample` derivation.
//!
//! The rule *"records per sample is `non_input_neuron_count + creature.input`,
//! and the addition must be `checked_add` because `creature.input` is
//! caller-supplied"* (Issue #1867) was re-typed verbatim at three sites in
//! `src/record/`. `records_per_sample` expresses it once; these tests pin the
//! helper's arithmetic, its overflow behaviour, and the recording pipeline
//! that depends on it.

use neat_ai_discovery::record::records_per_sample;

/// Happy path: the count is the plain sum of the two widths.
#[test]
fn sums_non_input_neurons_and_creature_inputs() {
    let count = records_per_sample(3, 2).expect("3 + 2 must not overflow");
    assert_eq!(count, 5, "records per sample is the sum of the two widths");
}

/// Edge case: both widths zero. The helper derives a size; rejecting a zero
/// count is the caller's job (`validate_and_resolve_indices` does it), so the
/// helper must return `0` rather than erroring.
#[test]
fn zero_widths_derive_zero_records() {
    let count = records_per_sample(0, 0).expect("0 + 0 must not overflow");
    assert_eq!(count, 0, "zero widths derive zero records per sample");
}

/// Edge case: the largest sum that still fits. `usize::MAX + 0` is exactly
/// representable and must not be reported as an overflow.
#[test]
fn maximum_representable_sum_is_accepted() {
    let count = records_per_sample(usize::MAX, 0).expect("usize::MAX + 0 fits");
    assert_eq!(count, usize::MAX, "the boundary value is not an overflow");
}

/// Error path: a caller-supplied `creature.input` near `usize::MAX` must
/// produce a typed error, never a wrapped-around count. Release builds set no
/// `overflow-checks`, so a bare `+` would wrap silently here.
#[test]
fn overflowing_sum_is_reported_not_wrapped() {
    let err = records_per_sample(1, usize::MAX).expect_err("1 + usize::MAX must not wrap");
    assert_eq!(
        err.to_string(),
        "Discovery records per sample would overflow usize",
        "the single shared message is what every call site reports"
    );
}

/// Symmetry: overflow is detected whichever side is oversized.
#[test]
fn overflow_is_detected_from_either_side() {
    records_per_sample(usize::MAX, 1).expect_err("usize::MAX + 1 must not wrap");
    records_per_sample(usize::MAX, usize::MAX).expect_err("MAX + MAX must not wrap");
}

/// The recording pipeline still refuses an overflowing `creature.input` rather
/// than sizing the Parquet writer from a wrapped count — the behaviour the
/// three duplicated copies existed to guarantee (Issue #1867).
#[test]
fn record_discovery_data_still_refuses_an_overflowing_input() {
    let mut input = overflow_input();
    input.creature.input = usize::MAX;

    let err = neat_ai_discovery::record::record_discovery_data(&input)
        .expect_err("usize::MAX input must not wrap");
    assert!(
        err.to_string().contains("overflow"),
        "error must name the overflow: {err}"
    );
}

/// Minimal recording input used by the pipeline regression test.
fn overflow_input() -> neat_ai_discovery::RecordDiscoveryInput {
    neat_ai_discovery::RecordDiscoveryInput {
        creature: neat_ai_discovery::CreatureJson {
            neurons: vec![neat_ai_discovery::NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
            input: 1,
            output: 1,
        },
        training_data: vec![neat_ai_discovery::TrainingRecord {
            input: vec![0.1],
            output: vec![0.5],
            neuron_data: Some(vec![neat_ai_discovery::NeuronData {
                neuron_uuid: "output-0".to_string(),
                activation: 0.5,
                value: Some(0.5),
                errors: vec![0.0],
            }]),
        }],
        temp_dir: std::env::temp_dir()
            .join("neat-ai-discovery-issue-2047-overflow")
            .to_string_lossy()
            .to_string(),
        binary_file_path: None,
        record_indices: None,
        timeout_seconds: None,
        task_descriptor: None,
    }
}
