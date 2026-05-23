//! Parameterised end-to-end tests for the seven built-in NEAT-AI cost
//! functions (Issue #1246).
//!
//! For each cost name in [`BUILT_IN_COST_NAMES`] the test:
//! 1. Builds a small toy creature ([`toy_cost_creature`]).
//! 2. Hand-crafts `TrainingRecord` entries whose output-neuron
//!    `errors` mimic that cost's per-record residual distribution.
//! 3. Calls `record_discovery_internal` to produce a Parquet file.
//! 4. Calls `analyze_parallel_internal` and asserts:
//!    - The call succeeds (`success == true`, no panic, no FFI error).
//!    - Every returned candidate carries a finite
//!      `expectedCreatureScoreGain`.
//!
//! `CATEGORICAL_ERROR` additionally gets a dedicated quantised-`{0, 1}`
//! test to exercise variance-based detectors against zero-variance
//! output errors (Issue #1246, acceptance criterion #3).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::common::{
    BUILT_IN_COST_NAMES, GainFloorDisableGuard, cost_shaped_error, toy_cost_creature,
};
use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::{analyze_parallel_internal, record_discovery_internal};
use serde_json::{Value, json};
use serial_test::serial;
use tempfile::tempdir;

/// Number of training observations per fixture. Kept small so each test
/// finishes well within the 10-second unit-test budget (Issue #603) while
/// still giving discovery enough samples to exercise its variance and
/// distribution detectors.
const FIXTURE_SAMPLE_COUNT: u32 = 64;

/// Build the JSON input expected by `record_discovery_internal`.
///
/// `neuron_errors_for_output` returns the per-output-neuron error vector
/// for a given observation index — this is where the cost-shape is
/// injected.
fn build_record_input(
    temp_dir: &str,
    creature: &Value,
    sample_count: u32,
    mut neuron_errors_for_output: impl FnMut(u32) -> [f32; 1],
) -> String {
    // Deterministic activations so discovery has signal to analyse, but
    // not so structured that detectors lock onto a single pattern.
    let mut training_data: Vec<Value> = Vec::with_capacity(sample_count as usize);
    for obs in 0..sample_count {
        let phase = obs as f32 * 0.21;
        let i0 = phase.sin();
        let i1 = (phase * 1.7).cos();
        // Forward-pass approximations — values must be valid f32 and
        // consistent across the batch. Discovery never reactivates the
        // creature, it just stores these as opaque samples.
        let h0_act = (0.7 * i0).tanh();
        let h1_act = 1.0 / (1.0 + (-(-0.5 * i1)).exp());
        let h2_act = (0.6 * h0_act + 0.4 * h1_act).tanh();
        let o0_act = 0.8 * h2_act;
        let o1_act = -0.3 * h2_act;

        // Output errors carry the cost-shaped residual; hidden neurons
        // record empty errors (mirrors how NEAT-AI populates the field
        // for non-output neurons in the existing test corpus).
        let out_errors = neuron_errors_for_output(obs);
        let o0_err = out_errors[0];
        // Second output gets a phase-shifted residual so we exercise
        // multi-output handling.
        let o1_err = cost_shaped_error("MSE", obs.wrapping_add(7), 1) * 0.5 + out_errors[0] * 0.25;

        training_data.push(json!({
            "input": [i0, i1],
            "output": [o0_act, o1_act],
            "neuron_data": [
                {"neuron_uuid": "h0", "activation": h0_act, "value": h0_act, "errors": []},
                {"neuron_uuid": "h1", "activation": h1_act, "value": h1_act, "errors": []},
                {"neuron_uuid": "h2", "activation": h2_act, "value": h2_act, "errors": []},
                {"neuron_uuid": "o0", "activation": o0_act, "value": o0_act, "errors": [o0_err]},
                {"neuron_uuid": "o1", "activation": o1_act, "value": o1_act, "errors": [o1_err]},
            ]
        }));
    }

    json!({
        "creature": creature,
        "training_data": training_data,
        "temp_dir": temp_dir,
    })
    .to_string()
}

/// Assert that every emitted candidate carries a finite
/// `expectedCreatureScoreGain`. The acceptance criterion is "finite for
/// every returned candidate" regardless of which candidate bucket it
/// lives in, so we sweep every bucket that the FFI exposes.
fn assert_finite_gain_for_all_candidates(cost: &str, output: &Value) {
    const BUCKETS: [&str; 5] = [
        "helpfulSynapses",
        "harmfulSynapses",
        "helpfulNeurons",
        "synapseWeightUpdates",
        "coordinatedStructuralCandidates",
    ];

    for bucket in BUCKETS {
        let Some(candidates) = output.get(bucket).and_then(|v| v.as_array()) else {
            continue;
        };
        for (idx, candidate) in candidates.iter().enumerate() {
            let gain = candidate
                .get("expectedCreatureScoreGain")
                .and_then(Value::as_f64)
                .unwrap_or_else(|| {
                    panic!(
                        "[{cost}] {bucket}[{idx}] missing expectedCreatureScoreGain: {candidate}"
                    )
                });
            assert!(
                gain.is_finite(),
                "[{cost}] {bucket}[{idx}] expectedCreatureScoreGain is not finite: {gain} \
                 (candidate: {candidate})"
            );
        }
    }
}

/// Drive the full `record_discovery` → `analyze_parallel` pipeline
/// against the toy creature using cost-shaped output errors. Returns the
/// parsed `analyze_parallel` output for further per-cost assertions.
fn run_pipeline_for_cost(cost: &str) -> Value {
    let temp_dir = tempdir().expect("tempdir");
    let temp_path = temp_dir
        .path()
        .to_str()
        .expect("temp path must be UTF-8")
        .to_string();

    let creature = toy_cost_creature();
    let creature_json = serde_json::to_value(&creature).expect("serialise creature");

    let record_input =
        build_record_input(&temp_path, &creature_json, FIXTURE_SAMPLE_COUNT, |obs| {
            [cost_shaped_error(cost, obs, 0)]
        });

    let record_result =
        record_discovery_internal(&record_input).expect("record_discovery returned Err");
    let record_output: Value =
        serde_json::from_str(&record_result).expect("record_discovery returned invalid JSON");
    assert_eq!(
        record_output["success"], true,
        "[{cost}] record_discovery failed: {record_output}"
    );

    let parquet_path = std::path::PathBuf::from(
        record_output["tempDir"]
            .as_str()
            .expect("tempDir field missing"),
    )
    .join(record_output["file"].as_str().expect("file field missing"));
    assert!(
        parquet_path.exists(),
        "[{cost}] parquet file not written at {parquet_path:?}"
    );

    let analyze_input = json!({
        "parquetFile": parquet_path.to_str().expect("path UTF-8"),
        "creature": creature_json,
        "focusNeurons": ["o0", "o1"],
        "maxSynapseCandidates": 16,
        "maxNeuronCandidates": 8,
        "randomSeed": 1246,
    })
    .to_string();

    let analyse_result =
        analyze_parallel_internal(&analyze_input).expect("analyze_parallel returned Err");
    let analyse_output: Value =
        serde_json::from_str(&analyse_result).expect("analyze_parallel returned invalid JSON");
    assert_eq!(
        analyse_output["success"], true,
        "[{cost}] analyze_parallel failed: {analyse_output}"
    );

    assert_finite_gain_for_all_candidates(cost, &analyse_output);

    analyse_output
}

/// Helper for the per-cost tests: skip cleanly on machines without a GPU
/// (discovery is GPU-only) and hold the noise-floor guard for the duration
/// of the run because the toy fixtures sit below the production
/// `MIN_EXPECTED_GAIN` threshold (Issue #1191).
fn run_cost_test(cost: &str) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping cost-compatibility test for {cost}: no GPU available");
        return;
    }
    let _gain_guard = GainFloorDisableGuard::new();
    let _output = run_pipeline_for_cost(cost);
}

// ---------------------------------------------------------------------------
// Per-cost tests — one per built-in cost name.
//
// Each test is `#[serial]` because `GainFloorDisableGuard` mutates a shared
// environment variable. The seven tests together complete in well under
// the 10-second-per-test budget (Issue #603) on a CI runner with a GPU.
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn cost_compatibility_mse() {
    run_cost_test("MSE");
}

#[test]
#[serial]
fn cost_compatibility_mae() {
    run_cost_test("MAE");
}

#[test]
#[serial]
fn cost_compatibility_mape() {
    run_cost_test("MAPE");
}

#[test]
#[serial]
fn cost_compatibility_msle() {
    run_cost_test("MSLE");
}

#[test]
#[serial]
fn cost_compatibility_hinge() {
    run_cost_test("HINGE");
}

#[test]
#[serial]
fn cost_compatibility_cross_entropy() {
    run_cost_test("CROSS_ENTROPY");
}

#[test]
#[serial]
fn cost_compatibility_categorical_error() {
    run_cost_test("CATEGORICAL_ERROR");
}

/// Issue #1246 acceptance criterion #6: `CATEGORICAL_ERROR` quantised
/// `{0, 1}` output errors must not cause variance-based detectors to
/// divide by zero, panic, or emit non-finite gains.
///
/// This fixture forces *all* output errors to be the same value (`1.0`)
/// for half the observations and `0.0` for the rest, so the per-sample
/// variance of the output residual is the maximum possible under the
/// `{0, 1}` quantisation. It also forces both outputs to share the
/// quantisation so any per-output variance reducer sees zero-variance
/// columns.
#[test]
#[serial]
fn cost_compatibility_categorical_error_quantised_zero_variance() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping quantised CATEGORICAL_ERROR cost-compatibility test: no GPU available");
        return;
    }
    let _gain_guard = GainFloorDisableGuard::new();

    let temp_dir = tempdir().expect("tempdir");
    let temp_path = temp_dir
        .path()
        .to_str()
        .expect("temp path must be UTF-8")
        .to_string();

    let creature = toy_cost_creature();
    let creature_json = serde_json::to_value(&creature).expect("serialise creature");

    // Build a fixture where every output neuron in every observation
    // sees either {0, 0} or {1, 1}. This is the worst case for
    // variance-based detectors because per-output columns collapse to
    // two constant clusters with zero within-cluster variance.
    let record_input = serde_json::to_string(&json!({
        "creature": creature_json,
        "training_data": (0..FIXTURE_SAMPLE_COUNT)
            .map(|obs| {
                let flag: f32 = if obs % 2 == 0 { 0.0 } else { 1.0 };
                let phase = obs as f32 * 0.21;
                let i0 = phase.sin();
                let i1 = (phase * 1.7).cos();
                let h0_act = (0.7 * i0).tanh();
                let h1_act = 1.0 / (1.0 + (-(-0.5 * i1)).exp());
                let h2_act = (0.6 * h0_act + 0.4 * h1_act).tanh();
                let o0_act = 0.8 * h2_act;
                let o1_act = -0.3 * h2_act;
                json!({
                    "input": [i0, i1],
                    "output": [o0_act, o1_act],
                    "neuron_data": [
                        {"neuron_uuid": "h0", "activation": h0_act, "value": h0_act, "errors": []},
                        {"neuron_uuid": "h1", "activation": h1_act, "value": h1_act, "errors": []},
                        {"neuron_uuid": "h2", "activation": h2_act, "value": h2_act, "errors": []},
                        {"neuron_uuid": "o0", "activation": o0_act, "value": o0_act, "errors": [flag]},
                        {"neuron_uuid": "o1", "activation": o1_act, "value": o1_act, "errors": [flag]},
                    ]
                })
            })
            .collect::<Vec<_>>(),
        "temp_dir": temp_path,
    }))
    .expect("serialise record input");

    let record_result = record_discovery_internal(&record_input)
        .expect("quantised CATEGORICAL_ERROR record_discovery returned Err");
    let record_output: Value = serde_json::from_str(&record_result)
        .expect("quantised CATEGORICAL_ERROR record_discovery returned invalid JSON");
    assert_eq!(
        record_output["success"], true,
        "quantised CATEGORICAL_ERROR record_discovery failed: {record_output}"
    );

    let parquet_path = std::path::PathBuf::from(
        record_output["tempDir"]
            .as_str()
            .expect("tempDir field missing"),
    )
    .join(record_output["file"].as_str().expect("file field missing"));
    assert!(
        parquet_path.exists(),
        "quantised CATEGORICAL_ERROR parquet not written at {parquet_path:?}"
    );

    let analyze_input = json!({
        "parquetFile": parquet_path.to_str().expect("path UTF-8"),
        "creature": creature_json,
        "focusNeurons": ["o0", "o1"],
        "maxSynapseCandidates": 16,
        "maxNeuronCandidates": 8,
        "randomSeed": 1246,
    })
    .to_string();

    let analyse_result = analyze_parallel_internal(&analyze_input)
        .expect("quantised CATEGORICAL_ERROR analyze_parallel returned Err");
    let analyse_output: Value = serde_json::from_str(&analyse_result)
        .expect("quantised CATEGORICAL_ERROR analyze_parallel returned invalid JSON");
    assert_eq!(
        analyse_output["success"], true,
        "quantised CATEGORICAL_ERROR analyze_parallel failed: {analyse_output}"
    );

    assert_finite_gain_for_all_candidates("CATEGORICAL_ERROR (quantised)", &analyse_output);
}

/// Belt-and-braces meta-test: the explicit per-cost tests above must
/// cover every entry in [`BUILT_IN_COST_NAMES`]. Adding a new built-in
/// cost to the list without a matching test is an oversight — this
/// guard test fails fast so the new cost gets its own fixture.
#[test]
fn every_built_in_cost_has_a_dedicated_test() {
    // Names of the per-cost tests defined in this file. Keep in sync
    // with the `#[test]` functions above.
    const COVERED: [&str; 7] = [
        "MSE",
        "MAE",
        "MAPE",
        "MSLE",
        "HINGE",
        "CROSS_ENTROPY",
        "CATEGORICAL_ERROR",
    ];
    for cost in BUILT_IN_COST_NAMES {
        assert!(
            COVERED.contains(&cost),
            "Cost '{cost}' is in BUILT_IN_COST_NAMES but has no dedicated \
             cost-compatibility test in tests/cost_compatibility/end_to_end.rs"
        );
    }
}
