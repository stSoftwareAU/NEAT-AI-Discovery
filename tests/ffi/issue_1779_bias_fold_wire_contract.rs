//! End-to-end wire-contract test for Issue #1779: the constant-neuron bias fold
//! must reach the FFI JSON for a **hidden** functionally-constant neuron.
//!
//! Every earlier test for this remedy asserted on the in-memory struct field with
//! fixtures declared `"type": "constant"` — the very class the gate required and
//! no producer emits. This test drives the real pipeline instead:
//! `write_records_to_parquet` → `analyze_parallel_internal` → parsed FFI JSON, and
//! pins the serialised names (`constantNeuronBiasFold`, `foldedTargets`,
//! `biasDelta`) plus the folded value `w × c`, so a serde rename or a regression
//! of the gate fails CI rather than passing silently.

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, analyze_parallel_internal};
use serde_json::{Value, json};

/// Observation count — comfortably above `MIN_DISCOVERY_SAMPLE_COUNT` (20) while
/// keeping the test well inside the unit-test speed budget.
const SAMPLE_COUNT: u32 = 64;

/// The constant activation `c` of the hidden neuron under test. Low enough to be
/// proposed for removal by the low-impact detector, and non-zero so the folded
/// delta is a real value rather than an uninformative zero.
const CONSTANT_ACTIVATION: f32 = 0.02;

/// Outgoing weight `w` from the constant neuron to the single output.
const OUTGOING_WEIGHT: f32 = 3.0;

/// A creature whose hidden neuron `quiet` holds a constant activation on every
/// observation, alongside a variance-carrying hidden neuron `live`.
fn creature_with_constant_hidden_neuron() -> CreatureJson {
    serde_json::from_str(
        r#"{
            "input": 2, "output": 1,
            "neurons": [
                {"uuid": "quiet", "type": "hidden", "squash": "IDENTITY", "bias": 0.02},
                {"uuid": "live", "type": "hidden", "squash": "TANH", "bias": 0.0},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY", "bias": 0.1}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "quiet", "weight": 0.0},
                {"fromUUID": "input-1", "toUUID": "live", "weight": 1.0},
                {"fromUUID": "quiet", "toUUID": "out-0", "weight": 3.0},
                {"fromUUID": "live", "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    )
    .expect("valid creature JSON")
}

/// Per-observation records: `quiet` constant, `live` varying, `out-0` carrying the
/// residual error discovery analyses.
fn records() -> Vec<DiscoverRecord> {
    let mut records = Vec::with_capacity(SAMPLE_COUNT as usize * 3);
    for obs in 0..SAMPLE_COUNT {
        let phase = f32::from(u16::try_from(obs).expect("small observation index")) * 0.21;
        let live = phase.sin().tanh();
        let out = 0.1 + OUTGOING_WEIGHT * CONSTANT_ACTIVATION + live;
        records.push(DiscoverRecord::new(
            obs,
            "quiet".to_string(),
            None,
            CONSTANT_ACTIVATION,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "live".to_string(),
            None,
            live,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "out-0".to_string(),
            None,
            out,
            vec![0.3 * phase.cos()],
        ));
    }
    records
}

/// `true` when the candidate's sole operation removes `uuid`.
fn is_sole_removal_of(candidate: &Value, uuid: &str) -> bool {
    let Some(ops) = candidate.get("operations").and_then(Value::as_array) else {
        return false;
    };
    ops.len() == 1
        && ops[0].get("type").and_then(Value::as_str) == Some("removeNeuron")
        && ops[0].get("neuronUuid").and_then(Value::as_str) == Some(uuid)
}

/// The full pipeline emits the removal of a hidden functionally-constant neuron
/// with its bias fold serialised into the FFI JSON.
#[test]
fn hidden_constant_neuron_removal_carries_bias_fold_in_ffi_json() {
    // Discovery is GPU-only. On machines without GPU, we skip.
    if !GpuAnalyzer::gpu_is_available() {
        return;
    }

    let creature = creature_with_constant_hidden_neuron();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let parquet_path = temp_dir.path().join("issue_1779.parquet");
    let parquet_file = parquet_path
        .to_str()
        .expect("temp path is UTF-8")
        .to_string();
    write_records_to_parquet(&parquet_file, &records()).expect("write parquet");

    let input = json!({
        "parquetFile": parquet_file,
        "creature": serde_json::to_value(&creature).expect("serialise creature"),
        "focusNeurons": ["out-0", "quiet", "live"],
        "randomSeed": 1779,
    })
    .to_string();

    let raw = analyze_parallel_internal(&input).expect("analyze_parallel returned Err");
    let output: Value = serde_json::from_str(&raw).expect("analyze_parallel returned invalid JSON");
    assert_eq!(output["success"], true, "analyze_parallel failed: {output}");

    let candidates = output["coordinatedStructuralCandidates"]
        .as_array()
        .expect("coordinatedStructuralCandidates array present");
    let removal = candidates
        .iter()
        .find(|c| is_sole_removal_of(c, "quiet"))
        .unwrap_or_else(|| {
            panic!(
                "the constant hidden neuron's removal must reach the FFI response, got: {}",
                serde_json::to_string_pretty(candidates).unwrap_or_default()
            )
        });

    // The wire contract: serialised names and the folded value w × c.
    let fold = removal
        .get("constantNeuronBiasFold")
        .unwrap_or_else(|| panic!("removal carries constantNeuronBiasFold: {removal}"));
    let constant = fold
        .get("constantActivation")
        .and_then(Value::as_f64)
        .expect("constantActivation present");
    assert!(
        (constant - f64::from(CONSTANT_ACTIVATION)).abs() < 1e-6,
        "constantActivation must be the measured mean {CONSTANT_ACTIVATION}, got {constant}"
    );

    let targets = fold
        .get("foldedTargets")
        .and_then(Value::as_array)
        .expect("foldedTargets present");
    assert_eq!(targets.len(), 1, "one downstream target: {targets:?}");
    assert_eq!(
        targets[0].get("targetNeuronUuid").and_then(Value::as_str),
        Some("out-0")
    );
    let delta = targets[0]
        .get("biasDelta")
        .and_then(Value::as_f64)
        .expect("biasDelta present");
    let expected = f64::from(OUTGOING_WEIGHT) * f64::from(CONSTANT_ACTIVATION);
    assert!(
        (delta - expected).abs() < 1e-6,
        "biasDelta must be w×c = {expected}, got {delta}"
    );

    // The constant neuron carries no per-sample variance, so it must not also
    // carry the #1559 redistribution remedy.
    assert!(
        removal.get("removeNeuronCompensation").is_none(),
        "a measured-constant removal must not also carry a redistribution remedy: {removal}"
    );
}

/// The variance-carrying hidden neuron must never be handed a bias fold, whatever
/// else the pipeline proposes for it.
#[test]
fn variance_carrying_neuron_never_carries_a_bias_fold_in_ffi_json() {
    // Discovery is GPU-only. On machines without GPU, we skip.
    if !GpuAnalyzer::gpu_is_available() {
        return;
    }

    let creature = creature_with_constant_hidden_neuron();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let parquet_path = temp_dir.path().join("issue_1779_variance.parquet");
    let parquet_file = parquet_path
        .to_str()
        .expect("temp path is UTF-8")
        .to_string();
    write_records_to_parquet(&parquet_file, &records()).expect("write parquet");

    let input = json!({
        "parquetFile": parquet_file,
        "creature": serde_json::to_value(&creature).expect("serialise creature"),
        "focusNeurons": ["out-0", "quiet", "live"],
        "randomSeed": 1779,
    })
    .to_string();

    let raw = analyze_parallel_internal(&input).expect("analyze_parallel returned Err");
    let output: Value = serde_json::from_str(&raw).expect("analyze_parallel returned invalid JSON");
    let candidates = output["coordinatedStructuralCandidates"]
        .as_array()
        .expect("coordinatedStructuralCandidates array present");

    for candidate in candidates.iter().filter(|c| is_sole_removal_of(c, "live")) {
        assert!(
            candidate.get("constantNeuronBiasFold").is_none(),
            "a varying neuron must be rejected by the fold gate: {candidate}"
        );
    }
}
