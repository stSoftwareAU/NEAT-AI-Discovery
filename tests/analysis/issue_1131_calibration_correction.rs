//! Integration tests for Issue #1131: Calibrate prediction factors using the
//! failure cache.
//!
//! Verifies that when a creature-specific `failure_cache` is supplied to
//! `analyze_all`, the derived per-change-type calibration corrections:
//!
//! 1. Appear in the returned metadata (synapse + neuron paths).
//! 2. Clamp to `MIN_CALIBRATION_CORRECTION` for a 20-entry over-estimation
//!    history of 1000× (expected 0.001, actual 0.000001).
//! 3. Are deterministic across repeated runs with the same input.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::analysis::gpu::GpuAnalyzer;
use neat_ai_discovery::analysis::scoring::calibration_correction::{
    CHANGE_TYPE_ADD_NEURONS, CHANGE_TYPE_ADD_SYNAPSES, CHANGE_TYPE_COORDINATED_STRUCTURAL,
    CalibrationCorrection, FailureCacheEntry, MIN_CALIBRATION_CORRECTION,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};

macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Build a small creature mirroring the shape used by other integration tests.
fn build_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 1,
    }
}

fn build_records(creature: &CreatureJson, count: usize) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for obs in 0..count as u32 {
        let t = obs as f32 / count as f32;
        for neuron in &creature.neurons {
            let (activation, value, errors) = match neuron.neuron_type.as_str() {
                "input" => ((t * std::f32::consts::TAU).sin(), None, Vec::new()),
                "hidden" => {
                    let act = 0.3 + 0.4 * (t * std::f32::consts::TAU).sin();
                    (act, Some(act), vec![0.05 * (t * 3.0).cos()])
                }
                "output" => {
                    let error = 0.1 * (t * std::f32::consts::PI).cos();
                    (0.5 + 0.2 * t, Some(0.5 + 0.2 * t), vec![error])
                }
                _ => continue,
            };
            records.push(DiscoverRecord {
                obs_index: obs,
                neuron_uuid: neuron.uuid.clone(),
                value,
                activation,
                errors,
            });
        }
    }
    records
}

/// Acceptance criterion: 20 entries of expected=0.001, actual=0.000001
/// yield a correction factor at the lower clamp (≈ 0.001).
#[test]
fn twenty_over_estimations_clamp_to_minimum_correction() {
    let cache: Vec<FailureCacheEntry> = (0..20)
        .map(|_| FailureCacheEntry {
            change_type: CHANGE_TYPE_ADD_NEURONS.to_string(),
            expected_error_reduction: 0.001,
            actual_error_reduction: 0.000_001,
            target_squash: None,
            variant_key: None,
            target_uuid: None,
            improved_count: None,
            total_count: None,
            age_epochs: None,
        })
        .collect();
    let correction = CalibrationCorrection::from_failure_cache(&cache);
    let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
    assert!(
        (value - MIN_CALIBRATION_CORRECTION).abs() < 1e-6,
        "expected ≈ {MIN_CALIBRATION_CORRECTION}, got {value}"
    );
}

/// With the same 20-entry failure cache, the `analyze_all` synapse metadata
/// exposes the derived per-change-type correction factors.
#[test]
fn analyze_all_exposes_calibration_corrections_in_metadata() {
    skip_without_gpu!();

    let creature = build_creature();
    let records = build_records(&creature, 60);

    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("test.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    // Seed failure cache: heavy over-estimation across all three change types.
    let mut failure_cache = Vec::new();
    for change_type in [
        CHANGE_TYPE_ADD_NEURONS,
        CHANGE_TYPE_ADD_SYNAPSES,
        CHANGE_TYPE_COORDINATED_STRUCTURAL,
    ] {
        for _ in 0..20 {
            failure_cache.push(FailureCacheEntry {
                change_type: change_type.to_string(),
                expected_error_reduction: 0.001,
                actual_error_reduction: 0.000_001,
                target_squash: None,
                variant_key: None,
                target_uuid: None,
                improved_count: None,
                total_count: None,
                age_epochs: None,
            });
        }
    }

    let focus_neurons: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output" || n.neuron_type == "hidden")
        .map(|n| n.uuid.clone())
        .collect();

    let input = AnalyzeAllInput {
        parquet_file,
        creature,
        focus_neurons,
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: Some(42),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: Some(failure_cache),
        discovery_outcome_log: None,
        cost_name: None,
    };

    let result = analyze_all(&input).expect("analyze_all should succeed");

    let synapse = result.synapse.expect("synapse analysis should run");
    let neuron = result.neuron.expect("neuron analysis should run");

    // Synapse path: both add-synapses and coordinated-structural corrections
    // should be at the floor.
    let synapse_corrections = &synapse.metadata.calibration_corrections;
    let syn_add = synapse_corrections
        .get(CHANGE_TYPE_ADD_SYNAPSES)
        .copied()
        .unwrap_or(1.0);
    let syn_coord = synapse_corrections
        .get(CHANGE_TYPE_COORDINATED_STRUCTURAL)
        .copied()
        .unwrap_or(1.0);
    assert!(
        (syn_add - MIN_CALIBRATION_CORRECTION).abs() < 1e-6,
        "expected add-synapses correction ≈ {MIN_CALIBRATION_CORRECTION}, got {syn_add}"
    );
    assert!(
        (syn_coord - MIN_CALIBRATION_CORRECTION).abs() < 1e-6,
        "expected coordinated-structural correction ≈ {MIN_CALIBRATION_CORRECTION}, got {syn_coord}"
    );

    // Neuron path: add-neurons correction should be at the floor too.
    let neuron_corrections = &neuron.metadata.calibration_corrections;
    let neu_add = neuron_corrections
        .get(CHANGE_TYPE_ADD_NEURONS)
        .copied()
        .unwrap_or(1.0);
    assert!(
        (neu_add - MIN_CALIBRATION_CORRECTION).abs() < 1e-6,
        "expected add-neurons correction ≈ {MIN_CALIBRATION_CORRECTION}, got {neu_add}"
    );
}

/// Two back-to-back `analyze_all` invocations with the same seed and the same
/// failure cache must produce identical calibration corrections.
#[test]
fn calibration_corrections_are_deterministic() {
    skip_without_gpu!();

    let creature = build_creature();
    let records = build_records(&creature, 40);

    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("test.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let failure_cache: Vec<FailureCacheEntry> = (0..15)
        .map(|i| FailureCacheEntry {
            change_type: CHANGE_TYPE_ADD_SYNAPSES.to_string(),
            expected_error_reduction: 0.01,
            // Vary slightly so the EWMA is a non-trivial value.
            actual_error_reduction: 0.002 + (i as f32) * 0.000_05,
            target_squash: None,
            variant_key: None,
            target_uuid: None,
            improved_count: None,
            total_count: None,
            age_epochs: None,
        })
        .collect();

    let focus_neurons: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone())
        .collect();

    let make_input = || AnalyzeAllInput {
        parquet_file: parquet_file.clone(),
        creature: creature.clone(),
        focus_neurons: focus_neurons.clone(),
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(false),
        random_seed: Some(12345),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: Some(failure_cache.clone()),
        discovery_outcome_log: None,
        cost_name: None,
    };

    let r1 = analyze_all(&make_input()).expect("run 1");
    let r2 = analyze_all(&make_input()).expect("run 2");

    let c1 = &r1.synapse.unwrap().metadata.calibration_corrections;
    let c2 = &r2.synapse.unwrap().metadata.calibration_corrections;
    assert_eq!(c1, c2, "corrections must be deterministic across runs");

    // And must match a fresh direct-compute against the same cache.
    let direct = CalibrationCorrection::from_failure_cache(&failure_cache);
    let direct_value = direct.get_correction(CHANGE_TYPE_ADD_SYNAPSES);
    let metadata_value = c1
        .get(CHANGE_TYPE_ADD_SYNAPSES)
        .copied()
        .expect("add-synapses correction should be present");
    assert!(
        (direct_value - metadata_value).abs() < 1e-6,
        "metadata value {metadata_value} should match direct compute {direct_value}"
    );
}

/// With no failure cache supplied, the metadata's `calibration_corrections` map
/// is empty and predictions fall through to the compiled constants unchanged.
#[test]
fn no_failure_cache_leaves_corrections_empty() {
    skip_without_gpu!();

    let creature = build_creature();
    let records = build_records(&creature, 40);

    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("test.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let focus_neurons: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone())
        .collect();

    let input = AnalyzeAllInput {
        parquet_file,
        creature,
        focus_neurons,
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(false),
        random_seed: Some(42),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        cost_name: None,
    };

    let result = analyze_all(&input).expect("analyze_all should succeed");
    let synapse = result.synapse.expect("synapse analysis should run");
    assert!(
        synapse.metadata.calibration_corrections.is_empty(),
        "expected empty corrections map when no failure cache is supplied"
    );
}
