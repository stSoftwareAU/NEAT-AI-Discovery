//! Pass-level wiring guard for the target-cooldown epoch (Issue #1790).
//!
//! `TargetFailureTracker::advance_epoch` had no production caller, so the
//! global tracker's epoch was frozen at `0` forever: `current_epoch <
//! failure_epoch + cooldown_epochs` was permanently true and any target that
//! entered cooldown stayed suppressed for the life of the process.
//!
//! These tests drive `analyze_all` — the single discovery-pass entry point —
//! and assert the global epoch moves from `E` to `E + 1` per pass: not `0`
//! (caller deleted or relocated by a refactor) and not `2` (advanced inside
//! each of the neuron and synapse preparation layers instead of once at the
//! head of the pass).
//!
//! The tests share process-global state, so they are `#[serial]`.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::analysis::gpu::GpuAnalyzer;
use neat_ai_discovery::analysis::target_failure_tracker::global_tracker;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;

/// Read the process-global tracker's epoch, recovering a poisoned lock the
/// same way production does.
fn current_global_epoch() -> u64 {
    match global_tracker().lock() {
        Ok(guard) => guard.current_epoch(),
        Err(poisoned) => poisoned.into_inner().current_epoch(),
    }
}

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

fn make_input(
    parquet_file: String,
    include_synapse: bool,
    include_neuron: bool,
) -> AnalyzeAllInput {
    let creature = build_creature();
    let focus_neurons: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output" || n.neuron_type == "hidden")
        .map(|n| n.uuid.clone())
        .collect();

    AnalyzeAllInput {
        parquet_file,
        creature,
        focus_neurons,
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(include_synapse),
        include_neuron_analysis: Some(include_neuron),
        random_seed: Some(42),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        cost_name: None,
    }
}

/// A full discovery pass running BOTH the neuron and synapse paths advances
/// the global tracker's epoch by exactly one — not zero, not two.
#[test]
#[serial]
fn full_discovery_pass_advances_global_epoch_by_exactly_one() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let creature = build_creature();
    let records = build_records(&creature, 60);
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let parquet_path = temp_dir.path().join("epoch.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    write_records_to_parquet(&parquet_file, &records).expect("write parquet");

    let input = make_input(parquet_file, true, true);

    let before = current_global_epoch();
    analyze_all(&input).expect("analyze_all");
    assert_eq!(
        current_global_epoch(),
        before + 1,
        "one full pass (neuron + synapse) must advance the epoch by exactly 1"
    );

    // A second pass proves the advance is per-pass, not a one-off at start-up.
    analyze_all(&input).expect("analyze_all");
    assert_eq!(
        current_global_epoch(),
        before + 2,
        "a second pass must advance the epoch by exactly 1 again"
    );
}

/// The advance is at the head of the pass, so it happens on every pass —
/// including GPU-less hosts and passes where both analyses are disabled. This
/// keeps the wiring guard effective in CI without a GPU.
#[test]
#[serial]
fn pass_with_both_analyses_disabled_still_advances_epoch_once() {
    let input = make_input("unused.parquet".to_string(), false, false);

    let before = current_global_epoch();
    analyze_all(&input).expect("analyze_all");
    assert_eq!(
        current_global_epoch(),
        before + 1,
        "every discovery pass must advance the epoch by exactly 1"
    );
}
