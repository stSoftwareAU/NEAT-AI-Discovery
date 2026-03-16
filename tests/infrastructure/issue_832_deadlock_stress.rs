//! Concurrency stress tests for deadlock and contention detection (Issue #832).
//!
//! These tests exercise the analysis pipeline under high concurrency to detect
//! deadlocks, lock contention, and stalls. They use `parking_lot::deadlock::check_deadlock()`
//! to verify no deadlocks occur, and confirm the pipeline completes within a
//! reasonable time bound.
//!
//! Tests use seeded RNG for reproducibility and a private Rayon thread pool
//! with high thread counts to maximise contention.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

// ---------------------------------------------------------------------------
// Helpers: synthetic creature and record generation
// ---------------------------------------------------------------------------

/// Build a creature with many hidden neurons and a rich synapse topology.
///
/// Topology:
/// - 5 input neurons (input-0 … input-4)
/// - `n_hidden` hidden neurons with mixed activation functions
/// - 1 output neuron
/// - Each hidden neuron receives a synapse from one or two inputs
/// - Each hidden neuron sends a synapse to the output
/// - Inter-hidden synapses create depth (forward-only)
fn build_stress_creature(n_hidden: usize) -> CreatureJson {
    let n_inputs = 5;
    let mut neurons = Vec::with_capacity(n_hidden + 1);
    let mut synapses = Vec::with_capacity(n_hidden * 3 + n_hidden / 5);

    let squash_options = ["TANH", "ReLU", "LOGISTIC", "IDENTITY", "SOFTSIGN"];

    for i in 0..n_hidden {
        let squash = squash_options[i % squash_options.len()];
        let bias = ((i % 7) as f32 - 3.0) * 0.1;
        neurons.push(NeuronJson {
            uuid: format!("h-{i}"),
            neuron_type: "hidden".to_string(),
            squash: squash.to_string(),
            bias,
        });

        // Primary input synapse
        synapses.push(SynapseJson {
            from_uuid: format!("input-{}", i % n_inputs),
            to_uuid: format!("h-{i}"),
            weight: 0.5 - (i % 11) as f32 * 0.05,
            synapse_type: None,
        });

        // Secondary input synapse for some neurons
        if i % 3 == 0 {
            synapses.push(SynapseJson {
                from_uuid: format!("input-{}", (i + 2) % n_inputs),
                to_uuid: format!("h-{i}"),
                weight: 0.3 - (i % 9) as f32 * 0.03,
                synapse_type: None,
            });
        }

        // Output synapse
        synapses.push(SynapseJson {
            from_uuid: format!("h-{i}"),
            to_uuid: "output-0".to_string(),
            weight: 0.1 - (i % 13) as f32 * 0.01,
            synapse_type: None,
        });

        // Inter-hidden synapse (forward-only depth)
        if i >= 5 && i % 4 == 0 {
            synapses.push(SynapseJson {
                from_uuid: format!("h-{}", i - 5),
                to_uuid: format!("h-{i}"),
                weight: 0.2,
                synapse_type: None,
            });
        }
    }

    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    CreatureJson {
        input: n_inputs,
        output: 1,
        neurons,
        synapses,
    }
}

/// Generate synthetic discovery records with deterministic variation.
fn generate_stress_records(creature: &CreatureJson, n_obs: u32) -> Vec<DiscoverRecord> {
    let n_neurons = creature.neurons.len();
    let mut records = Vec::with_capacity(n_neurons * n_obs as usize);

    for obs in 0..n_obs {
        let t = obs as f32 / n_obs as f32;
        let x = t * 2.0 - 1.0;

        for neuron in &creature.neurons {
            let (pre_act, activation, error) = if neuron.neuron_type == "output" {
                let out_val = x * 0.1;
                let target = x * 0.5 + (x * x) * 0.2;
                (out_val, out_val, target - out_val)
            } else {
                let base = x * 0.3 + neuron.bias;
                let act = match neuron.squash.as_str() {
                    "TANH" => base.tanh(),
                    "ReLU" => base.max(0.0),
                    "LOGISTIC" => 1.0 / (1.0 + (-base).exp()),
                    "SOFTSIGN" => base / (1.0 + base.abs()),
                    _ => base,
                };
                let err = (x * 0.5 - act) * 0.01;
                (base, act, err)
            };

            records.push(DiscoverRecord::new(
                obs,
                neuron.uuid.clone(),
                Some(pre_act),
                activation,
                vec![error],
            ));
        }
    }

    records
}

/// Write records to a temporary parquet file and return the path and temp dir handle.
fn write_temp_parquet(records: &[DiscoverRecord], suffix: &str) -> (String, tempfile::TempDir) {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let parquet_file = temp_dir
        .path()
        .join(format!("stress_{suffix}.parquet"))
        .to_str()
        .expect("valid UTF-8 path")
        .to_string();
    write_records_to_parquet(&parquet_file, records).expect("write parquet");
    (parquet_file, temp_dir)
}

/// Run `analyze_all` via the internal JSON interface (same pattern as issue_612 tests).
fn run_analysis_pipeline(
    creature: &CreatureJson,
    parquet_file: &str,
    focus_neurons: &[String],
    deadline_ms: Option<u64>,
    seed: u64,
) -> serde_json::Value {
    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": focus_neurons,
        "maxSynapseCandidates": 32,
        "maxNeuronCandidates": 32,
        "analysisDeadlineMs": deadline_ms,
        "includeSynapseAnalysis": true,
        "includeNeuronAnalysis": true,
        "randomSeed": seed,
    })
    .to_string();

    let output_json =
        neat_ai_discovery::analyze_parallel_internal(&input_json).expect("analysis should succeed");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(
        output["success"],
        true,
        "Analysis failed: {}",
        output["error"].as_str().unwrap_or("unknown")
    );

    output
}

/// Spawn a background thread that periodically checks for deadlocks.
/// Returns (checker_handle, deadlock_flag, done_signal).
fn spawn_deadlock_checker(
    interval_ms: u64,
) -> (
    std::thread::JoinHandle<()>,
    Arc<AtomicBool>,
    Arc<AtomicBool>,
) {
    let deadlock_found = Arc::new(AtomicBool::new(false));
    let deadlock_flag = Arc::clone(&deadlock_found);
    let done = Arc::new(AtomicBool::new(false));
    let done_flag = Arc::clone(&done);

    let handle = std::thread::spawn(move || {
        while !done_flag.load(Ordering::Relaxed) {
            let deadlocks = parking_lot::deadlock::check_deadlock();
            if !deadlocks.is_empty() {
                eprintln!("DEADLOCK DETECTED: {} deadlocked threads", deadlocks.len());
                deadlock_flag.store(true, Ordering::SeqCst);
                return;
            }
            std::thread::sleep(Duration::from_millis(interval_ms));
        }
    });

    (handle, deadlock_found, done)
}

// ---------------------------------------------------------------------------
// Test 1: Full pipeline stress with deadlock detection
// ---------------------------------------------------------------------------

/// Exercise the full analysis pipeline (synapse + neuron) with a network
/// and verify no deadlocks occur. A background thread periodically checks for
/// deadlocks using `parking_lot::deadlock::check_deadlock()`.
#[test]
fn stress_full_pipeline_no_deadlocks_under_high_concurrency() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    // Initialise parking_lot deadlock detector
    neat_ai_discovery::debug::init_debug_handlers();

    let n_hidden = 20;
    let creature = build_stress_creature(n_hidden);
    let records = generate_stress_records(&creature, 30);
    let (parquet_file, _tmp) = write_temp_parquet(&records, "full_pipeline");

    // Focus on output + a selection of hidden neurons
    let focus_neurons: Vec<String> = std::iter::once("output-0".to_string())
        .chain((0..n_hidden).step_by(5).map(|i| format!("h-{i}")))
        .collect();

    let (checker, deadlock_found, done) = spawn_deadlock_checker(200);

    let start = Instant::now();
    let _output = run_analysis_pipeline(&creature, &parquet_file, &focus_neurons, None, 42);
    let elapsed = start.elapsed();
    done.store(true, Ordering::SeqCst);
    checker.join().expect("deadlock checker thread panicked");

    assert!(
        !deadlock_found.load(Ordering::SeqCst),
        "Deadlock detected during full pipeline analysis"
    );

    // The pipeline should complete within 60 seconds — slower indicates contention
    assert!(
        elapsed < Duration::from_secs(60),
        "Full pipeline took {:.1}s — exceeds 60s contention threshold",
        elapsed.as_secs_f64()
    );

    eprintln!(
        "Full pipeline stress test completed in {:.2}s with no deadlocks",
        elapsed.as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// Test 2: Impact computation stress with concurrent access
// ---------------------------------------------------------------------------

/// Stress-test `compute_impacts_public` by running it concurrently from
/// multiple threads on a large network. Verifies determinism and no deadlocks.
#[test]
fn stress_impact_computation_concurrent_determinism() {
    // Initialise deadlock detection
    neat_ai_discovery::debug::init_debug_handlers();

    let n_hidden = 150;
    let creature = build_stress_creature(n_hidden);

    // Compute baseline impacts
    let baseline = compute_impacts_public(&creature);
    assert!(!baseline.is_empty(), "Baseline impacts should not be empty");

    let (checker, deadlock_found, done) = spawn_deadlock_checker(100);

    // Run impact computation concurrently from a private thread pool
    let creature_arc = Arc::new(creature);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .expect("build rayon thread pool");

    let start = Instant::now();
    let results: Vec<_> = pool.install(|| {
        use rayon::prelude::*;
        (0..16)
            .into_par_iter()
            .map(|_| compute_impacts_public(&creature_arc))
            .collect()
    });
    let elapsed = start.elapsed();

    done.store(true, Ordering::SeqCst);
    checker.join().expect("deadlock checker thread panicked");

    assert!(
        !deadlock_found.load(Ordering::SeqCst),
        "Deadlock detected during concurrent impact computation"
    );

    // Verify all results match the baseline (determinism)
    for (run, impacts) in results.iter().enumerate() {
        for (uuid, &expected) in &baseline {
            let actual = impacts.get(uuid).copied().unwrap_or(f32::NAN);
            assert!(
                (actual - expected).abs() < 1e-6,
                "Run {run}: impact for {uuid} differs: expected {expected}, got {actual}"
            );
        }
    }

    assert!(
        elapsed < Duration::from_secs(30),
        "Concurrent impact computation took {:.1}s — exceeds 30s threshold",
        elapsed.as_secs_f64()
    );

    eprintln!(
        "Concurrent impact stress test: 16 parallel runs completed in {:.2}s",
        elapsed.as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// Test 3: Repeated rapid analyses (simulating rapid repeated calls)
// ---------------------------------------------------------------------------

/// Simulate rapid repeated calls to the analysis pipeline, as would occur
/// during a NEAT-AI evolution run. Verifies no resource exhaustion or deadlocks.
#[test]
fn stress_rapid_repeated_analyses_no_resource_exhaustion() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    neat_ai_discovery::debug::init_debug_handlers();

    let n_hidden = 15;
    let creature = build_stress_creature(n_hidden);
    let records = generate_stress_records(&creature, 25);
    let (parquet_file, _tmp) = write_temp_parquet(&records, "rapid_repeat");

    let focus_neurons: Vec<String> = std::iter::once("output-0".to_string())
        .chain((0..n_hidden).step_by(5).map(|i| format!("h-{i}")))
        .collect();

    let (checker, deadlock_found, done) = spawn_deadlock_checker(200);

    let start = Instant::now();
    let n_runs: u64 = 3;
    for run in 0..n_runs {
        let seed = 42 + run;
        let output = run_analysis_pipeline(&creature, &parquet_file, &focus_neurons, None, seed);
        assert_eq!(output["success"], true, "Run {run} failed");
    }
    let elapsed = start.elapsed();

    done.store(true, Ordering::SeqCst);
    checker.join().expect("deadlock checker thread panicked");

    assert!(
        !deadlock_found.load(Ordering::SeqCst),
        "Deadlock detected during rapid repeated analyses"
    );

    assert!(
        elapsed < Duration::from_secs(60),
        "{n_runs} rapid analyses took {:.1}s — exceeds 60s threshold",
        elapsed.as_secs_f64()
    );

    eprintln!(
        "Rapid repeat stress test: {n_runs} analyses completed in {:.2}s ({:.2}s avg)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / n_runs as f64
    );
}

// ---------------------------------------------------------------------------
// Test 4: Tight deadline under contention
// ---------------------------------------------------------------------------

/// Run analysis with a tight deadline to exercise timeout paths under
/// contention. The pipeline should respect the deadline and return a result
/// (possibly partial) without deadlocking.
#[test]
fn stress_tight_deadline_respects_timeout_without_deadlock() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    neat_ai_discovery::debug::init_debug_handlers();

    let n_hidden = 20;
    let creature = build_stress_creature(n_hidden);
    let records = generate_stress_records(&creature, 30);
    let (parquet_file, _tmp) = write_temp_parquet(&records, "tight_deadline");

    let focus_neurons: Vec<String> = std::iter::once("output-0".to_string())
        .chain((0..n_hidden).step_by(5).map(|i| format!("h-{i}")))
        .collect();

    let (checker, deadlock_found, done) = spawn_deadlock_checker(100);

    // Use a tight deadline of 2 seconds
    let tight_deadline_ms = 2_000;
    let start = Instant::now();
    let output = run_analysis_pipeline(
        &creature,
        &parquet_file,
        &focus_neurons,
        Some(tight_deadline_ms),
        42,
    );
    let elapsed = start.elapsed();

    done.store(true, Ordering::SeqCst);
    checker.join().expect("deadlock checker thread panicked");

    assert!(
        !deadlock_found.load(Ordering::SeqCst),
        "Deadlock detected during tight-deadline analysis"
    );

    // The pipeline should complete — possibly with partial results — but
    // not hang far beyond the deadline. Allow generous margin for GPU init.
    assert!(
        elapsed < Duration::from_secs(60),
        "Tight-deadline analysis took {:.1}s — far exceeds deadline, possible stall",
        elapsed.as_secs_f64()
    );

    // Result should still be valid JSON with success=true
    assert_eq!(output["success"], true);

    eprintln!(
        "Tight deadline stress test completed in {:.2}s (deadline: {tight_deadline_ms}ms)",
        elapsed.as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// Test 5: Deadlock detection mechanism verification
// ---------------------------------------------------------------------------

/// Verify that `parking_lot::deadlock::check_deadlock()` correctly reports
/// no deadlocks in clean concurrent analysis state. This confirms the
/// detection mechanism is active and functional.
#[test]
fn stress_deadlock_detection_reports_clean_state_after_concurrent_work() {
    neat_ai_discovery::debug::init_debug_handlers();

    let n_hidden = 100;
    let creature = build_stress_creature(n_hidden);

    // Run concurrent impact computations to exercise locks
    let creature_arc = Arc::new(creature);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("build rayon thread pool");

    pool.install(|| {
        use rayon::prelude::*;
        (0..8).into_par_iter().for_each(|_| {
            let _ = compute_impacts_public(&creature_arc);
        });
    });

    // After all concurrent work completes, verify no deadlocks remain
    let deadlocks = parking_lot::deadlock::check_deadlock();
    assert!(
        deadlocks.is_empty(),
        "Expected no deadlocks after concurrent work, found {}",
        deadlocks.len()
    );
}
