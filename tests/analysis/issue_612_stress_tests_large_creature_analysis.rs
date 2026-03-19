//! Stress tests for large creature analysis (Issue #612).
//!
//! These tests verify that the full analysis pipeline handles large synthetic
//! creatures (1,000+ neurons, 5,000+ synapses) without panics, OOM errors,
//! or unbounded memory growth. All tests are marked `#[ignore]` so they do
//! not slow CI — run them on demand with:
//!
//! ```sh
//! cargo test --test issue_612_stress_tests_large_creature_analysis -- --ignored --test-threads=1
//! ```

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};

// ---------------------------------------------------------------------------
// Helper: run the full pipeline and return parsed JSON output
// ---------------------------------------------------------------------------

/// Write discovery records to parquet and run the full analysis pipeline.
/// Panics on infrastructure failure so tests focus on domain assertions.
fn run_pipeline(
    creature: &CreatureJson,
    records: &[DiscoverRecord],
    focus_neurons: &[&str],
) -> serde_json::Value {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("valid UTF-8 path")
        .to_string();

    write_records_to_parquet(&parquet_file, records).expect("write parquet");

    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": focus_neurons,
        "maxSynapseCandidates": 64,
        "maxNeuronCandidates": 64,
        "randomSeed": 42
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should return JSON");
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

// ---------------------------------------------------------------------------
// Helper: build a large synthetic creature with N hidden neurons
// ---------------------------------------------------------------------------

/// Build a creature with `n_hidden` hidden neurons and a rich synapse topology.
///
/// Topology:
/// - 5 input neurons (input-0 … input-4)
/// - `n_hidden` hidden neurons with mixed activation functions
/// - 1 output neuron
/// - Each hidden neuron receives a synapse from one or two inputs
/// - Each hidden neuron sends a synapse to the output
/// - Additional inter-hidden synapses create a deeper topology
///
/// This produces roughly `n_hidden * 3` synapses for moderate `n_hidden`.
fn build_large_creature(n_hidden: usize) -> CreatureJson {
    let n_inputs = 5;
    let mut neurons = Vec::with_capacity(n_hidden + 1);
    let mut synapses = Vec::with_capacity(n_hidden * 3 + n_hidden / 5);

    // Activation functions to cycle through
    let squash_options = ["TANH", "ReLU", "LOGISTIC", "IDENTITY", "SOFTSIGN"];

    for i in 0..n_hidden {
        let squash = squash_options[i % squash_options.len()];
        let bias = ((i % 7) as f32 - 3.0) * 0.1; // Small varied biases
        neurons.push(NeuronJson {
            uuid: format!("h-{i}"),
            neuron_type: "hidden".to_string(),
            squash: squash.to_string(),
            bias,
        });

        // Primary input synapse: input-(i % n_inputs) → h-i
        synapses.push(SynapseJson {
            from_uuid: format!("input-{}", i % n_inputs),
            to_uuid: format!("h-{i}"),
            weight: 0.5 - (i % 11) as f32 * 0.05,
            synapse_type: None,
        });

        // Secondary input synapse for some neurons: input-((i+2) % n_inputs) → h-i
        if i % 3 == 0 {
            synapses.push(SynapseJson {
                from_uuid: format!("input-{}", (i + 2) % n_inputs),
                to_uuid: format!("h-{i}"),
                weight: 0.3 - (i % 9) as f32 * 0.03,
                synapse_type: None,
            });
        }

        // Output synapse: h-i → output-0
        synapses.push(SynapseJson {
            from_uuid: format!("h-{i}"),
            to_uuid: "output-0".to_string(),
            weight: 0.1 - (i % 13) as f32 * 0.01,
            synapse_type: None,
        });

        // Inter-hidden synapse: h-(i-5) → h-i (creates depth, forward-only)
        if i >= 5 && i % 4 == 0 {
            synapses.push(SynapseJson {
                from_uuid: format!("h-{}", i - 5),
                to_uuid: format!("h-{i}"),
                weight: 0.2,
                synapse_type: None,
            });
        }
    }

    // Output neuron
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

/// Generate synthetic discovery records for a large creature.
///
/// Produces `n_obs` observations, each with one record per neuron (hidden + output).
/// Activations vary by observation to create realistic signal patterns. Output
/// has residual error so the pipeline has work to do.
fn generate_records(creature: &CreatureJson, n_obs: u32) -> Vec<DiscoverRecord> {
    let n_neurons = creature.neurons.len();
    let mut records = Vec::with_capacity(n_neurons * n_obs as usize);

    for obs in 0..n_obs {
        let t = obs as f32 / n_obs as f32; // Normalised [0, 1)
        let x = t * 2.0 - 1.0; // [-1, 1)

        for neuron in &creature.neurons {
            let (pre_act, activation, error) = if neuron.neuron_type == "output" {
                // Output neuron with residual error
                let out_val = x * 0.1;
                let target = x * 0.5 + (x * x) * 0.2;
                (out_val, out_val, target - out_val)
            } else {
                // Hidden neuron — varied activations based on squash type
                let base = x * 0.3 + neuron.bias;
                let act = match neuron.squash.as_str() {
                    "TANH" => base.tanh(),
                    "ReLU" => base.max(0.0),
                    "LOGISTIC" => 1.0 / (1.0 + (-base).exp()),
                    "SOFTSIGN" => base / (1.0 + base.abs()),
                    _ => base, // IDENTITY
                };
                let err = (x * 0.5 - act) * 0.01; // Small backpropagated error
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

// ---------------------------------------------------------------------------
// Test 1: 1,000 hidden neurons — pipeline completes without panic
// ---------------------------------------------------------------------------

/// Stress test: creature with 1,000 hidden neurons (~3,000 synapses).
/// Verifies the full analysis pipeline completes without panics or OOM.
#[test]
#[ignore]
fn stress_1000_neurons_pipeline_completes_without_panic() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let n_hidden = 1_000;
    let creature = build_large_creature(n_hidden);

    eprintln!(
        "Built creature: {} neurons, {} synapses",
        creature.neurons.len(),
        creature.synapses.len()
    );
    assert!(creature.neurons.len() > 1_000);
    assert!(creature.synapses.len() > 2_000);

    let records = generate_records(&creature, 50);
    eprintln!("Generated {} discovery records", records.len());

    // Focus on output + a sample of hidden neurons (focusing all 1000 would
    // be extreme; pick every 10th to exercise breadth while staying practical)
    let focus_uuids: Vec<String> = std::iter::once("output-0".to_string())
        .chain((0..n_hidden).step_by(10).map(|i| format!("h-{i}")))
        .collect();
    let focus_refs: Vec<&str> = focus_uuids
        .iter()
        .map(std::string::String::as_str)
        .collect();

    let output = run_pipeline(&creature, &records, &focus_refs);

    // Pipeline completed (success asserted inside run_pipeline).
    // Log candidate counts for visibility.
    let helpful_syn = output["helpfulSynapses"]
        .as_array()
        .map_or(0, std::vec::Vec::len);
    let helpful_neu = output["helpfulNeurons"]
        .as_array()
        .map_or(0, std::vec::Vec::len);
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .map_or(0, std::vec::Vec::len);
    let weight_upd = output["synapseWeightUpdates"]
        .as_array()
        .map_or(0, std::vec::Vec::len);

    eprintln!(
        "1,000-neuron stress test completed: helpfulSynapses={helpful_syn}, \
         helpfulNeurons={helpful_neu}, coordinated={coordinated}, \
         weightUpdates={weight_upd}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: 2,000 hidden neurons with 5,000+ synapses
// ---------------------------------------------------------------------------

/// Stress test: creature with 2,000 hidden neurons (~6,000+ synapses).
/// Exceeds the 5,000-synapse threshold specified in the issue. Verifies
/// the pipeline handles this scale without crashes.
#[test]
#[ignore]
fn stress_2000_neurons_5000_plus_synapses() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let n_hidden = 2_000;
    let creature = build_large_creature(n_hidden);

    eprintln!(
        "Built creature: {} neurons, {} synapses",
        creature.neurons.len(),
        creature.synapses.len()
    );
    assert!(
        creature.synapses.len() >= 5_000,
        "Expected 5,000+ synapses, got {}",
        creature.synapses.len()
    );

    let records = generate_records(&creature, 30);
    eprintln!("Generated {} discovery records", records.len());

    // Focus on output + every 20th hidden neuron
    let focus_uuids: Vec<String> = std::iter::once("output-0".to_string())
        .chain((0..n_hidden).step_by(20).map(|i| format!("h-{i}")))
        .collect();
    let focus_refs: Vec<&str> = focus_uuids
        .iter()
        .map(std::string::String::as_str)
        .collect();

    let output = run_pipeline(&creature, &records, &focus_refs);

    let total = output["helpfulSynapses"]
        .as_array()
        .map_or(0, std::vec::Vec::len)
        + output["helpfulNeurons"]
            .as_array()
            .map_or(0, std::vec::Vec::len)
        + output["coordinatedStructuralCandidates"]
            .as_array()
            .map_or(0, std::vec::Vec::len)
        + output["synapseWeightUpdates"]
            .as_array()
            .map_or(0, std::vec::Vec::len);

    eprintln!("2,000-neuron stress test completed: {total} total candidates across all types");
}

// ---------------------------------------------------------------------------
// Test 3: Memory stability — run pipeline twice, no unbounded growth
// ---------------------------------------------------------------------------

/// Verify that running the analysis pipeline multiple times on large creatures
/// does not cause unbounded memory growth. We compare memory usage before and
/// after two runs, checking that the increase is bounded.
#[test]
#[ignore]
fn stress_memory_no_unbounded_growth() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let n_hidden = 500;
    let creature = build_large_creature(n_hidden);
    let records = generate_records(&creature, 40);

    let focus_uuids: Vec<String> = std::iter::once("output-0".to_string())
        .chain((0..n_hidden).step_by(10).map(|i| format!("h-{i}")))
        .collect();
    let focus_refs: Vec<&str> = focus_uuids
        .iter()
        .map(std::string::String::as_str)
        .collect();

    // First run — warm up GPU, caches, etc.
    let _output1 = run_pipeline(&creature, &records, &focus_refs);

    // Capture memory after first run
    let mem_after_first = get_process_memory_bytes();

    // Second run — should not cause significant additional memory usage
    let _output2 = run_pipeline(&creature, &records, &focus_refs);

    // Third run — check for creeping growth
    let _output3 = run_pipeline(&creature, &records, &focus_refs);

    let mem_after_third = get_process_memory_bytes();

    // Allow up to 100 MB growth across two additional runs. Anything above
    // that suggests unbounded growth (leaking buffers, uncapped caches, etc.).
    let growth_bytes = mem_after_third.saturating_sub(mem_after_first);
    let growth_mb = growth_bytes as f64 / (1024.0 * 1024.0);

    eprintln!(
        "Memory after first run: {:.1} MB, after third run: {:.1} MB, growth: {:.1} MB",
        mem_after_first as f64 / (1024.0 * 1024.0),
        mem_after_third as f64 / (1024.0 * 1024.0),
        growth_mb,
    );

    assert!(
        growth_mb < 100.0,
        "Memory grew by {growth_mb:.1} MB across two additional pipeline runs — \
         possible unbounded growth (limit: 100 MB)"
    );
}

/// Get the current process memory usage in bytes (resident set size).
#[cfg(target_os = "macos")]
fn get_process_memory_bytes() -> u64 {
    use std::process::Command;
    let pid = std::process::id();
    Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
        .unwrap_or(0)
        * 1024 // ps reports in KB on macOS
}

#[cfg(target_os = "linux")]
fn get_process_memory_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|p| p.parse::<u64>().ok())
        })
        .unwrap_or(0)
        * 4096 // statm reports in pages (typically 4KB on Linux)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn get_process_memory_bytes() -> u64 {
    0 // Fallback — test will pass since growth from 0 is treated as unknown
}

// ---------------------------------------------------------------------------
// Test 4: GPU buffer allocation with large workload
// ---------------------------------------------------------------------------

/// Verify that GPU buffer allocation handles a large creature correctly.
/// A creature with many neurons means larger GPU buffers for activation data.
/// This test checks that the pipeline does not panic from buffer over-allocation.
#[test]
#[ignore]
fn stress_gpu_buffers_handle_large_creature() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let n_hidden = 1_500;
    let creature = build_large_creature(n_hidden);

    eprintln!(
        "Built creature: {} neurons, {} synapses",
        creature.neurons.len(),
        creature.synapses.len()
    );

    // Use a larger observation count to push GPU buffer sizes
    let records = generate_records(&creature, 100);
    eprintln!(
        "Generated {} discovery records ({} observations x {} neurons)",
        records.len(),
        100,
        creature.neurons.len()
    );

    // Focus on output + every 15th hidden neuron
    let focus_uuids: Vec<String> = std::iter::once("output-0".to_string())
        .chain((0..n_hidden).step_by(15).map(|i| format!("h-{i}")))
        .collect();
    let focus_refs: Vec<&str> = focus_uuids
        .iter()
        .map(std::string::String::as_str)
        .collect();

    let output = run_pipeline(&creature, &records, &focus_refs);

    // Verify GPU was actually used (not silently skipped)
    let synapse_gpu = output["synapseGpuUsed"].as_bool().unwrap_or(false);
    let neuron_gpu = output["neuronGpuUsed"].as_bool().unwrap_or(false);
    eprintln!("GPU usage: synapseGpuUsed={synapse_gpu}, neuronGpuUsed={neuron_gpu}");

    // At least one GPU path should have been exercised
    assert!(
        synapse_gpu || neuron_gpu,
        "Expected GPU to be used for analysis of a large creature"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Mixed activation functions at scale
// ---------------------------------------------------------------------------

/// Verify the pipeline handles a large creature where every activation function
/// type is well-represented. This exercises all GPU shader paths at scale.
#[test]
#[ignore]
fn stress_mixed_activations_at_scale() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let n_hidden = 1_000;
    let creature = build_large_creature(n_hidden);

    // Verify we have a good mix of activation functions
    let mut squash_counts = std::collections::HashMap::new();
    for neuron in &creature.neurons {
        *squash_counts
            .entry(neuron.squash.as_str())
            .or_insert(0usize) += 1;
    }
    eprintln!("Activation function distribution: {squash_counts:?}");
    assert!(
        squash_counts.len() >= 4,
        "Expected at least 4 distinct activation functions, got {}",
        squash_counts.len()
    );

    let records = generate_records(&creature, 50);

    // Focus on a cross-section of neurons with different activation functions
    let focus_uuids: Vec<String> = std::iter::once("output-0".to_string())
        .chain((0..n_hidden).step_by(5).map(|i| format!("h-{i}")))
        .collect();
    let focus_refs: Vec<&str> = focus_uuids
        .iter()
        .map(std::string::String::as_str)
        .collect();

    let output = run_pipeline(&creature, &records, &focus_refs);

    let total = output["helpfulSynapses"]
        .as_array()
        .map_or(0, std::vec::Vec::len)
        + output["helpfulNeurons"]
            .as_array()
            .map_or(0, std::vec::Vec::len)
        + output["coordinatedStructuralCandidates"]
            .as_array()
            .map_or(0, std::vec::Vec::len)
        + output["synapseWeightUpdates"]
            .as_array()
            .map_or(0, std::vec::Vec::len);

    eprintln!(
        "Mixed-activation stress test: {total} candidates from {} focus neurons",
        focus_refs.len()
    );
}
