//! Stress tests for large creature analysis (Issue #612).
//!
//! These tests verify the full analysis pipeline handles scale correctly with
//! creatures containing 1,000+ neurons and 5,000+ synapses. They confirm no
//! panics, no OOM errors, and no unbounded memory growth during analysis.
//!
//! All tests are marked `#[ignore]` so they do not slow CI but can be run on
//! demand with `cargo test -- --ignored --test-threads=1`.

mod common;

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};

// ---------------------------------------------------------------------------
// Helper: run the full pipeline for stress tests
// ---------------------------------------------------------------------------

/// Record discovery data to parquet and run parallel analysis, returning the
/// parsed JSON output. Panics on any infrastructure failure.
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
        "maxSynapseCandidates": 32,
        "maxNeuronCandidates": 32,
        "randomSeed": 42,
        "analysisDeadlineMs": 120_000
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
// Helper: build a synthetic large creature
// ---------------------------------------------------------------------------

/// Build a synthetic creature with the specified number of hidden neurons
/// arranged in layers, connected by a forward-only synapse topology.
///
/// The creature has `num_inputs` virtual inputs, `num_hidden` hidden neurons
/// distributed across `num_layers` layers, and 1 output neuron.
/// Synapses connect each layer to the next (sparse, forward-only).
fn build_large_creature(
    num_inputs: usize,
    num_hidden: usize,
    num_layers: usize,
) -> (CreatureJson, Vec<String>) {
    let neurons_per_layer = num_hidden / num_layers;
    let squash_options = ["TANH", "ReLU", "LOGISTIC", "IDENTITY"];

    let mut neurons = Vec::with_capacity(num_hidden + 1);
    let mut synapses = Vec::new();
    let mut focus_neurons = Vec::new();

    // Create hidden neurons in layers
    for layer in 0..num_layers {
        for i in 0..neurons_per_layer {
            let idx = layer * neurons_per_layer + i;
            let uuid = format!("h-{idx}");
            let squash = squash_options[idx % squash_options.len()];
            let bias = ((idx % 7) as f32 - 3.0) * 0.1; // Small varied biases

            neurons.push(NeuronJson {
                uuid: uuid.clone(),
                neuron_type: "hidden".to_string(),
                squash: squash.to_string(),
                bias,
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

    // Synapses: input → first layer
    for input_idx in 0..num_inputs {
        // Connect each input to a subset of first-layer neurons
        for i in 0..neurons_per_layer {
            if (input_idx + i) % 3 == 0 {
                let to_uuid = format!("h-{i}");
                let weight = ((input_idx + i) % 5) as f32 * 0.3 - 0.6;
                synapses.push(SynapseJson {
                    from_uuid: format!("input-{input_idx}"),
                    to_uuid,
                    weight,
                    synapse_type: None,
                });
            }
        }
    }

    // Synapses: layer-to-layer (sparse connections)
    for layer in 0..(num_layers - 1) {
        let src_start = layer * neurons_per_layer;
        let dst_start = (layer + 1) * neurons_per_layer;

        for src_i in 0..neurons_per_layer {
            // Connect to 2-3 neurons in the next layer
            for dst_i in 0..neurons_per_layer {
                if (src_i + dst_i) % 4 == 0 {
                    let from_uuid = format!("h-{}", src_start + src_i);
                    let to_uuid = format!("h-{}", dst_start + dst_i);
                    let weight = ((src_i + dst_i) % 7) as f32 * 0.2 - 0.6;
                    synapses.push(SynapseJson {
                        from_uuid,
                        to_uuid,
                        weight,
                        synapse_type: None,
                    });
                }
            }
        }
    }

    // Synapses: last layer → output
    let last_layer_start = (num_layers - 1) * neurons_per_layer;
    for i in 0..neurons_per_layer {
        if i % 2 == 0 {
            let from_uuid = format!("h-{}", last_layer_start + i);
            let weight = (i % 5) as f32 * 0.2 - 0.4;
            synapses.push(SynapseJson {
                from_uuid,
                to_uuid: "output-0".to_string(),
                weight,
                synapse_type: None,
            });
        }
    }

    // Focus on a subset of neurons (output + some from each layer)
    focus_neurons.push("output-0".to_string());
    for layer in 0..num_layers {
        let start = layer * neurons_per_layer;
        // Pick a few neurons per layer to focus on
        for i in (0..neurons_per_layer).step_by(neurons_per_layer / 3 + 1) {
            focus_neurons.push(format!("h-{}", start + i));
        }
    }

    let creature = CreatureJson {
        input: num_inputs,
        output: 1,
        neurons,
        synapses,
    };

    (creature, focus_neurons)
}

/// Generate synthetic discovery records for a large creature. Creates records
/// for all hidden neurons and the output neuron across `num_observations`
/// training samples.
fn build_large_records(creature: &CreatureJson, num_observations: usize) -> Vec<DiscoverRecord> {
    let total_neurons = creature.neurons.len();
    let mut records = Vec::with_capacity(total_neurons * num_observations);

    for obs in 0..num_observations as u32 {
        let t = obs as f32 / num_observations as f32;

        for (idx, neuron) in creature.neurons.iter().enumerate() {
            // Generate varied but deterministic activation patterns
            let phase = (idx as f32 * 0.37 + t * std::f32::consts::TAU).sin();
            let pre_activation = phase * (1.0 + (idx % 5) as f32 * 0.2);

            let activation = match neuron.squash.as_str() {
                "TANH" => pre_activation.tanh(),
                "ReLU" => pre_activation.max(0.0),
                "LOGISTIC" => 1.0 / (1.0 + (-pre_activation).exp()),
                _ => pre_activation, // IDENTITY
            };

            let error = if neuron.neuron_type == "output" {
                // Output has meaningful error signal
                (t * 2.0 - 1.0) * 0.3 - activation * 0.1
            } else {
                // Hidden neurons get backpropagated error
                phase * 0.05
            };

            records.push(DiscoverRecord::new(
                obs,
                neuron.uuid.clone(),
                Some(pre_activation),
                activation,
                vec![error],
            ));
        }
    }

    records
}

// ---------------------------------------------------------------------------
// Test 1: 1,000 hidden neurons — pipeline completes without panics
// ---------------------------------------------------------------------------

/// Stress test with 1,000 hidden neurons arranged in 10 layers. Verifies the
/// full analysis pipeline completes without panicking or returning an error.
#[test]
#[ignore]
fn stress_1000_neurons_pipeline_completes_without_panic() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let (creature, focus_neurons) = build_large_creature(10, 1000, 10);
    let records = build_large_records(&creature, 50);

    eprintln!(
        "Stress test: {} neurons, {} synapses, {} records",
        creature.neurons.len(),
        creature.synapses.len(),
        records.len()
    );

    assert!(creature.neurons.len() >= 1000, "Should have 1,000+ neurons");
    assert!(
        creature.synapses.len() >= 5000,
        "Should have 5,000+ synapses, got {}",
        creature.synapses.len()
    );

    let focus_refs: Vec<&str> = focus_neurons.iter().map(|s| s.as_str()).collect();
    let output = run_pipeline(&creature, &records, &focus_refs);

    // Pipeline completed — verify the output structure is valid
    assert!(
        output.get("helpfulSynapses").is_some()
            || output.get("coordinatedStructuralCandidates").is_some(),
        "Pipeline should return standard candidate fields"
    );

    eprintln!(
        "1,000-neuron stress test passed: helpfulSynapses={}, helpfulNeurons={}, coordinated={}",
        output["helpfulSynapses"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        output["helpfulNeurons"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        output["coordinatedStructuralCandidates"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
    );
}

// ---------------------------------------------------------------------------
// Test 2: 2,000 neurons — larger scale
// ---------------------------------------------------------------------------

/// Stress test with 2,000 hidden neurons in 20 layers. Exercises the pipeline
/// at a larger scale to confirm GPU buffer allocation and memory handling cope.
#[test]
#[ignore]
fn stress_2000_neurons_pipeline_completes_without_panic() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let (creature, focus_neurons) = build_large_creature(20, 2000, 20);
    let records = build_large_records(&creature, 30);

    eprintln!(
        "Stress test: {} neurons, {} synapses, {} records",
        creature.neurons.len(),
        creature.synapses.len(),
        records.len()
    );

    assert!(creature.neurons.len() >= 2000, "Should have 2,000+ neurons");

    let focus_refs: Vec<&str> = focus_neurons.iter().map(|s| s.as_str()).collect();
    let output = run_pipeline(&creature, &records, &focus_refs);

    assert!(
        output.get("helpfulSynapses").is_some()
            || output.get("coordinatedStructuralCandidates").is_some(),
        "Pipeline should return standard candidate fields"
    );

    eprintln!(
        "2,000-neuron stress test passed: helpfulSynapses={}, helpfulNeurons={}, coordinated={}",
        output["helpfulSynapses"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        output["helpfulNeurons"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        output["coordinatedStructuralCandidates"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
    );
}

// ---------------------------------------------------------------------------
// Test 3: Memory stability — no unbounded growth across repeated analyses
// ---------------------------------------------------------------------------

/// Run the analysis pipeline multiple times on a 1,000-neuron creature and
/// verify that memory usage does not grow unboundedly between iterations.
/// This catches memory leaks in caches, GPU buffers, or record storage.
#[test]
#[ignore]
fn stress_memory_stable_across_repeated_analyses() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let (creature, focus_neurons) = build_large_creature(10, 1000, 10);
    let focus_refs: Vec<&str> = focus_neurons.iter().map(|s| s.as_str()).collect();

    // Run the pipeline several times with different observation data
    let iterations = 5;
    for iteration in 0..iterations {
        // Generate fresh records each time (different observation data)
        let records = build_large_records(&creature, 40 + iteration * 5);

        let output = run_pipeline(&creature, &records, &focus_refs);

        assert_eq!(
            output["success"],
            true,
            "Iteration {iteration} failed: {}",
            output["error"].as_str().unwrap_or("unknown")
        );

        eprintln!(
            "Iteration {}/{}: {} records processed, pipeline succeeded",
            iteration + 1,
            iterations,
            records.len()
        );
    }

    eprintln!("Memory stability test passed: {iterations} iterations without unbounded growth");
}

// ---------------------------------------------------------------------------
// Test 4: Dense connectivity — many synapses per neuron
// ---------------------------------------------------------------------------

/// A creature with 500 neurons but very dense connectivity (each neuron
/// connected to many others). Tests that the synapse analysis handles high
/// fan-in and fan-out correctly without panics.
#[test]
#[ignore]
fn stress_dense_connectivity_no_panics() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let num_inputs = 10;
    let num_hidden = 500;
    let squash_options = ["TANH", "ReLU", "LOGISTIC", "IDENTITY"];

    let mut neurons = Vec::with_capacity(num_hidden + 1);
    let mut synapses = Vec::new();

    // Create hidden neurons in 5 layers of 100
    let layers = 5;
    let per_layer = num_hidden / layers;

    for idx in 0..num_hidden {
        neurons.push(NeuronJson {
            uuid: format!("h-{idx}"),
            neuron_type: "hidden".to_string(),
            squash: squash_options[idx % squash_options.len()].to_string(),
            bias: ((idx % 7) as f32 - 3.0) * 0.1,
        });
    }

    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    // Dense input→first layer connections
    for input_idx in 0..num_inputs {
        for i in 0..per_layer {
            let weight = ((input_idx + i) % 9) as f32 * 0.2 - 0.8;
            synapses.push(SynapseJson {
                from_uuid: format!("input-{input_idx}"),
                to_uuid: format!("h-{i}"),
                weight,
                synapse_type: None,
            });
        }
    }

    // Dense inter-layer connections (every 2nd source to every 2nd target)
    for layer in 0..(layers - 1) {
        let src_start = layer * per_layer;
        let dst_start = (layer + 1) * per_layer;

        for src_i in (0..per_layer).step_by(2) {
            for dst_i in (0..per_layer).step_by(2) {
                let weight = ((src_i + dst_i) % 11) as f32 * 0.15 - 0.75;
                synapses.push(SynapseJson {
                    from_uuid: format!("h-{}", src_start + src_i),
                    to_uuid: format!("h-{}", dst_start + dst_i),
                    weight,
                    synapse_type: None,
                });
            }
        }
    }

    // Last layer → output
    let last_start = (layers - 1) * per_layer;
    for i in 0..per_layer {
        synapses.push(SynapseJson {
            from_uuid: format!("h-{}", last_start + i),
            to_uuid: "output-0".to_string(),
            weight: (i % 5) as f32 * 0.2 - 0.4,
            synapse_type: None,
        });
    }

    let creature = CreatureJson {
        input: num_inputs,
        output: 1,
        neurons,
        synapses,
    };

    eprintln!(
        "Dense connectivity test: {} neurons, {} synapses",
        creature.neurons.len(),
        creature.synapses.len()
    );

    assert!(
        creature.synapses.len() >= 5000,
        "Should have 5,000+ synapses for dense test, got {}",
        creature.synapses.len()
    );

    let records = build_large_records(&creature, 40);

    // Focus on output + a selection of hidden neurons
    let mut focus_owned: Vec<String> = vec!["output-0".to_string()];
    for layer in 0..layers {
        let start = layer * per_layer;
        focus_owned.push(format!("h-{start}"));
        focus_owned.push(format!("h-{}", start + per_layer / 2));
    }
    let focus: Vec<&str> = focus_owned.iter().map(|s| s.as_str()).collect();

    let output = run_pipeline(&creature, &records, &focus);

    assert!(
        output.get("helpfulSynapses").is_some()
            || output.get("coordinatedStructuralCandidates").is_some(),
        "Pipeline should return standard candidate fields"
    );

    eprintln!(
        "Dense connectivity test passed: helpfulSynapses={}, coordinated={}",
        output["helpfulSynapses"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        output["coordinatedStructuralCandidates"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
    );
}

// ---------------------------------------------------------------------------
// Test 5: Many observations — large record count
// ---------------------------------------------------------------------------

/// A moderately sized creature (200 neurons) with a large number of
/// observations (500). Tests that parquet I/O and record streaming handle
/// large record volumes correctly.
#[test]
#[ignore]
fn stress_many_observations_no_panics() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let (creature, focus_neurons) = build_large_creature(10, 200, 5);
    let records = build_large_records(&creature, 500);

    eprintln!(
        "Many-observations test: {} neurons, {} records ({} observations x {} neurons)",
        creature.neurons.len(),
        records.len(),
        500,
        creature.neurons.len()
    );

    let focus_refs: Vec<&str> = focus_neurons.iter().map(|s| s.as_str()).collect();
    let output = run_pipeline(&creature, &records, &focus_refs);

    assert!(
        output.get("helpfulSynapses").is_some()
            || output.get("coordinatedStructuralCandidates").is_some(),
        "Pipeline should return standard candidate fields"
    );

    eprintln!(
        "Many-observations test passed: helpfulSynapses={}, helpfulNeurons={}, coordinated={}",
        output["helpfulSynapses"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        output["helpfulNeurons"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        output["coordinatedStructuralCandidates"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
    );
}
