use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};

/// Integration-style test using the public JSON API (`analyze_parallel_internal`).
///
/// This constructs the simplest "two thermometers" scenario and asserts we get a
/// coordinated (grouped) candidate that can be applied in TypeScript using only:
/// - remove synapse
/// - remove synapse
/// - add synapse (with adjusted weight)
///
/// ASCII art used for the test scenario:
///
/// Mercury Thermometer (°C) ----[ weight w₁ (lower trust) ]------\
///                                                                \
///                                                               (+) ----> Output Neuron ----> Temperature (°F)
///                                                                /
/// Digital Thermometer (°C) ----[ weight w₂ (higher trust) ]-----/
///                                        [ bias ]
///
/// Test intent:
/// - The mercury synapse is *harmful* (its signal pushes the output further in the wrong direction),
///   so we expect a candidate that removes it.
/// - The digital synapse is *under-weighted*, so we expect a candidate that removes the old digital
///   synapse and adds it back with a larger weight.
#[test]
fn coordinated_structural_mercury_noisy_digital_trusted_returns_remove_remove_add() {
    // Discovery is GPU-only. On machines without GPU, we skip.
    if !GpuAnalyzer::gpu_is_available() {
        return;
    }

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    // We keep everything IDENTITY so the math stays linear and deterministic.
    //
    // This is intentionally the "simple case":
    // - The *mean* of both signals is the same.
    // - The *weights* of both synapses start the same.
    // - The mercury signal is much noisier (higher variance).
    //
    // The desired coordinated fix is:
    // - remove mercury synapse entirely
    // - double the digital synapse weight (so the mean contribution stays the same)
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![
            // Mercury (noisy) → output: wrong sign so it should be removed.
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.05, // equal weights (w₁ == w₂)
                synapse_type: None,
            },
            // Digital (trusted) → output: under-weighted so it should be increased.
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.05, // equal weights (w₁ == w₂)
                synapse_type: None,
            },
        ],
    };

    // Synthetic records:
    //
    // Mercury signal (input-0) is noisy but has the same mean as digital:
    // - alternates between 0.0 and 2.0 (mean = 1.0)
    //
    // Digital signal (input-1) is stable:
    // - constant 1.0 (mean = 1.0)
    //
    // Output errors alternate because mercury noise pushes the output above/below the desired value.
    // If we remove mercury and double digital, the output becomes stable and errors go to ~0.
    let mut records = Vec::new();
    for obs_index in 0..64u32 {
        let mercury_activation = if obs_index % 2 == 0 { 0.0 } else { 2.0 };
        let digital_activation = 1.0;
        // With w₁ = w₂ = 0.05:
        // current output contribution = 0.05 * mercury + 0.05 * digital
        // desired output contribution = 0.05 * 1.0 + 0.05 * 1.0 = 0.10
        // error = desired - current
        let output_error = 0.10 - (0.05 * mercury_activation + 0.05 * digital_activation);

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(), // mercury
            Some(0.0),
            mercury_activation,
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(), // digital
            Some(0.0),
            digital_activation,
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![output_error],
        ));
    }
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write discovery records");

    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": ["output-0"],
        "maxSynapseCandidates": 10,
        "maxNeuronCandidates": 0,
        "analysisDeadlineMs": 10_000,
        "randomSeed": 1
    })
    .to_string();

    let output_json =
        analyze_parallel_internal(&input_json).expect("parallel analysis should return JSON");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(output["success"], true);

    let groups = output["coordinatedStructuralCandidates"]
        .as_array()
        .expect("should include coordinatedStructuralCandidates array");
    assert!(
        !groups.is_empty(),
        "expected at least one coordinated structural candidate"
    );

    // We only assert the first one - candidates are sorted by expected gain, highest first.
    let ops = groups[0]["operations"]
        .as_array()
        .expect("grouped candidate should have operations[]");
    assert_eq!(
        ops.len(),
        3,
        "expected remove/remove/add so TypeScript can apply without a set-weight op"
    );

    // op[0] remove mercury
    assert_eq!(ops[0]["type"], "removeSynapse");
    assert_eq!(ops[0]["fromNeuronUuid"], "input-0");
    assert_eq!(ops[0]["toNeuronUuid"], "output-0");

    // op[1] remove digital (old)
    assert_eq!(ops[1]["type"], "removeSynapse");
    assert_eq!(ops[1]["fromNeuronUuid"], "input-1");
    assert_eq!(ops[1]["toNeuronUuid"], "output-0");

    // op[2] add digital (new)
    assert_eq!(ops[2]["type"], "addSynapse");
    assert_eq!(ops[2]["fromNeuronUuid"], "input-1");
    assert_eq!(ops[2]["toNeuronUuid"], "output-0");
    assert!(
        (ops[2]["weight"].as_f64().unwrap_or(0.0) - 0.10).abs() < 1e-6,
        "expected the new digital weight to be doubled (0.10)"
    );
}
