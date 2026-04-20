//! Regression: "COMPLEMENT" / "INVERSE" discovery is represented via IDENTITY.
//!
//! We do not propose INVERSE as a new neuron activation because it's mathematically linear:
//! `INVERSE(x) = 1 - x`. This can be expressed with an IDENTITY neuron by using a bias and
//! negative incoming weights (and letting the outgoing weight scale it):
//!
//!   IDENTITY(incoming * x + bias) = k * (1 - x)  when  incoming = -k  and  bias = k
//!
//! This test ensures complement-like candidates are still discoverable, but as IDENTITY.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::skip_without_gpu;
use neat_ai_discovery::record_discovery_internal;
use neat_ai_discovery::{AnalyzeNeuronsInput, CreatureJson, NeuronJson};
use tempfile::TempDir;

#[test]
fn test_complement_is_discovered_as_identity_not_inverse() {
    skip_without_gpu!();

    let temp_dir = TempDir::new().expect("Failed to create temporary directory");
    let temp_path = temp_dir.path();

    // One output neuron, no synapses. Input neurons are implicit via `input` count.
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
        input: 1,
        output: 1,
    };

    // Build a dataset where the target error is proportional to the complement of input:
    //
    //   avg_error(x) = ERROR_SCALE * (1 - x)
    //
    // The best candidate correction is then:
    //
    //   correction(x) = outgoing * IDENTITY(incoming * x + bias)
    //                = outgoing * (incoming * x + bias)
    //
    // For complement behaviour (up to scale), we want:
    //   incoming ≈ -bias  and  correction(0) ≈ ERROR_SCALE, correction(1) ≈ 0
    // Issue #888: Reduced from 0.1 to 0.01 so that the optimal IDENTITY candidate
    // (incoming ≈ -1, bias ≈ 1, outgoing ≈ 0.01) falls within the tightened
    // weight constraints (MAX_OUTGOING_WEIGHT=0.01, MAX_BIAS_MAGNITUDE=2.0).
    const ERROR_SCALE: f32 = 0.01;

    let mut training_data = Vec::new();
    for i in 0..=100 {
        let x = (i as f32) / 100.0; // [0, 1]
        let error = ERROR_SCALE * (1.0 - x);
        training_data.push(serde_json::json!({
            "input": [x],
            "output": [0.0],
            "neuron_data": [{
                "neuron_uuid": "output-0",
                "activation": 0.0,
                "value": 0.0,
                "errors": [error]
            }]
        }));
    }

    // Record discovery data
    let input_json = serde_json::json!({
        "creature": creature,
        "training_data": training_data,
        "temp_dir": temp_path.to_str().expect("Temp path should be valid UTF-8")
    });
    let record_input = serde_json::to_string(&input_json).expect("Input JSON should serialise");
    let record_output_json =
        record_discovery_internal(&record_input).expect("Record should return JSON");
    let record_output: serde_json::Value =
        serde_json::from_str(&record_output_json).expect("Record output should be valid JSON");
    assert_eq!(
        record_output["success"], true,
        "Failed to record discovery data: {:?}",
        record_output["error"]
    );

    let parquet_file = temp_path.join(
        record_output["file"]
            .as_str()
            .expect("file should be a string"),
    );

    // Analyse neurons (GPU)
    let analyze_input = AnalyzeNeuronsInput {
        parquet_file: parquet_file
            .to_str()
            .expect("Parquet path should be valid UTF-8")
            .to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
    };

    let result = neat_ai_discovery::analysis::analyze_neurons(&analyze_input)
        .expect("Analysis should succeed");

    // Requirement: Do NOT return INVERSE candidates. Complement should be IDENTITY-based.
    assert!(
        !result.helpful_neurons.iter().any(|c| c.squash == "INVERSE"),
        "INVERSE candidates should not be returned. Found: {:?}",
        result
            .helpful_neurons
            .iter()
            .filter(|c| c.squash == "INVERSE")
            .map(|c| (
                &c.source_neuron_uuid,
                &c.target_neuron_uuid,
                c.incoming_weight,
                c.outgoing_weight,
                c.bias
            ))
            .collect::<Vec<_>>()
    );

    // Find an IDENTITY candidate that represents complement (up to scale).
    //
    // We avoid over-constraining exact magnitudes because the optimiser is free to trade:
    // - incoming scale
    // - bias scale
    // - outgoing weight
    //
    // while still representing the same correction shape.
    let complement_identity = result.helpful_neurons.iter().find(|c| {
        if c.source_neuron_uuid != "input-0" || c.target_neuron_uuid != "output-0" {
            return false;
        }
        if c.squash != "IDENTITY" {
            return false;
        }

        let incoming = c.incoming_weight;
        let bias = c.bias;
        let outgoing = c.outgoing_weight;

        // Complement shape constraint:
        // incoming*x + bias should be ~0 at x=1, so incoming + bias ≈ 0.
        let complement_shape_ok = (incoming + bias).abs() < 1e-2;

        // Ensure the correction points in the right direction: at x=0, error is positive.
        let correction0 = outgoing * bias;
        let direction_ok = correction0.is_finite() && correction0 > 0.0;

        // Check correlation between correction(x) and (1 - x) over a few points.
        // For a true complement-like candidate, the correlation should be strongly positive.
        let xs: [f32; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let mut corr = 0.0f32;
        {
            let mut mean_a = 0.0f32;
            let mut mean_b = 0.0f32;
            for &x in &xs {
                let a = outgoing * (incoming * x + bias); // candidate correction
                let b = 1.0 - x; // ideal complement shape (scale-free)
                mean_a += a;
                mean_b += b;
            }
            mean_a /= xs.len() as f32;
            mean_b /= xs.len() as f32;

            let mut cov = 0.0f32;
            let mut var_a = 0.0f32;
            let mut var_b = 0.0f32;
            for &x in &xs {
                let a = outgoing * (incoming * x + bias) - mean_a;
                let b = (1.0 - x) - mean_b;
                cov += a * b;
                var_a += a * a;
                var_b += b * b;
            }
            if var_a > 0.0 && var_b > 0.0 {
                corr = cov / (var_a.sqrt() * var_b.sqrt());
            }
        }
        let correlation_ok = corr.is_finite() && corr > 0.99;

        complement_shape_ok && direction_ok && correlation_ok
    });

    assert!(
        complement_identity.is_some(),
        "Expected an IDENTITY candidate that implements complement (1 - x) via bias/incoming. \
         Found candidates: {:?}",
        result
            .helpful_neurons
            .iter()
            .take(25)
            .map(|c| (
                &c.squash,
                &c.source_neuron_uuid,
                &c.target_neuron_uuid,
                c.incoming_weight,
                c.outgoing_weight,
                c.bias
            ))
            .collect::<Vec<_>>()
    );
}
