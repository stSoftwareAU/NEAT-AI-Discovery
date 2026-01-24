//! Tests for confidence intervals in discovery predictions (Issue #194)
//!
//! This module tests the confidence interval functionality that helps callers:
//! - Prioritise high-confidence candidates
//! - Understand prediction uncertainty
//! - Filter out unreliable predictions
//!
//! ## New Fields (Issue #194)
//!
//! Candidates now include:
//! - `predictionConfidence` - Overall confidence score (0.0 to 1.0)
//! - `expectedScoreGainConfidenceInterval` - [lower_bound, upper_bound] for the prediction
//!
//! ## Confidence Factors
//!
//! The confidence score is computed from:
//! 1. **Sample size** - More samples = higher confidence
//! 2. **Source variance** - Higher source variance = more reliable correlation
//! 3. **Model fit** - How well the linear model fits the data (R²)

mod common;

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{analyze_parallel_internal, CreatureJson, NeuronJson};
use tempfile::NamedTempFile;

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Helper to create an output neuron
fn output(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "output".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Build records with varying input activation to avoid constant-source folding.
fn records_with_varying_input(error: f32, sample_count: u32) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for obs_index in 0..sample_count {
        // Vary input activation to ensure we get synapse candidates, not setBias
        let input_activation = if (obs_index % 2) == 0 { 0.3 } else { 0.7 };
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(input_activation),
            input_activation,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![error],
        ));
    }
    records
}

// =============================================================================
// Test: predictionConfidence field should exist on synapse candidates
// =============================================================================

/// Issue #194: Synapse candidates should include `predictionConfidence` field.
///
/// This field shows the overall confidence in the prediction (0.0 to 1.0).
/// Higher values indicate more reliable predictions.
#[test]
fn test_prediction_confidence_field_exists_on_synapse_candidates() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records with error so we get candidates
    let records = records_with_varying_input(0.5, 100);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");

    // Verify the predictionConfidence field IS present in the JSON
    assert!(
        output_json.contains("predictionConfidence"),
        "predictionConfidence field should be present in JSON output. Got: {output_json}"
    );
}

// =============================================================================
// Test: expectedScoreGainConfidenceInterval field should exist
// =============================================================================

/// Issue #194: Synapse candidates should include `expectedScoreGainConfidenceInterval` field.
///
/// This field provides the confidence interval bounds [lower, upper] for the
/// expected score gain prediction.
#[test]
fn test_score_gain_confidence_interval_field_exists_on_synapse_candidates() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = records_with_varying_input(0.5, 100);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");

    // Verify the confidence interval field IS present in the JSON
    assert!(
        output_json.contains("expectedScoreGainConfidenceInterval"),
        "expectedScoreGainConfidenceInterval field should be present in JSON output. Got: {output_json}"
    );
}

// =============================================================================
// Test: predictionConfidence field should exist on neuron candidates
// =============================================================================

/// Issue #194: Neuron candidates should include `predictionConfidence` field.
#[test]
fn test_prediction_confidence_field_exists_on_neuron_candidates() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Need enough samples for neuron analysis to produce candidates
    let records = records_with_varying_input(0.5, 200);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");

    // Parse the output to check if we have neuron candidates
    let output: serde_json::Value = serde_json::from_str(&output_json).unwrap();
    let helpful_neurons = output.get("helpfulNeurons");

    // If we have neuron candidates, they should have predictionConfidence
    if let Some(serde_json::Value::Array(neurons)) = helpful_neurons {
        if !neurons.is_empty() {
            // At least one neuron candidate exists - check for the field
            assert!(
                neurons[0].get("predictionConfidence").is_some(),
                "predictionConfidence field should be present on neuron candidates. Got: {:?}",
                neurons[0]
            );
        }
    }
}

// =============================================================================
// Test: expectedScoreGainConfidenceInterval field should exist on neuron candidates
// =============================================================================

/// Issue #194: Neuron candidates should include `expectedScoreGainConfidenceInterval` field.
#[test]
fn test_score_gain_confidence_interval_field_exists_on_neuron_candidates() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = records_with_varying_input(0.5, 200);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");

    // Parse the output to check if we have neuron candidates
    let output: serde_json::Value = serde_json::from_str(&output_json).unwrap();
    let helpful_neurons = output.get("helpfulNeurons");

    // If we have neuron candidates, they should have expectedScoreGainConfidenceInterval
    if let Some(serde_json::Value::Array(neurons)) = helpful_neurons {
        if !neurons.is_empty() {
            assert!(
                neurons[0].get("expectedScoreGainConfidenceInterval").is_some(),
                "expectedScoreGainConfidenceInterval field should be present on neuron candidates. Got: {:?}",
                neurons[0]
            );
        }
    }
}

// =============================================================================
// Test: predictionConfidence should be between 0 and 1
// =============================================================================

/// Issue #194: predictionConfidence should be a valid probability (0.0 to 1.0).
#[test]
fn test_prediction_confidence_is_valid_probability() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = records_with_varying_input(0.5, 100);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");
    let output: serde_json::Value = serde_json::from_str(&output_json).unwrap();

    // Check synapse candidates
    if let Some(serde_json::Value::Array(synapses)) = output.get("helpfulSynapses") {
        for synapse in synapses {
            if let Some(confidence) = synapse.get("predictionConfidence") {
                let conf_value = confidence
                    .as_f64()
                    .expect("predictionConfidence should be a number");
                assert!(
                    (0.0..=1.0).contains(&conf_value),
                    "predictionConfidence should be between 0 and 1, got {conf_value}"
                );
            }
        }
    }
}

// =============================================================================
// Test: confidence interval lower bound should be <= expected score gain
// =============================================================================

/// Issue #194: The confidence interval lower bound should be <= the point estimate.
#[test]
fn test_confidence_interval_lower_bound_le_expected() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = records_with_varying_input(0.5, 100);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");
    let output: serde_json::Value = serde_json::from_str(&output_json).unwrap();

    // Check synapse candidates
    if let Some(serde_json::Value::Array(synapses)) = output.get("helpfulSynapses") {
        for synapse in synapses {
            if let (Some(interval), Some(expected)) = (
                synapse.get("expectedScoreGainConfidenceInterval"),
                synapse.get("expectedCreatureScoreGain"),
            ) {
                let interval_arr = interval.as_array().expect("interval should be an array");
                assert_eq!(interval_arr.len(), 2, "interval should have 2 elements");

                let lower = interval_arr[0]
                    .as_f64()
                    .expect("lower bound should be a number");
                let expected_val = expected
                    .as_f64()
                    .expect("expected score gain should be a number");

                assert!(
                    lower <= expected_val + 0.0001, // Allow tiny floating point tolerance
                    "Lower bound {lower} should be <= expected score gain {expected_val}"
                );
            }
        }
    }
}

// =============================================================================
// Test: confidence interval upper bound should be >= expected score gain
// =============================================================================

/// Issue #194: The confidence interval upper bound should be >= the point estimate.
#[test]
fn test_confidence_interval_upper_bound_ge_expected() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = records_with_varying_input(0.5, 100);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");
    let output: serde_json::Value = serde_json::from_str(&output_json).unwrap();

    // Check synapse candidates
    if let Some(serde_json::Value::Array(synapses)) = output.get("helpfulSynapses") {
        for synapse in synapses {
            if let (Some(interval), Some(expected)) = (
                synapse.get("expectedScoreGainConfidenceInterval"),
                synapse.get("expectedCreatureScoreGain"),
            ) {
                let interval_arr = interval.as_array().expect("interval should be an array");
                let upper = interval_arr[1]
                    .as_f64()
                    .expect("upper bound should be a number");
                let expected_val = expected
                    .as_f64()
                    .expect("expected score gain should be a number");

                assert!(
                    upper >= expected_val - 0.0001, // Allow tiny floating point tolerance
                    "Upper bound {upper} should be >= expected score gain {expected_val}"
                );
            }
        }
    }
}

// =============================================================================
// Test: higher sample count should increase confidence
// =============================================================================

/// Issue #194: Higher sample counts should generally result in higher confidence.
///
/// This test compares confidence scores between runs with different sample counts.
/// With more samples, predictions become more reliable (higher confidence).
#[test]
fn test_more_samples_higher_confidence() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    // Run with low sample count
    let temp_file_low = NamedTempFile::new().unwrap();
    let file_path_low = temp_file_low.path().to_str().unwrap();
    let records_low = records_with_varying_input(0.5, 20);
    write_records_to_parquet(file_path_low, &records_low).unwrap();

    let input_json_low = serde_json::json!({
        "parquetFile": file_path_low,
        "creature": &creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json_low =
        analyze_parallel_internal(&input_json_low).expect("analysis should succeed");
    let output_low: serde_json::Value = serde_json::from_str(&output_json_low).unwrap();

    // Run with high sample count
    let temp_file_high = NamedTempFile::new().unwrap();
    let file_path_high = temp_file_high.path().to_str().unwrap();
    let records_high = records_with_varying_input(0.5, 500);
    write_records_to_parquet(file_path_high, &records_high).unwrap();

    let input_json_high = serde_json::json!({
        "parquetFile": file_path_high,
        "creature": &creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json_high =
        analyze_parallel_internal(&input_json_high).expect("analysis should succeed");
    let output_high: serde_json::Value = serde_json::from_str(&output_json_high).unwrap();

    // Extract confidence from both runs
    let confidence_low = output_low
        .get("helpfulSynapses")
        .and_then(|s| s.as_array())
        .and_then(|arr| arr.first())
        .and_then(|s| s.get("predictionConfidence"))
        .and_then(|c| c.as_f64());

    let confidence_high = output_high
        .get("helpfulSynapses")
        .and_then(|s| s.as_array())
        .and_then(|arr| arr.first())
        .and_then(|s| s.get("predictionConfidence"))
        .and_then(|c| c.as_f64());

    if let (Some(conf_low), Some(conf_high)) = (confidence_low, confidence_high) {
        assert!(
            conf_high >= conf_low,
            "Higher sample count should have >= confidence: low={conf_low}, high={conf_high}"
        );
    }
}

// =============================================================================
// Test: low variance source should have lower confidence
// =============================================================================

/// Issue #194: Sources with lower variance should have lower confidence.
///
/// When the source neuron's activation doesn't vary much, the correlation
/// signal is weaker and predictions are less reliable.
#[test]
fn test_low_variance_source_lower_confidence() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    // Create records with HIGH source variance
    let temp_file_high_var = NamedTempFile::new().unwrap();
    let file_path_high_var = temp_file_high_var.path().to_str().unwrap();
    let mut records_high_var = Vec::new();
    for obs_index in 0..100u32 {
        // High variance: activation alternates between 0.0 and 1.0
        let input_activation = if (obs_index % 2) == 0 { 0.0 } else { 1.0 };
        records_high_var.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(input_activation),
            input_activation,
            vec![],
        ));
        records_high_var.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![0.5],
        ));
    }
    write_records_to_parquet(file_path_high_var, &records_high_var).unwrap();

    let input_json_high_var = serde_json::json!({
        "parquetFile": file_path_high_var,
        "creature": &creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json_high_var =
        analyze_parallel_internal(&input_json_high_var).expect("analysis should succeed");
    let output_high_var: serde_json::Value = serde_json::from_str(&output_json_high_var).unwrap();

    // Create records with LOW source variance
    let temp_file_low_var = NamedTempFile::new().unwrap();
    let file_path_low_var = temp_file_low_var.path().to_str().unwrap();
    let mut records_low_var = Vec::new();
    for obs_index in 0..100u32 {
        // Low variance: activation barely varies (0.49 to 0.51)
        let input_activation = if (obs_index % 2) == 0 { 0.49 } else { 0.51 };
        records_low_var.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(input_activation),
            input_activation,
            vec![],
        ));
        records_low_var.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![0.5],
        ));
    }
    write_records_to_parquet(file_path_low_var, &records_low_var).unwrap();

    let input_json_low_var = serde_json::json!({
        "parquetFile": file_path_low_var,
        "creature": &creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json_low_var =
        analyze_parallel_internal(&input_json_low_var).expect("analysis should succeed");
    let output_low_var: serde_json::Value = serde_json::from_str(&output_json_low_var).unwrap();

    // Extract confidence from both runs
    let confidence_high_var = output_high_var
        .get("helpfulSynapses")
        .and_then(|s| s.as_array())
        .and_then(|arr| arr.first())
        .and_then(|s| s.get("predictionConfidence"))
        .and_then(|c| c.as_f64());

    let confidence_low_var = output_low_var
        .get("helpfulSynapses")
        .and_then(|s| s.as_array())
        .and_then(|arr| arr.first())
        .and_then(|s| s.get("predictionConfidence"))
        .and_then(|c| c.as_f64());

    // Low variance source should either have no candidates (filtered) or lower confidence
    if let (Some(conf_high), Some(conf_low)) = (confidence_high_var, confidence_low_var) {
        assert!(
            conf_high >= conf_low,
            "High variance source should have >= confidence: high_var={conf_high}, low_var={conf_low}"
        );
    }
}

// =============================================================================
// Test: confidence interval width should be inversely related to sample count
// =============================================================================

/// Issue #194: Confidence interval width should decrease with more samples.
///
/// More samples = narrower confidence interval (more precise estimate).
#[test]
fn test_confidence_interval_narrows_with_more_samples() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    // Run with low sample count
    let temp_file_low = NamedTempFile::new().unwrap();
    let file_path_low = temp_file_low.path().to_str().unwrap();
    let records_low = records_with_varying_input(0.5, 20);
    write_records_to_parquet(file_path_low, &records_low).unwrap();

    let input_json_low = serde_json::json!({
        "parquetFile": file_path_low,
        "creature": &creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json_low =
        analyze_parallel_internal(&input_json_low).expect("analysis should succeed");
    let output_low: serde_json::Value = serde_json::from_str(&output_json_low).unwrap();

    // Run with high sample count
    let temp_file_high = NamedTempFile::new().unwrap();
    let file_path_high = temp_file_high.path().to_str().unwrap();
    let records_high = records_with_varying_input(0.5, 500);
    write_records_to_parquet(file_path_high, &records_high).unwrap();

    let input_json_high = serde_json::json!({
        "parquetFile": file_path_high,
        "creature": &creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json_high =
        analyze_parallel_internal(&input_json_high).expect("analysis should succeed");
    let output_high: serde_json::Value = serde_json::from_str(&output_json_high).unwrap();

    // Extract confidence interval width from both runs
    let extract_interval_width = |output: &serde_json::Value| -> Option<f64> {
        output
            .get("helpfulSynapses")
            .and_then(|s| s.as_array())
            .and_then(|arr| arr.first())
            .and_then(|s| s.get("expectedScoreGainConfidenceInterval"))
            .and_then(|ci| ci.as_array())
            .map(|arr| {
                let lower = arr[0].as_f64().unwrap_or(0.0);
                let upper = arr[1].as_f64().unwrap_or(0.0);
                upper - lower
            })
    };

    let width_low = extract_interval_width(&output_low);
    let width_high = extract_interval_width(&output_high);

    if let (Some(w_low), Some(w_high)) = (width_low, width_high) {
        assert!(
            w_high <= w_low + 0.0001, // Allow small tolerance
            "Higher sample count should have narrower CI: low_samples_width={w_low}, high_samples_width={w_high}"
        );
    }
}
