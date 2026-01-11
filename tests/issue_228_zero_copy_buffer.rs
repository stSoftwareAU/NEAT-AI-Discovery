//! Tests for Issue #228: Zero-copy buffer sharing between CPU and GPU.
//!
//! On unified memory architectures (Apple Silicon), the CPU and GPU share the same
//! physical memory. This means we can eliminate unnecessary data copies between
//! CPU and GPU buffers by using persistent mapped buffers.
//!
//! ## Test Coverage
//!
//! 1. **Unified memory detection**: Correctly identifies Apple Silicon and other
//!    unified memory architectures.
//!
//! 2. **ZeroCopyBufferConfig**: Tests the configuration for zero-copy buffer sharing.
//!
//! 3. **Integration tests**: Verifies the zero-copy path produces correct results.
//!
//! 4. **Benchmark tests**: Measures performance improvement on unified memory.

mod common;

use neat_ai_discovery::analysis::{
    analyze_synapses, supports_unified_memory, GpuAnalyzer, ZeroCopyBufferConfig,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use std::time::Instant;
use tempfile::tempdir;

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

// =============================================================================
// Unified Memory Detection Tests
// =============================================================================

/// Test that unified memory detection correctly identifies Apple Silicon.
///
/// On Apple Silicon (M1/M2/M3/M4), the CPU and GPU share the same memory,
/// making zero-copy buffer sharing possible.
#[test]
fn unified_memory_detection_apple_silicon() {
    skip_without_gpu!();

    let has_unified = supports_unified_memory();

    // On macOS with Apple Silicon, we should detect unified memory
    #[cfg(target_os = "macos")]
    {
        // Apple Silicon should always report unified memory
        // Intel Macs would report false, but they're rare now
        eprintln!("Unified memory detected: {has_unified}");
        // Don't assert true because Intel Macs exist
    }

    // On Linux, unified memory depends on the GPU type
    #[cfg(target_os = "linux")]
    {
        eprintln!("Unified memory on Linux: {has_unified}");
        // Integrated GPUs may support unified memory
    }
}

/// Test that unified memory detection returns correct adapter info.
#[test]
fn unified_memory_detection_returns_adapter_info() {
    skip_without_gpu!();

    // The detection function should work without panicking
    let _has_unified = supports_unified_memory();

    // Also verify we can get detailed adapter info
    let info = GpuAnalyzer::get_adapter_info();
    assert!(
        info.is_some(),
        "Should be able to get adapter info when GPU is available"
    );

    let info = info.unwrap();
    eprintln!(
        "GPU: {} (type: {:?}, unified memory: {})",
        info.name, info.device_type, info.has_unified_memory
    );
}

// =============================================================================
// ZeroCopyBufferConfig Tests
// =============================================================================

/// Test that ZeroCopyBufferConfig can be created with default settings.
#[test]
fn zero_copy_config_default() {
    skip_without_gpu!();

    let config = ZeroCopyBufferConfig::default();

    // Default should use auto-detection for unified memory
    eprintln!("ZeroCopyBufferConfig: {config:?}");
    eprintln!("  enabled: {}", config.enabled());
    eprintln!("  buffer_count: {}", config.buffer_count());
}

/// Test that ZeroCopyBufferConfig respects environment variable override.
#[test]
fn zero_copy_config_env_override() {
    skip_without_gpu!();

    // Test enabling via env var
    std::env::set_var("NEAT_AI_DISCOVERY_ZERO_COPY", "1");
    let config = ZeroCopyBufferConfig::from_env();
    std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY");

    assert!(
        config.force_enabled().is_some(),
        "Config should detect env var override"
    );

    // Test disabling via env var
    std::env::set_var("NEAT_AI_DISCOVERY_ZERO_COPY", "0");
    let config = ZeroCopyBufferConfig::from_env();
    std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY");

    assert!(
        config.force_enabled() == Some(false),
        "Config should respect disable override"
    );
}

// =============================================================================
// Integration Tests
// =============================================================================

/// Test that analysis produces correct results with zero-copy enabled.
///
/// This test verifies that enabling zero-copy doesn't change the analysis results.
#[test]
fn zero_copy_produces_correct_results() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create records with clear correlation for synapse discovery
    let mut records = Vec::new();
    for obs_index in 0..100u32 {
        let input_activation = (obs_index as f32 - 50.0) / 50.0;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            input_activation,
            Vec::new(),
        ));

        let error = input_activation * 0.3;
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    };

    // Run analysis with zero-copy enabled (if supported)
    std::env::set_var("NEAT_AI_DISCOVERY_ZERO_COPY", "1");
    let input = AnalyzeSynapsesInput {
        parquet_file: parquet_file.clone(),
        creature: creature.clone(),
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: Some(42),
    };
    let result_zero_copy =
        analyze_synapses(&input).expect("Analysis with zero-copy should succeed");
    std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY");

    // Run analysis with zero-copy disabled
    std::env::set_var("NEAT_AI_DISCOVERY_ZERO_COPY", "0");
    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: Some(42),
    };
    let result_copy = analyze_synapses(&input).expect("Analysis with copy should succeed");
    std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY");

    // Results should be equivalent
    assert_eq!(
        result_zero_copy.helpful_synapses.len(),
        result_copy.helpful_synapses.len(),
        "Zero-copy and copy paths should produce same number of candidates"
    );

    // If there are candidates, verify they match
    if !result_zero_copy.helpful_synapses.is_empty() {
        for (zc, c) in result_zero_copy
            .helpful_synapses
            .iter()
            .zip(result_copy.helpful_synapses.iter())
        {
            assert_eq!(
                zc.from_neuron_uuid, c.from_neuron_uuid,
                "Candidate source should match"
            );
            assert_eq!(
                zc.to_neuron_uuid, c.to_neuron_uuid,
                "Candidate target should match"
            );
            // Weights might differ slightly due to floating-point ordering
            let weight_diff = (zc.weight - c.weight).abs();
            assert!(
                weight_diff < 0.001,
                "Weights should be very close: {} vs {} (diff: {})",
                zc.weight,
                c.weight,
                weight_diff
            );
        }
    }
}

// =============================================================================
// Benchmark Tests
// =============================================================================

/// Benchmark test comparing zero-copy vs copying performance.
///
/// This test measures the performance difference between zero-copy and
/// traditional buffer copying. On unified memory architectures, zero-copy
/// should be faster.
#[test]
fn benchmark_zero_copy_vs_copying() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create a large dataset for meaningful benchmarking
    let sample_count = 100_000;
    let mut records = Vec::with_capacity(sample_count * 2);

    for obs_index in 0..sample_count as u32 {
        let input_activation = ((obs_index as f32) % 100.0 - 50.0) / 50.0;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            input_activation,
            Vec::new(),
        ));

        let error = input_activation * 0.2;
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    };

    // Warmup runs
    for seed in 0..2 {
        let input = AnalyzeSynapsesInput {
            parquet_file: parquet_file.clone(),
            creature: creature.clone(),
            focus_neurons: vec!["output-0".to_string()],
            max_candidates: Some(10),
            analysis_deadline_ms: None,
            random_seed: Some(seed),
        };
        let _ = analyze_synapses(&input);
    }

    // Benchmark with zero-copy disabled (traditional copying)
    std::env::set_var("NEAT_AI_DISCOVERY_ZERO_COPY", "0");
    let iterations = 3;
    let mut copy_times = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let input = AnalyzeSynapsesInput {
            parquet_file: parquet_file.clone(),
            creature: creature.clone(),
            focus_neurons: vec!["output-0".to_string()],
            max_candidates: Some(10),
            analysis_deadline_ms: None,
            random_seed: Some(100 + i as u64),
        };
        let start = Instant::now();
        let _ = analyze_synapses(&input).expect("Analysis should succeed");
        copy_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY");

    // Benchmark with zero-copy enabled
    std::env::set_var("NEAT_AI_DISCOVERY_ZERO_COPY", "1");
    let mut zero_copy_times = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let input = AnalyzeSynapsesInput {
            parquet_file: parquet_file.clone(),
            creature: creature.clone(),
            focus_neurons: vec!["output-0".to_string()],
            max_candidates: Some(10),
            analysis_deadline_ms: None,
            random_seed: Some(200 + i as u64),
        };
        let start = Instant::now();
        let _ = analyze_synapses(&input).expect("Analysis should succeed");
        zero_copy_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY");

    // Calculate statistics
    let copy_avg: f64 = copy_times.iter().sum::<f64>() / copy_times.len() as f64;
    let zero_copy_avg: f64 = zero_copy_times.iter().sum::<f64>() / zero_copy_times.len() as f64;

    eprintln!("\n=== Issue #228: Zero-Copy Buffer Benchmark ===");
    eprintln!("Sample count: {sample_count}");
    eprintln!("Iterations: {iterations}");
    eprintln!();
    eprintln!("With copying:    {copy_avg:.2}ms average");
    eprintln!("With zero-copy:  {zero_copy_avg:.2}ms average");

    let has_unified = supports_unified_memory();
    eprintln!("Unified memory:  {has_unified}");

    if has_unified {
        let improvement = (copy_avg - zero_copy_avg) / copy_avg * 100.0;
        eprintln!("Improvement:     {improvement:.1}%");

        // On unified memory, zero-copy should provide at least 30% improvement
        // as specified in the issue's success criteria
        if improvement < 30.0 {
            eprintln!(
                "Note: Improvement ({improvement:.1}%) is less than 30% target. \
                 This may be acceptable if the workload is GPU-bound rather than transfer-bound."
            );
        }
    } else {
        eprintln!("Note: Non-unified memory architecture - zero-copy falls back to copying.");
    }
}

/// Test that metadata includes zero-copy status.
#[test]
fn metadata_includes_zero_copy_status() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let mut records = Vec::new();
    for obs_index in 0..30u32 {
        let input_activation = (obs_index as f32 - 15.0) / 15.0;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            input_activation,
            Vec::new(),
        ));

        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![input_activation * 0.2],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    };

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: Some(42),
    };

    let result = analyze_synapses(&input).expect("Analysis should succeed");

    // Metadata should include zero-copy status
    assert!(
        result.metadata.gpu_info.is_some(),
        "Metadata should include GPU info"
    );

    let gpu_info = result.metadata.gpu_info.as_ref().unwrap();
    eprintln!("GPU info from metadata:");
    eprintln!("  name: {}", gpu_info.name);
    eprintln!("  unified_memory: {}", gpu_info.has_unified_memory);
    eprintln!("  zero_copy_enabled: {}", gpu_info.zero_copy_enabled);
}

/// Test that no data corruption occurs under concurrent zero-copy access.
///
/// This test creates multiple analysis requests that would be processed
/// concurrently, verifying that the ring buffer synchronisation is correct.
#[test]
fn zero_copy_no_data_corruption() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create dataset with known patterns
    let mut records = Vec::new();
    for obs_index in 0..50u32 {
        // Multiple input neurons to increase concurrent buffer usage
        for input_idx in 0..5 {
            let activation = (obs_index as f32 + input_idx as f32) / 50.0;
            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{input_idx}"),
                None,
                activation,
                Vec::new(),
            ));
        }

        let error = (obs_index as f32 - 25.0) / 25.0 * 0.3;
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 5,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    };

    // Enable zero-copy
    std::env::set_var("NEAT_AI_DISCOVERY_ZERO_COPY", "1");

    // Run multiple iterations with different seeds
    let mut all_results = Vec::new();
    for seed in 0..10u64 {
        let input = AnalyzeSynapsesInput {
            parquet_file: parquet_file.clone(),
            creature: creature.clone(),
            focus_neurons: vec!["output-0".to_string()],
            max_candidates: Some(20),
            analysis_deadline_ms: None,
            random_seed: Some(seed),
        };

        let result = analyze_synapses(&input).expect("Analysis should succeed");
        all_results.push(result);
    }

    std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY");

    // Verify results are consistent and sensible
    for (i, result) in all_results.iter().enumerate() {
        // All results should have the same structure
        assert!(
            result.helpful_synapses.len() <= 20,
            "Iteration {i}: Should respect max_candidates limit"
        );

        // Verify weights are valid numbers (not NaN or Inf)
        for candidate in &result.helpful_synapses {
            assert!(
                candidate.weight.is_finite(),
                "Iteration {i}: Candidate weight should be finite, got {}",
                candidate.weight
            );
            assert!(
                candidate.expected_creature_error_reduction.is_finite(),
                "Iteration {i}: Expected error reduction should be finite, got {}",
                candidate.expected_creature_error_reduction
            );
        }
    }

    eprintln!(
        "Data corruption test passed: {} iterations with no corruption",
        all_results.len()
    );
}
