//! Issue #718: Convert BUG: comments to proper error handling with tracing
//!
//! These tests verify the boundary conditions that previously used "BUG:" string
//! literals in log messages. The conditions are now handled with structured
//! `tracing` events and `debug_assert!` checks.
//!
//! ## Test Coverage
//!
//! - Target analysis: output neuron with only constant upstream neurons
//!   gracefully returns empty results instead of panicking

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::common::{neuron, output, record, synapse};
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_synapses};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson};
use serial_test::serial;
use tempfile::NamedTempFile;

/// Skip test if no GPU available.
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

// =============================================================================
// Target analysis: no eligible upstream neurons returns empty results
// =============================================================================

/// When a target neuron has no eligible upstream sources (e.g., all preceding
/// neurons are constants), the analysis should gracefully return empty results
/// rather than panicking.
///
/// This exercises the code path at `target_analysis/mod.rs` that previously
/// logged a "BUG:" message and now uses `tracing::error!`.
#[test]
#[serial]
fn target_with_only_constant_upstream_returns_empty_results() {
    skip_without_gpu!();

    // Build a creature where the output neuron's only potential upstream source
    // is a constant neuron, which gets filtered out during eligibility checks.
    // This means the output neuron has zero eligible upstream sources.
    let creature = CreatureJson {
        neurons: vec![
            neuron("const-0", "constant", "IDENTITY"),
            output("output-0", "TANH"),
        ],
        synapses: vec![synapse("const-0", "output-0", 1.0)],
        input: 0,
        output: 1,
    };

    // Create records for the output neuron
    let records: Vec<_> = (0..50)
        .map(|i| record("output-0", i, (i as f32) * 0.1, Some((i as f32) * 0.05)))
        .collect();

    let tmp = NamedTempFile::new().expect("failed to create temp file");
    let tmp_path = tmp.path().to_string_lossy().to_string();
    write_records_to_parquet(&tmp_path, &records).expect("failed to write parquet");

    let input = AnalyzeSynapsesInput {
        creature,
        parquet_file: tmp_path,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: Some(60_000),
        random_seed: Some(42),
        module_outcome_tracker: None,
        temperature: 1.0,
    };

    let result = analyze_synapses(&input);
    assert!(
        result.is_ok(),
        "Analysis should succeed even with no eligible upstream neurons"
    );

    let output = result.unwrap();
    // With no eligible sources, there should be no helpful or harmful candidates
    assert!(
        output.helpful_synapses.is_empty(),
        "Expected no helpful candidates when target has no eligible upstream neurons"
    );
}
