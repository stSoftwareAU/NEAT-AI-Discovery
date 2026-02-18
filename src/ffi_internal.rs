//! Internal business-logic functions for the FFI layer.
//!
//! These are the Rust-native entry points called by the FFI wrappers in
//! `src/ffi.rs` and by integration tests via `neat_ai_discovery::*_internal`.

use anyhow::Result;

use crate::ffi_types::*;
use crate::{analysis, export, focus, parquet_format, record};

/// Main entry point for recording discovery data
///
/// Takes JSON input and returns JSON output for easy integration with TypeScript/DenoJS
/// Always returns a JSON string, even on error (with success=false)
///
/// This is the internal Rust function. For FFI, use `record_discovery`.
pub fn record_discovery_internal(input_json: &str) -> Result<String> {
    // Parse input JSON - if this fails, return JSON error
    let input: RecordDiscoveryInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Process discovery data - if this fails, return JSON error
    let result = match record::record_discovery_data(&input) {
        Ok(result) => result,
        Err(e) => {
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(e.to_string()),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Success case
    let output = RecordDiscoveryOutput {
        success: true,
        temp_dir: Some(result.temp_dir),
        file: Some(result.file),
        error: None,
    };

    Ok(serde_json::to_string(&output)?)
}

pub fn merge_discovery_parquet_internal(input_json: &str) -> Result<String> {
    let input: MergeParquetInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let output = MergeParquetOutput {
                success: false,
                output_file: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    if input.input_files.is_empty() {
        let output = MergeParquetOutput {
            success: false,
            output_file: None,
            error: Some("No discovery parquet files provided for merge".to_string()),
        };
        return Ok(serde_json::to_string(&output)?);
    }

    match parquet_format::merge_parquet_files(&input.output_file, &input.input_files) {
        Ok(()) => {
            let output = MergeParquetOutput {
                success: true,
                output_file: Some(input.output_file),
                error: None,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let output = MergeParquetOutput {
                success: false,
                output_file: None,
                error: Some(e.to_string()),
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
}

pub fn analyze_parallel_internal(input_json: &str) -> Result<String> {
    let input: AnalyzeParallelInput = match serde_json::from_str::<AnalyzeParallelInput>(input_json)
    {
        Ok(value) => value,
        Err(e) => {
            let output = AnalyzeParallelOutput {
                success: false,
                helpful_synapses: None,
                harmful_synapses: None,
                synapse_diagnostics: None,
                synapse_gpu_used: None,
                synapse_metadata: None,
                helpful_neurons: None,
                synapse_weight_updates: None,
                coordinated_structural_candidates: None,
                candidate_clusters: None,
                neuron_diagnostics: None,
                neuron_gpu_used: None,
                neuron_metadata: None,
                neuron_fingerprints: None,
                fingerprint_cache_hits: None,
                fingerprint_cache_misses: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    let combined_input = build_analyze_all_input_from_parallel(input);

    match analysis::analyze_all(&combined_input) {
        Ok(result) => {
            let synapse = result.synapse;
            let neuron = result.neuron;
            let synapse_weight_updates = synapse.as_ref().and_then(|s| {
                if s.synapse_weight_updates.is_empty() {
                    None
                } else {
                    Some(s.synapse_weight_updates.clone())
                }
            });

            let coordinated_structural_candidates = synapse.as_ref().and_then(|s| {
                if s.coordinated_structural_candidates.is_empty() {
                    None
                } else {
                    Some(s.coordinated_structural_candidates.clone())
                }
            });

            let candidate_clusters = synapse.as_ref().and_then(|s| {
                if s.candidate_clusters.is_empty() {
                    None
                } else {
                    Some(s.candidate_clusters.clone())
                }
            });

            let output = AnalyzeParallelOutput {
                success: true,
                helpful_synapses: synapse.as_ref().map(|s| s.helpful_synapses.clone()),
                harmful_synapses: synapse.as_ref().map(|s| s.harmful_synapses.clone()),
                synapse_diagnostics: synapse
                    .as_ref()
                    .and_then(|s| synapse_diagnostics_json(s.no_candidate_reasons.as_slice())),
                synapse_gpu_used: synapse.as_ref().map(|s| s.gpu_used),
                synapse_metadata: synapse.as_ref().map(|s| SynapseAnalysisMetadataJson {
                    target_value_available: s.metadata.target_value_available,
                    saturation_aware_simulation_used: s.metadata.saturation_aware_simulation_used,
                    candidates_found: s.metadata.candidates_found,
                    candidates_returned: s.metadata.candidates_returned,
                    timed_out: s.metadata.timed_out,
                    completed_focus_neurons: s.metadata.completed_focus_neurons,
                    total_focus_neurons: s.metadata.total_focus_neurons,
                    input_index_min_seen_with_records: s.metadata.input_index_min_seen_with_records,
                    input_index_max_seen_with_records: s.metadata.input_index_max_seen_with_records,
                    timing: s.metadata.timing.as_ref().map(timing_to_json),
                    gpu_info: s.metadata.gpu_info.as_ref().map(gpu_info_to_json),
                    discovery_module_stats: s.metadata.discovery_module_stats.clone(),
                }),
                helpful_neurons: neuron.as_ref().map(|n| n.helpful_neurons.clone()),
                synapse_weight_updates,
                coordinated_structural_candidates,
                candidate_clusters,
                neuron_diagnostics: neuron
                    .as_ref()
                    .and_then(|n| neuron_diagnostics_json(n.no_candidate_reasons.as_slice())),
                neuron_gpu_used: neuron.as_ref().map(|n| n.gpu_used),
                neuron_metadata: neuron.as_ref().map(|n| NeuronAnalysisMetadataJson {
                    candidates_found: n.metadata.candidates_found,
                    candidates_returned: n.metadata.candidates_returned,
                    timed_out: n.metadata.timed_out,
                    completed_focus_neurons: n.metadata.completed_focus_neurons,
                    total_focus_neurons: n.metadata.total_focus_neurons,
                    timing: n.metadata.timing.as_ref().map(timing_to_json),
                    gpu_info: n.metadata.gpu_info.as_ref().map(gpu_info_to_json),
                }),
                neuron_fingerprints: result.neuron_fingerprints,
                fingerprint_cache_hits: if result.fingerprint_cache_hits > 0 {
                    Some(result.fingerprint_cache_hits)
                } else {
                    None
                },
                fingerprint_cache_misses: if result.fingerprint_cache_misses > 0 {
                    Some(result.fingerprint_cache_misses)
                } else {
                    None
                },
                error: None,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let output = AnalyzeParallelOutput {
                success: false,
                helpful_synapses: None,
                harmful_synapses: None,
                synapse_diagnostics: None,
                synapse_gpu_used: None,
                synapse_metadata: None,
                helpful_neurons: None,
                synapse_weight_updates: None,
                coordinated_structural_candidates: None,
                candidate_clusters: None,
                neuron_diagnostics: None,
                neuron_gpu_used: None,
                neuron_metadata: None,
                neuron_fingerprints: None,
                fingerprint_cache_hits: None,
                fingerprint_cache_misses: None,
                error: Some(e.to_string()),
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
}

pub(crate) fn build_analyze_all_input_from_parallel(
    input: AnalyzeParallelInput,
) -> AnalyzeAllInput {
    AnalyzeAllInput {
        parquet_file: input.parquet_file,
        creature: input.creature,
        focus_neurons: input.focus_neurons,
        max_synapse_candidates: input.max_synapse_candidates,
        max_neuron_candidates: input.max_neuron_candidates,
        analysis_deadline_ms: input.analysis_deadline_ms,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: input.random_seed,
        previous_neuron_fingerprints: input.previous_neuron_fingerprints,
    }
}

pub fn check_gpu_available_internal() -> Result<String> {
    let result = analysis::GpuAnalyzer::check_gpu_availability();

    // On macOS, missing GPU is an error (Metal should always work).
    // On Linux, missing GPU gracefully disables discovery (common on headless servers).
    let output = if result.is_error {
        CheckGpuOutput {
            success: false,
            gpu_available: false,
            reason: result.reason,
            error: Some("GPU required but not available".to_string()),
        }
    } else {
        CheckGpuOutput {
            success: true,
            gpu_available: result.available,
            reason: result.reason,
            error: None,
        }
    };
    Ok(serde_json::to_string(&output)?)
}

pub fn get_library_version_internal() -> Result<String> {
    let output = GetVersionOutput {
        success: true,
        version: crate::LIB_VERSION.to_string(),
        error: None,
    };
    Ok(serde_json::to_string(&output)?)
}

pub fn rank_focus_neurons_internal(input_json: &str) -> Result<String> {
    let input: RankFocusNeuronsInput = match serde_json::from_str(input_json) {
        Ok(value) => value,
        Err(e) => {
            let output = RankFocusNeuronsOutput {
                success: false,
                neurons: None,
                removal_candidates: None,
                constant_neuron_removals: None,
                max_output_error: None,
                processed_neurons: None,
                total_neurons: None,
                duration_ms: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    match focus::rank_focus_neurons(
        &input.parquet_file,
        &input.creature,
        input.max_results,
        input.cost_of_growth,
    ) {
        Ok(stats) => {
            let neurons: Vec<RankedNeuronJson> = stats
                .neurons
                .into_iter()
                .map(|neuron| RankedNeuronJson {
                    neuron_uuid: neuron.neuron_uuid,
                    total_error: neuron.total_error,
                    impact: neuron.impact,
                    mean_activation: neuron.mean_activation,
                    activation_weighted_impact: neuron.activation_weighted_impact,
                })
                .collect();
            let removal_candidates: Vec<RemovalCandidateJson> = stats
                .removal_candidates
                .into_iter()
                .map(|c| RemovalCandidateJson {
                    neuron_uuid: c.neuron_uuid,
                    total_error: c.total_error,
                    impact: c.impact,
                    mean_activation: c.mean_activation,
                    activation_weighted_impact: c.activation_weighted_impact,
                    incoming_synapses: c.incoming_synapses,
                    outgoing_synapses: c.outgoing_synapses,
                    removal_savings: c.removal_savings,
                    expected_error_reduction: c.expected_error_reduction,
                    reason: c.reason,
                })
                .collect();
            let output = RankFocusNeuronsOutput {
                success: true,
                neurons: Some(neurons),
                removal_candidates: if removal_candidates.is_empty() {
                    None
                } else {
                    Some(removal_candidates)
                },
                // Issue #306: Return constant neuron removal candidates
                constant_neuron_removals: if stats.constant_neuron_removals.is_empty() {
                    None
                } else {
                    Some(stats.constant_neuron_removals)
                },
                max_output_error: Some(stats.max_output_error),
                processed_neurons: Some(stats.processed_neurons),
                total_neurons: Some(stats.total_neurons),
                duration_ms: Some(stats.duration_ms.min(u64::MAX as u128) as u64),
                error: None,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let output = RankFocusNeuronsOutput {
                success: false,
                neurons: None,
                removal_candidates: None,
                constant_neuron_removals: None,
                max_output_error: None,
                processed_neurons: None,
                total_neurons: None,
                duration_ms: None,
                error: Some(e.to_string()),
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
}

/// Export a visualisation snapshot to JSON for debugging with NEAT-AI-Explore.
///
/// This is an optional debug tool that reads a Parquet recording and creature,
/// then writes a comprehensive JSON snapshot with recorded data, impacts, and
/// reconstruction checks.
pub fn export_visualisation_snapshot_internal(input_json: &str) -> Result<String> {
    let input: ExportVisualisationSnapshotInput = match serde_json::from_str(input_json) {
        Ok(value) => value,
        Err(e) => {
            let output = ExportVisualisationSnapshotOutput {
                success: false,
                out_file: None,
                stats: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    let options = export::ExportOptions {
        include_per_synapse_series: input.include_per_synapse_series,
        include_reconstruction_checks: input.include_reconstruction_checks,
        max_obs: input.max_obs,
        top_k_worst_samples: input.top_k_worst_samples.unwrap_or(20),
    };

    match export::export_visualisation_snapshot(
        &input.parquet_file,
        &input.creature,
        &input.out_file,
        &options,
    ) {
        Ok(stats) => {
            let output = ExportVisualisationSnapshotOutput {
                success: true,
                out_file: Some(input.out_file),
                stats: Some(ExportVisualisationStats {
                    obs_count: stats.obs_count,
                    neuron_count: stats.neuron_count,
                    synapse_count: stats.synapse_count,
                    output_count: stats.output_count,
                }),
                error: None,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let output = ExportVisualisationSnapshotOutput {
                success: false,
                out_file: None,
                stats: None,
                error: Some(e.to_string()),
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
}

/// Returns a calibration summary from discovery history (Issue #605).
///
/// Takes JSON input containing a serialised `DiscoveryHistory` and returns
/// calibration metrics (MAE, bias, calibration factor) per module/candidate type.
pub fn get_calibration_summary_internal(input_json: &str) -> Result<String> {
    let input: CalibrationSummaryInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let output = CalibrationSummaryOutput {
                success: false,
                calibration_summary: vec![],
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    let history: crate::discovery_history::DiscoveryHistory =
        match serde_json::from_str(&input.discovery_history) {
            Ok(h) => h,
            Err(e) => {
                let output = CalibrationSummaryOutput {
                    success: false,
                    calibration_summary: vec![],
                    error: Some(format!("Failed to parse discovery history: {e}")),
                };
                return Ok(serde_json::to_string(&output)?);
            }
        };

    let summary = history.calibration_summary();
    let output = CalibrationSummaryOutput {
        success: true,
        calibration_summary: summary,
        error: None,
    };
    Ok(serde_json::to_string(&output)?)
}

/// Read discovery records from Parquet file for a specific neuron
///
/// Takes JSON input and returns JSON output for easy integration with TypeScript/DenoJS
pub fn read_discovery_records(input_json: &str) -> Result<String> {
    use crate::parquet_format::read_records_from_parquet;
    use crate::types::DiscoverRecord;

    // Parse input JSON
    let input: ReadDiscoveryInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let output = ReadDiscoveryOutput {
                success: false,
                records: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Read records from Parquet
    let records: Vec<DiscoverRecord> =
        match read_records_from_parquet(&input.parquet_file, &input.neuron_uuid) {
            Ok(records) => records,
            Err(e) => {
                let output = ReadDiscoveryOutput {
                    success: false,
                    records: None,
                    error: Some(e.to_string()),
                };
                return Ok(serde_json::to_string(&output)?);
            }
        };

    // Convert to JSON format
    let json_records: Vec<DiscoverRecordJson> = records
        .into_iter()
        .map(|r| DiscoverRecordJson {
            obs_index: r.obs_index,
            neuron_uuid: r.neuron_uuid,
            value: r.value,
            activation: r.activation,
            errors: r.errors,
        })
        .collect();

    let output = ReadDiscoveryOutput {
        success: true,
        records: Some(json_records),
        error: None,
    };

    let json_string = serde_json::to_string(&output)?;

    Ok(json_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet_format::write_records_to_parquet;
    use crate::types::DiscoverRecord;
    use tempfile::tempdir;

    /// Helper macro to skip tests that require GPU when no GPU is available.
    /// This allows tests to pass gracefully in CI environments without GPUs.
    macro_rules! skip_if_no_gpu {
        () => {
            if !analysis::GpuAnalyzer::gpu_is_available() {
                eprintln!("⚠️  Skipping test: GPU not available");
                return;
            }
        };
    }

    /// Safely truncate a UTF-8 string at character boundaries
    ///
    /// Returns a string truncated to at most `max_bytes` bytes, ensuring the
    /// truncation occurs at a valid UTF-8 character boundary to avoid panics.
    fn truncate_utf8_safe(s: &str, max_bytes: usize) -> &str {
        if s.len() <= max_bytes {
            return s;
        }
        // Find the last valid character boundary at or before max_bytes
        // We iterate through character boundaries and keep the last one <= max_bytes
        let mut last_valid_boundary = 0;
        for (idx, _) in s.char_indices() {
            if idx > max_bytes {
                break;
            }
            last_valid_boundary = idx;
        }
        &s[..last_valid_boundary]
    }

    #[test]
    fn test_record_discovery_json_interface() {
        let input = r#"{
            "creature": {
                "neurons": [],
                "synapses": [],
                "input": 2,
                "output": 1
            },
            "training_data": [],
            "temp_dir": ".discovery/test"
        }"#;

        // Should parse without error
        let parsed: RecordDiscoveryInput = serde_json::from_str(input).unwrap();
        assert_eq!(parsed.creature.input, 2);
        assert_eq!(parsed.creature.output, 1);
    }

    #[test]
    fn test_error_message_json_escaping() {
        // Test that error messages with special characters are properly escaped
        let error_messages = vec![
            r#"Error with "quotes""#,
            r#"Error with \backslash"#,
            "Error with\nnewline",
            r#"Error with "quotes" and \backslash and\nnewline"#,
            r#"Error with "multiple" "quotes" and \multiple\backslashes"#,
        ];

        for error_msg in error_messages {
            // Test RecordDiscoveryOutput
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(error_msg.to_string()),
            };
            let json = serde_json::to_string(&output).unwrap();
            // Verify JSON is valid and can be parsed back
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed["success"], false);
            assert_eq!(parsed["error"].as_str(), Some(error_msg));

            // Test ReadDiscoveryOutput
            let read_output = ReadDiscoveryOutput {
                success: false,
                records: None,
                error: Some(error_msg.to_string()),
            };
            let read_json = serde_json::to_string(&read_output).unwrap();
            // Verify JSON is valid and can be parsed back
            let parsed_read: serde_json::Value = serde_json::from_str(&read_json).unwrap();
            assert_eq!(parsed_read["success"], false);
            assert_eq!(parsed_read["error"].as_str(), Some(error_msg));
        }
    }

    #[test]
    fn test_truncate_utf8_safe_ascii() {
        // Test with ASCII (single-byte characters)
        let s = "Hello, World!";
        assert_eq!(truncate_utf8_safe(s, 5), "Hello");
        assert_eq!(truncate_utf8_safe(s, 13), s);
        assert_eq!(truncate_utf8_safe(s, 100), s);
    }

    #[test]
    fn test_truncate_utf8_safe_multibyte() {
        // Test with multi-byte UTF-8 characters (each emoji is 4 bytes)
        let s = "Hello 🦀 World 🌍";
        // "Hello 🦀" is 11 bytes: "Hello " (6) + "🦀" (4) + " W" (2) = 12 bytes
        // But we want to test truncation at byte 11, which should cut before "🦀"
        let truncated = truncate_utf8_safe(s, 11);
        // Should truncate at character boundary, not in middle of emoji
        assert!(truncated.len() <= 11);
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    fn test_truncate_utf8_safe_boundary_at_500() {
        // Test the specific case mentioned in the issue: truncation at byte 500
        // Create a string with multi-byte characters near position 500
        let mut s = String::new();
        for _ in 0..100 {
            s.push('🦀'); // Each emoji is 4 bytes, so 100 emojis = 400 bytes
        }
        s.push_str("Hello"); // Add 5 more bytes = 405 bytes total

        // Truncate at 500 - should return full string since it's < 500
        assert_eq!(truncate_utf8_safe(&s, 500), s.as_str());

        // Truncate at 400 - should cut at character boundary (after 100 emojis)
        let truncated = truncate_utf8_safe(&s, 400);
        assert_eq!(truncated.len(), 400); // Exactly 100 emojis
        assert!(truncated.is_char_boundary(truncated.len()));

        // Truncate at 401 - should include first character of "Hello" (H = 1 byte)
        let truncated2 = truncate_utf8_safe(&s, 401);
        assert_eq!(truncated2.len(), 401);
        assert!(truncated2.is_char_boundary(truncated2.len()));
    }

    #[test]
    fn test_truncate_utf8_safe_empty() {
        let s = "";
        assert_eq!(truncate_utf8_safe(s, 0), "");
        assert_eq!(truncate_utf8_safe(s, 100), "");
    }

    #[test]
    fn test_truncate_utf8_safe_exact_boundary() {
        // Test truncation exactly at a character boundary
        let s = "Hello🦀World";
        // "Hello" = 5 bytes, "🦀" starts at byte 5
        let truncated = truncate_utf8_safe(s, 5);
        assert_eq!(truncated, "Hello");
        assert_eq!(truncated.len(), 5);
    }

    #[test]
    fn check_gpu_available_internal_returns_well_formed_json() {
        let json =
            check_gpu_available_internal().expect("GPU availability probe should return JSON");
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("GPU availability output should be valid JSON");

        assert!(
            value["success"].is_boolean(),
            "success flag should be a boolean"
        );
        assert!(
            value["gpuAvailable"].is_boolean(),
            "gpuAvailable flag should be a boolean"
        );

        // Platform-specific behaviour:
        // - On macOS: success=false when GPU unavailable (error condition)
        // - On Linux: success=true when GPU unavailable (graceful disable)
        // - reason field provides diagnostic information when GPU is unavailable
        if !value["gpuAvailable"].as_bool().unwrap_or(false) {
            assert!(
                value["reason"].is_string(),
                "reason should be provided when GPU is unavailable"
            );
        }
    }

    #[test]
    fn get_library_version_internal_returns_well_formed_json() {
        let json = get_library_version_internal().expect("Version query should return JSON");
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("Version output should be valid JSON");

        assert_eq!(value["success"], true);
        assert_eq!(
            value["version"]
                .as_str()
                .expect("version should be a string"),
            env!("CARGO_PKG_VERSION")
        );
        assert!(value["error"].is_null(), "error should be null on success");
    }

    #[test]
    fn analyze_parallel_internal_returns_combined_payload() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let parquet_file = temp_dir
            .path()
            .join("records.parquet")
            .to_str()
            .expect("temp path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..12u32 {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to persist discovery records");

        let input_json = serde_json::json!({
            "parquetFile": parquet_file,
            "creature": {
                "neurons": [{
                    "uuid": "output-0",
                    "type": "output",
                    "squash": "IDENTITY",
                    "bias": 0.0
                }],
                "synapses": [{
                    "from_uuid": "input-0",
                    "to_uuid": "output-0",
                    "weight": 0.4
                }],
                "input": 1,
                "output": 1
            },
            "focusNeurons": ["output-0"],
            "maxSynapseCandidates": 5,
            "maxNeuronCandidates": 5,
            "requireGpu": false
        })
        .to_string();

        let output_json =
            analyze_parallel_internal(&input_json).expect("parallel analysis should return JSON");
        let output: serde_json::Value =
            serde_json::from_str(&output_json).expect("output should be valid JSON");

        assert_eq!(output["success"], true);
        assert!(
            output["helpfulSynapses"].is_array(),
            "parallel analysis should include helpful synapse array"
        );
        assert!(
            output["helpfulNeurons"].is_array(),
            "parallel analysis should include helpful neuron array"
        );
        assert!(
            output["synapseGpuUsed"].is_boolean(),
            "parallel analysis should report GPU usage for synapses"
        );
        assert!(
            output["neuronGpuUsed"].is_boolean(),
            "parallel analysis should report GPU usage for neurons"
        );
    }

    #[test]
    fn analyze_parallel_threads_random_seed_into_combined_input() {
        let input_json = serde_json::json!({
            "parquetFile": "example.parquet",
            "creature": {
                "neurons": [],
                "synapses": [],
                "input": 1,
                "output": 1
            },
            "focusNeurons": ["output-0"],
            "maxSynapseCandidates": 5,
            "maxNeuronCandidates": 5,
            "analysisDeadlineMs": 1234,
            "randomSeed": 42
        })
        .to_string();

        let parsed: AnalyzeParallelInput =
            serde_json::from_str(&input_json).expect("input JSON should deserialize");
        let combined = build_analyze_all_input_from_parallel(parsed);

        assert_eq!(combined.random_seed, Some(42));
        assert_eq!(combined.analysis_deadline_ms, Some(1234));
    }

    #[test]
    fn coordinated_structural_candidates_are_exposed_via_analyze_parallel_output_shape() {
        // This test is intentionally light-weight and CPU-only.
        //
        // The end-to-end behaviour is covered by the integration test:
        // `tests/coordinated_structural_mercury_digital.rs`.
        let candidate = CoordinatedStructuralCandidateJson {
            operations: vec![
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: "input-0".to_string(),
                    to_neuron_uuid: "output-0".to_string(),
                },
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: "input-1".to_string(),
                    to_neuron_uuid: "output-0".to_string(),
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: "input-1".to_string(),
                    to_neuron_uuid: "output-0".to_string(),
                    weight: 0.1,
                },
            ],
            expected_creature_score_gain: 0.01,
            comment: Some("Example".to_string()),
        };

        let value = serde_json::to_value(&candidate).expect("candidate should serialise");
        assert!(value["operations"].is_array());
        assert!(value["expectedCreatureScoreGain"].is_number());
    }
}
