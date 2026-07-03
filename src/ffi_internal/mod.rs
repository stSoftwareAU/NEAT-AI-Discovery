//! Internal business-logic functions for the FFI layer.
//!
//! These are the Rust-native entry points called by the FFI wrappers in
//! `src/ffi/` and by integration tests via `neat_ai_discovery::*_internal`.
//!
//! Each sub-module mirrors the corresponding category in `src/ffi/`:
//! - `recording` — discovery data recording
//! - `analysis` — focus neuron ranking, parallel analysis, calibration
//! - `gpu` — GPU availability probe, library version
//! - `utilities` — parquet merge, record reading, visualisation export

// Guard the re-exported public API surface: every `*_internal` function
// re-exported at the crate root must carry rustdoc, so an rlib consumer never
// sees blank documentation for a crate-root entry point (Issue #1485). Scoped
// to this module because the wider crate exposes many undocumented public
// struct fields that are out of scope for this guard.
#![warn(missing_docs)]

mod analysis;
mod gpu;
mod recording;
mod utilities;

pub use analysis::*;
pub use gpu::*;
pub use recording::*;
pub use utilities::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis;
    use crate::ffi_types::*;
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
            let (error_kind, retryable) = error_fields(error_msg);
            let output = RecordDiscoveryOutput {
                success: false,
                schema_version: SCHEMA_VERSION.to_string(),
                temp_dir: None,
                file: None,
                error: Some(error_msg.to_string()),
                error_kind,
                retryable,
            };
            let json = serde_json::to_string(&output).unwrap();
            // Verify JSON is valid and can be parsed back
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed["success"], false);
            assert_eq!(parsed["error"].as_str(), Some(error_msg));

            // Test ReadDiscoveryOutput
            let (error_kind, retryable) = error_fields(error_msg);
            let read_output = ReadDiscoveryOutput {
                success: false,
                records: None,
                error: Some(error_msg.to_string()),
                error_kind,
                retryable,
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
