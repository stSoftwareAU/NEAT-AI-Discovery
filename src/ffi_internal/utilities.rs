//! Internal business-logic functions for utility FFI entry points.

use anyhow::Result;

use crate::ffi_types::*;
use crate::{export, parquet_format};

pub fn merge_discovery_parquet_internal(input_json: &str) -> Result<String> {
    let input: MergeParquetInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let err_msg = format!("Failed to parse input JSON: {e}");
            let (error_kind, retryable) = error_fields(&err_msg);
            let output = MergeParquetOutput {
                success: false,
                output_file: None,
                error: Some(err_msg),
                error_kind,
                retryable,
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    if input.input_files.is_empty() {
        let err_msg = "No discovery parquet files provided for merge".to_string();
        let (error_kind, retryable) = error_fields(&err_msg);
        let output = MergeParquetOutput {
            success: false,
            output_file: None,
            error: Some(err_msg),
            error_kind,
            retryable,
        };
        return Ok(serde_json::to_string(&output)?);
    }

    match parquet_format::merge_parquet_files(&input.output_file, &input.input_files) {
        Ok(()) => {
            let (error_kind, retryable) = no_error_fields();
            let output = MergeParquetOutput {
                success: true,
                output_file: Some(input.output_file),
                error: None,
                error_kind,
                retryable,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let err_msg = e.to_string();
            let (error_kind, retryable) = error_fields(&err_msg);
            let output = MergeParquetOutput {
                success: false,
                output_file: None,
                error: Some(err_msg),
                error_kind,
                retryable,
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
            let err_msg = format!("Failed to parse input JSON: {e}");
            let (error_kind, retryable) = error_fields(&err_msg);
            let output = ExportVisualisationSnapshotOutput {
                success: false,
                out_file: None,
                stats: None,
                error: Some(err_msg),
                error_kind,
                retryable,
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
            let (error_kind, retryable) = no_error_fields();
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
                error_kind,
                retryable,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let err_msg = e.to_string();
            let (error_kind, retryable) = error_fields(&err_msg);
            let output = ExportVisualisationSnapshotOutput {
                success: false,
                out_file: None,
                stats: None,
                error: Some(err_msg),
                error_kind,
                retryable,
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
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
            let err_msg = format!("Failed to parse input JSON: {e}");
            let (error_kind, retryable) = error_fields(&err_msg);
            let output = ReadDiscoveryOutput {
                success: false,
                records: None,
                error: Some(err_msg),
                error_kind,
                retryable,
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Read records from Parquet
    let records: Vec<DiscoverRecord> =
        match read_records_from_parquet(&input.parquet_file, &input.neuron_uuid) {
            Ok(records) => records,
            Err(e) => {
                let err_msg = e.to_string();
                let (error_kind, retryable) = error_fields(&err_msg);
                let output = ReadDiscoveryOutput {
                    success: false,
                    records: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
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

    let (error_kind, retryable) = no_error_fields();
    let output = ReadDiscoveryOutput {
        success: true,
        records: Some(json_records),
        error: None,
        error_kind,
        retryable,
    };

    let json_string = serde_json::to_string(&output)?;

    Ok(json_string)
}
