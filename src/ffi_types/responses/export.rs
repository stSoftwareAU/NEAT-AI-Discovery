//! Export, merge, read, and calibration output types for the FFI boundary.
//!
//! Contains output structures for visualisation snapshot export, Parquet merge,
//! discovery record reading, and calibration summary operations.

use serde::Serialize;

use crate::ffi_types::DiscoveryErrorKind;

// ============================================================================
// Visualisation Snapshot Export API Types
// ============================================================================

/// JSON output from `export_visualisation_snapshot` function
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportVisualisationSnapshotOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<ExportVisualisationStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

/// Statistics from the export operation
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportVisualisationStats {
    pub obs_count: usize,
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub output_count: usize,
}

// ============================================================================
// Merge Parquet output
// ============================================================================

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MergeParquetOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

// ============================================================================
// Read discovery records types
// ============================================================================

/// JSON output from `read_discovery_records` function
#[derive(Debug, Serialize)]
pub struct ReadDiscoveryOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub records: Option<Vec<DiscoverRecordJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

/// JSON representation of `DiscoverRecord` for serialization
#[derive(Debug, Serialize)]
pub struct DiscoverRecordJson {
    pub obs_index: u32,
    pub neuron_uuid: String,
    pub value: Option<f32>,
    pub activation: f32,
    pub errors: Vec<f32>,
}

// ============================================================================
// Calibration Summary (Issue #605)
// ============================================================================

/// JSON output from `get_calibration_summary` function.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationSummaryOutput {
    pub success: bool,
    /// Calibration summary entries, one per module/candidate-type combination.
    pub calibration_summary: Vec<crate::discovery_history::CalibrationSummaryEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}
