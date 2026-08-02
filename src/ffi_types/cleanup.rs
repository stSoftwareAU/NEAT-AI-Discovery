//! FFI types for discovery directory cleanup (Issue #1100).

use serde::{Deserialize, Serialize};

use super::DiscoveryErrorKind;

/// JSON input for `cleanup_discovery_dir` FFI function.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanupDiscoveryDirInput {
    /// Path to the discovery temp directory to remove.
    pub temp_dir: String,
}

/// JSON output from `cleanup_discovery_dir` FFI function.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanupDiscoveryDirOutput {
    pub success: bool,
    /// Whether the directory was removed or was already gone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub already_gone: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

/// JSON input for `clean_orphaned_discovery_dirs` FFI function.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanOrphanedDirsInput {
    /// Base directory containing discovery temp directories (e.g. `.discovery/`).
    pub base_dir: String,
}

/// JSON output from `clean_orphaned_discovery_dirs` FFI function.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanOrphanedDirsOutput {
    pub success: bool,
    /// Number of orphaned directories removed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub removed: Option<u32>,
    /// Number of directories that were already gone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub already_gone: Option<u32>,
    /// Number of directories a discovery session claimed mid-sweep, so they
    /// were deliberately left in place (Issue #1903).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed: Option<u32>,
    /// Error messages from directories that failed to remove.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub removal_errors: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}
