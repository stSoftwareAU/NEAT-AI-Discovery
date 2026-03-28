//! Issue #874 — Backward compatibility test for module splits.
//!
//! Verifies that all public types remain accessible via the same import paths
//! after splitting shared.rs, observability.rs, and responses.rs into submodules.

// Allow unused imports — these verify backward-compatible import paths compile.
#![allow(unused_imports)]

// =============================================================================
// analysis::shared re-exports
// =============================================================================

use neat_ai_discovery::analysis::shared::{
    AnalysisTiming, AnalyzeAllResult, AnalyzeNeuronsResult, AnalyzeSynapsesResult,
    CpuTimingBreakdown, GpuAdapterInfo, GpuDeviceType, GpuTimingBreakdown, NeuronAnalysisMetadata,
    NeuronNoCandidateDetail, NeuronNoCandidateReason, NeuronNoCandidateSummary, ShaderTiming,
    SynapseAnalysisMetadata, SynapseNoCandidateDetail, SynapseNoCandidateReason,
    SynapseNoCandidateSummary, TimingCategory, TimingCollector, ZeroCopyBufferConfig,
};

// =============================================================================
// observability re-exports
// =============================================================================

use neat_ai_discovery::observability::{
    GpuMetrics, PhaseTimer, ProfileData, ProfileMode, ScopedPhaseTimer, global_gpu_metrics,
    gpu_metrics_enabled, init_tracing, profile_mode, timing_enabled,
};

// =============================================================================
// ffi_types::responses re-exports (via crate-level wildcard)
// =============================================================================

use neat_ai_discovery::{
    AnalyzeParallelOutput, CalibrationSummaryOutput, CheckGpuOutput,
    ExportVisualisationSnapshotOutput, GetVersionOutput, MergeParquetOutput,
    NeuronAnalysisMetadataJson, RankFocusNeuronsOutput, ReadDiscoveryOutput, RecordDiscoveryOutput,
    SynapseAnalysisMetadataJson,
};

// GPU timing JSON types
use neat_ai_discovery::{
    AnalysisTimingJson, CpuTimingBreakdownJson, GpuAdapterInfoJson, GpuTimingBreakdownJson,
    ShaderTimingJson,
};

#[test]
fn shared_timing_types_accessible() {
    let timing = AnalysisTiming::default();
    assert_eq!(timing.total_analysis_ms, 0.0);

    let gpu_breakdown = GpuTimingBreakdown::default();
    assert!(gpu_breakdown.shader_timings.is_empty());

    let cpu_breakdown = CpuTimingBreakdown::default();
    assert_eq!(cpu_breakdown.sample_building_ms, 0.0);

    let shader = ShaderTiming::default();
    assert_eq!(shader.calls, 0);
}

#[test]
fn shared_timing_collector_accessible() {
    let collector = TimingCollector::new(false);
    assert!(!collector.is_enabled());
    assert!(collector.finalize().is_none());

    let _category = TimingCategory::Shader;
}

#[test]
fn shared_metadata_types_accessible() {
    let meta = SynapseAnalysisMetadata::default();
    assert_eq!(meta.candidates_found, 0);

    let meta = NeuronAnalysisMetadata::default();
    assert_eq!(meta.candidates_found, 0);
}

#[test]
fn shared_gpu_info_types_accessible() {
    let _device_type = GpuDeviceType::Discrete;
    let config = ZeroCopyBufferConfig::default();
    assert!(!config.enabled());
    assert_eq!(config.buffer_count(), 3);
}

#[test]
fn shared_no_candidate_types_accessible() {
    let _reason = SynapseNoCandidateReason::NoEligibleSources;
    let _reason = NeuronNoCandidateReason::NoEligibleSources;

    let _detail = SynapseNoCandidateDetail {
        source_uuid: None,
        sample_count: None,
        source_record_count: None,
        improved_count: None,
        worsened_count: None,
        expected_improvement: None,
        threshold: None,
        suggested_weight: None,
    };

    let _detail = NeuronNoCandidateDetail {
        source_uuid: None,
        orientation: None,
        sample_count: None,
        improved_count: None,
        worsened_count: None,
        expected_improvement: None,
        threshold: None,
        outgoing_weight: None,
    };
}

#[test]
fn observability_types_accessible() {
    let timer = PhaseTimer::new("test");
    let _ = timer.elapsed_ms();

    let metrics = GpuMetrics::new();
    assert_eq!(metrics.batch_count(), 0);

    let mut profile = ProfileData::new();
    profile.record_phase("test", 42);
    let json = profile.to_json();
    assert!(json["timing"]["phases"]["test"].is_number());

    let _enabled = timing_enabled();
    let _enabled = gpu_metrics_enabled();
    let _mode = profile_mode();
    let _global = global_gpu_metrics();

    // init_tracing is idempotent
    init_tracing();
}

#[test]
fn response_types_accessible() {
    // Verify the types can be constructed — proves they are accessible
    let _output = RecordDiscoveryOutput {
        success: true,
        schema_version: neat_ai_discovery::SCHEMA_VERSION.to_string(),
        temp_dir: None,
        file: None,
        error: None,
        error_kind: None,
        retryable: None,
    };

    let _output = CheckGpuOutput {
        success: true,
        gpu_available: false,
        reason: None,
        error: None,
        error_kind: None,
        retryable: None,
    };

    let _output = GetVersionOutput {
        success: true,
        version: "test".to_string(),
        schema_version: neat_ai_discovery::SCHEMA_VERSION.to_string(),
        error: None,
        error_kind: None,
        retryable: None,
    };

    let _output = MergeParquetOutput {
        success: true,
        output_file: None,
        error: None,
        error_kind: None,
        retryable: None,
    };
}
