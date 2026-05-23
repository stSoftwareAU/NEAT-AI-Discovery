//! Issue #1256 — verify the explicit public API surface from `src/lib.rs`.
//!
//! The crate root re-exports a curated list of FFI types and internal
//! business-logic functions from `ffi_types` and `ffi_internal`. This test
//! imports each name via the `neat_ai_discovery::TypeName` and
//! `neat_ai_discovery::function_internal` paths so that any accidental removal
//! from the explicit re-export lists produces a compile-time failure.
//!
//! If a name is intentionally removed from the public surface, this test
//! must be updated in the same PR.

#![allow(unused_imports)]

// Core FFI types declared in `ffi_types/mod.rs`.
use neat_ai_discovery::{
    CreatureJson, NeuronData, NeuronJson, NeuronStatsJson, SCHEMA_VERSION, SynapseJson,
    TrainingRecord,
};

// Candidate types.
use neat_ai_discovery::{
    CandidateNeuronJson, CandidateSynapseJson, CoordinatedStructuralCandidateJson,
    CoordinatedStructuralOpJson, RankedNeuronJson, RemovalCandidateJson,
    SynapseWeightUpdateCandidateJson,
};

// Cleanup request/response types.
use neat_ai_discovery::{
    CleanOrphanedDirsInput, CleanOrphanedDirsOutput, CleanupDiscoveryDirInput,
    CleanupDiscoveryDirOutput,
};

// Error classification.
use neat_ai_discovery::{
    DiscoveryError, DiscoveryErrorKind, classify_anyhow_error, classify_error, classify_panic,
    error_fields, error_fields_from_anyhow, no_error_fields, panic_error_fields,
};

// Forward-only validation helper.
use neat_ai_discovery::validate_forward_only_synapses;

// Request types.
use neat_ai_discovery::{
    AnalyzeAllInput, AnalyzeNeuronsInput, AnalyzeParallelInput, AnalyzeSynapsesInput,
    CalibrationSummaryInput, ExportVisualisationSnapshotInput, MergeParquetInput,
    RankFocusNeuronsInput, ReadDiscoveryInput, RecordDiscoveryInput,
};

// Response types (top-level).
use neat_ai_discovery::{GetVersionOutput, RankFocusNeuronsOutput, RecordDiscoveryOutput};

// Response types (analysis).
use neat_ai_discovery::{
    AcceptanceRateJson, AnalyzeParallelOutput, CalibrationMissEntryJson, DiversityMetricJson,
    McmcDiagnosticsJson, NeuronAnalysisMetadataJson, NeuronDiagnosticDetailJson,
    NeuronDiagnosticJson, NeuronDiagnosticReasonJson, ProposalQualityJson,
    SynapseAnalysisMetadataJson, SynapseDiagnosticDetailJson, SynapseDiagnosticJson,
    SynapseDiagnosticReasonJson,
};

// Response types (export / parquet).
use neat_ai_discovery::{
    CalibrationSummaryOutput, DiscoverRecordJson, ExportVisualisationSnapshotOutput,
    ExportVisualisationStats, MergeParquetOutput, ReadDiscoveryOutput,
};

// Response types (GPU / timing).
use neat_ai_discovery::{
    AnalysisTimingJson, CheckGpuOutput, CpuTimingBreakdownJson, GpuAdapterInfoJson,
    GpuTimingBreakdownJson, ShaderTimingJson,
};

// Session types.
use neat_ai_discovery::{
    AppendRecordsInput, AppendRecordsOutput, CancelSessionInput, CancelSessionOutput,
    FinishSessionInput, FinishSessionOutput, StartSessionInput, StartSessionOutput,
    StreamingObservation,
};

// Internal business-logic entry points (used by integration tests via
// `neat_ai_discovery::*_internal`).
use neat_ai_discovery::{
    analyze_parallel_internal, check_gpu_available_internal,
    export_visualisation_snapshot_internal, get_calibration_summary_internal,
    get_library_version_internal, merge_discovery_parquet_internal, rank_focus_neurons_internal,
    read_discovery_records, record_discovery_internal,
};

#[test]
fn schema_version_re_exported_at_crate_root() {
    // Exercise one re-exported constant so the test actually runs the code.
    assert!(!SCHEMA_VERSION.is_empty());
}

#[test]
fn creature_json_constructible_via_crate_root_path() {
    let creature = CreatureJson {
        neurons: Vec::new(),
        synapses: Vec::new(),
        input: 1,
        output: 1,
    };
    assert_eq!(creature.input, 1);
    assert_eq!(creature.output, 1);
}

#[test]
fn classify_error_re_exported_at_crate_root() {
    // The function returns a DiscoveryErrorKind; we only need a smoke check
    // that the re-export compiles and the function is callable.
    let kind = classify_error("file not found");
    let _ = format!("{kind:?}");
}
