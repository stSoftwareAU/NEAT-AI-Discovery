//! NEAT-AI Discovery Library
//!
//! High-performance Rust library for recording neuron activations and errors
//! during the discovery training phase, then scanning recorded data to identify
//! beneficial new synapses/neurons that would reduce error.

// Global tracking allocator — wraps the system allocator to report Rust-side
// memory usage via FFI (Issue #1027). Overhead is a single atomic add/sub per
// allocation, which is negligible for polling every 5-30 seconds. Implemented
// in-repo (Issue #1463) after the unmaintained `cap` crate was dropped.
#[global_allocator]
static ALLOCATOR: tracking_alloc::TrackingAlloc = tracking_alloc::TrackingAlloc::new();

pub mod activations;
pub mod analysis;
pub mod cancellation;
pub mod config;
pub mod debug;
pub mod discovery_cleanup;
pub mod discovery_history;
pub mod export;
pub mod ffi;
mod ffi_internal;
pub mod ffi_types;
pub mod focus;
pub mod intern;
pub mod observability;
pub mod parquet_format;
pub mod record;
pub mod streaming;
pub mod tracking_alloc;
pub mod types;
mod watchdog;

// Re-export the FFI boundary types so that `crate::TypeName` and
// `neat_ai_discovery::TypeName` continue to work for existing callers
// and integration tests.
//
// The list is intentionally explicit (Issue #1256): glob re-exports
// make the crate's public API surface implicit, so any new `pub` item
// added under `ffi_types` would silently leak. New items intended for
// the public surface must be added to this list; otherwise leave them
// `pub(crate)` or accessible only via the `ffi_types::` path.
pub use ffi_types::{
    // Response types — analysis (`ffi_types::responses::analysis`).
    AcceptanceRateJson,
    // Response types — GPU / timing (`ffi_types::responses::gpu`).
    AnalysisTimingJson,
    // Request types (`ffi_types::requests`).
    AnalyzeAllInput,
    AnalyzeNeuronsInput,
    AnalyzeParallelInput,
    AnalyzeParallelOutput,
    AnalyzeSynapsesInput,
    // Session types (`ffi_types::session`).
    AppendRecordsInput,
    AppendRecordsOutput,
    CalibrationMissEntryJson,
    CalibrationSummaryInput,
    // Response types — export / parquet (`ffi_types::responses::export`).
    CalibrationSummaryOutput,
    CancelSessionInput,
    CancelSessionOutput,
    // Candidate types (`ffi_types::candidates`).
    CandidateNeuronJson,
    CandidateSynapseJson,
    CheckGpuOutput,
    // Cleanup request/response types (`ffi_types::cleanup`).
    CleanOrphanedDirsInput,
    CleanOrphanedDirsOutput,
    CleanupDiscoveryDirInput,
    CleanupDiscoveryDirOutput,
    CoordinatedStructuralCandidateJson,
    CoordinatedStructuralOpJson,
    CpuTimingBreakdownJson,
    // Core FFI types declared in `ffi_types/mod.rs`.
    CreatureJson,
    DiscoverRecordJson,
    // Error classification (`ffi_types::error_classification`).
    DiscoveryError,
    DiscoveryErrorKind,
    DiversityMetricJson,
    // Zero-candidate summary types (`ffi_types::responses::analysis`, Issue #1446).
    EnvironmentalGatesJson,
    ExportVisualisationSnapshotInput,
    ExportVisualisationSnapshotOutput,
    ExportVisualisationStats,
    FinishSessionInput,
    FinishSessionOutput,
    // Response types — top-level (`ffi_types::responses`).
    GetVersionOutput,
    GpuAdapterInfoJson,
    GpuTimingBreakdownJson,
    McmcDiagnosticsJson,
    MergeParquetInput,
    MergeParquetOutput,
    NeuronAnalysisMetadataJson,
    NeuronData,
    NeuronDiagnosticDetailJson,
    NeuronDiagnosticJson,
    NeuronDiagnosticReasonJson,
    NeuronJson,
    NeuronStatsJson,
    ProposalQualityJson,
    RankFocusNeuronsInput,
    RankFocusNeuronsOutput,
    RankedNeuronJson,
    ReadDiscoveryInput,
    ReadDiscoveryOutput,
    RecordDiscoveryInput,
    RecordDiscoveryOutput,
    RemovalCandidateJson,
    RemoveNeuronCompensationJson,
    SCHEMA_VERSION,
    ShaderTimingJson,
    StartSessionInput,
    StartSessionOutput,
    StreamingObservation,
    SynapseAnalysisMetadataJson,
    SynapseDiagnosticDetailJson,
    SynapseDiagnosticJson,
    SynapseDiagnosticReasonJson,
    SynapseJson,
    SynapseWeightUpdateCandidateJson,
    TrainingRecord,
    ZeroCandidateSummary,
    build_zero_candidate_summary,
    classify_anyhow_error,
    classify_error,
    classify_panic,
    error_fields,
    error_fields_from_anyhow,
    no_error_fields,
    panic_error_fields,
    // Forward-only validation helper.
    validate_forward_only_synapses,
};

// Re-export the internal business-logic functions so that existing
// integration tests (`neat_ai_discovery::*_internal`) continue to work.
// The list is intentionally explicit (Issue #1256) — see the rationale
// on the `ffi_types` re-export above.
pub use ffi_internal::{
    analyze_parallel_internal, check_gpu_available_internal,
    export_visualisation_snapshot_internal, get_calibration_summary_internal,
    get_library_version_internal, merge_discovery_parquet_internal, rank_focus_neurons_internal,
    read_discovery_records, record_discovery_internal,
};

use std::sync::OnceLock;

// Library version from Cargo.toml
const LIB_VERSION: &str = env!("CARGO_PKG_VERSION");

// Static flag to ensure version is logged only once
static VERSION_LOGGED: OnceLock<()> = OnceLock::new();

/// Log library version on first initialisation.
///
/// Initialises the tracing subscriber (Issue #575) and debug handlers, then
/// logs the compiled library version.
pub(crate) fn log_version_once() {
    VERSION_LOGGED.get_or_init(|| {
        // Initialise structured logging subscriber before any tracing calls.
        observability::init_tracing();

        tracing::info!(
            version = LIB_VERSION,
            "NEAT-AI-Discovery library initialised"
        );

        // Issue #1422: log the effective drought-mitigation config once at
        // startup so an in-progress drought is diagnosable from one log line.
        config::log_effective_drought_mitigation_config();

        // Initialise debug handlers (deadlock detection + kill -3 thread dump)
        debug::init_debug_handlers();
    });
}
