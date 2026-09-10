//! Issue #2065 — `src/ffi_internal/` module docs must state a contract, not
//! restate their own file path.
//!
//! Each of the four public sub-modules previously opened with a header fully
//! derivable from the directory plus the file name ("Internal business-logic
//! functions for <X> FFI entry points."), which told a first-time reader
//! nothing about *why* the `_internal` split exists or what the layer
//! guarantees. These tests pin the two facts the headers must now carry:
//!
//!   1. The module is the internal counterpart of the matching `src/ffi/`
//!      entry points (the split, not the path).
//!   2. Every JSON entry point returns a JSON payload even on failure
//!      (`success: false`) instead of propagating a Rust `Err` — and the
//!      behavioural tests below prove that claim is true, so the doc cannot
//!      drift into fiction.

const ANALYSIS_DOC: &str = include_str!("../../src/ffi_internal/analysis.rs");
const GPU_DOC: &str = include_str!("../../src/ffi_internal/gpu.rs");
const RECORDING_DOC: &str = include_str!("../../src/ffi_internal/recording.rs");
const UTILITIES_DOC: &str = include_str!("../../src/ffi_internal/utilities.rs");

/// The leading `//!` block of a Rust source file, with the markers stripped.
fn module_doc(source: &str) -> String {
    source
        .lines()
        .take_while(|line| line.trim_start().starts_with("//!") || line.trim().is_empty())
        .filter_map(|line| line.trim_start().strip_prefix("//!"))
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

/// The four sub-modules under test, with the paraphrase each must no longer be.
fn modules() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "analysis.rs",
            ANALYSIS_DOC,
            "Internal business-logic functions for analysis FFI entry points.",
        ),
        (
            "gpu.rs",
            GPU_DOC,
            "Internal business-logic functions for GPU probe FFI entry points.",
        ),
        (
            "recording.rs",
            RECORDING_DOC,
            "Internal business-logic functions for recording FFI entry points.",
        ),
        (
            "utilities.rs",
            UTILITIES_DOC,
            "Internal business-logic functions for utility FFI entry points.",
        ),
    ]
}

// ============================================================================
// 1. The headers state a contract, not the file path.
// ============================================================================

#[test]
fn module_docs_are_not_a_bare_path_paraphrase() {
    for (name, source, paraphrase) in modules() {
        let doc = module_doc(source);
        assert!(
            !doc.is_empty(),
            "src/ffi_internal/{name} must carry a module doc (Issue #2065)"
        );
        assert_ne!(
            doc, paraphrase,
            "src/ffi_internal/{name} module doc must state a contract, not restate its own \
             directory and file name (Issue #2065)"
        );
    }
}

#[test]
fn module_docs_name_the_public_ffi_counterpart() {
    for (name, source, _) in modules() {
        let doc = module_doc(source);
        assert!(
            doc.contains("src/ffi/"),
            "src/ffi_internal/{name} module doc must name the `src/ffi/` counterpart it backs \
             (Issue #2065), got: {doc}"
        );
    }
}

#[test]
fn module_docs_state_the_json_always_returned_contract() {
    for (name, source, _) in modules() {
        let doc = module_doc(source);
        assert!(
            doc.contains("success: false") || doc.contains("`success: false`"),
            "src/ffi_internal/{name} module doc must state that failures come back as a \
             `success: false` JSON payload (Issue #2065), got: {doc}"
        );
        assert!(
            doc.contains("Err"),
            "src/ffi_internal/{name} module doc must say a Rust `Err` is not propagated across \
             the boundary for a caller-input failure (Issue #2065), got: {doc}"
        );
    }
}

// ============================================================================
// 2. The documented contract is true — malformed input yields success:false
//    JSON, not a Rust Err.
// ============================================================================

/// Every JSON-taking internal entry point in the four documented sub-modules.
type JsonEntryPoint = fn(&str) -> anyhow::Result<String>;

fn json_entry_points() -> Vec<(&'static str, JsonEntryPoint)> {
    use neat_ai_discovery as lib;
    vec![
        ("record_discovery_internal", lib::record_discovery_internal),
        ("analyze_parallel_internal", lib::analyze_parallel_internal),
        (
            "rank_focus_neurons_internal",
            lib::rank_focus_neurons_internal,
        ),
        (
            "get_calibration_summary_internal",
            lib::get_calibration_summary_internal,
        ),
        (
            "merge_discovery_parquet_internal",
            lib::merge_discovery_parquet_internal,
        ),
        (
            "export_visualisation_snapshot_internal",
            lib::export_visualisation_snapshot_internal,
        ),
        ("read_discovery_records", lib::read_discovery_records),
    ]
}

#[test]
fn malformed_input_returns_success_false_json_not_an_err() {
    for (name, entry_point) in json_entry_points() {
        let json = entry_point("{ not valid json")
            .unwrap_or_else(|e| panic!("{name} must not propagate a Rust Err on bad input: {e}"));
        let value: serde_json::Value = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("{name} must return parseable JSON, got {json}: {e}"));
        assert_eq!(
            value["success"], false,
            "{name} must report success=false on malformed input, got {json}"
        );
        assert!(
            value["error"].is_string(),
            "{name} must carry an error message on malformed input, got {json}"
        );
    }
}

#[test]
fn gpu_probe_module_returns_json_without_caller_input() {
    // The gpu sub-module's entry points take no JSON input; the header's
    // contract is that they still always answer in JSON.
    let json = neat_ai_discovery::get_library_version_internal()
        .expect("get_library_version_internal must return JSON");
    let value: serde_json::Value =
        serde_json::from_str(&json).expect("version output must be valid JSON");
    assert_eq!(value["success"], true);
    assert!(value["version"].is_string());
}
