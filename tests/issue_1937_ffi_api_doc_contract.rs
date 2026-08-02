//! Issue #1937 — `docs/FFI_API.md` must only promise what the FFI can honour.
//!
//! The reference documented request fields the library ignored, response keys
//! serialised as absent rather than `null`, `snake_case` spellings of a `camelCase`
//! wire struct, and the wrong home for the exported-symbol list. Each test
//! below pins the corrected contract to observable behaviour of the real types.

use neat_ai_discovery::{AnalyzeParallelInput, CheckGpuOutput, EnvironmentalGatesJson};

const FFI_API: &str = include_str!("../docs/FFI_API.md");

fn payload(extra: &str) -> String {
    format!(
        r#"{{
            "parquetFile": "/tmp/records.parquet",
            "creature": {{
                "neurons": [{{ "uuid": "hidden-0", "type": "hidden", "squash": "IDENTITY" }}],
                "synapses": [],
                "input": 1,
                "output": 1
            }},
            "focusNeurons": ["hidden-0"]{extra}
        }}"#
    )
}

fn parse(extra: &str) -> AnalyzeParallelInput {
    serde_json::from_str(&payload(extra)).expect("payload must deserialise")
}

/// Item 3 — the documented `camelCase` spellings are the ones serde binds; the
/// `snake_case` spellings the doc used to show are silently ignored.
#[test]
fn documented_camel_case_tuning_keys_are_the_ones_serde_binds() {
    let camel = parse(r#", "maxAnalysisMemoryMb": 4096, "analysisDeadlineMs": 90000"#);
    assert_eq!(camel.max_analysis_memory_mb, Some(4096));
    assert_eq!(camel.analysis_deadline_ms, Some(90_000));

    let snake = parse(r#", "max_analysis_memory_mb": 4096, "analysis_deadline_ms": 90000"#);
    assert_eq!(snake.max_analysis_memory_mb, None);
    assert_eq!(snake.analysis_deadline_ms, None);

    assert!(
        !FFI_API.contains("max_analysis_memory_mb") && !FFI_API.contains("analysis_deadline_ms"),
        "FFI_API.md must not spell camelCase wire fields in snake_case"
    );
}

/// Item 1 — the documented phase-gating fields reach the request type instead
/// of being dropped as unknown keys.
#[test]
fn documented_phase_gating_fields_bind_to_the_request_type() {
    let gated = parse(r#", "includeSynapseAnalysis": false, "includeNeuronAnalysis": true"#);
    assert_eq!(gated.include_synapse_analysis, Some(false));
    assert_eq!(gated.include_neuron_analysis, Some(true));

    let absent = parse("");
    assert_eq!(absent.include_synapse_analysis, None);
    assert_eq!(absent.include_neuron_analysis, None);
}

/// Item 4 — omitted optional fields are absent from the wire, never `null`, so
/// the examples must not show `null`.
#[test]
fn omitted_optional_response_fields_are_absent_not_null() {
    let gpu = CheckGpuOutput {
        success: true,
        gpu_available: true,
        reason: None,
        error: None,
        error_kind: None,
        retryable: None,
    };
    let wire = serde_json::to_value(&gpu).expect("CheckGpuOutput must serialise");
    assert!(
        wire.get("reason").is_none(),
        "`reason` must be absent when unset, not null"
    );

    let gates = EnvironmentalGatesJson {
        memory_budget_exceeded: false,
        memory_pressure_cancelled: false,
        cancelled: false,
        environmentally_disabled: None,
    };
    let wire = serde_json::to_value(&gates).expect("EnvironmentalGatesJson must serialise");
    assert!(
        wire.get("environmentallyDisabled").is_none(),
        "`environmentallyDisabled` must be absent when unset, not null"
    );

    assert!(
        !FFI_API.contains(r#""reason": null"#)
            && !FFI_API.contains(r#""environmentallyDisabled": null"#),
        "FFI_API.md examples must not show `null` for skip-if-none fields"
    );
}

/// Item 5 — the exported-symbol list lives under `src/ffi/`, not `src/lib.rs`.
#[test]
fn exported_symbol_list_points_at_the_ffi_module() {
    assert!(
        FFI_API.contains("lives under `src/ffi/`"),
        "FFI_API.md must point at src/ffi/ for the exported-symbol list"
    );
    assert!(
        !FFI_API.contains("symbols lives in `src/lib.rs`"),
        "FFI_API.md must not claim the symbol list lives in src/lib.rs"
    );
}
