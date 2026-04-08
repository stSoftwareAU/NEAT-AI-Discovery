//! Tests for memory budget in the analysis phase (Issue #1028).
//!
//! The analysis phase can accumulate unbounded data structures on
//! memory-constrained machines. This test verifies that a configurable
//! `maxAnalysisMemoryMb` parameter is accepted and that the library
//! reports `memoryBudgetExceeded` when the budget is breached.

// ============================================================================
// Parameter deserialisation — maxAnalysisMemoryMb is accepted
// ============================================================================

#[test]
fn analyze_parallel_input_accepts_memory_budget() {
    let json = r#"{
        "parquetFile": "/tmp/test.parquet",
        "creature": { "neurons": [], "synapses": [], "input": 1, "output": 1 },
        "focusNeurons": [],
        "maxAnalysisMemoryMb": 512
    }"#;
    let input: neat_ai_discovery::AnalyzeParallelInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.max_analysis_memory_mb, Some(512));
}

#[test]
fn analyze_parallel_input_defaults_to_none_when_omitted() {
    let json = r#"{
        "parquetFile": "/tmp/test.parquet",
        "creature": { "neurons": [], "synapses": [], "input": 1, "output": 1 },
        "focusNeurons": []
    }"#;
    let input: neat_ai_discovery::AnalyzeParallelInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.max_analysis_memory_mb, None);
}

#[test]
fn analyze_all_input_accepts_memory_budget() {
    let json = r#"{
        "parquetFile": "/tmp/test.parquet",
        "creature": { "neurons": [], "synapses": [], "input": 1, "output": 1 },
        "focusNeurons": [],
        "maxAnalysisMemoryMb": 1024
    }"#;
    let input: neat_ai_discovery::AnalyzeAllInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.max_analysis_memory_mb, Some(1024));
}

// ============================================================================
// Memory budget check utility — pure function tests
// ============================================================================

#[test]
fn check_memory_budget_returns_false_when_no_budget() {
    let exceeded = neat_ai_discovery::analysis::utils::memory::check_memory_budget_exceeded(
        None,
        500_000_000, // 500 MB allocated
    );
    assert!(!exceeded, "No budget set should never report exceeded");
}

#[test]
fn check_memory_budget_returns_false_when_under_budget() {
    let exceeded = neat_ai_discovery::analysis::utils::memory::check_memory_budget_exceeded(
        Some(1024),  // 1024 MB budget
        500_000_000, // 500 MB allocated
    );
    assert!(
        !exceeded,
        "500 MB used with 1024 MB budget should not be exceeded"
    );
}

#[test]
fn check_memory_budget_returns_true_when_over_budget() {
    let exceeded = neat_ai_discovery::analysis::utils::memory::check_memory_budget_exceeded(
        Some(256),   // 256 MB budget
        300_000_000, // ~300 MB allocated
    );
    assert!(
        exceeded,
        "300 MB used with 256 MB budget should be exceeded"
    );
}

#[test]
fn check_memory_budget_returns_true_when_approaching_budget() {
    // The check triggers at 90% of budget to allow graceful shutdown
    let budget_mb: u64 = 1000;
    let ninety_percent_bytes: u64 = 900 * 1024 * 1024; // 900 MB = 90% of 1000 MB
    let exceeded = neat_ai_discovery::analysis::utils::memory::check_memory_budget_exceeded(
        Some(budget_mb),
        ninety_percent_bytes + 1,
    );
    assert!(exceeded, "90%+ of budget should report exceeded");
}

#[test]
fn check_memory_budget_handles_zero_budget() {
    let exceeded = neat_ai_discovery::analysis::utils::memory::check_memory_budget_exceeded(
        Some(0),
        1, // any allocation exceeds a zero budget
    );
    assert!(exceeded, "Zero budget should always report exceeded");
}

// ============================================================================
// Output serialisation includes memoryBudgetExceeded when true
// ============================================================================

#[test]
fn analyze_parallel_output_serialises_memory_budget_exceeded() {
    // Construct an AnalyzeParallelOutput directly and verify the field serialises.
    let output = neat_ai_discovery::AnalyzeParallelOutput {
        success: true,
        schema_version: neat_ai_discovery::SCHEMA_VERSION.to_string(),
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
        module_outcome_tracker: None,
        memory_budget_exceeded: Some(true),
        error: None,
        error_kind: None,
        retryable: None,
    };

    let json_str = serde_json::to_string(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(
        parsed
            .get("memoryBudgetExceeded")
            .and_then(serde_json::Value::as_bool),
        Some(true),
        "memoryBudgetExceeded should be true in serialised JSON"
    );
}

#[test]
fn analyze_parallel_output_omits_memory_budget_exceeded_when_none() {
    // When memory budget is not set, the field should be omitted from JSON.
    let output = neat_ai_discovery::AnalyzeParallelOutput {
        success: true,
        schema_version: neat_ai_discovery::SCHEMA_VERSION.to_string(),
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
        module_outcome_tracker: None,
        memory_budget_exceeded: None,
        error: None,
        error_kind: None,
        retryable: None,
    };

    let json_str = serde_json::to_string(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert!(
        parsed.get("memoryBudgetExceeded").is_none(),
        "memoryBudgetExceeded should be omitted when None"
    );
}

// ============================================================================
// AnalyzeAllInput also deserialises maxAnalysisMemoryMb
// ============================================================================

#[test]
fn analyze_all_input_propagates_memory_budget_via_json() {
    // Verify that AnalyzeAllInput deserialises the field independently,
    // confirming it is available to the internal orchestration path.
    let json = r#"{
        "parquetFile": "/tmp/test.parquet",
        "creature": { "neurons": [], "synapses": [], "input": 1, "output": 1 },
        "focusNeurons": [],
        "maxAnalysisMemoryMb": 768
    }"#;
    let input: neat_ai_discovery::AnalyzeAllInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.max_analysis_memory_mb, Some(768));
}
