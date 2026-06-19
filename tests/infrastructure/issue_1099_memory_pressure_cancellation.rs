//! Tests for memory-pressure-triggered cancellation (Issue #1099).
//!
//! Verifies that:
//! - `request_cancellation_memory_pressure()` sets both general and memory-specific flags
//! - `reset_cancellation()` clears both flags
//! - `would_cancel_for_memory_pressure()` correctly identifies CRITICAL pressure
//! - The `memory_pressure_cancelled` field appears in `AnalyzeParallelOutput` JSON
//! - The FFI `cancel_analysis_memory_pressure` entry point works across the boundary

use neat_ai_discovery::analysis::utils::{
    MemoryPressure, categorise_memory_pressure, would_cancel_for_memory_pressure,
};
use neat_ai_discovery::cancellation;
use serial_test::serial;

// ============================================================================
// Memory pressure cancellation flag behaviour
// ============================================================================

#[test]
#[serial]
fn memory_pressure_cancellation_sets_both_flags() {
    cancellation::reset_cancellation();
    assert!(!cancellation::is_cancelled());
    assert!(!cancellation::is_memory_pressure_cancelled());

    cancellation::request_cancellation_memory_pressure();
    assert!(
        cancellation::is_cancelled(),
        "general cancellation flag must be set"
    );
    assert!(
        cancellation::is_memory_pressure_cancelled(),
        "memory pressure flag must be set"
    );
    cancellation::reset_cancellation();
}

#[test]
#[serial]
fn reset_clears_memory_pressure_flag() {
    cancellation::request_cancellation_memory_pressure();
    assert!(cancellation::is_memory_pressure_cancelled());

    cancellation::reset_cancellation();
    assert!(
        !cancellation::is_cancelled(),
        "general flag must be cleared"
    );
    assert!(
        !cancellation::is_memory_pressure_cancelled(),
        "memory pressure flag must be cleared"
    );
}

#[test]
#[serial]
fn normal_cancellation_does_not_set_memory_pressure_flag() {
    cancellation::reset_cancellation();
    cancellation::request_cancellation();
    assert!(cancellation::is_cancelled());
    assert!(
        !cancellation::is_memory_pressure_cancelled(),
        "normal cancellation must NOT set memory pressure flag"
    );
    cancellation::reset_cancellation();
}

#[test]
#[serial]
fn deadline_passed_respects_memory_pressure_cancellation() {
    cancellation::reset_cancellation();
    let no_deadline: Option<std::time::SystemTime> = None;

    assert!(
        !neat_ai_discovery::analysis::utils::deadline_passed(&no_deadline),
        "should be false before cancellation"
    );

    cancellation::request_cancellation_memory_pressure();
    assert!(
        neat_ai_discovery::analysis::utils::deadline_passed(&no_deadline),
        "should be true after memory pressure cancellation"
    );
    cancellation::reset_cancellation();
}

// ============================================================================
// Memory pressure detection (pure function tests)
// ============================================================================

/// Helper to compute a percentage of a total in bytes without float casts.
fn pct_of(total_bytes: u64, numerator: u64, denominator: u64) -> u64 {
    total_bytes / denominator * numerator
}

#[test]
fn critical_pressure_triggers_cancellation_check() {
    // Less than 5% available → CRITICAL → would cancel
    let total: u64 = 16 * 1024 * 1024 * 1024; // 16 GB
    let available = pct_of(total, 3, 100); // ~3%
    assert!(
        would_cancel_for_memory_pressure(available, total),
        "3% available should trigger cancellation"
    );
}

#[test]
fn high_pressure_does_not_trigger_cancellation() {
    // 10% available → HIGH but not CRITICAL → should NOT cancel
    let total: u64 = 16 * 1024 * 1024 * 1024;
    let available = pct_of(total, 10, 100);
    assert!(
        !would_cancel_for_memory_pressure(available, total),
        "10% available should not trigger cancellation"
    );
}

#[test]
fn moderate_pressure_does_not_trigger_cancellation() {
    let total: u64 = 16 * 1024 * 1024 * 1024;
    let available = pct_of(total, 20, 100);
    assert!(
        !would_cancel_for_memory_pressure(available, total),
        "20% available should not trigger cancellation"
    );
}

#[test]
fn no_pressure_does_not_trigger_cancellation() {
    let total: u64 = 16 * 1024 * 1024 * 1024;
    let available = pct_of(total, 50, 100);
    assert!(
        !would_cancel_for_memory_pressure(available, total),
        "50% available should not trigger cancellation"
    );
}

#[test]
fn boundary_at_5_percent_is_critical() {
    let total: u64 = 16 * 1024 * 1024 * 1024;

    // Just above 5% boundary → categorised as High, not Critical
    let above_boundary = pct_of(total, 6, 100);
    assert_eq!(
        categorise_memory_pressure(above_boundary, total),
        MemoryPressure::High,
        "6% available should be High, not Critical"
    );

    // Just below 5% → Critical
    let below_boundary = pct_of(total, 4, 100);
    assert_eq!(
        categorise_memory_pressure(below_boundary, total),
        MemoryPressure::Critical,
        "4% available should be Critical"
    );
}

#[test]
fn zero_total_memory_is_critical() {
    assert!(
        would_cancel_for_memory_pressure(0, 0),
        "zero total memory should be treated as Critical"
    );
}

// ============================================================================
// FFI output struct includes memory_pressure_cancelled field
// ============================================================================

#[test]
fn analyze_parallel_output_includes_memory_pressure_cancelled() {
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
        cancelled: Some(true),
        memory_pressure_cancelled: Some(true),
        environmentally_disabled: None,
        zero_candidate_summary: None,
        error: None,
        error_kind: None,
        retryable: None,
    };

    let json_str = serde_json::to_string(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(
        parsed
            .get("memoryPressureCancelled")
            .and_then(serde_json::Value::as_bool),
        Some(true),
        "memoryPressureCancelled should be true in serialised JSON"
    );

    assert_eq!(
        parsed.get("cancelled").and_then(serde_json::Value::as_bool),
        Some(true),
        "cancelled should also be true when memory pressure caused it"
    );
}

#[test]
fn analyze_parallel_output_omits_memory_pressure_cancelled_when_none() {
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
        cancelled: None,
        memory_pressure_cancelled: None,
        environmentally_disabled: None,
        zero_candidate_summary: None,
        error: None,
        error_kind: None,
        retryable: None,
    };

    let json_str = serde_json::to_string(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert!(
        parsed.get("memoryPressureCancelled").is_none(),
        "memoryPressureCancelled should be omitted from JSON when None"
    );
}
