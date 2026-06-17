//! Unit tests for memory module functions.
//!
//! These tests verify the memory detection and system requirements functionality
//! extracted from implementation.rs (Issue #267).

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::*;

// =============================================================================
// Memory Tier Tests
// =============================================================================

#[test]
fn test_memory_tier_low_threshold() {
    // Low tier: less than 8GB available
    assert_eq!(
        categorise_memory_tier(7.9 * 1024.0 * 1024.0 * 1024.0),
        MemoryTier::Low
    );
    assert_eq!(
        categorise_memory_tier(4.0 * 1024.0 * 1024.0 * 1024.0),
        MemoryTier::Low
    );
}

#[test]
fn test_memory_tier_standard_threshold() {
    // Standard tier: 8-16GB available
    assert_eq!(
        categorise_memory_tier(8.0 * 1024.0 * 1024.0 * 1024.0),
        MemoryTier::Standard
    );
    assert_eq!(
        categorise_memory_tier(12.0 * 1024.0 * 1024.0 * 1024.0),
        MemoryTier::Standard
    );
    assert_eq!(
        categorise_memory_tier(15.9 * 1024.0 * 1024.0 * 1024.0),
        MemoryTier::Standard
    );
}

#[test]
fn test_memory_tier_high_threshold() {
    // High tier: more than 16GB available
    assert_eq!(
        categorise_memory_tier(16.0 * 1024.0 * 1024.0 * 1024.0),
        MemoryTier::High
    );
    assert_eq!(
        categorise_memory_tier(32.0 * 1024.0 * 1024.0 * 1024.0),
        MemoryTier::High
    );
}

// =============================================================================
// Available-Memory Pre-load Decision Tests (Issue #1376)
// =============================================================================

const MB: u64 = 1024 * 1024;

#[test]
fn test_preload_fits_when_projection_under_available_minus_margin() {
    // 1 GB projection, 4 GB available, 1 GB margin → usable 3 GB → fits.
    assert!(parquet_preload_fits_available(
        1024 * MB,
        4096 * MB,
        1024 * MB
    ));
}

#[test]
fn test_preload_does_not_fit_when_projection_exceeds_usable() {
    // 3.5 GB projection, 4 GB available, 1 GB margin → usable 3 GB → lazy.
    assert!(!parquet_preload_fits_available(
        3584 * MB,
        4096 * MB,
        1024 * MB
    ));
}

#[test]
fn test_preload_fits_grq13_numbers() {
    // GRQ-13 regression: parquet 531.10 MB → ×3 ≈ 1593 MB projection, with
    // ~2990 MB available and the default 1 GB margin. usable = 2990 − 1024 =
    // 1966 MB ≥ 1593 MB → must pre-load (stay on the fast path), not lazy.
    let projected_bytes = 1593 * MB;
    let available_bytes = 2990 * MB;
    let margin_bytes = 1024 * MB;
    assert!(parquet_preload_fits_available(
        projected_bytes,
        available_bytes,
        margin_bytes
    ));
}

#[test]
fn test_preload_boundary_equal_fits() {
    // Exactly equal to usable budget still fits (<=, not <).
    assert!(parquet_preload_fits_available(
        3072 * MB,
        4096 * MB,
        1024 * MB
    ));
    // One byte over does not.
    assert!(!parquet_preload_fits_available(
        3072 * MB + 1,
        4096 * MB,
        1024 * MB
    ));
}

#[test]
fn test_preload_margin_larger_than_available_saturates_to_lazy() {
    // Margin larger than available → 0 usable bytes → only a 0-byte projection
    // "fits"; any real projection is lazy. No underflow panic.
    assert!(!parquet_preload_fits_available(MB, 512 * MB, 1024 * MB));
    assert!(parquet_preload_fits_available(0, 512 * MB, 1024 * MB));
}

#[test]
fn test_preload_zero_margin_uses_full_available() {
    // With no margin the full available memory is usable.
    assert!(parquet_preload_fits_available(2048 * MB, 2048 * MB, 0));
    assert!(!parquet_preload_fits_available(2048 * MB + 1, 2048 * MB, 0));
}

// =============================================================================
// Parquet Memory Check Tests
// =============================================================================

#[test]
fn test_check_parquet_memory_small_file() {
    // Small files should always pass on reasonable systems
    let result = validate_parquet_memory_requirements(
        100 * 1024 * 1024,       // 100MB file
        8 * 1024 * 1024 * 1024,  // 8GB available
        16 * 1024 * 1024 * 1024, // 16GB total
    );
    assert!(result.is_ok());
}

#[test]
fn test_check_parquet_memory_insufficient_available() {
    // File too large for available memory
    let result = validate_parquet_memory_requirements(
        2 * 1024 * 1024 * 1024, // 2GB file
        1024 * 1024 * 1024,     // 1GB available
        8 * 1024 * 1024 * 1024, // 8GB total
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Insufficient memory"));
}

#[test]
fn test_check_parquet_memory_exceeds_50_percent() {
    // File would use more than 50% of total RAM
    let result = validate_parquet_memory_requirements(
        3 * 1024 * 1024 * 1024,  // 3GB file → 9GB in memory
        12 * 1024 * 1024 * 1024, // 12GB available
        16 * 1024 * 1024 * 1024, // 16GB total → 8GB max allowed
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("too large for system memory"));
}

// =============================================================================
// System Requirements Tests
// =============================================================================

#[test]
fn test_system_requirements_sufficient_memory() {
    let result = check_system_memory_requirements(
        8 * 1024 * 1024 * 1024,  // 8GB available
        16 * 1024 * 1024 * 1024, // 16GB total
    );
    assert!(
        result.is_none(),
        "Should return None when requirements are met"
    );
}

#[test]
fn test_system_requirements_low_total_memory() {
    let result = check_system_memory_requirements(
        2 * 1024 * 1024 * 1024, // 2GB available
        3 * 1024 * 1024 * 1024, // 3GB total (below 4GB minimum)
    );
    assert!(
        result.is_some(),
        "Should return Some when total RAM is too low"
    );
    let reason = result.unwrap();
    assert!(reason.contains("Insufficient system memory"));
}

#[test]
fn test_system_requirements_low_available_memory() {
    // 400MB should fail on all platforms (below both 0.5GB macOS and 1GB Linux thresholds)
    let result = check_system_memory_requirements(
        400 * 1024 * 1024,      // 400MB available (below all minimum thresholds)
        8 * 1024 * 1024 * 1024, // 8GB total
    );
    assert!(
        result.is_some(),
        "Should return Some when available RAM is too low"
    );
    let reason = result.unwrap();
    assert!(reason.contains("Insufficient available memory"));
}

/// Test that macOS uses a lower memory threshold (0.5GB) vs Linux (1GB).
///
/// Issue #326: macOS aggressively caches files, so "available" memory appears low.
/// The kernel can quickly reclaim this memory, so we use a more lenient threshold.
#[cfg(target_os = "macos")]
#[test]
fn test_system_requirements_macos_lower_threshold() {
    // 0.6GB should pass on macOS (above 0.5GB threshold)
    let result = check_system_memory_requirements(
        (0.6 * 1024.0 * 1024.0 * 1024.0) as u64, // 0.6GB available
        8 * 1024 * 1024 * 1024,                  // 8GB total
    );
    assert!(
        result.is_none(),
        "macOS should allow 0.6GB available (threshold is 0.5GB)"
    );
}

/// Test edge case where memory is just above the macOS threshold.
///
/// Issue #326: The original bug showed "1.0GB available" but failed because
/// the actual value was slightly below 1.0GB (e.g., 0.95GB displayed as 1.0GB).
#[cfg(target_os = "macos")]
#[test]
fn test_system_requirements_macos_edge_case() {
    // 0.51GB should pass on macOS (just above 0.5GB threshold)
    let result = check_system_memory_requirements(
        (0.51 * 1024.0 * 1024.0 * 1024.0) as u64, // 0.51GB available
        8 * 1024 * 1024 * 1024,                   // 8GB total
    );
    assert!(
        result.is_none(),
        "macOS should allow 0.51GB available (just above 0.5GB threshold)"
    );

    // 0.49GB should fail on macOS (just below 0.5GB threshold)
    let result = check_system_memory_requirements(
        (0.49 * 1024.0 * 1024.0 * 1024.0) as u64, // 0.49GB available
        8 * 1024 * 1024 * 1024,                   // 8GB total
    );
    assert!(
        result.is_some(),
        "macOS should reject 0.49GB available (below 0.5GB threshold)"
    );
}

/// Test that Linux uses the stricter 1GB threshold.
#[cfg(target_os = "linux")]
#[test]
fn test_system_requirements_linux_stricter_threshold() {
    // 0.6GB should fail on Linux (below 1GB threshold)
    let result = check_system_memory_requirements(
        (0.6 * 1024.0 * 1024.0 * 1024.0) as u64, // 0.6GB available
        8 * 1024 * 1024 * 1024,                  // 8GB total
    );
    assert!(
        result.is_some(),
        "Linux should reject 0.6GB available (threshold is 1GB)"
    );

    // 1.1GB should pass on Linux (above 1GB threshold)
    let result = check_system_memory_requirements(
        (1.1 * 1024.0 * 1024.0 * 1024.0) as u64, // 1.1GB available
        8 * 1024 * 1024 * 1024,                  // 8GB total
    );
    assert!(
        result.is_none(),
        "Linux should allow 1.1GB available (above 1GB threshold)"
    );
}

// =============================================================================
// Configurable available-memory floor (Issue #1420)
// =============================================================================

const GB: f64 = 1024.0 * 1024.0 * 1024.0;

#[test]
fn floor_check_passes_above_explicit_floor() {
    // 1.5GB available with a 1.0GB floor → requirements met.
    let result =
        check_system_memory_requirements_with_floor((1.5 * GB) as u64, 8 * 1024 * 1024 * 1024, 1.0);
    assert!(
        result.is_none(),
        "1.5GB available should pass a 1.0GB floor"
    );
}

#[test]
fn floor_check_fails_below_explicit_floor() {
    // 0.5GB available with a 1.0GB floor → gated off.
    let result =
        check_system_memory_requirements_with_floor((0.5 * GB) as u64, 8 * 1024 * 1024 * 1024, 1.0);
    assert!(
        result.is_some(),
        "0.5GB available should fail a 1.0GB floor"
    );
    assert!(result.unwrap().contains("Insufficient available memory"));
}

#[test]
fn floor_check_boundary_is_inclusive() {
    // Exactly at the floor passes (uses `<`, so equal is allowed).
    let result =
        check_system_memory_requirements_with_floor((GB) as u64, 8 * 1024 * 1024 * 1024, 1.0);
    assert!(
        result.is_none(),
        "Available exactly at the floor should pass"
    );

    // Fractionally below the floor fails.
    let result = check_system_memory_requirements_with_floor(
        (0.99 * GB) as u64,
        8 * 1024 * 1024 * 1024,
        1.0,
    );
    assert!(
        result.is_some(),
        "Available just below the floor should fail"
    );
}

#[test]
fn floor_lowered_lets_small_host_proceed() {
    // Reproduces the Issue #1420 scenario: ~8GB host with ~0.15GB free that the
    // default 1.0GB Linux floor gates off. Lowering the floor to 0.1GB lets
    // discovery proceed; a 0 floor disables the available-memory gate entirely.
    let available = (0.15 * GB) as u64;
    let total = (7.6 * GB) as u64;

    // Default-ish floor (1.0GB) gates it off.
    let gated = check_system_memory_requirements_with_floor(available, total, 1.0);
    assert!(
        gated.is_some(),
        "Default floor should gate off a 0.15GB host"
    );

    // Lowered floor lets it run.
    let allowed = check_system_memory_requirements_with_floor(available, total, 0.1);
    assert!(
        allowed.is_none(),
        "A 0.1GB floor should allow a 0.15GB host"
    );

    // Disabling the gate (floor 0) also allows it.
    let disabled = check_system_memory_requirements_with_floor(available, total, 0.0);
    assert!(
        disabled.is_none(),
        "A 0 floor disables the available-memory gate"
    );
}

#[test]
fn floor_check_total_memory_gate_still_applies() {
    // Total-memory gate is independent of the available-memory floor: a host
    // below the 4GB total minimum is rejected even with a 0 available floor.
    let result =
        check_system_memory_requirements_with_floor((2.0 * GB) as u64, (3.0 * GB) as u64, 0.0);
    assert!(
        result.is_some(),
        "Sub-4GB total RAM should still be rejected"
    );
    assert!(result.unwrap().contains("Insufficient system memory"));
}

// =============================================================================
// macOS vm_stat Parsing Tests
// =============================================================================

#[cfg(target_os = "macos")]
mod macos_tests {
    use super::super::*;

    #[test]
    fn test_parse_vm_stat_line() {
        assert_eq!(
            parse_vm_stat_line("Pages free:                              123456."),
            123456
        );
        assert_eq!(
            parse_vm_stat_line("Pages inactive:                          789012."),
            789012
        );
        assert_eq!(parse_vm_stat_line("Invalid line"), 0);
    }

    #[test]
    fn test_parse_vm_stat_page_size_apple_silicon() {
        let output =
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 1234.";
        assert_eq!(parse_vm_stat_page_size(output), 16384);
    }

    #[test]
    fn test_parse_vm_stat_page_size_intel() {
        let output = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 1234.";
        assert_eq!(parse_vm_stat_page_size(output), 4096);
    }

    #[test]
    fn test_parse_vm_stat_page_size_fallback() {
        // If parsing fails, should return architecture-appropriate default
        let page_size = parse_vm_stat_page_size("Invalid header");
        assert!(page_size == 4096 || page_size == 16384);
    }
}

// =============================================================================
// Linux meminfo Parsing Tests
// =============================================================================

#[cfg(target_os = "linux")]
mod linux_tests {
    use super::super::*;

    #[test]
    fn test_parse_meminfo_line() {
        assert_eq!(
            parse_meminfo_line("MemTotal:       16384000 kB"),
            Some(16384000)
        );
        assert_eq!(
            parse_meminfo_line("MemAvailable:    8192000 kB"),
            Some(8192000)
        );
        assert_eq!(parse_meminfo_line("InvalidLine"), None);
    }
}

// =============================================================================
// GPU Batch Size Tests
// =============================================================================

#[test]
fn test_cap_gpu_batch_size_by_bytes_no_limit() {
    // When plenty of memory, batch size should not be capped
    let result = cap_gpu_batch_size_by_bytes(
        512,         // configured batch size
        1000,        // max sample length
        4,           // bytes per sample
        100_000_000, // 100MB max batch
    );
    assert_eq!(result, 512);
}

#[test]
fn test_cap_gpu_batch_size_by_bytes_capped() {
    // When memory limited, batch size should be capped
    let result = cap_gpu_batch_size_by_bytes(
        512,     // configured batch size
        10000,   // max sample length
        4,       // bytes per sample (40KB per op)
        200_000, // 200KB max batch → ~5 ops max
    );
    assert!(result < 512);
    assert!(result >= 1);
}

#[test]
fn test_cap_gpu_batch_size_by_bytes_zero_values() {
    // Should handle zero values safely
    assert_eq!(cap_gpu_batch_size_by_bytes(0, 1000, 4, 100000), 1);
    assert_eq!(cap_gpu_batch_size_by_bytes(512, 0, 4, 100000), 512);
    assert_eq!(cap_gpu_batch_size_by_bytes(512, 1000, 0, 100000), 512);
}

// =============================================================================
// Work Queue Capacity Tests
// =============================================================================

#[test]
fn test_work_queue_capacity_by_tier() {
    assert_eq!(get_work_queue_capacity_for_tier(MemoryTier::Low), 4);
    assert_eq!(get_work_queue_capacity_for_tier(MemoryTier::Standard), 8);
    assert_eq!(get_work_queue_capacity_for_tier(MemoryTier::High), 16);
}
