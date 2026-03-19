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
