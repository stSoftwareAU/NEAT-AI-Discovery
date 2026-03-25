//! Memory detection and system requirements checking utilities.
//!
//! This module provides platform-specific memory detection, memory tier classification,
//! and system requirements validation for GPU discovery operations.
//!
//! Extracted from `implementation.rs` as part of Issue #267.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use anyhow::Result;

// =============================================================================
// Constants
// =============================================================================

/// Default GPU batch size for standard hardware.
pub const DEFAULT_GPU_BATCH_SIZE: usize = 512;

/// GPU batch size for high-performance hardware (M4, Pro/Max variants).
pub const HIGH_PERF_GPU_BATCH_SIZE: usize = 1024;

/// GPU batch size for low-memory systems.
pub const LOW_MEMORY_GPU_BATCH_SIZE: usize = 256;

/// Memory threshold for low tier (less than 8GB).
const LOW_MEMORY_THRESHOLD_GB: f64 = 8.0;

/// Memory threshold for standard tier (8-16GB).
const STANDARD_MEMORY_THRESHOLD_GB: f64 = 16.0;

/// Minimum total system memory required for discovery (4GB).
const MINIMUM_TOTAL_MEMORY_GB: f64 = 4.0;

/// Minimum available memory required for discovery.
///
/// Platform-specific thresholds:
/// - **macOS (0.5GB)**: macOS aggressively caches files, so "available" memory appears low.
///   The kernel can quickly reclaim this cached memory when needed. Apple Silicon also has
///   unified memory where GPU shares system RAM.
/// - **Linux (1GB)**: Standard threshold for headless servers without aggressive file caching.
///
/// Issue #326: GPU detection failed on Mac due to overly strict memory check.
#[cfg(target_os = "macos")]
const MINIMUM_AVAILABLE_MEMORY_GB: f64 = 0.5;

#[cfg(not(target_os = "macos"))]
const MINIMUM_AVAILABLE_MEMORY_GB: f64 = 1.0;

// =============================================================================
// Memory Tier Classification
// =============================================================================

/// Memory tier for adaptive configuration.
///
/// Used to adjust GPU batch sizes and work queue capacities based on available
/// system memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// Less than 8GB available - use conservative settings
    Low,
    /// 8-16GB available - use standard settings
    Standard,
    /// More than 16GB available - use aggressive settings
    High,
}

/// Categorise available memory into a tier.
///
/// This is a pure function for testing. Use `detect_memory_tier()` for
/// production code which caches the result.
#[inline]
pub fn categorise_memory_tier(available_bytes: f64) -> MemoryTier {
    let available_gb = available_bytes / (1024.0 * 1024.0 * 1024.0);
    if available_gb < LOW_MEMORY_THRESHOLD_GB {
        MemoryTier::Low
    } else if available_gb < STANDARD_MEMORY_THRESHOLD_GB {
        MemoryTier::Standard
    } else {
        MemoryTier::High
    }
}

/// Detect available system memory and categorise into tiers.
///
/// The result is cached for the lifetime of the process.
pub fn detect_memory_tier() -> MemoryTier {
    use std::sync::OnceLock;
    static TIER: OnceLock<MemoryTier> = OnceLock::new();

    *TIER.get_or_init(|| {
        let (available, total) = get_memory_info();
        let available_gb = available as f64 / (1024.0 * 1024.0 * 1024.0);
        let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);

        let memory_tier = categorise_memory_tier(available as f64);

        // Log memory info once
        let tier_str = match memory_tier {
            MemoryTier::Low => "low",
            MemoryTier::Standard => "standard",
            MemoryTier::High => "high",
        };
        tracing::info!(
            available_gb = format_args!("{available_gb:.1}"),
            total_gb = format_args!("{total_gb:.1}"),
            tier = tier_str,
            "Memory detected"
        );

        memory_tier
    })
}

// =============================================================================
// Platform-Specific Memory Detection
// =============================================================================

/// Get memory information from the OS.
/// Returns (`available_bytes`, `total_bytes`).
#[cfg(target_os = "macos")]
pub fn get_memory_info() -> (u64, u64) {
    use std::process::Command;

    // Get total memory from sysctl
    let total = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
        .unwrap_or(8 * 1024 * 1024 * 1024); // Default 8GB

    // Get page size and free/inactive pages from vm_stat
    let vm_stat = Command::new("vm_stat").output().ok();
    let available = vm_stat.map_or(total / 2, |o| {
        let output = String::from_utf8_lossy(&o.stdout);
        // Parse page size from vm_stat header - handles both Apple Silicon (16KB)
        // and Intel Macs (4KB) correctly
        let page_size = parse_vm_stat_page_size(&output);

        let mut free_pages: u64 = 0;
        let mut inactive_pages: u64 = 0;
        let mut purgeable_pages: u64 = 0;

        for line in output.lines() {
            if line.starts_with("Pages free:") {
                free_pages = parse_vm_stat_line(line);
            } else if line.starts_with("Pages inactive:") {
                inactive_pages = parse_vm_stat_line(line);
            } else if line.starts_with("Pages purgeable:") {
                purgeable_pages = parse_vm_stat_line(line);
            }
        }

        // Available = free + inactive + purgeable (memory that can be reclaimed)
        (free_pages + inactive_pages + purgeable_pages) * page_size
    }); // Default to half of total

    (available, total)
}

/// Parse a page count from a `vm_stat` output line.
/// Example: "Pages free:                              123456." -> 123456
#[cfg(target_os = "macos")]
pub fn parse_vm_stat_line(line: &str) -> u64 {
    line.split(':')
        .nth(1)
        .and_then(|s| s.trim().trim_end_matches('.').parse::<u64>().ok())
        .unwrap_or(0)
}

/// Parse the page size from `vm_stat` output's header line.
/// Example: "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
/// Returns the page size in bytes, or a default based on architecture.
///
/// Apple Silicon uses 16KB pages, Intel Macs use 4KB pages.
/// Parsing dynamically ensures correct memory calculations on both.
#[cfg(target_os = "macos")]
pub fn parse_vm_stat_page_size(output: &str) -> u64 {
    // Default page sizes by architecture
    // Apple Silicon (ARM64): 16384 bytes (16KB)
    // Intel (x86_64): 4096 bytes (4KB)
    #[cfg(target_arch = "aarch64")]
    let default_page_size: u64 = 16384;
    #[cfg(not(target_arch = "aarch64"))]
    let default_page_size: u64 = 4096;

    // Parse from first line: "Mach Virtual Memory Statistics: (page size of XXXX bytes)"
    output
        .lines()
        .next()
        .and_then(|first_line| {
            // Find "page size of " and extract the number before " bytes"
            let marker = "page size of ";
            first_line.find(marker).and_then(|start| {
                let after_marker = &first_line[start + marker.len()..];
                after_marker
                    .split_whitespace()
                    .next()
                    .and_then(|num_str| num_str.parse::<u64>().ok())
            })
        })
        .unwrap_or(default_page_size)
}

/// Get memory information from the OS (Linux version).
/// Works on Ubuntu, AWS Linux (Amazon Linux 2/2023), and other Linux distributions.
/// All Linux systems expose memory info via /proc/meminfo.
#[cfg(target_os = "linux")]
pub fn get_memory_info() -> (u64, u64) {
    use std::fs;

    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mut total: u64 = 8 * 1024 * 1024; // Default 8GB in KB
    let mut available: u64 = 4 * 1024 * 1024; // Default 4GB in KB

    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            // Only update if parsing succeeds - don't overwrite defaults with 0
            if let Some(value) = parse_meminfo_line(line) {
                total = value;
            }
        } else if line.starts_with("MemAvailable:") {
            // Only update if parsing succeeds - don't overwrite defaults with 0
            if let Some(value) = parse_meminfo_line(line) {
                available = value;
            }
        }
    }

    // Convert KB to bytes
    (available * 1024, total * 1024)
}

/// Parse a memory value from a /proc/meminfo line.
/// Returns None if the line is malformed (missing or non-numeric value).
/// Example: "`MemTotal`:       16384000 kB" -> Some(16384000)
#[cfg(target_os = "linux")]
pub fn parse_meminfo_line(line: &str) -> Option<u64> {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
}

/// Fallback for other platforms (Windows, FreeBSD, etc.).
/// Returns conservative defaults since we don't have platform-specific memory detection.
/// GPU discovery will still work via wgpu (DirectX 12 on Windows, Vulkan elsewhere).
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn get_memory_info() -> (u64, u64) {
    // Conservative defaults: 8GB total, 4GB available
    // These are safe values that won't trigger low-memory protections on modern machines
    (4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
}

// =============================================================================
// Parquet Memory Validation
// =============================================================================

/// Validate that there's enough memory to load a parquet file.
///
/// This is a testable pure function. Use `check_memory_for_parquet()` for
/// production code which queries actual file size and system memory.
///
/// # Arguments
/// * `file_size_bytes` - Size of the parquet file in bytes
/// * `available_bytes` - Available system memory in bytes
/// * `total_bytes` - Total system memory in bytes
pub fn validate_parquet_memory_requirements(
    file_size_bytes: u64,
    available_bytes: u64,
    total_bytes: u64,
) -> Result<()> {
    const MEMORY_MULTIPLIER: f64 = 3.0;
    const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIN_HEADROOM_GB: f64 = 1.0; // Keep at least 1GB free for GPU/system
    const MAX_MEMORY_FRACTION: f64 = 0.5; // Don't use more than 50% of total RAM

    let file_size_gb = file_size_bytes as f64 / BYTES_PER_GB;
    let estimated_memory_gb = file_size_gb * MEMORY_MULTIPLIER;

    let available_gb = available_bytes as f64 / BYTES_PER_GB;
    let total_gb = total_bytes as f64 / BYTES_PER_GB;

    let file_size_mb = file_size_bytes as f64 / (1024.0 * 1024.0);
    let estimated_mb = estimated_memory_gb * 1024.0;
    let available_mb = available_gb * 1024.0;

    // Check 1: Do we have enough available memory (with headroom)?
    let memory_needed_with_headroom = estimated_memory_gb + MIN_HEADROOM_GB;
    if memory_needed_with_headroom > available_gb {
        return Err(anyhow::anyhow!(
            "Insufficient memory to load parquet file.\n\
             • Parquet file: {file_size_mb:.0} MB\n\
             • Estimated memory needed: {estimated_mb:.0} MB (parquet decompresses to ~3x in memory)\n\
             • Plus 1 GB headroom for GPU buffers and system\n\
             • Available memory: {available_mb:.0} MB / {total_gb:.1} GB total\n\n\
             Suggestions:\n\
             • Close other applications to free memory\n\
             • Reduce sample rate (e.g., discoverySampleRate: 0.02) to create smaller parquet files\n\
             • Reduce recording time (discoveryRecordTimeOutMinutes)\n\
             • Use a machine with more RAM (16GB+ recommended for large creatures)"
        ));
    }

    // Check 2: Don't exceed 50% of total RAM (prevents system instability)
    let max_allowed_gb = total_gb * MAX_MEMORY_FRACTION;
    if estimated_memory_gb > max_allowed_gb {
        let max_parquet_mb = (max_allowed_gb / MEMORY_MULTIPLIER) * 1024.0;
        return Err(anyhow::anyhow!(
            "Parquet file too large for system memory.\n\
             • Parquet file: {file_size_mb:.0} MB\n\
             • Estimated memory needed: {estimated_mb:.0} MB ({:.0}% of {total_gb:.0} GB total RAM)\n\
             • Maximum safe usage: {:.0} MB (50% of RAM = {max_allowed_gb:.1} GB)\n\
             • Maximum parquet file size for this machine: {max_parquet_mb:.0} MB\n\n\
             Suggestions:\n\
             • Reduce sample rate (e.g., discoverySampleRate: 0.02)\n\
             • Reduce recording time (discoveryRecordTimeOutMinutes)\n\
             • Use a machine with more RAM (16GB+ recommended for large creatures)",
            (estimated_memory_gb / total_gb) * 100.0,
            max_allowed_gb * 1024.0
        ));
    }

    Ok(())
}

/// Check if there's enough memory to load a parquet file.
///
/// Parquet files are compressed; in-memory representation is typically 2-4x larger.
/// This function estimates memory needed and returns an error if insufficient.
///
/// We use conservative checks to avoid memory pressure that causes the system to
/// become unresponsive (stuck with low CPU/GPU, eventually OOM killed):
/// 1. Require 3x file size for in-memory representation
/// 2. Require at least 1GB headroom after loading (for GPU buffers, etc.)
/// 3. Don't use more than 50% of total RAM for parquet data
///
/// This is a public function so it can be used by focus.rs and analysis.rs.
pub fn check_memory_for_parquet(parquet_file: &str) -> Result<()> {
    let file_size_bytes = std::fs::metadata(parquet_file)
        .map(|m| m.len())
        .unwrap_or(0);

    let (available_bytes, total_bytes) = get_memory_info();

    // Log memory usage for large files (helps diagnose issues)
    let file_size_gb = file_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if file_size_gb > 0.5 {
        let file_size_mb = file_size_bytes as f64 / (1024.0 * 1024.0);
        let estimated_mb = file_size_gb * 3.0 * 1024.0;
        let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let available_mb = available_bytes as f64 / (1024.0 * 1024.0);
        let usage_percent = (file_size_gb * 3.0 / total_gb) * 100.0;
        tracing::info!(
            file_size_mb = format_args!("{file_size_mb:.0}"),
            estimated_mb = format_args!("{estimated_mb:.0}"),
            usage_percent = format_args!("{usage_percent:.0}"),
            available_mb = format_args!("{available_mb:.0}"),
            "Loading parquet file"
        );
    }

    validate_parquet_memory_requirements(file_size_bytes, available_bytes, total_bytes)
        .map_err(|e| anyhow::anyhow!("{e}\n• File: {parquet_file}"))
}

// =============================================================================
// System Requirements Checking
// =============================================================================

/// Check if system memory meets minimum requirements for GPU discovery.
///
/// Returns `Some(reason)` if requirements are NOT met.
/// Returns `None` if requirements ARE met.
///
/// # Platform Differences
///
/// On macOS, we use a lower threshold for available memory because:
/// - macOS aggressively caches files in memory (appears as "inactive" or "purgeable")
/// - The kernel can instantly reclaim this cached memory when needed
/// - Apple Silicon has unified memory, so GPU shares system RAM efficiently
///
/// Issue #326: GPU detection failed on Mac due to overly strict memory check.
pub fn check_system_memory_requirements(available_bytes: u64, total_bytes: u64) -> Option<String> {
    let available_gb = available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // Check total system memory
    if total_gb < MINIMUM_TOTAL_MEMORY_GB {
        return Some(format!(
            "Insufficient system memory: {total_gb:.1}GB total (minimum: {MINIMUM_TOTAL_MEMORY_GB}GB required). \
             Discovery is disabled to prevent hangs on memory-constrained machines. \
             Evolution will continue without discovery."
        ));
    }

    // Check available memory - MINIMUM_AVAILABLE_MEMORY_GB is platform-specific
    // (0.5GB on macOS, 1GB on Linux). See constant definition for rationale.
    if available_gb < MINIMUM_AVAILABLE_MEMORY_GB {
        return Some(format!(
            "Insufficient available memory: {available_gb:.2}GB available (minimum: {MINIMUM_AVAILABLE_MEMORY_GB}GB required). \
             Discovery is disabled to prevent hangs from memory pressure. \
             Try closing other applications or reboot to free memory."
        ));
    }

    // Requirements met
    None
}

// =============================================================================
// Memory Pressure Detection (Issue #420)
// =============================================================================

/// Memory pressure level for adaptive behaviour under constrained systems.
///
/// Unlike `MemoryTier` which categorises total available memory, `MemoryPressure`
/// measures how constrained the system is *right now* based on the ratio of
/// available to total memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// More than 30% of total memory available — no special action needed.
    None,
    /// 15-30% of total memory available — reduce cache sizes, use compression.
    Moderate,
    /// 5-15% of total memory available — aggressive eviction, smaller blocks.
    High,
    /// Less than 5% of total memory available — minimal caching, streaming only.
    Critical,
}

/// Categorise current memory pressure based on available and total memory.
///
/// This is a pure function for testability. For production use, call
/// `detect_memory_pressure()` which queries live system memory.
pub fn categorise_memory_pressure(available_bytes: u64, total_bytes: u64) -> MemoryPressure {
    if total_bytes == 0 {
        return MemoryPressure::Critical;
    }
    let ratio = available_bytes as f64 / total_bytes as f64;
    if ratio > 0.30 {
        MemoryPressure::None
    } else if ratio > 0.15 {
        MemoryPressure::Moderate
    } else if ratio > 0.05 {
        MemoryPressure::High
    } else {
        MemoryPressure::Critical
    }
}

/// Detect current memory pressure from live system memory.
pub fn detect_memory_pressure() -> MemoryPressure {
    let (available, total) = get_memory_info();
    categorise_memory_pressure(available, total)
}

// =============================================================================
// GPU Batch Size Management
// =============================================================================

/// Cap GPU batch size based on memory constraints.
///
/// This ensures we don't submit GPU batches that would exceed available memory.
///
/// # Arguments
/// * `configured_batch_size` - The desired batch size
/// * `max_sample_len` - Maximum sample length in elements
/// * `bytes_per_sample` - Bytes per sample element
/// * `max_batch_bytes` - Maximum batch size in bytes
pub fn cap_gpu_batch_size_by_bytes(
    configured_batch_size: usize,
    max_sample_len: usize,
    bytes_per_sample: usize,
    max_batch_bytes: usize,
) -> usize {
    if configured_batch_size == 0 || max_sample_len == 0 || bytes_per_sample == 0 {
        return configured_batch_size.max(1);
    }
    let bytes_per_op = max_sample_len.saturating_mul(bytes_per_sample);
    if bytes_per_op == 0 {
        return configured_batch_size.max(1);
    }

    // How many operations (sources) can we safely submit in one batch chunk?
    let cap = (max_batch_bytes / bytes_per_op).max(1);
    configured_batch_size.min(cap).max(1)
}

/// Get work queue capacity for a given memory tier.
///
/// Lower capacity = more backpressure = less memory usage.
pub fn get_work_queue_capacity_for_tier(tier: MemoryTier) -> usize {
    match tier {
        MemoryTier::Low => 4,      // Aggressive backpressure
        MemoryTier::Standard => 8, // Moderate backpressure
        MemoryTier::High => 16,    // Allow more parallelism
    }
}

/// Get the GPU work queue capacity based on system resources.
///
/// This queries the current memory tier and returns an appropriate capacity.
pub fn get_work_queue_capacity() -> usize {
    get_work_queue_capacity_for_tier(detect_memory_tier())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
