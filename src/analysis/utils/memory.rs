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

/// Default minimum available memory required for discovery.
///
/// Platform-specific thresholds:
/// - **macOS (0.5GB)**: macOS aggressively caches files, so "available" memory appears low.
///   The kernel can quickly reclaim this cached memory when needed. Apple Silicon also has
///   unified memory where GPU shares system RAM.
/// - **Linux (1GB)**: Standard threshold for headless servers without aggressive file caching.
///
/// This is the *default* floor. Operators can override it at runtime via
/// `NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB` (Issue #1420) — on small-but-capable
/// hosts where the discovery runtime itself already holds most of the RAM, the
/// default would otherwise gate discovery off every pass.
///
/// Issue #326: GPU detection failed on Mac due to overly strict memory check.
#[cfg(target_os = "macos")]
pub const DEFAULT_MIN_AVAILABLE_MEMORY_GB: f64 = 0.5;

#[cfg(not(target_os = "macos"))]
pub const DEFAULT_MIN_AVAILABLE_MEMORY_GB: f64 = 1.0;

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

    // Reclaimable available memory is parsed from vm_stat (Issue #3173).
    let vm_stat = Command::new("vm_stat").output().ok();
    let available = vm_stat.map_or(total / 2, |o| {
        let output = String::from_utf8_lossy(&o.stdout);
        parse_macos_available_bytes(&output, total)
    }); // Default to half of total when vm_stat is unavailable

    (available, total)
}

/// Compute reclaimable available memory (bytes) from raw `vm_stat` output and
/// the total physical memory (Issue #3173).
///
/// macOS keeps a large share of RAM as reclaimable file-cache, purgeable and
/// free pages. The previous `free + inactive + purgeable` figure undercounted
/// true headroom: file-backed cache sitting in the **active** list was treated
/// as unavailable, so a ~14.3 GB pre-load projection on a 24 GB host was forced
/// onto the slow lazy path despite the machine being able to run eager. This
/// counts every genuinely-reclaimable page class:
///
/// - **Pages free** — unused RAM,
/// - **File-backed pages** — clean file cache (active **and** inactive), which
///   the kernel drops on demand without I/O,
/// - **Pages purgeable** — caches the kernel may discard on demand.
///
/// Wired, compressed and dirty anonymous (app) pages are deliberately excluded:
/// reclaiming them needs swap / compression — the very memory pressure the
/// pre-load margin guards against, so counting them would risk the OOM the
/// margin exists to prevent.
///
/// Falls back to `total_bytes / 2` when no reclaimable field can be parsed (an
/// unexpected `vm_stat` format) so the whole machine is never reported as
/// available, and the result is clamped to `total_bytes`.
#[cfg(any(target_os = "macos", test))]
#[must_use]
pub fn parse_macos_available_bytes(vm_stat_output: &str, total_bytes: u64) -> u64 {
    // Parse page size from the header - handles both Apple Silicon (16KB) and
    // Intel Macs (4KB) correctly.
    let page_size = parse_vm_stat_page_size(vm_stat_output);

    let mut free_pages: u64 = 0;
    let mut file_backed_pages: u64 = 0;
    let mut purgeable_pages: u64 = 0;
    let mut parsed_any = false;

    for line in vm_stat_output.lines() {
        if line.starts_with("Pages free:") {
            free_pages = parse_vm_stat_line(line);
            parsed_any = true;
        } else if line.starts_with("File-backed pages:") {
            file_backed_pages = parse_vm_stat_line(line);
            parsed_any = true;
        } else if line.starts_with("Pages purgeable:") {
            purgeable_pages = parse_vm_stat_line(line);
            parsed_any = true;
        }
    }

    if !parsed_any {
        // Unexpected format — be conservative rather than reporting the whole
        // machine as available.
        return total_bytes / 2;
    }

    macos_reclaimable_available_bytes(page_size, free_pages, file_backed_pages, purgeable_pages)
        .min(total_bytes)
}

/// Sum the reclaimable macOS page classes and convert to bytes (Issue #3173).
///
/// Pure arithmetic split out from [`parse_macos_available_bytes`] so the
/// reclaimable-memory accounting can be unit-tested against fixed page counts.
/// Uses saturating arithmetic so implausibly large counts cannot overflow.
#[cfg(any(target_os = "macos", test))]
#[must_use]
pub const fn macos_reclaimable_available_bytes(
    page_size: u64,
    free_pages: u64,
    file_backed_pages: u64,
    purgeable_pages: u64,
) -> u64 {
    free_pages
        .saturating_add(file_backed_pages)
        .saturating_add(purgeable_pages)
        .saturating_mul(page_size)
}

/// Parse a page count from a `vm_stat` output line.
/// Example: "Pages free:                              123456." -> 123456
#[cfg(any(target_os = "macos", test))]
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
#[cfg(any(target_os = "macos", test))]
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

/// Multiplier applied to the parquet file size to estimate the in-memory
/// representation (parquet decompresses to ~3× in memory).
pub const PARQUET_MEMORY_MULTIPLIER: u64 = 3;

/// Estimate the in-memory bytes required to fully pre-load a parquet file
/// (Issue #1172).
///
/// Returns `file_size × PARQUET_MEMORY_MULTIPLIER`. Returns `0` when the
/// file's metadata cannot be read so the caller can treat the projection as
/// unknown rather than aborting.
pub fn estimate_parquet_in_memory_bytes(parquet_file: &str) -> u64 {
    let file_size = std::fs::metadata(parquet_file).map_or(0, |m| m.len());
    file_size.saturating_mul(PARQUET_MEMORY_MULTIPLIER)
}

/// Decide whether a projected parquet pre-load fits within OS-available
/// memory after reserving a safety margin (Issue #1376).
///
/// Returns `true` (pre-load is safe) when
/// `projected_bytes <= available_bytes − margin_bytes`. Uses saturating
/// subtraction so a margin larger than available memory yields `0` usable
/// bytes (lazy) rather than underflowing.
///
/// This is a pure function so the eager-vs-lazy decision can be unit-tested
/// against fixed memory figures (e.g. the GRQ-13 numbers: ~1.6 GB projection
/// with ~3 GB available → fits → pre-load) without sampling live system
/// memory.
#[must_use]
pub const fn parquet_preload_fits_available(
    projected_bytes: u64,
    available_bytes: u64,
    margin_bytes: u64,
) -> bool {
    let usable = available_bytes.saturating_sub(margin_bytes);
    projected_bytes <= usable
}

/// Convert bytes to whole megabytes, rounding up so non-zero sizes always
/// produce at least 1 MB.
pub const fn bytes_to_mb_ceil(bytes: u64) -> u64 {
    const MB: u64 = 1024 * 1024;
    bytes.div_ceil(MB)
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
    let file_size_bytes = std::fs::metadata(parquet_file).map_or(0, |m| m.len());

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
///
/// The available-memory floor is resolved from
/// `NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB` (Issue #1420), falling back to
/// the platform default ([`DEFAULT_MIN_AVAILABLE_MEMORY_GB`]) when unset or
/// invalid. Use [`check_system_memory_requirements_with_floor`] to supply an
/// explicit floor (e.g. in unit tests).
pub fn check_system_memory_requirements(available_bytes: u64, total_bytes: u64) -> Option<String> {
    check_system_memory_requirements_with_floor(
        available_bytes,
        total_bytes,
        crate::config::min_available_memory_gb(),
    )
}

/// Check system memory against an explicit available-memory floor (in GB).
///
/// This is the pure core of [`check_system_memory_requirements`]: it takes the
/// floor as a parameter so the threshold boundary can be unit-tested without
/// mutating process-global environment state.
///
/// Returns `Some(reason)` if requirements are NOT met, `None` if they ARE.
pub fn check_system_memory_requirements_with_floor(
    available_bytes: u64,
    total_bytes: u64,
    min_available_gb: f64,
) -> Option<String> {
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

    // Check available memory against the (possibly operator-overridden) floor.
    // Default is platform-specific (0.5GB macOS, 1GB Linux); see
    // DEFAULT_MIN_AVAILABLE_MEMORY_GB and NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB.
    if available_gb < min_available_gb {
        return Some(format!(
            "Insufficient available memory: {available_gb:.2}GB available (minimum: {min_available_gb}GB required). \
             Discovery is disabled to prevent hangs from memory pressure. \
             Lower the floor via NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB if this host is capable, \
             or close other applications / reboot to free memory."
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
// Memory Budget Checking (Issue #1028)
// =============================================================================

/// Threshold fraction of the memory budget at which we consider it exceeded.
///
/// We trigger at 90% to allow the analysis to return partial results before
/// the OS kills the process.
const MEMORY_BUDGET_THRESHOLD: f64 = 0.9;

/// Check whether the current Rust heap usage exceeds the configured memory budget.
///
/// Returns `true` when `allocated_bytes` reaches 90% of `budget_mb` (converted to
/// bytes). Returns `false` when no budget is set (`budget_mb` is `None`).
///
/// This is a pure function to enable deterministic testing without relying on
/// the global allocator.
pub fn check_memory_budget_exceeded(budget_mb: Option<u64>, allocated_bytes: u64) -> bool {
    let Some(budget) = budget_mb else {
        return false;
    };
    if budget == 0 {
        return allocated_bytes > 0;
    }
    let budget_bytes = budget as f64 * 1024.0 * 1024.0;
    let threshold_bytes = budget_bytes * MEMORY_BUDGET_THRESHOLD;
    allocated_bytes as f64 >= threshold_bytes
}

/// Check memory budget against the live Rust heap allocator (Issue #1028).
///
/// Reads the current allocation from the global tracking allocator and
/// compares it to `budget_mb`. Returns `true` if the budget is approached
/// (90% threshold) or exceeded.
pub fn is_memory_budget_exceeded(budget_mb: Option<u64>) -> bool {
    let allocated = crate::ALLOCATOR.allocated() as u64;
    check_memory_budget_exceeded(budget_mb, allocated)
}

// =============================================================================
// Memory Pressure Cancellation (Issue #1099)
// =============================================================================

/// Check system memory pressure and cancel in-flight analysis if CRITICAL.
///
/// This is called at key phase boundaries in the analysis pipeline to
/// self-monitor memory pressure. When the system has less than 5% available
/// memory (CRITICAL), it triggers cancellation so the analysis returns partial
/// results and frees its buffers, preventing OOM.
///
/// Returns `true` if cancellation was triggered (or was already active).
pub fn check_memory_pressure_and_cancel() -> bool {
    // Short-circuit if already cancelled — avoid the system call.
    if crate::cancellation::is_cancelled() {
        return true;
    }

    let pressure = detect_memory_pressure();
    if pressure == MemoryPressure::Critical {
        let (available, total) = get_memory_info();
        let pct = if total > 0 {
            (available as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        tracing::warn!(
            available_mb = available / (1024 * 1024),
            total_mb = total / (1024 * 1024),
            available_pct = format!("{pct:.1}%"),
            "CRITICAL memory pressure detected — cancelling in-flight analysis (Issue #1099)"
        );
        crate::cancellation::request_cancellation_memory_pressure();
        return true;
    }

    false
}

/// Pure function variant for testing: check whether the given memory values
/// represent CRITICAL pressure and would trigger cancellation (Issue #1099).
pub fn would_cancel_for_memory_pressure(available_bytes: u64, total_bytes: u64) -> bool {
    categorise_memory_pressure(available_bytes, total_bytes) == MemoryPressure::Critical
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
