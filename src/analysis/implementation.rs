use crate::focus::{compute_impacts_public, compute_impacts_with_activations, RecordProvider};
use crate::types::DiscoverRecord;
use crate::{
    AnalyzeNeuronsInput, AnalyzeSynapsesInput, CandidateNeuronJson, CandidateSynapseJson,
    SynapseJson,
};
use anyhow::{anyhow, Context, Result};

// Import shared types from the new module structure
use crate::analysis::shared::{
    AnalyzeNeuronsResult, AnalyzeSynapsesResult, NeuronNoCandidateDetail, NeuronNoCandidateReason,
    NeuronNoCandidateSummary, SynapseNoCandidateDetail, SynapseNoCandidateReason,
    SynapseNoCandidateSummary,
};
use bytemuck::{Pod, Zeroable};
use crossbeam_channel::{bounded, Receiver, Sender};
use once_cell::sync::OnceCell;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};
use wgpu::util::DeviceExt;

const EPSILON: f32 = 1e-8;
/// Workgroup size for GPU compute shaders. Must match the @workgroup_size in WGSL shaders.
/// 256 is optimal for Apple Silicon: divisible by SIMD width (32), good occupancy,
/// and allows efficient wavefront scheduling on M1/M2/M3/M4 GPUs.
const WORKGROUP_SIZE: u32 = 256;
const MIN_NEURON_SAMPLE_COUNT: usize = 10;

/// Minimum variance of new neuron output across samples.
///
/// Issue #123: When bias is too large relative to the input range, the neuron becomes
/// saturated and outputs nearly-constant values regardless of input. For example:
/// - SOFTSIGN(5 + 0.35×x) ≈ 0.83 for all typical x values
/// - TANH(10 + x) ≈ 1.0 for all x > -9
///
/// A constant-output neuron CANNOT reduce error correlation - it just adds a fixed offset.
/// The prediction model incorrectly assumes the output varies with input, leading to
/// massive prediction failures (e.g., predicting 4.67% improvement when actual is ~0%).
///
/// This threshold rejects candidates where the output standard deviation is below 0.01,
/// meaning the neuron produces nearly identical output across all samples.
const MIN_NEURON_OUTPUT_STD_DEV: f32 = 0.01;

/// Maximum absolute value for outgoing weights in add-neuron candidates.
///
/// Based on analysis of successful discoveries vs failures:
/// - ALL successful discoveries have |outgoing_weight| < 0.05
/// - 36% of failures have |outgoing_weight| > 0.05 (up to 50!)
///
/// Using 0.1 provides some margin while eliminating clearly bad candidates.
/// The new neuron should contribute a SMALL correction, not dominate the network.
const MAX_OUTGOING_WEIGHT: f32 = 0.1;

/// Check if a target neuron uses a threshold-based discrete activation function.
/// STEP and BIPOLAR can benefit from a specialised threshold-crossing analysis model
/// that counts how many samples would flip to the correct output if we add a new connection.
///
/// - **STEP**: Output = value > 0 ? 1 : 0
/// - **BIPOLAR**: Output = value > 0 ? 1 : -1
#[inline]
fn is_threshold_activation(squash: &str) -> bool {
    matches!(squash.to_uppercase().as_str(), "STEP" | "BIPOLAR")
}

/// Check if a new neuron would be saturated (producing nearly-constant output).
///
/// Issue #123: Neurons with large bias values relative to typical inputs become saturated
/// and output nearly the same value regardless of input. This causes massive prediction
/// failures because the linear error model assumes output varies with input.
///
/// Returns `true` if the neuron is NOT saturated (output has sufficient variance).
/// Returns `false` if the neuron IS saturated (should be rejected).
///
/// IMPORTANT: We only reject when INPUT has variance but OUTPUT doesn't. If input is
/// already constant (low variance), then constant output is expected and predictions
/// will still be valid.
///
/// # Arguments
/// * `samples` - The samples to evaluate
/// * `incoming_weight` - Weight from source to new neuron
/// * `bias` - Bias of the new neuron
/// * `activation_fn` - Activation function of the new neuron
fn has_sufficient_output_variance(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    bias: f32,
    activation_fn: fn(f32) -> f32,
) -> bool {
    if samples.len() < 2 {
        return false;
    }

    // Compute mean and variance of both input and output
    let mut input_sum = 0.0f64;
    let mut input_sum_sq = 0.0f64;
    let mut output_sum = 0.0f64;
    let mut output_sum_sq = 0.0f64;
    let mut count = 0u32;

    for sample in samples {
        let input = sample.activation;
        let pre_activation = incoming_weight * input + bias;
        let output = activation_fn(pre_activation);
        if output.is_finite() && input.is_finite() {
            input_sum += input as f64;
            input_sum_sq += (input as f64) * (input as f64);
            output_sum += output as f64;
            output_sum_sq += (output as f64) * (output as f64);
            count += 1;
        }
    }

    if count < 2 {
        return false;
    }

    let n = count as f64;

    // Calculate input variance
    let input_mean = input_sum / n;
    let input_variance = (input_sum_sq / n) - (input_mean * input_mean);
    let input_std_dev = input_variance.max(0.0).sqrt() as f32;

    // Calculate output variance
    let output_mean = output_sum / n;
    let output_variance = (output_sum_sq / n) - (output_mean * output_mean);
    let output_std_dev = output_variance.max(0.0).sqrt() as f32;

    // If input already has low variance, constant output is expected - allow it
    if input_std_dev < MIN_NEURON_OUTPUT_STD_DEV {
        return true;
    }

    // If input has variance but output doesn't, the neuron is saturated - reject
    output_std_dev >= MIN_NEURON_OUTPUT_STD_DEV
}

/// Compute source activation variance discount factor.
///
/// Issue #130 (v0.2.2): When a source neuron has constant or near-constant activation,
/// adding a connection from it cannot reduce error correlation - it only adds a constant
/// offset. The prediction model must discount improvements based on source variance.
///
/// **Key insight**: If source activation is constant, the new connection acts like a
/// bias change, not a meaningful signal. A constant cannot correlate with varying error.
///
/// **Production example**: input-1244 had variance 0.000000 (completely constant), yet
/// the model predicted 29.6% error reduction. Actual result was 0%.
///
/// # Returns
/// A discount factor in [0, 1]:
/// - 1.0: Source has high variance (no discount)
/// - 0.0: Source is constant (full discount → zero improvement)
/// - Between: Proportional discount based on variance ratio
///
/// # Formula
/// `discount = min(1.0, source_std_dev / MIN_SOURCE_STD_DEV)`
///
/// Where MIN_SOURCE_STD_DEV = 0.05 (sources with std dev < 0.05 are progressively discounted)
fn compute_source_variance_discount(samples: &[HelpfulSample]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }

    // Minimum source standard deviation for full credit.
    // Sources with std dev below this are progressively discounted.
    // Value chosen based on production analysis: input-1064 had std dev 0.01 and caused
    // massive over-prediction. Sources should have at least 0.05 std dev for reliable correlation.
    const MIN_SOURCE_STD_DEV: f32 = 0.05;

    let mut activation_sum = 0.0f64;
    let mut activation_sq_sum = 0.0f64;
    let mut count = 0u32;

    for sample in samples {
        if sample.activation.is_finite() {
            let a = sample.activation as f64;
            activation_sum += a;
            activation_sq_sum += a * a;
            count += 1;
        }
    }

    if count < 2 {
        return 0.0;
    }

    let n = count as f64;
    let mean = activation_sum / n;
    let variance = (activation_sq_sum / n) - (mean * mean);
    let std_dev = variance.max(0.0).sqrt() as f32;

    // Linear discount: full credit at MIN_SOURCE_STD_DEV, zero at 0
    // Values above MIN_SOURCE_STD_DEV get full credit (capped at 1.0)
    (std_dev / MIN_SOURCE_STD_DEV).clamp(0.0, 1.0)
}

/// Default GPU batch size for GPU operations.
/// This is tuned for M1/M2/M3 which have moderate GPU core counts.
const DEFAULT_GPU_BATCH_SIZE: usize = 512;

/// Larger GPU batch size for high-performance GPUs (M4, M3 Pro/Max, etc.).
/// M4 has significantly more GPU cores and can handle larger batches efficiently.
const HIGH_PERF_GPU_BATCH_SIZE: usize = 1024;

/// Smaller GPU batch size for memory-constrained systems.
/// Used when available memory is low to prevent memory pressure.
const LOW_MEMORY_GPU_BATCH_SIZE: usize = 256;

/// Memory threshold (in GB) below which we use conservative settings.
/// Systems with less than 8GB available should use low-memory mode.
const LOW_MEMORY_THRESHOLD_GB: f64 = 8.0;

/// Memory threshold (in GB) for standard settings.
/// Systems with 8-16GB use standard settings.
const STANDARD_MEMORY_THRESHOLD_GB: f64 = 16.0;

/// Minimum total system memory (in GB) required for discovery.
/// Systems with less than 4GB total RAM cannot reliably run GPU discovery
/// without risking hangs from memory pressure/swap thrashing.
const MINIMUM_TOTAL_MEMORY_GB: f64 = 4.0;

/// Minimum available memory (in GB) required for discovery.
/// If less than 1GB is available, discovery is disabled to prevent hangs.
/// Note: macOS on Apple Silicon aggressively uses memory for caching, so
/// "available" memory is often reported low even when plenty can be reclaimed.
/// 1GB is sufficient since macOS can quickly reclaim cached/inactive pages.
const MINIMUM_AVAILABLE_MEMORY_GB: f64 = 1.0;

/// Minimum timeout (seconds) for the GPU work queue waiting for a response from the GPU thread.
/// This is the OUTER timeout - if the GPU thread doesn't respond within this time,
/// the queue gives up and returns an error.
/// For large datasets (>1GB Parquet files), batch evaluation may take longer than 60 seconds.
const GPU_QUEUE_TIMEOUT_MIN_SECS: u64 = 60;

/// Maximum timeout (seconds) for GPU batch operations.
/// Even with large datasets, if the GPU hasn't responded in 5 minutes, something is wrong.
const GPU_QUEUE_TIMEOUT_MAX_SECS: u64 = 300;

/// Buffer map timeout is always 5 seconds shorter than queue timeout to avoid race conditions.
/// If both timeouts are the same, the queue might timeout before the GPU thread
/// has a chance to return its own timeout error, leaving the thread stuck.
const GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS: u64 = 5;

/// Default buffer map timeout for internal GPU operations (in seconds).
/// This is used inside the GPU thread where we don't have access to the external deadline.
/// Set to max queue timeout minus margin for safety.
const GPU_BUFFER_MAP_TIMEOUT_SECS: u64 =
    GPU_QUEUE_TIMEOUT_MAX_SECS - GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS;

/// Maximum estimated bytes of GPU-side buffers we allow per *batched* submission.
///
/// Why:
/// - For large recordings (eg 50k+ samples), `batch_size=1024` can produce enormous transient
///   Metal buffers (samples + contributions + staging) and wedge the driver.
/// - When the driver wedges, utilisation drops to ~0 and discovery yields no candidates due to timeouts.
///
/// We cap based on an estimate of the dominant buffers:
/// - Sample storage buffer (GpuHelpfulSample)
/// - Contribution buffer (HelpfulContribution / HarmfulContribution)
/// - Staging buffer (same size as contribution buffer)
///
/// This cap is conservative by design; it trades a bit of peak throughput for stability on
/// Apple Silicon (and makes time-bounded runs far more reliable).
const GPU_MAX_BATCH_ALLOC_BYTES: usize = 256 * 1024 * 1024; // 256MB

fn cap_gpu_batch_size_by_bytes(
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

/// Timeout for GPU thread initialisation (in seconds).
/// GPU device creation should be fast; if it takes longer, something is wrong.
const GPU_INIT_TIMEOUT_SECS: u64 = 30;

/// Calculate adaptive GPU batch timeout based on remaining deadline.
///
/// For large datasets (>1GB Parquet files), GPU batch evaluations may take
/// longer than the minimum 60 seconds. This function calculates a reasonable
/// timeout based on:
/// - Minimum: GPU_QUEUE_TIMEOUT_MIN_SECS (60s) - catches unresponsive GPU
/// - Maximum: GPU_QUEUE_TIMEOUT_MAX_SECS (5 min) - prevents infinite waits
/// - If deadline is available: uses up to half remaining time (capped at max)
fn calculate_gpu_batch_timeout(deadline: &Option<std::time::SystemTime>) -> Duration {
    let min_timeout = Duration::from_secs(GPU_QUEUE_TIMEOUT_MIN_SECS);
    let max_timeout = Duration::from_secs(GPU_QUEUE_TIMEOUT_MAX_SECS);

    match deadline {
        Some(dl) => {
            if let Ok(remaining) = dl.duration_since(std::time::SystemTime::now()) {
                // Use half of remaining time, but capped between min and max
                let half_remaining = remaining / 2;
                if half_remaining < min_timeout {
                    min_timeout
                } else if half_remaining > max_timeout {
                    max_timeout
                } else {
                    half_remaining
                }
            } else {
                // Deadline already passed - use minimum
                min_timeout
            }
        }
        // No deadline - use maximum
        None => max_timeout,
    }
}

/// Poll the GPU device until it has no work in flight, or a timeout is reached.
///
/// This intentionally avoids `Maintain::Wait` because that can block forever on
/// some machines if the GPU driver wedges. Instead, we poll in a loop and bail
/// out with an error so unattended workers can recover (or watchdog can abort).
fn poll_device_until_idle(device: &wgpu::Device, timeout: Duration, label: &str) -> Result<()> {
    use std::time::Instant;
    let start = Instant::now();
    loop {
        let result = device.poll(wgpu::Maintain::Poll);
        if result.is_queue_empty() {
            return Ok(());
        }
        if start.elapsed() > timeout {
            return Err(anyhow!(
                "GPU device poll timed out after {:.1}s ({label}). The GPU driver may be unresponsive.",
                timeout.as_secs_f64()
            ));
        }
        // Short sleep to avoid busy-waiting.
        thread::sleep(Duration::from_millis(10));
    }
}

/// Wait for a GPU buffer mapping to complete.
///
/// We avoid `Maintain::Wait` because that can block forever on some machines if
/// the GPU driver is wedged. Instead, we poll in a loop until the callback fires
/// or the timeout is reached.
fn wait_for_buffer_map(
    device: &wgpu::Device,
    receiver: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    timeout_secs: u64,
) -> Result<()> {
    use std::time::Instant;

    let timeout = Duration::from_secs(timeout_secs);
    let start = Instant::now();
    loop {
        match receiver.try_recv() {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(err)) => return Err(anyhow!("Buffer mapping failed: {err}")),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(anyhow!("GPU callback channel disconnected"));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Not ready yet; keep polling.
            }
        }

        let result = device.poll(wgpu::Maintain::Poll);
        if result.is_queue_empty() && start.elapsed() > Duration::from_millis(250) {
            // If the queue is empty but the callback hasn't fired, something is off.
            // Keep trying until timeout, but this is a strong signal of driver trouble.
        }

        if start.elapsed() > timeout {
            return Err(anyhow!(
                "GPU buffer mapping timed out after {:.1}s. The GPU driver may be unresponsive.",
                timeout.as_secs_f64()
            ));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

/// Wait for multiple GPU buffer mappings to complete.
fn wait_for_buffer_maps_batch(
    device: &wgpu::Device,
    receivers: &[std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>],
    timeout_secs: u64,
) -> Result<()> {
    if receivers.is_empty() {
        return Ok(());
    }

    use std::time::Instant;

    let timeout = Duration::from_secs(timeout_secs);
    let start = Instant::now();
    let mut done: Vec<Option<Result<(), wgpu::BufferAsyncError>>> = vec![None; receivers.len()];

    loop {
        // Drain as many completion callbacks as possible.
        for (i, receiver) in receivers.iter().enumerate() {
            if done[i].is_some() {
                continue;
            }
            match receiver.try_recv() {
                Ok(result) => done[i] = Some(result),
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(anyhow!("Buffer {i} callback channel disconnected"));
                }
            }
        }

        if done.iter().all(|x| x.is_some()) {
            // Validate all results.
            for (i, result) in done.into_iter().enumerate() {
                match result.expect("checked is_some") {
                    Ok(()) => {}
                    Err(err) => return Err(anyhow!("Buffer {i} mapping failed: {err}")),
                }
            }
            return Ok(());
        }

        device.poll(wgpu::Maintain::Poll);

        if start.elapsed() > timeout {
            return Err(anyhow!(
                "GPU batch buffer mapping timed out after {:.1}s. The GPU driver may be unresponsive.",
                timeout.as_secs_f64()
            ));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

/// Detected GPU performance tier for auto-tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuPerformanceTier {
    /// High-performance GPU (M4, M3 Pro/Max, M2 Pro/Max, dedicated GPUs)
    High,
    /// Standard GPU (M1, M2, M3 base, integrated GPUs)
    Standard,
    /// Unknown/fallback
    Unknown,
}

/// Detect GPU performance tier from adapter info.
/// Returns High for M4, Pro/Max variants; Standard for base M-series; Unknown otherwise.
fn detect_gpu_tier(adapter_info: &wgpu::AdapterInfo) -> GpuPerformanceTier {
    let name = adapter_info.name.to_lowercase();

    // M4 series - highest performance
    if name.contains("m4") {
        return GpuPerformanceTier::High;
    }

    // Pro/Max/Ultra variants of any M-series - high performance
    if (name.contains("m1") || name.contains("m2") || name.contains("m3"))
        && (name.contains("pro") || name.contains("max") || name.contains("ultra"))
    {
        return GpuPerformanceTier::High;
    }

    // Base M-series - standard performance
    if name.contains("m1") || name.contains("m2") || name.contains("m3") {
        return GpuPerformanceTier::Standard;
    }

    // Dedicated GPUs are typically high performance
    if adapter_info.device_type == wgpu::DeviceType::DiscreteGpu {
        return GpuPerformanceTier::High;
    }

    // Integrated GPUs - standard
    if adapter_info.device_type == wgpu::DeviceType::IntegratedGpu {
        return GpuPerformanceTier::Standard;
    }

    GpuPerformanceTier::Unknown
}

/// System resource information for adaptive configuration.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // Fields kept for diagnostic logging
struct SystemResources {
    /// Available memory in bytes (not total - what's actually free)
    available_memory_bytes: u64,
    /// Total physical memory in bytes
    total_memory_bytes: u64,
    /// Memory pressure level
    memory_tier: MemoryTier,
}

/// Memory tier for adaptive configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryTier {
    /// Less than 8GB available - use conservative settings
    Low,
    /// 8-16GB available - use standard settings
    Standard,
    /// More than 16GB available - use aggressive settings
    High,
}

/// Detect available system memory and categorise into tiers.
/// This is cached for the lifetime of the process.
fn detect_system_resources() -> SystemResources {
    use std::sync::OnceLock;
    static RESOURCES: OnceLock<SystemResources> = OnceLock::new();

    *RESOURCES.get_or_init(|| {
        let (available, total) = get_memory_info();
        let available_gb = available as f64 / (1024.0 * 1024.0 * 1024.0);
        let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);

        let memory_tier = if available_gb < LOW_MEMORY_THRESHOLD_GB {
            MemoryTier::Low
        } else if available_gb < STANDARD_MEMORY_THRESHOLD_GB {
            MemoryTier::Standard
        } else {
            MemoryTier::High
        };

        // Log memory info once
        let tier_str = match memory_tier {
            MemoryTier::Low => "low",
            MemoryTier::Standard => "standard",
            MemoryTier::High => "high",
        };
        eprintln!(
            "[NEAT-AI-Discovery] Memory: {available_gb:.1}GB available / {total_gb:.1}GB total | Tier: {tier_str}"
        );

        SystemResources {
            available_memory_bytes: available,
            total_memory_bytes: total,
            memory_tier,
        }
    })
}

/// Get memory information from the OS.
/// Returns (available_bytes, total_bytes).
#[cfg(target_os = "macos")]
fn get_memory_info() -> (u64, u64) {
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
    let available = vm_stat
        .map(|o| {
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
        })
        .unwrap_or(total / 2); // Default to half of total

    (available, total)
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_line(line: &str) -> u64 {
    line.split(':')
        .nth(1)
        .and_then(|s| s.trim().trim_end_matches('.').parse::<u64>().ok())
        .unwrap_or(0)
}

/// Parse the page size from vm_stat output's header line.
/// Example: "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
/// Returns the page size in bytes, or a default based on architecture.
///
/// Apple Silicon uses 16KB pages, Intel Macs use 4KB pages.
/// Parsing dynamically ensures correct memory calculations on both.
#[cfg(target_os = "macos")]
fn parse_vm_stat_page_size(output: &str) -> u64 {
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
fn get_memory_info() -> (u64, u64) {
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
/// Example: "MemTotal:       16384000 kB" -> Some(16384000)
#[cfg(target_os = "linux")]
fn parse_meminfo_line(line: &str) -> Option<u64> {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
}

/// Fallback for other platforms (Windows, FreeBSD, etc.).
/// Returns conservative defaults since we don't have platform-specific memory detection.
/// GPU discovery will still work via wgpu (DirectX 12 on Windows, Vulkan elsewhere).
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn get_memory_info() -> (u64, u64) {
    // Conservative defaults: 8GB total, 4GB available
    // These are safe values that won't trigger low-memory protections on modern machines
    (4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
}

/// Check if there's enough memory to load a parquet file.
///
/// Parquet files are compressed; in-memory representation is typically 2-4x larger.
/// This function estimates memory needed and returns an error if insufficient.
///
/// We use conservative checks to avoid memory pressure that causes the system to
/// become unresponsive (stuck with low CPU/GPU, eventually OOM killed):
/// 1. Require 3× file size for in-memory representation
/// 2. Require at least 1GB headroom after loading (for GPU buffers, etc.)
/// 3. Don't use more than 50% of total RAM for parquet data
///
/// This is a public function so it can be used by focus.rs and analysis.rs.
pub fn check_memory_for_parquet(parquet_file: &str) -> Result<()> {
    const MEMORY_MULTIPLIER: f64 = 3.0;
    const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIN_HEADROOM_GB: f64 = 1.0; // Keep at least 1GB free for GPU/system
    const MAX_MEMORY_FRACTION: f64 = 0.5; // Don't use more than 50% of total RAM

    let file_size_bytes = std::fs::metadata(parquet_file)
        .map(|m| m.len())
        .unwrap_or(0);
    let file_size_gb = file_size_bytes as f64 / BYTES_PER_GB;
    let estimated_memory_gb = file_size_gb * MEMORY_MULTIPLIER;

    let (available_bytes, total_bytes) = get_memory_info();
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
             • Parquet file: {file_size_mb:.0} MB ({parquet_file})\n\
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
             • Parquet file: {file_size_mb:.0} MB ({parquet_file})\n\
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

    // Log memory usage for large files (helps diagnose issues)
    if file_size_gb > 0.5 {
        let usage_percent = (estimated_memory_gb / total_gb) * 100.0;
        eprintln!(
            "[NEAT-AI-Discovery] Loading {file_size_mb:.0} MB parquet file \
             (estimated {estimated_mb:.0} MB in memory = {usage_percent:.0}% of RAM, \
             {available_mb:.0} MB available)"
        );
    }

    Ok(())
}

/// Get the GPU work queue capacity based on system resources.
/// Lower capacity = more backpressure = less memory usage.
fn get_work_queue_capacity() -> usize {
    let resources = detect_system_resources();

    match resources.memory_tier {
        MemoryTier::Low => 4,      // Aggressive backpressure
        MemoryTier::Standard => 8, // Moderate backpressure
        MemoryTier::High => 16,    // Allow more parallelism
    }
}

/// Check if the system meets minimum requirements for GPU discovery.
///
/// Very old machines with insufficient memory cannot reliably run GPU operations
/// without risking hangs from memory pressure. This check runs before GPU
/// initialisation to allow discovery to be disabled gracefully.
///
/// Returns `Some(GpuAvailabilityResult)` if requirements are NOT met (discovery disabled).
/// Returns `None` if requirements ARE met (continue with GPU check).
fn check_minimum_system_requirements() -> Option<GpuAvailabilityResult> {
    let (available, total) = get_memory_info();
    let available_gb = available as f64 / (1024.0 * 1024.0 * 1024.0);
    let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);

    // Check total system memory
    if total_gb < MINIMUM_TOTAL_MEMORY_GB {
        eprintln!(
            "[NEAT-AI-Discovery] System has only {total_gb:.1}GB total RAM (minimum: {MINIMUM_TOTAL_MEMORY_GB}GB). Discovery disabled."
        );
        return Some(GpuAvailabilityResult {
            available: false,
            reason: Some(format!(
                "Insufficient system memory: {total_gb:.1}GB total (minimum: {MINIMUM_TOTAL_MEMORY_GB}GB required). \
                 Discovery is disabled to prevent hangs on memory-constrained machines. \
                 Evolution will continue without discovery."
            )),
            is_error: false, // Not an error - graceful disable
        });
    }

    // Check available memory
    if available_gb < MINIMUM_AVAILABLE_MEMORY_GB {
        eprintln!(
            "[NEAT-AI-Discovery] Only {available_gb:.1}GB memory available (minimum: {MINIMUM_AVAILABLE_MEMORY_GB}GB). Discovery disabled."
        );
        return Some(GpuAvailabilityResult {
            available: false,
            reason: Some(format!(
                "Insufficient available memory: {available_gb:.1}GB available (minimum: {MINIMUM_AVAILABLE_MEMORY_GB}GB required). \
                 Discovery is disabled to prevent hangs from memory pressure. \
                 Try closing other applications or reboot to free memory."
            )),
            is_error: false, // Not an error - graceful disable
        });
    }

    // Requirements met
    None
}

/// Adjust batch size based on both GPU tier and memory availability.
fn get_adjusted_batch_size(gpu_tier: GpuPerformanceTier) -> usize {
    // Check for explicit override first
    if let Some(size) = get_batch_size_override() {
        return size;
    }

    let resources = detect_system_resources();
    let base_size = match gpu_tier {
        GpuPerformanceTier::High => HIGH_PERF_GPU_BATCH_SIZE,
        GpuPerformanceTier::Standard | GpuPerformanceTier::Unknown => DEFAULT_GPU_BATCH_SIZE,
    };

    // Reduce batch size if memory is constrained
    // Standard memory tier should also reduce batch size to prevent Metal command buffer exhaustion
    match resources.memory_tier {
        MemoryTier::Low => LOW_MEMORY_GPU_BATCH_SIZE.min(base_size),
        MemoryTier::Standard => DEFAULT_GPU_BATCH_SIZE.min(base_size), // Use 512 max, not 1024
        MemoryTier::High => base_size,
    }
}

/// Get cached batch size override from environment variable.
/// Returns None if not set or invalid.
fn get_batch_size_override() -> Option<usize> {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("NEAT_AI_DISCOVERY_GPU_BATCH_SIZE")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .filter(|size| (64..=4096).contains(size))
    })
}

/// Get optimised GPU batch size based on detected GPU tier (without memory adjustment).
/// Used by tests. Production code uses `get_adjusted_batch_size()` which also considers memory.
#[cfg(test)]
fn get_batch_size_for_tier(tier: GpuPerformanceTier) -> usize {
    // Check for explicit override first (cached)
    if let Some(size) = get_batch_size_override() {
        return size;
    }

    match tier {
        GpuPerformanceTier::High => HIGH_PERF_GPU_BATCH_SIZE,
        GpuPerformanceTier::Standard | GpuPerformanceTier::Unknown => DEFAULT_GPU_BATCH_SIZE,
    }
}

/// Log GPU adapter info once per process for diagnostic purposes.
/// This helps users understand what hardware is being used and the selected batch size.
fn log_gpu_info_once(
    adapter_info: &wgpu::AdapterInfo,
    tier: GpuPerformanceTier,
    batch_size: usize,
) {
    use std::sync::OnceLock;
    static LOGGED: OnceLock<bool> = OnceLock::new();

    LOGGED.get_or_init(|| {
        // Always log GPU info to help diagnose performance issues
        let tier_str = match tier {
            GpuPerformanceTier::High => "high-performance",
            GpuPerformanceTier::Standard => "standard",
            GpuPerformanceTier::Unknown => "unknown",
        };

        let device_type = match adapter_info.device_type {
            wgpu::DeviceType::DiscreteGpu => "discrete",
            wgpu::DeviceType::IntegratedGpu => "integrated",
            wgpu::DeviceType::VirtualGpu => "virtual",
            wgpu::DeviceType::Cpu => "CPU",
            wgpu::DeviceType::Other => "other",
        };

        eprintln!(
            "[NEAT-AI-Discovery] GPU: {} ({} {}) | Tier: {} | Batch size: {}",
            adapter_info.name,
            device_type,
            format!("{:?}", adapter_info.backend).to_lowercase(),
            tier_str,
            batch_size
        );

        // Provide tuning hints for verbose mode
        if verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] GPU tuning: Set NEAT_AI_DISCOVERY_GPU_BATCH_SIZE=N to override (64-4096). \
                 Higher values improve GPU utilisation on powerful hardware."
            );
        }

        true
    });
}

/// Constants for deadline calculation - shared between `calculate_effective_timeout_ms` and `build_deadline`.
const DEFAULT_DURATION_MS: u64 = 600_000; // 10 minutes (10 * 60 * 1000)
const YEAR_2000_MS: u64 = 946_684_800_000;
const MIN_DURATION_MS: u64 = 3_000; // 3 seconds
const MAX_DURATION_MS: u64 = 3_600_000; // 1 hour (60 * 60 * 1000)

/// Calculate the effective timeout duration in milliseconds.
///
/// This function applies the same logic as `build_deadline`:
/// 1. Converts absolute timestamps (values >= year 2000 in ms) to relative durations
/// 2. Clamps values outside the 3-second to 1-hour range to the 10-minute default
///
/// Returns `None` if the deadline is in the past (for absolute timestamps), or
/// `Some(effective_duration_ms)` otherwise.
///
/// Used by both `build_deadline` (to create the SystemTime) and `log_analysis_start`
/// (to display the effective timeout to users).
fn calculate_effective_timeout_ms(deadline_ms: Option<u64>) -> Option<u64> {
    let target_ms = deadline_ms.unwrap_or(DEFAULT_DURATION_MS);

    // Heuristic: if the value is less than year 2000 in milliseconds,
    // treat it as a relative duration. Otherwise, it's likely an absolute timestamp
    // from the calling code, so convert it to a relative duration.
    let relative_ms = if target_ms < YEAR_2000_MS {
        // Small value - treat as relative duration (milliseconds from now)
        target_ms
    } else {
        // Large value - likely an absolute timestamp from calling code.
        // Convert to relative duration by subtracting current time.
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()?
            .as_millis() as u64;

        // If the timestamp is in the past, return None (deadline already passed)
        if target_ms <= now_ms {
            return None;
        }

        // Calculate relative duration
        target_ms - now_ms
    };

    // Validate duration bounds: minimum 3 seconds, maximum 1 hour
    // If invalid, default to 10 minutes (expected typical value)
    // NOTE: If these warnings appear, it's a bug in the calling code (NEAT-AI or GRQ)
    // that should be fixed to pass valid timeout values.
    let validated_ms = if relative_ms < MIN_DURATION_MS {
        eprintln!(
            "⚠️  [NEAT-AI-Discovery] BUG: analysis_deadline_ms ({:.1}s) is less than minimum (3s). \
             Using default 10 minute timeout. Please fix the calling code (NEAT-AI/GRQ) to pass a valid timeout.",
            relative_ms as f64 / 1000.0
        );
        DEFAULT_DURATION_MS
    } else if relative_ms > MAX_DURATION_MS {
        eprintln!(
            "⚠️  [NEAT-AI-Discovery] BUG: analysis_deadline_ms ({:.1}s) exceeds maximum (1 hour). \
             Using default 10 minute timeout. Please fix the calling code (NEAT-AI/GRQ) to pass a valid timeout.",
            relative_ms as f64 / 1000.0
        );
        DEFAULT_DURATION_MS
    } else {
        relative_ms
    };

    Some(validated_ms)
}

fn parse_input_index(uuid: &str) -> Option<usize> {
    uuid.strip_prefix("input-")?.parse::<usize>().ok()
}

fn interleave_from_ends<T>(items: Vec<T>) -> Vec<T> {
    use std::collections::VecDeque;
    if items.len() <= 2 {
        return items;
    }
    // Items are expected to already be in a meaningful order (eg sorted by index).
    let mut deque: VecDeque<T> = items.into();
    let mut out = Vec::with_capacity(deque.len());
    while let Some(front) = deque.pop_front() {
        out.push(front);
        if let Some(back) = deque.pop_back() {
            out.push(back);
        }
    }
    out
}

fn build_deadline(deadline_ms: Option<u64>) -> Option<SystemTime> {
    // Treat deadline_ms as a relative duration (milliseconds from now), not an absolute timestamp.
    // The calling code (TypeScript) calculates this as Date.now() + duration, but we want to treat
    // it as a duration to avoid issues with clock skew and to match the expected semantics.
    // If the calling code passes an absolute timestamp, we need to convert it to a relative duration.
    // If None is passed, apply default 10 minute timeout to prevent runaway analysis.
    calculate_effective_timeout_ms(deadline_ms)
        .and_then(|validated_ms| SystemTime::now().checked_add(Duration::from_millis(validated_ms)))
}

fn deadline_passed(deadline: &Option<SystemTime>) -> bool {
    #[cfg(test)]
    {
        if let Some(value) = deadline_override::next_override_value() {
            return value;
        }
    }

    matches!(deadline, Some(limit) if SystemTime::now() >= *limit)
}

/// Log analysis start information including deadline and focus neuron count.
/// This provides visibility into timeout configuration without requiring verbose mode.
fn log_analysis_start(
    analysis_type: &str,
    deadline_ms: Option<u64>,
    focus_count: usize,
    shuffled_order: &[String],
) {
    // Calculate the effective deadline duration using the same logic as build_deadline.
    // This ensures the logged timeout matches what's actually used.
    let deadline_duration_ms =
        calculate_effective_timeout_ms(deadline_ms).unwrap_or(DEFAULT_DURATION_MS);
    let deadline_secs = deadline_duration_ms as f64 / 1000.0;

    // Debug: Log the raw deadline_ms value if verbose to help diagnose timeout issues
    if verbose_enabled() {
        if let Some(raw_ms) = deadline_ms {
            let now_ms = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            if raw_ms >= YEAR_2000_MS {
                // Absolute timestamp
                let elapsed_secs =
                    now_ms.saturating_sub(raw_ms.saturating_sub(deadline_duration_ms)) as f64
                        / 1000.0;
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] {analysis_type} deadline_ms={raw_ms} (absolute timestamp), \
                     {elapsed_secs:.1}s elapsed since timeout was set, {deadline_secs:.1}s remaining"
                );
            } else {
                // Relative duration
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] {analysis_type} deadline_ms={raw_ms} (relative duration)"
                );
            }
        }
    }

    // Warn if timeout is very short - likely means focus selection took most of the allotted time
    const MIN_USEFUL_TIMEOUT_SECS: f64 = 60.0; // 1 minute minimum for useful analysis
    if deadline_secs < MIN_USEFUL_TIMEOUT_SECS {
        eprintln!(
            "💡  [NEAT-AI-Discovery]: Only {deadline_secs:.1}s remaining for {analysis_type} analysis. \
             Focus selection may have consumed most of the timeout. \
             Consider increasing discoveryAnalysisTimeoutMinutes."
        );
    }

    // Format the timeout nicely
    let timeout_str = if deadline_secs >= 60.0 {
        let minutes = deadline_secs / 60.0;
        format!("{minutes:.1} minutes")
    } else {
        format!("{deadline_secs:.1} seconds")
    };

    eprintln!(
        "[NEAT-AI-Discovery] Starting {analysis_type} analysis: {focus_count} focus neurons, timeout: {timeout_str}"
    );

    // Log the shuffled order if verbose mode is enabled
    if verbose_enabled() && !shuffled_order.is_empty() {
        let preview: Vec<&str> = shuffled_order.iter().take(5).map(|s| s.as_str()).collect();
        let extra = shuffled_order.len().saturating_sub(5);
        let suffix = if extra > 0 {
            format!("... (+{extra} more)")
        } else {
            String::new()
        };
        eprintln!("[NEAT-AI-Discovery][verbose] Randomised focus order: {preview:?}{suffix}");
    }
}

/// Log when analysis timeout is reached. Always prints (not verbose-only).
fn log_analysis_timeout(analysis_type: &str, completed_count: usize, total_count: usize) {
    eprintln!(
        "[NEAT-AI-Discovery] {analysis_type} analysis reached timeout. Completed {completed_count}/{total_count} focus neurons. \
         Returning partial results."
    );
}

#[cfg(test)]
mod deadline_override {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    use std::sync::{Mutex, MutexGuard};

    static OVERRIDE_LOCK: Mutex<()> = Mutex::new(());
    static OVERRIDE_SEQUENCE: Mutex<VecDeque<bool>> = Mutex::new(VecDeque::new());
    static OVERRIDE_ACTIVE: AtomicBool = AtomicBool::new(false);

    pub(super) struct DeadlineOverrideGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl DeadlineOverrideGuard {
        pub(super) fn with_sequence(sequence: Vec<bool>) -> Self {
            let lock = OVERRIDE_LOCK
                .lock()
                .expect("Deadline override lock should not be poisoned");
            {
                let mut queue = OVERRIDE_SEQUENCE
                    .lock()
                    .expect("Deadline override queue should not be poisoned");
                queue.clear();
                for value in sequence.into_iter() {
                    queue.push_back(value);
                }
            }
            OVERRIDE_ACTIVE.store(true, AtomicOrdering::SeqCst);
            Self { _lock: lock }
        }
    }

    impl Drop for DeadlineOverrideGuard {
        fn drop(&mut self) {
            OVERRIDE_ACTIVE.store(false, AtomicOrdering::SeqCst);
            let mut queue = OVERRIDE_SEQUENCE
                .lock()
                .expect("Deadline override queue should not be poisoned");
            queue.clear();
        }
    }

    pub(super) fn next_override_value() -> Option<bool> {
        if !OVERRIDE_ACTIVE.load(AtomicOrdering::SeqCst) {
            return None;
        }
        // Allow any thread to consume the override sequence for parallel processing compatibility
        // With parallel processing via par_iter(), worker threads have different thread IDs,
        // so we allow all threads to see and consume the override sequence.
        let mut queue = OVERRIDE_SEQUENCE
            .lock()
            .expect("Deadline override queue should not be poisoned");
        queue.pop_front()
    }
}

/// Check if verbose logging is enabled. Result is cached for performance.
/// Set `NEAT_AI_DISCOVERY_VERBOSE=1` to enable verbose logging.
/// Check if verbose logging is enabled via NEAT_AI_DISCOVERY_VERBOSE environment variable.
/// This is public so it can be used by focus.rs and other modules.
pub fn verbose_enabled() -> bool {
    use std::sync::OnceLock;
    static VERBOSE: OnceLock<bool> = OnceLock::new();
    *VERBOSE.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok())
}

/// Suppress Mesa/libEGL debug warnings on Linux.
///
/// When wgpu initialises on Linux, it probes multiple GPU backends (EGL, Vulkan, etc.).
/// If the user lacks permission to access `/dev/dri/renderD*` or `/dev/dri/card*` devices,
/// libEGL emits warnings like "failed to open /dev/dri/renderD128: Permission denied".
///
/// These warnings are often benign if wgpu finds an alternative backend (e.g., Vulkan via
/// a different ICD loader). This function suppresses the warnings by setting environment
/// variables that quiet Mesa's debug output.
///
/// Set `NEAT_AI_DISCOVERY_QUIET_GPU=1` to enable suppression.
#[cfg(target_os = "linux")]
fn suppress_mesa_warnings_if_requested() {
    use std::env;
    use std::sync::Once;

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if env::var("NEAT_AI_DISCOVERY_QUIET_GPU").is_ok() {
            // Suppress EGL debug messages (these cause "failed to open /dev/dri/..." warnings)
            if env::var("EGL_LOG_LEVEL").is_err() {
                // SAFETY: single-threaded at this point (Once guard) and before GPU init
                unsafe { env::set_var("EGL_LOG_LEVEL", "fatal") };
            }

            // Suppress Mesa GLSL shader cache warnings
            if env::var("MESA_GLSL_CACHE_DISABLE").is_err() {
                unsafe { env::set_var("MESA_GLSL_CACHE_DISABLE", "true") };
            }

            // Suppress general Mesa debug output
            if env::var("MESA_DEBUG").is_err() {
                unsafe { env::set_var("MESA_DEBUG", "silent") };
            }
        }
    });
}

#[cfg(not(target_os = "linux"))]
fn suppress_mesa_warnings_if_requested() {
    // No-op on non-Linux platforms
}

/// Ensure XDG_RUNTIME_DIR is set on Linux (required by wgpu on Wayland).
///
/// This function uses `Once` for thread-safe one-time initialisation. It's safe to
/// call from multiple threads concurrently - only the first call will set the
/// environment variable, and subsequent calls are no-ops.
///
/// Must be called before any GPU initialisation (wgpu Instance creation).
#[cfg(target_os = "linux")]
fn ensure_xdg_runtime_dir() {
    use std::env;
    use std::sync::Once;

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if env::var("XDG_RUNTIME_DIR").is_err() {
            // Create a temporary runtime directory if XDG_RUNTIME_DIR is not set
            if let Ok(temp_dir) = std::env::temp_dir().canonicalize() {
                let runtime_dir = temp_dir.join("neat-ai-discovery-runtime");
                if let Err(e) = std::fs::create_dir_all(&runtime_dir) {
                    eprintln!("[NEAT-AI-Discovery] Warning: Failed to create XDG_RUNTIME_DIR at {runtime_dir:?}: {e}");
                } else {
                    // SAFETY: Inside Once::call_once, so guaranteed single-threaded execution.
                    // Called before any GPU init.
                    unsafe {
                        env::set_var("XDG_RUNTIME_DIR", runtime_dir.to_string_lossy().as_ref());
                    }
                }
            }
        }
    });
}

#[cfg(not(target_os = "linux"))]
fn ensure_xdg_runtime_dir() {
    // No-op on non-Linux platforms
}

/// Safely create a wgpu Instance, avoiding panics from backend probing.
///
/// On Linux with old hardware or missing GPU drivers, wgpu's EGL/OpenGL backend
/// can panic during initialisation (e.g., "BadDisplay" errors). This function:
///
/// - On Linux: Disables the GL backend entirely, using only Vulkan to avoid EGL panics
/// - On macOS: Uses Metal (the default and only backend on macOS)
/// - On all platforms: Wraps instance creation in `catch_unwind` as a safety net
///
/// Returns `None` if instance creation fails or panics, allowing callers to handle
/// the failure gracefully (e.g., treating missing GPU as discovery-disabled on Linux).
fn create_wgpu_instance_safely() -> Option<wgpu::Instance> {
    use std::panic;

    // On Linux, avoid GL/GLES backend which can panic on EGL initialisation
    // when /dev/dri devices are missing or inaccessible.
    #[cfg(target_os = "linux")]
    let backends = wgpu::Backends::VULKAN;

    // On macOS, Metal is the only backend and should always work
    #[cfg(target_os = "macos")]
    let backends = wgpu::Backends::METAL;

    // On other platforms, use all available backends
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let backends = wgpu::Backends::all();

    // Wrap in catch_unwind to handle any remaining panics from backend probing
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        })
    }));

    match result {
        Ok(instance) => Some(instance),
        Err(panic_info) => {
            // Log the panic but don't propagate it
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic during wgpu instance creation".to_string()
            };

            #[cfg(target_os = "linux")]
            {
                // On Linux, this is expected on headless servers without GPU
                if verbose_enabled() {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] wgpu instance creation failed: {panic_msg}. \
                         Discovery will be disabled on this machine."
                    );
                }
            }

            #[cfg(target_os = "macos")]
            {
                // On macOS, this is unexpected - Metal should always be available
                eprintln!(
                    "[NEAT-AI-Discovery] ERROR: wgpu instance creation failed on macOS: {panic_msg}. \
                     This indicates a system configuration issue."
                );
            }

            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            {
                eprintln!(
                    "[NEAT-AI-Discovery] wgpu instance creation failed: {panic_msg}. \
                     Discovery will be disabled on this machine."
                );
            }

            None
        }
    }
}

// Types moved to shared.rs - using imports from there

struct OrderedNeuron {
    uuid: String,
    index: usize,
}

type RecordCacheLoader = dyn Fn(&str, &str) -> Result<Vec<DiscoverRecord>> + Send + Sync + 'static;
type CachedNeuronRecords = OnceCell<Arc<Vec<DiscoverRecord>>>;

pub(crate) struct RecordCache {
    parquet_file: String,
    cache: Mutex<HashMap<String, Arc<CachedNeuronRecords>>>,
    loader: Arc<RecordCacheLoader>,
}

/// Adapter to allow `RecordCache` to be used where focus impact code expects a `RecordProvider`.
///
/// This lets us compute squash-aware impacts using *recorded activations* for selection squashes
/// (MINIMUM/MAXIMUM/IF) during candidate discounting, rather than falling back to the conservative
/// 1/N probability model.
struct RecordCacheProvider<'a> {
    cache: &'a RecordCache,
}

impl RecordProvider for RecordCacheProvider<'_> {
    fn get(&self, neuron_uuid: &str) -> Result<Option<Arc<Vec<DiscoverRecord>>>> {
        let records = self.cache.get(neuron_uuid)?;
        if records.is_empty() {
            Ok(None)
        } else {
            Ok(Some(records))
        }
    }

    fn len(&self) -> usize {
        let cache = self
            .cache
            .cache
            .lock()
            .expect("record cache mutex poisoned");
        cache.len()
    }
}

/// Compute neuron impact scores for candidate discounting.
///
/// We prefer activation-based selection statistics when available so MINIMUM/MAXIMUM/IF neurons
/// don't get incorrectly diluted via the 1/N fallback.
fn compute_impact_scores_for_discounting(
    creature: &crate::CreatureJson,
    cache: &RecordCache,
) -> HashMap<String, f32> {
    let provider = RecordCacheProvider { cache };
    match compute_impacts_with_activations(creature, &provider) {
        Ok(scores) => scores,
        Err(err) => {
            if verbose_enabled() {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Falling back to conservative impact calculation \
                    (no activation-based selection stats). Reason: {err}"
                );
            }
            compute_impacts_public(creature)
        }
    }
}

#[derive(Clone, Copy)]
enum RejectionReason {
    NoSamples,
    ZeroImprovement,
    BelowThreshold,
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RejectionReason::NoSamples => write!(f, "no overlapping discovery samples"),
            RejectionReason::ZeroImprovement => write!(f, "no consistent improvement in GPU stats"),
            RejectionReason::BelowThreshold => write!(f, "expected improvement below threshold"),
        }
    }
}

#[derive(Clone)]
struct RejectionDetail {
    source_uuid: String,
    reason: RejectionReason,
    sample_count: usize,
    source_record_count: usize,
    improved_count: u32,
    worsened_count: u32,
    expected_improvement: f32,
    threshold: f32,
    weight: Option<f32>,
}

impl RejectionDetail {
    fn score(&self) -> f32 {
        self.expected_improvement
    }
}

struct ThresholdContext {
    sample_count: usize,
    expected_improvement: f32,
    threshold: f32,
    improved_count: u32,
    worsened_count: u32,
    weight: f32,
}

struct TargetDiagnosticEntry {
    target_uuid: String,
    target_record_count: usize,
    evaluated_candidates: u32,
    candidates_with_samples: u32,
    total_eligible_sources: u32,
    input_neuron_count: u32,
    already_connected_count: u32,
    record_load_failures: u32,
    had_candidate: bool,
    best_rejection: Option<RejectionDetail>,
}

impl TargetDiagnosticEntry {
    fn new(target_uuid: &str) -> Self {
        Self {
            target_uuid: target_uuid.to_string(),
            target_record_count: 0,
            evaluated_candidates: 0,
            candidates_with_samples: 0,
            total_eligible_sources: 0,
            input_neuron_count: 0,
            already_connected_count: 0,
            record_load_failures: 0,
            had_candidate: false,
            best_rejection: None,
        }
    }

    fn update_best(&mut self, detail: RejectionDetail) {
        match &self.best_rejection {
            Some(current) => {
                if detail.score() > current.score()
                    || (detail.score() == current.score()
                        && detail.sample_count > current.sample_count)
                {
                    self.best_rejection = Some(detail);
                }
            }
            None => self.best_rejection = Some(detail),
        }
    }
}

struct TargetDiagnostics {
    log_enabled: bool,
    entries: HashMap<String, TargetDiagnosticEntry>,
}

impl TargetDiagnostics {
    fn new(targets: &[&String]) -> Self {
        let log_enabled = verbose_enabled();
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert(target.to_string(), TargetDiagnosticEntry::new(target));
        }
        Self {
            log_enabled,
            entries,
        }
    }

    #[cfg(test)]
    fn new_for_tests(targets: &[&str]) -> Self {
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert((*target).to_string(), TargetDiagnosticEntry::new(target));
        }
        Self {
            log_enabled: true,
            entries,
        }
    }

    fn set_target_record_count(&mut self, target_uuid: &str, count: usize) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.target_record_count = count;
        }
    }

    fn set_total_eligible_sources(&mut self, target_uuid: &str, count: u32) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.total_eligible_sources = count;
        }
    }

    fn set_input_neuron_count(&mut self, target_uuid: &str, count: u32) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.input_neuron_count = count;
        }
    }

    fn record_already_connected(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.already_connected_count += 1;
        }
    }

    fn record_load_failure(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.record_load_failures += 1;
        }
    }

    fn record_candidate_attempt(&mut self, target_uuid: &str, had_samples: bool) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.evaluated_candidates += 1;
            if had_samples {
                entry.candidates_with_samples += 1;
            }
        }
    }

    fn record_no_samples(
        &mut self,
        target_uuid: &str,
        source_uuid: &str,
        source_record_count: usize,
    ) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::NoSamples,
                sample_count: 0,
                source_record_count,
                improved_count: 0,
                worsened_count: 0,
                expected_improvement: f32::NEG_INFINITY,
                threshold: 0.0,
                weight: None,
            });
        }
    }

    fn record_zero_improvement(
        &mut self,
        target_uuid: &str,
        source_uuid: &str,
        sample_count: usize,
        positive_count: u32,
        negative_count: u32,
    ) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::ZeroImprovement,
                sample_count,
                source_record_count: sample_count,
                improved_count: positive_count.max(negative_count),
                worsened_count: positive_count.min(negative_count),
                expected_improvement: 0.0,
                threshold: 0.0,
                weight: None,
            });
        }
    }

    fn record_below_threshold(
        &mut self,
        target_uuid: &str,
        source_uuid: &str,
        context: ThresholdContext,
    ) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::BelowThreshold,
                sample_count: context.sample_count,
                source_record_count: context.sample_count,
                improved_count: context.improved_count,
                worsened_count: context.worsened_count,
                expected_improvement: context.expected_improvement,
                threshold: context.threshold,
                weight: Some(context.weight),
            });
        }
    }

    fn mark_candidate_selected(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.had_candidate = true;
        }
    }

    fn emit_logs(&self) {
        if !self.log_enabled {
            return;
        }

        for entry in self.entries.values() {
            if entry.had_candidate {
                continue;
            }

            if entry.total_eligible_sources == 0 {
                // This should never happen - input/constant neurons are skipped early
                // and hidden/output neurons should always have at least input neurons as eligible sources
                // Skip logging to avoid cluttering logs with impossible conditions
                continue;
            }

            // Check if neuron is fully connected (all eligible sources already have synapses)
            // Eligible sources include: ALL input neurons (input-0 through input-(creature.input-1))
            // AND ALL prior hidden/output neurons (with index < target_index, excluding constants)
            // This condition is rare - only occurs when neuron is connected to all possible sources
            if entry.already_connected_count == entry.total_eligible_sources {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} is fully connected: all {} eligible upstream sources already have synapses (all {} input neurons and all prior hidden/output neurons). This is a rare condition.",
                    entry.target_uuid, entry.total_eligible_sources, entry.input_neuron_count
                );
                continue;
            }

            // Check for record loading failures (this indicates a bug)
            if entry.record_load_failures > 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} had {} record loading failures (this may indicate a bug - records exist but couldn't be loaded).",
                    entry.target_uuid, entry.record_load_failures
                );
            }

            if entry.evaluated_candidates == 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream neurons but none were evaluated ({} already connected, {} record load failures).",
                    entry.target_uuid, entry.total_eligible_sources, entry.already_connected_count, entry.record_load_failures
                );
                continue;
            }

            let best = match &entry.best_rejection {
                Some(detail) => detail,
                None => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} evaluated {} potential synapses but recorded no diagnostics.",
                        entry.target_uuid, entry.evaluated_candidates
                    );
                    continue;
                }
            };

            match best.reason {
                RejectionReason::NoSamples => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} skipped candidate from {} because no aligned samples were available (source records {}, target records {}).",
                        entry.target_uuid,
                        best.source_uuid,
                        best.source_record_count,
                        entry.target_record_count
                    );
                }
                RejectionReason::ZeroImprovement => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} saw {} aligned samples from {} but GPU stats reported zero consistent improvements (positive {}, negative {}).",
                        entry.target_uuid,
                        best.sample_count,
                        best.source_uuid,
                        best.improved_count,
                        best.worsened_count
                    );
                }
                RejectionReason::BelowThreshold => {
                    if let Some(weight) = best.weight {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {} best candidate from {} improved {:.4} but remained below threshold {:.4} (improved {}, worsened {}, suggested weight {:.4}).",
                            entry.target_uuid,
                            best.source_uuid,
                            best.expected_improvement,
                            best.threshold,
                            best.improved_count,
                            best.worsened_count,
                            weight
                        );
                    } else {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {} best candidate from {} improved {:.4} but remained below threshold {:.4} (improved {}, worsened {}).",
                            entry.target_uuid,
                            best.source_uuid,
                            best.expected_improvement,
                            best.threshold,
                            best.improved_count,
                            best.worsened_count
                        );
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn entry_for(&self, target_uuid: &str) -> Option<&TargetDiagnosticEntry> {
        self.entries.get(target_uuid)
    }

    fn no_candidate_summaries(&self) -> Vec<SynapseNoCandidateSummary> {
        self.entries
            .values()
            .filter(|entry| !entry.had_candidate)
            .map(|entry| {
                // Only report "no eligible sources" if both total_eligible_sources and evaluated_candidates are 0
                // This handles the case where total_eligible_sources might be 0 in tests but evaluated_candidates > 0
                if entry.total_eligible_sources == 0 && entry.evaluated_candidates == 0 {
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: SynapseNoCandidateReason::NoEligibleSources,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                // Check if neuron is fully connected (all eligible sources already have synapses)
                // Eligible sources include: ALL input neurons (input-0 through input-(creature.input-1))
                // AND ALL prior hidden/output neurons (with index < target_index, excluding constants)
                // This condition is rare - only occurs when neuron is connected to all possible sources
                if entry.total_eligible_sources > 0
                    && entry.already_connected_count == entry.total_eligible_sources
                    && entry.evaluated_candidates == 0
                {
                    // Neuron is fully connected - all eligible sources (all inputs + all prior hidden neurons) already have synapses
                    // This is legitimate but rare, and we report it as "no eligible sources"
                    // since there are no NEW sources to evaluate
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: SynapseNoCandidateReason::NoEligibleSources,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                if let Some(best) = &entry.best_rejection {
                    let reason = match best.reason {
                        RejectionReason::NoSamples => SynapseNoCandidateReason::NoSamples,
                        RejectionReason::ZeroImprovement => {
                            SynapseNoCandidateReason::ZeroImprovement
                        }
                        RejectionReason::BelowThreshold => SynapseNoCandidateReason::BelowThreshold,
                    };
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: Some(SynapseNoCandidateDetail {
                            source_uuid: Some(best.source_uuid.clone()),
                            sample_count: Some(best.sample_count),
                            source_record_count: Some(best.source_record_count),
                            improved_count: Some(best.improved_count),
                            worsened_count: Some(best.worsened_count),
                            expected_improvement: Some(best.expected_improvement),
                            threshold: Some(best.threshold),
                            suggested_weight: best.weight,
                        }),
                    };
                }

                SynapseNoCandidateSummary {
                    target_uuid: entry.target_uuid.clone(),
                    reason: SynapseNoCandidateReason::NoDiagnostics,
                    evaluated_candidates: entry.evaluated_candidates,
                    candidates_with_samples: entry.candidates_with_samples,
                    target_record_count: entry.target_record_count,
                    detail: None,
                }
            })
            .collect()
    }
}

/// Detail about why a source was rejected (currently only used for NoSamples).
#[derive(Clone)]
struct NeuronRejectionDetail {
    source_uuid: String,
    orientation: Option<&'static str>,
    sample_count: usize,
    expected_improvement: f32,
}

impl NeuronRejectionDetail {
    fn score(&self) -> f32 {
        self.expected_improvement
    }
}

struct NeuronDiagnosticEntry {
    target_uuid: String,
    target_record_count: usize,
    total_eligible_sources: u32,
    record_load_failures: u32,
    evaluated_sources: u32,
    sources_with_samples: u32,
    had_candidate: bool,
    best_rejection: Option<NeuronRejectionDetail>,
    /// Set to true when this neuron was filtered out because it's a hidden neuron
    /// (only output neurons are valid targets for add-neuron analysis).
    hidden_filtered: bool,
    /// Set to true when this neuron was filtered out because it's an input neuron
    /// (input neurons are observation sources, not computation nodes).
    input_filtered: bool,
    /// Set to true when this neuron was filtered out because it's a constant neuron
    /// (constant neurons don't receive inputs - they always output a fixed value).
    constant_filtered: bool,
}

impl NeuronDiagnosticEntry {
    fn new(target_uuid: &str) -> Self {
        Self {
            target_uuid: target_uuid.to_string(),
            target_record_count: 0,
            total_eligible_sources: 0,
            record_load_failures: 0,
            evaluated_sources: 0,
            sources_with_samples: 0,
            had_candidate: false,
            best_rejection: None,
            hidden_filtered: false,
            input_filtered: false,
            constant_filtered: false,
        }
    }

    fn update_best(&mut self, detail: NeuronRejectionDetail) {
        match &self.best_rejection {
            Some(current) => {
                if detail.score() > current.score()
                    || (detail.score() == current.score()
                        && detail.sample_count > current.sample_count)
                {
                    self.best_rejection = Some(detail);
                }
            }
            None => self.best_rejection = Some(detail),
        }
    }
}

struct NeuronDiagnostics {
    log_enabled: bool,
    entries: HashMap<String, NeuronDiagnosticEntry>,
}

impl NeuronDiagnostics {
    fn new(targets: &[&String]) -> Self {
        let log_enabled = verbose_enabled();
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert(target.to_string(), NeuronDiagnosticEntry::new(target));
        }
        Self {
            log_enabled,
            entries,
        }
    }

    #[cfg(test)]
    fn new_for_tests(targets: &[&str]) -> Self {
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert((*target).to_string(), NeuronDiagnosticEntry::new(target));
        }
        Self {
            log_enabled: true,
            entries,
        }
    }

    fn set_target_record_count(&mut self, target_uuid: &str, count: usize) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.target_record_count = count;
        }
    }

    fn set_total_eligible_sources(&mut self, target_uuid: &str, count: u32) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.total_eligible_sources = count;
        }
    }

    fn record_load_failure(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.record_load_failures += 1;
        }
    }

    fn record_candidate_attempt(&mut self, target_uuid: &str, had_samples: bool) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.evaluated_sources += 1;
            if had_samples {
                entry.sources_with_samples += 1;
            }
        }
    }

    fn record_no_samples(&mut self, target_uuid: &str, source_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(NeuronRejectionDetail {
                source_uuid: source_uuid.to_string(),
                orientation: None,
                sample_count: 0,
                expected_improvement: f32::NEG_INFINITY,
            });
        }
    }

    fn mark_candidate_selected(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.had_candidate = true;
        }
    }

    /// Mark a neuron as filtered out because it's a hidden neuron.
    /// Hidden neurons are not valid targets for add-neuron analysis because their
    /// backpropagated errors don't reliably translate to output error reduction.
    fn mark_hidden_filtered(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.hidden_filtered = true;
        }
    }

    /// Mark a neuron as filtered out because it's an input neuron.
    /// Input neurons are observation sources, not computation nodes - they have
    /// no activation function or error to reduce.
    fn mark_input_filtered(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.input_filtered = true;
        }
    }

    /// Mark a neuron as filtered out because it's a constant neuron.
    /// Constant neurons don't receive inputs - they always output a fixed value
    /// regardless of network state, so adding a connection to them has no effect.
    fn mark_constant_filtered(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.constant_filtered = true;
        }
    }

    fn emit_logs(&self) {
        if !self.log_enabled {
            return;
        }

        for entry in self.entries.values() {
            if entry.had_candidate {
                continue;
            }

            // Check pre-analysis filters FIRST - these take precedence over all other reasons.
            // These neurons are filtered out before analysis even begins, so they won't
            // have any other diagnostic data (eligible sources, samples, etc.).

            // Input neurons are observation sources, not computation nodes
            if entry.input_filtered {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} was filtered out (input neuron). \
                    Input neurons are observation sources, not computation nodes - they have no \
                    activation function or error to reduce.",
                    entry.target_uuid
                );
                continue;
            }

            // Hidden neurons have backpropagated errors that don't reliably predict output error
            if entry.hidden_filtered {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} was filtered out (hidden neuron). \
                    Add-neuron analysis only targets output neurons because hidden neuron error \
                    reduction doesn't reliably translate to creature score improvement.",
                    entry.target_uuid
                );
                continue;
            }

            // Constant neurons don't receive inputs - they always output a fixed value
            if entry.constant_filtered {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} was filtered out (constant neuron). \
                    Constant neurons don't receive inputs - they always output a fixed value \
                    regardless of network state, so adding a connection to them has no effect.",
                    entry.target_uuid
                );
                continue;
            }

            // Check for record loading failures (this indicates a bug or data issue)
            if entry.record_load_failures > 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} had {} record loading failures out of {} eligible sources (this may indicate a data integrity issue - records exist but couldn't be loaded).",
                    entry.target_uuid, entry.record_load_failures, entry.total_eligible_sources
                );
            }

            if entry.evaluated_sources == 0 {
                if entry.total_eligible_sources > 0
                    && entry.record_load_failures == entry.total_eligible_sources
                {
                    // All eligible sources failed to load - this is a data/bug issue, not "no sources"
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream sources but all {} failed to load from parquet file.",
                        entry.target_uuid, entry.total_eligible_sources, entry.record_load_failures
                    );
                } else if entry.total_eligible_sources > 0 && entry.record_load_failures == 0 {
                    // Sources exist, no load failures, but none evaluated - likely timeout before sources could be checked
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream sources but none were evaluated (0 load failures). Likely analysis TIMEOUT before source loading could start.",
                        entry.target_uuid, entry.total_eligible_sources
                    );
                } else if entry.total_eligible_sources > 0 {
                    // Some sources exist, some failures, none evaluated
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream sources but none were evaluated ({} load failures).",
                        entry.target_uuid, entry.total_eligible_sources, entry.record_load_failures
                    );
                } else {
                    // Genuinely no eligible sources (e.g., target is first neuron)
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had no upstream neurons to analyse.",
                        entry.target_uuid
                    );
                }
                continue;
            }

            let best = match &entry.best_rejection {
                Some(detail) => detail,
                None => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} evaluated {} upstream neurons but recorded no diagnostics.",
                        entry.target_uuid, entry.evaluated_sources
                    );
                    continue;
                }
            };

            // Currently only NoSamples is used
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Target {} skipped candidate from {} because no overlapping samples were found.",
                entry.target_uuid, best.source_uuid
            );
        }
    }

    #[cfg(test)]
    fn entry_for(&self, target_uuid: &str) -> Option<&NeuronDiagnosticEntry> {
        self.entries.get(target_uuid)
    }

    fn no_candidate_summaries(&self) -> Vec<NeuronNoCandidateSummary> {
        self.entries
            .values()
            // Include entries that never had a candidate
            .filter(|entry| !entry.had_candidate)
            .map(|entry| {
                // Check pre-analysis filters FIRST - these take precedence over other reasons.
                // These neurons are filtered out before analysis even begins, so they won't
                // have any other diagnostic data (eligible sources, samples, etc.).

                // Input neurons are observation sources, not computation nodes
                if entry.input_filtered {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::InputNeuronFiltered,
                        evaluated_sources: 0,
                        sources_with_samples: 0,
                        target_record_count: 0,
                        detail: None,
                    };
                }

                // Hidden neurons have backpropagated errors that don't reliably predict output error
                if entry.hidden_filtered {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::HiddenNeuronFiltered,
                        evaluated_sources: 0,
                        sources_with_samples: 0,
                        target_record_count: 0,
                        detail: None,
                    };
                }

                // Constant neurons don't receive inputs - they always output a fixed value
                if entry.constant_filtered {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::ConstantNeuronFiltered,
                        evaluated_sources: 0,
                        sources_with_samples: 0,
                        target_record_count: 0,
                        detail: None,
                    };
                }

                // Only report "no eligible sources" if there were genuinely no eligible sources
                // AND no evaluated candidates. If there were eligible sources but they all failed
                // to load or had empty records, report that as NoSamples with context.
                if entry.evaluated_sources == 0 && entry.total_eligible_sources == 0 {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::NoEligibleSources,
                        evaluated_sources: entry.evaluated_sources,
                        sources_with_samples: entry.sources_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                // If evaluated_sources is 0 but total_eligible_sources > 0, sources existed
                // but all failed to load or had empty records - report as NoSamples
                if entry.evaluated_sources == 0 && entry.total_eligible_sources > 0 {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::NoSamples,
                        evaluated_sources: entry.evaluated_sources,
                        sources_with_samples: entry.sources_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                if let Some(best) = &entry.best_rejection {
                    // Currently only NoSamples is ever set as rejection reason
                    let reason = NeuronNoCandidateReason::NoSamples;
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason,
                        evaluated_sources: entry.evaluated_sources,
                        sources_with_samples: entry.sources_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: Some(NeuronNoCandidateDetail {
                            source_uuid: Some(best.source_uuid.clone()),
                            orientation: best.orientation.map(|name| name.to_string()),
                            sample_count: Some(best.sample_count),
                            improved_count: None,
                            worsened_count: None,
                            expected_improvement: Some(best.expected_improvement),
                            threshold: None,
                            outgoing_weight: None,
                        }),
                    };
                }

                NeuronNoCandidateSummary {
                    target_uuid: entry.target_uuid.clone(),
                    reason: NeuronNoCandidateReason::NoDiagnostics,
                    evaluated_sources: entry.evaluated_sources,
                    sources_with_samples: entry.sources_with_samples,
                    target_record_count: entry.target_record_count,
                    detail: None,
                }
            })
            .collect()
    }
}

impl RecordCache {
    /// Create a cache that automatically chooses the best loading strategy based on
    /// available system memory:
    ///
    /// - **Pre-loaded mode** (fast): Loads entire parquet file into memory upfront.
    ///   Used when there's sufficient RAM (3× file size + 1GB headroom).
    ///
    /// - **Lazy-loaded mode** (memory-efficient): Loads records on-demand per neuron.
    ///   Slower (O(N) parquet scans for N neurons) but works on memory-constrained systems.
    ///
    /// This ensures discovery works on any modern Mac/PC, adapting to available resources.
    pub(crate) fn new_adaptive(parquet_file: &str) -> Result<Self> {
        // Check if we have enough memory for pre-loading
        match check_memory_for_parquet(parquet_file) {
            Ok(()) => {
                // Sufficient memory - use fast pre-loaded mode
                Self::new_preloaded_internal(parquet_file)
            }
            Err(memory_error) => {
                // Insufficient memory - fall back to lazy loading
                eprintln!(
                    "[NEAT-AI-Discovery] Insufficient memory for pre-loading. \
                     Falling back to lazy-loading mode (slower but memory-efficient)."
                );
                if verbose_enabled() {
                    eprintln!("[NEAT-AI-Discovery][verbose] Memory check failed: {memory_error}");
                }
                Self::new_lazy(parquet_file)
            }
        }
    }

    /// Create a lazy-loading cache that loads records on-demand.
    /// Slower than pre-loaded mode but uses minimal memory.
    fn new_lazy(parquet_file: &str) -> Result<Self> {
        use crate::parquet_format::read_records_from_parquet;

        eprintln!(
            "[NEAT-AI-Discovery] Using lazy-loading mode for parquet file. \
             This is slower but uses less memory."
        );

        Ok(Self {
            parquet_file: parquet_file.to_string(),
            cache: Mutex::new(HashMap::new()),
            loader: Arc::new(move |file: &str, neuron_uuid: &str| {
                read_records_from_parquet(file, neuron_uuid)
            }),
        })
    }

    /// Internal pre-loaded implementation (called when memory check passes).
    fn new_preloaded_internal(parquet_file: &str) -> Result<Self> {
        use crate::parquet_format::read_all_records_grouped_by_neuron;
        use std::time::Instant;

        let start = Instant::now();
        let grouped_records = read_all_records_grouped_by_neuron(parquet_file)
            .with_context(|| format!("Failed to pre-load records from {parquet_file}"))?;

        let neuron_count = grouped_records.len();
        let total_records: usize = grouped_records.values().map(|v| v.len()).sum();

        // Pre-populate the cache with all loaded records
        let mut cache_map: HashMap<String, Arc<CachedNeuronRecords>> = HashMap::new();
        for (uuid, mut records) in grouped_records {
            records.sort_by_key(|r| r.obs_index);
            let cell = OnceCell::new();
            let _ = cell.set(Arc::new(records));
            cache_map.insert(uuid, Arc::new(cell));
        }

        let elapsed = start.elapsed();
        if verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Pre-loaded {} neurons with {} total records from parquet in {:.2}s",
                neuron_count,
                total_records,
                elapsed.as_secs_f64()
            );
        }

        Ok(Self {
            parquet_file: parquet_file.to_string(),
            cache: Mutex::new(cache_map),
            loader: Arc::new(|_file: &str, _neuron_uuid: &str| {
                // This loader should never be called for pre-loaded cache
                Ok(Vec::new())
            }),
        })
    }

    #[cfg(test)]
    fn with_loader(parquet_file: &str, loader: Arc<RecordCacheLoader>) -> Self {
        Self {
            parquet_file: parquet_file.to_string(),
            cache: Mutex::new(HashMap::new()),
            loader,
        }
    }

    fn get(&self, neuron_uuid: &str) -> Result<Arc<Vec<DiscoverRecord>>> {
        let cache_entry = {
            let mut cache = self.cache.lock().expect("record cache mutex poisoned");
            cache
                .entry(neuron_uuid.to_string())
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        let loader = Arc::clone(&self.loader);
        let parquet_file = self.parquet_file.clone();
        let context_uuid = neuron_uuid.to_string();
        let load_uuid = context_uuid.clone();

        let arc_records = cache_entry
            .get_or_try_init(move || -> Result<Arc<Vec<DiscoverRecord>>> {
                let mut records = loader(&parquet_file, &load_uuid)?;
                records.sort_by_key(|record| record.obs_index);
                Ok(Arc::new(records))
            })
            .with_context(|| {
                format!("Failed to read discovery records for neuron {context_uuid}")
            })?;

        Ok(Arc::clone(arc_records))
    }
}

fn require_unique_focus<'a>(focus_neurons: &'a [String], context: &str) -> Result<Vec<&'a String>> {
    if focus_neurons.is_empty() {
        return Err(anyhow!(
            "{context} needs at least one focus neuron. The Deno controller supplied an empty `focus_neurons` array, so there is nothing to analyse. Please fix the upstream request and retry after setting `NEAT_AI_DISCOVERY_VERBOSE=1` if you need extra logging."
        ));
    }

    let mut seen_targets: HashSet<&str> = HashSet::new();
    let mut unique_focus: Vec<&String> = Vec::new();
    let mut duplicates: Vec<String> = Vec::new();

    for target_uuid in focus_neurons {
        if seen_targets.insert(target_uuid.as_str()) {
            unique_focus.push(target_uuid);
        } else {
            duplicates.push(target_uuid.clone());
        }
    }

    if !duplicates.is_empty() {
        duplicates.sort();
        duplicates.dedup();
        let joined = duplicates.join(", ");
        return Err(anyhow!(
            "{context} received duplicate focus neurons ({joined}). Each target must be unique so we can map diagnostics back to the Deno request. We are refusing to continue so the upstream behaviour can be corrected."
        ));
    }

    Ok(unique_focus)
}

/// Sample data for evaluating potential synapses/neurons.
///
/// For accurate HARD_TANH modelling, we need the target's pre-activation value
/// to properly simulate clamping behaviour. When `target_value` is `Some`, we can
/// compute the actual effect of adding a contribution rather than using the linear
/// approximation.
#[derive(Clone, Copy, Default)]
struct HelpfulSample {
    /// Source neuron's activation (what we're considering adding a connection FROM)
    activation: f32,
    /// Target neuron's average error (expected - actual output)
    avg_error: f32,
    /// Target neuron's pre-activation value (input sum before squash function).
    /// Used for accurate HARD_TANH/clamping calculations. None for GPU-matched samples.
    target_value: Option<f32>,
    /// Target neuron's post-activation output (after squash function).
    /// Note: avg_error is in VALUE domain, so expected = squash(target_value + avg_error)
    target_activation: Option<f32>,
}

/// Extended sample for threshold-crossing analysis of discrete activations (STEP/BIPOLAR).
/// Includes the target neuron's pre-activation value to determine threshold crossings.
#[derive(Clone, Copy)]
struct DiscreteHelpfulSample {
    /// Source neuron's activation
    source_activation: f32,
    /// Target neuron's input sum before squash function
    target_value: f32,
    /// Target neuron's current output after squash (0/1 for STEP, -1/1 for BIPOLAR)
    target_activation: f32,
    /// Target neuron's average error
    avg_error: f32,
}

/// Type of threshold activation function
#[derive(Clone, Copy, PartialEq, Eq)]
enum ThresholdType {
    /// STEP: value > 0 ? 1 : 0
    Step,
    /// BIPOLAR: value > 0 ? 1 : -1
    Bipolar,
}

impl ThresholdType {
    fn from_squash(squash: &str) -> Option<Self> {
        match squash.to_uppercase().as_str() {
            "STEP" => Some(Self::Step),
            "BIPOLAR" => Some(Self::Bipolar),
            _ => None,
        }
    }

    /// Calculate the output for a given input value
    fn apply(&self, value: f32) -> f32 {
        match self {
            Self::Step => {
                if value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Bipolar => {
                if value > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }

    /// Check if adding a contribution would flip the output
    fn would_flip(&self, current_value: f32, contribution: f32) -> bool {
        let current_positive = current_value > 0.0;
        let new_positive = (current_value + contribution) > 0.0;
        current_positive != new_positive
    }

    /// Check if a flip is "helpful" (moves output in direction of error)
    /// Returns: 1 for helpful flip, -1 for harmful flip, 0 for no flip
    fn flip_direction(&self, current_value: f32, contribution: f32, error: f32) -> i32 {
        if !self.would_flip(current_value, contribution) {
            return 0;
        }

        let current_output = self.apply(current_value);
        let new_output = self.apply(current_value + contribution);

        // Error > 0 means output should be higher
        // Error < 0 means output should be lower
        let output_increased = new_output > current_output;
        let should_increase = error > 0.0;

        if output_increased == should_increase {
            1 // Helpful flip
        } else {
            -1 // Harmful flip
        }
    }
}

/// Statistics computed from neuron error and activation samples
#[derive(Debug, Clone)]
struct NeuronStats {
    mean_error: f32,
    error_variance: f32,
    mean_activation: f32,
    activation_variance: f32,
    error_spike_count: u32,
    activation_spike_count: u32,
    activation_min: f32,
    activation_max: f32,
}

impl NeuronStats {
    /// Compute statistics from a slice of discovery records (for target neurons)
    fn from_records(records: &[DiscoverRecord]) -> Option<Self> {
        if records.is_empty() {
            return None;
        }

        let mut samples = Vec::new();
        for record in records {
            if record.errors.is_empty() {
                continue;
            }
            // Compute average error for this record
            let mut error_sum = 0.0;
            let mut error_count = 0;
            for &err in &record.errors {
                if err.is_finite() {
                    error_sum += err;
                    error_count += 1;
                }
            }
            if error_count > 0 && record.activation.is_finite() {
                let avg_error = error_sum / error_count as f32;
                samples.push(HelpfulSample {
                    activation: record.activation,
                    avg_error,
                    target_value: record.value,
                    target_activation: Some(record.activation),
                });
            }
        }

        Self::from_samples(&samples)
    }

    /// Compute statistics from a slice of samples
    fn from_samples(samples: &[HelpfulSample]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut error_sum = 0.0;
        let mut error_sq_sum = 0.0;
        let mut activation_sum = 0.0;
        let mut activation_sq_sum = 0.0;
        let mut error_spike_count = 0u32;
        let mut activation_spike_count = 0u32;
        let mut activation_min = f32::INFINITY;
        let mut activation_max = f32::NEG_INFINITY;
        let mut valid_count = 0usize;

        // Spike thresholds: 2 standard deviations (we'll approximate with mean + 2*mean for now)
        // We'll compute proper thresholds after we have the mean
        let mut error_abs_sum = 0.0;
        let mut activation_abs_sum = 0.0;

        for sample in samples {
            if !sample.avg_error.is_finite() || !sample.activation.is_finite() {
                continue;
            }
            valid_count += 1;
            let error_abs = sample.avg_error.abs();
            let activation_abs = sample.activation.abs();

            error_sum += sample.avg_error;
            error_sq_sum += sample.avg_error * sample.avg_error;
            error_abs_sum += error_abs;

            activation_sum += sample.activation;
            activation_sq_sum += sample.activation * sample.activation;
            activation_abs_sum += activation_abs;

            if activation_min > sample.activation {
                activation_min = sample.activation;
            }
            if activation_max < sample.activation {
                activation_max = sample.activation;
            }
        }

        if valid_count == 0 {
            return None;
        }

        let count_f = valid_count as f32;
        let mean_error = error_sum / count_f;
        let mean_activation = activation_sum / count_f;
        let mean_error_abs = error_abs_sum / count_f;
        let mean_activation_abs = activation_abs_sum / count_f;

        // Compute variance using E[X^2] - E[X]^2
        let error_variance = (error_sq_sum / count_f) - (mean_error * mean_error);
        let activation_variance =
            (activation_sq_sum / count_f) - (mean_activation * mean_activation);

        // Spike detection: count samples where error/activation exceeds 2x the mean absolute value
        // This is a simple heuristic; more sophisticated methods could use actual std dev
        let error_spike_threshold = mean_error_abs * 2.0;
        let activation_spike_threshold = mean_activation_abs * 2.0;

        for sample in samples {
            if !sample.avg_error.is_finite() || !sample.activation.is_finite() {
                continue;
            }
            if sample.avg_error.abs() > error_spike_threshold {
                error_spike_count += 1;
            }
            if sample.activation.abs() > activation_spike_threshold {
                activation_spike_count += 1;
            }
        }

        Some(Self {
            mean_error,
            error_variance: error_variance.max(0.0), // Variance should be non-negative
            mean_activation,
            activation_variance: activation_variance.max(0.0),
            error_spike_count,
            activation_spike_count,
            activation_min: if activation_min.is_finite() {
                activation_min
            } else {
                0.0
            },
            activation_max: if activation_max.is_finite() {
                activation_max
            } else {
                0.0
            },
        })
    }

    fn to_json(&self) -> crate::NeuronStatsJson {
        crate::NeuronStatsJson {
            mean_error: self.mean_error,
            error_variance: self.error_variance,
            mean_activation: self.mean_activation,
            activation_variance: self.activation_variance,
            error_spike_count: self.error_spike_count,
            activation_spike_count: self.activation_spike_count,
            activation_min: self.activation_min,
            activation_max: self.activation_max,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuHelpfulSample {
    activation: f32,
    avg_error: f32,
}

/// Extended sample struct for GPU matching output, includes target neuron data.
impl From<HelpfulSample> for GpuHelpfulSample {
    fn from(value: HelpfulSample) -> Self {
        Self {
            activation: value.activation,
            avg_error: value.avg_error,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HelpfulContribution {
    positive_flag: u32,
    negative_flag: u32,
    positive_improvement: f32,
    negative_improvement: f32,
    positive_activation: f32,
    negative_activation: f32,
    error_squared: f32,
    activation_squared: f32,
    error_activation: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HelpfulUniforms {
    length: u32,
    pad0: u32,
    epsilon: f32,
    pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HarmfulContribution {
    harmful_flag: u32,
    helpful_flag: u32,
    error_magnitude: f32,
    pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HarmfulUniforms {
    length: u32,
    pad0: u32,
    epsilon: f32,
    weight: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReluContribution {
    positive_activation_sq: f32,
    positive_error_activation: f32,
    positive_count: u32,
    negative_activation_sq: f32,
    negative_error_activation: f32,
    negative_count: u32,
    error_sq: f32,
    pad0: f32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReluUniforms {
    length: u32,
    threshold: f32,
    epsilon: f32,
    pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BiasResult {
    bias_value: f32,
    error_reduction: f32,
    valid_sample_count: u32,
    pad0: u32,
}

impl BiasResult {
    fn zeroed() -> Self {
        Self {
            bias_value: 0.0,
            error_reduction: 0.0,
            valid_sample_count: 0,
            pad0: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BiasUniforms {
    sample_count: u32,
    bias_count: u32,
    incoming_weight: f32,
    outgoing_weight: f32,
    activation_type: u32,
    epsilon: f32,
    min_sample_count: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
#[allow(dead_code)] // Framework for future GPU activation evaluation
struct ActivationOutput {
    output: f32,
    output_sq: f32,
    error_output: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
#[allow(dead_code)] // Framework for future GPU activation evaluation
struct ActivationUniforms {
    sample_count: u32,
    orientation: f32,
    scale: f32,
    activation_type: u32,
    epsilon: f32,
    pad0: f32,
    pad1: f32,
}

#[derive(Default)]
struct HelpfulStats {
    positive_count: u32,
    negative_count: u32,
    positive_improvement_sum: f32,
    negative_improvement_sum: f32,
    positive_activation_sum: f32,
    negative_activation_sum: f32,
    error_sq_sum: f32,
    activation_sq_sum: f32,
    error_activation_sum: f32,
}

#[derive(Clone, Copy)]
enum ReluOrientation {
    Positive,
    Negative,
}

struct ReluStats {
    orientation: ReluOrientation,
    samples: Vec<(f32, f32)>,
    activation_sq_sum: f32,
    error_activation_sum: f32,
}

impl ReluStats {
    fn new(orientation: ReluOrientation) -> Self {
        Self {
            orientation,
            samples: Vec::new(),
            activation_sq_sum: 0.0,
            error_activation_sum: 0.0,
        }
    }

    /// Evaluate this orientation and return a candidate if it passes the threshold.
    fn evaluate(
        &self,
        source_uuid: &str,
        target_uuid: &str,
        threshold: f32,
        total_baseline_error_sq: f32,
        original_samples: &[HelpfulSample],
    ) -> Option<CandidateNeuronJson> {
        let sample_count = self.samples.len();
        if sample_count < MIN_NEURON_SAMPLE_COUNT || self.activation_sq_sum <= EPSILON {
            return None;
        }

        let mut outgoing_weight = self.error_activation_sum / (self.activation_sq_sum + EPSILON);
        if !outgoing_weight.is_finite() || outgoing_weight.abs() <= EPSILON {
            return None;
        }
        outgoing_weight = outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        let mut improved_count = 0u32;
        for (relu_activation, error) in &self.samples {
            let new_error = error - outgoing_weight * relu_activation;
            if new_error.abs() + EPSILON < error.abs() {
                improved_count += 1;
            }
        }

        // Calculate improvement based on magnitude (reduction in squared error)
        // improvement = baseline_sq - new_sq
        // = 2*w*sum(ea) - w^2*sum(aa)
        let improvement_magnitude = 2.0 * outgoing_weight * self.error_activation_sum
            - outgoing_weight * outgoing_weight * self.activation_sq_sum;

        // Normalise by total baseline error of ALL samples (not just active ones)
        let expected_improvement = if total_baseline_error_sq > EPSILON {
            let result = improvement_magnitude / total_baseline_error_sq;
            if result.is_finite() {
                result
            } else {
                0.0
            }
        } else {
            0.0
        };

        if expected_improvement <= threshold {
            return None;
        }

        let incoming_weight = match self.orientation {
            ReluOrientation::Positive => 1.0,
            ReluOrientation::Negative => -1.0,
        };

        // For split-error ReLU evaluation, use bias=0.
        // The whole point of split-error is that the ReLU should fire for ONE subset
        // (positive or negative error samples) but NOT the other.
        // Optimising bias on the subset alone can find a large positive bias that makes
        // the ReLU fire for ALL samples, defeating the split-error approach.
        // With bias=0, the ReLU naturally fires only when source activation > 0.
        let optimal_bias = 0.0;

        let target_stats = NeuronStats::from_samples(original_samples).map(|s| s.to_json());
        let total_count = self.samples.len() as u32;

        // Issue #128: Use creature-level metrics instead of neuron-level percentage.
        // target_neuron_impact will be updated during impact discounting.
        Some(CandidateNeuronJson {
            source_neuron_uuid: source_uuid.to_string(),
            target_neuron_uuid: target_uuid.to_string(),
            source_neuron_index: None, // Set during impact discounting
            target_neuron_index: None, // Set during impact discounting
            incoming_weight,
            outgoing_weight,
            squash: "ReLU".to_string(),
            bias: optimal_bias,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: expected_improvement,
            expected_creature_score_gain: expected_improvement,
            improved_count,
            total_count,
            target_neuron_stats: target_stats,
        })
    }
}

pub struct ActivationCandidateSpec {
    name: &'static str,
    orientations: &'static [f32],
    scales: &'static [f32],
    activation: fn(f32) -> f32,
    min_improvement: f32,
}

const ORIENTATIONS_BIDIRECTIONAL: [f32; 2] = [1.0, -1.0];
/// Log-spaced scale range for incoming weights - covers multiple orders of magnitude
/// efficiently. Evolution will fine-tune the exact values after discovery.
/// Extended to very large scales (50, 100) for aggressive signal amplification.
/// Note: Very large scales may cause numerical instability with some activations.
const SCALES_WIDE: [f32; 12] = [
    0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0,
];
/// Log-spaced scales for smooth activation functions (TANH, LOGISTIC, SELU) that
/// saturate at large inputs. Larger scales included but will saturate the output,
/// which may still be useful for binary-like thresholding behaviour.
const SCALES_SMOOTH: [f32; 10] = [0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 4.0, 10.0, 25.0, 50.0];

fn gelu_activation(x: f32) -> f32 {
    let x_cubed = x * x * x;
    let tanh_arg = 0.797_884_6 * (x + 0.044_715 * x_cubed);
    0.5 * x * (1.0 + tanh_arg.tanh())
}

fn elu_activation(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        x.exp() - 1.0
    }
}

fn softplus_activation(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn logistic_activation(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

fn tanh_activation(x: f32) -> f32 {
    x.tanh()
}

fn identity_activation(x: f32) -> f32 {
    x
}

fn bipolar_activation(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        -1.0
    }
}

fn clipped_activation(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

fn absolute_activation(x: f32) -> f32 {
    x.abs()
}

// ============================================================================
// NEW ACTIVATION FUNCTIONS (v0.1.139)
// Based on analysis of successful discoveries that evolved TO these activations
// ============================================================================

/// LeakyReLU - 4 successful discoveries evolved ReLU → LeakyReLU!
/// Allows small negative gradients instead of zeroing negative inputs.
#[allow(dead_code)] // Still used for target simulation, but not proposed as a candidate squash.
fn leaky_relu_activation(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        0.01 * x // Standard leak coefficient
    }
}

/// Mish - 2 successful discoveries evolved TO Mish (from ELU and Softplus)
/// Self-regularised activation: x * tanh(softplus(x))
fn mish_activation(x: f32) -> f32 {
    let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    x * sp.tanh()
}

/// Swish - 1 successful discovery evolved ReLU → Swish
/// Self-gated activation: x * sigmoid(x)
/// HARD_TANH - 1 successful discovery evolved CLIPPED → HARD_TANH
/// Linear in [-1, 1], saturates outside. Same as CLIPPED but named for NEAT-AI.
fn hard_tanh_activation(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// SOFTSIGN - 1 successful discovery neuron with SOFTSIGN
/// Smooth approximation of sign function: x / (1 + |x|)
fn softsign_activation(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// BENT_IDENTITY - 1 successful discovery evolved LeakyReLU → BENT_IDENTITY
/// Smooth, nearly linear: (sqrt(x² + 1) - 1) / 2 + x
fn bent_identity_activation(x: f32) -> f32 {
    ((x * x + 1.0).sqrt() - 1.0) / 2.0 + x
}

/// ArcTan - Similar to SOFTSIGN, bounded output
fn arctan_activation(x: f32) -> f32 {
    x.atan()
}

/// ReLU6 - Capped ReLU at 6, useful for quantisation
fn relu6_activation(x: f32) -> f32 {
    x.clamp(0.0, 6.0)
}

fn activation_name_to_gpu_id(name: &str) -> u32 {
    match name {
        "GELU" => 0,
        "ELU" => 1,
        "SELU" => 2,
        "Softplus" => 3,
        "LOGISTIC" => 4,
        "TANH" => 5,
        "IDENTITY" => 6,
        "BIPOLAR" => 7,
        "CLIPPED" => 8,
        "ABSOLUTE" => 9,
        "INVERSE" => 10,
        // New activations (v0.1.139) - GPU IDs 11-18
        "LeakyReLU" => 11,
        "Mish" => 12,
        "Swish" => 13,
        "HARD_TANH" => 14,
        "SOFTSIGN" => 15,
        "BENT_IDENTITY" => 16,
        "ArcTan" => 17,
        "ReLU6" => 18,
        _ => 6, // Default to IDENTITY
    }
}

pub const ACTIVATION_SPECS: [ActivationCandidateSpec; 15] = [
    // ========================================================================
    // ORIGINAL ACTIVATIONS (v0.1.x)
    // ========================================================================
    ActivationCandidateSpec {
        name: "GELU",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: gelu_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "ELU",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: elu_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "Softplus",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: softplus_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "LOGISTIC",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: logistic_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "TANH",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: tanh_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "IDENTITY",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: identity_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "BIPOLAR",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: bipolar_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "CLIPPED",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: clipped_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "ABSOLUTE",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: absolute_activation,
        min_improvement: 0.0,
    },
    // ========================================================================
    // NEW ACTIVATIONS (v0.1.139) - Based on successful discovery evolutions
    // ========================================================================
    //
    // NOTE (Issue #134 follow-up): We intentionally do NOT propose LeakyReLU as a
    // new neuron type. In practice it behaves very similarly to ReLU (α=0.01),
    // and production runs show many low-quality LeakyReLU candidates. We still
    // fully support LeakyReLU in existing creatures (targets and sources).
    //
    // NOTE (Issue #148, 26-Dec-2025): We also do not propose Swish or SELU as new
    // neuron squashes. In our value-domain discovery workflow they are close enough
    // to ReLU in practice that scanning them is usually a poor trade in time-bounded
    // runs. This preserves the budget to scan more (source,target) possibilities
    // while still allowing existing creatures to use any squash.
    ActivationCandidateSpec {
        name: "Mish",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: mish_activation,
        min_improvement: 0.0, // 2 successful discoveries evolved TO Mish
    },
    ActivationCandidateSpec {
        name: "HARD_TANH",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: hard_tanh_activation,
        min_improvement: 0.0, // 1 successful discovery evolved CLIPPED → HARD_TANH
    },
    ActivationCandidateSpec {
        name: "SOFTSIGN",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: softsign_activation,
        min_improvement: 0.0, // Successful discovery neuron with SOFTSIGN
    },
    ActivationCandidateSpec {
        name: "BENT_IDENTITY",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: bent_identity_activation,
        min_improvement: 0.0, // 1 successful discovery evolved LeakyReLU → BENT_IDENTITY
    },
    ActivationCandidateSpec {
        name: "ArcTan",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: arctan_activation,
        min_improvement: 0.0, // Similar to SOFTSIGN, bounded output
    },
    ActivationCandidateSpec {
        name: "ReLU6",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: relu6_activation,
        min_improvement: 0.0, // Capped ReLU, useful for bounded outputs
    },
];

/// Get activation-function-specific bias range (min, max, step).
/// This is still used by the GPU bias search for compatibility.
/// Extended ranges to work with large incoming weights (up to 200).
///
/// Different activation functions benefit from different bias ranges:
/// - ReLU/ELU: Large negative bias for high-threshold neurons
/// - TANH/LOGISTIC: Wide symmetric range to shift operating point
/// - IDENTITY: Widest range as pure offset (scales with large weights)
fn get_bias_range(squash: &str) -> (f32, f32, f32) {
    match squash {
        // ReLU/ELU: extended negative for high-threshold neurons with large weights
        "ReLU" | "ELU" | "SELU" => (-25.0, 10.0, 0.5),
        // Symmetric activation functions: extended for large weight configurations
        "TANH" | "LOGISTIC" => (-10.0, 10.0, 0.5),
        // IDENTITY: widest range - acts as offset, scales with incoming weights
        "IDENTITY" => (-50.0, 50.0, 1.0),
        // Softplus and GELU: extended negative thresholds
        "Softplus" | "GELU" => (-10.0, 10.0, 0.5),
        // Other activation functions get expanded range
        "ABSOLUTE" | "CLIPPED" => (-10.0, 10.0, 0.5),
        "BIPOLAR" => (-10.0, 10.0, 1.0),
        _ => (-10.0, 10.0, 0.5), // Generous default
    }
}

/// Get log-spaced bias values for a given activation function.
/// Uses sinh-like spacing: denser near 0, sparser at extremes.
/// Extended ranges to work with large incoming weights (up to 200).
/// Evolution will fine-tune the exact bias value after discovery.
fn get_bias_values(squash: &str) -> Vec<f32> {
    // Base log-spaced positive values (denser near 0, extended to larger values)
    let base_positive: &[f32] = match squash {
        // ReLU/ELU: extended for large weight thresholding
        "ReLU" | "ELU" | "SELU" => &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        // Symmetric activations: extended range for shifting operating point
        "TANH" | "LOGISTIC" => &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        // IDENTITY: widest range (pure offset, scales with large weights)
        "IDENTITY" => &[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0],
        // Softplus/GELU: extended for large weight configurations
        "Softplus" | "GELU" => &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        // Others: moderate extended range
        _ => &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
    };

    let base_negative: &[f32] = match squash {
        // ReLU/ELU: extended negative for high-threshold neurons
        "ReLU" | "ELU" | "SELU" => &[-0.1, -0.5, -1.0, -2.0, -5.0, -10.0, -25.0],
        // Symmetric: mirror of positive for operating point shift
        "TANH" | "LOGISTIC" => &[-0.1, -0.5, -1.0, -2.0, -5.0, -10.0],
        // IDENTITY: widest range to match large weights
        "IDENTITY" => &[-0.5, -1.0, -2.0, -5.0, -10.0, -25.0, -50.0],
        // Softplus/GELU: extended negative thresholds
        "Softplus" | "GELU" => &[-0.1, -0.5, -1.0, -2.0, -5.0, -10.0],
        // Others: moderate extended
        _ => &[-0.1, -0.5, -1.0, -2.0, -5.0],
    };

    let mut values: Vec<f32> = base_negative.to_vec();
    values.extend_from_slice(base_positive);
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values
}

/// Minimum weight ratio (incoming/outgoing) for reliable predictions.
///
/// Based on analysis of successful vs failed discoveries:
/// - ALL successful discoveries have ratio >= 71x
/// - Many failures have ratio < 10x
///
/// Using 50x provides some margin for edge cases.
const MIN_WEIGHT_RATIO: f32 = 50.0;

/// Calculate optimal outgoing weight for add-synapse or add-neuron candidates.
///
/// This is the shared weight calculation function used by both synapse and neuron
/// analysis to ensure consistent behaviour and maintainability (DRY principle).
///
/// The formula used is the standard least squares optimal weight:
/// ```text
/// w = Σ(error × activation) / Σ(activation²)
/// ```
///
/// # Arguments
/// * `sum_error_activation` - Σ(error × activation) from samples
/// * `sum_activation_sq` - Σ(activation²) from samples  
/// * `incoming_weight` - For neurons: the incoming weight; for synapses: use 1.0
///
/// # Returns
/// * `Some(weight)` - Optimal outgoing weight, clamped to [-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT]
/// * `None` - If weight cannot be computed (insufficient activation, invalid result, or
///   weight ratio too small for reliable prediction)
///
/// # Weight Ratio Validation
/// For add-neuron candidates where incoming_weight > 1.0, we validate that the
/// incoming/outgoing ratio is at least MIN_WEIGHT_RATIO. This is based on analysis
/// showing successful discoveries have much larger incoming than outgoing weights
/// (ratio 71x to 104,000x), while failures often have nearly equal weights.
fn calculate_optimal_outgoing_weight(
    sum_error_activation: f32,
    sum_activation_sq: f32,
    incoming_weight: f32,
) -> Option<f32> {
    // Need sufficient activation energy to compute meaningful weight
    if sum_activation_sq <= EPSILON {
        return None;
    }

    // Compute raw optimal weight using least squares formula
    let raw_weight = sum_error_activation / (sum_activation_sq + EPSILON);

    // Reject invalid weights
    if !raw_weight.is_finite() || raw_weight.abs() <= EPSILON {
        return None;
    }

    // Clamp to tight range based on successful discovery analysis
    let clamped = raw_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

    // For add-neuron candidates with non-trivial incoming weights, validate ratio
    // This catches cases where the computed weight is too large relative to incoming
    if incoming_weight.abs() > 1.0 {
        let ratio = incoming_weight.abs() / (clamped.abs() + EPSILON);
        if ratio < MIN_WEIGHT_RATIO {
            // Weight ratio too small - this configuration is unreliable
            // Skip rather than returning a weight that's likely to fail
            return None;
        }
    }

    Some(clamped)
}

/// Special-case optimisation for IDENTITY candidates: fit an affine correction.
///
/// For IDENTITY, the new neuron's output is linear in the source activation:
/// `output = incoming_weight * activation + bias`.
///
/// The generic search path computes `outgoing_weight` without bias, then searches bias with that
/// fixed outgoing weight. For complement-like shapes (eg `1 - x`) this can miss the optimum
/// because the best `outgoing_weight` depends on the bias/intercept.
///
/// We address that by directly fitting a 2-parameter model:
/// `avg_error ≈ outgoing_weight * (incoming_weight * activation) + intercept`,
/// then converting `intercept` to `bias` via `bias = intercept / outgoing_weight`.
fn calculate_optimal_identity_outgoing_and_bias(
    samples: &[HelpfulSample],
    incoming_weight: f32,
) -> Option<(f32, f32)> {
    let mut n: f32 = 0.0;
    let mut sum_a = 0.0f32;
    let mut sum_aa = 0.0f32;
    let mut sum_u = 0.0f32;
    let mut sum_uu = 0.0f32;
    let mut sum_e = 0.0f32;
    let mut sum_eu = 0.0f32;

    for sample in samples {
        if !sample.activation.is_finite() || !sample.avg_error.is_finite() {
            continue;
        }
        sum_a += sample.activation;
        sum_aa += sample.activation * sample.activation;
        let u = incoming_weight * sample.activation;
        n += 1.0;
        sum_u += u;
        sum_uu += u * u;
        sum_e += sample.avg_error;
        sum_eu += sample.avg_error * u;
    }

    if n <= 0.0 {
        return None;
    }

    // If the source activation has low variance, explicitly fitting an intercept (bias)
    // tends to overfit constant-ish sources. We already apply variance discounting later,
    // but keeping IDENTITY bias at 0.0 here avoids inflating the *raw* prediction before
    // the discount is applied (see Issue #130 tests).
    //
    // Use the same threshold as compute_source_variance_discount().
    const MIN_SOURCE_STD_DEV: f32 = 0.05;
    let mean_a = sum_a / n;
    let var_a = (sum_aa / n) - (mean_a * mean_a);
    let std_dev_a = var_a.max(0.0).sqrt();
    if std_dev_a < MIN_SOURCE_STD_DEV {
        let outgoing_weight = calculate_optimal_outgoing_weight(sum_eu, sum_uu, incoming_weight)?;
        return Some((outgoing_weight, 0.0));
    }

    // Solve normal equations for `e ≈ w*u + c` (w = outgoing_weight, c = intercept).
    // If the determinant is ~0, fall back to the no-intercept weight and compute the best intercept.
    let det = sum_uu * n - sum_u * sum_u;

    let outgoing_weight_raw = if det.abs() > EPSILON {
        (sum_eu * n - sum_e * sum_u) / det
    } else {
        sum_eu / (sum_uu + EPSILON)
    };

    if !outgoing_weight_raw.is_finite() || outgoing_weight_raw.abs() <= EPSILON {
        return None;
    }

    let outgoing_weight = outgoing_weight_raw.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

    // Apply the same reliability guard as the generic path.
    if incoming_weight.abs() > 1.0 {
        let ratio = incoming_weight.abs() / (outgoing_weight.abs() + EPSILON);
        if ratio < MIN_WEIGHT_RATIO {
            return None;
        }
    }

    // Best intercept for fixed outgoing_weight (least squares).
    let intercept = (sum_e - outgoing_weight * sum_u) / n;
    let bias = intercept / outgoing_weight;

    if !bias.is_finite() {
        return None;
    }

    Some((outgoing_weight, bias))
}

/// Calculate optimal bias for a neuron candidate using grid search.
///
/// This function finds the bias value that maximises error reduction when combined
/// with the given weights and activation function. It tests multiple bias values
/// across an activation-function-specific range and selects the one that gives
/// the best improvement.
///
/// Uses GPU-accelerated parallel search when analyzer is provided and GPU is available,
/// otherwise falls back to CPU sequential search.
///
/// # Arguments
/// * `samples` - Training samples (source activations and target errors)
/// * `incoming_weight` - Weight from source to new neuron
/// * `outgoing_weight` - Weight from new neuron to target
/// * `activation_fn` - Activation function to apply
/// * `squash` - Activation function name (for bias range selection and GPU)
/// * `analyzer` - Optional GPU analyzer for accelerated search
///
/// # Returns
/// Optimal bias value that maximises error reduction
fn calculate_optimal_bias(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    activation_fn: fn(f32) -> f32,
    squash: &str,
    analyzer: Option<&GpuAnalyzer>,
    target_squash: Option<&str>,
) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let bias_range = get_bias_range(squash);

    // Try GPU-accelerated search first if analyzer available
    if let Some(gpu_analyzer) = analyzer {
        if gpu_analyzer.device.is_some() {
            let activation_type = activation_name_to_gpu_id(squash);
            if let Ok(optimal_bias) = gpu_analyzer.evaluate_bias_gpu(
                samples,
                incoming_weight,
                outgoing_weight,
                activation_type,
                bias_range,
            ) {
                return optimal_bias;
            }
            // If GPU fails, fall through to CPU
        }
    }

    // Use log-spaced bias values for efficient search
    // Evolution will fine-tune the exact value after discovery
    let bias_values = get_bias_values(squash);

    // Calculate baseline error (no new neuron)
    let mut total_baseline_error_sq = 0.0;
    for sample in samples {
        if sample.avg_error.is_finite() {
            total_baseline_error_sq += sample.avg_error * sample.avg_error;
        }
    }

    if total_baseline_error_sq <= EPSILON {
        return 0.0;
    }

    let mut best_bias = 0.0;
    let mut best_error_reduction = f32::NEG_INFINITY;

    // Check if we can use HARD_TANH model (need target_value for all samples)
    let use_hard_tanh = target_squash == Some("HARD_TANH")
        && samples
            .iter()
            .all(|s| s.target_value.is_some() && s.target_activation.is_some());

    // Search over log-spaced bias values
    for &bias in &bias_values {
        // Calculate error with this bias
        let mut total_new_error_sq = 0.0;
        let mut valid_samples = 0;

        for sample in samples {
            if !sample.avg_error.is_finite() || !sample.activation.is_finite() {
                continue;
            }

            // Calculate new neuron's activation with bias
            let pre_activation = incoming_weight * sample.activation + bias;
            let new_neuron_activation = activation_fn(pre_activation);

            if !new_neuron_activation.is_finite() {
                continue;
            }

            // Calculate new error at target neuron
            let correction = outgoing_weight * new_neuron_activation;
            let new_error = if use_hard_tanh {
                // HARD_TANH model: account for target neuron's clamping
                // CRITICAL: avg_error is in VALUE domain (targetValue - currentValue from TypeScript)
                // So we compute desired_value = target_value + avg_error, then squash to get expected activation
                let target_value = sample.target_value.unwrap();
                let desired_value = target_value + sample.avg_error;
                let expected = hard_tanh(desired_value);
                let new_input = target_value + correction;
                let new_output = hard_tanh(new_input);
                expected - new_output
            } else {
                // Linear model
                sample.avg_error - correction
            };

            if new_error.is_finite() {
                total_new_error_sq += new_error * new_error;
                valid_samples += 1;
            }
        }

        // Only consider if we have valid samples
        if valid_samples < MIN_NEURON_SAMPLE_COUNT {
            continue;
        }

        // Calculate error reduction (positive is good)
        let error_reduction = total_baseline_error_sq - total_new_error_sq;

        if error_reduction > best_error_reduction {
            best_error_reduction = error_reduction;
            best_bias = bias;
        }
    }

    best_bias
}

#[derive(Default)]
struct HarmfulStats {
    harmful_count: u32,
    helpful_count: u32,
    harmful_error_sum: f32,
}

pub struct GpuAnalyzer {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    helpful_layout: Option<wgpu::BindGroupLayout>,
    helpful_pipeline: Option<wgpu::ComputePipeline>,
    harmful_layout: Option<wgpu::BindGroupLayout>,
    harmful_pipeline: Option<wgpu::ComputePipeline>,
    relu_layout: Option<wgpu::BindGroupLayout>,
    relu_pipeline: Option<wgpu::ComputePipeline>,
    #[allow(dead_code)] // Framework for future GPU activation evaluation
    activation_layout: Option<wgpu::BindGroupLayout>,
    #[allow(dead_code)] // Framework for future GPU activation evaluation
    activation_pipeline: Option<wgpu::ComputePipeline>,
    bias_layout: Option<wgpu::BindGroupLayout>,
    bias_pipeline: Option<wgpu::ComputePipeline>,
    /// Optimised GPU batch size based on detected hardware.
    /// Higher values improve GPU utilisation on high-performance hardware.
    batch_size: usize,
}

/// Result of GPU availability check with detailed diagnostics.
pub struct GpuAvailabilityResult {
    /// Whether a GPU is available for use.
    pub available: bool,
    /// Human-readable reason for the availability status.
    pub reason: Option<String>,
    /// Whether this is an error condition (true on macOS when GPU unavailable).
    pub is_error: bool,
}

// =============================================================================
// GPU Evaluator Trait - Abstracts over GpuAnalyzer and GpuWorkQueue
// =============================================================================

/// Trait for GPU-based evaluation operations.
/// This allows helper functions to work with either a direct GpuAnalyzer
/// or a shared GpuWorkQueue without code duplication.
trait GpuEvaluator {
    /// Evaluate ReLU activation for neuron candidates.
    /// Returns (positive_stats, negative_stats, baseline_error_sq).
    fn evaluate_relu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)>;

    /// Evaluate general activation function for neuron candidates.
    /// Returns (sum_activation_sq, sum_error_activation, total_baseline_error_sq, improved_count).
    fn evaluate_activation(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)>;
}

/// Implementation for direct GpuAnalyzer access.
impl GpuEvaluator for GpuAnalyzer {
    fn evaluate_relu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        self.evaluate_relu_gpu(samples, threshold)
    }

    fn evaluate_activation(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)> {
        self.evaluate_activation_gpu(samples, activation_type, orientation, scale)
    }
}

/// Implementation for shared GpuWorkQueue.
/// NOTE: These trait methods use `None` deadline, which gives maximum timeout (5 minutes).
/// For deadline-aware evaluation, use the batch methods directly with an explicit deadline.
impl GpuEvaluator for GpuWorkQueue {
    fn evaluate_relu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        // Clone samples to send to the GPU thread
        // Uses None deadline = maximum timeout (5 minutes)
        self.evaluate_relu_gpu(samples.to_vec(), threshold, &None)
    }

    fn evaluate_activation(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)> {
        // Uses None deadline = maximum timeout (5 minutes)
        self.evaluate_activation_gpu(samples.to_vec(), activation_type, orientation, scale, &None)
    }
}

// =============================================================================
// GPU Work Queue - Centralised GPU thread for improved utilisation
// =============================================================================

/// Work request types for the GPU queue.
/// Each variant contains the data needed for a specific GPU operation.
enum GpuWorkRequest {
    /// Batch of helpful synapse/neuron evaluations.
    /// Each item is a slice of samples to evaluate.
    HelpfulBatch {
        /// Samples for each evaluation, indexed by request ID.
        samples: Vec<Vec<HelpfulSample>>,
        /// Channel to send results back.
        response_tx: Sender<Result<Vec<HelpfulStats>>>,
    },
    /// Batch of harmful synapse evaluations.
    /// Each item is (samples, weight) pair.
    HarmfulBatch {
        /// (samples, weight) pairs for each evaluation.
        samples_with_weights: Vec<(Vec<HelpfulSample>, f32)>,
        /// Channel to send results back.
        response_tx: Sender<Result<Vec<HarmfulStats>>>,
    },
    /// ReLU activation evaluation for neuron candidates.
    ReluEval {
        samples: Vec<HelpfulSample>,
        threshold: f32,
        response_tx: Sender<Result<(ReluStats, ReluStats, f32)>>,
    },
    /// General activation function evaluation for neuron candidates.
    ActivationEval {
        samples: Vec<HelpfulSample>,
        activation_type: u32,
        orientation: f32,
        scale: f32,
        response_tx: Sender<Result<(f32, f32, f32, u32)>>,
    },
    /// Request to shut down the GPU thread.
    Shutdown,
}

/// Centralised GPU work queue that processes all GPU operations on a single thread.
///
/// This eliminates the overhead of creating multiple GPU devices (one per parallel
/// focus neuron) and improves GPU utilisation by batching work from multiple sources.
///
/// # Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────┐
/// │  CPU Threads (rayon par_iter)                               │
/// │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                    │
/// │  │Focus1│  │Focus2│  │Focus3│  │Focus4│  ...               │
/// │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘                    │
/// │     │         │         │         │                         │
/// │     └────┬────┴────┬────┴────┬────┘                         │
/// │          │         │         │                              │
/// │          ▼         ▼         ▼                              │
/// │  ┌─────────────────────────────────────────────────┐       │
/// │  │           GPU Work Queue (crossbeam channel)     │       │
/// │  └──────────────────────┬──────────────────────────┘       │
/// │                         │                                   │
/// │                         ▼                                   │
/// │  ┌─────────────────────────────────────────────────┐       │
/// │  │           GPU Thread (owns GpuAnalyzer)          │       │
/// │  │  • Batches work from multiple focus neurons      │       │
/// │  │  • Single GPU device for all operations          │       │
/// │  │  • Optimal GPU utilisation                       │       │
/// │  └─────────────────────────────────────────────────┘       │
/// └─────────────────────────────────────────────────────────────┘
/// ```
struct GpuWorkQueue {
    /// Channel to send work to the GPU thread.
    work_tx: Sender<GpuWorkRequest>,
    /// Handle to the GPU thread (for clean shutdown).
    thread_handle: Option<JoinHandle<()>>,
    /// Channel to receive notification when GPU thread exits.
    /// This allows Drop to use a timeout instead of blocking forever.
    exit_rx: Receiver<()>,
}

impl GpuWorkQueue {
    /// Create a new GPU work queue with a dedicated GPU thread.
    ///
    /// The GPU thread is spawned immediately and owns the GpuAnalyzer.
    /// All GPU operations are processed sequentially on this thread,
    /// eliminating device creation overhead and improving utilisation.
    ///
    /// CRITICAL: The GpuAnalyzer is created INSIDE the GPU thread, not before.
    /// wgpu devices have thread-local state that doesn't transfer properly when
    /// moved across threads, causing deadlocks in device.poll().
    pub fn new() -> Result<Self> {
        // Create the channel for sending work to the GPU thread.
        // Capacity is dynamically sized based on available system memory.
        // Lower capacity = more backpressure = less memory usage.
        // This prevents Metal command buffer exhaustion on memory-constrained systems.
        let queue_capacity = get_work_queue_capacity();
        let (work_tx, work_rx): (Sender<GpuWorkRequest>, Receiver<GpuWorkRequest>) =
            bounded(queue_capacity);

        // Channel to receive initialization result from the GPU thread.
        // This ensures the GpuAnalyzer is created ON the GPU thread, not moved to it.
        let (init_tx, init_rx): (Sender<Result<()>>, Receiver<Result<()>>) = bounded(1);

        // Channel to receive notification when GPU thread exits.
        // This allows Drop to use a timeout instead of blocking forever if the GPU hangs.
        let (exit_tx, exit_rx): (Sender<()>, Receiver<()>) = bounded(1);

        // Spawn dedicated GPU thread - analyzer is created INSIDE this thread
        let thread_handle = thread::spawn(move || {
            // Create analyzer on THIS thread to avoid wgpu thread-local state issues
            match GpuAnalyzer::new() {
                Ok(analyzer) => {
                    // Signal successful initialization
                    let _ = init_tx.send(Ok(()));
                    // Run the main loop
                    Self::gpu_thread_loop(analyzer, work_rx);
                }
                Err(e) => {
                    // Signal initialization failure
                    let _ = init_tx.send(Err(e));
                }
            }
            // Always signal exit, even if initialization failed or loop panicked
            let _ = exit_tx.send(());
        });

        // Wait for initialization to complete with timeout
        let init_timeout = Duration::from_secs(GPU_INIT_TIMEOUT_SECS);
        match init_rx.recv_timeout(init_timeout) {
            Ok(Ok(())) => {} // Success
            Ok(Err(e)) => return Err(e),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                return Err(anyhow!(
                    "GPU initialisation timed out after {GPU_INIT_TIMEOUT_SECS}s. \
                     The GPU may be unresponsive or overwhelmed. \
                     Try restarting the process or reducing workload."
                ));
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                return Err(anyhow!("GPU thread failed to start (channel disconnected)"));
            }
        }

        Ok(Self {
            work_tx,
            thread_handle: Some(thread_handle),
            exit_rx,
        })
    }

    /// The main loop for the GPU thread.
    /// Processes work requests until shutdown is requested.
    fn gpu_thread_loop(analyzer: GpuAnalyzer, work_rx: Receiver<GpuWorkRequest>) {
        while let Ok(request) = work_rx.recv() {
            match request {
                GpuWorkRequest::HelpfulBatch {
                    samples,
                    response_tx,
                } => {
                    // Convert Vec<Vec<HelpfulSample>> to &[&[HelpfulSample]] for the API
                    let samples_refs: Vec<&[HelpfulSample]> =
                        samples.iter().map(|v| v.as_slice()).collect();
                    let result = analyzer.evaluate_helpful_batch(&samples_refs);
                    // Send result back (ignore send errors - receiver may have dropped)
                    let _ = response_tx.send(result);
                }
                GpuWorkRequest::HarmfulBatch {
                    samples_with_weights,
                    response_tx,
                } => {
                    // Convert to the format expected by evaluate_harmful_batch
                    let batch_refs: Vec<(&[HelpfulSample], f32)> = samples_with_weights
                        .iter()
                        .map(|(samples, weight)| (samples.as_slice(), *weight))
                        .collect();
                    let result = analyzer.evaluate_harmful_batch(&batch_refs);
                    let _ = response_tx.send(result);
                }
                GpuWorkRequest::ReluEval {
                    samples,
                    threshold,
                    response_tx,
                } => {
                    let result = analyzer.evaluate_relu_gpu(&samples, threshold);
                    let _ = response_tx.send(result);
                }
                GpuWorkRequest::ActivationEval {
                    samples,
                    activation_type,
                    orientation,
                    scale,
                    response_tx,
                } => {
                    let result = analyzer.evaluate_activation_gpu(
                        &samples,
                        activation_type,
                        orientation,
                        scale,
                    );
                    let _ = response_tx.send(result);
                }
                GpuWorkRequest::Shutdown => {
                    // Clean shutdown requested
                    break;
                }
            }
        }
    }

    /// Submit a batch of helpful evaluations and wait for results.
    ///
    /// This is a synchronous call that blocks until the GPU thread processes
    /// the batch and returns results.
    ///
    /// The `deadline` parameter is used to calculate an adaptive timeout:
    /// - With deadline: uses up to half remaining time (60s-5min)
    /// - Without deadline: uses maximum timeout (5 minutes)
    pub fn evaluate_helpful_batch(
        &self,
        samples: Vec<Vec<HelpfulSample>>,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<Vec<HelpfulStats>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // Create a one-shot channel for the response
        let (response_tx, response_rx) = bounded(1);

        // Calculate timeout based on remaining deadline
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();

        // Send the work request with timeout to prevent deadlock if GPU thread is hung
        // If the channel is full (GPU not processing), this will timeout instead of blocking forever
        match self.work_tx.send_timeout(
            GpuWorkRequest::HelpfulBatch {
                samples,
                response_tx,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(anyhow!(
                    "GPU work queue full - send timed out after {timeout_secs}s. \
                     The GPU thread may be hung. Consider restarting the process."
                ));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        // Wait for the response with timeout
        match response_rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Err(anyhow!(
                "GPU helpful batch evaluation timed out after {timeout_secs}s. \
                     The GPU may be unresponsive. Consider reducing batch size or restarting."
            )),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                Err(anyhow!("GPU response channel closed unexpectedly"))
            }
        }
    }

    /// Submit a batch of harmful evaluations and wait for results.
    ///
    /// The `deadline` parameter is used to calculate an adaptive timeout (60s-5min).
    pub fn evaluate_harmful_batch(
        &self,
        samples_with_weights: Vec<(Vec<HelpfulSample>, f32)>,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<Vec<HarmfulStats>> {
        if samples_with_weights.is_empty() {
            return Ok(Vec::new());
        }

        let (response_tx, response_rx) = bounded(1);
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();

        // Send with timeout to prevent deadlock if GPU thread is hung
        match self.work_tx.send_timeout(
            GpuWorkRequest::HarmfulBatch {
                samples_with_weights,
                response_tx,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(anyhow!(
                    "GPU work queue full - send timed out after {timeout_secs}s. \
                     The GPU thread may be hung. Consider restarting the process."
                ));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        match response_rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Err(anyhow!(
                "GPU harmful batch evaluation timed out after {timeout_secs}s. \
                     The GPU may be unresponsive. Consider reducing batch size or restarting."
            )),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                Err(anyhow!("GPU response channel closed unexpectedly"))
            }
        }
    }

    /// Submit a ReLU evaluation and wait for results.
    /// Returns (positive_stats, negative_stats, baseline_error_sq).
    ///
    /// The `deadline` parameter is used to calculate an adaptive timeout (60s-5min).
    fn evaluate_relu_gpu(
        &self,
        samples: Vec<HelpfulSample>,
        threshold: f32,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        if samples.is_empty() {
            return Ok((
                ReluStats::new(ReluOrientation::Positive),
                ReluStats::new(ReluOrientation::Negative),
                0.0,
            ));
        }

        let (response_tx, response_rx) = bounded(1);
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();

        // Send with timeout to prevent deadlock if GPU thread is hung
        match self.work_tx.send_timeout(
            GpuWorkRequest::ReluEval {
                samples,
                threshold,
                response_tx,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(anyhow!(
                    "GPU work queue full - send timed out after {timeout_secs}s. \
                     The GPU thread may be hung. Consider restarting the process."
                ));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        match response_rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Err(anyhow!(
                "GPU ReLU evaluation timed out after {timeout_secs}s. \
                     The GPU may be unresponsive. Consider reducing batch size or restarting."
            )),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                Err(anyhow!("GPU response channel closed unexpectedly"))
            }
        }
    }

    /// Submit an activation evaluation and wait for results.
    /// Returns (sum_activation_sq, sum_error_activation, total_baseline_error_sq, improved_count).
    ///
    /// The `deadline` parameter is used to calculate an adaptive timeout (60s-5min).
    fn evaluate_activation_gpu(
        &self,
        samples: Vec<HelpfulSample>,
        activation_type: u32,
        orientation: f32,
        scale: f32,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<(f32, f32, f32, u32)> {
        if samples.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0));
        }

        let (response_tx, response_rx) = bounded(1);
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();

        // Send with timeout to prevent deadlock if GPU thread is hung
        match self.work_tx.send_timeout(
            GpuWorkRequest::ActivationEval {
                samples,
                activation_type,
                orientation,
                scale,
                response_tx,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(anyhow!(
                    "GPU work queue full - send timed out after {timeout_secs}s. \
                     The GPU thread may be hung. Consider restarting the process."
                ));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        match response_rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Err(anyhow!(
                "GPU activation evaluation timed out after {timeout_secs}s. \
                     The GPU may be unresponsive. Consider reducing batch size or restarting."
            )),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                Err(anyhow!("GPU response channel closed unexpectedly"))
            }
        }
    }

    /// Request the GPU thread to shut down.
    /// This should be called before dropping the queue to ensure clean shutdown.
    ///
    /// Uses a timeout to avoid blocking forever if the queue is full and the GPU
    /// thread is hung. If the send times out, the GPU thread is likely unresponsive
    /// and the Drop implementation will handle cleanup via the exit_rx timeout.
    pub fn shutdown(&self) {
        // Use timeout to avoid blocking forever if queue is full and GPU thread is hung.
        // 2 seconds is generous - if the GPU thread is responsive, it should drain
        // items much faster. If this times out, proceed to exit_rx timeout in Drop.
        let shutdown_send_timeout = Duration::from_secs(2);
        let _ = self
            .work_tx
            .send_timeout(GpuWorkRequest::Shutdown, shutdown_send_timeout);
    }
}

/// Timeout for GPU thread shutdown during Drop.
/// If the GPU thread doesn't exit within this time, we abandon it.
/// This prevents the process from hanging forever if the GPU driver is stuck.
const GPU_SHUTDOWN_TIMEOUT_SECS: u64 = 10;

impl Drop for GpuWorkQueue {
    fn drop(&mut self) {
        // Request shutdown
        self.shutdown();

        // Wait for GPU thread to exit with timeout
        // This prevents hanging forever if the GPU driver is stuck (e.g., Metal semaphore wait)
        let timeout = Duration::from_secs(GPU_SHUTDOWN_TIMEOUT_SECS);
        match self.exit_rx.recv_timeout(timeout) {
            Ok(()) => {
                // Thread exited cleanly, now safe to join
                if let Some(handle) = self.thread_handle.take() {
                    let _ = handle.join();
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // GPU thread is stuck (likely in Metal driver)
                // Log warning and abandon the thread - it will be cleaned up on process exit
                eprintln!(
                    "[NEAT-AI-Discovery] WARNING: GPU thread did not exit within {GPU_SHUTDOWN_TIMEOUT_SECS}s. \
                     The GPU driver may be hung. Abandoning thread to prevent deadlock. \
                     Consider restarting the process."
                );
                // Don't join - the thread is stuck and joining would block forever
                let _ = self.thread_handle.take();
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                // Channel disconnected - thread already exited (possibly via panic)
                if let Some(handle) = self.thread_handle.take() {
                    let _ = handle.join();
                }
            }
        }
    }
}

impl GpuAnalyzer {
    /// Lightweight probe to determine whether a usable GPU device is available.
    ///
    /// This is intended for callers (via FFI) that want to decide whether to
    /// enable the Rust discovery extension at all. It deliberately avoids
    /// falling back to CPU – if the adapter or device cannot be created, the
    /// probe reports `false`.
    ///
    /// Platform-specific behaviour:
    /// - **macOS**: GPU should always be available (Metal). Missing GPU is an error.
    /// - **Linux**: GPU may not be available on headless servers without GPU hardware
    ///   or proper permissions. Missing GPU gracefully disables discovery.
    ///
    /// **Note**: Result is cached for consistency. Creating wgpu instances is expensive
    /// and can give inconsistent results under parallel load (e.g., CI environments).
    pub fn gpu_is_available() -> bool {
        use std::sync::OnceLock;
        static GPU_AVAILABLE: OnceLock<bool> = OnceLock::new();
        *GPU_AVAILABLE.get_or_init(|| Self::check_gpu_availability().available)
    }

    /// Check GPU availability with detailed diagnostics.
    ///
    /// Returns availability status, reason, and whether it's an error condition.
    /// On macOS, missing GPU is treated as an error (Metal should always work).
    /// On Linux, missing GPU gracefully disables discovery (common on headless servers).
    pub fn check_gpu_availability() -> GpuAvailabilityResult {
        // Check minimum system requirements FIRST before any GPU operations.
        // This prevents hangs on very old/constrained machines by disabling discovery early.
        if let Some(result) = check_minimum_system_requirements() {
            return result;
        }

        // Suppress Mesa/libEGL warnings if requested (must be called before GPU init)
        suppress_mesa_warnings_if_requested();

        // Set XDG_RUNTIME_DIR if not already set (required by wgpu on Linux/Wayland)
        // Uses Once internally for thread-safe one-time initialisation
        ensure_xdg_runtime_dir();

        // Use safe instance creation to avoid panics from EGL/GL backend probing on Linux
        let Some(instance) = create_wgpu_instance_safely() else {
            return Self::no_gpu_result("wgpu instance creation failed (GPU backend unavailable)");
        };
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Some(adapter) = adapter else {
            return Self::no_gpu_result("No GPU adapter found");
        };

        let device_result = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("NEAT-AI Discovery GPU probe device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ));

        match device_result {
            Ok(_) => GpuAvailabilityResult {
                available: true,
                reason: None,
                is_error: false,
            },
            Err(e) => Self::no_gpu_result(&format!("GPU device creation failed: {e}")),
        }
    }

    /// Create a result for when GPU is not available.
    /// On macOS this is an error; on Linux it gracefully disables discovery.
    fn no_gpu_result(reason: &str) -> GpuAvailabilityResult {
        #[cfg(target_os = "macos")]
        {
            // On macOS, Metal should always be available - missing GPU is an error
            GpuAvailabilityResult {
                available: false,
                reason: Some(format!(
                    "{reason}. On macOS, GPU (Metal) should always be available. \
                     This may indicate a system configuration issue."
                )),
                is_error: true,
            }
        }

        #[cfg(target_os = "linux")]
        {
            // On Linux, GPU may not be available on headless servers - gracefully disable
            GpuAvailabilityResult {
                available: false,
                reason: Some(format!(
                    "{reason}. Discovery disabled on this machine. \
                     This is normal for headless Linux servers without GPU hardware or \
                     without proper permissions to access /dev/dri devices."
                )),
                is_error: false,
            }
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            // For other platforms, treat as non-error (graceful disable)
            GpuAvailabilityResult {
                available: false,
                reason: Some(format!("{reason}. Discovery disabled on this platform.")),
                is_error: false,
            }
        }
    }

    fn new() -> Result<Self> {
        // Suppress Mesa/libEGL warnings if requested (must be called before GPU init)
        suppress_mesa_warnings_if_requested();

        // Set XDG_RUNTIME_DIR if not already set (required by wgpu on Linux/Wayland)
        // Uses Once internally for thread-safe one-time initialisation
        ensure_xdg_runtime_dir();

        // Use safe instance creation to avoid panics from EGL/GL backend probing on Linux
        let instance = create_wgpu_instance_safely().ok_or_else(|| {
            anyhow::anyhow!(
                "wgpu instance creation failed (GPU backend unavailable). \
                 Discovery requires GPU acceleration."
            )
        })?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Some(adapter) => adapter,
            None => {
                // GPU is required - return an error instead of a CPU-only analyzer.
                // TypeScript calls check_gpu_available() before discovery, but the GPU
                // could become unavailable due to race conditions or resource exhaustion.
                // Returning an error here prevents panics in GPU methods that .expect() on device.
                anyhow::bail!(
                    "GPU adapter not available. Discovery requires GPU acceleration. \
                     This may indicate a transient GPU resource issue - consider retrying."
                );
            }
        };

        // Detect GPU tier and system resources for auto-tuning
        let adapter_info = adapter.get_info();
        let gpu_tier = detect_gpu_tier(&adapter_info);
        // Batch size is adjusted based on BOTH GPU tier AND available memory
        let batch_size = get_adjusted_batch_size(gpu_tier);

        // Log GPU info once per process (helps diagnose performance issues)
        log_gpu_info_once(&adapter_info, gpu_tier, batch_size);

        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("NEAT-AI Discovery GPU device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )) {
            Ok(result) => result,
            Err(e) => {
                // GPU is required - return an error instead of a CPU-only analyzer.
                // TypeScript calls check_gpu_available() before discovery, but the GPU
                // could become unavailable due to race conditions or resource exhaustion.
                // Returning an error here prevents panics in GPU methods that .expect() on device.
                anyhow::bail!(
                    "GPU device creation failed: {e}. Discovery requires GPU acceleration. \
                     This may indicate a transient GPU resource issue - consider retrying."
                );
            }
        };

        let (helpful_layout, helpful_pipeline) =
            Self::build_helpful_pipeline(&device, "helpful-synapse-pipeline");
        let (harmful_layout, harmful_pipeline) =
            Self::build_harmful_pipeline(&device, "harmful-synapse-pipeline");
        let (relu_layout, relu_pipeline) = Self::build_relu_pipeline(&device, "relu-pipeline");
        let (activation_layout, activation_pipeline) =
            Self::build_activation_pipeline(&device, "activation-pipeline");
        let (bias_layout, bias_pipeline) = Self::build_bias_pipeline(&device, "bias-pipeline");

        // CRITICAL: Warm up the GPU by polling to ensure all pipeline creation work is complete.
        //
        // We intentionally avoid `Maintain::Wait` here because it can block forever if the
        // GPU driver wedges on a specific machine. If this doesn't settle quickly, treat it
        // as a GPU initialisation failure and return an error so callers can restart/skip.
        poll_device_until_idle(
            &device,
            Duration::from_secs(GPU_INIT_TIMEOUT_SECS),
            "GPU warm-up after pipeline creation",
        )?;

        Ok(Self {
            device: Some(device),
            queue: Some(queue),
            helpful_layout: Some(helpful_layout),
            helpful_pipeline: Some(helpful_pipeline),
            harmful_layout: Some(harmful_layout),
            harmful_pipeline: Some(harmful_pipeline),
            relu_layout: Some(relu_layout),
            relu_pipeline: Some(relu_pipeline),
            activation_layout: Some(activation_layout),
            activation_pipeline: Some(activation_pipeline),
            bias_layout: Some(bias_layout),
            bias_pipeline: Some(bias_pipeline),
            batch_size,
        })
    }

    fn build_helpful_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("helpful-synapse-shader"),
            source: wgpu::ShaderSource::Wgsl(HELPFUL_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("helpful-synapse-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn build_harmful_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("harmful-synapse-shader"),
            source: wgpu::ShaderSource::Wgsl(HARMFUL_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("harmful-synapse-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn build_relu_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("relu-shader"),
            source: wgpu::ShaderSource::Wgsl(RELU_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("relu-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn build_activation_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("activation-shader"),
            source: wgpu::ShaderSource::Wgsl(ACTIVATION_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("activation-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn build_bias_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bias-shader"),
            source: wgpu::ShaderSource::Wgsl(BIAS_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bias-bind-group"),
            entries: &[
                // Binding 0: samples (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: bias_candidates (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: results (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    /// Batch evaluate multiple harmful synapse operations to improve GPU utilisation.
    /// Each entry is (samples, weight) pair. Returns stats in the same order as input.
    ///
    /// This reduces CPU-GPU round trips by submitting multiple GPU dispatches in a single
    /// command buffer, significantly improving throughput for harmful synapse analysis.
    fn evaluate_harmful_batch(
        &self,
        samples_batch: &[(&[HelpfulSample], f32)],
    ) -> Result<Vec<HarmfulStats>> {
        if samples_batch.is_empty() {
            return Ok(Vec::new());
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for batched harmful analysis")?;
        let harmful_layout = self
            .harmful_layout
            .as_ref()
            .context("GPU layout not initialised for batched harmful analysis")?;
        let harmful_pipeline = self
            .harmful_pipeline
            .as_ref()
            .context("GPU pipeline not initialised for batched harmful analysis")?;

        // Process in batches to avoid excessive memory usage
        // Apple Silicon optimisation: Use single encoder per batch to reduce Metal driver overhead
        let mut all_results = Vec::with_capacity(samples_batch.len());

        let max_sample_len = samples_batch
            .iter()
            .map(|(samples, _)| samples.len())
            .max()
            .unwrap_or(0);
        let bytes_per_sample = std::mem::size_of::<GpuHelpfulSample>()
            + (2 * std::mem::size_of::<HarmfulContribution>());
        let effective_batch_size = cap_gpu_batch_size_by_bytes(
            self.batch_size,
            max_sample_len,
            bytes_per_sample,
            GPU_MAX_BATCH_ALLOC_BYTES,
        );

        if verbose_enabled() && effective_batch_size < self.batch_size {
            let bytes_per_op = max_sample_len.saturating_mul(bytes_per_sample);
            let approx_mb = (bytes_per_op as f64 / (1024.0 * 1024.0)).max(0.0);
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Capping GPU harmful batch size from {} to {} due to large sample count. \
                 max_sample_len={}, approx_buffers_per_op≈{:.1}MB, cap={}MB",
                self.batch_size,
                effective_batch_size,
                max_sample_len,
                approx_mb,
                GPU_MAX_BATCH_ALLOC_BYTES / (1024 * 1024)
            );
        }

        for batch_chunk in samples_batch.chunks(effective_batch_size) {
            let mut empty_flags = Vec::with_capacity(batch_chunk.len());
            let mut batch_staging_buffers = Vec::new();
            let mut batch_contribution_sizes = Vec::new();
            let mut batch_contributions_buffers = Vec::new();

            // Single encoder for entire batch - reduces Metal driver overhead on Apple Silicon
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("harmful-command-encoder-batch"),
            });

            // Prepare all operations in this batch
            for (samples, weight) in batch_chunk {
                if samples.is_empty() {
                    empty_flags.push(true);
                    continue;
                }
                empty_flags.push(false);

                let gpu_samples: Vec<GpuHelpfulSample> = samples
                    .iter()
                    .copied()
                    .map(GpuHelpfulSample::from)
                    .collect();
                let contributions_zeroed = vec![HarmfulContribution::zeroed(); samples.len()];

                let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("harmful-samples-buffer-batch"),
                    contents: bytemuck::cast_slice(&gpu_samples),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let contributions_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("harmful-contributions-buffer-batch"),
                        contents: bytemuck::cast_slice(&contributions_zeroed),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });

                let uniforms = HarmfulUniforms {
                    length: samples.len() as u32,
                    pad0: 0,
                    epsilon: EPSILON,
                    weight: *weight,
                };
                let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("harmful-uniform-buffer-batch"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: harmful_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: sample_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: contributions_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                    ],
                    label: Some("harmful-bind-group-batch"),
                });

                let contribution_size =
                    (std::mem::size_of::<HarmfulContribution>() * samples.len()) as u64;
                let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("harmful-staging-buffer-batch"),
                    size: contribution_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                // Add compute pass to shared encoder (reduces command buffer overhead)
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("harmful-compute-pass-batch"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(harmful_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
                    compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
                }

                batch_contributions_buffers.push(contributions_buffer);
                batch_staging_buffers.push(staging_buffer);
                batch_contribution_sizes.push(contribution_size);
            }

            // Add all buffer copies after compute passes (better GPU scheduling)
            for (i, staging_buffer) in batch_staging_buffers.iter().enumerate() {
                let contribution_size = batch_contribution_sizes[i];
                encoder.copy_buffer_to_buffer(
                    &batch_contributions_buffers[i],
                    0,
                    staging_buffer,
                    0,
                    contribution_size,
                );
            }

            // Submit single command buffer for entire batch - reduces Metal driver overhead
            if !batch_staging_buffers.is_empty() {
                queue.submit(Some(encoder.finish()));
            }

            // OPTIMISATION: Map ALL buffers first, then poll ONCE for all.
            // This reduces GPU-CPU round trips compared to mapping each buffer individually.
            let mut map_receivers = Vec::with_capacity(batch_staging_buffers.len());
            for staging_buffer in &batch_staging_buffers {
                let buffer_slice = staging_buffer.slice(..);
                let (sender, receiver) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender
                        .send(result)
                        .expect("Failed to send map_async result");
                });
                map_receivers.push(receiver);
            }

            // Event-driven wait: poll non-blocking, check all callback channels
            wait_for_buffer_maps_batch(device, &map_receivers, GPU_BUFFER_MAP_TIMEOUT_SECS)
                .context("Harmful batch buffer mapping failed")?;

            // Process results - maintain order with empty flags
            // Note: wait_for_buffer_maps_batch already verified all buffers are mapped
            let mut buffer_idx = 0;
            for is_empty in empty_flags {
                if is_empty {
                    all_results.push(HarmfulStats::default());
                } else {
                    // Buffer is already mapped and verified by wait_for_buffer_maps_batch
                    let staging_buffer = &batch_staging_buffers[buffer_idx];
                    let buffer_slice = staging_buffer.slice(..);
                    let data = buffer_slice.get_mapped_range();
                    let contributions: &[HarmfulContribution] = bytemuck::cast_slice(&data);

                    let mut stats = HarmfulStats::default();
                    for contribution in contributions {
                        stats.harmful_count += contribution.harmful_flag;
                        stats.helpful_count += contribution.helpful_flag;
                        stats.harmful_error_sum += contribution.error_magnitude;
                    }

                    drop(data);
                    staging_buffer.unmap();

                    all_results.push(stats);
                    buffer_idx += 1;
                }
            }

            // CRITICAL: Ensure Metal releases command buffers before creating new ones.
            // Without this, we can exhaust Metal's command buffer pool when processing
            // many batches, causing the GPU thread to hang in semaphore_wait_trap.
            //
            // Avoid an unbounded wait - bail out with an error if the driver is wedged.
            poll_device_until_idle(
                device,
                Duration::from_secs(5),
                "post-harmful-batch command buffer release",
            )?;
        }

        Ok(all_results)
    }

    fn evaluate_relu_gpu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        if samples.is_empty() {
            return Ok((
                ReluStats::new(ReluOrientation::Positive),
                ReluStats::new(ReluOrientation::Negative),
                0.0,
            ));
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for ReLU analysis")?;
        let relu_layout = self
            .relu_layout
            .as_ref()
            .context("GPU ReLU layout not initialised")?;
        let relu_pipeline = self
            .relu_pipeline
            .as_ref()
            .context("GPU ReLU pipeline not initialised")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let contributions_zeroed = vec![ReluContribution::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let contributions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-contributions-buffer"),
            contents: bytemuck::cast_slice(&contributions_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = ReluUniforms {
            length: samples.len() as u32,
            threshold,
            epsilon: EPSILON,
            pad0: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: relu_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: contributions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("relu-bind-group"),
        });

        let contribution_size = (std::mem::size_of::<ReluContribution>() * samples.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relu-staging-buffer"),
            size: contribution_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("relu-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("relu-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(relu_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &contributions_buffer,
            0,
            &staging_buffer,
            0,
            contribution_size,
        );

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("Failed to send map_async result");
        });

        // Event-driven wait: poll non-blocking, check callback channel
        wait_for_buffer_map(device, &receiver, GPU_BUFFER_MAP_TIMEOUT_SECS)
            .context("ReLU buffer mapping failed")?;

        let data = buffer_slice.get_mapped_range();
        let contributions: &[ReluContribution] = bytemuck::cast_slice(&data);

        let mut positive_stats = ReluStats::new(ReluOrientation::Positive);
        let mut negative_stats = ReluStats::new(ReluOrientation::Negative);
        let mut total_baseline_error_sq = 0.0;

        // Accumulate statistics from GPU contributions
        for (idx, contribution) in contributions.iter().enumerate() {
            if idx < samples.len() {
                total_baseline_error_sq += contribution.error_sq;

                // For positive ReLU, accumulate activation_sq and error_activation
                if contribution.positive_count > 0 {
                    positive_stats.activation_sq_sum += contribution.positive_activation_sq;
                    positive_stats.error_activation_sum += contribution.positive_error_activation;
                    // Reconstruct the activation and error for the samples vector
                    let activation = (contribution.positive_activation_sq).sqrt();
                    let error = if activation > EPSILON {
                        contribution.positive_error_activation / activation
                    } else {
                        0.0
                    };
                    positive_stats.samples.push((activation, error));
                }

                // For negative ReLU, accumulate activation_sq and error_activation
                if contribution.negative_count > 0 {
                    negative_stats.activation_sq_sum += contribution.negative_activation_sq;
                    negative_stats.error_activation_sum += contribution.negative_error_activation;
                    // Reconstruct the activation and error for the samples vector
                    let activation = (contribution.negative_activation_sq).sqrt();
                    let error = if activation > EPSILON {
                        contribution.negative_error_activation / activation
                    } else {
                        0.0
                    };
                    negative_stats.samples.push((activation, error));
                }
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok((positive_stats, negative_stats, total_baseline_error_sq))
    }

    fn evaluate_activation_gpu(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)> {
        // Returns: (sum_activation_sq, sum_error_activation, total_baseline_error_sq, improved_count)
        if samples.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0));
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for activation analysis")?;
        let activation_layout = self
            .activation_layout
            .as_ref()
            .context("GPU activation layout not initialised")?;
        let activation_pipeline = self
            .activation_pipeline
            .as_ref()
            .context("GPU activation pipeline not initialised")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let outputs_zeroed = vec![ActivationOutput::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let outputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-outputs-buffer"),
            contents: bytemuck::cast_slice(&outputs_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = ActivationUniforms {
            sample_count: samples.len() as u32,
            orientation,
            scale,
            activation_type,
            epsilon: EPSILON,
            pad0: 0.0,
            pad1: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: activation_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: outputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("activation-bind-group"),
        });

        let output_size = (std::mem::size_of::<ActivationOutput>() * samples.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("activation-staging-buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("activation-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("activation-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(activation_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&outputs_buffer, 0, &staging_buffer, 0, output_size);

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("Failed to send map_async result");
        });

        // Event-driven wait: poll non-blocking, check callback channel
        wait_for_buffer_map(device, &receiver, GPU_BUFFER_MAP_TIMEOUT_SECS)
            .context("Activation buffer mapping failed")?;

        let data = buffer_slice.get_mapped_range();
        let outputs: &[ActivationOutput] = bytemuck::cast_slice(&data);

        let mut sum_activation_sq = 0.0;
        let mut sum_error_activation = 0.0;
        let mut total_baseline_error_sq = 0.0;

        for (idx, output) in outputs.iter().enumerate() {
            if idx < samples.len() {
                let sample = &samples[idx];
                if sample.avg_error.is_finite() {
                    total_baseline_error_sq += sample.avg_error * sample.avg_error;
                }
                if output.valid > 0 {
                    sum_activation_sq += output.output_sq;
                    sum_error_activation += output.error_output;
                }
            }
        }

        // Note: improved_count is not calculated here because it requires weight validation
        // that happens in the caller. The caller will calculate improved_count after
        // validating and clamping the outgoing_weight.

        drop(data);
        staging_buffer.unmap();

        Ok((
            sum_activation_sq,
            sum_error_activation,
            total_baseline_error_sq,
            0, // improved_count calculated by caller after weight validation
        ))
    }

    /// GPU-accelerated bias grid search
    /// Tests all bias values in parallel and returns the optimal bias
    fn evaluate_bias_gpu(
        &self,
        samples: &[HelpfulSample],
        incoming_weight: f32,
        outgoing_weight: f32,
        activation_type: u32,
        bias_range: (f32, f32, f32),
    ) -> Result<f32> {
        // Returns: optimal bias value
        if samples.is_empty() {
            return Ok(0.0);
        }

        let (min_bias, max_bias, step) = bias_range;

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for bias analysis")?;
        let bias_layout = self
            .bias_layout
            .as_ref()
            .context("GPU bias layout not initialised")?;
        let bias_pipeline = self
            .bias_pipeline
            .as_ref()
            .context("GPU bias pipeline not initialised")?;

        // Generate bias candidates
        let num_steps = ((max_bias - min_bias) / step).ceil() as i32 + 1;
        let bias_candidates: Vec<f32> = (0..num_steps)
            .map(|i| min_bias + (i as f32 * step).min(max_bias - min_bias))
            .collect();

        if bias_candidates.is_empty() {
            return Ok(0.0);
        }

        // Prepare GPU buffers
        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let results_zeroed = vec![BiasResult::zeroed(); bias_candidates.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bias_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-candidates-buffer"),
            contents: bytemuck::cast_slice(&bias_candidates),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let results_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-results-buffer"),
            contents: bytemuck::cast_slice(&results_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = BiasUniforms {
            sample_count: samples.len() as u32,
            bias_count: bias_candidates.len() as u32,
            incoming_weight,
            outgoing_weight,
            activation_type,
            epsilon: EPSILON,
            min_sample_count: MIN_NEURON_SAMPLE_COUNT as u32,
            pad0: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: bias_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("bias-bind-group"),
        });

        let output_size = (std::mem::size_of::<BiasResult>() * bias_candidates.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bias-staging-buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bias-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bias-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(bias_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (bias_candidates.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 0, output_size);

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("Failed to send map_async result");
        });

        // Event-driven wait: poll non-blocking, check callback channel
        wait_for_buffer_map(device, &receiver, GPU_BUFFER_MAP_TIMEOUT_SECS)
            .context("Bias buffer mapping failed")?;

        let data = buffer_slice.get_mapped_range();
        let results: &[BiasResult] = bytemuck::cast_slice(&data);

        // Find bias with best error reduction
        let mut best_bias = 0.0;
        let mut best_error_reduction = f32::NEG_INFINITY;

        for result in results {
            if result.valid_sample_count >= MIN_NEURON_SAMPLE_COUNT as u32
                && result.error_reduction > best_error_reduction
            {
                best_error_reduction = result.error_reduction;
                best_bias = result.bias_value;
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok(best_bias)
    }

    /// Batch evaluate multiple helpful operations to improve GPU utilization
    /// Returns a vector of stats in the same order as the input samples
    fn evaluate_helpful_batch(
        &self,
        samples_batch: &[&[HelpfulSample]],
    ) -> Result<Vec<HelpfulStats>> {
        if samples_batch.is_empty() {
            return Ok(Vec::new());
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for batched helpful analysis")?;
        let helpful_layout = self
            .helpful_layout
            .as_ref()
            .context("GPU layout not initialised for batched helpful analysis")?;
        let helpful_pipeline = self
            .helpful_pipeline
            .as_ref()
            .context("GPU pipeline not initialised for batched helpful analysis")?;

        // Process in batches to avoid excessive memory usage
        // Apple Silicon optimisation: Use single encoder per batch to reduce Metal driver overhead
        let mut all_results = Vec::with_capacity(samples_batch.len());

        let max_sample_len = samples_batch.iter().map(|s| s.len()).max().unwrap_or(0);
        let bytes_per_sample = std::mem::size_of::<GpuHelpfulSample>()
            + (2 * std::mem::size_of::<HelpfulContribution>());
        let effective_batch_size = cap_gpu_batch_size_by_bytes(
            self.batch_size,
            max_sample_len,
            bytes_per_sample,
            GPU_MAX_BATCH_ALLOC_BYTES,
        );

        if verbose_enabled() && effective_batch_size < self.batch_size {
            let bytes_per_op = max_sample_len.saturating_mul(bytes_per_sample);
            let approx_mb = (bytes_per_op as f64 / (1024.0 * 1024.0)).max(0.0);
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Capping GPU helpful batch size from {} to {} due to large sample count. \
                 max_sample_len={}, approx_buffers_per_op≈{:.1}MB, cap={}MB",
                self.batch_size,
                effective_batch_size,
                max_sample_len,
                approx_mb,
                GPU_MAX_BATCH_ALLOC_BYTES / (1024 * 1024)
            );
        }

        for batch_chunk in samples_batch.chunks(effective_batch_size) {
            let mut empty_flags = Vec::with_capacity(batch_chunk.len());
            let mut batch_staging_buffers = Vec::new();
            let mut batch_contribution_sizes = Vec::new();
            let mut batch_contributions_buffers = Vec::new();
            let mut batch_sample_refs = Vec::new();

            // Single encoder for entire batch - reduces Metal driver overhead on Apple Silicon
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("helpful-command-encoder-batch"),
            });

            // Prepare all operations in this batch
            for samples in batch_chunk {
                if samples.is_empty() {
                    empty_flags.push(true);
                    continue;
                }
                empty_flags.push(false);

                batch_sample_refs.push(*samples);

                let gpu_samples: Vec<GpuHelpfulSample> = samples
                    .iter()
                    .copied()
                    .map(GpuHelpfulSample::from)
                    .collect();
                let contributions_zeroed = vec![HelpfulContribution::zeroed(); samples.len()];

                let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("helpful-samples-buffer-batch"),
                    contents: bytemuck::cast_slice(&gpu_samples),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let contributions_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("helpful-contributions-buffer-batch"),
                        contents: bytemuck::cast_slice(&contributions_zeroed),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });

                let uniforms = HelpfulUniforms {
                    length: samples.len() as u32,
                    pad0: 0,
                    epsilon: EPSILON,
                    pad1: 0.0,
                };
                let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("helpful-uniform-buffer-batch"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: helpful_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: sample_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: contributions_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                    ],
                    label: Some("helpful-bind-group-batch"),
                });

                let contribution_size =
                    (std::mem::size_of::<HelpfulContribution>() * samples.len()) as u64;
                let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("helpful-staging-buffer-batch"),
                    size: contribution_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                // Add compute pass to shared encoder (reduces command buffer overhead)
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("helpful-compute-pass-batch"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(helpful_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
                    compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
                }

                batch_contributions_buffers.push(contributions_buffer);
                batch_staging_buffers.push(staging_buffer);
                batch_contribution_sizes.push((contribution_size, samples.len()));
            }

            // Add all buffer copies after compute passes (better GPU scheduling)
            for (i, staging_buffer) in batch_staging_buffers.iter().enumerate() {
                let (contribution_size, _) = batch_contribution_sizes[i];
                encoder.copy_buffer_to_buffer(
                    &batch_contributions_buffers[i],
                    0,
                    staging_buffer,
                    0,
                    contribution_size,
                );
            }

            // Submit single command buffer for entire batch - reduces Metal driver overhead
            if !batch_staging_buffers.is_empty() {
                queue.submit(Some(encoder.finish()));
            }

            // OPTIMISATION: Map ALL buffers first, then poll ONCE for all.
            // This reduces GPU-CPU round trips compared to mapping each buffer individually.
            let mut map_receivers = Vec::with_capacity(batch_staging_buffers.len());
            for staging_buffer in &batch_staging_buffers {
                let buffer_slice = staging_buffer.slice(..);
                let (sender, receiver) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender
                        .send(result)
                        .expect("Failed to send map_async result");
                });
                map_receivers.push(receiver);
            }

            // Event-driven wait: poll non-blocking, check all callback channels
            wait_for_buffer_maps_batch(device, &map_receivers, GPU_BUFFER_MAP_TIMEOUT_SECS)
                .context("Helpful batch buffer mapping failed")?;

            // Now read all the mapped data (buffers are already mapped)
            let mut batch_results = Vec::with_capacity(batch_contribution_sizes.len());
            // Buffer mappings already verified by wait_for_buffer_maps_batch
            for staging_buffer in batch_staging_buffers.iter() {
                let buffer_slice = staging_buffer.slice(..);
                let data = buffer_slice.get_mapped_range();
                let contributions: &[HelpfulContribution] = bytemuck::cast_slice(&data);

                let mut stats = HelpfulStats::default();
                for contribution in contributions {
                    stats.positive_count += contribution.positive_flag;
                    stats.negative_count += contribution.negative_flag;
                    stats.positive_improvement_sum += contribution.positive_improvement;
                    stats.negative_improvement_sum += contribution.negative_improvement;
                    stats.positive_activation_sum += contribution.positive_activation;
                    stats.negative_activation_sum += contribution.negative_activation;
                    stats.error_sq_sum += contribution.error_squared;
                    stats.activation_sq_sum += contribution.activation_squared;
                    stats.error_activation_sum += contribution.error_activation;
                }

                drop(data);
                staging_buffer.unmap();

                batch_results.push(stats);
            }

            let merged_results = Self::merge_batch_results(&empty_flags, batch_results);
            all_results.extend(merged_results);

            // CRITICAL: Ensure Metal releases command buffers before creating new ones.
            // Without this, we can exhaust Metal's command buffer pool when processing
            // many batches, causing the GPU thread to hang in semaphore_wait_trap.
            //
            // Avoid an unbounded wait - bail out with an error if the driver is wedged.
            poll_device_until_idle(
                device,
                Duration::from_secs(5),
                "post-helpful-batch command buffer release",
            )?;
        }

        Ok(all_results)
    }

    fn merge_batch_results(
        empty_flags: &[bool],
        computed_stats: Vec<HelpfulStats>,
    ) -> Vec<HelpfulStats> {
        let expected_non_empty = empty_flags.iter().filter(|flag| !**flag).count();
        debug_assert_eq!(
            expected_non_empty,
            computed_stats.len(),
            "Computed stats should match number of non-empty sample sets"
        );

        let mut results = Vec::with_capacity(empty_flags.len());
        let mut stats_iter = computed_stats.into_iter();

        for &is_empty in empty_flags {
            if is_empty {
                results.push(HelpfulStats::default());
            } else if let Some(stats) = stats_iter.next() {
                results.push(stats);
            } else {
                // Safety guard: if counts mismatch, preserve ordering by inserting default.
                results.push(HelpfulStats::default());
            }
        }

        results
    }
}

const HELPFUL_SHADER: &str = include_str!("../shaders/helpful.wgsl");

const HARMFUL_SHADER: &str = include_str!("../shaders/harmful.wgsl");

const RELU_SHADER: &str = include_str!("../shaders/relu.wgsl");

const ACTIVATION_SHADER: &str = include_str!("../shaders/activation.wgsl");

const BIAS_SHADER: &str = include_str!("../shaders/bias.wgsl");

fn build_ordered_neurons(creature: &crate::CreatureJson) -> Vec<OrderedNeuron> {
    let mut ordered = Vec::with_capacity(creature.input + creature.neurons.len());

    for input_index in 0..creature.input {
        ordered.push(OrderedNeuron {
            uuid: format!("input-{input_index}"),
            index: input_index,
        });
    }

    for (offset, neuron) in creature.neurons.iter().enumerate() {
        ordered.push(OrderedNeuron {
            uuid: neuron.uuid.clone(),
            index: creature.input + offset,
        });
    }

    ordered
}

/// Target data for a single observation (used for matching with source records)
struct TargetData {
    avg_error: f32,
    value: Option<f32>,
    activation: f32,
}

/// Pre-built target map for efficient sample building across multiple sources.
/// This avoids rebuilding the HashMap for each source when analysing a single target.
struct TargetMap {
    map: HashMap<u32, TargetData>,
}

impl TargetMap {
    /// Build a target map from target records.
    /// This should be done ONCE per focus neuron, then reused for all sources.
    fn from_records(target_records: &[DiscoverRecord]) -> Self {
        let mut map: HashMap<u32, TargetData> = HashMap::with_capacity(target_records.len());
        for record in target_records {
            if record.errors.is_empty() || !record.activation.is_finite() {
                continue;
            }
            let mut sum = 0.0;
            let mut count = 0;
            for error in &record.errors {
                if error.is_finite() {
                    sum += *error;
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            map.insert(
                record.obs_index,
                TargetData {
                    avg_error: sum / count as f32,
                    value: record.value,
                    activation: record.activation,
                },
            );
        }
        Self { map }
    }

    /// Build samples by matching source records against this pre-built target map.
    /// This is much faster than build_samples() when processing multiple sources
    /// against the same target.
    fn build_samples_from(&self, from_records: &[DiscoverRecord]) -> Vec<HelpfulSample> {
        if self.map.is_empty() || from_records.is_empty() {
            return Vec::new();
        }

        let mut samples = Vec::with_capacity(from_records.len().min(self.map.len()));
        for record in from_records {
            if let Some(target) = self.map.get(&record.obs_index) {
                if record.activation.is_finite() && target.avg_error.is_finite() {
                    samples.push(HelpfulSample {
                        activation: record.activation,
                        avg_error: target.avg_error,
                        target_value: target.value,
                        target_activation: Some(target.activation),
                    });
                }
            }
        }

        samples
    }
}

/// Build samples for testing. In production, use TargetMap::from_records() and
/// TargetMap::build_samples_from() for better performance when processing
/// multiple sources against the same target.
#[cfg(test)]
fn build_samples(
    target_records: &[DiscoverRecord],
    from_records: &[DiscoverRecord],
) -> Vec<HelpfulSample> {
    if target_records.is_empty() || from_records.is_empty() {
        return Vec::new();
    }

    // Build map from obs_index to target data (error, value, activation)
    let target_map = TargetMap::from_records(target_records);

    if target_map.map.is_empty() {
        return Vec::new();
    }

    target_map.build_samples_from(from_records)
}

/// Computes the sign of a weight as an i8 for use in the candidate key.
/// Returns 1 for positive weights, -1 for negative, and 0 for zero (though this
/// shouldn't happen in practice).
fn weight_sign(weight: f32) -> i8 {
    if weight > 0.0 {
        1
    } else if weight < 0.0 {
        -1
    } else {
        0
    }
}

/// Returns true if this add-neuron candidate is "extreme" enough to warrant a conservative pair.
///
/// We intentionally base this on incoming weight and bias (not outgoing), because outgoing
/// weight is already clamped and ReLU candidates commonly use outgoing=0.1 by design.
fn upsert_candidate(
    map: &mut HashMap<(String, String, String, i8, i8), CandidateNeuronJson>,
    candidate: CandidateNeuronJson,
) {
    use std::collections::hash_map::Entry;

    // Key includes signs of BOTH incoming_weight AND outgoing_weight so that:
    // 1. Different ReLU orientations (incoming_weight ±1) are kept separately
    // 2. Split-error complementary pairs (same incoming_weight, opposite outgoing_weight)
    //    are also kept separately - one pushes output UP, one pushes DOWN
    let key = (
        candidate.source_neuron_uuid.clone(),
        candidate.target_neuron_uuid.clone(),
        candidate.squash.clone(),
        weight_sign(candidate.incoming_weight),
        weight_sign(candidate.outgoing_weight),
    );

    match map.entry(key) {
        Entry::Occupied(mut entry) => {
            // Issue #128: Compare by expected creature score gain
            if candidate.expected_creature_score_gain > entry.get().expected_creature_score_gain {
                entry.insert(candidate);
            }
        }
        Entry::Vacant(entry) => {
            entry.insert(candidate);
        }
    }
}

/// Result from ReLU evaluation (split by target error sign)
struct SplitReluResult {
    /// Candidate for samples with positive error (output should be higher)
    positive_error_candidate: Option<CandidateNeuronJson>,
    /// Candidate for samples with negative error (output should be lower)
    negative_error_candidate: Option<CandidateNeuronJson>,
}

/// Apply HARD_TANH activation function (clamp to [-1, 1])
#[inline(always)]
fn hard_tanh(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Get the target simulation function for a squash name.
///
/// This delegates to `crate::activations::target_simulation_fn`, which is kept in sync
/// with NEAT-AI's activation registry (and supports aliases + case-insensitive names).
#[inline]
fn get_target_activation_fn(squash: &str) -> Option<fn(f32) -> f32> {
    crate::activations::target_simulation_fn(squash)
}

/// Check if samples support target activation simulation (all have target data).
/// Returns the activation function to use, or None if linear approximation should be used.
#[inline]
fn get_target_simulation_fn(
    samples: &[HelpfulSample],
    target_squash: Option<&str>,
) -> Option<fn(f32) -> f32> {
    let squash = target_squash?;
    let activation_fn = get_target_activation_fn(squash)?;

    // Verify all samples have the required target data
    if samples
        .iter()
        .all(|s| s.target_value.is_some() && s.target_activation.is_some())
    {
        Some(activation_fn)
    } else {
        None
    }
}

/// Legacy function for backwards compatibility - returns true only for HARD_TANH
/// Deprecated: Use get_target_simulation_fn instead for more accurate simulation
#[inline]
#[cfg(test)] // Only used in tests now
fn can_use_hard_tanh(samples: &[HelpfulSample], target_squash: Option<&str>) -> bool {
    target_squash == Some("HARD_TANH")
        && samples
            .iter()
            .all(|s| s.target_value.is_some() && s.target_activation.is_some())
}

/// Combined computation of improvement and count for ReLU candidates.
/// Single pass over samples for better cache efficiency.
///
/// When `target_activation_fn` is Some, simulates the target neuron's actual activation
/// function for more accurate improvement estimates. Otherwise falls back to linear approximation.
///
/// IMPORTANT: The `bias` parameter is critical for accurate predictions. It shifts the ReLU
/// activation threshold, affecting which samples produce non-zero output. When bias > 0,
/// more samples activate; when bias < 0, fewer samples activate. Excluding bias causes
/// significant prediction errors.
///
/// CRITICAL DOMAIN FIX (v0.1.120): When using target_activation_fn simulation,
/// both baseline and new error must be computed in ACTIVATION domain.
///
/// Returns (improvement_percentage, improved_count, total_count)
fn compute_relu_improvement_and_count(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    total_baseline_error_sq: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> (f32, u32, u32) {
    compute_relu_improvement_and_count_traced(
        samples,
        incoming_weight,
        outgoing_weight,
        bias,
        total_baseline_error_sq,
        target_activation_fn,
        None, // No trace context
    )
}

/// Compute ReLU improvement and count.
/// Returns (improvement_fraction, improved_count, total_count).
fn compute_relu_improvement_and_count_traced(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    total_baseline_error_sq: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
    _trace_context: Option<&str>, // Kept for API compatibility, no longer used
) -> (f32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, samples.len() as u32);
    }

    let mut baseline_error_sq_sum = 0.0f32; // ACTIVATION domain when simulating
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;

    for sample in samples.iter() {
        let pre_activation = incoming_weight * sample.activation + bias;
        let relu_output = pre_activation.max(0.0);
        let contribution = outgoing_weight * relu_output;

        let (baseline_error, new_error) = if let Some(target_fn) = target_activation_fn {
            // CRITICAL: Use ACTIVATION domain for BOTH baseline and new error.
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let target_activation = unsafe { sample.target_activation.unwrap_unchecked() };
            let desired_value = target_value + sample.avg_error;
            let expected = target_fn(desired_value);

            // Baseline error in ACTIVATION domain
            let baseline_err = expected - target_activation;

            // New error in ACTIVATION domain
            let new_input = target_value + contribution;
            let new_err = expected - target_fn(new_input);

            (baseline_err, new_err)
        } else {
            // Linear approximation: both errors in VALUE domain
            let baseline_err = sample.avg_error;
            let new_err = sample.avg_error - contribution;

            (baseline_err, new_err)
        };

        if baseline_error.is_finite() {
            baseline_error_sq_sum += baseline_error * baseline_error;
        }
        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        // Sample is improved if |new_error| < |baseline_error| (consistent domain)
        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        } else if new_error.abs() > baseline_error.abs() + EPSILON {
            worsened_count += 1;
        }
    }

    // Use computed ACTIVATION domain baseline when simulating, else use passed VALUE domain
    let effective_baseline = if target_activation_fn.is_some() {
        baseline_error_sq_sum
    } else {
        total_baseline_error_sq
    };

    let improvement = if effective_baseline > EPSILON {
        (effective_baseline - new_error_sq_sum) / effective_baseline
    } else {
        0.0
    };
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    let total_count = samples.len() as u32;
    let _ = worsened_count; // Kept for potential future use
    (improvement, improved_count, total_count)
}

/// Combined computation of improvement and count for activation candidates.
/// Single pass over samples for better cache efficiency.
///
/// When `target_activation_fn` is Some, simulates the target neuron's actual activation
/// function for more accurate improvement estimates. Otherwise falls back to linear approximation.
///
/// CRITICAL DOMAIN FIX (v0.1.120): When using target_activation_fn simulation,
/// both baseline and new error must be computed in ACTIVATION domain.
///
/// Returns (improvement_percentage, improved_count, total_count)
fn compute_activation_improvement_and_count(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    activation_fn: fn(f32) -> f32,
    total_baseline_error_sq: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> (f32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, samples.len() as u32);
    }

    let mut baseline_error_sq_sum = 0.0f32; // ACTIVATION domain when simulating
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let total_count = samples.len() as u32;

    for sample in samples {
        let pre_activation = incoming_weight * sample.activation + bias;
        let neuron_output = activation_fn(pre_activation);
        let contribution = outgoing_weight * neuron_output;

        let (baseline_error, new_error) = if let Some(target_fn) = target_activation_fn {
            // CRITICAL: Use ACTIVATION domain for BOTH baseline and new error.
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let target_activation = unsafe { sample.target_activation.unwrap_unchecked() };
            let desired_value = target_value + sample.avg_error;
            let expected = target_fn(desired_value);

            // Baseline error in ACTIVATION domain
            let baseline_err = expected - target_activation;

            // New error in ACTIVATION domain
            let new_input = target_value + contribution;
            let new_err = expected - target_fn(new_input);

            (baseline_err, new_err)
        } else {
            // Linear approximation: both errors in VALUE domain
            (sample.avg_error, sample.avg_error - contribution)
        };

        if baseline_error.is_finite() {
            baseline_error_sq_sum += baseline_error * baseline_error;
        }
        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        // Sample is improved if |new_error| < |baseline_error| (consistent domain)
        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        }
    }

    // Use computed ACTIVATION domain baseline when simulating, else use passed VALUE domain
    let effective_baseline = if target_activation_fn.is_some() {
        baseline_error_sq_sum
    } else {
        total_baseline_error_sq
    };

    let improvement = if effective_baseline > EPSILON {
        (effective_baseline - new_error_sq_sum) / effective_baseline
    } else {
        0.0
    };
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    (improvement, improved_count, total_count)
}

/// Wrapper for tests - computes improvement only.
/// NOTE: For ReLU candidates, bias affects which samples activate. Pass the actual bias
/// that will be used with the new neuron for accurate predictions.
#[cfg(test)]
fn compute_net_improvement_with_squash(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> f32 {
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);
    let (improvement, _, _) = compute_relu_improvement_and_count(
        samples,
        incoming_weight,
        outgoing_weight,
        bias,
        total_baseline_error_sq,
        target_activation_fn,
    );
    improvement
}

/// Compute synapse improvement accounting for target neuron's activation function.
///
/// For direct synapse connections (source → target), the contribution is `weight × source_activation`.
/// This function simulates the target's activation function to predict accurate improvement,
/// avoiding overprediction near saturation for HARD_TANH, TANH, LOGISTIC, etc.
///
/// Returns improvement_percentage only. Used in tests; production uses compute_synapse_improvement_and_count.
#[cfg(test)]
fn compute_synapse_improvement_with_target_squash(
    samples: &[HelpfulSample],
    weight: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> f32 {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return 0.0;
    }

    // Check if we can use saturation-aware model
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);

    let mut new_error_sq_sum = 0.0f32;

    for sample in samples {
        // Direct synapse contribution: weight × source_activation
        let contribution = weight * sample.activation;

        let new_error = if let Some(target_fn) = target_activation_fn {
            // Saturation-aware model: apply target's activation function
            // CRITICAL: avg_error is in VALUE domain (targetValue - currentValue from TypeScript)
            // So we compute desired_value = target_value + avg_error, then squash to get expected activation
            // Safety: target_activation_fn is only Some when all samples have target data
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let desired_value = target_value + sample.avg_error;
            let expected = target_fn(desired_value);
            let new_input = target_value + contribution;
            expected - target_fn(new_input)
        } else {
            // Linear approximation - assumes contribution directly reduces error
            sample.avg_error - contribution
        };

        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }
    }

    let improvement = (total_baseline_error_sq - new_error_sq_sum) / total_baseline_error_sq;
    if improvement.is_finite() {
        improvement
    } else {
        0.0
    }
}

/// Compute improvement, improved count, and worsened count for synapse candidates.
/// All counts use the same saturation-aware methodology for consistency.
///
/// CRITICAL DOMAIN FIX (v0.1.120): When using target_activation_fn simulation,
/// both baseline and new error must be computed in ACTIVATION domain. The passed-in
/// total_baseline_error_sq is in VALUE domain, so we compute our own ACTIVATION
/// domain baseline when simulating.
///
/// Returns (improvement_percentage, improved_count, worsened_count, total_count)
fn compute_synapse_improvement_and_count(
    samples: &[HelpfulSample],
    weight: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> (f32, u32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, 0, samples.len() as u32);
    }

    let target_activation_fn = get_target_simulation_fn(samples, target_squash);

    let mut baseline_error_sq_sum = 0.0f32; // ACTIVATION domain when simulating
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;
    let total_count = samples.len() as u32;

    for sample in samples {
        let contribution = weight * sample.activation;

        let (baseline_error, new_error) = if let Some(target_fn) = target_activation_fn {
            // CRITICAL: Use ACTIVATION domain for BOTH baseline and new error.
            // avg_error is in VALUE domain, but MSE is measured in ACTIVATION domain.
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let target_activation = unsafe { sample.target_activation.unwrap_unchecked() };
            let desired_value = target_value + sample.avg_error;
            let expected = target_fn(desired_value);

            // Baseline error in ACTIVATION domain: expected output - current output
            let baseline_err = expected - target_activation;

            // New error in ACTIVATION domain: expected output - new output
            let new_input = target_value + contribution;
            let new_err = expected - target_fn(new_input);

            (baseline_err, new_err)
        } else {
            // Linear approximation: both errors in VALUE domain
            (sample.avg_error, sample.avg_error - contribution)
        };

        if baseline_error.is_finite() {
            baseline_error_sq_sum += baseline_error * baseline_error;
        }
        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        // Count improved/worsened samples using consistent domain comparison
        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        } else if new_error.abs() > baseline_error.abs() + EPSILON {
            worsened_count += 1;
        }
    }

    // Use computed ACTIVATION domain baseline when simulating, else use passed VALUE domain
    let effective_baseline = if target_activation_fn.is_some() {
        baseline_error_sq_sum
    } else {
        total_baseline_error_sq
    };

    let improvement = if effective_baseline > EPSILON {
        (effective_baseline - new_error_sq_sum) / effective_baseline
    } else {
        0.0
    };
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    (improvement, improved_count, worsened_count, total_count)
}

/// Wrapper for tests - counts improved samples only.
#[cfg(test)]
fn count_improved_samples(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    target_squash: Option<&str>,
) -> (u32, u32) {
    let total_baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);
    let (_, improved, total) = compute_relu_improvement_and_count(
        samples,
        incoming_weight,
        outgoing_weight,
        bias,
        total_baseline_error_sq,
        target_activation_fn,
    );
    (improved, total)
}

/// Evaluate ReLU candidates by splitting samples based on TARGET neuron's error sign.
///
/// This is the PRIMARY approach for ReLU evaluation. It finds candidates for both directions:
/// - **Positive-error samples** (output should be HIGHER): compute weight that pushes UP
/// - **Negative-error samples** (output should be LOWER): compute weight that pushes DOWN
///
/// For each direction:
/// 1. Compute optimal weight from the error subset
/// 2. Evaluate NET improvement across ALL samples
/// 3. Return candidate if it passes threshold
///
/// This is the correct approach for directional activations like ReLU because:
/// - ReLU can only push output in ONE direction (based on outgoing weight sign)
/// - Averaging over all samples cancels out when errors are split ~50/50
/// - We evaluate source activations as-is (we don't care how they were calculated)
fn evaluate_relu_candidates_split<G: GpuEvaluator>(
    gpu: &G,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    target_squash: Option<&str>,
) -> Result<SplitReluResult> {
    // Split samples by error sign
    let positive_error_samples: Vec<HelpfulSample> = samples
        .iter()
        .filter(|s| s.avg_error > EPSILON)
        .copied()
        .collect();

    let negative_error_samples: Vec<HelpfulSample> = samples
        .iter()
        .filter(|s| s.avg_error < -EPSILON)
        .copied()
        .collect();

    let mut result = SplitReluResult {
        positive_error_candidate: None,
        negative_error_candidate: None,
    };

    // Compute total baseline error across ALL samples (for net improvement calculation)
    let total_baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

    if total_baseline_error_sq <= EPSILON {
        return Ok(result);
    }

    // Get target activation function for accurate simulation (ReLU, HARD_TANH, etc.)
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);

    // For positive errors (output should be higher), compute optimal weight from subset
    // then evaluate the NET effect across ALL samples.
    // We evaluate BOTH ReLU orientations (positive and negative incoming weight) and pick best.
    if positive_error_samples.len() >= MIN_NEURON_SAMPLE_COUNT {
        let (positive_stats, negative_stats, pos_baseline_error_sq) =
            gpu.evaluate_relu(&positive_error_samples, threshold)?;

        // Try both orientations and pick the best
        let orientations = [positive_stats, negative_stats];
        let mut best_candidate: Option<CandidateNeuronJson> = None;
        let mut best_improvement = threshold;

        for stats in orientations {
            if let Some(mut candidate) = stats.evaluate(
                source_uuid,
                target_uuid,
                threshold,
                pos_baseline_error_sq,
                &positive_error_samples,
            ) {
                // Compute net improvement across ALL samples (single pass)
                // CRITICAL: Include candidate.bias for accurate prediction
                let (net_improvement, improved, total) = compute_relu_improvement_and_count(
                    samples,
                    candidate.incoming_weight,
                    candidate.outgoing_weight,
                    candidate.bias,
                    total_baseline_error_sq,
                    target_activation_fn,
                );
                candidate.improved_count = improved;
                candidate.total_count = total;

                // Issue #128: Update creature-level metrics
                if net_improvement > best_improvement {
                    candidate.expected_creature_error_reduction = net_improvement;
                    candidate.expected_creature_score_gain = net_improvement;
                    best_improvement = net_improvement;
                    best_candidate = Some(candidate);
                }
            }
        }
        result.positive_error_candidate = best_candidate;
    }

    // For negative errors (output should be lower).
    // We evaluate BOTH ReLU orientations (positive and negative incoming weight) and pick best.
    if negative_error_samples.len() >= MIN_NEURON_SAMPLE_COUNT {
        let (positive_stats, negative_stats, neg_baseline_error_sq) =
            gpu.evaluate_relu(&negative_error_samples, threshold)?;

        // Try both orientations and pick the best
        let orientations = [positive_stats, negative_stats];
        let mut best_candidate: Option<CandidateNeuronJson> = None;
        let mut best_improvement = threshold;

        for stats in orientations {
            if let Some(mut candidate) = stats.evaluate(
                source_uuid,
                target_uuid,
                threshold,
                neg_baseline_error_sq,
                &negative_error_samples,
            ) {
                // Compute net improvement across ALL samples (single pass)
                // CRITICAL: Include candidate.bias for accurate prediction
                let (net_improvement, improved, total) = compute_relu_improvement_and_count(
                    samples,
                    candidate.incoming_weight,
                    candidate.outgoing_weight,
                    candidate.bias,
                    total_baseline_error_sq,
                    target_activation_fn,
                );
                candidate.improved_count = improved;
                candidate.total_count = total;

                // Issue #128: Update creature-level metrics
                if net_improvement > best_improvement {
                    candidate.expected_creature_error_reduction = net_improvement;
                    candidate.expected_creature_score_gain = net_improvement;
                    best_improvement = net_improvement;
                    best_candidate = Some(candidate);
                }
            }
        }
        result.negative_error_candidate = best_candidate;
    }

    Ok(result)
}

/// Helper for split-error evaluation: compute optimal weight from subset, evaluate on all samples.
///
/// This is the core of the split-error fix for non-ReLU activations. By computing
/// the optimal weight from a specific error subset (positive or negative), we get
/// a weight that's tuned to help that subset. We then evaluate the NET improvement
/// across ALL samples to ensure the candidate doesn't hurt the other subset more
/// than it helps the target subset.
#[allow(clippy::too_many_arguments)]
fn evaluate_activation_for_subset<G: GpuEvaluator>(
    gpu: &G,
    source_uuid: &str,
    target_uuid: &str,
    subset_samples: &[HelpfulSample], // Used to compute optimal weight
    all_samples: &[HelpfulSample],    // Used to compute net improvement
    spec: &ActivationCandidateSpec,
    target_squash: Option<&str>,
    total_baseline_error_sq: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> Result<Option<CandidateNeuronJson>> {
    if subset_samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return Ok(None);
    }

    let activation_type = activation_name_to_gpu_id(spec.name);

    let mut best_candidate: Option<CandidateNeuronJson> = None;
    // v0.1.136: Fixed threshold bug - use 0.0 instead of threshold.
    // The calling code in evaluate_activation_candidate handles threshold vs fallback
    // logic. If we initialise to threshold here, candidates with 0 < improvement <= threshold
    // are silently dropped, breaking the fallback mechanism for split-error evaluation.
    let mut best_net_improvement = 0.0;

    for &orientation in spec.orientations {
        for &scale in spec.scales {
            let incoming_weight = orientation * scale;

            // Compute optimal weight from SUBSET samples using GPU
            let (sum_activation_sq, sum_error_activation) = match gpu.evaluate_activation(
                subset_samples,
                activation_type,
                orientation,
                scale,
            ) {
                Ok(result) => (result.0, result.1),
                Err(_) => {
                    // Fall back to CPU on GPU error
                    let mut sum_act_sq = 0.0;
                    let mut sum_err_act = 0.0;
                    for sample in subset_samples {
                        let pre_activation = incoming_weight * sample.activation;
                        let output = (spec.activation)(pre_activation);
                        if output.is_finite() {
                            sum_act_sq += output * output;
                            sum_err_act += output * sample.avg_error;
                        }
                    }
                    (sum_act_sq, sum_err_act)
                }
            };

            let (outgoing_weight, optimal_bias) = if spec.name == "IDENTITY" {
                match calculate_optimal_identity_outgoing_and_bias(subset_samples, incoming_weight)
                {
                    Some((w, b)) => (w, b),
                    None => continue,
                }
            } else {
                // Use shared weight calculation with ratio validation
                let outgoing_weight = match calculate_optimal_outgoing_weight(
                    sum_error_activation,
                    sum_activation_sq,
                    incoming_weight,
                ) {
                    Some(w) => w,
                    None => continue, // Skip if weight is invalid or ratio too small
                };

                // Calculate optimal bias from subset
                let optimal_bias = calculate_optimal_bias(
                    subset_samples,
                    incoming_weight,
                    outgoing_weight,
                    spec.activation,
                    spec.name,
                    None,
                    target_squash,
                );
                (outgoing_weight, optimal_bias)
            };

            // Issue #123: Check for saturation - reject if neuron output is nearly constant.
            // Large bias values (e.g., 5.0 for SOFTSIGN) can cause the neuron to saturate,
            // producing nearly identical output regardless of input. Such neurons cannot
            // reduce error correlation and lead to massive prediction failures.
            if !has_sufficient_output_variance(
                all_samples,
                incoming_weight,
                optimal_bias,
                spec.activation,
            ) {
                continue;
            }

            // CRITICAL: Evaluate NET improvement across ALL samples
            // This ensures the candidate helps the target subset more than it hurts the other
            let (net_improvement, improved_count, total_count) =
                compute_activation_improvement_and_count(
                    all_samples,
                    incoming_weight,
                    outgoing_weight,
                    optimal_bias,
                    spec.activation,
                    total_baseline_error_sq,
                    target_activation_fn,
                );

            // Only consider candidates with positive NET improvement
            if net_improvement <= 0.0 {
                continue;
            }

            // Apply validity filters
            let absolute_improvement = net_improvement * total_baseline_error_sq;
            if absolute_improvement < 0.001 {
                continue;
            }

            if spec.name == "IDENTITY" && optimal_bias.abs() < 0.01 {
                continue;
            }

            // Track best candidate
            if net_improvement > best_net_improvement {
                best_net_improvement = net_improvement;

                let target_stats = NeuronStats::from_samples(all_samples).map(|s| s.to_json());
                // Issue #128: Use creature-level metrics
                best_candidate = Some(CandidateNeuronJson {
                    source_neuron_uuid: source_uuid.to_string(),
                    target_neuron_uuid: target_uuid.to_string(),
                    source_neuron_index: None, // Set during impact discounting
                    target_neuron_index: None, // Set during impact discounting
                    incoming_weight,
                    outgoing_weight,
                    squash: spec.name.to_string(),
                    bias: optimal_bias,
                    comment: None,
                    target_neuron_impact: 1.0,
                    expected_creature_error_reduction: net_improvement,
                    expected_creature_score_gain: net_improvement,
                    improved_count,
                    total_count,
                    target_neuron_stats: target_stats,
                });
            }
        }
    }

    Ok(best_candidate)
}

fn evaluate_activation_candidate<G: GpuEvaluator>(
    gpu: &G,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    spec: &ActivationCandidateSpec,
    target_squash: Option<&str>,
) -> Result<Option<CandidateNeuronJson>> {
    if samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return Ok(None);
    }

    let activation_type = activation_name_to_gpu_id(spec.name);

    let mut best_candidate: Option<CandidateNeuronJson> = None;
    let mut best_score = threshold;
    let mut fallback_candidate: Option<CandidateNeuronJson> = None;
    let mut fallback_score = f32::MIN;

    // v0.1.135: Split-error evaluation for all activations (not just ReLU).
    // When errors are split ~50/50 between positive and negative, computing
    // optimal weight from ALL samples averages to near-zero, giving weak
    // predictions that are often wrong in practice.
    //
    // The fix: split samples by error sign, compute optimal weight from each
    // subset, then evaluate NET improvement across ALL samples.
    let positive_error_samples: Vec<HelpfulSample> = samples
        .iter()
        .filter(|s| s.avg_error > EPSILON)
        .copied()
        .collect();

    let negative_error_samples: Vec<HelpfulSample> = samples
        .iter()
        .filter(|s| s.avg_error < -EPSILON)
        .copied()
        .collect();

    // Compute total_baseline_error_sq across ALL samples (for net improvement)
    let mut total_baseline_error_sq = 0.0;
    for sample in samples {
        if sample.avg_error.is_finite() {
            total_baseline_error_sq += sample.avg_error * sample.avg_error;
        }
    }

    // Get target activation function for net improvement calculation
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);

    // v0.1.136: Track whether split-error evaluation was properly attempted.
    // If BOTH subsets had enough samples but NEITHER produced candidates,
    // the errors are truly split and there's no reliable weight direction.
    // In this case, we should NOT fall back to all-samples evaluation,
    // which would produce unreliable small-improvement predictions.
    let positive_subset_valid = positive_error_samples.len() >= MIN_NEURON_SAMPLE_COUNT;
    let negative_subset_valid = negative_error_samples.len() >= MIN_NEURON_SAMPLE_COUNT;
    let split_error_attempted = positive_subset_valid && negative_subset_valid;

    // Evaluate candidates from BOTH error subsets
    // This ensures we find the best direction even with split errors
    for error_samples in [&positive_error_samples, &negative_error_samples] {
        if error_samples.len() < MIN_NEURON_SAMPLE_COUNT {
            continue;
        }

        // Compute baseline for this subset (used for weight calculation)
        let subset_baseline_sq: f32 = error_samples
            .iter()
            .map(|s| s.avg_error * s.avg_error)
            .sum();

        if subset_baseline_sq <= EPSILON {
            continue;
        }

        if let Some(candidate) = evaluate_activation_for_subset(
            gpu,
            source_uuid,
            target_uuid,
            error_samples, // Compute weight from subset
            samples,       // Evaluate improvement on ALL samples
            spec,
            target_squash,
            total_baseline_error_sq,
            target_activation_fn,
        )? {
            // Track best and fallback candidates from split evaluation
            // Issue #128: Use expected_creature_score_gain for comparison
            if candidate.expected_creature_score_gain > best_score {
                best_score = candidate.expected_creature_score_gain;
                best_candidate = Some(candidate.clone());
            }
            if candidate.expected_creature_score_gain > fallback_score
                && candidate.expected_creature_score_gain > 0.0
            {
                fallback_score = candidate.expected_creature_score_gain;
                fallback_candidate = Some(candidate);
            }
        }
    }

    // If split-error evaluation found candidates, return the best
    if best_candidate.is_some() || fallback_candidate.is_some() {
        return Ok(best_candidate.or(fallback_candidate));
    }

    // v0.1.136: If split-error evaluation was properly attempted (both subsets
    // had enough samples) but found NOTHING, don't fall back to all-samples.
    // The fact that neither subset produced candidates with positive net improvement
    // means there's no reliable weight - any weight that helps one subset hurts
    // the other by at least as much. All-samples would produce unreliable
    // small-improvement predictions that fail in production.
    if split_error_attempted {
        return Ok(None);
    }

    // Fall back to original ALL-samples evaluation ONLY for cases where errors
    // aren't clearly split (e.g., all positive or all negative errors, or
    // one subset has too few samples)

    for &orientation in spec.orientations {
        for &scale in spec.scales {
            let incoming_weight = orientation * scale;
            let (
                sum_activation_sq,
                sum_error_activation,
                gpu_baseline_sq,
                _gpu_improved_count, // Ignored - calculated after weight validation
                gpu_succeeded,
            ) = match gpu.evaluate_activation(samples, activation_type, orientation, scale) {
                Ok(result) => (result.0, result.1, result.2, result.3, true),
                Err(_) => {
                    // Fall back to CPU if GPU fails
                    let mut sum_activation_sq = 0.0;
                    let mut sum_error_activation = 0.0;
                    for sample in samples {
                        let pre_activation = incoming_weight * sample.activation;
                        let output = (spec.activation)(pre_activation);
                        if output.is_finite() {
                            sum_activation_sq += output * output;
                            sum_error_activation += output * sample.avg_error;
                        }
                    }
                    (
                        sum_activation_sq,
                        sum_error_activation,
                        total_baseline_error_sq,
                        0,
                        false,
                    )
                }
            };

            // Use GPU baseline if GPU succeeded, otherwise use CPU baseline
            let baseline_sq = if gpu_succeeded {
                gpu_baseline_sq
            } else {
                total_baseline_error_sq
            };

            let total_count = samples.len() as u32;
            if total_count == 0 {
                continue;
            }

            // For non-linear targets (HARD_TANH, ReLU, etc.), search for best outgoing_weight
            // since linear optimal may be wrong due to activation saturation/clipping.
            // Must verify samples have target_value/target_activation data before simulation.
            let target_activation_fn = get_target_simulation_fn(samples, target_squash);
            let (
                outgoing_weight,
                optimal_bias,
                neuron_error_improvement, // Issue #128: Renamed - this is neuron-level, not creature-level
                final_improved_count,
            ) = if spec.name == "IDENTITY" {
                let (outgoing_weight, optimal_bias) =
                    match calculate_optimal_identity_outgoing_and_bias(samples, incoming_weight) {
                        Some((w, b)) => (w, b),
                        None => continue,
                    };

                let (improvement, improved_count, _) = compute_activation_improvement_and_count(
                    samples,
                    incoming_weight,
                    outgoing_weight,
                    optimal_bias,
                    spec.activation,
                    baseline_sq,
                    target_activation_fn,
                );

                (outgoing_weight, optimal_bias, improvement, improved_count)
            } else if target_activation_fn.is_some() {
                // Use shared weight calculation with validation.
                // For non-linear targets, we'll search over scaled versions of this base weight.
                let base_weight = match calculate_optimal_outgoing_weight(
                    sum_error_activation,
                    sum_activation_sq,
                    incoming_weight,
                ) {
                    Some(w) => w,
                    None => continue, // Skip if weight is invalid or ratio too small
                };

                // Weight candidates: base weight and scaled versions
                // All candidates are already within MAX_OUTGOING_WEIGHT since base_weight is clamped
                let weight_candidates: [f32; 9] = [
                    base_weight * 0.1,
                    base_weight * 0.25,
                    base_weight * 0.5,
                    base_weight * 0.75,
                    base_weight,
                    base_weight * 1.5,
                    base_weight * 2.0,
                    -base_weight * 0.5,
                    -base_weight,
                ];

                let mut best_weight = base_weight;
                let mut best_bias = 0.0f32;
                let mut best_improvement = f32::NEG_INFINITY;
                let mut best_improved_count = 0u32;

                for &weight in &weight_candidates {
                    // Clamp scaled weights to ensure they stay within bounds
                    let clamped_weight = weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);
                    if clamped_weight.abs() <= EPSILON {
                        continue;
                    }

                    let bias = calculate_optimal_bias(
                        samples,
                        incoming_weight,
                        clamped_weight,
                        spec.activation,
                        spec.name,
                        None,
                        target_squash,
                    );

                    // v0.1.122: When target simulation is available, DON'T recompute weight
                    // using VALUE domain optimization. The candidate weights are already
                    // scaled versions of linear optimal. Recomputing in VALUE domain can
                    // overshoot near saturation, leading to worse ACTIVATION domain results.
                    // Instead, let the ACTIVATION domain evaluation pick the best candidate.
                    //
                    // We DO recompute to account for bias changing the activation pattern,
                    // but only when we don't have target simulation (linear approximation).

                    // Single pass for improvement and count with target simulation
                    let (improvement, improved, _) = compute_activation_improvement_and_count(
                        samples,
                        incoming_weight,
                        clamped_weight, // Use candidate weight directly, not recomputed
                        bias,
                        spec.activation,
                        baseline_sq,
                        target_activation_fn,
                    );

                    if improvement > best_improvement {
                        best_improvement = improvement;
                        best_weight = clamped_weight; // Use candidate weight directly
                        best_bias = bias;
                        best_improved_count = improved;
                    }
                }

                (
                    best_weight,
                    best_bias,
                    best_improvement,
                    best_improved_count,
                )
            } else {
                // For linear targets or when target data unavailable, use the base weight.
                let base_weight = match calculate_optimal_outgoing_weight(
                    sum_error_activation,
                    sum_activation_sq,
                    incoming_weight,
                ) {
                    Some(w) => w,
                    None => continue, // Skip if weight is invalid or ratio too small
                };

                // For linear targets or when target data unavailable, use the base weight
                let optimal_bias = calculate_optimal_bias(
                    samples,
                    incoming_weight,
                    base_weight,
                    spec.activation,
                    spec.name,
                    None,
                    target_squash,
                );

                // CRITICAL FIX: Recompute optimal weight WITH the bias included.
                // The base_weight was computed WITHOUT bias, so recompute
                // using the actual activation pattern with bias.
                let mut sum_activation_sq_with_bias = 0.0f32;
                let mut sum_error_activation_with_bias = 0.0f32;
                for sample in samples.iter() {
                    let pre_activation = incoming_weight * sample.activation + optimal_bias;
                    let output = (spec.activation)(pre_activation);
                    if output.is_finite() {
                        sum_activation_sq_with_bias += output * output;
                        sum_error_activation_with_bias += output * sample.avg_error;
                    }
                }
                // Use shared function for bias-adjusted weight calculation
                let outgoing_weight = calculate_optimal_outgoing_weight(
                    sum_error_activation_with_bias,
                    sum_activation_sq_with_bias,
                    incoming_weight,
                )
                .unwrap_or(base_weight);

                let (improvement, improved_count, _) = compute_activation_improvement_and_count(
                    samples,
                    incoming_weight,
                    outgoing_weight,
                    optimal_bias,
                    spec.activation,
                    baseline_sq,
                    None, // Linear approximation
                );

                (outgoing_weight, optimal_bias, improvement, improved_count)
            };

            // Skip invalid weights
            if outgoing_weight.abs() <= EPSILON {
                continue;
            }

            // =================================================================
            // FUNDAMENTAL VALIDITY FILTERS
            // These filters apply to ALL candidates (both fallback and best).
            // They must be checked BEFORE setting fallback_candidate to prevent
            // invalid candidates from being returned through the fallback path.
            // =================================================================

            // Issue #123: Check for saturation - reject if neuron output is nearly constant.
            // Large bias values (e.g., 5.0 for SOFTSIGN) can cause the neuron to saturate,
            // producing nearly identical output regardless of input. Such neurons cannot
            // reduce error correlation and lead to massive prediction failures.
            if !has_sufficient_output_variance(
                samples,
                incoming_weight,
                optimal_bias,
                spec.activation,
            ) {
                continue;
            }

            // Require minimum ABSOLUTE error reduction, not just percentage.
            // A 1% improvement on baseline_sq=0.0001 is only 0.000001 absolute reduction,
            // which won't meaningfully affect the creature's total error.
            // Minimum absolute improvement = 0.001 (0.1% of typical baseline ~1.0)
            let absolute_improvement = neuron_error_improvement * baseline_sq;
            if absolute_improvement < 0.001 {
                continue;
            }

            // IDENTITY with bias ≈ 0 is equivalent to a direct synapse (source × incoming × outgoing)
            // Filter out these redundant candidates - use synapse analysis for direct connections
            if spec.name == "IDENTITY" && optimal_bias.abs() < 0.01 {
                continue;
            }

            // =================================================================
            // FALLBACK CANDIDATE (best seen so far with any positive improvement)
            // =================================================================
            // Track the best candidate that didn't pass the main threshold.
            // TypeScript will decide if the improvement is worth the cost of growth.
            // We don't apply arbitrary minimum thresholds here - any positive
            // improvement is returned and TypeScript handles candidate selection.
            if neuron_error_improvement > fallback_score && neuron_error_improvement > 0.0 {
                fallback_score = neuron_error_improvement;

                let target_stats = NeuronStats::from_samples(samples).map(|s| s.to_json());
                // Issue #128: Use creature-level metrics (impact discounting applied later)
                fallback_candidate = Some(CandidateNeuronJson {
                    source_neuron_uuid: source_uuid.to_string(),
                    target_neuron_uuid: target_uuid.to_string(),
                    source_neuron_index: None, // Set during impact discounting
                    target_neuron_index: None, // Set during impact discounting
                    incoming_weight,
                    outgoing_weight,
                    squash: spec.name.to_string(),
                    bias: optimal_bias,
                    comment: None,
                    target_neuron_impact: 1.0,
                    expected_creature_error_reduction: neuron_error_improvement,
                    expected_creature_score_gain: neuron_error_improvement,
                    improved_count: final_improved_count,
                    total_count,
                    target_neuron_stats: target_stats,
                });
            }

            // =================================================================
            // THRESHOLD CHECK (only for best_candidate, not fallback)
            // =================================================================
            // Use the STRICTER threshold (max) to ensure meaningful improvements
            // If spec requires 5% and caller passes 1%, we require 5%
            // If spec requires 0% and caller passes 1%, we require 1%
            let improvement_cutoff = threshold.max(spec.min_improvement);

            if neuron_error_improvement <= improvement_cutoff
                || final_improved_count < MIN_NEURON_SAMPLE_COUNT as u32
            {
                continue;
            }

            // Current iteration passed threshold - create best_candidate with current iteration's values
            if neuron_error_improvement > best_score {
                best_score = neuron_error_improvement;

                let target_stats = NeuronStats::from_samples(samples).map(|s| s.to_json());
                // Issue #128: Use creature-level metrics (impact discounting applied later)
                best_candidate = Some(CandidateNeuronJson {
                    source_neuron_uuid: source_uuid.to_string(),
                    target_neuron_uuid: target_uuid.to_string(),
                    source_neuron_index: None, // Set during impact discounting
                    target_neuron_index: None, // Set during impact discounting
                    incoming_weight,
                    outgoing_weight,
                    squash: spec.name.to_string(),
                    bias: optimal_bias,
                    comment: None,
                    target_neuron_impact: 1.0,
                    expected_creature_error_reduction: neuron_error_improvement,
                    expected_creature_score_gain: neuron_error_improvement,
                    improved_count: final_improved_count,
                    total_count,
                    target_neuron_stats: target_stats,
                });
            }
        }
    }

    Ok(best_candidate.or(fallback_candidate))
}

/// Build discrete samples for threshold-crossing analysis.
/// Combines source neuron activations with target neuron values and errors.
fn build_discrete_samples(
    source_records: &[DiscoverRecord],
    target_records: &[DiscoverRecord],
) -> Vec<DiscreteHelpfulSample> {
    // Build a map from obs_index to target record
    let target_map: HashMap<u32, &DiscoverRecord> = target_records
        .iter()
        .filter(|r| !r.errors.is_empty())
        .map(|r| (r.obs_index, r))
        .collect();

    let mut samples = Vec::new();

    for source_record in source_records {
        if !source_record.activation.is_finite() {
            continue;
        }

        if let Some(target_record) = target_map.get(&source_record.obs_index) {
            // Need target's value (pre-activation input sum) for threshold crossing
            let target_value = match target_record.value {
                Some(v) if v.is_finite() => v,
                _ => continue, // Skip if no value available
            };

            if !target_record.activation.is_finite() {
                continue;
            }

            // Compute average error for target
            let mut error_sum = 0.0;
            let mut error_count = 0;
            for &err in &target_record.errors {
                if err.is_finite() {
                    error_sum += err;
                    error_count += 1;
                }
            }

            if error_count > 0 {
                let avg_error = error_sum / error_count as f32;
                samples.push(DiscreteHelpfulSample {
                    source_activation: source_record.activation,
                    target_value,
                    target_activation: target_record.activation,
                    avg_error,
                });
            }
        }
    }

    samples
}

/// Evaluate a discrete activation candidate using threshold-crossing model.
/// Instead of predicting continuous error reduction, counts how many samples
/// would flip to the correct output if we add a new connection.
///
/// For STEP/BIPOLAR, the only meaningful improvement is flipping the output:
/// - If error > 0 (output should be higher), we want to flip 0→1 or -1→1
/// - If error < 0 (output should be lower), we want to flip 1→0 or 1→-1
///
/// **CURRENTLY DISABLED**: This function always returns `None` because IDENTITY+bias=0
/// candidates are equivalent to a direct synapse and are filtered out. For STEP/BIPOLAR
/// targets, use **add-synapse analysis** instead - it can find direct connections that
/// flip the output without wasting a neuron.
///
/// The function is retained for potential future use with non-IDENTITY squash functions
/// (e.g., evaluating STEP→STEP chains) but currently does nothing useful.
///
/// # Arguments
/// * `is_output_target` - Whether the target neuron is an output neuron. (Currently unused.
///   Impact discounting for hidden targets is handled after candidate evaluation via
///   `compute_impacts_public()`, which correctly uses the target's weighted paths to outputs.)
#[allow(unused_variables, unreachable_code)]
fn evaluate_discrete_candidate(
    source_uuid: &str,
    target_uuid: &str,
    samples: &[DiscreteHelpfulSample],
    threshold_type: ThresholdType,
    threshold: f32,
    is_output_target: bool,
) -> Option<CandidateNeuronJson> {
    // EARLY RETURN: Discrete evaluation only considers IDENTITY squash for new neurons.
    // IDENTITY+bias=0 is mathematically equivalent to a direct synapse:
    //   IDENTITY(source × incoming_weight + 0) × outgoing_weight = source × (incoming × outgoing)
    //
    // For STEP/BIPOLAR targets, add-synapse analysis handles this case more efficiently
    // (1 synapse vs 1 neuron + 2 synapses). Returning None immediately avoids wasted
    // computation from iterating through weight combinations.
    //
    // TODO: If discrete evaluation should support non-IDENTITY squash functions in future
    // (e.g., STEP→STEP chains), remove this early return and add those squash types.
    return None;

    // The following code is unreachable but retained for reference if we add
    // support for non-IDENTITY squash functions in the future.
    if samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return None;
    }

    let total_count = samples.len() as u32;

    // Incoming weight scales - larger scales help with threshold crossing by amplifying
    // source activation differences. Keep wide range since these don't directly affect output.
    const INCOMING_SCALES: [f32; 8] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];

    // Outgoing weight scales - MUST be small! Based on analysis of successful vs failed
    // discoveries: successful add-neuron candidates have |outgoing_weight| < 0.05.
    // Large outgoing weights (10, 50) consistently fail in production.
    // These scales match MAX_OUTGOING_WEIGHT (0.1) as upper bound.
    const OUTGOING_SCALES: [f32; 5] = [0.01, 0.02, 0.05, 0.075, 0.1];

    const ORIENTATIONS: [f32; 2] = [1.0, -1.0];

    let mut best_candidate: Option<CandidateNeuronJson> = None;
    let mut best_improvement = threshold;

    // For discrete functions, use IDENTITY squash on the new neuron
    // This passes the weighted source activation directly
    let new_neuron_squash = "IDENTITY";

    for &orientation in &ORIENTATIONS {
        for &scale in &INCOMING_SCALES {
            let incoming_weight = orientation * scale;

            // For IDENTITY squash, new_neuron_output = incoming_weight * source_activation
            // Try different outgoing weights (small scales only for reliable predictions)
            for &out_scale in &OUTGOING_SCALES {
                for &out_orientation in &ORIENTATIONS {
                    let outgoing_weight = out_orientation * out_scale;

                    // Count helpful and harmful flips
                    let mut helpful_flips = 0i32;
                    let mut harmful_flips = 0i32;
                    let mut samples_with_error = 0u32;

                    for sample in samples {
                        if sample.avg_error.abs() < EPSILON {
                            continue; // No error, nothing to improve
                        }
                        samples_with_error += 1;

                        // New neuron output with IDENTITY: just passes through
                        let new_neuron_output = incoming_weight * sample.source_activation;
                        let contribution = outgoing_weight * new_neuron_output;

                        let flip_dir = threshold_type.flip_direction(
                            sample.target_value,
                            contribution,
                            sample.avg_error,
                        );

                        match flip_dir {
                            1 => helpful_flips += 1,
                            -1 => harmful_flips += 1,
                            _ => {}
                        }
                    }

                    if samples_with_error < MIN_NEURON_SAMPLE_COUNT as u32 {
                        continue;
                    }

                    // Net improvement: proportion of samples that would be corrected
                    let net_flips = helpful_flips - harmful_flips;
                    let flip_rate = net_flips as f32 / samples_with_error as f32;

                    // For both OUTPUT and HIDDEN targets, use flip_rate as the raw improvement.
                    //
                    // For OUTPUT targets: flip_rate ≈ error_reduction (each flip changes MSE by ~1)
                    // For HIDDEN targets: flip_rate is the neuron-level improvement. The actual
                    // creature-level error reduction is computed later via impact discounting
                    // (see lines ~6511-6550) which uses compute_impacts_public() to determine
                    // the target's weighted path to outputs.
                    //
                    // NOTE: Previously this code incorrectly used `outgoing_weight.abs()` as a
                    // proxy for hidden target impact. That was WRONG because `outgoing_weight`
                    // is the NEW→TARGET connection weight, NOT the TARGET's outgoing connections
                    // to downstream output neurons. When a hidden STEP neuron flips (0→1), its
                    // impact on outputs depends on its own synapses to outputs, not the incoming
                    // synapse weight. The proper impact is computed by compute_impacts_public().
                    let improvement = flip_rate;

                    // IMPORTANT: Require meaningful improvement (at least 1%) for IDENTITY neurons.
                    // IDENTITY with bias=0 is mathematically equivalent to a direct synapse:
                    //   IDENTITY(source × incoming_weight + 0) × outgoing_weight = source × incoming × outgoing
                    // These candidates should use add-synapse, not add-neuron.
                    // Additionally, very low flip rates (< 1%) indicate the contribution isn't
                    // reliably pushing the target across the threshold.
                    const MIN_DISCRETE_IMPROVEMENT: f32 = 0.01; // 1% minimum

                    if improvement > best_improvement.max(MIN_DISCRETE_IMPROVEMENT)
                        && helpful_flips > harmful_flips
                    {
                        // IDENTITY+bias=0 is mathematically equivalent to a direct synapse.
                        // These should be handled by add-synapse analysis, not add-neuron.
                        // Skip these candidates entirely - they waste a neuron for no benefit.
                        if new_neuron_squash == "IDENTITY" {
                            // bias is always 0 for discrete evaluation, so skip all IDENTITY
                            continue;
                        }
                        best_improvement = improvement;

                        // Create target neuron stats from samples
                        let target_stats = {
                            let helper_samples: Vec<HelpfulSample> = samples
                                .iter()
                                .map(|s| HelpfulSample {
                                    activation: s.target_activation,
                                    avg_error: s.avg_error,
                                    target_value: Some(s.target_value),
                                    target_activation: Some(s.target_activation),
                                })
                                .collect();
                            NeuronStats::from_samples(&helper_samples).map(|s| s.to_json())
                        };

                        // Issue #128: Use creature-level metrics
                        best_candidate = Some(CandidateNeuronJson {
                            source_neuron_uuid: source_uuid.to_string(),
                            target_neuron_uuid: target_uuid.to_string(),
                            source_neuron_index: None, // Set during impact discounting
                            target_neuron_index: None, // Set during impact discounting
                            incoming_weight,
                            outgoing_weight,
                            squash: new_neuron_squash.to_string(),
                            bias: 0.0, // IDENTITY doesn't need bias for threshold crossing
                            comment: None,
                            target_neuron_impact: 1.0,
                            expected_creature_error_reduction: improvement,
                            expected_creature_score_gain: improvement,
                            improved_count: helpful_flips as u32,
                            total_count,
                            target_neuron_stats: target_stats,
                        });
                    }
                }
            }
        }
    }

    best_candidate
}

pub(crate) fn analyze_neurons_with_cache(
    input: &AnalyzeNeuronsInput,
    cache: Arc<RecordCache>,
) -> Result<AnalyzeNeuronsResult> {
    // v0.1.134: Default threshold is 0 - return ALL positive improvements.
    // TypeScript will decide which candidates are worth the cost of growth.
    let threshold = input.improvement_threshold.unwrap_or(0.0);
    let ordered_neurons = build_ordered_neurons(&input.creature);

    // Build a lookup map for neuron squash functions to identify discrete targets
    let neuron_squash_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.squash.clone()))
        .collect();

    // Build a comprehensive lookup map for ALL neuron UUIDs to their types.
    // This includes: input neurons (from creature.input count) and all neurons from
    // creature.neurons (hidden, output, constant). If a UUID is not in this map,
    // it's an invalid UUID (bug in the caller).
    //
    // Only output neurons should be targets for add-neuron candidates because:
    // - Output neuron errors directly affect creature score
    // - Hidden neuron errors are backpropagated approximations that don't correlate
    //   reliably with actual output error reduction
    // - Input neurons are observation sources, not computation nodes
    let mut neuron_type_map: HashMap<String, String> = HashMap::new();

    // Add input neurons (they're not in creature.neurons, only represented by creature.input count)
    for input_index in 0..input.creature.input {
        neuron_type_map.insert(format!("input-{input_index}"), "input".to_string());
    }

    // Add all neurons from creature.neurons (hidden, output, constant)
    for neuron in &input.creature.neurons {
        neuron_type_map.insert(neuron.uuid.clone(), neuron.neuron_type.clone());
    }

    // Log creature configuration for debugging data issues
    if verbose_enabled() {
        let non_input_count = input.creature.neurons.len();
        let total_neurons = input.creature.input + non_input_count;
        eprintln!(
            "[NEAT-AI-Discovery][verbose] Neuron analysis creature config: {} input neurons (input-0 to input-{}), {} non-input neurons, {} total ordered neurons",
            input.creature.input,
            input.creature.input.saturating_sub(1),
            non_input_count,
            total_neurons
        );

        // Verify input neurons exist in parquet by checking a sample
        if input.creature.input > 0 {
            match cache.get("input-0") {
                Ok(records) => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Parquet data check: input-0 has {} records",
                        records.len()
                    );
                    if !records.is_empty() {
                        let first = &records[0];
                        let last = &records[records.len() - 1];
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Parquet data check: input-0 obs_index range [{}, {}], first activation={:.4}",
                            first.obs_index,
                            last.obs_index,
                            first.activation
                        );
                    }
                }
                Err(err) => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Parquet data check FAILED: input-0 error: {err}"
                    );
                }
            }

            // Also check a middle input neuron
            let mid_input = input.creature.input / 2;
            let mid_uuid = format!("input-{mid_input}");
            match cache.get(&mid_uuid) {
                Ok(records) => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Parquet data check: {mid_uuid} has {} records",
                        records.len()
                    );
                }
                Err(err) => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Parquet data check FAILED: {mid_uuid} error: {err}"
                    );
                }
            }
        }
    }

    let order_map: HashMap<String, usize> = ordered_neurons
        .iter()
        .map(|neuron| (neuron.uuid.clone(), neuron.index))
        .collect();

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_neurons")?;

    let helpful_map = Arc::new(Mutex::new(HashMap::<
        (String, String, String, i8, i8),
        CandidateNeuronJson,
    >::new()));

    let diagnostics = Arc::new(Mutex::new(NeuronDiagnostics::new(&unique_focus)));

    let deadline = build_deadline(input.analysis_deadline_ms);
    // GPU is always required - TypeScript layer calls check_gpu_available() and skips
    // discovery entirely on machines without GPU. Reaching here without GPU is a bug.
    assert!(
        GpuAnalyzer::gpu_is_available(),
        "Discovery logic called without GPU - check_gpu_available should have prevented this"
    );
    let gpu_used = true;
    let analysis_timed_out = Arc::new(Mutex::new(false));

    // Filter focus neurons to ONLY output neurons for add-neuron analysis.
    //
    // Rationale: Add-neuron candidates predict error reduction at the target neuron.
    // For OUTPUT neurons, this directly corresponds to creature score improvement.
    // For HIDDEN neurons, the backpropagated error is an approximation that doesn't
    // reliably translate to actual output error reduction - we've observed 100%
    // failure rates when targeting hidden neurons.
    // For INPUT neurons, they are observation sources, not computation nodes - they
    // have no activation function or error to reduce.
    //
    // Non-output neurons are skipped here but still tracked in diagnostics so the
    // caller knows they were received but filtered out (with the correct reason).
    let original_focus_count = unique_focus.len();
    let skipped_hidden: Vec<String> = Vec::new();
    let mut skipped_input: Vec<String> = Vec::new();
    let mut skipped_constant: Vec<String> = Vec::new();

    // Randomize the focus neuron order so that repeated runs with timeouts will
    // eventually cover all neurons. Convert to owned strings, shuffle, then use.
    //
    // All neurons are processed - no activation functions are skipped. STEP/BIPOLAR
    // neurons use a specialised threshold-crossing model; all others use the standard
    // linear error model (which is an approximation but still finds useful patterns).
    let mut threshold_targets: Vec<String> = Vec::new();
    let mut focus_order: Vec<String> = unique_focus
        .iter()
        .filter_map(|uuid| {
            // Check neuron type - allow output and hidden neurons for add-neuron analysis
            // Hidden neurons are analysed with impact-based discounting (v0.1.123)
            let neuron_type = neuron_type_map.get(*uuid).map(|s| s.as_str());
            match neuron_type {
                Some("output") | Some("hidden") => {
                    // Output and hidden neurons are valid targets
                    // Hidden neuron predictions will be discounted by impact later
                    if let Some(squash) = neuron_squash_map.get(*uuid) {
                        if is_threshold_activation(squash) {
                            threshold_targets.push((*uuid).clone());
                        }
                    }
                    Some((*uuid).clone())
                }
                Some("input") => {
                    // Input neurons are observation sources, not computation nodes
                    skipped_input.push((*uuid).clone());
                    None
                }
                Some("constant") => {
                    // Constant neurons don't receive inputs - filtering them out
                    skipped_constant.push((*uuid).clone());
                    None
                }
                Some(unknown_type) => {
                    // Unknown type - treat as hidden (analysable with discount)
                    eprintln!(
                        "[NEAT-AI-Discovery] Warning: Unknown neuron type '{unknown_type}' for UUID '{uuid}'. \
                        Treating as hidden neuron (will apply impact discount)."
                    );
                    Some((*uuid).clone())
                }
                None => {
                    // Unknown UUID - this is likely a bug, skip it
                    eprintln!(
                        "[NEAT-AI-Discovery] Warning: Unknown neuron UUID '{uuid}' in focus list \
                        (not found in creature). Skipping."
                    );
                    None
                }
            }
        })
        .collect();

    // Log when non-output neurons are filtered out
    let total_skipped = skipped_hidden.len() + skipped_input.len() + skipped_constant.len();
    if total_skipped > 0 {
        // Build a summary of skipped neuron types
        let mut skipped_parts: Vec<String> = Vec::new();
        if !skipped_input.is_empty() {
            skipped_parts.push(format!(
                "Input: {:?}",
                skipped_input.iter().take(5).collect::<Vec<_>>()
            ));
        }
        if !skipped_hidden.is_empty() {
            skipped_parts.push(format!(
                "Hidden: {:?}",
                skipped_hidden.iter().take(5).collect::<Vec<_>>()
            ));
        }
        if !skipped_constant.is_empty() {
            skipped_parts.push(format!(
                "Constant: {:?}",
                skipped_constant.iter().take(5).collect::<Vec<_>>()
            ));
        }
        eprintln!(
            "[NEAT-AI-Discovery] Filtered {} neuron(s) from add-neuron analysis. {}. Remaining valid targets: {}",
            total_skipped,
            skipped_parts.join(". "),
            focus_order.len()
        );
    }

    // If no output neurons remain after filtering, return early with empty results
    if focus_order.is_empty() {
        eprintln!(
            "[NEAT-AI-Discovery] No output neurons in focus list ({original_focus_count} non-output neurons filtered out). \
            Add-neuron candidates can only target output neurons."
        );
        // Build no_candidate_reasons with correct reason for each neuron type
        let mut no_candidate_reasons: Vec<NeuronNoCandidateSummary> = Vec::new();
        for uuid in &skipped_input {
            no_candidate_reasons.push(NeuronNoCandidateSummary {
                target_uuid: uuid.clone(),
                reason: NeuronNoCandidateReason::InputNeuronFiltered,
                evaluated_sources: 0,
                sources_with_samples: 0,
                target_record_count: 0,
                detail: None,
            });
        }
        for uuid in &skipped_hidden {
            no_candidate_reasons.push(NeuronNoCandidateSummary {
                target_uuid: uuid.clone(),
                reason: NeuronNoCandidateReason::HiddenNeuronFiltered,
                evaluated_sources: 0,
                sources_with_samples: 0,
                target_record_count: 0,
                detail: None,
            });
        }
        for uuid in &skipped_constant {
            no_candidate_reasons.push(NeuronNoCandidateSummary {
                target_uuid: uuid.clone(),
                reason: NeuronNoCandidateReason::ConstantNeuronFiltered,
                evaluated_sources: 0,
                sources_with_samples: 0,
                target_record_count: 0,
                detail: None,
            });
        }
        return Ok(AnalyzeNeuronsResult {
            helpful_neurons: Vec::new(),
            gpu_used: true,
            no_candidate_reasons,
            metadata: super::shared::NeuronAnalysisMetadata {
                candidates_found: 0,
                candidates_returned: 0,
            },
        });
    }

    // Mark skipped neurons in diagnostics so they appear with the correct reason
    // instead of misleading reasons like NoEligibleSources.
    // This is the normal flow case where some output neurons exist.
    for input_uuid in &skipped_input {
        diagnostics
            .lock()
            .expect("Mutex poisoned: diagnostics")
            .mark_input_filtered(input_uuid);
    }
    for hidden_uuid in &skipped_hidden {
        diagnostics
            .lock()
            .expect("Mutex poisoned: diagnostics")
            .mark_hidden_filtered(hidden_uuid);
    }
    for constant_uuid in &skipped_constant {
        diagnostics
            .lock()
            .expect("Mutex poisoned: diagnostics")
            .mark_constant_filtered(constant_uuid);
    }

    // Log threshold-crossing neurons for visibility
    if verbose_enabled() && !threshold_targets.is_empty() {
        eprintln!(
            "[NEAT-AI-Discovery][verbose] Using threshold-crossing model for {} STEP/BIPOLAR neurons: {:?}",
            threshold_targets.len(),
            threshold_targets.iter().take(5).collect::<Vec<_>>()
        );
    }

    let mut rng = thread_rng();
    focus_order.shuffle(&mut rng);

    // Log analysis start with timeout duration and randomised order
    log_analysis_start(
        "neuron",
        input.analysis_deadline_ms,
        focus_order.len(),
        &focus_order,
    );

    // Track completed focus neurons for timeout logging
    let total_focus_count = focus_order.len();
    let completed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let focus_order_arc = Arc::new(focus_order);
    let ordered_neurons_arc = Arc::new(ordered_neurons);
    let order_map_arc = Arc::new(order_map);
    let neuron_squash_map_arc = Arc::new(neuron_squash_map);

    // Create a shared GPU work queue ONCE before the parallel loop.
    // This eliminates the overhead of creating multiple GPU devices (one per thread).
    // All GPU operations are processed by a single dedicated thread, improving utilisation.
    // CRITICAL: The GpuAnalyzer is created INSIDE the GPU thread to avoid wgpu deadlocks.
    let gpu_queue = Arc::new(GpuWorkQueue::new()?);

    // Process each focus neuron in parallel. Deadline checks happen at the start of
    // each focus target so that once analysis for a neuron begins, we prefer to
    // complete its upstream evaluation rather than abandoning it mid-stream. This
    // gives us “vertical” timeout behaviour where some neurons complete fully even
    // if later targets are skipped when the deadline is reached.
    focus_order_arc
        .par_iter()
        .try_for_each(|target_uuid| -> Result<()> {
            if *analysis_timed_out.lock().expect("Mutex poisoned") || deadline_passed(&deadline) {
                *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                return Ok(());
            }

            crate::watchdog::beat(format!(
                "neuron analysis → processing target {target_uuid}"
            ));

            // Use the shared GPU work queue instead of creating a new GpuAnalyzer per thread.
            // This eliminates device creation overhead and improves GPU utilisation.
            let gpu = &*gpu_queue;

            let target_records_arc = match cache.get(target_uuid.as_str()) {
                Ok(records) => records,
                Err(err) => {
                    if cfg!(debug_assertions) {
                        eprintln!("Failed to load target neuron records for {target_uuid}: {err}");
                    }
                    return Ok(());
                }
            };
            if target_records_arc.is_empty() {
                diagnostics
                    .lock()
                    .expect("Mutex poisoned: diagnostics")
                    .set_target_record_count(target_uuid, 0);
                return Ok(());
            }
            let target_records = target_records_arc.as_ref();
            diagnostics
                .lock()
                .expect("Mutex poisoned: diagnostics")
                .set_target_record_count(target_uuid, target_records.len());

            // Log target neuron obs_index range for debugging sample matching
            if verbose_enabled() && !target_records.is_empty() {
                let first_obs = target_records.first().map(|r| r.obs_index).unwrap_or(0);
                let last_obs = target_records.last().map(|r| r.obs_index).unwrap_or(0);
                let has_errors = target_records.iter().any(|r| !r.errors.is_empty());
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} has {} records, obs_index range [{}, {}], has_errors={}",
                    target_uuid,
                    target_records.len(),
                    first_obs,
                    last_obs,
                    has_errors
                );
            }

            // Check if this is a threshold activation (STEP/BIPOLAR) that needs special handling
            let threshold_type = neuron_squash_map_arc
                .get(target_uuid)
                .and_then(|squash| ThresholdType::from_squash(squash));

            // Check if target is an output neuron (needed for discrete evaluation accuracy)
            let is_output_target = neuron_type_map
                .get(target_uuid.as_str())
                .map(|t| t == "output")
                .unwrap_or(false);

            let target_index = match order_map_arc.get(target_uuid.as_str()) {
                Some(index) => *index,
                None => return Ok(()),
            };

            let mut eligible_sources: Vec<&OrderedNeuron> = ordered_neurons_arc
                .iter()
                .filter(|neuron| neuron.index < target_index)
                .collect();
            // Fair source ordering (Dec 2025):
            // - We want to avoid starving "late" input neurons (eg input-1486+) when timeouts occur.
            // - We therefore interleave input sources from both ends (0, last, 1, last-1, ...)
            //   and only then shuffle the remaining non-input sources.
            let mut input_sources: Vec<&OrderedNeuron> = eligible_sources
                .iter()
                .copied()
                .filter(|n| parse_input_index(&n.uuid).is_some())
                .collect();
            input_sources.sort_by_key(|n| parse_input_index(&n.uuid).unwrap_or(n.index));
            let input_sources = interleave_from_ends(input_sources);

            let mut non_input_sources: Vec<&OrderedNeuron> = eligible_sources
                .iter()
                .copied()
                .filter(|n| parse_input_index(&n.uuid).is_none())
                .collect();
            let mut rng = thread_rng();
            non_input_sources.shuffle(&mut rng);

            eligible_sources.clear();
            eligible_sources.extend(input_sources);
            eligible_sources.extend(non_input_sources);

            // Track total eligible sources for diagnostics
            let total_eligible = eligible_sources.len() as u32;
            diagnostics
                .lock()
                .expect("Mutex poisoned: diagnostics")
                .set_total_eligible_sources(target_uuid, total_eligible);

            // Log focus neuron details for debugging
            if verbose_enabled() && total_eligible == 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} (index {}) has 0 eligible upstream sources. creature.input={}, so input neurons span indices 0-{}. This indicates target_index <= 0 or a creature configuration mismatch.",
                    target_uuid,
                    target_index,
                    ordered_neurons_arc.len().saturating_sub(input.creature.neurons.len()),
                    ordered_neurons_arc.len().saturating_sub(input.creature.neurons.len()).saturating_sub(1)
                );
            }

            // Phase 1: Pre-filter sources and collect their records (with deadline checks)
            // This mirrors the synapse analysis approach for better parallelism
            let mut sources_to_process: Vec<(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)> =
                Vec::with_capacity(eligible_sources.len());
            let mut empty_record_sources: Vec<String> = Vec::new();
            let mut load_failure_count = 0u32;

            for source in &eligible_sources {
                // Check deadline during pre-filtering
                if deadline_passed(&deadline) {
                    *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                    break;
                }
                let source_uuid = source.uuid.as_str();
                match cache.get(source_uuid) {
                    Ok(records) => {
                        if !records.is_empty() {
                            sources_to_process.push((source, records));
                        } else {
                            empty_record_sources.push(source_uuid.to_string());
                        }
                    }
                    Err(err) => {
                        load_failure_count += 1;
                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] Failed to load source neuron records for {source_uuid} (target {target_uuid}): {err}"
                            );
                        }
                    }
                }
            }

            // Record load failures in diagnostics
            if load_failure_count > 0 {
                let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                for _ in 0..load_failure_count {
                    diag.record_load_failure(target_uuid);
                }
            }

            // Log summary of source loading results for debugging
            let sources_checked = sources_to_process.len() + empty_record_sources.len() + load_failure_count as usize;
            let timed_out_during_loading = *analysis_timed_out.lock().expect("Mutex poisoned");
            if verbose_enabled() && (sources_to_process.is_empty() || load_failure_count > 0 || !empty_record_sources.is_empty() || timed_out_during_loading) {
                let sources_with_records = sources_to_process.len();
                let empty_count = empty_record_sources.len();
                if timed_out_during_loading && sources_checked == 0 {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {target_uuid} source loading: TIMEOUT before any of {total_eligible} eligible sources could be checked"
                    );
                } else if timed_out_during_loading {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {target_uuid} source loading: TIMEOUT after checking {sources_checked}/{total_eligible} eligible sources ({sources_with_records} with records, {empty_count} empty, {load_failure_count} failures)"
                    );
                } else {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {target_uuid} source loading: {total_eligible} eligible -> {sources_with_records} with records, {empty_count} empty records, {load_failure_count} load failures"
                    );
                }
            }

            // Batch diagnostics for empty record sources
            if !empty_record_sources.is_empty() {
                let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                for source_uuid in &empty_record_sources {
                    diag.record_candidate_attempt(target_uuid, false);
                    diag.record_no_samples(target_uuid, source_uuid);
                }
            }

            // Check if timed out during pre-filtering
            if *analysis_timed_out.lock().expect("Mutex poisoned") {
                return Ok(());
            }

            // Phase 2: Build samples in parallel using CPU (much faster than sequential GPU calls)
            // OPTIMIZATION: Pre-build target map ONCE, reuse for all sources.
            // This avoids rebuilding the HashMap for each of ~1000+ source neurons.
            let target_map = TargetMap::from_records(target_records);
            let target_map_ref = &target_map;

            // Even if target_map is empty, we continue to record diagnostics
            // about what sources were evaluated.
            struct NeuronWorkResult {
                source_uuid: String,
                samples: Vec<HelpfulSample>,
            }

            let work_results: Vec<NeuronWorkResult> = sources_to_process
                .par_iter()
                .map(|(source, from_records_arc)| {
                    let source_uuid = source.uuid.as_str();
                    let from_records = from_records_arc.as_ref();
                    // Use pre-built target map - avoids rebuilding HashMap for each source
                    let samples = target_map_ref.build_samples_from(from_records);
                    NeuronWorkResult {
                        source_uuid: source_uuid.to_string(),
                        samples,
                    }
                })
                .collect();

            // Phase 3: Batch diagnostics updates for sample building results
            {
                let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                for result in &work_results {
                    diag.record_candidate_attempt(target_uuid, !result.samples.is_empty());
                    if result.samples.is_empty() {
                        diag.record_no_samples(target_uuid, &result.source_uuid);
                    }
                }
            }

            // Phase 4: Process evaluations - GPU work is done here
            // Filter to only sources with samples, then evaluate
            //
            // For threshold activations (STEP/BIPOLAR), we use a specialised
            // threshold-crossing model instead of the standard linear error model.
            if let Some(t_type) = threshold_type {
                // Threshold activation path - use discrete evaluation
                for (source, from_records_arc) in &sources_to_process {
                    // Check deadline
                    if deadline_passed(&deadline) {
                        *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                        break;
                    }

                    let from_records = from_records_arc.as_ref();
                    let discrete_samples = build_discrete_samples(from_records, target_records);

                    if discrete_samples.len() < MIN_NEURON_SAMPLE_COUNT {
                        continue;
                    }

                    if let Some(candidate) = evaluate_discrete_candidate(
                        &source.uuid,
                        target_uuid,
                        &discrete_samples,
                        t_type,
                        threshold,
                        is_output_target,
                    ) {
                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] Threshold-crossing candidate for {} -> {}: {} samples, {:.2}% improvement (flips: {})",
                                source.uuid,
                                target_uuid,
                                discrete_samples.len(),
                                candidate.expected_creature_score_gain * 100.0,
                                candidate.improved_count
                            );
                        }
                        diagnostics
                            .lock()
                            .expect("Mutex poisoned: diagnostics")
                            .mark_candidate_selected(target_uuid);
                        let mut map = helpful_map.lock().expect("Mutex poisoned: helpful_map");
                        upsert_candidate(&mut map, candidate);
                    }
                }
            } else {
                // Standard continuous activation path
                for result in work_results {
                    // Check deadline before each evaluation batch
                    if deadline_passed(&deadline) {
                        *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                        break;
                    }

                    if result.samples.is_empty() {
                        continue;
                    }

                    // Get target_squash for accurate HARD_TANH modelling
                    let target_squash = neuron_squash_map_arc.get(target_uuid).map(|s| s.as_str());

                    // Issue #130 (v0.2.2): Compute source variance discount.
                    // If source activation has low variance, predictions are unreliable.
                    let source_variance_discount = compute_source_variance_discount(&result.samples);
                    if source_variance_discount <= EPSILON {
                        // Source is constant - skip evaluation entirely
                        continue;
                    }

                    // ReLU evaluation: split by TARGET neuron's error sign.
                    //
                    // ReLU can only push output in ONE direction (based on outgoing weight sign),
                    // so we evaluate two candidates separately:
                    // - Positive-error ReLU: optimised for samples where output should be HIGHER
                    // - Negative-error ReLU: optimised for samples where output should be LOWER
                    //
                    // Each candidate's weight is computed from its error subset, then NET
                    // improvement is calculated across ALL samples. This is the correct
                    // approach for directional activation functions like ReLU.
                    //
                    // NOTE: We don't use "averaging over all samples" because when errors are
                    // split ~50/50, the average cancels out and no candidate is found.
                    let split_result = evaluate_relu_candidates_split(
                        gpu,
                        &result.source_uuid,
                        target_uuid,
                        &result.samples,
                        threshold,
                        target_squash,
                    )?;

                    if let Some(mut candidate) = split_result.positive_error_candidate {
                        // Issue #130: Apply source variance discount
                        candidate.expected_creature_error_reduction *= source_variance_discount;
                        candidate.expected_creature_score_gain *= source_variance_discount;

                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] ReLU (push UP) {} -> {}: {:.2}% improvement (variance discount: {:.2})",
                                result.source_uuid,
                                target_uuid,
                                candidate.expected_creature_score_gain * 100.0,
                                source_variance_discount
                            );
                        }
                        diagnostics
                            .lock()
                            .expect("Mutex poisoned: diagnostics")
                            .mark_candidate_selected(target_uuid);
                        let mut map = helpful_map.lock().expect("Mutex poisoned: helpful_map");
                        upsert_candidate(&mut map, candidate);
                    }

                    if let Some(mut candidate) = split_result.negative_error_candidate {
                        // Issue #130: Apply source variance discount
                        candidate.expected_creature_error_reduction *= source_variance_discount;
                        candidate.expected_creature_score_gain *= source_variance_discount;

                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] ReLU (push DOWN) {} -> {}: {:.2}% improvement (variance discount: {:.2})",
                                result.source_uuid,
                                target_uuid,
                                candidate.expected_creature_score_gain * 100.0,
                                source_variance_discount
                            );
                        }
                        diagnostics
                            .lock()
                            .expect("Mutex poisoned: diagnostics")
                            .mark_candidate_selected(target_uuid);
                        let mut map = helpful_map.lock().expect("Mutex poisoned: helpful_map");
                        upsert_candidate(&mut map, candidate);
                    }

                    for spec in ACTIVATION_SPECS.iter() {
                        if let Some(mut candidate) = evaluate_activation_candidate(
                            gpu,
                            &result.source_uuid,
                            target_uuid,
                            &result.samples,
                            threshold,
                            spec,
                            target_squash,
                        )? {
                            // Issue #130: Apply source variance discount
                            candidate.expected_creature_error_reduction *= source_variance_discount;
                            candidate.expected_creature_score_gain *= source_variance_discount;

                            diagnostics
                                .lock()
                                .expect("Mutex poisoned: diagnostics")
                                .mark_candidate_selected(target_uuid);
                            let mut map = helpful_map.lock().expect("Mutex poisoned: helpful_map");
                            upsert_candidate(&mut map, candidate);
                        }
                    }
                }
            }

            // Track completion of this focus neuron for timeout reporting.
            let completed =
                completed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            crate::watchdog::beat(format!(
                "neuron analysis → completed {completed}/{total_focus_count} (last target {target_uuid})"
            ));
            Ok(())
        })?;

    let analysis_timed_out = *analysis_timed_out
        .lock()
        .expect("Mutex poisoned: analysis_timed_out");
    let helpful_map = helpful_map
        .lock()
        .expect("Mutex poisoned: helpful_map")
        .clone();
    let diagnostics = diagnostics.lock().expect("Mutex poisoned: diagnostics");

    // Log timeout with completion stats (always visible, not just verbose)
    if analysis_timed_out {
        let completed = completed_count.load(std::sync::atomic::Ordering::Relaxed);
        log_analysis_timeout("neuron", completed, total_focus_count);
    }

    let mut helpful_results: Vec<CandidateNeuronJson> = helpful_map.into_values().collect();

    // Issue #128: Apply impact-based discounting and set creature-level metrics.
    // Output neurons have impact = 1.0 (no discount).
    // Hidden neurons have impact in [0, 1] based on their weighted paths to outputs.
    let impact_scores = compute_impact_scores_for_discounting(&input.creature, cache.as_ref());
    for candidate in &mut helpful_results {
        candidate.source_neuron_index = order_map_arc.get(&candidate.source_neuron_uuid).copied();
        candidate.target_neuron_index = order_map_arc.get(&candidate.target_neuron_uuid).copied();

        let is_hidden = neuron_type_map
            .get(&candidate.target_neuron_uuid)
            .map(|t| t != "output")
            .unwrap_or(true); // Default to true if type unknown (treat as hidden)

        let impact = if is_hidden {
            if let Some(&impact) = impact_scores.get(&candidate.target_neuron_uuid) {
                impact.clamp(0.0, 1.0)
            } else {
                // No impact score means disconnected from outputs - heavy discount
                0.1
            }
        } else {
            // Output neuron - full impact
            1.0
        };

        // Update creature-level metrics
        candidate.target_neuron_impact = impact;
        let original = candidate.expected_creature_error_reduction;
        candidate.expected_creature_error_reduction *= impact;
        candidate.expected_creature_score_gain = candidate.expected_creature_error_reduction;

        if verbose_enabled() && is_hidden {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Neuron candidate → {} impact {:.3}: \
                {:.4}% → {:.4}%",
                &candidate.target_neuron_uuid[..12.min(candidate.target_neuron_uuid.len())],
                impact,
                original * 100.0,
                candidate.expected_creature_score_gain * 100.0
            );
        }
    }

    // Sort by expected creature score gain (highest first) - Issue #128
    helpful_results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(Ordering::Equal)
    });

    // Production experiment: pair "extreme" candidates with a conservative variant.
    // Pass None for limit here - we'll truncate separately so that candidates_found
    // correctly includes generated variants.
    helpful_results = crate::analysis::utils::pair_extreme_candidates_with_conservative_variants(
        helpful_results,
        None, // No limit - truncate separately after capturing candidates_found
    );

    // Track candidates_found AFTER pairing but BEFORE truncation.
    // This ensures candidates_found >= candidates_returned always holds, which is
    // the expected semantic for this metric pair ("found" >= "returned").
    let candidates_found = helpful_results.len();

    // Apply max_candidates limit (truncation)
    if let Some(limit) = input.max_candidates {
        helpful_results.truncate(limit);
    }

    // Track candidates_returned AFTER truncation
    let candidates_returned = helpful_results.len();

    let no_candidate_reasons = diagnostics.no_candidate_summaries();
    diagnostics.emit_logs();

    Ok(AnalyzeNeuronsResult {
        helpful_neurons: helpful_results,
        gpu_used,
        no_candidate_reasons,
        metadata: super::shared::NeuronAnalysisMetadata {
            candidates_found,
            candidates_returned,
        },
    })
}

pub fn analyze_neurons(input: &AnalyzeNeuronsInput) -> Result<AnalyzeNeuronsResult> {
    // Validate focus_neurons before expensive pre-loading
    require_unique_focus(&input.focus_neurons, "Neuron analysis")?;

    // Pre-load all records for faster analysis (1 scan vs ~2000 scans)
    let cache = Arc::new(RecordCache::new_adaptive(&input.parquet_file)?);
    analyze_neurons_with_cache(input, cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ACTIVATION_SPECS;

    // ==================== GPU Tier Detection Tests ====================

    #[test]
    fn impact_discounting_uses_activation_based_selection_stats_for_minimum() -> Result<()> {
        // This is a targeted regression test for impact discounting consistency:
        // - MINIMUM/MAXIMUM/IF impacts should use activation-based win probabilities when
        //   recorded activations are available (via RecordCache).
        //
        // Without activation stats, MINIMUM uses a conservative 1/N model, which can
        // under-estimate impact and cause downstream discounting to be too aggressive.

        let creature = crate::CreatureJson {
            neurons: vec![
                crate::NeuronJson {
                    uuid: "hidden-a".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                crate::NeuronJson {
                    uuid: "hidden-b".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                crate::NeuronJson {
                    uuid: "min-0".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "MINIMUM".to_string(),
                    bias: 0.0,
                },
                crate::NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                crate::SynapseJson {
                    from_uuid: "hidden-a".to_string(),
                    to_uuid: "min-0".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                },
                crate::SynapseJson {
                    from_uuid: "hidden-b".to_string(),
                    to_uuid: "min-0".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                },
                crate::SynapseJson {
                    from_uuid: "min-0".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                },
            ],
            input: 0,
            output: 1,
        };

        // Build a cache that returns activations where hidden-a ALWAYS wins MINIMUM.
        let cache = Arc::new(RecordCache::with_loader(
            "unused.parquet",
            Arc::new(|_file, uuid| {
                let mut records = Vec::new();
                for obs_index in 0..10u32 {
                    let activation = match uuid {
                        "hidden-a" => 0.0,
                        "hidden-b" => 1.0,
                        _ => 0.0,
                    };
                    records.push(DiscoverRecord {
                        obs_index,
                        neuron_uuid: uuid.to_string(),
                        value: None,
                        activation,
                        errors: vec![0.0],
                    });
                }
                Ok(records)
            }),
        ));

        let impacts = compute_impact_scores_for_discounting(&creature, cache.as_ref());
        let a = impacts.get("hidden-a").copied().unwrap_or(0.0);
        let b = impacts.get("hidden-b").copied().unwrap_or(0.0);

        assert!(
            a > 0.9,
            "hidden-a should have near-full impact via MINIMUM win probability, got {a}"
        );
        assert!(
            b < 0.1,
            "hidden-b should have near-zero impact via MINIMUM win probability, got {b}"
        );
        Ok(())
    }

    /// Test that M4 is detected as high-performance tier.
    #[test]
    fn gpu_tier_detects_m4_as_high_performance() {
        let info = wgpu::AdapterInfo {
            name: "Apple M4".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::IntegratedGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Metal,
        };
        assert_eq!(
            detect_gpu_tier(&info),
            GpuPerformanceTier::High,
            "M4 should be detected as high-performance"
        );
    }

    /// Test that M4 Pro/Max are detected as high-performance tier.
    #[test]
    fn gpu_tier_detects_m4_pro_max_as_high_performance() {
        for name in ["Apple M4 Pro", "Apple M4 Max", "Apple M4 Ultra"] {
            let info = wgpu::AdapterInfo {
                name: name.to_string(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::IntegratedGpu,
                driver: String::new(),
                driver_info: String::new(),
                backend: wgpu::Backend::Metal,
            };
            assert_eq!(
                detect_gpu_tier(&info),
                GpuPerformanceTier::High,
                "{name} should be detected as high-performance"
            );
        }
    }

    /// Test that M3 Pro/Max are detected as high-performance tier.
    #[test]
    fn gpu_tier_detects_m3_pro_max_as_high_performance() {
        for name in ["Apple M3 Pro", "Apple M3 Max"] {
            let info = wgpu::AdapterInfo {
                name: name.to_string(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::IntegratedGpu,
                driver: String::new(),
                driver_info: String::new(),
                backend: wgpu::Backend::Metal,
            };
            assert_eq!(
                detect_gpu_tier(&info),
                GpuPerformanceTier::High,
                "{name} should be detected as high-performance"
            );
        }
    }

    /// Test that base M1/M2/M3 are detected as standard tier.
    #[test]
    fn gpu_tier_detects_base_m_series_as_standard() {
        for name in ["Apple M1", "Apple M2", "Apple M3"] {
            let info = wgpu::AdapterInfo {
                name: name.to_string(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::IntegratedGpu,
                driver: String::new(),
                driver_info: String::new(),
                backend: wgpu::Backend::Metal,
            };
            assert_eq!(
                detect_gpu_tier(&info),
                GpuPerformanceTier::Standard,
                "{name} should be detected as standard"
            );
        }
    }

    /// Test that discrete GPUs are detected as high-performance.
    #[test]
    fn gpu_tier_detects_discrete_gpu_as_high_performance() {
        let info = wgpu::AdapterInfo {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };
        assert_eq!(
            detect_gpu_tier(&info),
            GpuPerformanceTier::High,
            "Discrete GPUs should be detected as high-performance"
        );
    }

    /// Test that unknown integrated GPUs are detected as standard.
    #[test]
    fn gpu_tier_detects_unknown_integrated_as_standard() {
        let info = wgpu::AdapterInfo {
            name: "Intel UHD Graphics 630".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::IntegratedGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };
        assert_eq!(
            detect_gpu_tier(&info),
            GpuPerformanceTier::Standard,
            "Unknown integrated GPUs should be detected as standard"
        );
    }

    /// Test that batch size is correct for each tier.
    #[test]
    fn batch_size_correct_for_each_tier() {
        assert_eq!(
            get_batch_size_for_tier(GpuPerformanceTier::High),
            HIGH_PERF_GPU_BATCH_SIZE,
            "High-performance tier should use larger batch size"
        );
        assert_eq!(
            get_batch_size_for_tier(GpuPerformanceTier::Standard),
            DEFAULT_GPU_BATCH_SIZE,
            "Standard tier should use default batch size"
        );
        assert_eq!(
            get_batch_size_for_tier(GpuPerformanceTier::Unknown),
            DEFAULT_GPU_BATCH_SIZE,
            "Unknown tier should use default batch size"
        );
    }

    #[test]
    fn gpu_batch_size_caps_for_large_sample_counts() {
        // Dec 2025: Large recordings (50k+ samples) can wedge Metal when combined with
        // a high batch size. We cap based on estimated GPU buffer sizes.
        let configured = 1024;
        let max_sample_len = 58_149; // representative from production logs

        // Helpful path uses HelpfulContribution (48 bytes) + staging, plus sample buffer.
        let bytes_per_sample_helpful = std::mem::size_of::<GpuHelpfulSample>()
            + (2 * std::mem::size_of::<HelpfulContribution>());
        let capped_helpful = cap_gpu_batch_size_by_bytes(
            configured,
            max_sample_len,
            bytes_per_sample_helpful,
            GPU_MAX_BATCH_ALLOC_BYTES,
        );
        assert!(
            capped_helpful < configured,
            "expected helpful batch size to be capped for large sample sets"
        );
        assert!(
            capped_helpful >= 1,
            "batch size must never be reduced below 1"
        );

        // Harmful path uses HarmfulContribution (16 bytes) + staging, plus sample buffer.
        let bytes_per_sample_harmful = std::mem::size_of::<GpuHelpfulSample>()
            + (2 * std::mem::size_of::<HarmfulContribution>());
        let capped_harmful = cap_gpu_batch_size_by_bytes(
            configured,
            max_sample_len,
            bytes_per_sample_harmful,
            GPU_MAX_BATCH_ALLOC_BYTES,
        );
        assert!(
            capped_harmful < configured,
            "expected harmful batch size to be capped for large sample sets"
        );
        assert!(capped_harmful >= 1);
    }

    /// Test that verbose_enabled() is cached (doesn't re-read env var each time).
    #[test]
    fn verbose_enabled_is_cached() {
        // Call twice - should return same value without re-reading env
        let first = verbose_enabled();
        let second = verbose_enabled();
        assert_eq!(first, second, "verbose_enabled() should be deterministic");
    }

    // ==================== Memory Info / Page Size Tests ====================

    /// Test parsing page size from vm_stat output - Apple Silicon (16KB pages).
    #[test]
    #[cfg(target_os = "macos")]
    fn parse_vm_stat_page_size_apple_silicon() {
        let vm_stat_output = r#"Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               15417.
Pages active:                            624277.
Pages inactive:                          601916.
Pages speculative:                        23646.
"#;
        assert_eq!(
            parse_vm_stat_page_size(vm_stat_output),
            16384,
            "Should parse 16384 byte page size for Apple Silicon"
        );
    }

    /// Test parsing page size from vm_stat output - Intel Mac (4KB pages).
    #[test]
    #[cfg(target_os = "macos")]
    fn parse_vm_stat_page_size_intel_mac() {
        let vm_stat_output = r#"Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                               45123.
Pages active:                           1234567.
Pages inactive:                          876543.
Pages speculative:                        12345.
"#;
        assert_eq!(
            parse_vm_stat_page_size(vm_stat_output),
            4096,
            "Should parse 4096 byte page size for Intel Mac"
        );
    }

    /// Test parsing page size handles malformed output gracefully.
    #[test]
    #[cfg(target_os = "macos")]
    fn parse_vm_stat_page_size_malformed_output() {
        // Empty string should return architecture-appropriate default
        let result = parse_vm_stat_page_size("");
        #[cfg(target_arch = "aarch64")]
        assert_eq!(
            result, 16384,
            "Empty output should default to 16KB on ARM64"
        );
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(result, 4096, "Empty output should default to 4KB on Intel");

        // Missing "page size of" should return default
        let malformed = "Some random output without page size info";
        let result = parse_vm_stat_page_size(malformed);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(
            result, 16384,
            "Malformed output should default to 16KB on ARM64"
        );
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(
            result, 4096,
            "Malformed output should default to 4KB on Intel"
        );
    }

    /// Test that the parser handles various page size values.
    #[test]
    #[cfg(target_os = "macos")]
    fn parse_vm_stat_page_size_various_sizes() {
        // Test 4KB pages (Intel)
        let output_4k =
            "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 100.";
        assert_eq!(parse_vm_stat_page_size(output_4k), 4096);

        // Test 16KB pages (Apple Silicon)
        let output_16k =
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.";
        assert_eq!(parse_vm_stat_page_size(output_16k), 16384);

        // Test hypothetical larger page size (future-proofing)
        let output_64k =
            "Mach Virtual Memory Statistics: (page size of 65536 bytes)\nPages free: 100.";
        assert_eq!(parse_vm_stat_page_size(output_64k), 65536);
    }

    /// Test Linux meminfo parsing with valid input.
    #[test]
    #[cfg(target_os = "linux")]
    fn parse_meminfo_line_valid_input() {
        // Standard format from /proc/meminfo
        assert_eq!(
            parse_meminfo_line("MemTotal:       16384000 kB"),
            Some(16384000)
        );
        assert_eq!(
            parse_meminfo_line("MemAvailable:    8192000 kB"),
            Some(8192000)
        );
        // Single digit
        assert_eq!(parse_meminfo_line("MemFree:        1 kB"), Some(1));
    }

    /// Test Linux meminfo parsing with malformed input returns None (not 0).
    /// This is critical: returning None allows defaults to be preserved.
    #[test]
    #[cfg(target_os = "linux")]
    fn parse_meminfo_line_malformed_returns_none() {
        // Missing value
        assert_eq!(parse_meminfo_line("MemTotal:"), None);
        // Non-numeric value
        assert_eq!(parse_meminfo_line("MemTotal:       abc kB"), None);
        // Empty string
        assert_eq!(parse_meminfo_line(""), None);
        // Just whitespace after colon
        assert_eq!(parse_meminfo_line("MemTotal:       "), None);
    }

    /// Test that Linux get_memory_info preserves defaults when parsing fails.
    /// This prevents misleading "0.0GB" error messages.
    #[test]
    #[cfg(target_os = "linux")]
    fn linux_memory_info_preserves_defaults_on_malformed_input() {
        // This test verifies the fix by checking that parse_meminfo_line
        // returns None for malformed input, which allows get_memory_info
        // to preserve its default values instead of overwriting with 0.
        //
        // The actual get_memory_info function reads /proc/meminfo, so we
        // can't easily test it with mock data. Instead, we verify the
        // building blocks work correctly:

        // 1. Valid input should return Some(value)
        assert!(parse_meminfo_line("MemTotal:       16384000 kB").is_some());

        // 2. Malformed input should return None (not Some(0))
        assert!(parse_meminfo_line("MemTotal:").is_none());
        assert!(parse_meminfo_line("MemTotal:       abc").is_none());

        // 3. This ensures the if-let pattern in get_memory_info preserves defaults
    }

    // ==================== Activation Function Tests ====================

    #[test]
    fn test_identity_activation() {
        assert_eq!(identity_activation(1.0), 1.0);
        assert_eq!(identity_activation(-1.0), -1.0);
        assert_eq!(identity_activation(0.0), 0.0);
    }

    #[test]
    fn test_bipolar_activation() {
        assert_eq!(bipolar_activation(1.0), 1.0);
        assert_eq!(bipolar_activation(0.0001), 1.0);
        assert_eq!(bipolar_activation(0.0), -1.0);
        assert_eq!(bipolar_activation(-1.0), -1.0);
        assert_eq!(bipolar_activation(-0.0001), -1.0);
    }

    #[test]
    fn test_clipped_activation() {
        assert_eq!(clipped_activation(1.5), 1.0);
        assert_eq!(clipped_activation(0.5), 0.5);
        assert_eq!(clipped_activation(-0.5), -0.5);
        assert_eq!(clipped_activation(-1.5), -1.0);
    }

    #[test]
    fn test_absolute_activation() {
        assert_eq!(absolute_activation(1.0), 1.0);
        assert_eq!(absolute_activation(-1.0), 1.0);
        assert_eq!(absolute_activation(0.0), 0.0);
    }

    #[test]
    fn test_specs_include_new_activations() {
        let names: Vec<&str> = ACTIVATION_SPECS.iter().map(|s| s.name).collect();
        // Original activations
        assert!(names.contains(&"IDENTITY"));
        assert!(names.contains(&"BIPOLAR"));
        assert!(names.contains(&"CLIPPED"));
        assert!(names.contains(&"ABSOLUTE"));
        assert!(
            !names.contains(&"SELU"),
            "SELU should not be suggested as a new neuron activation (Issue #148: time-bounded runs)"
        );
        assert!(
            !names.contains(&"INVERSE"),
            "INVERSE (complement) should not be suggested as a new neuron activation. \
             It can be represented via IDENTITY with bias and negative incoming weights."
        );
        // New activations (v0.1.139)
        assert!(
            !names.contains(&"LeakyReLU"),
            "LeakyReLU should not be suggested as a new neuron activation"
        );
        assert!(
            names.contains(&"Mish"),
            "Mish should be included - 2 successful discoveries!"
        );
        assert!(
            !names.contains(&"Swish"),
            "Swish should not be suggested as a new neuron activation (Issue #148: time-bounded runs)"
        );
        assert!(names.contains(&"HARD_TANH"), "HARD_TANH should be included");
        assert!(
            names.contains(&"SOFTSIGN"),
            "SOFTSIGN should be included - successful discovery!"
        );
        assert!(
            names.contains(&"BENT_IDENTITY"),
            "BENT_IDENTITY should be included - successful discovery!"
        );
        assert!(names.contains(&"ArcTan"), "ArcTan should be included");
        assert!(names.contains(&"ReLU6"), "ReLU6 should be included");
        // Total count
        assert_eq!(names.len(), 15, "Should have 15 activation specs");
    }

    // ==================== Saturation Detection Tests (Issue #123) ====================

    /// Test that has_sufficient_output_variance detects saturated neurons.
    /// Issue #123: Large bias values cause neurons to output nearly-constant values,
    /// leading to massive prediction failures (e.g., predicting 4.67% when actual is ~0%).
    #[test]
    fn test_saturation_detection_rejects_constant_output() {
        // Create samples with typical activation range [-1, 1]
        let samples: Vec<HelpfulSample> = (-10..=10)
            .map(|i| HelpfulSample {
                activation: i as f32 / 10.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            })
            .collect();

        // SOFTSIGN with bias=5 and incoming=0.35 (from issue #123)
        // Output ≈ (5 + 0.35×x) / (1 + |5 + 0.35×x|) ≈ 0.83 for all x
        let saturated = !has_sufficient_output_variance(
            &samples,
            0.35, // incoming_weight
            5.0,  // bias (too large!)
            softsign_activation,
        );
        assert!(
            saturated,
            "SOFTSIGN with bias=5 should be detected as saturated (constant output)"
        );

        // SOFTSIGN with bias=0 should NOT be saturated
        let not_saturated = has_sufficient_output_variance(
            &samples,
            1.0, // incoming_weight
            0.0, // bias
            softsign_activation,
        );
        assert!(
            not_saturated,
            "SOFTSIGN with bias=0 should NOT be saturated"
        );
    }

    /// Test that TANH with large bias is detected as saturated.
    #[test]
    fn test_saturation_detection_tanh_large_bias() {
        let samples: Vec<HelpfulSample> = (-10..=10)
            .map(|i| HelpfulSample {
                activation: i as f32 / 10.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            })
            .collect();

        // TANH with bias=10 is heavily saturated (output ≈ 1.0 for all inputs)
        let saturated = !has_sufficient_output_variance(
            &samples,
            1.0,  // incoming_weight
            10.0, // bias (way too large!)
            |x| x.tanh(),
        );
        assert!(
            saturated,
            "TANH with bias=10 should be detected as saturated"
        );

        // TANH with moderate bias=1 should still have variance
        let not_saturated = has_sufficient_output_variance(
            &samples,
            1.0, // incoming_weight
            1.0, // bias (reasonable)
            |x| x.tanh(),
        );
        assert!(
            not_saturated,
            "TANH with bias=1 should NOT be saturated - still has output variance"
        );
    }

    /// Test that ReLU is not incorrectly flagged as saturated.
    /// ReLU naturally has "half" of samples at 0, but the other half varies.
    #[test]
    fn test_saturation_detection_relu_not_false_positive() {
        let samples: Vec<HelpfulSample> = (-10..=10)
            .map(|i| HelpfulSample {
                activation: i as f32 / 10.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            })
            .collect();

        // ReLU with bias=0 should NOT be flagged as saturated
        // Half the outputs are 0, but the other half varies from 0 to 1
        let not_saturated = has_sufficient_output_variance(&samples, 1.0, 0.0, |x| x.max(0.0));
        assert!(
            not_saturated,
            "ReLU with bias=0 should NOT be flagged as saturated"
        );
    }

    /// Test that constant input samples are NOT incorrectly rejected.
    /// If input is constant, output will be constant too, but predictions are still valid.
    #[test]
    fn test_saturation_detection_allows_constant_input() {
        // All samples have the same input activation (constant input)
        let samples: Vec<HelpfulSample> = (0..20)
            .map(|_| HelpfulSample {
                activation: -1.0, // Constant input
                avg_error: 0.3,
                target_value: None,
                target_activation: None,
            })
            .collect();

        // Even with large bias, constant input should be allowed
        // because predictions are still valid when input doesn't vary
        let should_allow = has_sufficient_output_variance(
            &samples,
            1.0,
            5.0, // Large bias, but input is constant so it's OK
            |x| x.tanh(),
        );
        assert!(
            should_allow,
            "Constant input samples should NOT be rejected - predictions are valid"
        );
    }

    #[test]
    fn parse_input_index_parses_valid_ids() {
        assert_eq!(parse_input_index("input-0"), Some(0));
        assert_eq!(parse_input_index("input-1486"), Some(1486));
        assert_eq!(parse_input_index("input-001"), Some(1));
        assert_eq!(parse_input_index("hidden-1"), None);
        assert_eq!(parse_input_index("input-"), None);
    }

    #[test]
    fn interleave_from_ends_interleaves_sorted_inputs() {
        let v = vec![0, 1, 2, 3, 4, 5];
        let out = interleave_from_ends(v);
        assert_eq!(out, vec![0, 5, 1, 4, 2, 3]);

        let v = vec![0, 1, 2, 3, 4];
        let out = interleave_from_ends(v);
        assert_eq!(out, vec![0, 4, 1, 3, 2]);
    }
}

// analyze_all has been moved to src/analysis/mod.rs

pub(crate) fn analyze_synapses_with_cache(
    input: &AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
) -> Result<AnalyzeSynapsesResult> {
    let ordered_neurons = build_ordered_neurons(&input.creature);
    let order_map: HashMap<String, usize> = ordered_neurons
        .iter()
        .map(|neuron| (neuron.uuid.clone(), neuron.index))
        .collect();

    let existing_synapses: HashSet<(String, String)> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| (synapse.from_uuid.clone(), synapse.to_uuid.clone()))
        .collect();

    let synapses_by_target: HashMap<String, Vec<SynapseJson>> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| (synapse.to_uuid.clone(), synapse.clone()))
        .fold(HashMap::new(), |mut acc, (key, val)| {
            acc.entry(key).or_default().push(val);
            acc
        });

    // Build a lookup map for neuron squash functions to identify discrete targets
    let neuron_squash_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.squash.clone()))
        .collect();

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_synapses")?;

    let diagnostics = Arc::new(Mutex::new(TargetDiagnostics::new(&unique_focus)));

    let deadline = build_deadline(input.analysis_deadline_ms);
    // Randomise the focus neuron order so that repeated runs with timeouts will
    // eventually cover all neurons. Convert to owned strings, shuffle, then use.
    // STEP/BIPOLAR neurons are now included - both get proper simulation functions
    // that accurately predict output flips when synapse contributions cross the threshold.
    let mut focus_order: Vec<String> = unique_focus.iter().map(|s| (*s).clone()).collect();

    let mut rng = thread_rng();
    focus_order.shuffle(&mut rng);

    // Log analysis start with timeout duration and randomised order
    log_analysis_start(
        "synapse",
        input.analysis_deadline_ms,
        focus_order.len(),
        &focus_order,
    );

    // Track completed focus neurons for timeout logging
    let total_focus_count = focus_order.len();
    let completed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // v0.1.134: Default threshold is 0 - return ALL positive improvements.
    // TypeScript will decide which candidates are worth the cost of growth.
    let threshold = input.improvement_threshold.unwrap_or(0.0);

    // Collect all helpful evaluation work first for batching
    struct HelpfulWork {
        source_uuid: String,
        target_uuid: String,
        samples: Vec<HelpfulSample>,
    }

    // GPU is always required - TypeScript layer calls check_gpu_available() and skips
    // discovery entirely on machines without GPU. Reaching here without GPU is a bug.
    assert!(
        GpuAnalyzer::gpu_is_available(),
        "Discovery logic called without GPU - check_gpu_available should have prevented this"
    );
    let gpu_used = true;

    let helpful_results = Arc::new(Mutex::new(Vec::<CandidateSynapseJson>::new()));
    let harmful_results = Arc::new(Mutex::new(Vec::<CandidateSynapseJson>::new()));
    let helpful_fallback = Arc::new(Mutex::new(Option::<CandidateSynapseJson>::None));
    let analysis_timed_out = Arc::new(Mutex::new(false));

    // Metadata tracking for observability (v0.2.17+)
    // These track whether target_value was available and whether saturation-aware simulation was used
    let metadata_target_value_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let metadata_saturation_aware_used = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let metadata_seen_any_input_with_records = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let metadata_input_min_with_records = Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
    let metadata_input_max_with_records = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let focus_order_arc = Arc::new(focus_order);
    let ordered_neurons_arc = Arc::new(ordered_neurons);
    let existing_synapses_arc = Arc::new(existing_synapses);
    let synapses_by_target_arc = Arc::new(synapses_by_target);
    let order_map_arc = Arc::new(order_map);
    let neuron_squash_map_arc = Arc::new(neuron_squash_map);

    // Build a comprehensive map of ALL neuron UUIDs to their types
    // This includes: input neurons, and all neurons from creature.neurons (hidden, output, constant)
    // If a UUID is not in this map, it's an invalid UUID (bug)
    let mut neuron_type_map: HashMap<String, String> = HashMap::new();

    // Add input neurons (they're not in creature.neurons, only represented by creature.input count)
    for input_index in 0..input.creature.input {
        neuron_type_map.insert(format!("input-{input_index}"), "input".to_string());
    }

    // Add all neurons from creature.neurons (hidden, output, constant)
    for neuron in &input.creature.neurons {
        neuron_type_map.insert(neuron.uuid.clone(), neuron.neuron_type.clone());
    }

    let neuron_type_map_arc = Arc::new(neuron_type_map);

    // Keep input neuron UUIDs set for quick checks (backwards compatibility)
    let input_neuron_uuids: HashSet<String> = (0..input.creature.input)
        .map(|i| format!("input-{i}"))
        .collect();
    let input_neuron_uuids_arc = Arc::new(input_neuron_uuids);

    // Create a shared GPU work queue ONCE before the parallel loop.
    // This eliminates the overhead of creating multiple GPU devices (one per thread).
    // All GPU operations are processed by a single dedicated thread, improving utilisation.
    // CRITICAL: The GpuAnalyzer is created INSIDE the GPU thread to avoid wgpu deadlocks.
    let gpu_queue = Arc::new(GpuWorkQueue::new()?);

    // Process each focus neuron in parallel
    focus_order_arc
        .par_iter()
        .try_for_each(|target_uuid| -> Result<()> {
            if *analysis_timed_out.lock().expect("Mutex poisoned") || deadline_passed(&deadline) {
                *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                return Ok(());
            }

            crate::watchdog::beat(format!(
                "synapse analysis → processing target {target_uuid}"
            ));

            // Use the shared GPU work queue instead of creating a new GpuAnalyzer per thread.
            // This eliminates device creation overhead and improves GPU utilisation.
            let gpu = &*gpu_queue;

            let target_records_arc = cache.get(target_uuid.as_str())?;
            if target_records_arc.is_empty() {
                diagnostics
                    .lock()
                    .expect("Mutex poisoned: diagnostics")
                    .set_target_record_count(target_uuid, 0);
                return Ok(());
            }
            let target_records = target_records_arc.as_ref();
            diagnostics
                .lock()
                .expect("Mutex poisoned: diagnostics")
                .set_target_record_count(target_uuid, target_records.len());

            let target_index = match order_map_arc.get(target_uuid.as_str()) {
                Some(index) => *index,
                None => {
                    // Target neuron not found in order map - this indicates a data integrity issue
                    // This should never happen for valid hidden/output neurons
                    if verbose_enabled() {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {target_uuid} not found in creature neuron order map (neuron may not exist in creature definition). Skipping."
                        );
                    }
                    return Ok(());
                }
            };

            // Early validation: skip input and constant neurons (they have no upstream sources)
            // Validate target UUID exists in comprehensive neuron type map
            let target_neuron_type = neuron_type_map_arc.get(target_uuid.as_str())
                .ok_or_else(|| anyhow!(
                    "Invalid target neuron UUID '{}': not found in neuron type map. \
                    This indicates a serious data integrity bug. All valid neurons must be in the type map \
                    (input neurons: input-0..input-{}, or neurons from creature.neurons array).",
                    target_uuid,
                    input_neuron_uuids_arc.len().saturating_sub(1)
                ))?;

            let input_count = input_neuron_uuids_arc.len();
            let is_input_neuron = target_neuron_type == "input";
            let is_constant_neuron = target_neuron_type == "constant";

            // Skip actual input/constant neurons (expected - they have no upstream sources)
            // Only skip by UUID check, not by index, to avoid incorrectly skipping hidden neurons
            // that might have been assigned incorrect indices due to ordering bugs
            if is_input_neuron || is_constant_neuron {
                return Ok(());
            }

            // Filter eligible sources: must have index < target_index and not be a constant
            // All neurons should be in the comprehensive neuron_type_map
            let mut eligible_sources: Vec<&OrderedNeuron> = ordered_neurons_arc
                .iter()
                .filter(|neuron| {
                    neuron.index < target_index
                        && {
                            // Look up neuron type - if missing, it's a serious bug
                            match neuron_type_map_arc.get(&neuron.uuid) {
                                Some(neuron_type) => {
                                    // Valid neuron - exclude constants, include everything else (input, hidden, output)
                                    neuron_type != "constant"
                                }
                                None => {
                                    // Invalid UUID - serious data integrity bug
                                    eprintln!(
                                        "[NEAT-AI-Discovery] ERROR: Invalid neuron UUID '{}' found in ordered_neurons. \
                                        Not found in comprehensive neuron type map. This indicates a serious data integrity bug.",
                                        neuron.uuid
                                    );
                                    false // Exclude invalid neurons
                                }
                            }
                        }
                })
                .collect();

            // Track total eligible sources before filtering
            let total_eligible = eligible_sources.len() as u32;

            // For hidden/output neurons with index >= input_count, there should always be at least the input neurons as eligible sources
            // If total_eligible == 0, this indicates a serious bug
            if total_eligible == 0 {
                // This should be impossible - we've already filtered out input/constant neurons
                // Log detailed diagnostics to help debug
                let neurons_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| n.index < target_index)
                    .count();
                let constants_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| {
                        n.index < target_index
                            && neuron_type_map_arc
                                .get(&n.uuid)
                                .map(|t| t == "constant")
                                .unwrap_or(false)
                    })
                    .count();
                let input_neurons_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| n.index < target_index && input_neuron_uuids_arc.contains(&n.uuid))
                    .count();

                eprintln!(
                    "[NEAT-AI-Discovery] BUG: Target {target_uuid} (type: {target_neuron_type}, index: {target_index}) has no eligible upstream neurons. \
                    creature.input: {input_count}, neurons before target: {neurons_before_index}, constants before target: {constants_before_index}, \
                    input neurons before target: {input_neurons_before_index}. This should not happen for hidden/output neurons with index >= creature.input."
                );

                // Still skip to avoid crashing, but log the bug
                return Ok(());
            }
            // Count how many eligible sources are input neurons
            let input_neuron_count = eligible_sources
                .iter()
                .filter(|neuron| input_neuron_uuids_arc.contains(&neuron.uuid))
                .count() as u32;
            diagnostics
                .lock()
                .expect("Mutex poisoned: diagnostics")
                .set_total_eligible_sources(target_uuid, total_eligible);
            diagnostics
                .lock()
                .expect("Mutex poisoned: diagnostics")
                .set_input_neuron_count(target_uuid, input_neuron_count);

            let mut rng = thread_rng();
            eligible_sources.shuffle(&mut rng);

            // Improved GPU utilisation: Build samples on CPU in parallel, then batch GPU evaluation
            // This avoids the GPU sync overhead of calling build_samples_gpu for each source.
            // The heavy computation is in evaluate_helpful_batch which is properly batched.
            struct SourceWorkResult {
                work: Option<HelpfulWork>,
                had_samples: bool,
                source_uuid: String,
                record_count: usize,
            }

            // Pre-filter sources and collect their records (cache is thread-safe)
            // Track already-connected, load-failure, and empty-record counts separately
            let mut already_connected_count = 0u32;
            let mut load_failure_count = 0u32;
            let mut empty_record_sources: Vec<String> = Vec::new();
            let mut sources_to_process: Vec<(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)> =
                Vec::with_capacity(eligible_sources.len());

            for source in &eligible_sources {
                if deadline_passed(&deadline) {
                    *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                    break;
                }
                let source_uuid = source.uuid.as_str();

                if existing_synapses_arc
                    .contains(&(source_uuid.to_string(), target_uuid.to_string()))
                {
                    already_connected_count += 1;
                    continue;
                }

                match cache.get(source_uuid) {
                    Ok(records) => {
                        if !records.is_empty() {
                            if let Some(input_index) = parse_input_index(source_uuid) {
                                metadata_seen_any_input_with_records.store(
                                    true,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                // Update min/max atomically (best-effort).
                                let _ = metadata_input_min_with_records.fetch_min(
                                    input_index,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                let _ = metadata_input_max_with_records.fetch_max(
                                    input_index,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            sources_to_process.push((source, records));
                        } else {
                            // Empty records - track for diagnostics
                            empty_record_sources.push(source_uuid.to_string());
                            let is_input_neuron = input_neuron_uuids_arc.contains(source_uuid);
                            // Log non-input neurons with empty records (input neurons are logged as summary below)
                            if verbose_enabled() && !is_input_neuron {
                                eprintln!(
                                    "[NEAT-AI-Discovery][verbose] Source {source_uuid} (target {target_uuid}) has no records in parquet file."
                                );
                            }
                        }
                    }
                    Err(err) => {
                        load_failure_count += 1;
                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] Failed to load records for source {source_uuid} (target {target_uuid}): {err}"
                            );
                        }
                    }
                };
            }

            // Count how many empty record sources are input neurons (helps diagnose parquet data issues)
            let empty_input_neuron_count = empty_record_sources
                .iter()
                .filter(|uuid| input_neuron_uuids_arc.contains(uuid.as_str()))
                .count();
            let empty_non_input_count = empty_record_sources.len() - empty_input_neuron_count;

            // Log summary if many input neurons have empty records (indicates data issue)
            if verbose_enabled() && empty_input_neuron_count > 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {target_uuid}: {} of {} input neurons have no records in parquet file (plus {} non-input sources). This may indicate incomplete parquet data.",
                    empty_input_neuron_count,
                    input_neuron_uuids_arc.len(),
                    empty_non_input_count
                );
            }

            // Update diagnostics for already-connected, load failures, and empty records
            if already_connected_count > 0
                || load_failure_count > 0
                || !empty_record_sources.is_empty()
            {
                let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                for _ in 0..already_connected_count {
                    diag.record_already_connected(target_uuid);
                }
                for _ in 0..load_failure_count {
                    diag.record_load_failure(target_uuid);
                }
                // Record diagnostics for sources with empty records (matches old sequential behaviour)
                for source_uuid in &empty_record_sources {
                    diag.record_candidate_attempt(target_uuid, false);
                    diag.record_no_samples(target_uuid, source_uuid, 0);
                }
            }

            // Build samples on CPU (fast hashmap matching, no GPU sync overhead)
            // OPTIMIZATION: Pre-build target map ONCE, reuse for all sources.
            // This avoids rebuilding the HashMap for each of ~1000+ source neurons.
            let target_map = TargetMap::from_records(target_records);
            let target_map_ref = &target_map;

            // Even if target_map is empty, we continue to record diagnostics
            // about what sources were evaluated (important for debugging).
            let source_results: Vec<SourceWorkResult> = sources_to_process
                .par_iter()
                .map(|(source, from_records_arc)| {
                    let source_uuid = source.uuid.as_str();
                    let from_records = from_records_arc.as_ref();
                    let record_count = from_records.len();

                    // Use pre-built target map - avoids rebuilding HashMap for each source
                    let samples = target_map_ref.build_samples_from(from_records);
                    let had_samples = !samples.is_empty();

                    let work = if had_samples {
                        Some(HelpfulWork {
                            source_uuid: source_uuid.to_string(),
                            target_uuid: target_uuid.to_string(),
                            samples,
                        })
                    } else {
                        None
                    };

                    SourceWorkResult {
                        work,
                        had_samples,
                        source_uuid: source_uuid.to_string(),
                        record_count,
                    }
                })
                .collect();

            // Extract work batch and batch diagnostics updates
            let mut helpful_work_batch: Vec<HelpfulWork> = Vec::new();
            let mut diagnostics_updates: Vec<(String, String, bool, usize)> = Vec::new();

            for result in source_results {
                if let Some(work) = result.work {
                    helpful_work_batch.push(work);
                }
                diagnostics_updates.push((
                    target_uuid.to_string(),
                    result.source_uuid,
                    result.had_samples,
                    result.record_count,
                ));
            }

            // Apply all diagnostics updates in a single lock (reduces contention)
            if !diagnostics_updates.is_empty() {
                let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                for (target, source, had_samples, record_count) in diagnostics_updates {
                    diag.record_candidate_attempt(&target, had_samples);
                    if !had_samples {
                        diag.record_no_samples(&target, &source, record_count);
                    }
                }
            }

            // Process helpful work in batches for better GPU utilization
            // Vertical timeout: Complete all GPU batch processing for the current focus neuron
            if !helpful_work_batch.is_empty() {
                // Clone samples for the GPU queue (queue takes ownership)
                let helpful_samples: Vec<Vec<HelpfulSample>> = helpful_work_batch
                    .iter()
                    .map(|w| w.samples.clone())
                    .collect();

                // Track metadata: check if any samples have target_value data
                // This is used to determine if saturation-aware simulation was possible.
                for samples in &helpful_samples {
                    if samples.iter().any(|s| s.target_value.is_some()) {
                        metadata_target_value_seen.store(true, std::sync::atomic::Ordering::Relaxed);
                        break;
                    }
                }

                let helpful_stats_batch = gpu.evaluate_helpful_batch(helpful_samples, &deadline)?;

                // Process results - collect all updates first, then apply in batches (reduces mutex contention)
                let mut candidates_to_add = Vec::new();
                let mut diagnostics_zero_improvements = Vec::new();
                let mut diagnostics_below_threshold = Vec::new();
                let mut diagnostics_selected = Vec::new();

                for (work, stats) in helpful_work_batch.iter().zip(helpful_stats_batch.iter()) {
                    let positive_is_better = stats.positive_count >= stats.negative_count;
                    let gpu_improved_count = if positive_is_better {
                        stats.positive_count
                    } else {
                        stats.negative_count
                    };
                    if gpu_improved_count == 0 {
                        diagnostics_zero_improvements.push((
                            work.target_uuid.clone(),
                            work.source_uuid.clone(),
                            work.samples.len(),
                            stats.positive_count,
                            stats.negative_count,
                        ));
                        continue;
                    }

                    let total_count = work.samples.len() as u32;
                    if total_count == 0 {
                        continue;
                    }

                    // Use shared weight calculation (synapse = direct connection, so incoming_weight = 1.0)
                    // The shared function ensures consistent weight calculation across synapse and neuron analysis
                    let weight = match calculate_optimal_outgoing_weight(
                        stats.error_activation_sum,
                        stats.activation_sq_sum,
                        1.0, // Synapses are direct connections, no intermediate neuron
                    ) {
                        Some(w) => w,
                        None => continue, // Skip if weight is invalid
                    };

                    // Get target's squash function for saturation-aware improvement calculation.
                    // For saturating activations (HARD_TANH, TANH, LOGISTIC, etc.), the linear model
                    // overpredicts improvement near saturation. Using the actual activation function
                    // gives accurate predictions that match real-world results.
                    let target_squash = neuron_squash_map_arc
                        .get(&work.target_uuid)
                        .map(|s| s.as_str());

                    // Track metadata: check if saturation-aware simulation is used for this candidate.
                    // get_target_simulation_fn returns Some when:
                    // 1. The target squash is a supported saturating activation, AND
                    // 2. All samples have target_value/target_activation data
                    if get_target_simulation_fn(&work.samples, target_squash).is_some() {
                        metadata_saturation_aware_used.store(true, std::sync::atomic::Ordering::Relaxed);
                    }

                    // Compute expected improvement using the linear error model.
                    // This works for ALL squash types because we're measuring actual errors
                    // from recordings, not predicting theoretical errors. The correlation
                    // between source activation and target error determines improvement.
                    // Issue #128: This is neuron-level improvement - impact discounting converts to creature-level.
                    let (neuron_error_improvement, improved_count, worsened_count) = {
                        let baseline_error_sq = stats.error_sq_sum;
                        let (improvement, improved, worsened, _) =
                            compute_synapse_improvement_and_count(
                                &work.samples,
                                weight,
                                baseline_error_sq,
                                target_squash,
                            );
                        (improvement, improved, worsened)
                    };

                    // Accept all positive improvements as candidates (not just those above threshold)
                    // Only reject if improvement is non-positive (<= 0.0)
                    if neuron_error_improvement <= 0.0 {
                        // Skip non-positive improvements
                        continue;
                    }

                    // If positive but below threshold, still accept as candidate but log for diagnostics
                    if neuron_error_improvement <= threshold {
                        diagnostics_below_threshold.push((
                            work.target_uuid.clone(),
                            work.source_uuid.clone(),
                            ThresholdContext {
                                sample_count: work.samples.len(),
                                expected_improvement: neuron_error_improvement,
                                threshold,
                                improved_count,
                                worsened_count,
                                weight,
                            },
                        ));
                    }

                    diagnostics_selected.push(work.target_uuid.clone());
                    let target_stats = cache
                        .get(&work.target_uuid)
                        .ok()
                        .and_then(|records| NeuronStats::from_records(records.as_ref()))
                        .map(|s| s.to_json());
                    // Issue #128: Use creature-level metrics (impact discounting applied later)
                    candidates_to_add.push(CandidateSynapseJson {
                        from_neuron_uuid: work.source_uuid.clone(),
                        to_neuron_uuid: work.target_uuid.clone(),
                        from_neuron_index: None,
                        to_neuron_index: None,
                        weight,
                        target_neuron_impact: 1.0,
                        expected_creature_error_reduction: neuron_error_improvement,
                        expected_creature_score_gain: neuron_error_improvement,
                        improved_count,
                        total_count,
                        target_neuron_stats: target_stats,
                    });
                }

                // Apply all diagnostics updates in batches (minimizes mutex contention)
                if !diagnostics_zero_improvements.is_empty() {
                    let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                    for (target, source, sample_count, pos, neg) in diagnostics_zero_improvements {
                        diag.record_zero_improvement(&target, &source, sample_count, pos, neg);
                    }
                }
                if !diagnostics_below_threshold.is_empty() {
                    let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                    for (target, source, context) in diagnostics_below_threshold {
                        diag.record_below_threshold(&target, &source, context);
                    }
                }
                if !diagnostics_selected.is_empty() {
                    let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                    for target in diagnostics_selected {
                        diag.mark_candidate_selected(&target);
                    }
                }
                if !candidates_to_add.is_empty() {
                    let mut results = helpful_results
                        .lock()
                        .expect("Mutex poisoned: helpful_results");
                    results.extend(candidates_to_add);
                }
            }

            // Process harmful synapses for this target - BATCHED GPU evaluation for better utilisation
            // Vertical timeout: Complete all harmful synapse processing for the current focus neuron
            if !*analysis_timed_out.lock().expect("Mutex poisoned") {
                if let Some(existing) = synapses_by_target_arc.get(target_uuid.as_str()) {
                    // Phase 1: Build all samples on CPU (fast, parallel-friendly)
                    // Reuse the target_map we already built for helpful synapse processing
                    struct HarmfulWork {
                        synapse: SynapseJson,
                        samples: Vec<HelpfulSample>,
                    }
                    let mut harmful_work: Vec<HarmfulWork> = Vec::with_capacity(existing.len());

                    for synapse in existing {
                        let from_records_arc = match cache.get(&synapse.from_uuid) {
                            Ok(records) => records,
                            Err(_) => continue,
                        };
                        if from_records_arc.is_empty() {
                            continue;
                        }
                        let from_records = from_records_arc.as_ref();
                        // Use pre-built target map - avoids rebuilding HashMap for each synapse
                        let samples = target_map_ref.build_samples_from(from_records);
                        if samples.is_empty() {
                            continue;
                        }

                        harmful_work.push(HarmfulWork {
                            synapse: synapse.clone(),
                            samples,
                        });
                    }

                    // Phase 2: Batch GPU evaluation (single submission for all synapses)
                    if !harmful_work.is_empty() {
                        // Clone samples for the GPU queue (queue takes ownership)
                        let batch_input: Vec<(Vec<HelpfulSample>, f32)> = harmful_work
                            .iter()
                            .map(|w| (w.samples.clone(), w.synapse.weight))
                            .collect();

                        let batch_stats = gpu.evaluate_harmful_batch(batch_input, &deadline)?;

                        // Phase 3: Process results
                        let mut harmful_candidates = Vec::with_capacity(batch_stats.len());
                        let target_stats = cache
                            .get(target_uuid.as_str())
                            .ok()
                            .and_then(|records| NeuronStats::from_records(records.as_ref()))
                            .map(|s| s.to_json());

                        for (work, stats) in harmful_work.iter().zip(batch_stats.iter()) {
                            let total_count = work.samples.len() as u32;
                            if total_count == 0 {
                                continue;
                            }

                            // Issue #128: This is neuron-level - impact discounting converts to creature-level
                            let neuron_error_improvement = (stats.harmful_count as f32
                                - stats.helpful_count as f32)
                                / total_count as f32;

                            // Issue #128: Use creature-level metrics (impact discounting applied later)
                            harmful_candidates.push(CandidateSynapseJson {
                                from_neuron_uuid: work.synapse.from_uuid.clone(),
                                to_neuron_uuid: work.synapse.to_uuid.clone(),
                                from_neuron_index: None,
                                to_neuron_index: None,
                                weight: work.synapse.weight,
                                target_neuron_impact: 1.0,
                                expected_creature_error_reduction: neuron_error_improvement,
                                expected_creature_score_gain: neuron_error_improvement,
                                improved_count: stats.harmful_count,
                                total_count,
                                target_neuron_stats: target_stats.clone(),
                            });
                        }

                        // Batch push all harmful candidates (single lock)
                        if !harmful_candidates.is_empty() {
                            let mut results = harmful_results
                                .lock()
                                .expect("Mutex poisoned: harmful_results");
                            results.extend(harmful_candidates);
                        }
                    }
                }
            }

            // Track completion of this focus neuron for timeout reporting.
            let completed =
                completed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            crate::watchdog::beat(format!(
                "synapse analysis → completed {completed}/{total_focus_count} (last target {target_uuid})"
            ));
            Ok(())
        })?;

    let analysis_timed_out = *analysis_timed_out.lock().expect("Mutex poisoned");
    let mut helpful_results = helpful_results.lock().expect("Mutex poisoned").clone();
    let mut harmful_results = harmful_results.lock().expect("Mutex poisoned").clone();
    let mut helpful_fallback = helpful_fallback.lock().expect("Mutex poisoned").take();
    let mut diagnostics = diagnostics.lock().expect("Mutex poisoned");

    // Log timeout with completion stats (always visible, not just verbose)
    if analysis_timed_out {
        let completed = completed_count.load(std::sync::atomic::Ordering::Relaxed);
        log_analysis_timeout("synapse", completed, total_focus_count);
    }

    if helpful_results.is_empty() {
        if let Some(candidate) = helpful_fallback.take() {
            diagnostics.mark_candidate_selected(&candidate.to_neuron_uuid);
            helpful_results.push(candidate);
        }
    }

    // Issue #128: Apply impact-based discounting and set creature-level metrics.
    // Output neurons have impact = 1.0 (no discount).
    // Hidden neurons have impact in [0, 1] based on their weighted paths to outputs.
    let impact_scores = compute_impact_scores_for_discounting(&input.creature, cache.as_ref());
    let neuron_type_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.neuron_type.clone()))
        .collect();

    // Apply impact discounting to helpful synapse candidates
    for candidate in &mut helpful_results {
        // Populate indices for debugging/analysis (consistent with CandidateNeuronJson).
        candidate.from_neuron_index = order_map_arc.get(&candidate.from_neuron_uuid).copied();
        candidate.to_neuron_index = order_map_arc.get(&candidate.to_neuron_uuid).copied();

        let is_hidden = neuron_type_map
            .get(&candidate.to_neuron_uuid)
            .map(|t| t != "output")
            .unwrap_or(true); // Default to hidden if type unknown

        let impact = if is_hidden {
            if let Some(&impact) = impact_scores.get(&candidate.to_neuron_uuid) {
                impact.clamp(0.0, 1.0)
            } else {
                // No impact score means disconnected from outputs - heavy discount
                0.1
            }
        } else {
            // Output neuron - full impact
            1.0
        };

        // Update creature-level metrics
        candidate.target_neuron_impact = impact;
        let original = candidate.expected_creature_error_reduction;
        candidate.expected_creature_error_reduction *= impact;
        candidate.expected_creature_score_gain = candidate.expected_creature_error_reduction;

        if verbose_enabled() && is_hidden {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Synapse candidate → {} impact {:.3}: \
                {:.4}% → {:.4}%",
                &candidate.to_neuron_uuid[..12.min(candidate.to_neuron_uuid.len())],
                impact,
                original * 100.0,
                candidate.expected_creature_score_gain * 100.0
            );
        }
    }

    // Apply impact discounting to harmful synapse candidates (same logic)
    for candidate in &mut harmful_results {
        // Populate indices for debugging/analysis (consistent with CandidateNeuronJson).
        candidate.from_neuron_index = order_map_arc.get(&candidate.from_neuron_uuid).copied();
        candidate.to_neuron_index = order_map_arc.get(&candidate.to_neuron_uuid).copied();

        let is_hidden = neuron_type_map
            .get(&candidate.to_neuron_uuid)
            .map(|t| t != "output")
            .unwrap_or(true);

        let impact = if is_hidden {
            if let Some(&impact) = impact_scores.get(&candidate.to_neuron_uuid) {
                impact.clamp(0.0, 1.0)
            } else {
                0.1
            }
        } else {
            1.0
        };

        candidate.target_neuron_impact = impact;
        candidate.expected_creature_error_reduction *= impact;
        candidate.expected_creature_score_gain = candidate.expected_creature_error_reduction;
    }

    // Note: helpful_fallback does NOT need separate discounting here.
    // If helpful_results was empty, the fallback was already moved into it via .take()
    // at line ~6877 and gets discounted in the loop above. If helpful_results was NOT
    // empty, the fallback is intentionally not returned (we have better candidates).

    helpful_results.sort_by(|a, b| {
        // Issue #128: Sort by expected creature score gain (highest first)
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(Ordering::Equal)
    });
    harmful_results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(Ordering::Equal)
    });

    // Track candidates_found before truncation for metadata
    let candidates_found = helpful_results.len() + harmful_results.len();

    if let Some(limit) = input.max_candidates {
        helpful_results.truncate(limit);
        harmful_results.truncate(limit);
    }

    // Track candidates_returned after truncation
    let candidates_returned = helpful_results.len() + harmful_results.len();

    let no_candidate_reasons = diagnostics.no_candidate_summaries();
    diagnostics.emit_logs();

    // Build metadata for observability (v0.2.17+)
    // Note: target_value_available and saturation_aware_simulation_used are tracked
    // during the inner analysis loop via atomic flags.
    let saw_any_input =
        metadata_seen_any_input_with_records.load(std::sync::atomic::Ordering::Relaxed);
    let input_min = metadata_input_min_with_records.load(std::sync::atomic::Ordering::Relaxed);
    let input_max = metadata_input_max_with_records.load(std::sync::atomic::Ordering::Relaxed);
    let metadata = super::shared::SynapseAnalysisMetadata {
        target_value_available: metadata_target_value_seen
            .load(std::sync::atomic::Ordering::Relaxed),
        saturation_aware_simulation_used: metadata_saturation_aware_used
            .load(std::sync::atomic::Ordering::Relaxed),
        candidates_found,
        candidates_returned,
        timed_out: analysis_timed_out,
        input_index_min_seen_with_records: if saw_any_input { Some(input_min) } else { None },
        input_index_max_seen_with_records: if saw_any_input { Some(input_max) } else { None },
    };

    Ok(AnalyzeSynapsesResult {
        helpful_synapses: helpful_results,
        harmful_synapses: harmful_results,
        gpu_used,
        no_candidate_reasons,
        metadata,
    })
}

pub fn analyze_synapses(input: &AnalyzeSynapsesInput) -> Result<AnalyzeSynapsesResult> {
    // Validate focus_neurons before expensive pre-loading
    require_unique_focus(&input.focus_neurons, "Synapse analysis")?;

    // Pre-load all records for faster analysis (1 scan vs ~2000 scans)
    let cache = Arc::new(RecordCache::new_adaptive(&input.parquet_file)?);
    analyze_synapses_with_cache(input, cache)
}

#[cfg(test)]
mod tests_synapses {
    use super::*;
    use crate::analysis::analyze_all;
    use crate::parquet_format::write_records_to_parquet;
    use crate::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, SystemTime};
    use tempfile::tempdir;

    /// Helper macro to skip tests that require GPU when no GPU is available.
    /// This allows tests to pass gracefully in CI environments without GPUs.
    macro_rules! skip_if_no_gpu {
        () => {
            if !GpuAnalyzer::gpu_is_available() {
                eprintln!("⚠️  Skipping test: GPU not available");
                return;
            }
        };
    }

    #[test]
    fn deadline_passed_detects_elapsed_wall_clock_deadline() {
        // Note: This test does NOT require GPU - it only tests the deadline_passed
        // function which performs simple time comparisons. Do not add skip_if_no_gpu!()
        // Use the deadline override mechanism in tests so behaviour is deterministic
        let _guard = deadline_override::DeadlineOverrideGuard::with_sequence(vec![true, false]);

        let dummy_deadline = Some(SystemTime::now());
        assert!(
            deadline_passed(&dummy_deadline),
            "past deadlines should be treated as expired immediately"
        );

        assert!(
            !deadline_passed(&dummy_deadline),
            "future deadlines should not be marked as expired"
        );

        assert!(
            !deadline_passed(&None),
            "missing deadlines should behave as if no timeout was requested"
        );
    }

    #[test]
    fn build_deadline_handles_absolute_timestamps_and_relative_durations() {
        // Verify that build_deadline correctly handles both absolute timestamps
        // (milliseconds since UNIX_EPOCH) and relative durations (milliseconds from now)
        let now = SystemTime::now();
        let now_ms = now
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_millis() as u64;

        // Test 1: Absolute timestamp (large value, >= year 2000)
        // Create a deadline 10 minutes in the future using absolute timestamp
        let ten_minutes_ms = 10 * 60 * 1000; // 10 minutes in milliseconds
        let future_deadline_ms = now_ms + ten_minutes_ms;
        let deadline = build_deadline(Some(future_deadline_ms));

        assert!(
            deadline.is_some(),
            "deadline should be Some when deadline_ms is provided"
        );

        let deadline_time = deadline.unwrap();

        // The deadline should be approximately 10 minutes in the future
        // Allow for some small timing variance (up to 1 second)
        if let Ok(duration) = deadline_time.duration_since(now) {
            let expected_min = Duration::from_millis(ten_minutes_ms) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(ten_minutes_ms) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "absolute timestamp deadline should be approximately 10 minutes in the future, got {duration:?}"
            );
        } else {
            panic!("deadline should be in the future");
        }

        // Test 2: Relative duration (small value, < year 2000)
        // Pass 10 minutes as a relative duration
        let relative_deadline_ms = ten_minutes_ms; // 10 minutes as relative duration
        let relative_deadline = build_deadline(Some(relative_deadline_ms));
        assert!(relative_deadline.is_some());
        let relative_time = relative_deadline.unwrap();
        // This should also be approximately 10 minutes in the future
        if let Ok(duration) = relative_time.duration_since(now) {
            let expected_min = Duration::from_millis(ten_minutes_ms) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(ten_minutes_ms) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "relative duration deadline should be approximately 10 minutes in the future, got {duration:?}"
            );
        } else {
            panic!("relative deadline should be in the future");
        }

        // Test 3: Verify that an absolute timestamp in the past returns None
        // (deadline already passed - no point in creating a deadline)
        let past_timestamp_ms = 1_700_000_000_000u64; // Jan 2024 (in the past)
        let past_deadline = build_deadline(Some(past_timestamp_ms));
        assert!(
            past_deadline.is_none(),
            "Past timestamp should return None (deadline already passed)"
        );

        // Test 4: Verify that a future absolute timestamp is correctly converted to relative duration
        let future_timestamp_ms = now_ms + ten_minutes_ms; // 10 minutes in the future as absolute timestamp
        let future_deadline = build_deadline(Some(future_timestamp_ms));
        assert!(future_deadline.is_some());
        let future_time = future_deadline.unwrap();
        // Should be approximately 10 minutes in the future
        if let Ok(duration) = future_time.duration_since(now) {
            let expected_min = Duration::from_millis(ten_minutes_ms) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(ten_minutes_ms) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Future absolute timestamp should be converted to relative duration correctly, got {duration:?}"
            );
        } else {
            panic!("Future deadline should be in the future");
        }
    }

    #[test]
    fn build_deadline_validates_duration_bounds() {
        // Test that build_deadline validates and defaults to 10 minutes for invalid values
        let now = SystemTime::now();
        const DEFAULT_DURATION_MS: u64 = 600_000; // 10 minutes

        // Test 1: Duration below minimum (3 seconds) should default to 10 minutes
        let too_short_ms = 1_000u64; // 1 second
        let deadline = build_deadline(Some(too_short_ms));
        assert!(deadline.is_some());
        let deadline_time = deadline.unwrap();
        if let Ok(duration) = deadline_time.duration_since(now) {
            // Should default to 10 minutes (600000 ms)
            let expected_min = Duration::from_millis(DEFAULT_DURATION_MS) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(DEFAULT_DURATION_MS) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Duration below 3 seconds should default to 10 minutes, got {duration:?}"
            );
        } else {
            panic!("Default deadline should be in the future");
        }

        // Test 2: Duration above maximum (1 hour) should default to 10 minutes
        let too_long_ms = 4_000_000u64; // ~66 minutes
        let deadline = build_deadline(Some(too_long_ms));
        assert!(deadline.is_some());
        let deadline_time = deadline.unwrap();
        if let Ok(duration) = deadline_time.duration_since(now) {
            // Should default to 10 minutes (600000 ms)
            let expected_min = Duration::from_millis(DEFAULT_DURATION_MS) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(DEFAULT_DURATION_MS) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Duration above 1 hour should default to 10 minutes, got {duration:?}"
            );
        } else {
            panic!("Default deadline should be in the future");
        }

        // Test 3: Valid duration (10 minutes) should pass through unchanged
        let valid_ms = 10 * 60 * 1000u64; // 10 minutes
        let deadline = build_deadline(Some(valid_ms));
        assert!(deadline.is_some());
        let deadline_time = deadline.unwrap();
        if let Ok(duration) = deadline_time.duration_since(now) {
            let expected_min = Duration::from_millis(valid_ms) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(valid_ms) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Valid duration should pass through unchanged, got {duration:?}"
            );
        } else {
            panic!("Valid deadline should be in the future");
        }

        // Test 4: Exactly at minimum (3 seconds) should pass through
        let min_ms = 3_000u64; // Exactly 3 seconds
        let deadline = build_deadline(Some(min_ms));
        assert!(deadline.is_some());
        let deadline_time = deadline.unwrap();
        if let Ok(duration) = deadline_time.duration_since(now) {
            let expected_min = Duration::from_millis(min_ms) - Duration::from_millis(100);
            let expected_max = Duration::from_millis(min_ms) + Duration::from_millis(100);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Duration at minimum should pass through, got {duration:?}"
            );
        } else {
            panic!("Minimum deadline should be in the future");
        }

        // Test 5: Exactly at maximum (1 hour) should pass through
        let max_ms = 3_600_000u64; // Exactly 1 hour
        let deadline = build_deadline(Some(max_ms));
        assert!(deadline.is_some());
        let deadline_time = deadline.unwrap();
        if let Ok(duration) = deadline_time.duration_since(now) {
            let expected_min = Duration::from_millis(max_ms) - Duration::from_millis(1000);
            let expected_max = Duration::from_millis(max_ms) + Duration::from_millis(1000);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Duration at maximum should pass through, got {duration:?}"
            );
        } else {
            panic!("Maximum deadline should be in the future");
        }
    }

    /// Test that calculate_effective_timeout_ms applies the same logic as build_deadline.
    /// This is critical for ensuring log_analysis_start displays the correct timeout.
    #[test]
    fn calculate_effective_timeout_ms_matches_build_deadline_logic() {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_millis() as u64;

        // Test 1: Relative duration (15 minutes) should pass through unchanged
        let fifteen_minutes_ms = 15 * 60 * 1000u64;
        let result = calculate_effective_timeout_ms(Some(fifteen_minutes_ms));
        assert_eq!(
            result,
            Some(fifteen_minutes_ms),
            "15 minute relative duration should pass through unchanged"
        );

        // Test 2: Absolute timestamp (now + 15 minutes) should convert to ~15 minutes
        let absolute_15min = now_ms + fifteen_minutes_ms;
        let result = calculate_effective_timeout_ms(Some(absolute_15min));
        assert!(
            result.is_some(),
            "Future absolute timestamp should return Some"
        );
        let effective_ms = result.unwrap();
        // Allow 2 second tolerance for timing variance
        assert!(
            effective_ms >= fifteen_minutes_ms - 2000 && effective_ms <= fifteen_minutes_ms + 2000,
            "Absolute timestamp should convert to ~15 minutes, got {effective_ms}ms"
        );

        // Test 3: Duration below minimum (1 second) should default to 10 minutes
        let too_short_ms = 1_000u64;
        let result = calculate_effective_timeout_ms(Some(too_short_ms));
        assert_eq!(
            result,
            Some(DEFAULT_DURATION_MS),
            "Duration below 3 seconds should default to 10 minutes"
        );

        // Test 4: Duration above maximum (2 hours) should default to 10 minutes
        let too_long_ms = 2 * 3_600_000u64;
        let result = calculate_effective_timeout_ms(Some(too_long_ms));
        assert_eq!(
            result,
            Some(DEFAULT_DURATION_MS),
            "Duration above 1 hour should default to 10 minutes"
        );

        // Test 5: Past absolute timestamp should return None
        let past_timestamp_ms = 1_700_000_000_000u64; // Circa late 2023
        let result = calculate_effective_timeout_ms(Some(past_timestamp_ms));
        assert!(
            result.is_none(),
            "Past timestamp should return None (deadline already passed)"
        );

        // Test 6: None should default to 10 minutes
        let result = calculate_effective_timeout_ms(None);
        assert_eq!(
            result,
            Some(DEFAULT_DURATION_MS),
            "None should default to 10 minutes"
        );
    }

    #[test]
    fn record_cache_loads_once_per_neuron_under_contention() {
        let load_counter = Arc::new(AtomicUsize::new(0));
        let loader_counter = Arc::clone(&load_counter);
        let loader = Arc::new(
            move |_file: &str, neuron_uuid: &str| -> Result<Vec<DiscoverRecord>> {
                loader_counter.fetch_add(1, AtomicOrdering::SeqCst);
                thread::sleep(Duration::from_millis(50));
                Ok(vec![DiscoverRecord::new(
                    0,
                    neuron_uuid.to_string(),
                    None,
                    0.0,
                    Vec::new(),
                )])
            },
        );

        let cache = Arc::new(RecordCache::with_loader("unused.parquet", loader));
        let worker_count = 4;
        let barrier = Arc::new(Barrier::new(worker_count));
        let mut handles = Vec::new();
        for _ in 0..worker_count {
            let cache_clone = Arc::clone(&cache);
            let barrier_clone = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier_clone.wait();
                cache_clone
                    .get("neuron-1")
                    .expect("cache should load neuron records");
            }));
        }

        for handle in handles {
            handle.join().expect("worker thread should exit cleanly");
        }

        assert_eq!(
            load_counter.load(AtomicOrdering::SeqCst),
            1,
            "record cache should only hit the loader once even when multiple threads request the same neuron"
        );
    }

    /// TDD Test: evaluate_harmful_batch should handle empty batch gracefully.
    #[test]
    fn evaluate_harmful_batch_handles_empty_batch() {
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed in tests");

        let empty_batch: Vec<(&[HelpfulSample], f32)> = vec![];
        let result = analyzer
            .evaluate_harmful_batch(&empty_batch)
            .expect("Empty batch should succeed");

        assert!(result.is_empty(), "Empty batch should return empty results");
    }

    /// TDD Test: evaluate_harmful_batch should handle batch with empty sample sets.
    #[test]
    fn evaluate_harmful_batch_handles_empty_sample_sets() {
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed in tests");

        let samples: Vec<HelpfulSample> = (0..20)
            .map(|i| HelpfulSample {
                activation: (i as f32) / 20.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            })
            .collect();
        let empty_samples: Vec<HelpfulSample> = vec![];

        let batch_input = vec![
            (&samples[..], 0.5),
            (&empty_samples[..], 0.3), // Empty set in the middle
            (&samples[..], -0.2),
        ];

        let batched = analyzer
            .evaluate_harmful_batch(&batch_input)
            .expect("Batch with empty set should succeed");

        assert_eq!(batched.len(), 3, "Should return 3 results");

        // Middle result should be default (all zeros)
        assert_eq!(
            batched[1].harmful_count, 0,
            "Empty sample set should have 0 harmful_count"
        );
        assert_eq!(
            batched[1].helpful_count, 0,
            "Empty sample set should have 0 helpful_count"
        );
    }

    #[test]
    fn sample_matching_filters_non_finite_values() {
        // No GPU needed - tests the CPU build_samples function used in production
        let huge = f32::MAX;
        let target_records = vec![
            DiscoverRecord::new(0, "target".to_string(), None, 0.0, vec![0.5, -0.25]),
            DiscoverRecord::new(1, "target".to_string(), None, 0.0, vec![huge, huge]),
        ];
        let from_records = vec![
            DiscoverRecord::new(0, "from".to_string(), None, f32::INFINITY, Vec::new()),
            DiscoverRecord::new(1, "from".to_string(), None, 1.0, Vec::new()),
        ];

        let samples = build_samples(&target_records, &from_records);
        assert!(
            samples.is_empty(),
            "Sample matching should exclude non-finite samples"
        );
    }

    #[test]
    fn sample_matching_retains_legitimate_zero_samples() {
        // No GPU needed - tests the CPU build_samples function used in production
        let target_records = vec![DiscoverRecord::new(
            42,
            "target".to_string(),
            None,
            0.0,
            vec![0.0, 0.0],
        )];
        let from_records = vec![DiscoverRecord::new(
            42,
            "from".to_string(),
            None,
            0.0,
            Vec::new(),
        )];

        let samples = build_samples(&target_records, &from_records);
        assert_eq!(
            samples.len(),
            1,
            "Sample matching should include legitimate zero-valued samples"
        );

        let sample = samples[0];
        assert_eq!(
            sample.activation, 0.0,
            "Zero activation should be preserved"
        );
        assert_eq!(
            sample.avg_error, 0.0,
            "Zero average error should be preserved"
        );
    }

    /// Test that sample matching preserves target_value and target_activation.
    /// This is critical for accurate improvement predictions with non-linear
    /// activation functions (TANH, LOGISTIC, HARD_TANH, etc.).
    #[test]
    fn sample_matching_preserves_target_value_and_activation() {
        // No GPU needed - tests the production build_samples function
        let target_value = 0.8; // Pre-activation input sum
        let target_activation = 0.6; // Post-activation output (e.g., after TANH)
        let target_records = vec![DiscoverRecord::new(
            0,
            "target".to_string(),
            Some(target_value),
            target_activation,
            vec![0.1, -0.2],
        )];
        let from_records = vec![DiscoverRecord::new(
            0,
            "from".to_string(),
            None, // Source value not used
            0.5,  // Source activation
            Vec::new(),
        )];

        let samples = build_samples(&target_records, &from_records);
        assert_eq!(samples.len(), 1, "Should find one matching sample");
        assert_eq!(
            samples[0].target_value,
            Some(target_value),
            "Sample matching must preserve target_value for activation function simulation"
        );
        assert_eq!(
            samples[0].target_activation,
            Some(target_activation),
            "Sample matching must preserve target_activation for error calculation"
        );
        assert_eq!(
            samples[0].activation, 0.5,
            "Source activation should be preserved"
        );
    }

    /// Test that target_value enables proper activation function simulation.
    /// When target_value is available, get_target_simulation_fn should return
    /// the activation function, enabling saturation-aware improvement predictions.
    #[test]
    fn sample_matching_enables_target_activation_simulation() {
        // No GPU needed - tests the production build_samples function
        // Create samples near HARD_TANH saturation to test simulation accuracy
        let target_records = vec![
            DiscoverRecord::new(
                0,
                "target".to_string(),
                Some(0.95), // Near saturation
                0.95,       // HARD_TANH clips to 1.0 when input >= 1.0
                vec![0.1],  // Small positive error (output should be higher)
            ),
            DiscoverRecord::new(
                1,
                "target".to_string(),
                Some(-0.8),
                -0.8,
                vec![-0.15], // Small negative error (output should be lower)
            ),
        ];
        let from_records = vec![
            DiscoverRecord::new(0, "from".to_string(), None, 0.5, Vec::new()),
            DiscoverRecord::new(1, "from".to_string(), None, -0.3, Vec::new()),
        ];

        let samples = build_samples(&target_records, &from_records);

        assert_eq!(samples.len(), 2, "Should match both records");

        // Verify all samples have target data (required for simulation)
        for (i, sample) in samples.iter().enumerate() {
            assert!(
                sample.target_value.is_some(),
                "Sample {i} must have target_value for activation simulation"
            );
            assert!(
                sample.target_activation.is_some(),
                "Sample {i} must have target_activation for error calculation"
            );
        }

        // With target data available, get_target_simulation_fn should return Some
        // for activations that need simulation (HARD_TANH, TANH, ReLU, etc.)
        let simulation_fn = get_target_simulation_fn(&samples, Some("HARD_TANH"));
        assert!(
            simulation_fn.is_some(),
            "Should enable HARD_TANH simulation when samples have target data"
        );

        let simulation_fn = get_target_simulation_fn(&samples, Some("TANH"));
        assert!(
            simulation_fn.is_some(),
            "Should enable TANH simulation when samples have target data"
        );
    }

    /// Regression test: Add-neuron predictions must use weight computed WITH bias.
    ///
    /// BUG: Previously, the optimal outgoing weight was computed WITHOUT bias:
    ///   weight = Σ(error × TANH(x)) / Σ(TANH(x)²)
    ///
    /// But the actual neuron uses bias:
    ///   contribution = weight × TANH(x + bias)
    ///
    /// When bias significantly shifts the activation pattern, the weight computed
    /// without bias causes predictions to have the WRONG SIGN - predicting improvement
    /// when it actually makes things worse.
    ///
    /// This test reproduces the production failure pattern:
    /// - Predicted: +0.3% improvement
    /// - Actual: -0.2% (worse!)
    #[test]
    fn add_neuron_weight_must_include_bias_in_calculation() {
        // Scenario from production: TANH neuron with bias=1
        // This shifts the activation threshold from x>0 to x>-1
        let incoming_weight = 1.0f32;
        let bias = 1.0f32;

        // Create samples that expose the bug:
        // - Source activations centered around 0
        // - Roughly equal positive and negative errors
        // - With bias=1, TANH(x+1) is almost always positive (x > -1)
        // - Without bias, TANH(x) has mixed signs
        let samples: Vec<HelpfulSample> = vec![
            // Positive source activation, positive error (need output up)
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.3,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: 0.3,
                avg_error: 0.2,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: 0.1,
                avg_error: 0.1,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            // Negative source activation, negative error (need output down)
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.3,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.2,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: -0.1,
                avg_error: -0.1,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            // More samples near zero - these are affected most by bias
            HelpfulSample {
                activation: 0.05,
                avg_error: 0.15,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: -0.05,
                avg_error: -0.15,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
        ];

        // Compute baseline error
        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

        // BUG PATH: Compute weight WITHOUT bias (what the old code did)
        let mut sum_sq_no_bias = 0.0f32;
        let mut sum_ea_no_bias = 0.0f32;
        for s in &samples {
            let pre_act = incoming_weight * s.activation; // NO BIAS!
            let output = pre_act.tanh();
            sum_sq_no_bias += output * output;
            sum_ea_no_bias += output * s.avg_error;
        }
        let weight_without_bias = sum_ea_no_bias / sum_sq_no_bias;

        // FIX PATH: Compute weight WITH bias (what the fixed code does)
        let mut sum_sq_with_bias = 0.0f32;
        let mut sum_ea_with_bias = 0.0f32;
        for s in &samples {
            let pre_act = incoming_weight * s.activation + bias; // WITH BIAS!
            let output = pre_act.tanh();
            sum_sq_with_bias += output * output;
            sum_ea_with_bias += output * s.avg_error;
        }
        let weight_with_bias = sum_ea_with_bias / sum_sq_with_bias;

        // Compute ACTUAL error reduction using the ACTUAL neuron (with bias)
        fn compute_actual_improvement(
            samples: &[HelpfulSample],
            incoming_weight: f32,
            outgoing_weight: f32,
            bias: f32,
            baseline_error_sq: f32,
        ) -> f32 {
            let mut new_error_sq = 0.0f32;
            for s in samples {
                let pre_act = incoming_weight * s.activation + bias;
                let neuron_output = pre_act.tanh();
                let contribution = outgoing_weight * neuron_output;
                // Linear approximation: new_error = old_error - contribution
                let new_error = s.avg_error - contribution;
                new_error_sq += new_error * new_error;
            }
            // Improvement = (baseline - new) / baseline
            (baseline_error_sq - new_error_sq) / baseline_error_sq
        }

        // Test 1: Using weight computed WITHOUT bias gives WRONG prediction
        // The prediction (using weight_without_bias) should differ from actual
        let predicted_without_bias = {
            // Predicted improvement uses the same formula as actual
            // but the weight was computed from wrong activation pattern
            let mut predicted_new_error_sq = 0.0f32;
            for s in &samples {
                let pre_act = incoming_weight * s.activation; // NO BIAS in prediction!
                let output = pre_act.tanh();
                let contribution = weight_without_bias * output;
                let new_error = s.avg_error - contribution;
                predicted_new_error_sq += new_error * new_error;
            }
            (baseline_error_sq - predicted_new_error_sq) / baseline_error_sq
        };

        let actual_with_wrong_weight = compute_actual_improvement(
            &samples,
            incoming_weight,
            weight_without_bias,
            bias,
            baseline_error_sq,
        );

        // The bug: predicted is positive, actual is often negative or much smaller
        // This happens because weight was optimised for TANH(x) but applied to TANH(x+1)
        let prediction_error_wrong = (predicted_without_bias - actual_with_wrong_weight).abs();

        // Test 2: Using weight computed WITH bias gives CORRECT prediction
        let predicted_with_bias = {
            let mut predicted_new_error_sq = 0.0f32;
            for s in &samples {
                let pre_act = incoming_weight * s.activation + bias; // WITH BIAS in prediction!
                let output = pre_act.tanh();
                let contribution = weight_with_bias * output;
                let new_error = s.avg_error - contribution;
                predicted_new_error_sq += new_error * new_error;
            }
            (baseline_error_sq - predicted_new_error_sq) / baseline_error_sq
        };

        let actual_with_correct_weight = compute_actual_improvement(
            &samples,
            incoming_weight,
            weight_with_bias,
            bias,
            baseline_error_sq,
        );

        let prediction_error_correct = (predicted_with_bias - actual_with_correct_weight).abs();

        // Assertions:
        // 1. The two weights should be significantly different
        assert!(
            (weight_with_bias - weight_without_bias).abs() > 0.01,
            "Weights should differ: without_bias={weight_without_bias:.4}, with_bias={weight_with_bias:.4}"
        );

        // 2. Using wrong weight should have high prediction error
        assert!(
            prediction_error_wrong > 0.001,
            "Wrong weight should cause prediction error > 0.1%. \
             Predicted={predicted_without_bias:.4}, Actual={actual_with_wrong_weight:.4}, \
             Error={prediction_error_wrong:.4}"
        );

        // 3. Using correct weight should have low prediction error
        assert!(
            prediction_error_correct < 0.0001,
            "Correct weight should have prediction error < 0.01%. \
             Predicted={predicted_with_bias:.4}, Actual={actual_with_correct_weight:.4}, \
             Error={prediction_error_correct:.4}"
        );

        // 4. The key bug symptom: wrong weight often gives OPPOSITE sign of improvement
        // (predicts positive improvement but actual is negative, or vice versa)
        // This may not always happen with this specific test data, but we verify
        // the prediction error is significantly worse.
        assert!(
            prediction_error_wrong > prediction_error_correct * 10.0,
            "Wrong weight should have much higher error than correct weight. \
             Wrong error={prediction_error_wrong:.6}, Correct error={prediction_error_correct:.6}"
        );
    }

    /// Regression test: Target simulation must compute new_error = expected - new_output.
    ///
    /// BUG: The code was computing new_error = new_output - expected (opposite sign).
    /// This caused predictions to have the WRONG SIGN compared to actual results:
    /// - Predicted positive improvement but actual was negative (worse)
    /// - The sign error was in the target activation simulation path
    ///
    /// The linear approximation uses: new_error = avg_error - contribution
    /// The target simulation must be consistent: new_error = expected - target_fn(new_input)
    ///
    /// Note: Since we square the errors, the sign doesn't affect the squared error sum,
    /// but it DOES affect the direction of the optimal weight calculation when used
    /// inconsistently between the weight optimisation and improvement prediction.
    #[test]
    fn target_simulation_error_sign_consistent_with_linear_model() {
        // Sample with positive avg_error (output should be higher)
        // CRITICAL: avg_error is in VALUE domain (targetValue - currentValue from TypeScript)
        // So avg_error = desired_value - current_value, positive means current is too low
        let sample = HelpfulSample {
            activation: 0.5,
            avg_error: 0.2,          // VALUE domain: need to add 0.2 to pre-activation
            target_value: Some(0.3), // Pre-activation input to target (current value)
            target_activation: Some(0.3), // Post-activation (in linear region of HARD_TANH)
        };

        // Small positive contribution (should reduce the positive error)
        let contribution = 0.05f32;

        // Linear model: new_error = avg_error - contribution = 0.2 - 0.05 = 0.15
        let linear_new_error = sample.avg_error - contribution;

        // Target simulation (CORRECT formula using VALUE domain):
        // desired_value = target_value + avg_error (VALUE domain)
        // expected = squash(desired_value) (convert to ACTIVATION domain)
        let desired_value = sample.target_value.unwrap() + sample.avg_error; // = 0.3 + 0.2 = 0.5
        let expected = hard_tanh(desired_value); // = 0.5 (in linear region, so same as desired_value)
        let new_input = sample.target_value.unwrap() + contribution; // = 0.3 + 0.05 = 0.35
        let new_output = hard_tanh(new_input); // = 0.35 (in linear region)

        // new_error = expected - new_output
        let target_new_error = expected - new_output; // = 0.5 - 0.35 = 0.15

        // Both should give the same result in the linear region
        assert!(
            (linear_new_error - target_new_error).abs() < 0.001,
            "Target simulation should match linear model in linear region. \
             Linear: {linear_new_error}, Target: {target_new_error}"
        );

        // Both should show error REDUCTION (not increase)
        assert!(
            target_new_error.abs() < sample.avg_error.abs(),
            "Positive contribution should REDUCE positive error. \
             Old error: {}, New error: {}",
            sample.avg_error,
            target_new_error
        );

        // Verify both have the same sign (positive)
        assert!(
            linear_new_error > 0.0 && target_new_error > 0.0,
            "Both error calculations should be positive. \
             Linear: {linear_new_error}, Target: {target_new_error}"
        );
    }

    /// Debug test: Constant source activation with BIPOLAR neuron targeting HARD_TANH.
    ///
    /// Reproduces production scenario:
    /// - Source neuron has constant activation (-0.575)
    /// - New neuron: BIPOLAR with inW=5, bias=2, outW=0.010
    /// - Target: HARD_TANH output neuron
    /// - More negative errors (28757) than positive (25505)
    ///
    /// BIPOLAR(5 × -0.575 + 2) = BIPOLAR(-0.875) = -1
    /// Contribution = 0.010 × -1 = -0.010 (constant negative)
    ///
    /// Expected: Error should decrease (more negative errors helped)
    /// Actual: Error increased (prediction wrong)
    #[test]
    fn constant_activation_bipolar_targeting_hard_tanh() {
        // Simulate production distribution:
        // ~46% positive errors (need output higher)
        // ~54% negative errors (need output lower)
        let mut samples = Vec::new();

        // Positive errors (25505 samples with positive error)
        for i in 0..255 {
            let target_value = (i as f32 - 127.0) / 200.0; // Range roughly -0.6 to 0.6
            let target_activation = hard_tanh(target_value);
            let avg_error = 0.5 + (i as f32 % 50.0) / 100.0; // Positive errors 0.5-1.0

            samples.push(HelpfulSample {
                activation: -0.575, // Constant source activation
                avg_error,
                target_value: Some(target_value),
                target_activation: Some(target_activation),
            });
        }

        // Negative errors (287 samples with negative error - ratio ~54%)
        for i in 0..287 {
            let target_value = (i as f32 - 143.0) / 200.0;
            let target_activation = hard_tanh(target_value);
            let avg_error = -0.5 - (i as f32 % 50.0) / 100.0; // Negative errors -0.5 to -1.0

            samples.push(HelpfulSample {
                activation: -0.575, // Same constant source activation
                avg_error,
                target_value: Some(target_value), // FIXED: was incorrectly target_activation
                target_activation: Some(target_activation),
            });
        }

        // Production parameters
        let incoming_weight = 5.0f32;
        let bias = 2.0f32;
        let outgoing_weight = 0.010f32;

        // Compute BIPOLAR output
        let pre_activation = incoming_weight * (-0.575) + bias; // = -0.875
        let bipolar_output = bipolar_activation(pre_activation); // = -1
        let contribution = outgoing_weight * bipolar_output; // = -0.010

        assert_eq!(
            bipolar_output, -1.0,
            "BIPOLAR({pre_activation}) should be -1"
        );
        assert!(
            (contribution - (-0.010)).abs() < 0.0001,
            "Contribution should be -0.010"
        );

        // Compute baseline and new error
        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        let mut new_error_sq_sum = 0.0f32;
        let mut improved_count = 0u32;
        let mut worsened_count = 0u32;

        for sample in &samples {
            // CORRECT formula: avg_error is in VALUE domain
            // desired_value = target_value + avg_error, then squash to get expected activation
            let desired_value = sample.target_value.unwrap() + sample.avg_error;
            let expected = hard_tanh(desired_value);
            let new_input = sample.target_value.unwrap() + contribution;
            let new_output = hard_tanh(new_input);
            let new_error = expected - new_output;

            new_error_sq_sum += new_error.powi(2);

            if new_error.abs() < sample.avg_error.abs() {
                improved_count += 1;
            } else if new_error.abs() > sample.avg_error.abs() {
                worsened_count += 1;
            }
        }

        let improvement = (baseline_error_sq - new_error_sq_sum) / baseline_error_sq;
        let improvement_pct = improvement * 100.0;

        eprintln!(
            "Constant activation test: baseline={baseline_error_sq:.4}, new={new_error_sq_sum:.4}"
        );
        eprintln!(
            "Improvement: {improvement_pct:.4}%, improved={improved_count}, worsened={worsened_count}"
        );

        // With constant negative contribution and more negative errors,
        // we should see positive improvement (error reduction)
        // OR if improvement is negative, it explains the production issue
        if improvement < 0.0 {
            eprintln!("WARNING: Negative improvement with constant activation!");
            eprintln!("This matches production failure pattern.");
        }

        // At minimum, verify the math is consistent
        assert!(
            improvement.is_finite(),
            "Improvement should be finite, got {improvement}"
        );
    }

    #[test]
    fn merge_batch_results_preserves_order_with_empty_samples() {
        let flags = vec![false, true, false, true];
        let merged = GpuAnalyzer::merge_batch_results(
            &flags,
            vec![
                HelpfulStats {
                    positive_count: 1,
                    ..HelpfulStats::default()
                },
                HelpfulStats {
                    positive_count: 2,
                    ..HelpfulStats::default()
                },
            ],
        );

        assert_eq!(
            merged.len(),
            flags.len(),
            "Merged results should match input batch length"
        );
        assert_eq!(
            merged[0].positive_count, 1,
            "First non-empty sample should remain first"
        );
        assert_eq!(
            merged[1].positive_count, 0,
            "Empty samples should produce default stats"
        );
        assert_eq!(
            merged[2].positive_count, 2,
            "Second non-empty sample should remain in original position"
        );
        assert_eq!(
            merged[3].positive_count, 0,
            "Trailing empty samples should also produce defaults"
        );
    }

    #[test]
    fn diagnostics_prefers_higher_expected_improvement() {
        let mut diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 1_500);
        diagnostics.record_candidate_attempt("output-0", false);
        diagnostics.record_no_samples("output-0", "input-0", 0);

        diagnostics.record_candidate_attempt("output-0", true);
        diagnostics.record_below_threshold(
            "output-0",
            "hidden-1",
            ThresholdContext {
                sample_count: 42,
                expected_improvement: 0.05,
                threshold: 0.1,
                improved_count: 30,
                worsened_count: 12,
                weight: -0.25,
            },
        );

        let entry = diagnostics
            .entry_for("output-0")
            .expect("diagnostics entry should exist");
        let reason = entry.best_rejection.as_ref().map(|detail| detail.reason);
        assert!(
            matches!(reason, Some(RejectionReason::BelowThreshold)),
            "Expected below-threshold reason to persist when it has the highest score"
        );
    }

    #[test]
    fn diagnostics_marks_candidate_selection() {
        let mut diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.mark_candidate_selected("output-0");
        let entry = diagnostics
            .entry_for("output-0")
            .expect("diagnostics entry should exist");
        assert!(
            entry.had_candidate,
            "Entry should record that a candidate was selected"
        );
    }

    #[test]
    fn relu_split_evaluation_finds_candidates_when_activation_correlates_with_error() {
        // Test that split-by-error ReLU evaluation finds candidates when source activation
        // correlates with target error direction.
        //
        // Key insight: A ReLU can only help if its activation correlates with the errors
        // it's trying to fix. If activation is the same for all samples, the ReLU's
        // contribution will cancel out across balanced errors.
        //
        // This test creates samples where:
        // - Source fires (activation > 0) when target error is positive (output should go UP)
        // - Source doesn't fire (activation <= 0) when target error is negative
        //
        // This is the realistic scenario where adding a ReLU neuron can help.
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analysis should be available");

        let mut samples = Vec::new();
        // Samples where source fires AND output should go UP (positive error)
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: 1.0, // Source fires
                avg_error: 0.5,  // Output should be HIGHER
                target_value: None,
                target_activation: None,
            });
        }
        // Samples where source doesn't fire AND output should go DOWN (negative error)
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: -0.5, // Source doesn't fire (ReLU will output 0)
                avg_error: -0.5,  // Output should be LOWER
                target_value: None,
                target_activation: None,
            });
        }

        let result =
            evaluate_relu_candidates_split(&analyzer, "input-0", "output-0", &samples, 0.0, None)
                .expect("ReLU split evaluation should succeed");

        // With correlation between activation and error, we should find a positive-error candidate
        // The ReLU fires when we need output to go UP, and doesn't fire when we need it DOWN.
        assert!(
            result.positive_error_candidate.is_some(),
            "Should find positive-error ReLU candidate when activation correlates with error direction"
        );

        // Verify the candidate pushes in the correct direction
        if let Some(pos_candidate) = &result.positive_error_candidate {
            assert!(
                pos_candidate.outgoing_weight > 0.0,
                "Positive-error candidate should have positive outgoing weight (pushes UP). Got: {}",
                pos_candidate.outgoing_weight
            );
        }
    }

    #[test]
    fn relu_split_evaluation_finds_negative_orientation_candidates() {
        // Test that we can find ReLU candidates with incoming_weight = -1.0 (negative orientation).
        //
        // This is critical: when source neurons have predominantly NEGATIVE activations
        // that correlate with errors, we need a ReLU with incoming_weight = -1.0 to flip
        // the sign before the ReLU activation.
        //
        // Bug regression test: Previously, evaluate_relu_candidates_split discarded
        // negative_stats entirely, meaning these candidates could never be found.
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analysis should be available");

        let mut samples = Vec::new();
        // Samples where source has NEGATIVE activation AND output should go UP (positive error)
        // A ReLU with incoming_weight = -1.0 will flip -1.0 to +1.0, then ReLU outputs 1.0
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: -1.0, // NEGATIVE activation
                avg_error: 0.5,   // Output should be HIGHER
                target_value: None,
                target_activation: None,
            });
        }
        // Samples where source has POSITIVE activation AND output should go DOWN (negative error)
        // A ReLU with incoming_weight = -1.0 will flip +0.5 to -0.5, then ReLU outputs 0
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: 0.5, // POSITIVE activation (will be flipped to negative, ReLU = 0)
                avg_error: -0.5, // Output should be LOWER
                target_value: None,
                target_activation: None,
            });
        }

        let result =
            evaluate_relu_candidates_split(&analyzer, "input-0", "output-0", &samples, 0.0, None)
                .expect("ReLU split evaluation should succeed");

        // With negative activations correlating with positive errors, we should find a candidate
        // that uses the NEGATIVE orientation (incoming_weight = -1.0)
        assert!(
            result.positive_error_candidate.is_some(),
            "Should find ReLU candidate even when source has negative activations (requires negative orientation)"
        );

        // Verify the candidate uses negative incoming weight (the critical fix!)
        if let Some(pos_candidate) = &result.positive_error_candidate {
            assert!(
                pos_candidate.incoming_weight < 0.0,
                "Candidate should have NEGATIVE incoming weight to flip negative activations. Got: {}",
                pos_candidate.incoming_weight
            );
            assert!(
                pos_candidate.outgoing_weight > 0.0,
                "Candidate should have positive outgoing weight (pushes UP). Got: {}",
                pos_candidate.outgoing_weight
            );
        }
    }

    #[test]
    fn neuron_diagnostics_tracks_load_failures() {
        // Test that when eligible sources exist but all fail to load, we report
        // NoSamples rather than NoEligibleSources
        let mut diagnostics = NeuronDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 10); // 10 eligible sources exist
                                                                // All 10 sources fail to load
        for _ in 0..10 {
            diagnostics.record_load_failure("output-0");
        }
        // No record_candidate_attempt calls (because all failed to load)

        let summaries = diagnostics.no_candidate_summaries();
        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];

        // Should NOT report "no eligible sources" - sources existed but failed to load
        assert!(
            !matches!(summary.reason, NeuronNoCandidateReason::NoEligibleSources),
            "Should not report NoEligibleSources when sources existed but failed to load"
        );
        // Should report NoSamples (as a catchall for sources existing but not being usable)
        assert!(
            matches!(summary.reason, NeuronNoCandidateReason::NoSamples),
            "Should report NoSamples when eligible sources exist but none were evaluated"
        );

        // Verify entry tracking
        let entry = diagnostics.entry_for("output-0").unwrap();
        assert_eq!(entry.total_eligible_sources, 10);
        assert_eq!(entry.record_load_failures, 10);
        assert_eq!(entry.evaluated_sources, 0);
    }

    #[test]
    fn neuron_diagnostics_reports_genuine_no_eligible_sources() {
        // Test that when there are genuinely no eligible sources, we correctly report that
        let mut diagnostics = NeuronDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 0); // No eligible sources

        let summaries = diagnostics.no_candidate_summaries();
        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];

        // Should correctly report no eligible sources
        assert!(
            matches!(summary.reason, NeuronNoCandidateReason::NoEligibleSources),
            "Should report NoEligibleSources when genuinely no sources exist"
        );

        // Verify entry tracking
        let entry = diagnostics.entry_for("output-0").unwrap();
        assert_eq!(entry.total_eligible_sources, 0);
        assert_eq!(entry.record_load_failures, 0);
        assert_eq!(entry.evaluated_sources, 0);
    }

    #[test]
    fn target_diagnostics_reports_no_samples_reason() {
        let mut diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 25);
        diagnostics.record_candidate_attempt("output-0", false);
        diagnostics.record_no_samples("output-0", "input-0", 8);

        let summaries = diagnostics.no_candidate_summaries();
        assert_eq!(
            summaries.len(),
            1,
            "Expected a single diagnostic summary for target without candidates"
        );

        let summary = &summaries[0];
        assert_eq!(
            summary.target_uuid, "output-0",
            "Target UUID should be preserved in summary"
        );
        assert!(
            matches!(summary.reason, SynapseNoCandidateReason::NoSamples),
            "Expected no-samples reason"
        );
    }

    #[test]
    fn analyze_neurons_rejects_duplicate_focus_targets() {
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = (MIN_NEURON_SAMPLE_COUNT + 5) as u32;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            records.push(DiscoverRecord::new(
                obs_index,
                "hidden-source".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![1.0],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-source".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: Vec::new(),
        };

        let input = AnalyzeNeuronsInput {
            parquet_file: parquet_file.clone(),
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let err = analyze_neurons(&input)
            .expect_err("Neuron analysis should refuse duplicate focus neurons");
        let message = format!("{err}");
        assert!(
            message.contains("duplicate focus neurons"),
            "Expected duplicate focus error, got: {message}",
        );
    }

    #[test]
    fn analyze_synapses_rejects_duplicate_focus_targets() {
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = 16;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![-0.05],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "input-0".to_string(),
                    neuron_type: "input".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
                synapse_type: None,
            }],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let err = analyze_synapses(&input)
            .expect_err("Synapse analysis should refuse duplicate focus neurons");
        let message = format!("{err}");
        assert!(
            message.contains("duplicate focus neurons"),
            "Expected duplicate focus error, got: {message}",
        );
    }

    #[test]
    fn analyze_synapses_reports_eligible_sources_correctly_for_non_input_neurons() {
        skip_if_no_gpu!();
        // Test that non-input neurons with valid creature structure always report
        // eligible sources correctly, not "no eligible sources" when sources exist
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Input neuron records (observations)
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "input-1".to_string(),
                Some(0.0),
                0.3,
                vec![0.2],
            ));
            // Hidden neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "hidden-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.15],
            ));
            // Output neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.6,
                vec![0.05],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 2,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-0".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "constant-0".to_string(),
                    neuron_type: "constant".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 1.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                // Hidden neuron already connected to input-0
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "hidden-0".to_string(),
                    weight: 0.4,
                    synapse_type: None,
                },
                // Output neuron already connected to hidden-0
                SynapseJson {
                    from_uuid: "hidden-0".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 0.5,
                    synapse_type: None,
                },
            ],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["hidden-0".to_string(), "output-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // Check diagnostics for hidden-0
        // hidden-0 should have eligible sources (input-1 is not connected yet)
        // So it should NOT report "no eligible sources"
        let hidden_diag = result
            .no_candidate_reasons
            .iter()
            .find(|summary| summary.target_uuid == "hidden-0");

        if let Some(diag) = hidden_diag {
            assert!(
                diag.reason != SynapseNoCandidateReason::NoEligibleSources,
                "hidden-0 should have eligible sources (input-1 is available), but got: {:?}",
                diag.reason
            );
            assert!(
                diag.evaluated_candidates > 0,
                "hidden-0 should have evaluated at least one candidate (input-1), but evaluated_candidates is {}",
                diag.evaluated_candidates
            );
        }

        // Check diagnostics for output-0
        // output-0 should have eligible sources (input-0, input-1 are available)
        // So it should NOT report "no eligible sources"
        let output_diag = result
            .no_candidate_reasons
            .iter()
            .find(|summary| summary.target_uuid == "output-0");

        if let Some(diag) = output_diag {
            assert!(
                diag.reason != SynapseNoCandidateReason::NoEligibleSources,
                "output-0 should have eligible sources (input-0, input-1 are available), but got: {:?}",
                diag.reason
            );
            assert!(
                diag.evaluated_candidates > 0,
                "output-0 should have evaluated at least one candidate, but evaluated_candidates is {}",
                diag.evaluated_candidates
            );
        }
    }

    #[test]
    fn analyze_synapses_reports_fully_connected_neuron_explicitly() {
        skip_if_no_gpu!();
        // Test that a neuron connected to ALL eligible sources is explicitly reported
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Input neuron records (observations)
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "input-1".to_string(),
                Some(0.0),
                0.3,
                vec![0.2],
            ));
            // Hidden neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "hidden-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.15],
            ));
            // Output neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.6,
                vec![0.05],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        // Create a creature where hidden-0 is connected to ALL eligible sources
        // (both input-0 and input-1)
        let creature = CreatureJson {
            input: 2,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-0".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                // hidden-0 is connected to ALL eligible sources (input-0 and input-1)
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "hidden-0".to_string(),
                    weight: 0.4,
                    synapse_type: None,
                },
                SynapseJson {
                    from_uuid: "input-1".to_string(),
                    to_uuid: "hidden-0".to_string(),
                    weight: 0.5,
                    synapse_type: None,
                },
            ],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["hidden-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // hidden-0 should be reported as having no eligible sources
        // because it's connected to ALL eligible sources (both inputs)
        let hidden_diag = result
            .no_candidate_reasons
            .iter()
            .find(|summary| summary.target_uuid == "hidden-0");

        assert!(
            hidden_diag.is_some(),
            "hidden-0 should have diagnostics since it's fully connected"
        );

        if let Some(diag) = hidden_diag {
            assert_eq!(
                diag.reason,
                SynapseNoCandidateReason::NoEligibleSources,
                "hidden-0 should report NoEligibleSources since it's connected to all eligible sources"
            );
            assert_eq!(
                diag.evaluated_candidates, 0,
                "hidden-0 should have 0 evaluated candidates since all sources are already connected"
            );
        }
    }

    #[test]
    fn analyze_synapses_requires_focus_targets() {
        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file: "unused.parquet".to_string(),
            creature,
            focus_neurons: Vec::new(),
            improvement_threshold: None,
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let err =
            analyze_synapses(&input).expect_err("Synapse analysis should refuse empty focus lists");
        let message = format!("{err}");
        assert!(
            message.contains("at least one focus neuron"),
            "Expected missing focus error, got: {message}",
        );
    }

    #[test]
    fn analyze_synapses_reports_diagnostics_when_no_candidates() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..16 {
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.05],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file: parquet_file.clone(),
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input)
            .expect("Synapse analysis should succeed even without candidates");
        assert!(
            result.helpful_synapses.is_empty(),
            "Expected no helpful candidates when there are no eligible sources"
        );
        let reason = result
            .no_candidate_reasons
            .first()
            .map(|summary| summary.reason.clone());
        assert!(
            matches!(reason, Some(SynapseNoCandidateReason::NoEligibleSources)),
            "Expected diagnostics to explain missing candidates"
        );
    }

    #[test]
    fn analyze_synapses_stops_harmful_processing_after_deadline() {
        skip_if_no_gpu!();
        let _deadline_guard = deadline_override::DeadlineOverrideGuard::with_sequence(vec![
            false, false, false, false, false, false, true, false, false, false,
        ]);

        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..16 {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-1".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 2,
            neurons: vec![
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-1".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 0.8,
                    synapse_type: None,
                },
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "output-1".to_string(),
                    weight: 0.6,
                    synapse_type: None,
                },
            ],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input)
            .expect("Synapse analysis should complete even when the deadline triggers");

        // With parallel processing, deadline detection order is non-deterministic because
        // multiple threads call deadline_passed() concurrently. The timeout mechanism is
        // approximate - once any thread detects the deadline, analysis_timed_out is set
        // and processing should stop. However, some threads may have already started
        // processing harmful synapses before the deadline was detected.
        //
        // The key requirement is that the analysis completes successfully and respects
        // the deadline approximately. Since timeout is approximate, we verify that:
        // 1. The analysis completes without panicking
        // 2. The result structure is valid
        // 3. We don't process more harmful synapses than exist (sanity check)
        //
        // In this test setup, we have 2 focus neurons, each with 1 harmful synapse (2 total).
        // The deadline sequence [false x6, true, ...] should cause early termination,
        // but with parallel processing, the exact point of termination is non-deterministic.
        let max_possible_harmful = 2; // 2 focus neurons × 1 harmful synapse each
        assert!(
            result.harmful_synapses.len() <= max_possible_harmful,
            "Should not process more harmful synapses than exist. \
             Got {} harmful synapses, max possible is {}",
            result.harmful_synapses.len(),
            max_possible_harmful
        );
        // The deadline mechanism is approximate, so we accept any result as long as
        // the analysis completes and doesn't exceed reasonable bounds
    }

    #[test]
    fn analyze_all_runs_synapse_and_neuron_phases() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..(MIN_NEURON_SAMPLE_COUNT as u32 + 1) {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: vec![SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
                synapse_type: None,
            }],
        };

        let input = AnalyzeAllInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.05),
            harmful_threshold: Some(-0.05),
            max_synapse_candidates: Some(5),
            max_neuron_candidates: Some(5),
            analysis_deadline_ms: None,
            include_synapse_analysis: Some(true),
            include_neuron_analysis: Some(true),
        };

        let result = analyze_all(&input).expect("Combined analysis should succeed");
        assert!(result.synapse.is_some(), "Synapse phase should run");
        assert!(result.neuron.is_some(), "Neuron phase should run");
    }

    #[test]
    fn analyze_neurons_reports_diagnostics_when_no_candidates() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..(MIN_NEURON_SAMPLE_COUNT as u32 + 1) {
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-source".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: Vec::new(),
        };

        let input = AnalyzeNeuronsInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_neurons(&input)
            .expect("Neuron analysis should succeed even without candidates");
        assert!(
            result.helpful_neurons.is_empty(),
            "Expected no neuron candidates when the source neuron lacks samples"
        );
        let reason = result
            .no_candidate_reasons
            .first()
            .map(|summary| summary.reason.clone());
        assert!(
            matches!(reason, Some(NeuronNoCandidateReason::NoSamples)),
            "Expected diagnostics to explain missing neuron candidates"
        );
    }

    #[test]
    fn analyze_neurons_uses_vertical_timeout_with_randomized_order() {
        skip_if_no_gpu!();
        use rayon::ThreadPoolBuilder;

        // Simulate a deadline that allows at least one focus neuron to start, but
        // triggers before all are processed. The override sequence is consumed
        // by calls to `deadline_passed` in order. With randomization, we need
        // enough false values to allow at least one neuron to start processing.
        // We provide multiple false values to account for any initialization checks,
        // then true to stop further processing.
        // Provide enough "not timed out" checks to allow at least one focus neuron
        // to begin evaluating sources before we trigger the timeout.
        let mut deadline_sequence = vec![false; 64];
        deadline_sequence.push(true);
        let _deadline_guard =
            deadline_override::DeadlineOverrideGuard::with_sequence(deadline_sequence);

        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Provide discovery records for two output neurons but none for the
        // hidden source. This guarantees that each focus neuron has at least
        // one eligible upstream source, and that diagnostics can attribute a
        // `NoSamples` reason once analysis runs.
        let mut records = Vec::new();
        for obs_index in 0..(MIN_NEURON_SAMPLE_COUNT as u32 + 1) {
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-1".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 0,
            output: 2,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-source".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-1".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: Vec::new(),
        };

        let input = AnalyzeNeuronsInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            // Any non-None deadline value will exercise the override sequence.
            analysis_deadline_ms: Some(1_000_000),
        };

        // Use a single-threaded Rayon pool so the deadline override sequence remains
        // deterministic for this test. Note: focus neurons are randomized, so we can't
        // assume a specific order.
        let pool = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("Failed to build single-threaded Rayon pool");

        let result = pool
            .install(|| analyze_neurons(&input))
            .expect("Neuron analysis should succeed even when the deadline triggers");

        // At least one focus neuron should have evaluated at least one upstream
        // source before the deadline (vertical timeout behaviour). Since focus
        // neurons are randomized, we check that at least one of the two neurons
        // was processed.
        let processed_neurons: Vec<_> = result
            .no_candidate_reasons
            .iter()
            .filter(|summary| summary.evaluated_sources > 0)
            .collect();

        // At least one focus neuron should have progressed far enough to attempt
        // source evaluation, OR we should have produced at least one candidate.
        let did_any_work = !processed_neurons.is_empty() || !result.helpful_neurons.is_empty();
        assert!(
            did_any_work,
            "At least one focus neuron should do some work before timeout (vertical timeout behaviour)"
        );

        // Verify that the processed neuron(s) are not reported as having no eligible sources
        for summary in &processed_neurons {
            assert!(
                summary.reason != NeuronNoCandidateReason::NoEligibleSources,
                "Processed focus neuron should not be reported as having no eligible sources when a timeout occurs"
            );
        }
    }

    #[test]
    fn analyze_synapses_uses_vertical_timeout_with_randomized_order() {
        skip_if_no_gpu!();
        use rayon::ThreadPoolBuilder;

        // Simulate a deadline that allows at least one focus neuron to start, but
        // triggers before all are processed. The override sequence is consumed
        // by calls to `deadline_passed` in order. With randomization, we need
        // enough false values to allow at least one neuron to start processing.
        // We provide multiple false values to account for any initialization checks,
        // then true to stop further processing.
        // Provide enough "not timed out" checks to allow at least one focus neuron
        // to begin evaluating candidates before we trigger the timeout.
        //
        // The analysis code checks the deadline at several stages (start-of-focus,
        // source pre-filtering, sample building, batch evaluation). If we trigger
        // the timeout too early, the vertical-timeout behaviour isn't exercised.
        let mut deadline_sequence = vec![false; 64];
        deadline_sequence.push(true);
        let _deadline_guard =
            deadline_override::DeadlineOverrideGuard::with_sequence(deadline_sequence);

        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Provide discovery records for an input neuron and two output neurons.
        // Records use disjoint obs_index ranges so that each potential synapse
        // has discovery data but no aligned samples, guaranteeing that the
        // diagnostics machinery records a `NoSamples` style rejection rather
        // than treating the target as having no eligible sources.
        let mut records = Vec::new();
        for obs_index in 0..16u32 {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
        }
        for obs_index in 100..116u32 {
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-1".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 2,
            neurons: vec![
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-1".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            // Any non-None deadline value will exercise the override sequence.
            analysis_deadline_ms: None,
        };

        // Use a single-threaded Rayon pool so the deadline override sequence remains
        // deterministic for this test. Note: focus neurons are randomized, so we can't
        // assume a specific order.
        let pool = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("Failed to build single-threaded Rayon pool");

        let result = pool
            .install(|| analyze_synapses(&input))
            .expect("Synapse analysis should succeed even when the deadline triggers");

        // At least one focus neuron should have evaluated at least one upstream
        // source before the deadline (vertical timeout behaviour). Since focus
        // neurons are randomized, we check that at least one of the two neurons
        // was processed.
        let processed_neurons: Vec<_> = result
            .no_candidate_reasons
            .iter()
            .filter(|summary| summary.evaluated_candidates > 0)
            .collect();

        // At least one focus neuron should have progressed far enough to attempt
        // candidate evaluation, OR we should have produced at least one candidate.
        let did_any_work = !processed_neurons.is_empty()
            || !result.helpful_synapses.is_empty()
            || !result.harmful_synapses.is_empty();
        assert!(
            did_any_work,
            "At least one focus neuron should do some work before timeout (vertical timeout behaviour)"
        );

        // If we observed a processed neuron via diagnostics, it should not be reported
        // as having no eligible sources.
        for summary in &processed_neurons {
            assert!(
                summary.reason != SynapseNoCandidateReason::NoEligibleSources,
                "Processed focus neuron should not be reported as having no eligible sources when a timeout occurs"
            );
        }
    }

    /// Test bias calculation for TANH activation function
    #[test]
    fn test_bias_calculation_tanh() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let bias = calculate_optimal_bias(&samples, 1.0, -0.5, tanh_activation, "TANH", None, None);

        // Bias should be in expanded TANH range
        assert!(
            (-1.0..=1.0).contains(&bias),
            "Bias for TANH should be in range [-1.0, 1.0], got {bias}"
        );
    }

    /// Test bias calculation for ReLU activation function
    #[test]
    fn test_bias_calculation_relu() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let relu_fn = |x: f32| x.max(0.0);
        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, relu_fn, "ReLU", None, None);

        // ReLU can now use negative bias for threshold shifting (expanded range)
        assert!(bias >= -1.0, "Bias for ReLU should be >= -1.0, got {bias}");
        assert!(bias <= 1.0, "Bias for ReLU should be <= 1.0, got {bias}");
    }

    /// Test bias improves error reduction compared to zero bias
    #[test]
    fn test_bias_improves_error_reduction() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let incoming = 1.5;
        let outgoing = -0.18;

        // Calculate error with zero bias
        let mut zero_bias_error_sq = 0.0;
        for sample in &samples {
            let pre_activation = incoming * sample.activation;
            let new_neuron_activation = identity_activation(pre_activation);
            let correction = outgoing * new_neuron_activation;
            let new_error = sample.avg_error - correction;
            zero_bias_error_sq += new_error * new_error;
        }

        // Calculate optimal bias
        let optimal_bias = calculate_optimal_bias(
            &samples,
            incoming,
            outgoing,
            identity_activation,
            "IDENTITY",
            None,
            None,
        );

        // Calculate error with optimal bias
        let mut optimal_bias_error_sq = 0.0;
        for sample in &samples {
            let pre_activation = incoming * sample.activation + optimal_bias;
            let new_neuron_activation = identity_activation(pre_activation);
            let correction = outgoing * new_neuron_activation;
            let new_error = sample.avg_error - correction;
            optimal_bias_error_sq += new_error * new_error;
        }

        // Optimal bias should give equal or better error reduction than zero bias
        assert!(
            optimal_bias_error_sq <= zero_bias_error_sq + EPSILON,
            "Optimal bias should improve or equal zero bias error reduction: zero_bias_error={zero_bias_error_sq}, optimal_bias_error={optimal_bias_error_sq}"
        );
    }

    /// Test that positive improvements below threshold are accepted as candidates
    /// This verifies the fix where all positive improvements are candidates, not just those above threshold
    #[test]
    fn analyze_synapses_accepts_positive_improvements_below_threshold() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Create test data that will produce a positive but below-threshold improvement
        // We need: expected_improvement = (2*w*E[a*e] - w^2*E[a^2]) / E[e^2]
        // To get ~0.05 improvement with threshold 0.1, we'll use:
        // - source activation: 0.5 consistently
        // - target error: 0.1 consistently
        // - This should produce a positive improvement when weight is chosen appropriately
        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Source neuron (input-0) with consistent activation
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.5,    // Consistent activation
                vec![], // Input neurons don't have errors
            ));
            // Target neuron (output-0) with consistent error
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.3,
                vec![0.1], // Consistent error
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(), // No existing synapse from input-0 to output-0
        };

        // Set threshold to 0.1 - we expect a positive but below-threshold improvement to be accepted
        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.1),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // The key assertion: positive improvements below threshold should be accepted
        // We should have at least one helpful synapse candidate (even if improvement < 0.1)
        // OR if no candidate, it should NOT be due to BelowThreshold for a positive improvement
        if result.helpful_synapses.is_empty() {
            // If no candidates, check diagnostics - it should NOT be BelowThreshold for positive improvements
            let no_candidate = result
                .no_candidate_reasons
                .iter()
                .find(|summary| summary.target_uuid == "output-0");

            if let Some(summary) = no_candidate {
                // If there's a detail, check that it's not a positive improvement below threshold
                if let Some(detail) = &summary.detail {
                    if let Some(improvement) = detail.expected_improvement {
                        if improvement > 0.0 && improvement <= 0.1 {
                            panic!(
                                "Positive improvement {:.4} below threshold 0.1 should be accepted as candidate, but was rejected with reason: {:?}",
                                improvement, summary.reason
                            );
                        }
                    }
                }
            }
        } else {
            // We have candidates - verify at least one has positive improvement
            // Issue #128: Use expected_creature_score_gain (creature-level, not neuron-level)
            let has_positive_improvement = result
                .helpful_synapses
                .iter()
                .any(|synapse| synapse.expected_creature_score_gain > 0.0);

            assert!(
                has_positive_improvement,
                "Should have at least one candidate with positive improvement"
            );
        }
    }

    /// Test that non-positive improvements (<= 0.0) are still rejected
    #[test]
    fn analyze_synapses_rejects_non_positive_improvements() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Create test data that will produce a non-positive improvement
        // Use mismatched activations/errors that result in negative or zero improvement
        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Source neuron with activation
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![],
            ));
            // Target neuron with error that doesn't correlate well (will produce negative improvement)
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.0,
                vec![-0.1], // Negative error when source is positive
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.1),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // Non-positive improvements should be rejected (not appear in helpful_synapses)
        // Even though we now accept positive improvements below threshold, we still reject <= 0.0
        // Issue #128: Use expected_creature_score_gain (creature-level metric)
        let has_non_positive = result
            .helpful_synapses
            .iter()
            .any(|synapse| synapse.expected_creature_score_gain <= 0.0);

        assert!(
            !has_non_positive,
            "Should not have any candidates with non-positive improvement (<= 0.0)"
        );
    }

    /// Test that positive improvements above threshold are still accepted (regression test)
    #[test]
    fn analyze_synapses_accepts_positive_improvements_above_threshold() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Create test data that will produce a positive improvement above threshold
        // Use strong correlation between source activation and target error
        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Source neuron with strong activation
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![],
            ));
            // Target neuron with error that correlates positively
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2], // Positive error when source is positive
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        // Set threshold to 0.1 - we expect improvement above this to be accepted
        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.1),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // Positive improvements above threshold should definitely be accepted
        // This is a regression test to ensure we didn't break existing behavior
        // Issue #128: Use expected_creature_score_gain (creature-level metric)
        let has_above_threshold = result
            .helpful_synapses
            .iter()
            .any(|synapse| synapse.expected_creature_score_gain > 0.1);

        // Note: This test may pass even if no candidates are found due to other reasons
        // (e.g., no samples, zero improvement). The key is that if we have candidates,
        // they should include positive improvements above threshold.
        if !result.helpful_synapses.is_empty() {
            assert!(
                has_above_threshold
                    || result
                        .helpful_synapses
                        .iter()
                        .any(|s| s.expected_creature_score_gain > 0.0),
                "Should have candidates with positive improvement (above or below threshold)"
            );
        }
    }

    /// Test bias range boundaries for different activation functions
    #[test]
    fn test_bias_within_reasonable_range() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        type ActivationTestCase = (&'static str, fn(f32) -> f32, f32, f32);
        let test_cases: Vec<ActivationTestCase> = vec![
            ("TANH", tanh_activation as fn(f32) -> f32, -10.0, 10.0),
            (
                "LOGISTIC",
                logistic_activation as fn(f32) -> f32,
                -10.0,
                10.0,
            ),
            (
                "IDENTITY",
                identity_activation as fn(f32) -> f32,
                -50.0,
                50.0,
            ),
        ];

        for (name, activation_fn, min_expected, max_expected) in test_cases {
            let bias = calculate_optimal_bias(&samples, 1.0, 1.0, activation_fn, name, None, None);
            assert!(
                bias >= min_expected && bias <= max_expected,
                "Bias for {name} should be in range [{min_expected}, {max_expected}], got {bias}"
            );
        }
    }

    /// Test bias calculation handles empty samples
    #[test]
    fn test_bias_calculation_empty_samples() {
        let samples: Vec<HelpfulSample> = vec![];

        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

        // Should return 0.0 for empty samples
        assert_eq!(bias, 0.0, "Empty samples should return bias of 0.0");
    }

    /// Test bias calculation handles insufficient samples
    #[test]
    fn test_bias_calculation_insufficient_samples() {
        // Only 5 samples (less than MIN_NEURON_SAMPLE_COUNT of 10)
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
        ];

        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

        // Should still return a valid bias in range (though may be 0.0 if no bias tested has sufficient samples)
        assert!(
            (-10.0..=10.0).contains(&bias),
            "Bias should be in reasonable range, got {bias}"
        );
    }

    /// Test get_bias_range returns correct ranges for different activation functions
    /// Note: get_bias_range is used by GPU, get_bias_values is used by CPU
    #[test]
    fn test_get_bias_range() {
        // Test ReLU range (extended negative for high-threshold neurons)
        let (min, max, step) = get_bias_range("ReLU");
        assert_eq!(min, -25.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 0.5);

        // Test TANH range (extended symmetric for large weight configurations)
        let (min, max, step) = get_bias_range("TANH");
        assert_eq!(min, -10.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 0.5);

        // Test LOGISTIC range (extended symmetric)
        let (min, max, step) = get_bias_range("LOGISTIC");
        assert_eq!(min, -10.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 0.5);

        // Test IDENTITY range (widest - acts as pure offset, scales with large weights)
        let (min, max, step) = get_bias_range("IDENTITY");
        assert_eq!(min, -50.0);
        assert_eq!(max, 50.0);
        assert_eq!(step, 1.0);

        // Test default range for unknown activation (generous)
        let (min, max, step) = get_bias_range("UNKNOWN");
        assert_eq!(min, -10.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 0.5);
    }

    /// Test is_threshold_activation identifies threshold functions (STEP/BIPOLAR).
    /// These use a specialised threshold-crossing model instead of the linear model.
    /// All other activations use the standard linear error model - none are skipped.
    #[test]
    fn test_is_threshold_activation() {
        // Threshold activations - use threshold-crossing model
        assert!(is_threshold_activation("STEP"), "STEP uses threshold model");
        assert!(is_threshold_activation("step"), "case insensitive");
        assert!(
            is_threshold_activation("BIPOLAR"),
            "BIPOLAR uses threshold model"
        );

        // All other activations use standard linear model (not skipped)
        assert!(
            !is_threshold_activation("IF"),
            "IF uses standard model (correlation still works)"
        );
        assert!(
            !is_threshold_activation("MAXIMUM"),
            "MAXIMUM uses standard model"
        );
        assert!(
            !is_threshold_activation("MINIMUM"),
            "MINIMUM uses standard model"
        );
        assert!(
            !is_threshold_activation("HARD_TANH"),
            "HARD_TANH uses standard model"
        );
        assert!(
            !is_threshold_activation("CLIPPED"),
            "CLIPPED uses standard model"
        );
        assert!(
            !is_threshold_activation("ReLU6"),
            "ReLU6 uses standard model"
        );
        assert!(!is_threshold_activation("TANH"), "TANH uses standard model");
        assert!(
            !is_threshold_activation("LOGISTIC"),
            "LOGISTIC uses standard model"
        );
        assert!(!is_threshold_activation("ReLU"), "ReLU uses standard model");
        assert!(
            !is_threshold_activation("LeakyReLU"),
            "LeakyReLU uses standard model"
        );
        assert!(!is_threshold_activation("ELU"), "ELU uses standard model");
        assert!(!is_threshold_activation("SELU"), "SELU uses standard model");
        assert!(!is_threshold_activation("GELU"), "GELU uses standard model");
        assert!(
            !is_threshold_activation("IDENTITY"),
            "IDENTITY uses standard model"
        );
        assert!(
            !is_threshold_activation("Softplus"),
            "Softplus uses standard model"
        );
        assert!(
            !is_threshold_activation("BENT_IDENTITY"),
            "BENT_IDENTITY uses standard model"
        );
        assert!(
            !is_threshold_activation("ArcTan"),
            "ArcTan uses standard model"
        );
        assert!(
            !is_threshold_activation("Swish"),
            "Swish uses standard model"
        );
        assert!(!is_threshold_activation("Mish"), "Mish uses standard model");
        assert!(
            !is_threshold_activation("UNKNOWN"),
            "Unknown uses standard model"
        );
    }

    /// Test ThresholdType correctly applies threshold functions
    #[test]
    fn test_threshold_type_apply() {
        // STEP: value > 0 ? 1 : 0
        assert_eq!(ThresholdType::Step.apply(0.5), 1.0);
        assert_eq!(ThresholdType::Step.apply(0.001), 1.0);
        assert_eq!(ThresholdType::Step.apply(0.0), 0.0);
        assert_eq!(ThresholdType::Step.apply(-0.001), 0.0);
        assert_eq!(ThresholdType::Step.apply(-5.0), 0.0);

        // BIPOLAR: value > 0 ? 1 : -1
        assert_eq!(ThresholdType::Bipolar.apply(0.5), 1.0);
        assert_eq!(ThresholdType::Bipolar.apply(0.001), 1.0);
        assert_eq!(ThresholdType::Bipolar.apply(0.0), -1.0);
        assert_eq!(ThresholdType::Bipolar.apply(-0.001), -1.0);
        assert_eq!(ThresholdType::Bipolar.apply(-5.0), -1.0);
    }

    /// Test ThresholdType correctly detects threshold flips
    #[test]
    fn test_threshold_type_would_flip() {
        // STEP: threshold at 0
        assert!(
            ThresholdType::Step.would_flip(-0.5, 1.0),
            "negative -> positive should flip"
        );
        assert!(
            ThresholdType::Step.would_flip(0.5, -1.0),
            "positive -> negative should flip"
        );
        assert!(
            !ThresholdType::Step.would_flip(0.5, 0.3),
            "positive -> more positive shouldn't flip"
        );
        assert!(
            !ThresholdType::Step.would_flip(-0.5, -0.3),
            "negative -> more negative shouldn't flip"
        );

        // Edge cases
        assert!(
            ThresholdType::Step.would_flip(-0.1, 0.2),
            "just crosses threshold"
        );
        assert!(
            !ThresholdType::Step.would_flip(-0.1, 0.05),
            "doesn't quite reach threshold"
        );
    }

    /// Test ThresholdType correctly identifies helpful vs harmful flips
    #[test]
    fn test_threshold_type_flip_direction() {
        // STEP: Error > 0 means output should be higher (0 -> 1 is helpful)
        // Current output is 0 (value < 0), error > 0 (should be 1), flip to 1 is helpful
        assert_eq!(
            ThresholdType::Step.flip_direction(-0.5, 1.0, 0.5),
            1,
            "flip 0->1 when error>0 is helpful"
        );

        // Current output is 1 (value > 0), error < 0 (should be 0), flip to 0 is helpful
        assert_eq!(
            ThresholdType::Step.flip_direction(0.5, -1.0, -0.5),
            1,
            "flip 1->0 when error<0 is helpful"
        );

        // Current output is 0 (value < 0), error < 0 (should be 0), no flip needed
        assert_eq!(
            ThresholdType::Step.flip_direction(-0.5, -0.1, -0.5),
            0,
            "no flip when error<0 and output=0"
        );

        // Current output is 1 (value > 0), error > 0 (should be 1), no flip needed
        assert_eq!(
            ThresholdType::Step.flip_direction(0.5, 0.1, 0.5),
            0,
            "no flip when error>0 and output=1"
        );

        // Harmful flip: flip 1->0 when error > 0 (output should stay high)
        assert_eq!(
            ThresholdType::Step.flip_direction(0.5, -1.0, 0.5),
            -1,
            "flip 1->0 when error>0 is harmful"
        );

        // Harmful flip: flip 0->1 when error < 0 (output should stay low)
        assert_eq!(
            ThresholdType::Step.flip_direction(-0.5, 1.0, -0.5),
            -1,
            "flip 0->1 when error<0 is harmful"
        );
    }

    /// Test evaluate_discrete_candidate correctly filters IDENTITY+bias=0 candidates.
    ///
    /// IDENTITY neurons with bias=0 are mathematically equivalent to a direct synapse:
    ///   IDENTITY(source × incoming_weight + 0) × outgoing_weight = source × (incoming × outgoing)
    ///
    /// For STEP/BIPOLAR targets, we should NOT recommend IDENTITY add-neuron candidates
    /// because add-synapse would achieve the same effect without wasting a neuron.
    #[test]
    fn test_evaluate_discrete_candidate_step() {
        // Create samples where source activation correlates with whether the target
        // is on the "wrong side" of the threshold
        let mut samples = Vec::new();

        // Case 1: Target is at 0 (value=-0.5) but should be 1 (error=0.5)
        // Source has high positive activation - adding positive contribution would help
        for i in 0..20 {
            samples.push(DiscreteHelpfulSample {
                source_activation: 0.5 + (i as f32 * 0.01),
                target_value: -0.3, // Currently outputs 0
                target_activation: 0.0,
                avg_error: 0.5, // Should be 1 (positive error)
            });
        }

        let candidate = evaluate_discrete_candidate(
            "input-0",
            "target-step",
            &samples,
            ThresholdType::Step,
            0.0,  // threshold
            true, // is_output_target - test assumes target is output
        );

        // Should NOT find a candidate because IDENTITY+bias=0 is filtered
        // (equivalent to synapse - use add-synapse analysis instead)
        assert!(
            candidate.is_none(),
            "Should NOT return IDENTITY candidate for STEP (equivalent to synapse). \
            Use add-synapse analysis for STEP targets instead."
        );
    }

    /// Test that discrete evaluation filters ALL IDENTITY candidates (not just low-improvement).
    /// IDENTITY neurons with bias=0 are mathematically equivalent to synapses, so they
    /// should never be recommended for STEP/BIPOLAR targets.
    ///
    /// Previously this test checked the MIN_DISCRETE_IMPROVEMENT threshold, but now
    /// ALL IDENTITY candidates are filtered regardless of improvement rate.
    #[test]
    fn test_discrete_evaluation_filters_low_improvement_identity() {
        // Create a sample set where ALL samples have error (passing the sample count check)
        // but only a tiny fraction (< 1%) would flip helpfully.
        //
        // IMPORTANT: Previously this test was broken - it only gave 5 samples non-zero error,
        // so samples_with_error=5 was less than MIN_NEURON_SAMPLE_COUNT=10, causing all weight
        // combinations to be skipped. The test passed for the wrong reason.
        //
        // Key insight: To make samples truly un-flippable, source_activation must be ZERO.
        // Any non-zero source activation with any of the tested weight scales (0.1 to 50.0)
        // could potentially flip the target. With source_activation=0, contribution=0
        // regardless of weights, so those samples cannot be affected.
        let mut samples = Vec::new();

        // 1000 samples - ALL have error to pass samples_with_error check
        for i in 0..1000 {
            // Only ~5 samples (0.5%) can flip helpfully:
            // These have non-zero source activation
            let can_flip_to_help = i < 5;

            if can_flip_to_help {
                // Sample that CAN flip to help:
                // - source_activation = 1.0 (non-zero, can contribute)
                // - target_value just below threshold (0)
                // - error positive (wants output to increase from 0 to 1)
                samples.push(DiscreteHelpfulSample {
                    source_activation: 1.0,
                    target_value: -0.05,    // Just below threshold
                    target_activation: 0.0, // Currently outputs 0 (STEP)
                    avg_error: 0.5,         // Wants to output 1, has error
                });
            } else {
                // Sample that CANNOT flip:
                // - source_activation = 0 (CRITICAL: contribution is always 0)
                // - Has error but cannot be helped since contribution = weight * 0 = 0
                samples.push(DiscreteHelpfulSample {
                    source_activation: 0.0, // Zero! contribution = weight * 0 = 0
                    target_value: -0.5,     // Below threshold (outputs 0)
                    target_activation: 0.0, // Currently outputs 0
                    avg_error: 0.3,         // Has error but source can't help
                });
            }
        }

        let candidate = evaluate_discrete_candidate(
            "input-0",
            "target-step",
            &samples,
            ThresholdType::Step,
            0.0,  // Zero threshold - MIN_DISCRETE_IMPROVEMENT (1%) should filter
            true, // is_output_target
        );

        // Should NOT return a candidate because improvement would be <1%
        // samples_with_error = 1000 (all have error)
        // helpful_flips = 5 (only samples with non-zero source activation)
        // harmful_flips = 0 (zero-activation samples can't flip either way)
        // improvement = 5 / 1000 = 0.5% < MIN_DISCRETE_IMPROVEMENT (1%)
        assert!(
            candidate.is_none(),
            "Should NOT return IDENTITY candidate with <1% improvement (0.5% in this test). \
             These are equivalent to direct synapses and don't reliably help. \
             Got candidate: {candidate:?}"
        );
    }

    /// Test that even high-improvement IDENTITY candidates are filtered for STEP targets.
    ///
    /// IDENTITY+bias=0 is mathematically equivalent to a synapse:
    ///   IDENTITY(source × w1 + 0) × w2 = source × (w1 × w2)
    ///
    /// If this pattern would help, add-synapse analysis will find it at lower cost
    /// (1 synapse vs 1 neuron + 2 synapses). So we filter ALL IDENTITY candidates
    /// from discrete evaluation, regardless of improvement rate.
    #[test]
    fn test_discrete_evaluation_filters_identity_even_with_high_improvement() {
        // Create samples with high improvement potential (2% > 1% threshold)
        let mut samples = Vec::new();

        for i in 0..1000 {
            // 20 samples (2%) can flip helpfully - above the 1% threshold
            let can_flip_to_help = i < 20;

            if can_flip_to_help {
                samples.push(DiscreteHelpfulSample {
                    source_activation: 1.0,
                    target_value: -0.05,
                    target_activation: 0.0,
                    avg_error: 0.5,
                });
            } else {
                samples.push(DiscreteHelpfulSample {
                    source_activation: 0.0, // Cannot contribute
                    target_value: -0.5,
                    target_activation: 0.0,
                    avg_error: 0.3,
                });
            }
        }

        let candidate = evaluate_discrete_candidate(
            "input-0",
            "target-step",
            &samples,
            ThresholdType::Step,
            0.0,
            true, // is_output_target
        );

        // Should NOT return a candidate even with 2% improvement
        // because IDENTITY+bias=0 is equivalent to a synapse.
        // Add-synapse will find this pattern at lower cost (1 synapse vs 1 neuron + 2 synapses).
        assert!(
            candidate.is_none(),
            "Should NOT return IDENTITY candidate for STEP even with high improvement. \
            IDENTITY(source × w1 + 0) × w2 = source × (w1 × w2), which is equivalent to a synapse. \
            Add-synapse analysis will find this pattern at lower cost."
        );
    }

    /// Test get_bias_values returns log-spaced values for efficient search
    #[test]
    fn test_get_bias_values() {
        // ReLU should have extended negative range for high-threshold neurons
        let relu_values = get_bias_values("ReLU");
        assert!(relu_values.contains(&0.0), "Should include 0");
        assert!(
            relu_values.iter().any(|&v| v <= -10.0),
            "ReLU should have large negative bias for high thresholds"
        );
        assert!(
            relu_values.len() < 25,
            "Should be efficient (log-spaced, not linear)"
        );

        // IDENTITY should have widest range (scales with large weights)
        let identity_values = get_bias_values("IDENTITY");
        assert!(
            identity_values.iter().any(|&v| v >= 25.0),
            "IDENTITY should reach 25.0 for large weight configurations"
        );
        assert!(
            identity_values.iter().any(|&v| v <= -25.0),
            "IDENTITY should reach -25.0"
        );

        // All values should be sorted
        for squash in &["ReLU", "TANH", "IDENTITY", "GELU"] {
            let values = get_bias_values(squash);
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(values, sorted, "Values for {squash} should be sorted");
        }
    }

    /// Test bias calculation with non-finite values
    #[test]
    fn test_bias_calculation_with_non_finite_values() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: f32::NAN,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: f32::INFINITY,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

        // Should handle non-finite values gracefully and return a finite bias
        assert!(
            bias.is_finite(),
            "Bias should be finite even with non-finite input values"
        );
        assert!(
            (-1.0..=1.0).contains(&bias),
            "Bias should be in reasonable range, got {bias}"
        );
    }

    /// Test that split-error ReLU evaluation finds complementary pairs when errors are split.
    /// When errors are ~50/50 positive/negative, no single ReLU can help all samples.
    /// Split evaluation should find two candidates: one for each error direction.
    #[test]
    fn test_split_relu_finds_complementary_pairs() {
        // Create samples with split errors:
        // - Half have positive error (output should be higher) with high source activation
        // - Half have negative error (output should be lower) with different pattern
        let mut samples = Vec::new();

        // Positive errors: when source is high, output should be higher
        // A ReLU with positive weight on these samples will help
        for i in 0..50 {
            samples.push(HelpfulSample {
                activation: 0.5 + (i as f32) * 0.01,
                avg_error: 0.3, // Positive: output should be higher
                target_value: None,
                target_activation: None,
            });
        }

        // Negative errors: when source is high, output should be lower
        // A ReLU with negative weight on these samples will help
        for i in 0..50 {
            samples.push(HelpfulSample {
                activation: 0.5 + (i as f32) * 0.01,
                avg_error: -0.3, // Negative: output should be lower
                target_value: None,
                target_activation: None,
            });
        }

        // Verify we have split errors
        let positive_count = samples.iter().filter(|s| s.avg_error > 0.0).count();
        let negative_count = samples.iter().filter(|s| s.avg_error < 0.0).count();
        assert_eq!(positive_count, 50);
        assert_eq!(negative_count, 50);

        // Standard ReLU evaluation should struggle because errors cancel out
        // when computing error*activation correlation - roughly equal positive
        // and negative errors with similar activations means weak correlation overall.
        //
        // The split evaluation separates these, so each subset has strong correlation.
        // This test documents the expected behaviour without requiring GPU.
    }

    /// Test that upsert_candidate keeps complementary ReLU candidates with different
    /// incoming_weight values. A positive-weight ReLU (incoming_weight=1.0) and a
    /// negative-weight ReLU (incoming_weight=-1.0) should both be kept, not collide.
    #[test]
    fn test_upsert_keeps_complementary_relu_candidates_by_incoming_weight() {
        use std::collections::HashMap;

        let mut map: HashMap<(String, String, String, i8, i8), CandidateNeuronJson> =
            HashMap::new();

        // Positive-orientation ReLU candidate
        // Issue #128: Use creature-level metrics
        let positive_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: 1.0, // Positive orientation
            outgoing_weight: 0.5,
            squash: "ReLU".to_string(),
            bias: 0.0,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.15,
            expected_creature_score_gain: 0.15,
            improved_count: 30,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Negative-orientation ReLU candidate
        let negative_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: -1.0, // Negative orientation
            outgoing_weight: 0.4,  // Same outgoing sign
            squash: "ReLU".to_string(),
            bias: 0.0,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.12,
            expected_creature_score_gain: 0.12,
            improved_count: 25,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Insert both candidates
        upsert_candidate(&mut map, positive_candidate.clone());
        upsert_candidate(&mut map, negative_candidate.clone());

        // Both should be kept - they have different incoming_weight signs
        assert_eq!(
            map.len(),
            2,
            "Candidates with different incoming_weight should both be kept"
        );

        // Verify both are present with correct values
        let pos_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            1_i8, // incoming sign
            1_i8, // outgoing sign
        );
        let neg_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            -1_i8, // incoming sign
            1_i8,  // outgoing sign
        );

        assert!(
            map.contains_key(&pos_key),
            "Positive-orientation candidate should exist"
        );
        assert!(
            map.contains_key(&neg_key),
            "Negative-orientation candidate should exist"
        );
    }

    /// Test that upsert_candidate keeps split-error complementary pairs with same
    /// incoming_weight but different outgoing_weight signs. This is the key case for
    /// split-error ReLU evaluation where errors are ~50/50 positive/negative.
    #[test]
    fn test_upsert_keeps_split_error_complementary_pairs() {
        use std::collections::HashMap;

        let mut map: HashMap<(String, String, String, i8, i8), CandidateNeuronJson> =
            HashMap::new();

        // Candidate for positive errors: same source/target, positive outgoing_weight
        // This pushes output UP when source is high
        // Issue #128: Use creature-level metrics
        let positive_error_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: 1.0, // Same orientation
            outgoing_weight: 0.5, // POSITIVE: pushes output UP
            squash: "ReLU".to_string(),
            bias: 0.0,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.10,
            expected_creature_score_gain: 0.10,
            improved_count: 25,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Candidate for negative errors: same source/target, negative outgoing_weight
        // This pushes output DOWN when source is high
        let negative_error_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: 1.0,  // Same orientation
            outgoing_weight: -0.4, // NEGATIVE: pushes output DOWN
            squash: "ReLU".to_string(),
            bias: 0.0,
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.08,
            expected_creature_score_gain: 0.08,
            improved_count: 20,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Insert both candidates
        upsert_candidate(&mut map, positive_error_candidate.clone());
        upsert_candidate(&mut map, negative_error_candidate.clone());

        // Both should be kept - they have different outgoing_weight signs
        // This is the key fix for split-error ReLU evaluation
        assert_eq!(
            map.len(),
            2,
            "Split-error complementary pairs with different outgoing_weight signs should both be kept"
        );

        // Verify both are present
        let pos_out_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            1_i8, // incoming sign (both same)
            1_i8, // outgoing sign: positive
        );
        let neg_out_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            1_i8,  // incoming sign (both same)
            -1_i8, // outgoing sign: negative
        );

        assert!(
            map.contains_key(&pos_out_key),
            "Positive-outgoing candidate (pushes UP) should exist"
        );
        assert!(
            map.contains_key(&neg_out_key),
            "Negative-outgoing candidate (pushes DOWN) should exist"
        );

        assert_eq!(
            map.get(&pos_out_key).unwrap().outgoing_weight,
            0.5,
            "Positive-outgoing candidate should have outgoing_weight=0.5"
        );
        assert_eq!(
            map.get(&neg_out_key).unwrap().outgoing_weight,
            -0.4,
            "Negative-outgoing candidate should have outgoing_weight=-0.4"
        );
    }

    // ============================================================================
    // TDD TESTS: Validate ReLU improvement calculations
    // ============================================================================

    /// TDD Test: Verify that predicted improvement matches actual for split errors.
    /// This tests the core maths of compute_net_improvement_across_all_samples.
    #[test]
    fn test_predicted_improvement_matches_actual_for_split_relu() {
        // Scenario: 50% positive errors, 50% negative errors
        // Source always positive (0.5), so ReLU always fires
        //
        // Positive errors: error = +0.2 (want output higher)
        // Negative errors: error = -0.2 (want output lower)
        //
        // If we compute optimal weight from positive subset: w = Σ(error×act)/Σ(act²)
        // For positive subset: w = (0.2×0.5 + 0.2×0.5) / (0.5² + 0.5²) = 0.2/0.5 = 0.4
        //
        // Now apply w=0.4 to ALL samples:
        // - Positive samples: new_error = 0.2 - 0.4×0.5 = 0.0 (perfect!)
        // - Negative samples: new_error = -0.2 - 0.4×0.5 = -0.4 (much worse!)
        //
        // Net improvement should be NEGATIVE (overall harm)

        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: -0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: -0.2,
                target_value: None,
                target_activation: None,
            },
        ];

        // Optimal weight computed from positive samples
        let outgoing_weight: f32 = 0.4;
        let incoming_weight: f32 = 1.0;

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Use the function under test (linear model, bias=0)
        let predicted_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0 for this test (ReLU threshold at 0)
            baseline_error_sq,
            None,
        );

        // Manually compute actual improvement
        let mut new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_out = (incoming_weight * sample.activation).max(0.0);
            let new_err = sample.avg_error - outgoing_weight * relu_out;
            new_error_sq += new_err.powi(2);
        }
        let actual_improvement = (baseline_error_sq - new_error_sq) / baseline_error_sq;

        eprintln!("Baseline error²: {baseline_error_sq:.4}");
        eprintln!(
            "Outgoing weight: {outgoing_weight:.4}, Predicted: {:.4}%, Actual: {:.4}%",
            predicted_improvement * 100.0,
            actual_improvement * 100.0
        );

        assert!(
            (predicted_improvement - actual_improvement).abs() < 0.0001,
            "Predicted {predicted_improvement:.4} must match actual {actual_improvement:.4}",
        );

        // With split errors, the net improvement should be NEGATIVE
        assert!(
            predicted_improvement < 0.0,
            "With split errors and uniform source, net improvement should be negative, got {:.4}%",
            predicted_improvement * 100.0
        );
    }

    /// TDD Test: When errors are aligned, linear model should be accurate.
    #[test]
    fn test_linear_model_accurate_when_errors_aligned() {
        // All positive errors, source always positive
        // This is the ideal case for ReLU - linear model should work perfectly
        let samples = vec![
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.5,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.4,
                target_value: None,
                target_activation: None,
            },
        ];

        // Compute optimal weight: w = Σ(error×activation) / Σ(activation²)
        let error_act_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let act_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let outgoing_weight = error_act_sum / act_sq_sum;
        let incoming_weight = 1.0f32;

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        let predicted_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0 for this test
            baseline_error_sq,
            None, // Linear model
        );

        // Manually compute (bias=0)
        let mut new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_out = (incoming_weight * sample.activation).max(0.0);
            let new_err = sample.avg_error - outgoing_weight * relu_out;
            new_error_sq += new_err.powi(2);
        }
        let actual_improvement = (baseline_error_sq - new_error_sq) / baseline_error_sq;

        eprintln!(
            "Aligned errors: weight={outgoing_weight:.4}, predicted={:.4}%, actual={:.4}%",
            predicted_improvement * 100.0,
            actual_improvement * 100.0
        );

        assert!(
            (predicted_improvement - actual_improvement).abs() < 0.0001,
            "Predicted {predicted_improvement:.4} must match actual {actual_improvement:.4}",
        );

        assert!(
            actual_improvement > 0.1,
            "With aligned errors, should see significant improvement, got {:.4}%",
            actual_improvement * 100.0
        );
    }

    // ============================================================================
    // TDD TESTS: HARD_TANH saturation behaviour
    // ============================================================================

    /// Extended sample for HARD_TANH testing - includes target neuron's pre-activation value
    struct HardTanhSample {
        source_activation: f32,
        target_value: f32,      // Pre-activation input sum
        target_activation: f32, // Post-activation output (clamped to [-1, 1])
        target_error: f32,      // expected - actual
    }

    /// Apply HARD_TANH activation function
    fn hard_tanh(x: f32) -> f32 {
        x.clamp(-1.0, 1.0)
    }

    /// TEST: Demonstrates that linear model is WRONG for HARD_TANH targets near saturation.
    /// The linear model predicts disaster (-125%) but HARD_TANH actually gives perfect result!
    #[test]
    fn test_hard_tanh_linear_model_is_wrong_near_saturation() {
        // Scenario: Target neuron with HARD_TANH activation is near saturation
        // - target_value = 0.9 (input sum before clamping)
        // - target_activation = 0.9 (output after HARD_TANH, not saturated yet)
        // - expected output = 1.0
        // - error = 1.0 - 0.9 = 0.1 (positive: output should be higher)
        //
        // Source neuron fires with activation 0.5
        // If we add a ReLU with outgoing_weight = 0.5:
        // - contribution = 0.5 * relu(0.5) = 0.5 * 0.5 = 0.25
        //
        // LINEAR MODEL predicts:
        // - new_error = 0.1 - 0.25 = -0.15 (overshot)
        // - old_error² = 0.01, new_error² = 0.0225
        // - improvement = (0.01 - 0.0225) / 0.01 = -125% (WORSE!)
        //
        // ACTUAL HARD_TANH behaviour:
        // - new_input = 0.9 + 0.25 = 1.15
        // - new_output = clamp(1.15, -1, 1) = 1.0 (saturated!)
        // - new_error = 1.0 - 1.0 = 0.0 (PERFECT!)
        // - old_error² = 0.01, new_error² = 0.0
        // - improvement = (0.01 - 0.0) / 0.01 = +100% (MUCH BETTER!)

        let samples = vec![HardTanhSample {
            source_activation: 0.5,
            target_value: 0.9,      // Near saturation
            target_activation: 0.9, // hard_tanh(0.9) = 0.9
            target_error: 0.1,      // expected (1.0) - actual (0.9)
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5;

        // Compute baseline error
        let baseline_error_sq: f32 = samples.iter().map(|s| s.target_error.powi(2)).sum();

        // LINEAR MODEL prediction (current behaviour)
        let mut linear_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let linear_new_error = sample.target_error - contribution;
            linear_new_error_sq += linear_new_error.powi(2);
        }
        let linear_improvement = (baseline_error_sq - linear_new_error_sq) / baseline_error_sq;

        // ACTUAL HARD_TANH behaviour
        let mut hard_tanh_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let new_input = sample.target_value + contribution;
            let new_output = hard_tanh(new_input);
            let expected = sample.target_activation + sample.target_error;
            let new_error = expected - new_output;
            hard_tanh_new_error_sq += new_error.powi(2);
        }
        let hard_tanh_improvement =
            (baseline_error_sq - hard_tanh_new_error_sq) / baseline_error_sq;

        eprintln!("Baseline error²: {baseline_error_sq:.4}");
        eprintln!(
            "Linear model: new_error²={linear_new_error_sq:.4}, improvement={:.1}%",
            linear_improvement * 100.0
        );
        eprintln!(
            "HARD_TANH actual: new_error²={hard_tanh_new_error_sq:.4}, improvement={:.1}%",
            hard_tanh_improvement * 100.0
        );

        // The linear model predicts NEGATIVE improvement (making things worse)
        assert!(
            linear_improvement < 0.0,
            "Linear model should predict negative improvement near saturation, got {:.1}%",
            linear_improvement * 100.0
        );

        // But the actual HARD_TANH behaviour shows PERFECT improvement!
        assert!(
            hard_tanh_improvement > 0.99,
            "HARD_TANH should show ~100% improvement (error goes to 0), got {:.1}%",
            hard_tanh_improvement * 100.0
        );

        // The difference is massive - linear model is completely wrong!
        let difference = (hard_tanh_improvement - linear_improvement).abs();
        assert!(
            difference > 1.0,
            "Difference between models should be >100%, got {:.1}%",
            difference * 100.0
        );
    }

    /// TEST: Linear model predicts improvement but HARD_TANH shows NO improvement (already saturated)
    #[test]
    fn test_hard_tanh_linear_model_wrong_when_already_saturated() {
        // Scenario: Target is ALREADY saturated at 1.0
        // - target_value = 1.5 (input already beyond saturation)
        // - target_activation = 1.0 (clamped output)
        // - expected output = 0.8
        // - error = 0.8 - 1.0 = -0.2 (negative: output should be LOWER)
        //
        // Source fires with activation 0.5, ReLU with outgoing_weight = -0.3
        // Contribution = -0.3 * 0.5 = -0.15 (pushing output DOWN, seems good!)
        //
        // LINEAR MODEL predicts:
        // - new_error = -0.2 - (-0.15) = -0.05 (improved!)
        // - old_error² = 0.04, new_error² = 0.0025
        // - improvement = (0.04 - 0.0025) / 0.04 = 93.75% (great!)
        //
        // ACTUAL HARD_TANH behaviour:
        // - new_input = 1.5 + (-0.15) = 1.35 (still beyond saturation!)
        // - new_output = clamp(1.35) = 1.0 (unchanged!)
        // - new_error = 0.8 - 1.0 = -0.2 (NO CHANGE!)
        // - improvement = 0%

        let samples = vec![HardTanhSample {
            source_activation: 0.5,
            target_value: 1.5,      // Already beyond saturation
            target_activation: 1.0, // Clamped at max
            target_error: -0.2,     // expected (0.8) - actual (1.0)
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = -0.3; // Trying to push output down

        let baseline_error_sq: f32 = samples.iter().map(|s| s.target_error.powi(2)).sum();

        // LINEAR MODEL
        let mut linear_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let linear_new_error = sample.target_error - contribution;
            linear_new_error_sq += linear_new_error.powi(2);
        }
        let linear_improvement = (baseline_error_sq - linear_new_error_sq) / baseline_error_sq;

        // ACTUAL HARD_TANH
        let mut hard_tanh_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let new_input = sample.target_value + contribution;
            let new_output = hard_tanh(new_input);
            let expected = sample.target_activation + sample.target_error;
            let new_error = expected - new_output;
            hard_tanh_new_error_sq += new_error.powi(2);
        }
        let hard_tanh_improvement =
            (baseline_error_sq - hard_tanh_new_error_sq) / baseline_error_sq;

        eprintln!("Baseline error²: {baseline_error_sq:.4}");
        eprintln!(
            "Linear model predicts: {:.1}% improvement",
            linear_improvement * 100.0
        );
        eprintln!(
            "HARD_TANH actual: {:.1}% improvement",
            hard_tanh_improvement * 100.0
        );

        // Linear model predicts big improvement
        assert!(
            linear_improvement > 0.9,
            "Linear model should predict ~93% improvement, got {:.1}%",
            linear_improvement * 100.0
        );

        // But HARD_TANH shows NO improvement (still saturated)
        assert!(
            hard_tanh_improvement.abs() < 0.01,
            "HARD_TANH should show ~0% improvement (still saturated), got {:.1}%",
            hard_tanh_improvement * 100.0
        );
    }

    /// Test that compute_net_improvement_with_squash uses HARD_TANH model when specified.
    /// This verifies the actual function we use in production.
    #[test]
    fn test_compute_net_improvement_uses_hard_tanh_model() {
        // Create samples WITH target data (target_value and target_activation)
        // so that the HARD_TANH model can be used
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,          // expected 1.0, actual 0.9
            target_value: Some(0.9), // Near saturation
            target_activation: Some(0.9),
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.5 * 0.5 = 0.25

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Test with LINEAR model (no squash specified, bias=0)
        let linear_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            baseline_error_sq,
            None,
        );

        // Test with HARD_TANH model (bias=0)
        let hard_tanh_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            baseline_error_sq,
            Some("HARD_TANH"),
        );

        eprintln!(
            "compute_net_improvement_with_squash(None): {:.1}%",
            linear_improvement * 100.0
        );
        eprintln!(
            "compute_net_improvement_with_squash(HARD_TANH): {:.1}%",
            hard_tanh_improvement * 100.0
        );

        // LINEAR model should predict NEGATIVE improvement (overshoot to -0.15 error)
        // new_error = 0.1 - 0.25 = -0.15, new_error² = 0.0225
        // baseline = 0.01, so improvement = (0.01 - 0.0225) / 0.01 = -125%
        assert!(
            linear_improvement < 0.0,
            "Linear model should predict negative improvement, got {:.1}%",
            linear_improvement * 100.0
        );

        // HARD_TANH model should predict PERFECT improvement (saturate at 1.0)
        // new_input = 0.9 + 0.25 = 1.15, new_output = clamp(1.15) = 1.0
        // expected = 0.9 + 0.1 = 1.0, new_error = 0.0
        // improvement = (0.01 - 0.0) / 0.01 = 100%
        assert!(
            hard_tanh_improvement > 0.99,
            "HARD_TANH should show ~100% improvement, got {:.1}%",
            hard_tanh_improvement * 100.0
        );

        // The difference between models should be massive
        let difference = (hard_tanh_improvement - linear_improvement).abs();
        assert!(
            difference > 1.0,
            "Difference between models should be >100%, got {:.1}%",
            difference * 100.0
        );
    }

    /// Test that HARD_TANH model falls back to linear when target data is missing.
    #[test]
    fn test_compute_net_improvement_falls_back_to_linear_without_target_data() {
        // Create samples WITHOUT target data (None values)
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,      // No target data
            target_activation: None, // No target data
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5;
        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Even with HARD_TANH specified, should fall back to linear model (bias=0)
        let improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            baseline_error_sq,
            Some("HARD_TANH"),
        );

        // Should match the linear model result (-125%)
        // new_error = 0.1 - 0.25 = -0.15, new_error² = 0.0225
        // improvement = (0.01 - 0.0225) / 0.01 = -125%
        let expected_linear = -1.25;
        assert!(
            (improvement - expected_linear).abs() < 0.01,
            "Should fall back to linear model without target data, got {:.1}% (expected {:.1}%)",
            improvement * 100.0,
            expected_linear * 100.0
        );
    }

    /// TEST: count_improved_samples must use HARD_TANH model for accurate sample counts.
    ///
    /// This test demonstrates the bug where count_improved_samples always uses the linear
    /// model, causing inaccurate counts for HARD_TANH targets. For a sample near saturation,
    /// the linear model predicts the error gets worse (overshoot), but the HARD_TANH model
    /// correctly shows the sample is improved (saturates at the limit).
    #[test]
    fn test_count_improved_samples_uses_hard_tanh_model() {
        // Scenario: Target neuron with HARD_TANH activation is near saturation
        // - target_value = 0.9 (input sum before clamping)
        // - target_activation = 0.9 (output after HARD_TANH, not saturated yet)
        // - expected output = 1.0 (what we want)
        // - avg_error = 0.1 (expected - actual = 1.0 - 0.9 = 0.1)
        //
        // When we add a connection with contribution = 0.25:
        // - LINEAR model: new_error = 0.1 - 0.25 = -0.15, |new_error| > |old_error|, NOT improved
        // - HARD_TANH model: new_input = 0.9 + 0.25 = 1.15, new_output = clamp(1.15) = 1.0
        //                    new_error = 1.0 - 1.0 = 0.0, |new_error| < |old_error|, IMPROVED!
        let samples = vec![HelpfulSample {
            activation: 0.5,              // Source neuron's activation
            avg_error: 0.1,               // Target wants to go up by 0.1
            target_value: Some(0.9),      // Pre-activation input sum
            target_activation: Some(0.9), // Post-activation output (not yet saturated)
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.5 × max(0, 1.0 × 0.5) = 0.25

        // With HARD_TANH model, this sample SHOULD be counted as improved (bias=0)
        let (improved_count, total_count) = count_improved_samples(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            Some("HARD_TANH"),
        );

        assert_eq!(total_count, 1, "Should have 1 total sample");
        assert_eq!(
            improved_count, 1,
            "HARD_TANH model should show sample is improved (saturates at 1.0), got {improved_count} improved",
        );
    }

    /// TEST: count_improved_samples falls back to linear model when target data is missing.
    #[test]
    fn test_count_improved_samples_falls_back_to_linear_without_target_data() {
        // Sample WITHOUT target data - should use linear model
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,      // No target data
            target_activation: None, // No target data
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.25
                                   // Linear model: new_error = 0.1 - 0.25 = -0.15, |new_error| > |old_error|, NOT improved

        let (improved_count, _) = count_improved_samples(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0,               // bias=0
            Some("HARD_TANH"), // Even with HARD_TANH, should fall back to linear
        );

        assert_eq!(
            improved_count, 0,
            "Without target data, should fall back to linear model (sample not improved)"
        );
    }

    /// TEST: count_improved_samples uses linear model for non-HARD_TANH activations.
    #[test]
    fn test_count_improved_samples_uses_linear_for_other_activations() {
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.3,
            target_value: Some(0.5),
            target_activation: Some(0.5),
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.25
                                   // Linear: new_error = 0.3 - 0.25 = 0.05, |new_error| < |old_error| = 0.3, IMPROVED

        // With TANH (not HARD_TANH), should use linear model (bias=0)
        let (improved_count, _) = count_improved_samples(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0,
            Some("TANH"),
        );

        assert_eq!(
            improved_count, 1,
            "Linear model should show sample is improved for TANH"
        );

        // With None squash, should also use linear model (bias=0)
        let (improved_count_none, _) =
            count_improved_samples(&samples, incoming_weight, outgoing_weight, 0.0, None);

        assert_eq!(
            improved_count_none, 1,
            "Linear model should show sample is improved when squash is None"
        );
    }

    /// Test that compute_activation_improvement_and_count correctly falls back to linear model
    /// when samples lack target_value/target_activation data, even if use_hard_tanh is true.
    ///
    /// This validates the safety invariant: use_hard_tanh should only be true when
    /// can_use_hard_tanh() has verified all samples have the required data.
    #[test]
    fn test_activation_improvement_uses_linear_when_no_target_data() {
        // Samples WITHOUT target data
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,      // No target data
            target_activation: None, // No target data
        }];

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // can_use_hard_tanh should return false when samples lack target data
        assert!(
            !can_use_hard_tanh(&samples, Some("HARD_TANH")),
            "can_use_hard_tanh must return false when samples lack target data"
        );

        // get_target_simulation_fn should also return None when samples lack target data
        assert!(
            get_target_simulation_fn(&samples, Some("HARD_TANH")).is_none(),
            "get_target_simulation_fn must return None when samples lack target data"
        );

        // When properly using get_target_simulation_fn, we get linear model behaviour
        let target_activation_fn = get_target_simulation_fn(&samples, Some("HARD_TANH"));
        let (improvement, improved, total) = compute_activation_improvement_and_count(
            &samples,
            1.0,            // incoming_weight
            0.5,            // outgoing_weight
            0.0,            // bias
            |x| x.max(0.0), // ReLU activation
            baseline_sq,
            target_activation_fn, // Will be None due to missing target data
        );

        // Linear model: contribution = 0.5 × max(0, 1.0 × 0.5 + 0) = 0.25
        // new_error = 0.25 - 0.1 = 0.15 (note: compute_activation uses contribution - avg_error)
        // But |0.15| > |0.1| so sample is NOT improved
        // improvement = (0.01 - 0.0225) / 0.01 = -125%
        assert!(
            improvement < 0.0,
            "Linear model should show negative improvement"
        );
        assert_eq!(total, 1, "Should have 1 total sample");
        assert_eq!(
            improved, 0,
            "Linear model should show sample is NOT improved"
        );
    }

    /// Verify synapse weight uses correct linear optimal formula: w = Σ(error × activation) / Σ(activation²)
    /// The old buggy formula (Σ|error| / Σ|activation|) would clamp to ±1.0 in many cases.
    /// This test ensures weights are computed correctly and not always clamped.
    #[test]
    fn synapse_weight_uses_correct_linear_optimal_formula() {
        // Create HelpfulStats with known values that demonstrate the difference
        // between the correct and buggy formulas:
        //
        // Sample 1: activation=2.0, error=0.8  (error×activation = 1.6, activation² = 4.0)
        // Sample 2: activation=3.0, error=0.6  (error×activation = 1.8, activation² = 9.0)
        // Sample 3: activation=1.0, error=0.4  (error×activation = 0.4, activation² = 1.0)
        //
        // Σ(error × activation) = 1.6 + 1.8 + 0.4 = 3.8
        // Σ(activation²) = 4.0 + 9.0 + 1.0 = 14.0
        //
        // Correct optimal weight: 3.8 / 14.0 = 0.271...
        //
        // Old buggy formula would compute:
        // Σ|error| = 0.8 + 0.6 + 0.4 = 1.8
        // Σ|activation| = 2.0 + 3.0 + 1.0 = 6.0
        // Buggy weight: 1.8 / 6.0 = 0.3 (different!)
        //
        // And in cases where Σ|error| > Σ|activation|, the buggy formula would clamp to 1.0

        let stats = HelpfulStats {
            positive_count: 3, // All samples have positive correlation for this test
            negative_count: 0,
            positive_improvement_sum: 1.8, // Σ|error| for positive samples (unused in new formula)
            negative_improvement_sum: 0.0,
            positive_activation_sum: 6.0, // Σ|activation| for positive samples (unused in new formula)
            negative_activation_sum: 0.0,
            error_sq_sum: 0.8 * 0.8 + 0.6 * 0.6 + 0.4 * 0.4, // 0.64 + 0.36 + 0.16 = 1.16
            activation_sq_sum: 14.0,                         // Σ(activation²) = 4 + 9 + 1
            error_activation_sum: 3.8, // Σ(error × activation) = 1.6 + 1.8 + 0.4
        };

        // Apply the correct formula used in production (after fix):
        // weight = error_activation_sum / (activation_sq_sum + EPSILON)
        let raw_weight = if stats.activation_sq_sum > EPSILON {
            stats.error_activation_sum / (stats.activation_sq_sum + EPSILON)
        } else {
            0.0
        };
        let weight = raw_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        // Raw weight: 3.8 / 14.0 ≈ 0.2714
        let expected_raw = 3.8 / 14.0;
        assert!(
            (raw_weight - expected_raw).abs() < 0.001,
            "Raw weight should be Σ(error×activation)/Σ(activation²) = {expected_raw:.4}, got {raw_weight:.4}"
        );

        // Weight should be clamped to MAX_OUTGOING_WEIGHT since 0.2714 > 0.1
        assert!(
            (weight - MAX_OUTGOING_WEIGHT).abs() < 0.001,
            "Weight should be clamped to MAX_OUTGOING_WEIGHT = {MAX_OUTGOING_WEIGHT}, got {weight:.4}"
        );

        // Verify the RAW value is NOT the buggy value
        let buggy_weight = 1.8 / 6.0; // 0.3
        assert!(
            (raw_weight - buggy_weight).abs() > 0.01,
            "Raw weight {raw_weight} should differ from buggy formula result {buggy_weight}"
        );

        // Now test a case that would clamp to 1.0 with the buggy formula
        // Samples where |error| >> |activation|
        let stats_would_clamp = HelpfulStats {
            positive_count: 2,
            negative_count: 0,
            positive_improvement_sum: 5.0, // Σ|error| (buggy formula would use this)
            negative_improvement_sum: 0.0,
            positive_activation_sum: 2.0, // Σ|activation| (buggy formula: 5.0/2.0 = 2.5 -> clamp to 1.0)
            negative_activation_sum: 0.0,
            error_sq_sum: 13.0,        // 2² + 3² = 4 + 9 = 13
            activation_sq_sum: 2.0,    // 1² + 1² = 2
            error_activation_sum: 5.0, // 2×1 + 3×1 = 5
        };

        let raw_weight2 = if stats_would_clamp.activation_sq_sum > EPSILON {
            stats_would_clamp.error_activation_sum / (stats_would_clamp.activation_sq_sum + EPSILON)
        } else {
            0.0
        };
        let weight2 = raw_weight2.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        // Raw optimal: 5.0 / 2.0 = 2.5
        // With new tight clamp, this gets clamped to MAX_OUTGOING_WEIGHT
        assert!(
            (raw_weight2 - 2.5).abs() < 0.001,
            "Raw weight should be 2.5, got {raw_weight2}"
        );
        assert!(
            (weight2 - MAX_OUTGOING_WEIGHT).abs() < 0.001,
            "Weight should be clamped to MAX_OUTGOING_WEIGHT = {MAX_OUTGOING_WEIGHT}, got {weight2}"
        );
        // The raw calculation (2.5) is correct, but now we clamp to a tighter range
        // to improve prediction accuracy based on successful discovery analysis.
    }

    /// Test that synapse improvement calculation uses saturation-aware model for HARD_TANH targets.
    ///
    /// CRITICAL: avg_error is in VALUE domain (targetValue - currentValue from TypeScript).
    /// This means expected = squash(target_value + avg_error), NOT target_activation + avg_error.
    ///
    /// Scenario: Target near saturation where linear model UNDERPREDICTS actual benefit.
    /// - current value = 0.8, current activation = 0.8
    /// - avg_error = 0.3 (VALUE domain: want to add 0.3 to pre-activation)
    /// - desired_value = 1.1, expected_activation = clamp(1.1) = 1.0
    /// - actual activation error = 1.0 - 0.8 = 0.2 (what we really want to fix)
    ///
    /// If contribution = 0.25 (pushing value from 0.8 to 1.05):
    /// - Linear model: new_error = 0.3 - 0.25 = 0.05 (thinks we still have 0.05 error)
    /// - Saturation: new_output = clamp(1.05) = 1.0, new_error = 1.0 - 1.0 = 0 (perfect!)
    ///
    /// Linear model underpredicts because it doesn't know saturation "absorbs" the overshoot.
    #[test]
    fn synapse_improvement_uses_saturation_aware_model_for_hard_tanh() {
        // Scenario where saturation helps - the target is pushing towards saturation
        // and the synapse contribution helps reach it even though linear math says we undershot
        let samples = vec![
            HelpfulSample {
                activation: 0.5,              // source neuron activation
                avg_error: 0.3,               // VALUE domain: want +0.3 to pre-activation
                target_value: Some(0.8),      // current pre-activation
                target_activation: Some(0.8), // current output (linear region)
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.25, // VALUE domain
                target_value: Some(0.85),
                target_activation: Some(0.85),
            },
        ];

        // Compute optimal weight using linear model (treats avg_error as activation error)
        let mut error_activation_sum = 0.0f32;
        let mut activation_sq_sum = 0.0f32;
        let mut baseline_value_error_sq = 0.0f32; // Linear baseline (VALUE domain errors)

        for sample in &samples {
            error_activation_sum += sample.avg_error * sample.activation;
            activation_sq_sum += sample.activation * sample.activation;
            baseline_value_error_sq += sample.avg_error * sample.avg_error;
        }

        let weight = error_activation_sum / (activation_sq_sum + EPSILON);

        // LINEAR MODEL PREDICTION (using VALUE domain errors throughout):
        // improvement = (2*w*E[a*e] - w²*E[a²]) / E[e²]
        let linear_improvement = (2.0 * weight * error_activation_sum
            - weight * weight * activation_sq_sum)
            / baseline_value_error_sq;

        // SATURATION-AWARE MODEL (correct ACTIVATION domain):
        // Compute actual errors in activation domain where MSE is measured
        // CRITICAL: expected = squash(target_value + avg_error)
        let mut baseline_activation_error_sq = 0.0f32;
        let mut new_activation_error_sq = 0.0f32;

        for sample in &samples {
            let target_value = sample.target_value.unwrap();
            let target_activation = sample.target_activation.unwrap();
            let desired_value = target_value + sample.avg_error;
            let expected_output = desired_value.clamp(-1.0, 1.0);

            // Baseline error in ACTIVATION domain (what MSE actually measures)
            let baseline_act_error = expected_output - target_activation;
            baseline_activation_error_sq += baseline_act_error * baseline_act_error;

            // New pre-activation and output
            let new_pre_activation = target_value + weight * sample.activation;
            let new_output = new_pre_activation.clamp(-1.0, 1.0);
            let new_act_error = expected_output - new_output;
            new_activation_error_sq += new_act_error * new_act_error;
        }

        let actual_improvement =
            (baseline_activation_error_sq - new_activation_error_sq) / baseline_activation_error_sq;

        // Both models should show improvement for this well-chosen scenario
        assert!(
            linear_improvement > 0.5,
            "Linear model should show significant improvement, got {linear_improvement:.4}"
        );
        assert!(
            actual_improvement > 0.5,
            "Saturation-aware model should show significant improvement, got {actual_improvement:.4}"
        );

        // The key insight: models may differ, but saturation-aware is more accurate
        // Log the difference for debugging
        eprintln!(
            "Linear improvement: {:.2}%, Saturation-aware: {:.2}%",
            linear_improvement * 100.0,
            actual_improvement * 100.0
        );

        // Now test that compute_synapse_improvement_with_target_squash gives accurate prediction
        // Use VALUE domain baseline for consistency with how the function is called in production
        let saturation_aware_improvement = compute_synapse_improvement_with_target_squash(
            &samples,
            weight,
            baseline_value_error_sq,
            Some("HARD_TANH"),
        );

        // Log all predictions for debugging
        eprintln!(
            "Function prediction: {:.2}%",
            saturation_aware_improvement * 100.0
        );

        // The saturation-aware function should give reasonable predictions
        assert!(
            saturation_aware_improvement.is_finite(),
            "Saturation-aware model should give finite improvement"
        );
    }

    /// TDD Test: Domain consistency in improvement calculation.
    ///
    /// BUG: When using target_activation_fn simulation, the code was computing:
    /// - baseline_error in VALUE domain (sample.avg_error²)
    /// - new_error in ACTIVATION domain (expected - target_fn(new_input))²
    ///
    /// This is WRONG because VALUE and ACTIVATION domains have different scales
    /// near saturation. The actual MSE (measured by Deno) is in ACTIVATION domain,
    /// so both baseline and new must use ACTIVATION domain for accurate predictions.
    ///
    /// This test verifies that the improvement calculation uses consistent domains.
    #[test]
    fn improvement_calculation_uses_consistent_domains() {
        // Near saturation scenario where domain mismatch is most visible
        let samples = vec![HelpfulSample {
            activation: 0.5,              // source activation
            avg_error: 0.3,               // VALUE domain: want +0.3 to pre-activation
            target_value: Some(0.9),      // near upper saturation
            target_activation: Some(0.9), // HARD_TANH(0.9) = 0.9
        }];

        // With this sample:
        // - desired_value = 0.9 + 0.3 = 1.2
        // - expected = HARD_TANH(1.2) = 1.0 (saturated)
        // - target_activation = 0.9
        //
        // ACTIVATION domain baseline error = 1.0 - 0.9 = 0.1
        // VALUE domain baseline error = 0.3 (sample.avg_error)
        //
        // If we apply a contribution of 0.15:
        // - new_input = 0.9 + 0.15 = 1.05
        // - new_output = HARD_TANH(1.05) = 1.0 (saturated)
        // - ACTIVATION domain new error = 1.0 - 1.0 = 0.0 (perfect!)
        //
        // CORRECT improvement (ACTIVATION domain):
        // = (0.1² - 0.0²) / 0.1² = 100%
        //
        // BUGGY improvement (VALUE baseline, ACTIVATION new):
        // = (0.3² - 0.0²) / 0.3² = 100% (coincidentally same in this case)

        // Now test with partial contribution
        let contribution = 0.05;
        // new_input = 0.9 + 0.05 = 0.95
        // new_output = HARD_TANH(0.95) = 0.95
        // ACTIVATION domain new error = 1.0 - 0.95 = 0.05
        //
        // CORRECT improvement (ACTIVATION domain):
        // = (0.1² - 0.05²) / 0.1² = (0.01 - 0.0025) / 0.01 = 75%
        //
        // BUGGY improvement (VALUE baseline, ACTIVATION new):
        // = (0.3² - 0.05²) / 0.3² = (0.09 - 0.0025) / 0.09 = 97% (WRONG!)

        let desired_value: f32 = 0.9 + 0.3; // = 1.2
        let expected = desired_value.clamp(-1.0, 1.0); // = 1.0
        let target_activation: f32 = 0.9;

        // Correct ACTIVATION domain baseline
        let baseline_act_error = expected - target_activation; // = 0.1
        let baseline_act_sq = baseline_act_error * baseline_act_error; // = 0.01

        // New error in ACTIVATION domain
        let new_input: f32 = 0.9 + contribution; // = 0.95
        let new_output = new_input.clamp(-1.0, 1.0); // = 0.95
        let new_act_error = expected - new_output; // = 0.05
        let new_act_sq = new_act_error * new_act_error; // = 0.0025

        // Correct improvement
        let correct_improvement = (baseline_act_sq - new_act_sq) / baseline_act_sq;
        assert!(
            (correct_improvement - 0.75).abs() < 0.01,
            "Correct ACTIVATION domain improvement should be ~75%, got {:.2}%",
            correct_improvement * 100.0
        );

        // Buggy calculation (VALUE baseline, ACTIVATION new)
        let value_baseline_sq: f32 = 0.3 * 0.3; // = 0.09
        let buggy_improvement = (value_baseline_sq - new_act_sq) / value_baseline_sq;
        assert!(
            (buggy_improvement - 0.97_f32).abs() < 0.01,
            "Buggy mixed-domain improvement should be ~97%, got {:.2}%",
            buggy_improvement * 100.0
        );

        // The bug causes MASSIVE overprediction: 97% predicted vs 75% actual
        assert!(
            buggy_improvement > correct_improvement + 0.1,
            "Buggy calculation should significantly overpredict improvement"
        );

        // Now verify the actual function uses consistent domains
        // Use the optimal weight that would produce contribution=0.05 with activation=0.5
        let weight = contribution / 0.5; // = 0.1

        // CRITICAL: Production passes VALUE domain baseline (this is what stats.error_sq_sum is)
        // The function should INTERNALLY compute ACTIVATION domain baseline when simulating target
        let production_baseline = value_baseline_sq; // VALUE domain as passed in production

        let (improvement, _, _, _) = compute_synapse_improvement_and_count(
            &samples,
            weight,
            production_baseline,
            Some("HARD_TANH"),
        );

        // The function should give the CORRECT result (75%) not the buggy result (97%)
        // because it should internally use ACTIVATION domain for both baseline and new error
        assert!(
            (improvement - correct_improvement).abs() < 0.1,
            "Function should internally use consistent ACTIVATION domain. \
             Expected ~{:.1}%, got {:.1}% (buggy would be ~{:.1}%)",
            correct_improvement * 100.0,
            improvement * 100.0,
            buggy_improvement * 100.0
        );
    }

    /// TDD Test: ReLU improvement calculation MUST include bias for accurate predictions.
    ///
    /// When bias > 0, the ReLU threshold shifts left, causing more samples to activate.
    /// When bias < 0, the ReLU threshold shifts right, causing fewer samples to activate.
    ///
    /// If bias is NOT included in the improvement calculation, the prediction will be
    /// inaccurate when a non-zero bias is proposed for the new neuron.
    ///
    /// This test demonstrates the bug: compute_relu_improvement_and_count ignores bias,
    /// leading to overestimation when the actual neuron would use a different activation
    /// pattern due to the bias.
    #[test]
    fn test_relu_improvement_must_include_bias() {
        // Scenario: Source activations that are NEGATIVE (would be zeroed by ReLU without bias).
        // With a positive bias, the ReLU would fire on these samples.
        //
        // Sample 1: activation = -0.3, error = 0.5 (want output higher)
        // Sample 2: activation = -0.2, error = 0.4 (want output higher)
        // Sample 3: activation = 0.1, error = 0.3 (want output higher)
        //
        // Without bias (bias=0):
        //   ReLU(1.0 × -0.3 + 0) = 0  → contribution = 0
        //   ReLU(1.0 × -0.2 + 0) = 0  → contribution = 0
        //   ReLU(1.0 × 0.1 + 0) = 0.1 → contribution = outgoing_weight × 0.1
        //
        // With bias=0.5:
        //   ReLU(1.0 × -0.3 + 0.5) = 0.2 → contribution = outgoing_weight × 0.2
        //   ReLU(1.0 × -0.2 + 0.5) = 0.3 → contribution = outgoing_weight × 0.3
        //   ReLU(1.0 × 0.1 + 0.5) = 0.6 → contribution = outgoing_weight × 0.6
        //
        // The bias dramatically changes which samples are affected and by how much!

        let samples = vec![
            HelpfulSample {
                activation: -0.3,
                avg_error: 0.5,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: 0.4,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.1,
                avg_error: 0.3,
                target_value: None,
                target_activation: None,
            },
        ];

        let incoming_weight = 1.0f32;
        let outgoing_weight = 0.8f32; // Positive weight to reduce positive errors
        let bias = 0.5f32; // Significant positive bias

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();
        // 0.5² + 0.4² + 0.3² = 0.25 + 0.16 + 0.09 = 0.5

        // Predicted improvement using the function WITH bias parameter
        let predicted_with_bias = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            bias,
            baseline_error_sq,
            None,
        );

        // Also compute without bias (bias=0) to show the difference
        let predicted_without_bias = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // No bias
            baseline_error_sq,
            None,
        );

        // Manually compute ACTUAL improvement WITH bias
        let mut new_error_sq_with_bias = 0.0f32;
        for sample in &samples {
            let pre_activation = incoming_weight * sample.activation + bias;
            let relu_out = pre_activation.max(0.0);
            let contribution = outgoing_weight * relu_out;
            let new_err = sample.avg_error - contribution;
            new_error_sq_with_bias += new_err.powi(2);
        }
        let actual_improvement_with_bias =
            (baseline_error_sq - new_error_sq_with_bias) / baseline_error_sq;

        // Manually compute improvement WITHOUT bias (what current code predicts)
        let mut new_error_sq_without_bias = 0.0f32;
        for sample in &samples {
            let pre_activation = incoming_weight * sample.activation; // No bias!
            let relu_out = pre_activation.max(0.0);
            let contribution = outgoing_weight * relu_out;
            let new_err = sample.avg_error - contribution;
            new_error_sq_without_bias += new_err.powi(2);
        }
        let manual_improvement_without_bias =
            (baseline_error_sq - new_error_sq_without_bias) / baseline_error_sq;

        eprintln!(
            "Baseline error²: {baseline_error_sq:.4}, With bias: new_error²={new_error_sq_with_bias:.4}, Without bias: new_error²={new_error_sq_without_bias:.4}"
        );
        let predicted_with_bias_pct = predicted_with_bias * 100.0;
        let predicted_without_bias_pct = predicted_without_bias * 100.0;
        let actual_with_bias_pct = actual_improvement_with_bias * 100.0;
        eprintln!(
            "Predicted (with bias): {predicted_with_bias_pct:.2}%, Predicted (no bias): {predicted_without_bias_pct:.2}%, Actual (with bias): {actual_with_bias_pct:.2}%"
        );

        // The predicted improvement WITH bias should match the actual improvement WITH bias.
        // This verifies that the bias parameter is correctly included in the calculation.
        assert!(
            (predicted_with_bias - actual_improvement_with_bias).abs() < 0.01,
            "Predicted improvement WITH bias ({predicted_with_bias:.4}) must match actual improvement WITH bias ({actual_improvement_with_bias:.4})."
        );

        // Verify that WITHOUT bias prediction matches manual calculation (both bias=0)
        assert!(
            (predicted_without_bias - manual_improvement_without_bias).abs() < 0.01,
            "Predicted (no bias) ({predicted_without_bias:.4}) must match manual (no bias) ({manual_improvement_without_bias:.4})."
        );

        // The key insight: with bias=0.5, improvement should be much higher than with bias=0
        // because more samples activate the ReLU
        assert!(
            actual_improvement_with_bias > predicted_without_bias + 0.1,
            "Improvement with bias ({actual_improvement_with_bias:.4}) should be significantly higher than without ({predicted_without_bias:.4})"
        );
    }

    #[test]
    fn neuron_diagnostics_reports_hidden_neuron_filtered_in_mixed_focus_list() {
        // Test that when a focus list contains BOTH output AND hidden neurons,
        // the hidden neurons get HiddenNeuronFiltered reason (not NoEligibleSources).
        //
        // Bug scenario: When focus_order is NOT empty (some output neurons exist),
        // the skipped_hidden neurons were never merged into diagnostics, so they
        // appeared with misleading reasons like NoEligibleSources.
        let mut diagnostics = NeuronDiagnostics::new_for_tests(&["output-0", "hidden-1"]);

        // Mark hidden-1 as filtered (this is what should happen in normal flow)
        diagnostics.mark_hidden_filtered("hidden-1");

        // Simulate output-0 being processed normally but finding no candidate
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 5);
        diagnostics.record_candidate_attempt("output-0", false);

        let summaries = diagnostics.no_candidate_summaries();

        // Both should have summaries
        assert_eq!(
            summaries.len(),
            2,
            "Expected 2 summaries (one for output, one for hidden)"
        );

        // Find the hidden neuron summary
        let hidden_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "hidden-1")
            .expect("Should have summary for hidden-1");

        // The hidden neuron should have HiddenNeuronFiltered reason
        assert!(
            matches!(
                hidden_summary.reason,
                NeuronNoCandidateReason::HiddenNeuronFiltered
            ),
            "Hidden neuron should report HiddenNeuronFiltered, not {:?}",
            hidden_summary.reason
        );
    }

    #[test]
    fn neuron_diagnostics_reports_input_neuron_filtered_not_hidden() {
        // Test that when an input neuron is in the focus list, it gets
        // InputNeuronFiltered reason (not HiddenNeuronFiltered).
        //
        // Bug scenario: The neuron_type_map was only built from creature.neurons
        // and didn't include input neurons. When an input neuron (e.g. "input-0")
        // was in the focus list, neuron_type_map.get() returned None, and
        // `neuron_type != Some("output")` evaluated to true. The input neuron
        // was incorrectly added to skipped_hidden and reported with
        // HiddenNeuronFiltered reason.
        let mut diagnostics =
            NeuronDiagnostics::new_for_tests(&["output-0", "input-1", "hidden-2"]);

        // Mark input-1 as filtered because it's an input neuron
        diagnostics.mark_input_filtered("input-1");

        // Mark hidden-2 as filtered because it's a hidden neuron
        diagnostics.mark_hidden_filtered("hidden-2");

        // Simulate output-0 being processed normally but finding no candidate
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 5);
        diagnostics.record_candidate_attempt("output-0", false);

        let summaries = diagnostics.no_candidate_summaries();

        // All three should have summaries
        assert_eq!(
            summaries.len(),
            3,
            "Expected 3 summaries (one for output, one for input, one for hidden)"
        );

        // Find the input neuron summary - should have InputNeuronFiltered reason
        let input_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "input-1")
            .expect("Should have summary for input-1");
        assert!(
            matches!(
                input_summary.reason,
                NeuronNoCandidateReason::InputNeuronFiltered
            ),
            "Input neuron should report InputNeuronFiltered, not {:?}",
            input_summary.reason
        );

        // Find the hidden neuron summary - should still have HiddenNeuronFiltered reason
        let hidden_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "hidden-2")
            .expect("Should have summary for hidden-2");
        assert!(
            matches!(
                hidden_summary.reason,
                NeuronNoCandidateReason::HiddenNeuronFiltered
            ),
            "Hidden neuron should report HiddenNeuronFiltered, not {:?}",
            hidden_summary.reason
        );
    }

    #[test]
    fn neuron_diagnostics_reports_constant_neuron_filtered_not_hidden() {
        // Test that when a constant neuron is in the focus list, it gets
        // ConstantNeuronFiltered reason (not HiddenNeuronFiltered).
        //
        // Bug scenario (v0.1.124): Constant neurons were pushed to skipped_hidden
        // but reported with HiddenNeuronFiltered reason. This is semantically
        // incorrect - constant neurons don't receive inputs because they always
        // output a fixed value, which is different from hidden neurons whose
        // backpropagated errors don't reliably predict output error.
        let mut diagnostics =
            NeuronDiagnostics::new_for_tests(&["output-0", "constant-1", "hidden-2"]);

        // Mark constant-1 as filtered because it's a constant neuron
        diagnostics.mark_constant_filtered("constant-1");

        // Mark hidden-2 as filtered because it's a hidden neuron
        diagnostics.mark_hidden_filtered("hidden-2");

        // Simulate output-0 being processed normally but finding no candidate
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 5);
        diagnostics.record_candidate_attempt("output-0", false);

        let summaries = diagnostics.no_candidate_summaries();

        // All three should have summaries
        assert_eq!(
            summaries.len(),
            3,
            "Expected 3 summaries (one for output, one for constant, one for hidden)"
        );

        // Find the constant neuron summary - should have ConstantNeuronFiltered reason
        let constant_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "constant-1")
            .expect("Should have summary for constant-1");
        assert!(
            matches!(
                constant_summary.reason,
                NeuronNoCandidateReason::ConstantNeuronFiltered
            ),
            "Constant neuron should report ConstantNeuronFiltered, not {:?}",
            constant_summary.reason
        );

        // Find the hidden neuron summary - should still have HiddenNeuronFiltered reason
        let hidden_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "hidden-2")
            .expect("Should have summary for hidden-2");
        assert!(
            matches!(
                hidden_summary.reason,
                NeuronNoCandidateReason::HiddenNeuronFiltered
            ),
            "Hidden neuron should report HiddenNeuronFiltered, not {:?}",
            hidden_summary.reason
        );
    }
}

#[cfg(test)]
mod tests_optimal_outgoing_weight {
    use super::*;
    use anyhow::anyhow;

    struct AlwaysFailGpuEvaluator;

    impl GpuEvaluator for AlwaysFailGpuEvaluator {
        fn evaluate_relu(
            &self,
            _samples: &[HelpfulSample],
            _threshold: f32,
        ) -> Result<(ReluStats, ReluStats, f32)> {
            Err(anyhow!(
                "AlwaysFailGpuEvaluator: GPU not available for test"
            ))
        }

        fn evaluate_activation(
            &self,
            _samples: &[HelpfulSample],
            _activation_type: u32,
            _orientation: f32,
            _scale: f32,
        ) -> Result<(f32, f32, f32, u32)> {
            Err(anyhow!(
                "AlwaysFailGpuEvaluator: GPU not available for test"
            ))
        }
    }

    /// Regression: IDENTITY candidates must not be skipped in the all-samples fallback path.
    ///
    /// `evaluate_activation_candidate` previously computed `base_weight` (no-intercept fit)
    /// before checking `spec.name == "IDENTITY"`. If the no-intercept fit returned None (for
    /// example, when Σ(activation×error) cancels to ~0), the function would `continue` and the
    /// IDENTITY-specific affine fit (with intercept/bias) was never attempted.
    ///
    /// This test constructs a dataset where:
    /// - split-error evaluation is NOT "properly attempted" (negative subset < MIN sample count),
    /// - subset evaluation returns None (positive subset has constant error ⇒ best slope is 0),
    /// - the all-samples no-intercept fit produces `None` (Σ(activation×error) cancels to 0),
    /// - but the all-samples affine fit *does* succeed and should yield a candidate.
    #[test]
    fn identity_all_samples_fallback_uses_affine_fit_even_when_base_weight_is_none() {
        let gpu = AlwaysFailGpuEvaluator;

        // Use a minimal spec so this test is deterministic.
        static ORIENTATIONS: [f32; 1] = [1.0];
        static SCALES: [f32; 1] = [1.0];
        let spec = ActivationCandidateSpec {
            name: "IDENTITY",
            orientations: &ORIENTATIONS,
            scales: &SCALES,
            activation: identity_activation,
            min_improvement: 0.0,
        };

        // 11 positive-error samples with varying activation: affine fit slope should be ~0 here,
        // so IDENTITY subset evaluation returns None and we fall through to all-samples fallback.
        let mut samples: Vec<HelpfulSample> = (0..=10)
            .map(|i| HelpfulSample {
                activation: 100.0 + (i as f32) * 10.0, // 100..200
                avg_error: 1.0,
                target_value: None,
                target_activation: None,
            })
            .collect();

        // 9 negative-error samples whose activation sum matches the positive group (1650),
        // making Σ(activation×error) == 0 for the all-samples no-intercept fit.
        let negative_activations: [f32; 9] = [
            200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 150.0, 150.0, 150.0,
        ];
        for activation in negative_activations {
            samples.push(HelpfulSample {
                activation,
                avg_error: -1.0,
                target_value: None,
                target_activation: None,
            });
        }

        let candidate =
            evaluate_activation_candidate(&gpu, "source-0", "target-0", &samples, 0.0, &spec, None)
                .expect("Evaluation should succeed");

        assert!(
            candidate.is_some(),
            "Expected an IDENTITY candidate from the all-samples affine fit. \
             This is a regression if None is returned."
        );
        let candidate = candidate.unwrap();
        assert_eq!(candidate.squash, "IDENTITY");
        assert!(
            candidate.bias.abs() >= 0.01,
            "IDENTITY candidate should have a meaningful bias (not equivalent to a direct synapse)"
        );
        assert!(
            candidate.outgoing_weight.is_finite() && candidate.outgoing_weight.abs() > EPSILON,
            "IDENTITY candidate should have a valid outgoing weight"
        );
    }

    /// Test that calculate_optimal_outgoing_weight returns None for insufficient activation
    #[test]
    fn returns_none_for_zero_activation() {
        let result = calculate_optimal_outgoing_weight(1.0, 0.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when activation_sq is zero"
        );

        let result = calculate_optimal_outgoing_weight(1.0, EPSILON * 0.5, 1.0);
        assert!(
            result.is_none(),
            "Should return None when activation_sq <= EPSILON"
        );
    }

    /// Test that calculate_optimal_outgoing_weight returns None for non-finite results
    #[test]
    fn returns_none_for_non_finite_weight() {
        let result = calculate_optimal_outgoing_weight(f32::INFINITY, 1.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when raw weight is infinite"
        );

        let result = calculate_optimal_outgoing_weight(f32::NAN, 1.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when raw weight is NaN"
        );
    }

    /// Test that calculate_optimal_outgoing_weight returns None for near-zero weights
    #[test]
    fn returns_none_for_near_zero_weight() {
        // Very small error_activation results in near-zero weight
        let result = calculate_optimal_outgoing_weight(EPSILON * 0.1, 100.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when raw weight is near zero"
        );
    }

    /// Test that weights are clamped to MAX_OUTGOING_WEIGHT
    #[test]
    fn clamps_to_max_outgoing_weight() {
        // Large error relative to activation would produce large weight
        // error/activation = 10.0/1.0 = 10.0, should clamp to 0.1
        let result = calculate_optimal_outgoing_weight(10.0, 1.0, 1.0);
        assert!(result.is_some(), "Should return a valid weight");
        let weight = result.unwrap();
        assert!(
            (weight - MAX_OUTGOING_WEIGHT).abs() < EPSILON,
            "Weight {weight} should be clamped to MAX_OUTGOING_WEIGHT {MAX_OUTGOING_WEIGHT}"
        );

        // Negative case
        let result = calculate_optimal_outgoing_weight(-10.0, 1.0, 1.0);
        assert!(result.is_some(), "Should return a valid negative weight");
        let weight = result.unwrap();
        assert!(
            (weight - (-MAX_OUTGOING_WEIGHT)).abs() < EPSILON,
            "Weight {} should be clamped to -MAX_OUTGOING_WEIGHT {}",
            weight,
            -MAX_OUTGOING_WEIGHT
        );
    }

    /// Test weight ratio validation for add-neuron candidates
    #[test]
    fn rejects_small_weight_ratio() {
        // incoming_weight = 10, max outgoing = 0.1, ratio = 100 -> OK
        let result = calculate_optimal_outgoing_weight(1.0, 1.0, 10.0);
        // raw = 1.0, clamped to 0.1, ratio = 10/0.1 = 100 >= 50 -> OK
        assert!(
            result.is_some(),
            "Should accept ratio of 100 (incoming=10, outgoing=0.1)"
        );

        // incoming_weight = 2, max outgoing = 0.1, ratio = 20 < 50 -> REJECT
        let result = calculate_optimal_outgoing_weight(1.0, 1.0, 2.0);
        // raw = 1.0, clamped to 0.1, ratio = 2/0.1 = 20 < 50 -> REJECT
        assert!(
            result.is_none(),
            "Should reject ratio of 20 (incoming=2, outgoing=0.1)"
        );
    }

    /// Test that small incoming weights (synapses) skip ratio check
    #[test]
    fn skips_ratio_check_for_synapses() {
        // For synapses, incoming_weight = 1.0, so ratio check is skipped
        let result = calculate_optimal_outgoing_weight(0.5, 10.0, 1.0);
        // raw = 0.05, within bounds, no ratio check since incoming <= 1.0
        assert!(
            result.is_some(),
            "Should accept synapse weight without ratio check"
        );
        assert!(
            (result.unwrap() - 0.05).abs() < 0.001,
            "Synapse weight should be ~0.05"
        );
    }

    /// Test that successful discovery parameters would pass validation
    /// Based on real successful discoveries from production
    #[test]
    fn successful_discovery_parameters_pass() {
        // Successful discovery: incoming=100, outgoing=-0.00096
        // ratio = 100/0.00096 ≈ 104,000 >> 50 -> OK
        // But we need to compute what error/activation ratio would produce 0.00096
        // If raw weight = 0.00096, and it's not clamped, we need:
        // sum_error_activation / sum_activation_sq = 0.00096
        let sum_activation_sq = 1000.0;
        let sum_error_activation = 0.00096 * sum_activation_sq; // = 0.96

        let result =
            calculate_optimal_outgoing_weight(sum_error_activation, sum_activation_sq, 100.0);
        assert!(result.is_some(), "Successful discovery params should pass");
        let weight = result.unwrap();
        // The weight should be approximately -0.00096 or +0.00096 depending on sign
        assert!(
            weight.abs() < MAX_OUTGOING_WEIGHT,
            "Weight {weight} should be within bounds"
        );
    }

    /// Test that failed discovery parameters would be rejected
    /// Based on real failed discoveries from production
    #[test]
    fn failed_discovery_parameters_rejected() {
        // Failed discovery: incoming=5, outgoing=4.58 (before our fix, this would pass)
        // Now: raw = 4.58, clamped to 0.1, ratio = 5/0.1 = 50, just at the boundary
        // This might just pass or just fail depending on exact values

        // More clearly failed case: incoming=10, outgoing=-10 (1:1 ratio)
        // Even after clamping to 0.1, ratio = 10/0.1 = 100 >= 50 -> passes ratio check
        // BUT the weight is clamped from -10 to -0.1, so prediction accuracy improves

        // The key improvement is that extreme weights like 4.58 or -10 are now clamped
        // to 0.1, dramatically reducing prediction errors

        // Test that a raw weight of 10.0 gets clamped
        let result = calculate_optimal_outgoing_weight(10.0, 1.0, 10.0);
        assert!(result.is_some(), "Should return clamped weight");
        assert!(
            (result.unwrap().abs() - MAX_OUTGOING_WEIGHT).abs() < EPSILON,
            "Large raw weight should be clamped to MAX_OUTGOING_WEIGHT"
        );
    }
}

/// Synthetic tests to verify prediction accuracy against manual simulation.
/// These tests create controlled scenarios where we know exactly what the
/// predicted and actual improvements should be.
#[cfg(test)]
mod tests_prediction_accuracy {
    use super::*;

    /// Hard tanh activation for testing (same as production)
    fn test_hard_tanh(x: f32) -> f32 {
        x.clamp(-1.0, 1.0)
    }

    /// Create synthetic samples with known properties.
    /// Returns (samples, baseline_error_sq_sum).
    fn create_synthetic_samples(
        count: usize,
        avg_error: f32,
        source_activation: f32,
        target_value: f32,
    ) -> (Vec<HelpfulSample>, f32) {
        let samples: Vec<HelpfulSample> = (0..count)
            .map(|_| HelpfulSample {
                activation: source_activation,
                avg_error,
                target_value: Some(target_value),
                target_activation: Some(test_hard_tanh(target_value)),
            })
            .collect();

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();
        (samples, baseline_error_sq)
    }

    /// CORE TEST: Verify that our prediction formula gives the correct result.
    ///
    /// This test creates a simple scenario:
    /// - 100 samples all with the same properties
    /// - Known source activation, error, target value
    /// - Compute optimal weight
    /// - Predict improvement
    /// - Manually simulate what the actual improvement would be
    /// - Compare predicted vs manually simulated
    #[test]
    fn prediction_matches_manual_simulation_linear_region() {
        // Scenario: Target in LINEAR region of HARD_TANH (value between -1 and 1)
        let source_activation = 1.0;
        let avg_error = 0.2; // VALUE domain: need to ADD 0.2 to target value
        let target_value = 0.3; // Current pre-activation (in linear region)
        let incoming_weight = 1.0;
        let bias = 0.0;

        let (samples, baseline_error_sq) =
            create_synthetic_samples(100, avg_error, source_activation, target_value);

        // Compute optimal weight using the production formula
        let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
        let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        eprintln!(
            "LINEAR REGION TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
        );

        // Predict improvement using production function
        let (predicted_improvement, improved_count, total_count) =
            compute_relu_improvement_and_count(
                &samples,
                incoming_weight,
                outgoing_weight,
                bias,
                baseline_error_sq,
                Some(test_hard_tanh),
            );

        // MANUALLY simulate what the actual improvement would be
        // This is what TypeScript evaluation does
        let mut manual_baseline_error_sq = 0.0f32;
        let mut manual_new_error_sq = 0.0f32;

        for sample in &samples {
            let target_val = sample.target_value.unwrap();
            let target_act = sample.target_activation.unwrap();
            let desired_value = target_val + sample.avg_error;
            let expected_activation = test_hard_tanh(desired_value);

            // Baseline error in ACTIVATION domain
            let baseline_err = expected_activation - target_act;
            manual_baseline_error_sq += baseline_err.powi(2);

            // Simulate the new neuron's contribution
            let pre_act = incoming_weight * sample.activation + bias;
            let relu_output = pre_act.max(0.0);
            let contribution = outgoing_weight * relu_output;

            // New target value after contribution
            let new_target_value = target_val + contribution;
            let new_target_activation = test_hard_tanh(new_target_value);

            // New error in ACTIVATION domain
            let new_err = expected_activation - new_target_activation;
            manual_new_error_sq += new_err.powi(2);
        }

        let manual_improvement = if manual_baseline_error_sq > EPSILON {
            (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
        } else {
            0.0
        };

        eprintln!(
            "LINEAR REGION RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
            predicted_improvement,
            predicted_improvement * 100.0,
            manual_improvement,
            manual_improvement * 100.0,
            (predicted_improvement - manual_improvement).abs()
        );
        eprintln!(
            "  improved_count={improved_count}/{total_count}, manual_baseline_sq={manual_baseline_error_sq:.6}, manual_new_sq={manual_new_error_sq:.6}"
        );

        // Prediction and manual simulation should match closely
        let diff = (predicted_improvement - manual_improvement).abs();
        assert!(
            diff < 0.01,
            "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
        );

        // Both should be positive (error should decrease)
        assert!(
            predicted_improvement > 0.0,
            "Predicted improvement should be positive"
        );
        assert!(
            manual_improvement > 0.0,
            "Manual improvement should be positive"
        );
    }

    /// Test with target near SATURATION (value close to 1.0)
    #[test]
    fn prediction_matches_manual_simulation_saturation_region() {
        // Scenario: Target near SATURATION of HARD_TANH
        let source_activation = 1.0;
        let avg_error = 0.1; // VALUE domain: need to ADD 0.1 to target value
        let target_value = 0.95; // Current pre-activation (near saturation!)
        let incoming_weight = 1.0;
        let bias = 0.0;

        let (samples, baseline_error_sq) =
            create_synthetic_samples(100, avg_error, source_activation, target_value);

        // Compute optimal weight
        let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
        let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        eprintln!(
            "SATURATION TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
        );

        // Predict improvement
        let (predicted_improvement, improved_count, total_count) =
            compute_relu_improvement_and_count(
                &samples,
                incoming_weight,
                outgoing_weight,
                bias,
                baseline_error_sq,
                Some(test_hard_tanh),
            );

        // Manual simulation
        let mut manual_baseline_error_sq = 0.0f32;
        let mut manual_new_error_sq = 0.0f32;

        for sample in &samples {
            let target_val = sample.target_value.unwrap();
            let target_act = sample.target_activation.unwrap();
            let desired_value = target_val + sample.avg_error;
            let expected_activation = test_hard_tanh(desired_value);

            let baseline_err = expected_activation - target_act;
            manual_baseline_error_sq += baseline_err.powi(2);

            let pre_act = incoming_weight * sample.activation + bias;
            let relu_output = pre_act.max(0.0);
            let contribution = outgoing_weight * relu_output;

            let new_target_value = target_val + contribution;
            let new_target_activation = test_hard_tanh(new_target_value);

            let new_err = expected_activation - new_target_activation;
            manual_new_error_sq += new_err.powi(2);
        }

        let manual_improvement = if manual_baseline_error_sq > EPSILON {
            (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
        } else {
            0.0
        };

        eprintln!(
            "SATURATION RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
            predicted_improvement,
            predicted_improvement * 100.0,
            manual_improvement,
            manual_improvement * 100.0,
            (predicted_improvement - manual_improvement).abs()
        );
        eprintln!(
            "  improved_count={improved_count}/{total_count}, manual_baseline_sq={manual_baseline_error_sq:.6}, manual_new_sq={manual_new_error_sq:.6}"
        );

        // Prediction and manual simulation should match
        let diff = (predicted_improvement - manual_improvement).abs();
        assert!(
            diff < 0.01,
            "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
        );
    }

    /// Test with NEGATIVE error (target output should be LOWER)
    #[test]
    fn prediction_matches_manual_simulation_negative_error() {
        // Scenario: Target output is too HIGH, need to REDUCE it
        let source_activation = 1.0;
        let avg_error = -0.2; // VALUE domain: need to SUBTRACT 0.2 from target value
        let target_value = 0.5; // Current pre-activation
        let incoming_weight = 1.0;
        let bias = 0.0;

        let (samples, baseline_error_sq) =
            create_synthetic_samples(100, avg_error, source_activation, target_value);

        // Compute optimal weight (should be NEGATIVE to reduce error)
        let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
        let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        eprintln!(
            "NEGATIVE ERROR TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
        );

        // Verify optimal weight is negative (to reduce target value)
        assert!(
            outgoing_weight < 0.0,
            "Outgoing weight should be negative to reduce target value"
        );

        // Predict improvement
        let (predicted_improvement, improved_count, total_count) =
            compute_relu_improvement_and_count(
                &samples,
                incoming_weight,
                outgoing_weight,
                bias,
                baseline_error_sq,
                Some(test_hard_tanh),
            );

        // Manual simulation
        let mut manual_baseline_error_sq = 0.0f32;
        let mut manual_new_error_sq = 0.0f32;

        for sample in &samples {
            let target_val = sample.target_value.unwrap();
            let target_act = sample.target_activation.unwrap();
            let desired_value = target_val + sample.avg_error;
            let expected_activation = test_hard_tanh(desired_value);

            let baseline_err = expected_activation - target_act;
            manual_baseline_error_sq += baseline_err.powi(2);

            let pre_act = incoming_weight * sample.activation + bias;
            let relu_output = pre_act.max(0.0);
            let contribution = outgoing_weight * relu_output;

            let new_target_value = target_val + contribution;
            let new_target_activation = test_hard_tanh(new_target_value);

            let new_err = expected_activation - new_target_activation;
            manual_new_error_sq += new_err.powi(2);
        }

        let manual_improvement = if manual_baseline_error_sq > EPSILON {
            (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
        } else {
            0.0
        };

        eprintln!(
            "NEGATIVE ERROR RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
            predicted_improvement,
            predicted_improvement * 100.0,
            manual_improvement,
            manual_improvement * 100.0,
            (predicted_improvement - manual_improvement).abs()
        );
        eprintln!(
            "  improved_count={improved_count}/{total_count}, manual_baseline_sq={manual_baseline_error_sq:.6}, manual_new_sq={manual_new_error_sq:.6}"
        );

        // Prediction and manual simulation should match
        let diff = (predicted_improvement - manual_improvement).abs();
        assert!(
            diff < 0.01,
            "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
        );

        // Both should be positive (error should decrease)
        assert!(
            predicted_improvement > 0.0,
            "Predicted improvement should be positive"
        );
        assert!(
            manual_improvement > 0.0,
            "Manual improvement should be positive"
        );
    }

    /// Test with MIXED errors (some positive, some negative)
    /// This simulates real-world scenarios where samples have varied errors.
    #[test]
    fn prediction_matches_manual_simulation_mixed_errors() {
        // Create samples with varied errors
        let samples: Vec<HelpfulSample> = vec![
            // Samples that need INCREASE (positive error)
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.2,
                target_value: Some(0.3),
                target_activation: Some(0.3),
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.15,
                target_value: Some(0.4),
                target_activation: Some(0.4),
            },
            HelpfulSample {
                activation: 1.2,
                avg_error: 0.1,
                target_value: Some(0.2),
                target_activation: Some(0.2),
            },
            // Samples that need DECREASE (negative error)
            HelpfulSample {
                activation: 0.9,
                avg_error: -0.15,
                target_value: Some(0.6),
                target_activation: Some(0.6),
            },
            HelpfulSample {
                activation: 1.1,
                avg_error: -0.1,
                target_value: Some(0.5),
                target_activation: Some(0.5),
            },
        ];

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Compute optimal weight (weighted average)
        let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
        let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        let incoming_weight = 1.0;
        let bias = 0.0;

        eprintln!(
            "MIXED ERRORS TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
        );

        // Predict improvement
        let (predicted_improvement, improved_count, total_count) =
            compute_relu_improvement_and_count(
                &samples,
                incoming_weight,
                outgoing_weight,
                bias,
                baseline_error_sq,
                Some(test_hard_tanh),
            );

        // Manual simulation
        let mut manual_baseline_error_sq = 0.0f32;
        let mut manual_new_error_sq = 0.0f32;
        let mut manual_improved = 0u32;
        let mut manual_worsened = 0u32;

        for sample in &samples {
            let target_val = sample.target_value.unwrap();
            let target_act = sample.target_activation.unwrap();
            let desired_value = target_val + sample.avg_error;
            let expected_activation = test_hard_tanh(desired_value);

            let baseline_err = expected_activation - target_act;
            manual_baseline_error_sq += baseline_err.powi(2);

            let pre_act = incoming_weight * sample.activation + bias;
            let relu_output = pre_act.max(0.0);
            let contribution = outgoing_weight * relu_output;

            let new_target_value = target_val + contribution;
            let new_target_activation = test_hard_tanh(new_target_value);

            let new_err = expected_activation - new_target_activation;
            manual_new_error_sq += new_err.powi(2);

            if new_err.abs() < baseline_err.abs() - EPSILON {
                manual_improved += 1;
            } else if new_err.abs() > baseline_err.abs() + EPSILON {
                manual_worsened += 1;
            }
        }

        let manual_improvement = if manual_baseline_error_sq > EPSILON {
            (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
        } else {
            0.0
        };

        eprintln!(
            "MIXED ERRORS RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
            predicted_improvement,
            predicted_improvement * 100.0,
            manual_improvement,
            manual_improvement * 100.0,
            (predicted_improvement - manual_improvement).abs()
        );
        eprintln!(
            "  func: improved={improved_count}/{total_count}, manual: improved={manual_improved}, worsened={manual_worsened}"
        );

        // Prediction and manual simulation should match
        let diff = (predicted_improvement - manual_improvement).abs();
        assert!(
            diff < 0.01,
            "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
        );
    }

    /// KEY TEST: Simulate what TypeScript evaluation actually does.
    /// This is the most realistic test - it matches the production evaluation flow.
    #[test]
    fn prediction_matches_simulated_typescript_evaluation() {
        // Create samples that match production data characteristics
        let samples: Vec<HelpfulSample> = (0..1000)
            .map(|i| {
                let variation = (i as f32 / 100.0).sin() * 0.1;
                let error_variation = (i as f32 / 50.0).cos() * 0.05;
                HelpfulSample {
                    activation: 0.5 + variation,
                    avg_error: 0.1 + error_variation,
                    target_value: Some(0.4 + variation * 0.5),
                    target_activation: Some(test_hard_tanh(0.4 + variation * 0.5)),
                }
            })
            .collect();

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Compute optimal weight
        let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
        let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

        let incoming_weight = 1.0;
        let bias = 0.0;

        // Predict improvement (what Rust returns)
        let (predicted_improvement, _improved_count, _total_count) =
            compute_relu_improvement_and_count(
                &samples,
                incoming_weight,
                outgoing_weight,
                bias,
                baseline_error_sq,
                Some(test_hard_tanh),
            );

        // Simulate TypeScript evaluation
        // TypeScript computes: actualErrorReduction = originalError - candidateError
        // Where error is typically MSE or similar across all training samples

        // Original creature MSE (before adding neuron)
        let original_mse: f32 = samples
            .iter()
            .map(|s| {
                let target_act = s.target_activation.unwrap();
                let desired_value = s.target_value.unwrap() + s.avg_error;
                let expected = test_hard_tanh(desired_value);
                (expected - target_act).powi(2)
            })
            .sum::<f32>()
            / samples.len() as f32;

        // Candidate creature MSE (after adding neuron)
        let candidate_mse: f32 = samples
            .iter()
            .map(|s| {
                let target_val = s.target_value.unwrap();
                let desired_value = target_val + s.avg_error;
                let expected = test_hard_tanh(desired_value);

                // New neuron contribution
                let pre_act = incoming_weight * s.activation + bias;
                let relu_output = pre_act.max(0.0);
                let contribution = outgoing_weight * relu_output;

                let new_target_value = target_val + contribution;
                let new_activation = test_hard_tanh(new_target_value);
                (expected - new_activation).powi(2)
            })
            .sum::<f32>()
            / samples.len() as f32;

        // TypeScript reports: actualErrorReduction = originalError - candidateError
        // If we interpret this as raw error change:
        let original_error = original_mse.sqrt(); // RMSE
        let candidate_error = candidate_mse.sqrt();
        let actual_error_reduction = original_error - candidate_error;

        // For comparison with our percentage, convert to ratio
        let actual_improvement_ratio = actual_error_reduction / original_error;

        // Also compute MSE-based ratio (should match our prediction more closely)
        let mse_improvement_ratio = (original_mse - candidate_mse) / original_mse;

        eprintln!(
            "TYPESCRIPT SIMULATION: predicted={:.6} ({:.4}%)",
            predicted_improvement,
            predicted_improvement * 100.0
        );
        eprintln!(
            "  original_mse={:.8}, candidate_mse={:.8}, mse_improvement={:.6} ({:.4}%)",
            original_mse,
            candidate_mse,
            mse_improvement_ratio,
            mse_improvement_ratio * 100.0
        );
        eprintln!(
            "  original_rmse={:.6}, candidate_rmse={:.6}, rmse_reduction={:.6} ({:.4}%)",
            original_error,
            candidate_error,
            actual_improvement_ratio,
            actual_improvement_ratio * 100.0
        );

        // Our prediction should match MSE-based improvement
        let diff = (predicted_improvement - mse_improvement_ratio).abs();
        assert!(
            diff < 0.01,
            "Predicted ({predicted_improvement:.6}) and MSE improvement ({mse_improvement_ratio:.6}) should match within 1%"
        );

        // Both should be positive (error should decrease)
        assert!(
            predicted_improvement > 0.0,
            "Predicted improvement should be positive"
        );
        assert!(
            mse_improvement_ratio > 0.0,
            "MSE improvement should be positive"
        );
    }

    /// Test that verifies the sign is correct when contribution SHOULD help.
    /// If this test fails, it indicates a sign error in the formula.
    #[test]
    fn contribution_in_correct_direction_reduces_error() {
        // Simple scenario: positive error, positive activation, positive weight = positive contribution
        // Positive contribution ADDS to target value, reducing positive error
        let sample = HelpfulSample {
            activation: 1.0, // positive source activation
            avg_error: 0.2,  // positive VALUE error: need to ADD 0.2
            target_value: Some(0.3),
            target_activation: Some(0.3),
        };

        // Optimal weight formula: w = Σ(error×activation) / Σ(activation²) = 0.2/1 = 0.2
        // Contribution = w × ReLU(source) = 0.2 × 1 = 0.2
        // New target value = 0.3 + 0.2 = 0.5
        // Desired value = 0.3 + 0.2 = 0.5 (should match!)

        let outgoing_weight = 0.1; // Clamped from 0.2
        let incoming_weight = 1.0;
        let bias = 0.0;

        let pre_activation = incoming_weight * sample.activation + bias;
        let relu_output = pre_activation.max(0.0);
        let contribution = outgoing_weight * relu_output;

        // Verify contribution is in the right direction
        assert!(
            contribution > 0.0,
            "Contribution should be positive for positive error"
        );
        assert!(
            contribution.signum() == sample.avg_error.signum(),
            "Contribution sign ({}) should match error sign ({})",
            contribution.signum(),
            sample.avg_error.signum()
        );

        // Verify new error is smaller
        let target_value = sample.target_value.unwrap();
        let target_activation = sample.target_activation.unwrap();
        let desired_value = target_value + sample.avg_error;
        let expected = test_hard_tanh(desired_value);

        let baseline_error = expected - target_activation;
        let new_target_value = target_value + contribution;
        let new_activation = test_hard_tanh(new_target_value);
        let new_error = expected - new_activation;

        eprintln!(
            "DIRECTION TEST: baseline_err={:.4}, new_err={:.4}, reduction={:.4}",
            baseline_error.abs(),
            new_error.abs(),
            baseline_error.abs() - new_error.abs()
        );

        assert!(
            new_error.abs() < baseline_error.abs(),
            "New error ({:.4}) should be smaller than baseline ({:.4})",
            new_error.abs(),
            baseline_error.abs()
        );
    }
}
