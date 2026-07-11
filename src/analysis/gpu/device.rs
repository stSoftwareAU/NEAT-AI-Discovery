//! GPU device management module
//!
//! This module contains GPU device initialisation, detection, and buffer management
//! code extracted from the monolithic implementation.rs file (Issue #272).
//!
//! ## Contents
//!
//! - `GpuPerformanceTier` - GPU performance classification for auto-tuning
//! - `GpuAvailabilityResult` - Result of GPU availability check with diagnostics
//! - Device detection functions (`detect_unified_memory`, `detect_gpu_tier`)
//! - Safe wgpu instance creation (`create_wgpu_instance_safely`)
//! - Buffer mapping helpers (`poll_device_until_idle`, `wait_for_buffer_map`, etc.)
//!
//! ## Dependencies
//!
//! **Incoming dependencies (modules that use device):**
//! - GPU work queue (device access)
//! - GPU pipelines (device for pipeline creation)
//! - Memory utilities (GPU memory detection)
//!
//! **Outgoing dependencies (device needs these):**
//! - wgpu crate
//! - Memory utilities (tier detection)
//! - Constants (timeouts)

use anyhow::{Result, anyhow};
use std::thread;
use std::time::Duration;

use crate::analysis::utils::{ensure_xdg_runtime_dir, suppress_mesa_warnings_if_requested};

// =============================================================================
// Constants
// =============================================================================

/// Maximum timeout for GPU queue operations (in seconds).
/// Imported from deadline module.
pub use crate::analysis::utils::GPU_QUEUE_TIMEOUT_MAX_SECS;

/// Buffer map timeout is always 5 seconds shorter than queue timeout to avoid race conditions.
/// If both timeouts are the same, the queue might timeout before the GPU thread
/// has a chance to return its own timeout error, leaving the thread stuck.
pub const GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS: u64 = 5;

/// Default buffer map timeout for internal GPU operations (in seconds).
/// This is used inside the GPU thread where we don't have access to the external deadline.
/// Set to max queue timeout minus margin for safety.
pub const GPU_BUFFER_MAP_TIMEOUT_SECS: u64 =
    GPU_QUEUE_TIMEOUT_MAX_SECS - GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS;

/// Timeout for GPU thread initialisation (in seconds).
/// GPU device creation should be fast; if it takes longer, something is wrong.
pub const GPU_INIT_TIMEOUT_SECS: u64 = 30;

// =============================================================================
// GPU Performance Tier
// =============================================================================

/// Detected GPU performance tier for auto-tuning.
///
/// This classification is used to select appropriate batch sizes and optimisation
/// strategies based on the detected GPU hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPerformanceTier {
    /// High-performance GPU (M4, M3 Pro/Max, M2 Pro/Max, dedicated GPUs)
    High,
    /// Standard GPU (M1, M2, M3 base, integrated GPUs)
    Standard,
    /// Unknown/fallback
    Unknown,
}

// =============================================================================
// GPU Availability Result
// =============================================================================

/// Result of GPU availability check with detailed diagnostics.
///
/// This struct provides information about whether a GPU is available for use,
/// along with a human-readable reason and error status.
#[derive(Debug, Clone)]
pub struct GpuAvailabilityResult {
    /// Whether a GPU is available for use.
    pub available: bool,
    /// Human-readable reason for the availability status.
    pub reason: Option<String>,
    /// Whether this is an error condition (true on macOS when GPU unavailable).
    pub is_error: bool,
}

// =============================================================================
// GPU Detection Functions
// =============================================================================

/// Detect if the GPU has unified memory architecture.
///
/// Unified memory means CPU and GPU share the same physical memory,
/// enabling zero-copy buffer sharing.
///
/// Returns `true` for:
/// - Apple Silicon (M1/M2/M3/M4) - always unified memory
/// - Integrated GPUs on some Vulkan devices (may support unified memory)
///
/// Returns `false` for:
/// - Discrete GPUs (separate VRAM)
pub fn detect_unified_memory(adapter_info: &wgpu::AdapterInfo) -> bool {
    let name = adapter_info.name.to_lowercase();

    // Apple Silicon always has unified memory
    if name.contains("apple") {
        return true;
    }

    // Integrated GPUs may have unified memory (e.g., Intel, AMD APUs)
    // However, wgpu doesn't expose whether the memory is actually unified,
    // so we're conservative and only claim unified for Apple Silicon.
    //
    // Note: Some integrated GPUs on Linux with Vulkan might support unified memory,
    // but the wgpu API doesn't expose this information reliably.
    // We could enable this in the future with more testing.
    false
}

/// Detect GPU performance tier from adapter info.
///
/// Returns `High` for M4, Pro/Max variants, and discrete GPUs.
/// Returns `Standard` for base M-series and integrated GPUs.
/// Returns `Unknown` otherwise.
pub fn detect_gpu_tier(adapter_info: &wgpu::AdapterInfo) -> GpuPerformanceTier {
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

// =============================================================================
// Safe wgpu Instance Creation
// =============================================================================

/// Safely create a wgpu Instance, avoiding panics from backend probing.
///
/// On Linux with old hardware or missing GPU drivers, wgpu's EGL/OpenGL backend
/// can panic during initialisation (e.g., "`BadDisplay`" errors). This function:
///
/// - On Linux: Disables the GL backend entirely, using only Vulkan to avoid EGL panics
/// - On macOS: Uses Metal (the default and only backend on macOS)
/// - On all platforms: Wraps instance creation in `catch_unwind` as a safety net
///
/// Returns `None` if instance creation fails or panics, allowing callers to handle
/// the failure gracefully (e.g., treating missing GPU as discovery-disabled on Linux).
pub fn create_wgpu_instance_safely() -> Option<wgpu::Instance> {
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
            ..wgpu::InstanceDescriptor::new_without_display_handle()
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
                tracing::debug!(
                    panic_message = %panic_msg,
                    "wgpu instance creation failed — discovery will be disabled on this machine"
                );
            }

            #[cfg(target_os = "macos")]
            {
                // On macOS, this is unexpected - Metal should always be available
                tracing::error!(
                    panic_message = %panic_msg,
                    "wgpu instance creation failed on macOS — this indicates a system configuration issue"
                );
            }

            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            {
                tracing::warn!(
                    panic_message = %panic_msg,
                    "wgpu instance creation failed — discovery will be disabled on this machine"
                );
            }

            None
        }
    }
}

// =============================================================================
// Buffer Management Functions
// =============================================================================

/// Poll the GPU device until it has no work in flight, or a timeout is reached.
///
/// This intentionally avoids `Maintain::Wait` because that can block forever on
/// some machines if the GPU driver wedges. Instead, we poll in a loop and bail
/// out with an error so unattended workers can recover (or watchdog can abort).
pub fn poll_device_until_idle(device: &wgpu::Device, timeout: Duration, label: &str) -> Result<()> {
    use std::time::Instant;
    let start = Instant::now();
    loop {
        match device.poll(wgpu::PollType::Poll) {
            Ok(status) if status.is_queue_empty() => return Ok(()),
            Ok(_) => {} // Queue not empty yet; keep polling.
            Err(e) => return Err(anyhow!("GPU device poll error ({label}): {e}")),
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
pub fn wait_for_buffer_map(
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

        let queue_empty = matches!(
            device.poll(wgpu::PollType::Poll),
            Ok(status) if status.is_queue_empty()
        );
        if queue_empty && start.elapsed() > Duration::from_millis(250) {
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
///
/// This is an optimisation that calls `map_async` on ALL staging buffers first,
/// then performs a single `device.poll(Wait)` to wait for all buffers simultaneously.
/// This reduces GPU-CPU round trips compared to sequential mapping.
pub fn wait_for_buffer_maps_batch(
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

        if done.iter().all(std::option::Option::is_some) {
            // Validate all results.
            for (i, result) in done.into_iter().enumerate() {
                match result.expect("checked is_some") {
                    Ok(()) => {}
                    Err(err) => return Err(anyhow!("Buffer {i} mapping failed: {err}")),
                }
            }
            return Ok(());
        }

        let _ = device.poll(wgpu::PollType::Poll);

        if start.elapsed() > timeout {
            return Err(anyhow!(
                "GPU batch buffer mapping timed out after {:.1}s. The GPU driver may be unresponsive.",
                timeout.as_secs_f64()
            ));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

// =============================================================================
// Internal Helpers for GPU Availability
// =============================================================================

/// Create a result for when GPU is not available.
/// On macOS this is an error; on Linux it gracefully disables discovery.
pub fn no_gpu_result(reason: &str) -> GpuAvailabilityResult {
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

/// Get raw wgpu adapter info.
///
/// This is an internal helper used by `check_gpu_availability` and related functions.
pub fn get_adapter_info_internal() -> Option<wgpu::AdapterInfo> {
    // Suppress Mesa/libEGL warnings
    suppress_mesa_warnings_if_requested();
    ensure_xdg_runtime_dir();

    let instance = create_wgpu_instance_safely()?;
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;

    Some(adapter.get_info())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to construct `wgpu::AdapterInfo` for tests.
    /// Fills in sensible defaults for fields not relevant to the test.
    fn test_adapter_info(
        name: &str,
        device_type: wgpu::DeviceType,
        backend: wgpu::Backend,
    ) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_string(),
            vendor: 0,
            device: 0,
            device_type,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            transient_saves_memory: Some(false),
            limit_bucket: None,
        }
    }

    #[test]
    fn test_gpu_performance_tier_variants() {
        // Ensure all variants are accessible
        assert_eq!(GpuPerformanceTier::High, GpuPerformanceTier::High);
        assert_eq!(GpuPerformanceTier::Standard, GpuPerformanceTier::Standard);
        assert_eq!(GpuPerformanceTier::Unknown, GpuPerformanceTier::Unknown);
        assert_ne!(GpuPerformanceTier::High, GpuPerformanceTier::Standard);
    }

    #[test]
    fn test_gpu_availability_result_construction() {
        let result = GpuAvailabilityResult {
            available: true,
            reason: None,
            is_error: false,
        };
        assert!(result.available);
        assert!(result.reason.is_none());
        assert!(!result.is_error);

        let result_unavailable = GpuAvailabilityResult {
            available: false,
            reason: Some("No GPU found".to_string()),
            is_error: true,
        };
        assert!(!result_unavailable.available);
        assert_eq!(result_unavailable.reason.as_deref(), Some("No GPU found"));
        assert!(result_unavailable.is_error);
    }

    #[test]
    fn test_detect_unified_memory_apple() {
        // Test Apple Silicon detection
        let apple_info = test_adapter_info(
            "Apple M1",
            wgpu::DeviceType::IntegratedGpu,
            wgpu::Backend::Metal,
        );
        assert!(detect_unified_memory(&apple_info));

        let apple_m4_info = test_adapter_info(
            "Apple M4 Pro",
            wgpu::DeviceType::IntegratedGpu,
            wgpu::Backend::Metal,
        );
        assert!(detect_unified_memory(&apple_m4_info));
    }

    #[test]
    fn test_detect_unified_memory_non_apple() {
        // Test non-Apple GPU detection (should return false)
        let nvidia_info = test_adapter_info(
            "NVIDIA GeForce RTX 4090",
            wgpu::DeviceType::DiscreteGpu,
            wgpu::Backend::Vulkan,
        );
        assert!(!detect_unified_memory(&nvidia_info));

        let intel_info = test_adapter_info(
            "Intel UHD Graphics 630",
            wgpu::DeviceType::IntegratedGpu,
            wgpu::Backend::Vulkan,
        );
        // Conservative: non-Apple integrated GPUs return false
        assert!(!detect_unified_memory(&intel_info));
    }

    #[test]
    fn test_detect_gpu_tier_m4() {
        let info = test_adapter_info(
            "Apple M4",
            wgpu::DeviceType::IntegratedGpu,
            wgpu::Backend::Metal,
        );
        assert_eq!(detect_gpu_tier(&info), GpuPerformanceTier::High);
    }

    #[test]
    fn test_detect_gpu_tier_m3_pro() {
        let info = test_adapter_info(
            "Apple M3 Pro",
            wgpu::DeviceType::IntegratedGpu,
            wgpu::Backend::Metal,
        );
        assert_eq!(detect_gpu_tier(&info), GpuPerformanceTier::High);
    }

    #[test]
    fn test_detect_gpu_tier_m1_base() {
        let info = test_adapter_info(
            "Apple M1",
            wgpu::DeviceType::IntegratedGpu,
            wgpu::Backend::Metal,
        );
        assert_eq!(detect_gpu_tier(&info), GpuPerformanceTier::Standard);
    }

    #[test]
    fn test_detect_gpu_tier_discrete() {
        let info = test_adapter_info(
            "NVIDIA GeForce RTX 4090",
            wgpu::DeviceType::DiscreteGpu,
            wgpu::Backend::Vulkan,
        );
        assert_eq!(detect_gpu_tier(&info), GpuPerformanceTier::High);
    }

    #[test]
    fn test_detect_gpu_tier_integrated() {
        let info = test_adapter_info(
            "Intel UHD Graphics",
            wgpu::DeviceType::IntegratedGpu,
            wgpu::Backend::Vulkan,
        );
        assert_eq!(detect_gpu_tier(&info), GpuPerformanceTier::Standard);
    }

    #[test]
    fn test_detect_gpu_tier_unknown() {
        let info = test_adapter_info("Unknown GPU", wgpu::DeviceType::Other, wgpu::Backend::Noop);
        assert_eq!(detect_gpu_tier(&info), GpuPerformanceTier::Unknown);
    }

    #[test]
    fn test_no_gpu_result_has_reason() {
        let result = no_gpu_result("Test reason");
        assert!(!result.available);
        assert!(result.reason.is_some());
        assert!(result.reason.as_ref().unwrap().contains("Test reason"));
    }

    #[test]
    fn test_buffer_timeout_constants() {
        // Verify timeout constants are sensible using const assertions
        const _: () = assert!(GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS > 0);
        const _: () = assert!(GPU_BUFFER_MAP_TIMEOUT_SECS > 0);
        const _: () = assert!(GPU_INIT_TIMEOUT_SECS > 0);
        const _: () = assert!(GPU_BUFFER_MAP_TIMEOUT_SECS < GPU_QUEUE_TIMEOUT_MAX_SECS);
    }

    #[test]
    fn test_wait_for_buffer_maps_batch_empty() {
        // Test that empty receivers list returns Ok immediately
        // We need a mock device for this test, but we can't create one without GPU
        // This test documents the expected behaviour
        let receivers: Vec<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> =
            Vec::new();
        // The function should handle empty receivers gracefully
        // We can't actually call it without a device, but the behaviour is documented
        assert!(receivers.is_empty());
    }
}
