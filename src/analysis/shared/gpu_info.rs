//! GPU adapter information and zero-copy buffer configuration.
//!
//! Contains [`GpuAdapterInfo`] for reporting GPU hardware capabilities,
//! [`GpuDeviceType`] for classifying GPU device types, and
//! [`ZeroCopyBufferConfig`] for controlling zero-copy buffer sharing on
//! unified memory architectures (Apple Silicon).

// =============================================================================
// GPU Info and Zero-Copy Types (Issue #228)
// =============================================================================

/// Information about the GPU adapter being used.
///
/// Provides details about the GPU hardware and its capabilities,
/// particularly for unified memory detection (Apple Silicon).
#[derive(Debug, Clone)]
pub struct GpuAdapterInfo {
    /// Human-readable name of the GPU (e.g., "Apple M4 Pro").
    pub name: String,
    /// Device type (discrete, integrated, software, etc.).
    pub device_type: GpuDeviceType,
    /// Whether the GPU has unified memory architecture.
    ///
    /// On unified memory systems (Apple Silicon), CPU and GPU share the same
    /// physical memory, enabling zero-copy buffer sharing.
    pub has_unified_memory: bool,
    /// Whether zero-copy buffer sharing is currently enabled.
    ///
    /// This may differ from `has_unified_memory` if the user has explicitly
    /// disabled zero-copy via environment variable.
    pub zero_copy_enabled: bool,
}

/// GPU device type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    /// Discrete GPU (separate VRAM, e.g., NVIDIA/AMD cards).
    Discrete,
    /// Integrated GPU (shares system memory, e.g., Intel integrated).
    Integrated,
    /// Software/CPU-based rendering (fallback).
    Software,
    /// Virtual GPU (e.g., cloud instances).
    Virtual,
    /// Other/unknown device type.
    Other,
}

impl From<wgpu::DeviceType> for GpuDeviceType {
    fn from(device_type: wgpu::DeviceType) -> Self {
        match device_type {
            wgpu::DeviceType::DiscreteGpu => GpuDeviceType::Discrete,
            wgpu::DeviceType::IntegratedGpu => GpuDeviceType::Integrated,
            wgpu::DeviceType::Cpu => GpuDeviceType::Software,
            wgpu::DeviceType::VirtualGpu => GpuDeviceType::Virtual,
            wgpu::DeviceType::Other => GpuDeviceType::Other,
        }
    }
}

/// Configuration for zero-copy buffer sharing.
///
/// Zero-copy buffer sharing eliminates unnecessary CPU-GPU data copies on
/// unified memory architectures like Apple Silicon. This configuration
/// allows fine-grained control over when zero-copy is used.
#[derive(Debug, Clone)]
pub struct ZeroCopyBufferConfig {
    /// Whether zero-copy is explicitly enabled or disabled via environment variable.
    ///
    /// - `Some(true)`: Force-enabled via NEAT_AI_DISCOVERY_ZERO_COPY=1
    /// - `Some(false)`: Force-disabled via NEAT_AI_DISCOVERY_ZERO_COPY=0
    /// - `None`: Auto-detect based on unified memory support
    force_enabled: Option<bool>,
    /// Number of buffers in the ring buffer for pipelining.
    ///
    /// Triple buffering (3) is the default, allowing one buffer for CPU writes,
    /// one for GPU reads, and one in flight.
    buffer_count: usize,
}

impl Default for ZeroCopyBufferConfig {
    fn default() -> Self {
        Self {
            force_enabled: None,
            buffer_count: 3, // Triple buffering
        }
    }
}

impl ZeroCopyBufferConfig {
    /// Create configuration from environment variables.
    ///
    /// Delegates to [`crate::config::zero_copy_override()`].
    pub fn from_env() -> Self {
        Self {
            force_enabled: crate::config::zero_copy_override(),
            buffer_count: 3,
        }
    }

    /// Whether zero-copy should be enabled for the given hardware.
    ///
    /// Uses force setting if present, otherwise auto-detects based on
    /// unified memory support.
    pub fn enabled(&self) -> bool {
        // Will be computed with actual hardware check
        self.force_enabled.unwrap_or(false)
    }

    /// Whether zero-copy was explicitly enabled or disabled.
    pub fn force_enabled(&self) -> Option<bool> {
        self.force_enabled
    }

    /// Number of buffers in the ring buffer.
    pub fn buffer_count(&self) -> usize {
        self.buffer_count
    }

    /// Check if enabled with hardware detection.
    pub fn enabled_with_hardware(&self, has_unified_memory: bool) -> bool {
        self.force_enabled.unwrap_or(has_unified_memory)
    }
}
