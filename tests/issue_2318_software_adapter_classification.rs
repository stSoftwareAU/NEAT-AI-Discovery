//! Regression tests pinning the Issue #2318 fix: a software (CPU) wgpu
//! adapter must not silently pass the GPU capability gate as an
//! indistinguishable-from-real-GPU verdict.
//!
//! `GpuDeviceType` classifies `wgpu::DeviceType::Cpu` as
//! [`GpuDeviceType::Software`] and exposes `is_software()` so callers can
//! branch on it, and `GpuAdapterInfoJson` (the `gpuInfo` field on analysis
//! responses) now carries `device_type`, serialised as `deviceType`, so the
//! device type is no longer dropped at the FFI boundary.

use neat_ai_discovery::analysis::shared::GpuDeviceType;
use neat_ai_discovery::ffi_types::GpuAdapterInfoJson;

/// Issue #2318: the software (CPU) wgpu device type must classify as
/// `GpuDeviceType::Software`, with `is_software()` reporting `true`.
#[test]
fn software_cpu_adapter_is_classified_as_software() {
    let device_type = GpuDeviceType::from(wgpu::DeviceType::Cpu);

    assert_eq!(device_type, GpuDeviceType::Software);
    assert!(device_type.is_software());
}

/// Issue #2318: real (and virtual/other) GPU device types must never be
/// classified as software, so controllers gating on `softwareAdapter` do not
/// reject genuine GPU hardware.
#[test]
fn hardware_adapters_are_not_classified_as_software() {
    let hardware_types = [
        wgpu::DeviceType::DiscreteGpu,
        wgpu::DeviceType::IntegratedGpu,
        wgpu::DeviceType::VirtualGpu,
        wgpu::DeviceType::Other,
    ];

    for wgpu_device_type in hardware_types {
        let device_type = GpuDeviceType::from(wgpu_device_type);
        assert!(
            !device_type.is_software(),
            "{wgpu_device_type:?} must not be classified as software, got {device_type:?}"
        );
    }
}

/// Issue #2318: `GpuAdapterInfoJson` (the `gpuInfo` field on analysis
/// responses) must serialise `device_type` as `deviceType`, so a software
/// adapter is no longer indistinguishable from a real GPU at the FFI
/// boundary.
#[test]
fn gpu_info_json_carries_device_type() {
    let info = GpuAdapterInfoJson {
        name: "llvmpipe (LLVM 17.0.6, 256 bits)".to_string(),
        device_type: GpuDeviceType::Software,
        unified_memory: false,
        zero_copy_enabled: false,
    };

    let value = serde_json::to_value(&info).expect("GpuAdapterInfoJson must serialise");

    assert_eq!(value["deviceType"], "software");
}
