//! Regression tests for Issue #2332 — the GPU capability probe
//! (`check_gpu_availability` / `get_adapter_info_internal`, both reached via
//! `OnceLock::get_or_init`) used to call `pollster::block_on` with no
//! deadline, so a hung driver wedged every caller forever. These tests cover
//! the bounded seam added to fix that: `run_gpu_probe_with_timeout` and
//! `check_gpu_availability_with`.
//!
//! Every test uses an isolated `GpuCircuitBreaker::new()` — never the global
//! breaker — and none of them touch real GPU hardware.

use std::time::Duration;

use crate::analysis::gpu::analyzer::check_gpu_availability_with;
use crate::analysis::gpu::breaker::{GpuCircuitBreaker, GpuTripReason};
use crate::analysis::gpu::device::{
    BoundedProbe, GpuAvailabilityResult, run_gpu_probe_with_timeout,
};
use crate::analysis::gpu::shaders::GPU_INIT_TIMEOUT_SECS;

/// Short timeout for probes that must actually time out — kept tight so the
/// hung-probe tests stay fast.
const HUNG_PROBE_TIMEOUT: Duration = Duration::from_millis(100);

/// Generous timeout for probes that resolve or panic immediately — a long
/// deadline costs nothing since they never wait for it, and it avoids flaking
/// on a loaded CI runner where thread spawn/scheduling can exceed 100 ms.
const RESOLVING_PROBE_TIMEOUT: Duration = Duration::from_secs(30);

#[test]
fn hung_probe_times_out_and_trips_breaker() {
    let breaker = GpuCircuitBreaker::new();
    let (tx, rx) = crossbeam_channel::unbounded::<()>();

    let result = check_gpu_availability_with(HUNG_PROBE_TIMEOUT, &breaker, move || {
        // Never resolves until the sender is dropped below.
        let _ = rx.recv();
        GpuAvailabilityResult {
            available: true,
            reason: None,
            is_error: false,
            device_type: None,
        }
    });

    assert!(!result.available);
    assert!(result.is_error);
    assert_eq!(
        result.reason,
        Some(format!(
            "GPU capability probe timed out after {GPU_INIT_TIMEOUT_SECS}s"
        ))
    );
    assert!(result.device_type.is_none());
    assert!(breaker.is_tripped());
    assert_eq!(breaker.trip_reason(), Some(GpuTripReason::InitTimeout));

    // Good hygiene: let the leaked thread exit instead of leaving it parked
    // on `recv()` for the rest of the test binary's life.
    drop(tx);
}

#[test]
fn resolving_probe_returns_its_result_without_tripping() {
    let breaker = GpuCircuitBreaker::new();

    let result = check_gpu_availability_with(RESOLVING_PROBE_TIMEOUT, &breaker, || {
        GpuAvailabilityResult {
            available: true,
            reason: None,
            is_error: false,
            device_type: Some(crate::analysis::shared::GpuDeviceType::Discrete),
        }
    });

    assert!(result.available);
    assert!(!result.is_error);
    assert!(result.reason.is_none());
    assert_eq!(
        result.device_type,
        Some(crate::analysis::shared::GpuDeviceType::Discrete)
    );
    assert!(!breaker.is_tripped());
    assert!(breaker.trip_reason().is_none());
}

#[test]
fn panicking_probe_fails_without_tripping() {
    let breaker = GpuCircuitBreaker::new();

    let result = check_gpu_availability_with(
        RESOLVING_PROBE_TIMEOUT,
        &breaker,
        || -> GpuAvailabilityResult {
            panic!("probe exploded");
        },
    );

    assert!(!result.available);
    assert!(
        result
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("GPU capability probe thread failed")),
        "got: {:?}",
        result.reason
    );
    assert!(
        !breaker.is_tripped(),
        "a panicking probe is not evidence the GPU itself is wedged"
    );
}

#[test]
fn run_gpu_probe_with_timeout_times_out_on_a_hung_closure() {
    let breaker = GpuCircuitBreaker::new();
    let (tx, rx) = crossbeam_channel::unbounded::<()>();

    let outcome = run_gpu_probe_with_timeout(HUNG_PROBE_TIMEOUT, &breaker, move || {
        let _ = rx.recv();
        42
    });

    assert!(matches!(outcome, BoundedProbe::TimedOut));
    assert!(breaker.is_tripped());
    assert_eq!(breaker.trip_reason(), Some(GpuTripReason::InitTimeout));

    drop(tx);
}

#[test]
fn run_gpu_probe_with_timeout_completes_on_a_resolving_closure() {
    let breaker = GpuCircuitBreaker::new();

    let outcome = run_gpu_probe_with_timeout(RESOLVING_PROBE_TIMEOUT, &breaker, || 42);

    assert!(matches!(outcome, BoundedProbe::Completed(42)));
    assert!(!breaker.is_tripped());
}
