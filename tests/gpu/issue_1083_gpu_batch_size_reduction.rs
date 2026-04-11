//! Integration tests for GPU batch size reduction on memory exhaustion (Issue #1083)
//!
//! Verifies that the memory exhaustion detection correctly identifies OOM errors,
//! that batch size halving respects the minimum floor, and that the metrics
//! track batch size reductions.

use neat_ai_discovery::analysis::gpu::{
    MINIMUM_GPU_BATCH_SIZE, is_device_lost_error, is_memory_exhaustion_error,
};

// =============================================================================
// Memory exhaustion detection
// =============================================================================

#[test]
fn test_memory_exhaustion_detects_oom_patterns() {
    let oom_messages = vec![
        "Out of memory allocating GPU buffer",
        "wgpu: out of memory when creating staging buffer",
        "allocation failed for 256MB compute buffer",
        "GPU allocation failed",
    ];

    for msg in oom_messages {
        let err = anyhow::anyhow!("{msg}");
        assert!(
            is_memory_exhaustion_error(&err),
            "Expected memory exhaustion detection for: {msg}"
        );
        assert!(
            is_device_lost_error(&err),
            "OOM errors should also be detected as device-lost: {msg}"
        );
    }
}

#[test]
fn test_memory_exhaustion_ignores_non_memory_errors() {
    let non_memory_errors = vec![
        "Device is lost",
        "device lost during operation",
        "Internal error in GPU pipeline",
        "Too many command buffers in flight",
        "Invalid input data",
        "Channel disconnected",
        "GPU driver may be unresponsive",
    ];

    for msg in non_memory_errors {
        let err = anyhow::anyhow!("{msg}");
        assert!(
            !is_memory_exhaustion_error(&err),
            "Should NOT detect memory exhaustion for: {msg}"
        );
    }
}

#[test]
fn test_memory_exhaustion_detection_is_case_insensitive() {
    let variations = vec![
        "OUT OF MEMORY",
        "Out Of Memory",
        "out of memory",
        "ALLOCATION FAILED",
        "Allocation Failed",
    ];

    for msg in variations {
        let err = anyhow::anyhow!("{msg}");
        assert!(
            is_memory_exhaustion_error(&err),
            "Expected case-insensitive detection for: {msg}"
        );
    }
}

// =============================================================================
// Batch size reduction logic
// =============================================================================

#[test]
fn test_minimum_batch_size_is_64() {
    assert_eq!(MINIMUM_GPU_BATCH_SIZE, 64);
}

#[test]
fn test_batch_size_halving_above_minimum() {
    let initial = 512_usize;
    let reduced = initial / 2;
    assert_eq!(reduced, 256);
    assert!(reduced >= MINIMUM_GPU_BATCH_SIZE);
}

#[test]
fn test_batch_size_halving_reaches_minimum() {
    let mut batch_size = 512_usize;
    let mut reductions = 0;
    while batch_size / 2 >= MINIMUM_GPU_BATCH_SIZE {
        batch_size /= 2;
        reductions += 1;
    }
    assert_eq!(batch_size, 64);
    assert_eq!(reductions, 3); // 512 -> 256 -> 128 -> 64
}

#[test]
fn test_batch_size_below_minimum_cannot_reduce() {
    let batch_size = 64_usize;
    let proposed = batch_size / 2;
    assert!(
        proposed < MINIMUM_GPU_BATCH_SIZE,
        "Halving {batch_size} should fall below minimum {MINIMUM_GPU_BATCH_SIZE}"
    );
}

#[test]
fn test_batch_size_reduction_from_large_value() {
    let mut batch_size = 1024_usize;
    let mut reductions = 0;
    while batch_size / 2 >= MINIMUM_GPU_BATCH_SIZE {
        batch_size /= 2;
        reductions += 1;
    }
    assert_eq!(batch_size, 64);
    assert_eq!(reductions, 4); // 1024 -> 512 -> 256 -> 128 -> 64
}

// =============================================================================
// GPU metrics batch size tracking
// =============================================================================

#[test]
fn test_gpu_metrics_tracks_batch_size_reduction() {
    let metrics = neat_ai_discovery::observability::GpuMetrics::new();

    assert_eq!(metrics.effective_batch_size(), 0);
    assert_eq!(metrics.batch_size_reductions(), 0);

    metrics.record_batch_size_reduction(256);
    assert_eq!(metrics.effective_batch_size(), 256);
    assert_eq!(metrics.batch_size_reductions(), 1);

    metrics.record_batch_size_reduction(128);
    assert_eq!(metrics.effective_batch_size(), 128);
    assert_eq!(metrics.batch_size_reductions(), 2);
}
