//! Tests for oversized helpful/harmful sample sets exceeding the GPU device's
//! `max_storage_buffer_binding_size` limit (Issue #2314).
//!
//! With the device created at `wgpu::Limits::default()` (128 MiB), a helpful
//! sample set of 2,796,203 samples needs a 48-byte-per-sample contributions
//! binding that exceeds the limit, and a harmful set of 8,388,609 samples
//! exceeds it with 16-byte `HarmfulContribution` entries. The fix must return
//! `Err` naming `max_storage_buffer_binding_size` before allocating, rather than
//! letting wgpu panic on the uncaptured validation error and kill the GPU
//! thread. These tests assert the error is returned and that the GPU analyser
//! survives to evaluate a normal-sized batch afterwards.

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::analysis::samples::HelpfulSample;

/// Skip test if no GPU available.
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Build a cheap deterministic helpful sample set of the requested length.
fn make_samples(len: usize) -> Vec<HelpfulSample> {
    (0..len)
        .map(|_| HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,
            target_activation: None,
        })
        .collect()
}

#[test]
fn oversized_helpful_set_returns_err_and_gpu_survives() {
    skip_without_gpu!();

    let gpu = GpuAnalyzer::new().expect("GPU analyser should initialise on a GPU-equipped machine");

    // 2,796,203 samples * 48 bytes/sample contributions binding = 134,217,744
    // bytes, which exceeds the default max_storage_buffer_binding_size of
    // 134,217,728 bytes (128 MiB).
    let oversized = make_samples(2_796_203);
    let batch: Vec<&[HelpfulSample]> = vec![&oversized];

    let result = gpu.evaluate_helpful_batch(&batch);
    let err = match result {
        Ok(_) => panic!("oversized helpful sample set must be rejected, not panic"),
        Err(err) => err,
    };
    let message = format!("{err:#}");
    assert!(
        message.contains("max_storage_buffer_binding_size"),
        "error should name the offending limit, got: {message}"
    );

    // The GPU thread must survive the rejected oversized request: a normal
    // batch submitted afterwards should still succeed.
    let small = make_samples(1024);
    let small_batch: Vec<&[HelpfulSample]> = vec![&small];
    let recovered = gpu
        .evaluate_helpful_batch(&small_batch)
        .expect("GPU analyser should still work after an oversized batch was rejected");
    assert_eq!(recovered.len(), 1);
}

#[test]
fn oversized_harmful_set_returns_err_and_gpu_survives() {
    skip_without_gpu!();

    let gpu = GpuAnalyzer::new().expect("GPU analyser should initialise on a GPU-equipped machine");

    // 8,388,609 samples * 16 bytes/sample HarmfulContribution binding exceeds
    // the default max_storage_buffer_binding_size of 134,217,728 bytes (128 MiB).
    let oversized = make_samples(8_388_609);
    let batch: Vec<(&[HelpfulSample], f32)> = vec![(&oversized, 1.0)];

    let result = gpu.evaluate_harmful_batch(&batch);
    let err = match result {
        Ok(_) => panic!("oversized harmful sample set must be rejected, not panic"),
        Err(err) => err,
    };
    let message = format!("{err:#}");
    assert!(
        message.contains("max_storage_buffer_binding_size"),
        "error should name the offending limit, got: {message}"
    );

    // The GPU thread must survive the rejected oversized request: a normal
    // batch submitted afterwards should still succeed.
    let small = make_samples(1024);
    let small_batch: Vec<(&[HelpfulSample], f32)> = vec![(&small, 1.0)];
    let recovered = gpu
        .evaluate_harmful_batch(&small_batch)
        .expect("GPU analyser should still work after an oversized batch was rejected");
    assert_eq!(recovered.len(), 1);
}
