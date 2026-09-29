//! Every `GpuAnalyzer` entry point returns `Err` on an all-`None` analyser
//! (Issue #2241).
//!
//! Only the device check is reachable here: a `wgpu::Device` cannot be built
//! without a GPU, so the queue/layout/pipeline checks behind it are pinned by
//! the `.context("…")` string check in
//! `tests/issue_2113_chunk_09c_device_sweep.rs` instead. `FakeGpuEvaluator`
//! replaces `GpuAnalyzer` wholesale, so it cannot prove this property. The
//! `RequestEvaluator` delegation is covered in `queue/none_field_tests.rs`.

use anyhow::Result;

use super::analyzer::{GpuAnalyzer, GpuEvaluator};
use super::budget::GpuTimeBudget;
use crate::analysis::samples::HelpfulSample;

pub(super) const DEVICE_UNAVAILABLE: &str = "GPU device unavailable";

pub(super) fn all_none_analyzer() -> GpuAnalyzer {
    GpuAnalyzer {
        device: None,
        queue: None,
        helpful_layout: None,
        helpful_pipeline: None,
        harmful_layout: None,
        harmful_pipeline: None,
        relu_layout: None,
        relu_pipeline: None,
        activation_layout: None,
        activation_pipeline: None,
        helpful_reduce_layout: None,
        helpful_reduce_pipeline: None,
        harmful_reduce_layout: None,
        harmful_reduce_pipeline: None,
        activation_reduce_layout: None,
        activation_reduce_pipeline: None,
        batch_size: 64,
    }
}

/// Non-empty input, so no empty-input short-circuit can answer `Ok`.
pub(super) fn samples() -> Vec<HelpfulSample> {
    vec![
        HelpfulSample {
            activation: 0.5,
            avg_error: -0.25,
            target_value: None,
            target_activation: None,
        };
        4
    ]
}

pub(super) const CONFIGS: [(u32, f32, f32); 2] = [(0, 1.0, 1.0), (1, -1.0, 0.5)];

pub(super) fn assert_device_err<T>(entry: &str, result: Result<T>) {
    match result {
        Ok(_) => panic!("{entry} returned Ok on an all-None GpuAnalyzer"),
        Err(err) => {
            let message = format!("{err:#}");
            assert!(
                message.contains(DEVICE_UNAVAILABLE),
                "{entry} failed with {message:?}, not {DEVICE_UNAVAILABLE:?}"
            );
        }
    }
}

#[test]
fn every_inherent_entry_point_errs_when_the_device_is_none() {
    let gpu = all_none_analyzer();
    let s = samples();
    let batch: [&[HelpfulSample]; 2] = [&s, &s];
    let harmful: [(&[HelpfulSample], f32); 2] = [(&s, 0.5), (&s, -0.5)];
    let budget = GpuTimeBudget::unbounded;

    assert_device_err("evaluate_relu_gpu", gpu.evaluate_relu_gpu(&s, 0.0));
    assert_device_err(
        "evaluate_relu_gpu_with_budget",
        gpu.evaluate_relu_gpu_with_budget(&s, 0.0, budget()),
    );
    assert_device_err(
        "evaluate_activation_gpu",
        gpu.evaluate_activation_gpu(&s, 0, 1.0, 1.0),
    );
    assert_device_err(
        "evaluate_activation_gpu_with_budget",
        gpu.evaluate_activation_gpu_with_budget(&s, 0, 1.0, 1.0, budget()),
    );
    assert_device_err(
        "evaluate_activations_batched_gpu",
        gpu.evaluate_activations_batched_gpu(&s, &CONFIGS),
    );
    assert_device_err(
        "evaluate_activations_batched_gpu_with_budget",
        gpu.evaluate_activations_batched_gpu_with_budget(&s, &CONFIGS, budget()),
    );
    assert_device_err(
        "evaluate_harmful_batch",
        gpu.evaluate_harmful_batch(&harmful),
    );
    assert_device_err(
        "evaluate_harmful_batch_with_budget",
        gpu.evaluate_harmful_batch_with_budget(&harmful, budget()),
    );
    assert_device_err("evaluate_helpful_batch", gpu.evaluate_helpful_batch(&batch));
    assert_device_err(
        "evaluate_helpful_batch_with_budget",
        gpu.evaluate_helpful_batch_with_budget(&batch, budget()),
    );
}

#[test]
fn the_gpu_evaluator_delegation_errs_when_the_device_is_none() {
    let gpu = all_none_analyzer();
    let s = samples();

    assert_device_err(
        "GpuEvaluator::evaluate_relu",
        GpuEvaluator::evaluate_relu(&gpu, &s, 0.0),
    );
    assert_device_err(
        "GpuEvaluator::evaluate_activation",
        GpuEvaluator::evaluate_activation(&gpu, &s, 0, 1.0, 1.0),
    );
    assert_device_err(
        "GpuEvaluator::evaluate_activations_batched",
        GpuEvaluator::evaluate_activations_batched(&gpu, &s, &CONFIGS),
    );
}
