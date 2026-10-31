//! The `RequestEvaluator for GpuAnalyzer` delegation returns `Err` on an
//! all-`None` analyser (Issue #2241).
//!
//! The helpers live in `gpu/none_field_tests.rs`, which covers the inherent
//! entry points and the `GpuEvaluator` delegation and says why only the device
//! check is reachable without a GPU.

use super::executor::RequestEvaluator;
use crate::analysis::gpu::budget::GpuTimeBudget;
use crate::analysis::gpu::none_field_tests::{CONFIGS, all_none_analyzer, samples};
use crate::analysis::samples::HelpfulSample;

// The wrapper's message distinguishes it from the inner evaluators' own
// missing-device errors, so this guards the production wiring (Issue #2363).
fn assert_scoped_device_err<T>(entry: &str, result: anyhow::Result<T>) {
    match result {
        Ok(_) => panic!("{entry} returned Ok on an all-None GpuAnalyzer"),
        Err(err) => {
            let message = format!("{err:#}");
            assert!(
                message.contains("GPU device unavailable for error-scoped evaluation"),
                "{entry} did not enter the device error scopes: {message:?}"
            );
        }
    }
}

#[test]
fn the_request_evaluator_delegation_errs_when_the_device_is_none() {
    let gpu = all_none_analyzer();
    let s = samples();
    let batch: [&[HelpfulSample]; 2] = [&s, &s];
    let harmful: [(&[HelpfulSample], f32); 2] = [(&s, 0.5), (&s, -0.5)];
    let budget = GpuTimeBudget::unbounded;

    assert_scoped_device_err(
        "RequestEvaluator::evaluate_helpful_batch",
        RequestEvaluator::evaluate_helpful_batch(&gpu, &batch, budget()),
    );
    assert_scoped_device_err(
        "RequestEvaluator::evaluate_harmful_batch",
        RequestEvaluator::evaluate_harmful_batch(&gpu, &harmful, budget()),
    );
    assert_scoped_device_err(
        "RequestEvaluator::evaluate_relu",
        RequestEvaluator::evaluate_relu(&gpu, &s, 0.0, budget()),
    );
    assert_scoped_device_err(
        "RequestEvaluator::evaluate_activation",
        RequestEvaluator::evaluate_activation(&gpu, &s, 0, 1.0, 1.0, budget()),
    );
    assert_scoped_device_err(
        "RequestEvaluator::evaluate_activations_batched",
        RequestEvaluator::evaluate_activations_batched(&gpu, &s, &CONFIGS, budget()),
    );
}
