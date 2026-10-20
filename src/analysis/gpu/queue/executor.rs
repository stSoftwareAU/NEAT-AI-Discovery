//! The evaluation surface the GPU thread loop drives (Issue #1929).
//!
//! `gpu_thread_loop` used to call `GpuAnalyzer` inherent methods directly, so
//! the loop could only run on a machine with a working GPU. The skip-stale and
//! abort-retry behaviour added for Issue #1929 is defined by what the loop does
//! *not* call, which is untestable without a substitute analyser.
//!
//! `RequestEvaluator` is that seam: the production loop runs on `GpuAnalyzer`
//! exactly as before, while tests drive it with a counting stub and assert the
//! analyser is never invoked for an abandoned request.
//!
//! Issue #2363: wgpu 30 reports a lost device as an opaque error (for example
//! "Error occurred when trying to async map a buffer") or as a panic from the
//! default uncaptured-error handler. The production evaluations therefore run
//! inside wgpu out-of-memory and internal error scopes, so a captured device
//! error surfaces as a typed `GPU out of memory` or `GPU internal error`
//! message. The device-lost callback sets a flag that `device_lost()` exposes,
//! so recovery routes on the source of the failure rather than on its text.

use anyhow::Result;
use std::sync::atomic::Ordering;

use crate::analysis::gpu::analyzer::GpuAnalyzer;
use crate::analysis::gpu::budget::GpuTimeBudget;
use crate::analysis::samples::{HarmfulStats, HelpfulSample, HelpfulStats, ReluStats};

/// The GPU evaluations a queued work request can ask for.
///
/// One method per `GpuWorkRequest` variant, each mirroring the corresponding
/// `GpuAnalyzer` method exactly.
pub(crate) trait RequestEvaluator {
    /// Current effective batch size, used by the OOM back-off path.
    fn batch_size(&self) -> usize;

    /// Whether the device-lost callback has fired for this evaluator (Issue
    /// #2363).
    ///
    /// wgpu 30 reports a lost device through `set_device_lost_callback`, not
    /// through a stable error string. A failed request is routed to recovery
    /// when this returns `true`, even if its error text does not match.
    fn device_lost(&self) -> bool;

    fn evaluate_helpful_batch(
        &self,
        samples_batch: &[&[HelpfulSample]],
        budget: GpuTimeBudget,
    ) -> Result<Vec<HelpfulStats>>;

    fn evaluate_harmful_batch(
        &self,
        samples_batch: &[(&[HelpfulSample], f32)],
        budget: GpuTimeBudget,
    ) -> Result<Vec<HarmfulStats>>;

    fn evaluate_relu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
        budget: GpuTimeBudget,
    ) -> Result<(ReluStats, ReluStats, f32)>;

    fn evaluate_activation(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
        budget: GpuTimeBudget,
    ) -> Result<(f32, f32, f32, u32)>;

    fn evaluate_activations_batched(
        &self,
        samples: &[HelpfulSample],
        activation_configs: &[(u32, f32, f32)],
        budget: GpuTimeBudget,
    ) -> Result<Vec<(f32, f32, f32, u32)>>;
}

/// Formats a wgpu error captured by an error scope as a typed message (Issue
/// #2363).
///
/// The `GPU out of memory` and `GPU internal error` prefixes let the recovery
/// matchers classify the failure by its source. Pure, so it can be tested
/// without a device.
pub(crate) fn scoped_device_error_message(err: &wgpu::Error) -> String {
    match err {
        wgpu::Error::OutOfMemory { .. } => format!("GPU out of memory: {err}"),
        wgpu::Error::Internal { .. } => format!("GPU internal error: {err}"),
        _ => format!("GPU device error: {err}"),
    }
}

impl GpuAnalyzer {
    /// Runs one evaluation with wgpu out-of-memory and internal error scopes
    /// pushed around it (Issue #2363).
    ///
    /// Scopes are popped innermost first, on the calling thread, which is the
    /// thread that pushed them. A captured error takes precedence over the
    /// evaluation result, because a batch that hit a device error cannot be
    /// trusted even if it returned `Ok`. Out-of-memory is preferred over
    /// internal when both were captured.
    fn with_device_error_scopes<T>(&self, evaluate: impl FnOnce() -> Result<T>) -> Result<T> {
        let Some(device) = self.device.as_ref() else {
            return evaluate();
        };
        let oom_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let result = evaluate();
        let internal_error = pollster::block_on(internal_scope.pop());
        let oom_error = pollster::block_on(oom_scope.pop());
        if let Some(err) = oom_error.or(internal_error) {
            return Err(anyhow::anyhow!(scoped_device_error_message(&err)));
        }
        result
    }
}

impl RequestEvaluator for GpuAnalyzer {
    fn batch_size(&self) -> usize {
        Self::batch_size(self)
    }

    fn device_lost(&self) -> bool {
        self.device_lost.load(Ordering::SeqCst)
    }

    fn evaluate_helpful_batch(
        &self,
        samples_batch: &[&[HelpfulSample]],
        budget: GpuTimeBudget,
    ) -> Result<Vec<HelpfulStats>> {
        self.with_device_error_scopes(|| {
            self.evaluate_helpful_batch_with_budget(samples_batch, budget)
        })
    }

    fn evaluate_harmful_batch(
        &self,
        samples_batch: &[(&[HelpfulSample], f32)],
        budget: GpuTimeBudget,
    ) -> Result<Vec<HarmfulStats>> {
        self.with_device_error_scopes(|| {
            self.evaluate_harmful_batch_with_budget(samples_batch, budget)
        })
    }

    fn evaluate_relu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
        budget: GpuTimeBudget,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        self.with_device_error_scopes(|| {
            self.evaluate_relu_gpu_with_budget(samples, threshold, budget)
        })
    }

    fn evaluate_activation(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
        budget: GpuTimeBudget,
    ) -> Result<(f32, f32, f32, u32)> {
        self.with_device_error_scopes(|| {
            self.evaluate_activation_gpu_with_budget(
                samples,
                activation_type,
                orientation,
                scale,
                budget,
            )
        })
    }

    fn evaluate_activations_batched(
        &self,
        samples: &[HelpfulSample],
        activation_configs: &[(u32, f32, f32)],
        budget: GpuTimeBudget,
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        self.with_device_error_scopes(|| {
            self.evaluate_activations_batched_gpu_with_budget(samples, activation_configs, budget)
        })
    }
}

/// Creates a replacement evaluator after a device-lost error.
///
/// `batch_size_override` is `Some` only on the memory-exhaustion path, where the
/// loop retries with a halved batch size (Issue #1083).
pub(crate) trait EvaluatorFactory<E: RequestEvaluator> {
    fn create(&self, batch_size_override: Option<usize>) -> Result<E>;
}

/// Production factory: re-initialises a real `GpuAnalyzer`.
pub(crate) struct GpuAnalyzerFactory;

impl EvaluatorFactory<GpuAnalyzer> for GpuAnalyzerFactory {
    fn create(&self, batch_size_override: Option<usize>) -> Result<GpuAnalyzer> {
        match batch_size_override {
            Some(size) => GpuAnalyzer::new_with_batch_size(size),
            None => GpuAnalyzer::new(),
        }
    }
}
