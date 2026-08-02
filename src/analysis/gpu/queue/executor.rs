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

use anyhow::Result;

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

impl RequestEvaluator for GpuAnalyzer {
    fn batch_size(&self) -> usize {
        Self::batch_size(self)
    }

    fn evaluate_helpful_batch(
        &self,
        samples_batch: &[&[HelpfulSample]],
        budget: GpuTimeBudget,
    ) -> Result<Vec<HelpfulStats>> {
        self.evaluate_helpful_batch_with_budget(samples_batch, budget)
    }

    fn evaluate_harmful_batch(
        &self,
        samples_batch: &[(&[HelpfulSample], f32)],
        budget: GpuTimeBudget,
    ) -> Result<Vec<HarmfulStats>> {
        self.evaluate_harmful_batch_with_budget(samples_batch, budget)
    }

    fn evaluate_relu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
        budget: GpuTimeBudget,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        self.evaluate_relu_gpu_with_budget(samples, threshold, budget)
    }

    fn evaluate_activation(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
        budget: GpuTimeBudget,
    ) -> Result<(f32, f32, f32, u32)> {
        self.evaluate_activation_gpu_with_budget(
            samples,
            activation_type,
            orientation,
            scale,
            budget,
        )
    }

    fn evaluate_activations_batched(
        &self,
        samples: &[HelpfulSample],
        activation_configs: &[(u32, f32, f32)],
        budget: GpuTimeBudget,
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        self.evaluate_activations_batched_gpu_with_budget(samples, activation_configs, budget)
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
