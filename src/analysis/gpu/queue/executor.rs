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
//! inside wgpu out-of-memory, internal and validation error scopes, so a
//! captured device error surfaces as a typed `GPU out of memory`, `GPU internal
//! error` or `GPU device error` message. The device-lost callback sets a flag that `device_lost()` exposes,
//! so recovery routes on the source of the failure rather than on its text.

use anyhow::{Context, Result};
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

/// Picks the captured error to report, preferring out-of-memory over internal
/// over validation (Issue #2363).
///
/// Out-of-memory comes first because it is what batch halving (#1083) reacts
/// to. A validation error is usually a follow-on from the invalid resource an
/// out-of-memory buffer creation hands back, so it ranks last. Pure, so the
/// precedence can be tested without a device.
fn select_scoped_error(
    oom: Option<wgpu::Error>,
    internal: Option<wgpu::Error>,
    validation: Option<wgpu::Error>,
) -> Option<wgpu::Error> {
    oom.or(internal).or(validation)
}

impl GpuAnalyzer {
    /// Runs one evaluation with wgpu out-of-memory, internal and validation
    /// error scopes pushed around it (Issue #2363).
    ///
    /// In wgpu-core 30 a buffer-creation out-of-memory does not lose the device:
    /// it hands back an invalid buffer and the out-of-memory error. The
    /// evaluation then uses that buffer (for example in `create_bind_group`),
    /// which raises a *validation* error. With no validation scope, wgpu's
    /// default uncaptured-error handler panics on it before the scopes are
    /// popped, so the outermost scope captures validation errors too.
    ///
    /// Scopes are popped innermost first, on the calling thread, which is the
    /// thread that pushed them. A captured error takes precedence over the
    /// evaluation result, because a batch that hit a device error cannot be
    /// trusted even if it returned `Ok`. See `select_scoped_error` for which
    /// error wins when several were captured.
    fn with_device_error_scopes<T>(&self, evaluate: impl FnOnce() -> Result<T>) -> Result<T> {
        let device = self
            .device
            .as_ref()
            .context("GPU device unavailable for error-scoped evaluation")?;
        let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let oom_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let result = evaluate();
        let internal_error = pollster::block_on(internal_scope.pop());
        let oom_error = pollster::block_on(oom_scope.pop());
        let validation_error = pollster::block_on(validation_scope.pop());
        if let Some(err) = select_scoped_error(oom_error, internal_error, validation_error) {
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

#[cfg(test)]
mod tests {
    use super::{GpuAnalyzer, RequestEvaluator, scoped_device_error_message, select_scoped_error};
    use crate::analysis::gpu::queue::recovery::{is_device_lost_error, is_memory_exhaustion_error};
    use std::cell::Cell;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    /// Issue #2363: an out-of-memory capture gets the typed `GPU out of memory`
    /// prefix, and the wrapped message is recognised by both recovery matchers.
    #[test]
    fn out_of_memory_capture_gets_out_of_memory_prefix() {
        let err = wgpu::Error::OutOfMemory {
            source: Box::new(std::io::Error::other("x")),
        };
        let msg = scoped_device_error_message(&err);
        assert!(
            msg.starts_with("GPU out of memory: "),
            "expected the out-of-memory prefix, got: {msg}"
        );
        // wgpu's `Display` for `OutOfMemory` is "Out of Memory" and omits the
        // source, so the Metal "Not enough memory left" wording is not produced here.
        assert!(msg.contains("Out of Memory"), "got: {msg}");
        let wrapped = anyhow::anyhow!(msg);
        assert!(
            is_device_lost_error(&wrapped),
            "the out-of-memory message must route to device loss"
        );
        assert!(
            is_memory_exhaustion_error(&wrapped),
            "the out-of-memory message must classify as memory exhaustion"
        );
    }

    /// Issue #2363: an internal capture gets the `GPU internal error` prefix.
    #[test]
    fn internal_capture_gets_internal_error_prefix() {
        let err = wgpu::Error::Internal {
            source: Box::new(std::io::Error::other("x")),
            description: "pipeline exploded".to_owned(),
        };
        let msg = scoped_device_error_message(&err);
        assert!(
            msg.starts_with("GPU internal error: "),
            "expected the internal-error prefix, got: {msg}"
        );
        assert!(msg.contains("pipeline exploded"), "got: {msg}");
    }

    /// Issue #2363: any other capture (here a validation error) falls through to
    /// the generic `GPU device error` prefix.
    #[test]
    fn validation_capture_gets_generic_device_error_prefix() {
        let err = wgpu::Error::Validation {
            source: Box::new(std::io::Error::other("x")),
            description: "bad binding".to_owned(),
        };
        let msg = scoped_device_error_message(&err);
        assert!(
            msg.starts_with("GPU device error: "),
            "expected the generic device-error prefix, got: {msg}"
        );
        assert!(msg.contains("bad binding"), "got: {msg}");
    }

    fn oom_error() -> wgpu::Error {
        wgpu::Error::OutOfMemory {
            source: Box::new(std::io::Error::other("oom")),
        }
    }

    fn internal_error() -> wgpu::Error {
        wgpu::Error::Internal {
            source: Box::new(std::io::Error::other("internal")),
            description: "internal".to_owned(),
        }
    }

    fn validation_error() -> wgpu::Error {
        wgpu::Error::Validation {
            source: Box::new(std::io::Error::other("validation")),
            description: "validation".to_owned(),
        }
    }

    /// Issue #2363: out-of-memory wins over internal and validation, so batch
    /// halving (#1083) still sees the cause rather than its follow-on error.
    #[test]
    fn out_of_memory_outranks_internal_and_validation() {
        let picked = select_scoped_error(
            Some(oom_error()),
            Some(internal_error()),
            Some(validation_error()),
        );
        assert!(
            matches!(picked, Some(wgpu::Error::OutOfMemory { .. })),
            "got: {picked:?}"
        );
    }

    /// Issue #2363: internal wins over validation when there is no
    /// out-of-memory error.
    #[test]
    fn internal_outranks_validation() {
        let picked = select_scoped_error(None, Some(internal_error()), Some(validation_error()));
        assert!(
            matches!(picked, Some(wgpu::Error::Internal { .. })),
            "got: {picked:?}"
        );
    }

    /// Issue #2363: a validation error alone is reported, and no capture at all
    /// reports nothing.
    #[test]
    fn validation_alone_is_reported_and_none_is_none() {
        let picked = select_scoped_error(None, None, Some(validation_error()));
        assert!(
            matches!(picked, Some(wgpu::Error::Validation { .. })),
            "got: {picked:?}"
        );
        assert!(select_scoped_error(None, None, None).is_none());
    }

    /// Issue #2363 (PR #2433 review): a buffer-creation failure hands back an
    /// invalid buffer, and the evaluation then uses it. The follow-on bind-group
    /// error is a *validation* error, which wgpu's default handler turns into a
    /// panic unless a validation scope captures it. Needs a GPU, so it skips on
    /// a machine without one.
    #[test]
    fn invalid_buffer_use_inside_scopes_returns_err_not_panic() {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
        let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed");
        let device = analyzer
            .device
            .as_ref()
            .expect("a real analyser has a device");

        // Positive control: a scoped evaluation that raises no error returns its value.
        let ok = analyzer.with_device_error_scopes(|| Ok::<_, anyhow::Error>(7));
        assert_eq!(ok.expect("a clean evaluation must pass through"), 7);

        let err = analyzer
            .with_device_error_scopes(|| {
                // One byte over the device limit: creation fails and returns an
                // invalid buffer instead of losing the device.
                let invalid = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("issue-2363-invalid-buffer"),
                    size: device.limits().max_buffer_size + 1,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("issue-2363-layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
                // Using the invalid buffer here is the follow-on error.
                let _bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("issue-2363-bind-group"),
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: invalid.as_entire_binding(),
                    }],
                });
                Ok::<_, anyhow::Error>(())
            })
            .expect_err("a captured device error must become an Err, not a panic or Ok");
        let text = format!("{err:#}");
        assert!(
            [
                "GPU out of memory: ",
                "GPU internal error: ",
                "GPU device error: "
            ]
            .iter()
            .any(|prefix| text.starts_with(prefix)),
            "expected a typed GPU error message, got: {text}"
        );
    }

    /// Issue #2363 (PR #2433 review): the production device-lost callback must
    /// feed the flag that `RequestEvaluator::device_lost` reads. The loop tests
    /// use a fake evaluator, so only this test notices if `GpuAnalyzer::new`
    /// stops registering the callback or `device_lost` reads another field.
    /// Needs a GPU, so it skips on a machine without one.
    #[test]
    fn real_analyzer_device_lost_flag_follows_the_device_lost_callback() {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
        let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed");
        assert!(
            !RequestEvaluator::device_lost(&analyzer),
            "a freshly created device must not be flagged lost"
        );

        let device = analyzer
            .device
            .as_ref()
            .expect("a real analyser has a device");
        device.destroy();
        // wgpu-core invokes the lost closure from `maintain`, which a poll drives.
        let _ = device.poll(wgpu::PollType::Poll);
        assert!(
            RequestEvaluator::device_lost(&analyzer),
            "destroying the device must set the flag through the registered callback"
        );
    }

    /// Issue #2363: with no device, a scoped evaluation must fail loudly and must
    /// not run the evaluation closure.
    #[test]
    fn scoped_evaluation_without_device_fails_loudly_without_evaluating() {
        let analyzer = GpuAnalyzer {
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
            batch_size: 1,
            device_lost: Arc::new(AtomicBool::new(false)),
        };
        let evaluated = Cell::new(false);
        let result = analyzer.with_device_error_scopes(|| {
            evaluated.set(true);
            Ok::<_, anyhow::Error>(1)
        });
        let err = result.expect_err("a missing device must not yield a result");
        let text = format!("{err:#}");
        assert!(
            text.contains("GPU device unavailable for error-scoped evaluation"),
            "expected the missing-device message, got: {text}"
        );
        assert!(
            !evaluated.get(),
            "the evaluation must not run without a device"
        );
    }
}
