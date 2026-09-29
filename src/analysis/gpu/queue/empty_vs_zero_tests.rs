//! Empty-input short-circuits vs an all-zero GPU answer (Issue #2243, chunk
//! 9d-1 queue-core audit, `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`).
//!
//! Every empty-input short-circuit in `submission.rs` is driven through a
//! real [`GpuWorkQueue`] wired to the **production**
//! [`GpuWorkQueue::run_work_loop`] with [`FakeGpuEvaluator`]
//! (`WedgeBehaviour::Completes`) — only the device is fake. Case (a) is an
//! empty request; case (b) is a non-empty request answered with all-zero
//! statistics. Where the ledger says the two are distinguishable, this file
//! asserts they differ; where it says they are not, it pins the current
//! identical output so a fix that makes them differ has to update both this
//! test and the ledger row it cites.
//!
//! Each test also reads [`FakeGpuProbe::calls`] to prove (a) never reached the
//! device and (b) reached it exactly once — the proof that (a) really took
//! the short-circuit rather than merely returning the same answer by luck.

use crossbeam_channel::{Sender, bounded};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use super::fake_evaluator::{FakeEvaluatorFactory, FakeGpuEvaluator, WedgeBehaviour};
use super::{GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::breaker::GpuCircuitBreaker;
use crate::analysis::gpu::heartbeat::GpuHeartbeat;
use crate::analysis::samples::{HelpfulSample, ReluOrientation, ReluStats};

fn sample() -> HelpfulSample {
    HelpfulSample {
        activation: 1.0,
        avg_error: 0.5,
        target_value: None,
        target_activation: None,
    }
}

/// A fake GPU device running the production work loop on its own thread, plus
/// a `GpuWorkQueue` wired to the same channel and an isolated breaker.
///
/// Modelled directly on `WedgedGpu` in `wedge_tests.rs`: the production loop,
/// bounded wait and circuit breaker are all real, only the device is fake.
struct Harness {
    queue: GpuWorkQueue,
    breaker: &'static GpuCircuitBreaker,
    probe: super::fake_evaluator::FakeGpuProbe,
    work_tx: Sender<GpuWorkRequest>,
    worker: Option<JoinHandle<()>>,
}

impl Harness {
    fn spawn() -> Self {
        let (work_tx, work_rx) = bounded::<GpuWorkRequest>(8);
        let heartbeat = Arc::new(GpuHeartbeat::new());
        let evaluator = FakeGpuEvaluator::new(WedgeBehaviour::Completes, Arc::clone(&heartbeat));
        let probe = evaluator.probe();
        let factory = Arc::new(FakeEvaluatorFactory::new());

        let worker = std::thread::spawn(move || {
            GpuWorkQueue::run_work_loop(evaluator, work_rx, &*factory, &heartbeat);
        });

        // Drop the exit sender at once so `Drop for GpuWorkQueue` takes the
        // disconnected path instead of sitting out the shutdown timeout.
        let (_, exit_rx) = bounded::<()>(1);
        let breaker: &'static GpuCircuitBreaker = Box::leak(Box::new(GpuCircuitBreaker::new()));
        let queue = GpuWorkQueue {
            work_tx: work_tx.clone(),
            thread_handle: None,
            exit_rx,
            deadline: None,
            breaker,
        };

        Self {
            queue,
            breaker,
            probe,
            work_tx,
            worker: Some(worker),
        }
    }

    /// Release the fake device, shut the loop down and join the worker thread.
    fn stop(mut self) {
        let _ = self
            .work_tx
            .send_timeout(GpuWorkRequest::Shutdown, Duration::from_secs(1));
        if let Some(worker) = self.worker.take() {
            worker.join().expect("the fake GPU worker thread panicked");
        }
        self.breaker.reset();
    }
}

// =============================================================================
// 1. `submit_helpful_batch` (submission.rs:187) — distinguishable: yes
// =============================================================================

#[test]
fn submit_helpful_batch_empty_vs_zero_are_distinguishable() {
    let harness = Harness::spawn();

    let empty = harness
        .queue
        .submit_helpful_batch(Vec::new(), &None)
        .expect("empty batch must succeed")
        .collect()
        .expect("pre-resolved future collects");
    assert!(empty.is_empty(), "case (a): empty input yields no stats");
    assert_eq!(
        harness.probe.calls(),
        0,
        "case (a) never reaches the device"
    );

    let one = harness
        .queue
        .submit_helpful_batch(vec![Arc::new(vec![sample()])], &None)
        .expect("non-empty batch must enqueue")
        .collect()
        .expect("the fake device answers");
    assert_eq!(one.len(), 1, "case (b): one set in, one stat out");
    assert_eq!(
        one[0].total_count(),
        0,
        "the fake answer is all-zero, not a real evaluation"
    );
    assert_eq!(one[0].samples_evaluated, 0);
    assert_eq!(
        harness.probe.calls(),
        1,
        "case (b) reaches the device exactly once"
    );

    assert_ne!(
        empty.len(),
        one.len(),
        "distinguishable: yes — lengths differ (0 vs 1)"
    );

    harness.stop();
}

// =============================================================================
// 2. `evaluate_helpful_batch` (submission.rs:258) — distinguishable: yes
// =============================================================================

#[test]
fn evaluate_helpful_batch_empty_vs_zero_are_distinguishable() {
    let harness = Harness::spawn();

    let empty = harness
        .queue
        .evaluate_helpful_batch(Vec::new(), &None)
        .expect("empty batch must succeed");
    assert!(empty.is_empty(), "case (a): empty input yields no stats");
    assert_eq!(
        harness.probe.calls(),
        0,
        "case (a) never reaches the device"
    );

    let one = harness
        .queue
        .evaluate_helpful_batch(vec![Arc::new(vec![sample()])], &None)
        .expect("non-empty batch must be answered");
    assert_eq!(one.len(), 1, "case (b): one set in, one stat out");
    assert_eq!(one[0].total_count(), 0, "the fake answer is all-zero");
    assert_eq!(one[0].samples_evaluated, 0);
    assert_eq!(
        harness.probe.calls(),
        1,
        "case (b) reaches the device exactly once"
    );

    assert_ne!(
        empty.len(),
        one.len(),
        "distinguishable: yes — lengths differ (0 vs 1)"
    );

    harness.stop();
}

// =============================================================================
// 3. `evaluate_harmful_batch` (submission.rs:321) — distinguishable: yes
// =============================================================================

#[test]
fn evaluate_harmful_batch_empty_vs_zero_are_distinguishable() {
    let harness = Harness::spawn();

    let empty = harness
        .queue
        .evaluate_harmful_batch(Vec::new(), &None)
        .expect("empty batch must succeed");
    assert!(empty.is_empty(), "case (a): empty input yields no stats");
    assert_eq!(
        harness.probe.calls(),
        0,
        "case (a) never reaches the device"
    );

    let one = harness
        .queue
        .evaluate_harmful_batch(vec![(Arc::new(vec![sample()]), 1.0)], &None)
        .expect("non-empty batch must be answered");
    assert_eq!(one.len(), 1, "case (b): one set in, one stat out");
    assert_eq!(one[0].harmful_count, 0, "the fake answer is all-zero");
    assert_eq!(one[0].helpful_count, 0, "the fake answer is all-zero");
    assert_eq!(
        harness.probe.calls(),
        1,
        "case (b) reaches the device exactly once"
    );

    assert_ne!(
        empty.len(),
        one.len(),
        "distinguishable: yes — lengths differ (0 vs 1)"
    );

    harness.stop();
}

// =============================================================================
// 4. `evaluate_relu_gpu` (submission.rs:382) — indistinguishable: no
// =============================================================================

/// `ReluStats` has neither `PartialEq` nor `Debug`, so the comparison is
/// field-by-field.
fn assert_relu_equal(a: &(ReluStats, ReluStats, f32), b: &(ReluStats, ReluStats, f32)) {
    assert!(
        matches!(a.0.orientation, ReluOrientation::Positive)
            && matches!(b.0.orientation, ReluOrientation::Positive),
        "first stats must both be the Positive orientation"
    );
    assert!(
        matches!(a.1.orientation, ReluOrientation::Negative)
            && matches!(b.1.orientation, ReluOrientation::Negative),
        "second stats must both be the Negative orientation"
    );
    for (name, sa, sb) in [("positive", &a.0, &b.0), ("negative", &a.1, &b.1)] {
        assert!(
            sa.samples.is_empty() && sb.samples.is_empty(),
            "{name}: both must carry no per-sample data"
        );
        assert_eq!(
            sa.activation_sq_sum, sb.activation_sq_sum,
            "{name}: activation_sq_sum must match"
        );
        assert_eq!(
            sa.error_activation_sum, sb.error_activation_sum,
            "{name}: error_activation_sum must match"
        );
    }
    assert_eq!(a.2, b.2, "the f32 baseline must match");
}

/// Indistinguishable (ledger verdict `no`), refuted rather than filed: every
/// production caller submits only with at least `MIN_NEURON_SAMPLE_COUNT`
/// samples (`src/analysis/synapse/relu_evaluation.rs:81`, `:128`), so this
/// short-circuit is unreachable from production, and an all-zero answer
/// yields no candidate either way. See the queue-core rows of
/// `## Refuted / not findings` in `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`
/// (Issue #2243). A change that makes the two answers distinguishable must
/// update this test and that row.
#[test]
fn evaluate_relu_gpu_empty_vs_zero_are_pinned_equal() {
    let harness = Harness::spawn();

    let empty = harness
        .queue
        .evaluate_relu_gpu(&[], 0.0, &None)
        .expect("empty samples must succeed");
    assert_eq!(
        harness.probe.calls(),
        0,
        "case (a) never reaches the device"
    );

    let one = harness
        .queue
        .evaluate_relu_gpu(&[sample()], 0.0, &None)
        .expect("non-empty samples must be answered");
    assert_eq!(
        harness.probe.calls(),
        1,
        "case (b) reaches the device exactly once"
    );

    assert_relu_equal(&empty, &one);

    harness.stop();
}

// =============================================================================
// 5. `evaluate_activation_gpu` (submission.rs:445) — indistinguishable: no
// =============================================================================

/// Indistinguishable (ledger verdict `no`), refuted rather than filed: every
/// production caller submits only with at least `MIN_NEURON_SAMPLE_COUNT`
/// samples (`src/analysis/synapse/activation_evaluation.rs:48`,
/// `src/analysis/synapse/activation_subset_evaluation.rs:48`), so this
/// short-circuit is unreachable from production, and an all-zero answer
/// yields no candidate either way. See the queue-core rows of
/// `## Refuted / not findings` in `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`
/// (Issue #2243). A change that makes the two answers distinguishable must
/// update this test and that row.
#[test]
fn evaluate_activation_gpu_empty_vs_zero_are_pinned_equal() {
    let harness = Harness::spawn();

    let empty = harness
        .queue
        .evaluate_activation_gpu(&[], 0, 1.0, 1.0, &None)
        .expect("empty samples must succeed");
    assert_eq!(
        harness.probe.calls(),
        0,
        "case (a) never reaches the device"
    );

    let one = harness
        .queue
        .evaluate_activation_gpu(&[sample()], 0, 1.0, 1.0, &None)
        .expect("non-empty samples must be answered");
    assert_eq!(
        harness.probe.calls(),
        1,
        "case (b) reaches the device exactly once"
    );

    assert_eq!(empty, (0.0, 0.0, 0.0, 0));
    assert_eq!(one, (0.0, 0.0, 0.0, 0));
    assert_eq!(empty, one, "distinguishable: no — both are all-zero");

    harness.stop();
}

// =============================================================================
// 6. `evaluate_activations_batched_gpu`, empty configs (submission.rs:515)
//    — distinguishable: yes
// =============================================================================

#[test]
fn evaluate_activations_batched_gpu_empty_configs_vs_one_config_are_distinguishable() {
    let harness = Harness::spawn();

    // Case (a): a config list that is empty short-circuits regardless of
    // whether samples are present.
    let no_configs_no_samples = harness
        .queue
        .evaluate_activations_batched_gpu(&[], &[], &None)
        .expect("empty configs, empty samples must succeed");
    assert!(no_configs_no_samples.is_empty());
    assert_eq!(harness.probe.calls(), 0, "no GPU call for empty configs");

    let no_configs_with_samples = harness
        .queue
        .evaluate_activations_batched_gpu(&[sample()], &[], &None)
        .expect("empty configs with samples must still short-circuit");
    assert!(no_configs_with_samples.is_empty());
    assert_eq!(
        harness.probe.calls(),
        0,
        "an empty config list never reaches the device, even with samples"
    );

    // Case (b): one config, one sample.
    let one = harness
        .queue
        .evaluate_activations_batched_gpu(&[sample()], &[(0, 1.0, 1.0)], &None)
        .expect("one config must be answered");
    assert_eq!(one.len(), 1, "case (b): one config in, one result out");
    assert_eq!(
        harness.probe.calls(),
        1,
        "case (b) reaches the device exactly once"
    );

    assert_ne!(
        no_configs_with_samples.len(),
        one.len(),
        "distinguishable: yes — lengths differ (0 vs 1)"
    );

    harness.stop();
}

// =============================================================================
// 7. `evaluate_activations_batched_gpu`, empty samples (submission.rs:519)
//    — indistinguishable: no
// =============================================================================

/// Indistinguishable (ledger verdict `no`), refuted rather than filed: every
/// production caller submits only with at least `MIN_NEURON_SAMPLE_COUNT`
/// samples (`src/analysis/synapse/gpu_evaluation.rs:50`), so this
/// short-circuit is unreachable from production, and an all-zero answer per
/// config yields no candidate either way. See the queue-core rows of
/// `## Refuted / not findings` in `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`
/// (Issue #2243). A change that makes the two answers distinguishable must
/// update this test and that row.
#[test]
fn evaluate_activations_batched_gpu_empty_samples_vs_zero_are_pinned_equal() {
    let harness = Harness::spawn();
    let configs = [(0u32, 1.0f32, 1.0f32), (1u32, -1.0f32, 2.0f32)];

    let empty = harness
        .queue
        .evaluate_activations_batched_gpu(&[], &configs, &None)
        .expect("empty samples must succeed");
    assert_eq!(
        harness.probe.calls(),
        0,
        "case (a) never reaches the device"
    );

    let one = harness
        .queue
        .evaluate_activations_batched_gpu(&[sample()], &configs, &None)
        .expect("non-empty samples must be answered");
    assert_eq!(
        harness.probe.calls(),
        1,
        "case (b) reaches the device exactly once"
    );

    let expected: Vec<(f32, f32, f32, u32)> = vec![(0.0, 0.0, 0.0, 0); configs.len()];
    assert_eq!(empty, expected);
    assert_eq!(one, expected);
    assert_eq!(
        empty, one,
        "distinguishable: no — both are all-zero per config"
    );

    harness.stop();
}
