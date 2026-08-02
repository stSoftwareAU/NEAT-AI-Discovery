//! Deterministic wedged-GPU test double (Issue #1935).
//!
//! The wedge behind Issue #1926 only ever reproduced on one Apple M2 Ultra, so
//! every defence built for it — deadline-derived inner timeouts (#1928), stale
//! request skipping (#1929), the circuit breaker (#1930), the signalled partial
//! result (#1931), the typed non-retryable error (#1932) and the liveness
//! heartbeat (#1933) — risked shipping unverified.
//!
//! [`FakeGpuEvaluator`] is the substitute device those defences can be driven
//! against on a machine with no GPU at all. It implements [`RequestEvaluator`],
//! the seam [`GpuWorkQueue::run_work_loop`](super::GpuWorkQueue::run_work_loop)
//! drives, so the **production** loop, the **production** bounded wait and the
//! **production** breaker all run unchanged; only the device is fake.
//!
//! ```text
//! caller ──▶ work channel ──▶ run_work_loop ──▶ FakeGpuEvaluator
//!    ▲                            │ beats            │ selectable behaviour
//!    └── wait_for_gpu_response ◀──┘ heartbeat        └── never answers / beats
//!                                                        only / honours budget
//! ```
//!
//! ## Nothing here may wedge CI
//!
//! Every blocking behaviour is bounded twice: by [`HARNESS_HARD_CAP`], and by a
//! stop flag ([`FakeGpuProbe::release`]) so a test can free the worker thread
//! and join it. A harness bug therefore shows up as a test that fails its own
//! elapsed-time assertion, never as a suite that hangs.

use anyhow::{Result, anyhow};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::executor::{EvaluatorFactory, RequestEvaluator};
use crate::analysis::gpu::budget::GpuTimeBudget;
use crate::analysis::gpu::heartbeat::{GpuHeartbeat, STEP_SUB_BATCH_SUBMITTED};
use crate::analysis::samples::{
    HarmfulStats, HelpfulSample, HelpfulStats, ReluOrientation, ReluStats,
};

/// Upper bound on any blocking behaviour, whatever the test asks for.
///
/// A wedged device is modelled as "answers nothing", but a test double that
/// genuinely never returned would leave the suite hanging instead of failing.
/// Every wait ends here at the latest.
pub(crate) const HARNESS_HARD_CAP: Duration = Duration::from_secs(5);

/// How often a blocking behaviour re-checks its stop flag and its budget.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// How the fake device answers every evaluation it is handed.
#[derive(Debug, Clone, Copy)]
pub(crate) enum WedgeBehaviour {
    /// A healthy device: answers straight away.
    Completes,
    /// A slow but healthy device: silent for `delay`, then answers.
    CompletesAfter(Duration),
    /// A slow but *progressing* device: publishes `beats` progress steps
    /// `interval` apart, then answers. The stall guard must never flag this.
    BeatsThenCompletes {
        /// Gap between successive progress steps.
        interval: Duration,
        /// How many steps to publish before answering.
        beats: u32,
    },
    /// Publishes progress forever and never answers, so only the absolute
    /// batch timeout can end the caller's wait.
    BeatsWithoutCompleting {
        /// Gap between successive progress steps.
        interval: Duration,
    },
    /// The wedge from Issue #1926: silent forever, no progress, no answer.
    NeverAnswers,
    /// A wedged device whose inner waits honour the caller's budget (Issue
    /// #1928): blocks only while the request's [`GpuTimeBudget`] allows, then
    /// fails loudly instead of going silent.
    WedgesUntilBudgetExpires,
}

/// What the fake device saw for one evaluation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Observation {
    /// Whether the request carried a caller-derived budget at all.
    pub(crate) budget_bounded: bool,
    /// Budget left when the evaluation started.
    pub(crate) budget_remaining: Duration,
}

/// Test-side handles onto a [`FakeGpuEvaluator`] that has been moved into the
/// worker thread.
#[derive(Clone)]
pub(crate) struct FakeGpuProbe {
    calls: Arc<AtomicUsize>,
    observations: Arc<Mutex<Vec<Observation>>>,
    stop: Arc<AtomicBool>,
}

impl FakeGpuProbe {
    /// How many evaluations the device was actually asked to perform.
    pub(crate) fn calls(&self) -> usize {
        self.calls.load(Ordering::Acquire)
    }

    /// What the device saw, in call order.
    pub(crate) fn observations(&self) -> Vec<Observation> {
        self.observations
            .lock()
            .expect("fake GPU observation buffer poisoned")
            .clone()
    }

    /// Un-wedge the device so a blocked worker thread returns and can be joined.
    pub(crate) fn release(&self) {
        self.stop.store(true, Ordering::Release);
    }
}

/// A substitute GPU device with selectable wedge behaviour.
pub(crate) struct FakeGpuEvaluator {
    behaviour: WedgeBehaviour,
    /// The same heartbeat the work loop beats on and the caller watches.
    heartbeat: Arc<GpuHeartbeat>,
    probe: FakeGpuProbe,
}

impl FakeGpuEvaluator {
    /// Build a device with the given behaviour, publishing progress on
    /// `heartbeat`.
    pub(crate) fn new(behaviour: WedgeBehaviour, heartbeat: Arc<GpuHeartbeat>) -> Self {
        Self {
            behaviour,
            heartbeat,
            probe: FakeGpuProbe {
                calls: Arc::new(AtomicUsize::new(0)),
                observations: Arc::new(Mutex::new(Vec::new())),
                stop: Arc::new(AtomicBool::new(false)),
            },
        }
    }

    /// A handle that survives moving the device into the worker thread.
    pub(crate) fn probe(&self) -> FakeGpuProbe {
        self.probe.clone()
    }

    /// Whether a test has released the device.
    fn released(&self) -> bool {
        self.probe.stop.load(Ordering::Acquire)
    }

    /// Sleep in short steps until `is_done` or [`HARNESS_HARD_CAP`], returning
    /// `true` when `is_done` ended the wait.
    fn block_until(&self, mut is_done: impl FnMut() -> bool) -> bool {
        let started = Instant::now();
        loop {
            if is_done() {
                return true;
            }
            if self.released() || started.elapsed() >= HARNESS_HARD_CAP {
                return false;
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    /// Run the configured behaviour and produce this evaluation's result.
    fn respond<T>(&self, budget: GpuTimeBudget, value: T) -> Result<T> {
        self.probe.calls.fetch_add(1, Ordering::AcqRel);
        self.probe
            .observations
            .lock()
            .expect("fake GPU observation buffer poisoned")
            .push(Observation {
                budget_bounded: budget.is_bounded(),
                budget_remaining: budget.remaining(),
            });

        match self.behaviour {
            WedgeBehaviour::Completes => Ok(value),
            WedgeBehaviour::CompletesAfter(delay) => {
                let deadline = Instant::now() + delay;
                self.block_until(|| Instant::now() >= deadline);
                Ok(value)
            }
            WedgeBehaviour::BeatsThenCompletes { interval, beats } => {
                for _ in 0..beats {
                    let next = Instant::now() + interval;
                    self.block_until(|| Instant::now() >= next);
                    self.heartbeat.beat(STEP_SUB_BATCH_SUBMITTED);
                }
                Ok(value)
            }
            WedgeBehaviour::BeatsWithoutCompleting { interval } => {
                let cap = Instant::now() + HARNESS_HARD_CAP;
                while Instant::now() < cap && !self.released() {
                    let next = Instant::now() + interval;
                    self.block_until(|| Instant::now() >= next);
                    self.heartbeat.beat(STEP_SUB_BATCH_SUBMITTED);
                }
                Err(anyhow!(
                    "fake GPU published progress without ever completing (harness cap reached)"
                ))
            }
            WedgeBehaviour::NeverAnswers => {
                self.block_until(|| false);
                Err(anyhow!(
                    "fake GPU never answered (harness cap reached — the caller \
                     must have given up long before this)"
                ))
            }
            WedgeBehaviour::WedgesUntilBudgetExpires => {
                // Issue #1928: a correctly budgeted request bounds this wait.
                // Without a budget the loop below runs to the harness cap, which
                // is exactly the regression the calling test measures.
                self.block_until(|| budget.is_expired());
                budget.check("wedged sub-batch")?;
                Err(anyhow!(
                    "fake wedged GPU reached the {HARNESS_HARD_CAP:?} harness cap \
                     without an expiring budget — the request carried no caller deadline"
                ))
            }
        }
    }
}

impl RequestEvaluator for FakeGpuEvaluator {
    fn batch_size(&self) -> usize {
        1024
    }

    fn evaluate_helpful_batch(
        &self,
        _samples_batch: &[&[HelpfulSample]],
        budget: GpuTimeBudget,
    ) -> Result<Vec<HelpfulStats>> {
        self.respond(budget, vec![HelpfulStats::default()])
    }

    fn evaluate_harmful_batch(
        &self,
        _samples_batch: &[(&[HelpfulSample], f32)],
        budget: GpuTimeBudget,
    ) -> Result<Vec<HarmfulStats>> {
        self.respond(budget, Vec::new())
    }

    fn evaluate_relu(
        &self,
        _samples: &[HelpfulSample],
        _threshold: f32,
        budget: GpuTimeBudget,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        self.respond(
            budget,
            (
                ReluStats::new(ReluOrientation::Positive),
                ReluStats::new(ReluOrientation::Negative),
                0.0,
            ),
        )
    }

    fn evaluate_activation(
        &self,
        _samples: &[HelpfulSample],
        _activation_type: u32,
        _orientation: f32,
        _scale: f32,
        budget: GpuTimeBudget,
    ) -> Result<(f32, f32, f32, u32)> {
        self.respond(budget, (0.0, 0.0, 0.0, 0))
    }

    fn evaluate_activations_batched(
        &self,
        _samples: &[HelpfulSample],
        _activation_configs: &[(u32, f32, f32)],
        budget: GpuTimeBudget,
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        self.respond(budget, Vec::new())
    }
}

/// Factory that counts re-initialisation attempts and never recovers, so a
/// recovery the loop *did* attempt is unmistakable in the count.
pub(crate) struct FakeEvaluatorFactory {
    creates: AtomicUsize,
}

impl FakeEvaluatorFactory {
    pub(crate) const fn new() -> Self {
        Self {
            creates: AtomicUsize::new(0),
        }
    }

    /// How many times the loop tried to rebuild the device.
    pub(crate) fn create_count(&self) -> usize {
        self.creates.load(Ordering::Acquire)
    }
}

impl EvaluatorFactory<FakeGpuEvaluator> for FakeEvaluatorFactory {
    fn create(&self, _batch_size_override: Option<usize>) -> Result<FakeGpuEvaluator> {
        self.creates.fetch_add(1, Ordering::AcqRel);
        Err(anyhow!("the fake GPU never recovers"))
    }
}
