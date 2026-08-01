//! Neuron ranking and selection.
//!
//! Core types and functions for ranking neurons by their discovery potential.
//! Includes record providers (eager and lazy), ranking metrics, removal
//! candidate identification, and constant neuron removal.
//!
//! ## Sub-module Structure (Issue #564)
//!
//! - `record_providers` — Record provider trait and implementations (eager/lazy)
//! - `score_calculation` — Individual neuron ranking score computation
//! - `removal_candidates` — Removal candidate identification and constant neuron removal
//! - `removal_triage` — Structure-only removal triage, no parquet (Issue #1767)
//! - `reconstruction` — Reconstruction-mismatch focus signal (Issue #1634)

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
mod reconstruction;
pub(super) mod record_providers;
mod removal_candidates;
mod removal_triage;
mod score_calculation;

// Re-export public API — all items remain accessible via `crate::focus::ranking::*`
pub use record_providers::RecordProvider;
pub use removal_candidates::{RemovalCandidate, SynapseCounts, calculate_removal_savings};
// Issue #1767: structure-only removal triage helper — re-exported so `focus/mod.rs`
// (and the FFI layer) can reach it via `crate::focus::ranking::*`.
pub(crate) use removal_candidates::identify_structural_removal_candidates;
pub use removal_triage::{
    StructuralRemovalCandidate, StructuralRemovalTriage, triage_removal_candidates,
};
pub use score_calculation::{RankedNeuron, SelectionStats};

// Issue #1172 — `decide_records_loading` budget plumbing.
// Items defined in this module that are part of the public API.

// Re-export internal types needed by focus/tests.rs (unit tests for record providers)
pub(super) use record_providers::LazyRecordProvider;

use record_providers::{EagerRecordProvider, get_records_or_error};
use removal_candidates::{
    detect_constant_neuron_removals, effective_cost_of_growth, functionally_constant_focus_uuids,
    identify_removal_candidates,
};
use score_calculation::{
    activation_frequency_from_records, average_absolute_error_from_records,
    compute_frequency_factor, mean_absolute_activation_from_records,
    weighted_average_absolute_error_from_records,
};

use super::gradient::{
    build_squash_map, compute_gradient_flow_factor, compute_gradient_flow_for_neuron,
};
use super::impact::{
    compute_impacts_with_activations, compute_per_obs_margins, margin_weights_from_margins,
};
use crate::analysis::task_descriptor::{TargetTopology, TaskDescriptor};
use crate::analysis::utils::{
    bytes_to_mb_ceil, estimate_parquet_in_memory_bytes, get_memory_info,
    parquet_preload_fits_available, remaining_ms_until, verbose_enabled,
};
use crate::config::{
    FOCUS_RANKING_BUDGET_GRACE_MS, effective_focus_ranking_budget_ms,
    focus_ranking_memory_budget_mb, focus_ranking_memory_margin_mb, focus_ranking_perf_cliff_ms,
};
use crate::discovery_history::DiscoveryHistory;
use crate::ffi_types::DiscoveryError;
use crate::parquet_format::read_all_records_grouped_by_neuron_with_deadline;
use crate::parquet_format::shared_records::load_grouped_records_shared;
use crate::{CoordinatedStructuralCandidateJson, CreatureJson, NeuronJson};
use anyhow::{Context, Result};
use rayon::prelude::*;

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// Loading mode chosen by [`rank_focus_neurons`] for accessing recorded
/// discovery data (Issue #1172).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FocusLoadingMode {
    /// All records pre-loaded into memory up-front. Fast but memory-heavy.
    #[default]
    Preload,
    /// Records loaded on demand from parquet with a bounded LRU cache.
    /// Slower but memory-efficient.
    Lazy,
}

impl FocusLoadingMode {
    /// Stable lower-case identifier for structured logging and FFI surfaces.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Preload => "preload",
            Self::Lazy => "lazy",
        }
    }
}

/// Reason the focus ranker chose lazy mode (Issue #1172).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FocusLazyReason {
    /// Lazy mode was not selected (the run used preload).
    #[default]
    None,
    /// `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB` was set and the
    /// projected pre-load size exceeded it.
    Budget,
    /// No explicit budget was set and the projected pre-load did not fit
    /// within real OS-available memory after the safety margin (Issue #1376).
    MemoryPressure,
}

impl FocusLazyReason {
    /// Stable lower-case identifier for structured logging and FFI surfaces.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Budget => "budget",
            Self::MemoryPressure => "memory_pressure",
        }
    }
}

/// Outcome of a focus-ranking pass: the ranked neurons plus removal candidates,
/// run statistics, and diagnostics produced by [`rank_focus_neurons`].
#[derive(Debug, Default)]
pub struct RankFocusStats {
    pub neurons: Vec<RankedNeuron>,
    /// Neurons with impact below costOfGrowth - candidates for removal
    pub removal_candidates: Vec<RemovalCandidate>,
    /// Issue #306: Coordinated structural candidates for removing constant-value neurons.
    /// When a hidden neuron has near-zero activation variance (constant output), it can be
    /// removed and its effect folded into bias adjustments for downstream neurons.
    /// Each candidate contains:
    /// - A `RemoveNeuron` operation for the constant neuron
    /// - `SetBias` operations for all downstream neurons with adjusted biases
    pub constant_neuron_removals: Vec<CoordinatedStructuralCandidateJson>,
    pub max_output_error: f32,
    pub processed_neurons: usize,
    pub total_neurons: usize,
    pub duration_ms: u128,
    /// Aggregate rejection counts keyed by stable reason name (Issue #1142,
    /// reusing the Issue #1129 rejection-reason vocabulary).
    ///
    /// Currently populated with [`crate::analysis::diagnostics::rejection_reasons::REJECTION_REMOVAL_BELOW_NOISE_FLOOR`] counts
    /// for removal candidates dropped by the noise-floor gate. Surfaced
    /// verbatim into `RankFocusNeuronsOutput.rejection_breakdown`.
    pub rejection_breakdown: std::collections::HashMap<String, u32>,
    /// Record loading mode chosen for the run (Issue #1172).
    pub loading_mode: FocusLoadingMode,
    /// Reason lazy mode was selected, if any (Issue #1172). Set to
    /// [`FocusLazyReason::None`] when [`Self::loading_mode`] is
    /// [`FocusLoadingMode::Preload`].
    pub lazy_reason: FocusLazyReason,
    /// Configured memory budget in megabytes when
    /// `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB` was set
    /// (Issue #1172). `None` when no explicit budget was provided.
    pub budget_mb: Option<u64>,
    /// Projected in-memory size of the parquet pre-load in megabytes
    /// (Issue #1172). `0` when the file size could not be determined.
    pub projected_mb: u64,
    /// Number of functionally-constant hidden neurons excluded from focus-slot
    /// eligibility this pass (Issue #1624). Non-zero only when
    /// `NEAT_AI_DISCOVERY_FOCUS_EXCLUDE_CONSTANT_NEURONS` is enabled; the
    /// slot-waste measurement the exclusion recovered.
    pub focus_ineligible_constant: usize,
    /// Number of near-zero-impact neurons gated out of focus-slot eligibility
    /// this pass (Issue #1635). Non-zero only when
    /// `NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE` is enabled; these neurons vary
    /// across samples (so the constant filter leaves them in) yet carry a
    /// structural impact magnitude below the configured gate, so no change
    /// feeding them can move the output.
    pub focus_ineligible_low_impact: usize,
}

pub(super) fn is_selectable_type(neuron_type: &str) -> bool {
    neuron_type != "input" && neuron_type != "constant"
}

/// Wall-clock deadline guarding a focus-ranking run (Issue #1375).
///
/// Focus ranking previously had no time bound, so a pathological run (the
/// #1373 incident: 1h 11m) could blow the entire discovery wall-clock budget.
/// This mirrors the per-chunk Rust FFI analysis budget: the ranking pipeline
/// checks the deadline between passes and inside the per-neuron loops, and
/// aborts with a structured [`DiscoveryError::Timeout`] when exceeded so the
/// TypeScript caller falls back to its instant local ranking path.
#[derive(Debug, Clone, Copy)]
pub struct FocusDeadline {
    /// Instant beyond which the run must abort (budget + grace from `start`).
    expires_at: Instant,
    /// Configured budget in milliseconds, surfaced in the timeout error.
    budget_ms: u64,
}

impl FocusDeadline {
    /// Build a deadline `budget_ms` (+ grace) after `start`.
    #[must_use]
    pub fn new(start: Instant, budget_ms: u64) -> Self {
        let total = budget_ms.saturating_add(FOCUS_RANKING_BUDGET_GRACE_MS);
        Self {
            expires_at: start + Duration::from_millis(total),
            budget_ms,
        }
    }

    /// Build a deadline with an explicit `grace_ms` instead of the fixed
    /// production grace (Issue #1760).
    ///
    /// Test-only seam: the production grace is a full second, which dominates a
    /// unit test's wall clock and made the abort test contention-sensitive.
    /// Injecting a small grace lets the deadline fire promptly so the test can
    /// assert the *observable* abort outcome rather than a machine-speed timing.
    #[cfg(test)]
    pub(in crate::focus) fn with_grace_for_tests(
        start: Instant,
        budget_ms: u64,
        grace_ms: u64,
    ) -> Self {
        let total = budget_ms.saturating_add(grace_ms);
        Self {
            expires_at: start + Duration::from_millis(total),
            budget_ms,
        }
    }

    /// Resolve the optional focus-ranking deadline for a run starting at
    /// `start` (Issue #1407).
    ///
    /// Two budgets bound focus ranking and the **earlier** wins:
    /// 1. The shared **absolute** discovery deadline (`shared_deadline_ms`,
    ///    ms-since-epoch) that the synapse/neuron analysis phase also bills
    ///    against. Threading the same deadline through both phases means time
    ///    spent selecting focus neurons reduces the window left for analysis
    ///    instead of each phase opening a fresh independent window.
    /// 2. Focus ranking's own wall-clock safety net
    ///    (`NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS`, default
    ///    [`crate::config::DEFAULT_FOCUS_RANKING_BUDGET_MS`]) which still guards
    ///    a pathological ranking run even when no shared deadline is supplied.
    ///    Issue #3172: this safety net is **scaled** by the resolved loading
    ///    `mode` + `projected_mb` — a lazy fallback on a large dataset is
    ///    materially slower, so it earns a larger budget than the fast eager
    ///    path, which keeps the unscaled default. An explicit env override still
    ///    wins verbatim.
    ///
    /// Returns `None` only when both bounds are absent — no shared deadline AND
    /// the focus budget explicitly disabled with `0`.
    fn resolve(
        start: Instant,
        shared_deadline_ms: Option<u64>,
        mode: FocusLoadingMode,
        projected_mb: u64,
    ) -> Option<Self> {
        let is_lazy = mode == FocusLoadingMode::Lazy;
        let budget_deadline = effective_focus_ranking_budget_ms(is_lazy, projected_mb)
            .map(|budget_ms| Self::new(start, budget_ms));
        let shared_deadline = Self::from_shared_deadline(start, shared_deadline_ms);

        match (budget_deadline, shared_deadline) {
            (Some(budget), Some(shared)) => Some(if shared.expires_at <= budget.expires_at {
                shared
            } else {
                budget
            }),
            (Some(budget), None) => Some(budget),
            (None, Some(shared)) => Some(shared),
            (None, None) => None,
        }
    }

    /// Build a focus deadline anchored on `start` from the shared absolute
    /// discovery deadline (Issue #1407).
    ///
    /// The shared deadline is the same value the analysis phase enforces, so no
    /// focus-ranking grace is added here — the budget path keeps its own grace.
    /// Returns `None` when no shared deadline is supplied or the system clock is
    /// before the UNIX epoch.
    fn from_shared_deadline(start: Instant, shared_deadline_ms: Option<u64>) -> Option<Self> {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()?
            .as_millis() as u64;
        let remaining_ms = remaining_ms_until(shared_deadline_ms, now_ms)?;
        Some(Self {
            expires_at: start + Duration::from_millis(remaining_ms),
            budget_ms: remaining_ms,
        })
    }

    /// Abort with a structured timeout error if the deadline has passed.
    ///
    /// `context` names the pass that observed the overrun so logs make the
    /// abort point obvious. Kept cheap (a single `Instant::now()`) so the
    /// per-neuron checks add no measurable overhead on fast runs.
    fn check(&self, context: &str) -> Result<()> {
        if Instant::now() >= self.expires_at {
            tracing::warn!(
                target: "neat_ai_discovery::focus::ranking",
                budget_ms = self.budget_ms,
                grace_ms = FOCUS_RANKING_BUDGET_GRACE_MS,
                context,
                "focus::ranking aborted: wall-clock budget exceeded",
            );
            return Err(DiscoveryError::Timeout {
                deadline_ms: self.budget_ms,
            }
            .into());
        }
        Ok(())
    }

    /// Project this deadline onto the wall clock so the parquet reader — which
    /// checks a [`SystemTime`] at every record-batch boundary — can bill against
    /// the same budget (Issue #3686).
    ///
    /// Returns `SystemTime::now()` when the deadline has already passed, so the
    /// reader aborts immediately instead of wrapping into a distant future.
    fn as_system_time(&self) -> SystemTime {
        let now = Instant::now();
        let remaining = self.expires_at.saturating_duration_since(now);
        SystemTime::now() + remaining
    }
}

/// Project an optional focus deadline onto the wall clock (Issue #3686).
fn deadline_as_system_time(deadline: Option<FocusDeadline>) -> Option<SystemTime> {
    deadline.map(|d| d.as_system_time())
}

/// Optional deadline check helper — a no-op when no deadline is configured.
fn check_deadline(deadline: Option<FocusDeadline>, context: &str) -> Result<()> {
    match deadline {
        Some(d) => d.check(context),
        None => Ok(()),
    }
}

/// Loading-decision metadata threaded into the shared ranking core so both
/// public entry points report identical mode/projection stats (Issue #1375).
#[derive(Debug, Clone, Copy)]
struct LoadingMeta {
    mode: FocusLoadingMode,
    reason: FocusLazyReason,
    budget_mb: Option<u64>,
    projected_mb: u64,
}

/// Cost per hidden neuron when the caller supplies none — matches NEAT-AI's
/// `Score.ts` formula.
///
/// The **single** definition of the default (Issue #1807): every caller,
/// including the FFI entry point, resolves through this constant rather than
/// repeating the literal, so the two can never drift apart.
pub const DEFAULT_COST_OF_GROWTH: f32 = 1e-7;
const IMPACT_EPSILON: f32 = 0.0001;
const IMPACT_GAMMA: f32 = 0.8;

/// Loading plan: the eager-vs-lazy decision plus its projection, decided
/// **before** the (possibly expensive) record provider is built (Issue #3172).
///
/// Separating the cheap decision from the provider build lets the wall-clock
/// deadline be scaled to the chosen `mode` + `projected_mb` before the lazy warm
/// pass (a full parquet decode) starts consuming the budget.
#[derive(Debug, Clone, Copy)]
struct LoadingPlan {
    mode: FocusLoadingMode,
    reason: FocusLazyReason,
    budget_mb: Option<u64>,
    projected_mb: u64,
}

/// Pure decision helper for the configurable memory budget (Issue #1172).
///
/// Compares the projected in-memory size against the configured budget in
/// **bytes** so 1 MB granularity rounding does not distort comparisons for
/// small parquet files. Returns the chosen mode and a stable lazy reason.
///
/// Exposed publicly so unit tests can exercise the budget logic directly
/// without needing to materialise a parquet file large enough to exceed
/// 1 MB after the 3× decompression multiplier.
#[must_use]
pub fn decide_loading_mode_for_budget(
    projected_bytes: u64,
    budget_mb: u64,
) -> (FocusLoadingMode, FocusLazyReason) {
    const BYTES_PER_MB: u64 = 1024 * 1024;
    let budget_bytes = budget_mb.saturating_mul(BYTES_PER_MB);
    if projected_bytes > budget_bytes {
        (FocusLoadingMode::Lazy, FocusLazyReason::Budget)
    } else {
        (FocusLoadingMode::Preload, FocusLazyReason::None)
    }
}

/// Pure decision helper for the auto-detect (no explicit budget) path
/// (Issue #1376).
///
/// Bases the eager-vs-lazy decision on **real OS-available memory** with a
/// safety margin reserved for the system / GPU buffers, rather than the
/// 50%-of-total-RAM cap that previously rejected mid-sized parquet files on
/// hosts with plenty of free memory (the production memory regression: ~1.6 GB projection
/// dropped to lazy despite ~3 GB free).
///
/// Pre-loads when `projected_bytes <= available_bytes − margin_bytes`,
/// otherwise selects lazy mode with [`FocusLazyReason::MemoryPressure`].
///
/// Exposed publicly so unit tests can exercise the available-memory branch
/// directly without sampling live system memory.
#[must_use]
pub fn decide_loading_mode_for_available_memory(
    projected_bytes: u64,
    available_bytes: u64,
    margin_bytes: u64,
) -> (FocusLoadingMode, FocusLazyReason) {
    if parquet_preload_fits_available(projected_bytes, available_bytes, margin_bytes) {
        (FocusLoadingMode::Preload, FocusLazyReason::None)
    } else {
        (FocusLoadingMode::Lazy, FocusLazyReason::MemoryPressure)
    }
}

/// Pure perf-cliff decision for a completed focus-ranking pass (Issue #1377).
///
/// Returns `true` only for a **lazy** pass whose wall-clock `elapsed_ms` reaches
/// the configured `threshold_ms`. The fast preload path never trips it, and a
/// `threshold_ms` of `0` disables the perf-cliff warning entirely.
///
/// Exposed publicly so unit tests can exercise the boundary directly without
/// timing a real ranking pass.
#[must_use]
pub fn lazy_pass_exceeds_perf_cliff(
    mode: FocusLoadingMode,
    elapsed_ms: u128,
    threshold_ms: u64,
) -> bool {
    mode == FocusLoadingMode::Lazy && threshold_ms > 0 && elapsed_ms >= u128::from(threshold_ms)
}

/// Decide eager-vs-lazy loading and log the decision, **without** building the
/// (possibly expensive) record provider (Issue #3172).
///
/// This is the cheap half of loading — a parquet-size estimate plus an
/// OS-memory query — split out so [`FocusDeadline::resolve`] can scale the
/// wall-clock budget to the chosen `mode` + `projected_mb` before the lazy warm
/// pass (a full parquet decode) begins consuming that budget.
///
/// Behaviour (Issue #1172):
/// - When `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB` is set, the
///   projected in-memory size (file size × 3) is compared against the budget.
///   Lazy mode is selected with a structured `WARN` when the projection exceeds
///   the budget.
/// - When the budget is unset, the auto-detect path compares the projection
///   against real OS-available memory minus a configurable safety margin
///   (Issue #1376). Pre-load is chosen whenever the projection fits, so hosts
///   with GBs free stay on the fast path; lazy mode (with a `WARN`) is reserved
///   for genuinely memory-constrained hosts.
fn plan_loading(parquet_file: &str, selectable_len: usize) -> LoadingPlan {
    const BYTES_PER_MB: u64 = 1024 * 1024;
    let budget_mb = focus_ranking_memory_budget_mb();
    let projected_bytes = estimate_parquet_in_memory_bytes(parquet_file);
    let projected_mb = bytes_to_mb_ceil(projected_bytes);

    if let Some(budget) = budget_mb {
        let (mode, reason) = decide_loading_mode_for_budget(projected_bytes, budget);
        if mode == FocusLoadingMode::Lazy {
            // Issue #1377: escalate to WARN and record available memory
            // alongside projected/budget so the eager-vs-lazy trade-off is
            // visible at the decision point, not split across log lines.
            let (available_bytes, _total_bytes) = get_memory_info();
            tracing::warn!(
                target: "neat_ai_discovery::focus::ranking",
                mode = FocusLoadingMode::Lazy.as_str(),
                reason = reason.as_str(),
                budget_mb = budget,
                projected_mb,
                available_mb = bytes_to_mb_ceil(available_bytes),
                selectable = selectable_len,
                "focus::ranking selected lazy mode: projected pre-load exceeds configured budget",
            );
        }
        return LoadingPlan {
            mode,
            reason,
            budget_mb: Some(budget),
            projected_mb,
        };
    }

    // Issue #1376: base the decision on real OS-available memory minus a safety
    // margin, rather than the 50%-of-total-RAM cap that dropped mid-sized
    // parquet files onto the slow lazy path while GBs of RAM were free.
    let (available_bytes, _total_bytes) = get_memory_info();
    let margin_mb = focus_ranking_memory_margin_mb();
    let margin_bytes = margin_mb.saturating_mul(BYTES_PER_MB);
    let available_mb = bytes_to_mb_ceil(available_bytes);
    let (mode, reason) =
        decide_loading_mode_for_available_memory(projected_bytes, available_bytes, margin_bytes);

    if mode == FocusLoadingMode::Lazy {
        tracing::warn!(
            target: "neat_ai_discovery::focus::ranking",
            mode = FocusLoadingMode::Lazy.as_str(),
            reason = reason.as_str(),
            projected_mb,
            available_mb,
            // Issue #1377: no explicit budget on the auto-detect path; log 0
            // so the field is uniform with the budget path's lazy log.
            budget_mb = 0u64,
            margin_mb,
            selectable = selectable_len,
            "Insufficient available memory for full pre-load in focus ranking. \
             Using lazy-loading mode (slower but memory-efficient).",
        );
    }

    LoadingPlan {
        mode,
        reason,
        budget_mb: None,
        projected_mb,
    }
}

/// Build the record provider for a resolved [`LoadingPlan`] (Issue #3172).
///
/// The expensive half of loading: an eager pre-load decodes the whole parquet
/// file through the process-shared cache (Issue #1406) so the analysis phase can
/// reuse it; a lazy plan warms a bounded, working-set-sized cache
/// ([`build_lazy_provider`]). Kept separate from [`plan_loading`] so this work
/// runs *after* the wall-clock deadline has been scaled to the plan.
fn build_provider(
    parquet_file: &str,
    selectable: &[&NeuronJson],
    plan: LoadingPlan,
    deadline: Option<FocusDeadline>,
) -> Result<Arc<dyn RecordProvider>> {
    // Issue #3686: the provider build is the single most expensive step of a
    // ranking run — it decodes the whole recorded dataset — and it used to run
    // with NO deadline. On a memory-constrained host (12.5 GB of records against
    // 7 GB available) the lazy warm pass ground for 2 h 28 m past a 14-minute
    // budget, and the run was only aborted afterwards, by which time the calling
    // worker's 3 h wall-clock cap had killed it outright. Bill the build against
    // the same deadline the ranking passes use.
    let read_deadline = deadline_as_system_time(deadline);
    match plan.mode {
        FocusLoadingMode::Lazy => build_lazy_provider(parquet_file, selectable, deadline),
        FocusLoadingMode::Preload => match load_grouped_records_shared(parquet_file, read_deadline)
        {
            Ok(shared) => Ok(Arc::new(EagerRecordProvider::from_shared(&shared))),
            Err(read_err) => {
                // Issue #1768: if the eager pre-load overran the wall-clock
                // budget, abort with a distinct `eager_preload` context (a
                // structured timeout) rather than a generic read error, so the
                // abort point is unambiguous in the logs. A genuine read error
                // *within* budget still surfaces verbatim.
                check_deadline(deadline, "eager_preload")?;
                Err(read_err).context("Failed to read discovery records from parquet file")
            }
        },
    }
}

/// Build a lazy record provider whose cache is sized to the ranking working set
/// and warmed by a single grouped parquet pass (Issue #1374).
///
/// The ranking pipeline sweeps over every selectable neuron several times. The
/// previous lazy provider used an 8-entry cache and re-read the entire parquet
/// file on every cache miss, so a working set larger than 8 neurons thrashed
/// into `O(passes × neurons)` full-file decodes. Here we:
/// 1. size the cache to the selectable set so nothing is evicted mid-run, and
/// 2. warm it with **one** grouped decode, retaining only the selectable
///    neurons' records.
///
/// The per-neuron loader remains as a fallback for any neuron missing from the
/// warm pass (e.g. neurons with no recorded data), and the bounded-but-
/// sufficient cache still guarantees each is loaded at most once.
///
/// Issue #3686: the warm pass is bounded by `deadline` — the reader checks it at
/// every record-batch boundary, so the decode cannot grind for hours past the
/// budget.
///
/// Issue #1768: a warm pass that fails **because the wall-clock budget overran**
/// aborts loudly with a distinct [`DiscoveryError::Timeout`] under the
/// `lazy_warm_pass` context, rather than silently degrading to on-demand
/// loading. The old soft fallback masked the timeout as a clean provider, so the
/// observable abort only surfaced hours-equivalent later in the unrelated
/// `verify_selectable_records` per-neuron loop — a misleading context and, worse,
/// an on-demand path that re-reads the whole parquet file per neuron. Only a
/// **genuine read error within budget** now earns the soft on-demand fallback,
/// where the caller's own per-neuron deadline check still decides the run's fate.
fn build_lazy_provider(
    parquet_file: &str,
    selectable: &[&NeuronJson],
    deadline: Option<FocusDeadline>,
) -> Result<Arc<dyn RecordProvider>> {
    let provider = LazyRecordProvider::with_capacity(parquet_file, selectable.len());
    let read_deadline = deadline_as_system_time(deadline);

    match read_all_records_grouped_by_neuron_with_deadline(parquet_file, read_deadline) {
        Ok(mut grouped) => {
            let wanted: std::collections::HashSet<&str> =
                selectable.iter().map(|n| n.uuid.as_str()).collect();
            grouped.retain(|uuid, _| wanted.contains(uuid.as_str()));
            if let Err(seed_err) = provider.seed(grouped) {
                tracing::warn!(
                    error = %seed_err,
                    "Lazy focus-ranking cache seed failed; falling back to on-demand loading",
                );
            }
        }
        Err(read_err) => {
            // If the deadline has passed, the read aborted BECAUSE of the budget:
            // fail loud with a distinct `lazy_warm_pass` timeout so the run stops
            // within budget at an unambiguous point. `check_deadline` returns the
            // structured `DiscoveryError::Timeout` (and logs the context) only
            // when expired; otherwise it is a no-op and we degrade gracefully.
            check_deadline(deadline, "lazy_warm_pass")?;
            tracing::warn!(
                error = %read_err,
                "Lazy focus-ranking cache warm pass failed within budget; \
                 falling back to on-demand per-neuron loading",
            );
        }
    }

    Ok(Arc::new(provider))
}

/// Emit the structured end-of-pass summary (Issue #1172). Always logged at
/// `info` level so callers can scrape mode/projection metrics without enabling
/// verbose mode.
fn log_focus_ranking_summary(
    decision_mode: FocusLoadingMode,
    decision_reason: FocusLazyReason,
    budget_mb: Option<u64>,
    projected_mb: u64,
    entries: usize,
    elapsed_ms: u128,
) {
    match decision_mode {
        FocusLoadingMode::Preload => tracing::info!(
            target: "neat_ai_discovery::focus::ranking",
            mode = decision_mode.as_str(),
            entries,
            elapsed_ms = elapsed_ms.min(u64::MAX as u128) as u64,
            "focus::ranking pass complete",
        ),
        FocusLoadingMode::Lazy => tracing::info!(
            target: "neat_ai_discovery::focus::ranking",
            mode = decision_mode.as_str(),
            reason = decision_reason.as_str(),
            budget_mb = budget_mb.unwrap_or(0),
            projected_mb,
            entries,
            elapsed_ms = elapsed_ms.min(u64::MAX as u128) as u64,
            "focus::ranking pass complete",
        ),
    }

    maybe_warn_perf_cliff(
        decision_mode,
        decision_reason,
        projected_mb,
        entries,
        elapsed_ms,
    );
}

/// Emit a single, clearly-labelled perf-cliff `WARN` when a lazy focus-ranking
/// pass exceeds the configured threshold (Issue #1377).
///
/// The #1373 incident — a lazy ranking pass running for 1h 11m — surfaced only
/// as an opaque per-phase timing figure with no attribution to the lazy Rust
/// path. This makes the cliff loud at the moment it happens, naming the neuron
/// count and projected dataset size so the trade-off and likely remedy (raise
/// the memory budget or free host memory to re-enable preload) are obvious.
fn maybe_warn_perf_cliff(
    mode: FocusLoadingMode,
    reason: FocusLazyReason,
    projected_mb: u64,
    entries: usize,
    elapsed_ms: u128,
) {
    let threshold_ms = focus_ranking_perf_cliff_ms();
    if lazy_pass_exceeds_perf_cliff(mode, elapsed_ms, threshold_ms) {
        tracing::warn!(
            target: "neat_ai_discovery::focus::ranking",
            mode = mode.as_str(),
            reason = reason.as_str(),
            neurons = entries,
            projected_mb,
            elapsed_ms = elapsed_ms.min(u64::MAX as u128) as u64,
            threshold_ms,
            "focus::ranking PERF CLIFF: lazy ranking pass exceeded the perf-cliff \
             threshold — raise NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB or free \
             host memory to re-enable the faster preload path",
        );
    }
}

/// Compute max output error across all output neurons.
///
/// When `obs_weights` is provided (`OneHot` / `Margin` descriptors), the per-output
/// error is reweighted by the per-observation margin weight so the clamp scale
/// stays consistent with the margin-weighted per-neuron error (Issue #1318).
fn compute_max_output_error(
    creature: &CreatureJson,
    records_provider: &dyn RecordProvider,
    obs_weights: Option<&std::collections::HashMap<u32, f32>>,
) -> Result<f32> {
    let output_neurons: Vec<&NeuronJson> = creature
        .neurons
        .iter()
        .filter(|neuron| neuron.neuron_type == "output")
        .collect();

    if output_neurons.is_empty() {
        return Ok(0.0);
    }

    let errors: Vec<f32> = output_neurons
        .iter()
        .map(|neuron| {
            let records = get_records_or_error(records_provider, &neuron.uuid)?;
            Ok(weighted_average_absolute_error_from_records(
                &records,
                obs_weights,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(errors.into_iter().fold(0.0, f32::max))
}

/// Read-only per-neuron metric inputs shared across the ranked-neuron build
/// pass. Grouped into a struct to avoid `clippy::too_many_arguments` (mirrors
/// the pattern in [`super::impact`]).
struct RankedNeuronInputs<'a> {
    impact_map: &'a std::collections::HashMap<String, f32>,
    squash_map: &'a std::collections::HashMap<String, String>,
    max_output_error: f32,
    /// `OneHot` / `Margin` per-observation margin weights (Issue #1318).
    obs_weights: Option<&'a std::collections::HashMap<u32, f32>>,
    /// Per-neuron reconstruction activation delta (Issue #1634); empty when the
    /// signal is disabled.
    reconstruction_mismatch: &'a std::collections::HashMap<String, f32>,
}

/// Build ranked neurons from selectable neurons with their metrics.
///
/// When `inputs.obs_weights` is provided (`OneHot` / `Margin` descriptors),
/// per-neuron errors are aggregated using the per-observation margin weight
/// rather than the unweighted mean. Issue #1318.
fn build_ranked_neurons(
    selectable: &[&NeuronJson],
    records_provider: &dyn RecordProvider,
    inputs: &RankedNeuronInputs<'_>,
    deadline: Option<FocusDeadline>,
) -> Result<Vec<RankedNeuron>> {
    let obs_weights = inputs.obs_weights;
    let max_output_error = inputs.max_output_error;
    selectable
        .par_iter()
        .map(|neuron| -> Result<RankedNeuron> {
            // Issue #1375: bound the per-neuron ranking loop so a pathological
            // run aborts within budget instead of grinding for an hour.
            check_deadline(deadline, "build_ranked_neurons")?;
            let records = get_records_or_error(records_provider, &neuron.uuid)?;
            let raw_error = if obs_weights.is_some() {
                weighted_average_absolute_error_from_records(&records, obs_weights)
            } else {
                average_absolute_error_from_records(&records)
            };
            let structural_impact = *inputs.impact_map.get(&neuron.uuid).unwrap_or(&0.0);
            let mean_activation = mean_absolute_activation_from_records(&records);

            // Activation-weighted impact reflects the ACTUAL contribution during inference.
            // A neuron with tiny structural impact but massive activations still contributes
            // significantly: actual_contribution ≈ weight × activation
            //
            // Only neurons with BOTH low structural impact AND low activation should be
            // removal candidates. If either is high, the neuron is contributing.
            let activation_weighted_impact = structural_impact * mean_activation;

            let total_error = if max_output_error > 0.0 {
                raw_error.min(max_output_error)
            } else {
                raw_error
            };

            // Issue #206: Compute gradient flow stats for this neuron
            let gradient_flow =
                compute_gradient_flow_for_neuron(&neuron.uuid, inputs.squash_map, &records);

            // Issue #204: Compute activation frequency for focus neuron ranking
            let activation_frequency = activation_frequency_from_records(&records);

            // Issue #1634: reconstruction mismatch for this neuron (0.0 when the
            // signal is disabled or no reconstruction was available).
            let reconstruction_mismatch = inputs
                .reconstruction_mismatch
                .get(&neuron.uuid)
                .copied()
                .unwrap_or(0.0);

            Ok(RankedNeuron {
                neuron_uuid: neuron.uuid.clone(),
                total_error,
                raw_error,
                impact: structural_impact,
                mean_activation,
                activation_weighted_impact,
                gradient_flow,
                activation_frequency,
                // Issue #1445: populated after construction by the ranking pass
                // via compute_focus_weighted_score; 0.0 is a safe placeholder.
                weighted_score: 0.0,
                reconstruction_mismatch,
            })
        })
        .collect::<Result<Vec<_>>>()
}

/// Compute the combined impact-weighted ranking score for a neuron (Issue
/// #1445). This is the single source of truth for both the focus-list ordering
/// and the roulette weight used by diversity-aware focus selection.
///
/// Score = `error × (impact + ε)^γ × gradient_factor × frequency_factor`, scaled
/// by the optional Bayesian history multiplier `0.5 + history` (Issue #227) when
/// a per-neuron history factor is supplied. With no history the multiplier is
/// absent and the ordering matches the non-history path.
///
/// Issue #1634: when the reconstruction-mismatch signal is enabled, an additive
/// term `recon_weight × reconstruction_mismatch` is added *after* the
/// multiplicative history scaling so poorly-reconstructed neurons rise in the
/// focus budget. `recon_weight` is `0.0` when the signal is disabled, which adds
/// nothing and keeps the score byte-identical to the pre-#1634 path.
#[must_use]
pub(super) fn compute_focus_weighted_score(
    neuron: &RankedNeuron,
    history_factor: Option<f32>,
    recon_weight: f32,
) -> f32 {
    let base = neuron.total_error * (neuron.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);
    let gradient_factor = compute_gradient_flow_factor(&neuron.gradient_flow);
    let frequency_factor = compute_frequency_factor(neuron.activation_frequency);
    let with_gradient = base * gradient_factor * frequency_factor;
    let scaled = match history_factor {
        Some(h) => with_gradient * (0.5 + h),
        None => with_gradient,
    };
    // Additive reconstruction-mismatch bonus (Issue #1634). recon_weight is 0.0
    // when the signal is disabled, so this is a no-op for the legacy path.
    scaled + recon_weight * neuron.reconstruction_mismatch
}

/// Whether a neuron's structural impact magnitude falls **below** the focus gate
/// (Issue #1635).
///
/// The gate is applied on `|impact|`. A neuron whose magnitude is at or above the
/// gate is retained (`false`); one strictly below is gated out (`true`). The
/// boundary rule is therefore **retain-on-equal**: a neuron exactly at the gate
/// stays eligible. Non-finite impacts (`NaN`, `±inf` are impossible here but
/// guarded anyway) are treated as below the gate — they carry no usable signal.
#[must_use]
fn impact_below_gate(impact: f32, gate: f32) -> bool {
    let magnitude = impact.abs();
    // Non-finite magnitude (`NaN`) carries no usable signal → gate it out.
    // `magnitude < gate` gives retain-on-equal at the boundary.
    !magnitude.is_finite() || magnitude < gate
}

/// Drop near-zero-impact neurons from the ranked focus list (Issue #1635),
/// returning the number gated out.
///
/// Complements the constant-neuron filter (#1624): a neuron can vary across
/// samples (so it is not functionally constant) yet still have a structural
/// impact magnitude below `gate`, in which case no add-synapse / add-neuron
/// change feeding it can move the output — the focus slot is wasted. Retains the
/// caller's ordering for the survivors.
#[must_use]
fn apply_focus_impact_gate(neurons: &mut Vec<RankedNeuron>, gate: f32) -> usize {
    let before = neurons.len();
    neurons.retain(|n| !impact_below_gate(n.impact, gate));
    before - neurons.len()
}

/// Ranks a creature's focus neurons by impact, loading recorded samples from
/// `parquet_file`.
///
/// This is the primary public entry point for focus selection. It computes the
/// per-neuron ranking score (error × impact, gradient- and frequency-weighted),
/// orders the neurons, and returns the ranking alongside removal candidates and
/// run statistics in a [`RankFocusStats`]. `max_results` caps the size of the
/// returned ranked pool; `cost_of_growth` sets the impact threshold below which
/// neurons become removal candidates. This is the descriptor-free, deadline-free
/// convenience wrapper around [`rank_focus_neurons_with_descriptor`].
///
/// # Errors
///
/// Returns an error if the Parquet file cannot be read, its schema is invalid,
/// or the underlying impact computation fails.
///
/// # Examples
///
/// ```rust,ignore
/// use neat_ai_discovery::rank_focus_neurons;
///
/// let stats = rank_focus_neurons("discovery.parquet", &creature, Some(6), Some(1e-7))?;
/// for neuron in &stats.neurons {
///     println!("{}: {}", neuron.neuron_uuid, neuron.weighted_score);
/// }
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn rank_focus_neurons(
    parquet_file: &str,
    creature: &CreatureJson,
    max_results: Option<usize>,
    cost_of_growth: Option<f32>,
) -> Result<RankFocusStats> {
    rank_focus_neurons_with_descriptor(parquet_file, creature, max_results, cost_of_growth, None)
}

/// Returns `true` when the descriptor's target topology should activate
/// margin-aware focus ranking (Issue #1318). Currently `OneHot` and `Margin`.
fn descriptor_activates_margin_ranking(descriptor: Option<&TaskDescriptor>) -> bool {
    descriptor.is_some_and(|d| {
        matches!(
            d.target_topology,
            TargetTopology::OneHot | TargetTopology::Margin
        )
    })
}

/// Rank focus neurons with an optional [`TaskDescriptor`] (Issue #1318).
///
/// When the descriptor reports a `OneHot` or `Margin` topology, the per-neuron
/// error component of the ranking score is replaced with a **margin-weighted**
/// average — observations where the network's decision margin (top-1 vs top-2
/// output activation) is small contribute more, observations where the margin
/// is wide contribute less. This targets the plateau where margin-improving
/// changes that don't yet flip an argmax otherwise look worthless.
///
/// For `OTHER` / `Unknown` / `Independent` / `Simplex` topologies and for
/// `None`, the ranking falls back to the existing arithmetic-mean error path
/// (regression guard).
///
/// # Errors
///
/// Returns an error if the underlying record provider or impact computation
/// fails.
pub fn rank_focus_neurons_with_descriptor(
    parquet_file: &str,
    creature: &CreatureJson,
    max_results: Option<usize>,
    cost_of_growth: Option<f32>,
    descriptor: Option<&TaskDescriptor>,
) -> Result<RankFocusStats> {
    rank_focus_core(&RankCoreArgs {
        parquet_file,
        creature,
        max_results,
        cost_of_growth,
        descriptor,
        history: None,
        shared_deadline_ms: None,
    })
}

/// Focus-ranking entry point that bills against the shared discovery deadline
/// (Issue #1407).
///
/// Identical to [`rank_focus_neurons_with_descriptor`] but additionally accepts
/// the absolute discovery deadline (`analysis_deadline_ms`, ms-since-epoch) that
/// the subsequent synapse/neuron analysis phase also enforces. Focus selection
/// aborts at whichever is sooner: this shared deadline or the focus-ranking
/// wall-clock budget. Passing `None` reproduces the budget-only behaviour, so
/// the two phases no longer open independent time windows.
///
/// # Errors
///
/// Returns an error if the underlying record provider or impact computation
/// fails, or a [`DiscoveryError::Timeout`] if the shared deadline (or focus
/// budget) is exceeded mid-run.
pub fn rank_focus_neurons_with_descriptor_and_deadline(
    parquet_file: &str,
    creature: &CreatureJson,
    max_results: Option<usize>,
    cost_of_growth: Option<f32>,
    descriptor: Option<&TaskDescriptor>,
    analysis_deadline_ms: Option<u64>,
) -> Result<RankFocusStats> {
    rank_focus_core(&RankCoreArgs {
        parquet_file,
        creature,
        max_results,
        cost_of_growth,
        descriptor,
        history: None,
        shared_deadline_ms: analysis_deadline_ms,
    })
}

/// Inputs shared by both public focus-ranking entry points (Issue #1375).
///
/// Unifying the two near-identical functions behind one core lets the
/// wall-clock deadline be threaded through a single code path. The optional
/// `history` reproduces the history-aware sort multiplier — when `None`, the
/// ranking is byte-for-byte identical to the non-history path.
struct RankCoreArgs<'a> {
    parquet_file: &'a str,
    creature: &'a CreatureJson,
    max_results: Option<usize>,
    cost_of_growth: Option<f32>,
    descriptor: Option<&'a TaskDescriptor>,
    history: Option<&'a DiscoveryHistory>,
    /// Shared absolute discovery deadline (ms-since-epoch) billed against by
    /// both focus selection and analysis (Issue #1407). `None` keeps the
    /// budget-only behaviour.
    shared_deadline_ms: Option<u64>,
}

/// Shared focus-ranking core (Issue #1375).
///
/// Resolves the optional wall-clock deadline from configuration, chooses the
/// record loading mode, and delegates the ranking passes to
/// [`rank_selectable`]. Returns an empty result (no loading, no deadline) when
/// the creature has no selectable neurons.
fn rank_focus_core(args: &RankCoreArgs<'_>) -> Result<RankFocusStats> {
    let start = Instant::now();

    let selectable: Vec<&NeuronJson> = args
        .creature
        .neurons
        .iter()
        .filter(|neuron| is_selectable_type(&neuron.neuron_type))
        .collect();

    if selectable.is_empty() {
        return Ok(RankFocusStats {
            neurons: Vec::new(),
            removal_candidates: Vec::new(),
            constant_neuron_removals: Vec::new(),
            max_output_error: 0.0,
            processed_neurons: 0,
            total_neurons: 0,
            duration_ms: start.elapsed().as_millis(),
            rejection_breakdown: std::collections::HashMap::new(),
            loading_mode: FocusLoadingMode::Preload,
            lazy_reason: FocusLazyReason::None,
            budget_mb: focus_ranking_memory_budget_mb(),
            projected_mb: 0,
            focus_ineligible_constant: 0,
            focus_ineligible_low_impact: 0,
        });
    }

    // Issue #3172: decide the loading mode + projection *first* (cheap), so the
    // wall-clock deadline can be scaled to the chosen mode and dataset size
    // before the (possibly expensive) provider build starts spending it. A lazy
    // fallback on a large dataset thereby earns a larger budget instead of being
    // set up to abort at the fixed default.
    let plan = plan_loading(args.parquet_file, selectable.len());
    let deadline =
        FocusDeadline::resolve(start, args.shared_deadline_ms, plan.mode, plan.projected_mb);
    log_effective_budget(plan, deadline);

    let provider = build_provider(args.parquet_file, &selectable, plan, deadline)?;
    let meta = LoadingMeta {
        mode: plan.mode,
        reason: plan.reason,
        budget_mb: plan.budget_mb,
        projected_mb: plan.projected_mb,
    };

    rank_selectable(args, &selectable, provider, meta, deadline, start)
}

/// Log the resolved wall-clock budget for a run so the mode + effective
/// `budget_ms` are visible for diagnosis (Issue #3172 acceptance criterion).
///
/// `budget_enabled == false` marks a run whose budget was explicitly disabled
/// (`..._BUDGET_MS=0`) and which has no shared discovery deadline either — i.e.
/// fully unbounded.
fn log_effective_budget(plan: LoadingPlan, deadline: Option<FocusDeadline>) {
    let budget_ms = deadline.map(|d| d.budget_ms);
    tracing::info!(
        target: "neat_ai_discovery::focus::ranking",
        mode = plan.mode.as_str(),
        projected_mb = plan.projected_mb,
        budget_ms = budget_ms.unwrap_or(0),
        budget_enabled = budget_ms.is_some(),
        "focus::ranking wall-clock budget resolved",
    );
}

/// Run the ranking passes over an already-loaded record provider (Issue #1375).
///
/// Separated from provider construction so the wall-clock deadline can be
/// exercised in isolation with a deliberately slow injected provider. The
/// deadline is checked between passes and inside the per-neuron loops; on
/// exceed the run aborts with [`DiscoveryError::Timeout`].
fn rank_selectable(
    args: &RankCoreArgs<'_>,
    selectable: &[&NeuronJson],
    records_provider: Arc<dyn RecordProvider>,
    meta: LoadingMeta,
    deadline: Option<FocusDeadline>,
    start: Instant,
) -> Result<RankFocusStats> {
    let creature = args.creature;
    let total_neurons = selectable.len();

    if meta.mode == FocusLoadingMode::Lazy && verbose_enabled() {
        tracing::debug!(
            cached_neurons = records_provider.len(),
            "Lazy record cache initialised"
        );
    }

    // Verify that all selectable neurons have records (restore old error
    // behaviour). Issue #1375: check the deadline per neuron so a slow record
    // loader aborts within budget instead of grinding for an hour.
    for neuron in selectable {
        check_deadline(deadline, "verify_selectable_records")?;
        get_records_or_error(records_provider.as_ref(), &neuron.uuid)
            .context("Failed to read discovery records for all selectable neurons")?;
    }

    // Issue #1318: Under OneHot / Margin descriptors, compute per-observation
    // margin weights from output activations. These reweight per-neuron error
    // aggregation so candidates whose error mass lands on close-margin
    // observations rank above those whose error mass lands on already-dominant
    // decisions. Other topologies (Independent / Simplex / Unknown / None) fall
    // back to the unweighted mean (regression guard).
    check_deadline(deadline, "compute_margins")?;
    let obs_weights = if descriptor_activates_margin_ranking(args.descriptor) {
        let margins = compute_per_obs_margins(creature, records_provider.as_ref())?;
        if margins.is_empty() {
            None
        } else {
            Some(margin_weights_from_margins(&margins))
        }
    } else {
        None
    };

    check_deadline(deadline, "compute_max_output_error")?;
    let max_output_error =
        compute_max_output_error(creature, records_provider.as_ref(), obs_weights.as_ref())?;

    // Use activation-based impact calculation for more accurate MIN/MAX/IF statistics
    check_deadline(deadline, "compute_impacts")?;
    let impact_map = compute_impacts_with_activations(creature, records_provider.as_ref())?;

    // Issue #208: Pre-compute synapse counts to eliminate O(n×m) complexity.
    let synapse_counts = SynapseCounts::new(creature);

    // Issue #206: Build squash map for gradient flow analysis
    let squash_map = build_squash_map(creature);

    // Issue #1634: When the reconstruction-mismatch signal is enabled, compute
    // each selectable neuron's mean activation delta (recorded vs reconstructed
    // from its inbound synapses) so poorly-explained neurons rise in the focus
    // budget. Disabled by default — the map is empty and the score is unchanged.
    check_deadline(deadline, "compute_reconstruction_mismatch")?;
    let reconstruction_mismatch = if crate::config::focus_reconstruction_mismatch_enabled() {
        reconstruction::compute_reconstruction_mismatch_map(
            creature,
            selectable,
            records_provider.as_ref(),
        )?
    } else {
        std::collections::HashMap::new()
    };

    let mut neurons = build_ranked_neurons(
        selectable,
        records_provider.as_ref(),
        &RankedNeuronInputs {
            impact_map: &impact_map,
            squash_map: &squash_map,
            max_output_error,
            obs_weights: obs_weights.as_ref(),
            reconstruction_mismatch: &reconstruction_mismatch,
        },
        deadline,
    )?;

    // Sort by weighted score (error × impact × gradient_factor × frequency_factor) to prioritise neurons that:
    // 1. Have high error (potential for improvement)
    // 2. Have high impact (changes will affect output)
    // 3. Have good gradient flow (can actually learn - Issue #206)
    //
    // Dec 2025: We deliberately soften (but do not remove) the output bias by applying a
    // sub-linear exponent to impact. This increases exploration of hidden neurons without
    // letting low-impact neurons dominate purely due to noisy per-neuron errors.
    //
    // Jan 2026 (Issue #206): We further adjust ranking by gradient flow factor:
    // - Neurons stuck in saturation (high saturation_ratio) are de-prioritised
    // - Dead ReLU neurons (high dead_ratio) are de-prioritised
    // - Neurons with good gradient flow get higher priority
    //
    // Jan 2026 (Issue #204): We also adjust ranking by activation frequency factor:
    // - Rarely-firing neurons (< 10% activation rate) are de-prioritised (0.8x penalty)
    // - Always-firing neurons (> 90% activation rate) are de-prioritised (0.8x penalty)
    // - Moderate-frequency neurons (10-90%) get no penalty
    //
    // Issue #227: When discovery history is provided, the score is additionally
    // scaled by a Bayesian success multiplier (0.5 + history). With no history
    // the multiplier is absent and the ordering matches the non-history path.
    let history = args.history;

    // Issue #1445: Precompute and store each neuron's combined weighted score
    // once (error × impact^γ × gradient × frequency × history). Storing it on
    // the neuron gives the sort comparator and the downstream diversity-aware
    // focus selection a single source of truth — and avoids recomputing the
    // score on every comparison.
    //
    // History factor is in [0, 1], where 0.5 is neutral (Issue #227):
    // - 0.5 (neutral) → multiplier of 1.0 (no change)
    // - 1.0 (perfect success) → multiplier of 1.5 (50% boost)
    // - 0.0 (complete failure) → multiplier of 0.5 (50% penalty)
    // Issue #1634: resolve the additive reconstruction-mismatch weight once.
    // 0.0 when the signal is disabled, keeping the score byte-identical to the
    // pre-#1634 path.
    let recon_weight = if crate::config::focus_reconstruction_mismatch_enabled() {
        crate::config::focus_reconstruction_mismatch_weight()
    } else {
        0.0
    };
    for neuron in &mut neurons {
        let history_factor = history.map(|h| h.bayesian_score_for(&neuron.neuron_uuid) as f32);
        neuron.weighted_score = compute_focus_weighted_score(neuron, history_factor, recon_weight);
    }

    neurons.sort_by(|a, b| {
        b.weighted_score
            .total_cmp(&a.weighted_score)
            .then_with(|| b.impact.total_cmp(&a.impact))
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    // Identify removal candidates.
    //
    // Issue #1872: resolve the threshold through the #1783 validator rather than
    // taking the host value raw. A non-finite or non-positive `costOfGrowth`
    // makes `savings` NaN, and every removal gate is NaN-false, so the raw value
    // emitted *every* ranked neuron as a removal candidate. The validator warns
    // and substitutes `DEFAULT_COST_OF_GROWTH`.
    let cost_of_growth_threshold = effective_cost_of_growth(args.cost_of_growth);

    // Issue #414: High-error exploratory ablation DISABLED
    //
    // Previously, neurons with raw_error >= 10× max_output_error were returned as
    // "exploratory ablation candidates". This discovery type had a 0% success rate
    // (0 successes from 2 attempts) because the fundamental assumption was flawed:
    //
    // **High error ≠ harmful neuron**
    //
    // A neuron with high recorded error is often:
    // 1. Handling the most difficult samples (it's the only path for hard cases)
    // 2. Receiving bad inputs from upstream (the error is a symptom, not a cause)
    // 3. Fighting against incorrect biases elsewhere in the network
    //
    // Removing such neurons typically makes performance WORSE because:
    // - The difficult samples lose their only computation path
    // - The network loses the only neuron attempting to handle a specific pattern
    //
    // Error magnitude measures how WRONG the neuron's output is, not how HARMFUL
    // the neuron is to the network's overall score. This is why predicted
    // improvements (based on error magnitude) did not match actual outcomes.
    //
    // The legitimate removal candidate detection (based on activation_weighted_impact
    // < costOfGrowth) remains active and has a 17.6% success rate.
    let removal_outcome =
        identify_removal_candidates(&neurons, &synapse_counts, cost_of_growth_threshold);

    // Issue #1624: make functionally-constant hidden neurons (zero activation
    // variance) ineligible for focus slots. Their output never varies, so no
    // add-synapse / add-neuron change feeding them can move the network — every
    // focus slot they occupy is wasted and displaces a productive neuron,
    // starving successful-candidate throughput. The exclusion runs *after*
    // removal-candidate identification and does not touch the `selectable` set
    // fed to `detect_constant_neuron_removals` below, so the constant-removal
    // path (which folds these neurons into downstream biases, #306) is
    // unchanged. Opt-in via `NEAT_AI_DISCOVERY_FOCUS_EXCLUDE_CONSTANT_NEURONS`.
    let focus_ineligible_constant = if crate::config::focus_exclude_constant_neurons() {
        let constant_uuids =
            functionally_constant_focus_uuids(selectable, &records_provider, creature);
        if constant_uuids.is_empty() {
            0
        } else {
            let before = neurons.len();
            neurons.retain(|n| !constant_uuids.contains(&n.neuron_uuid));
            let removed = before - neurons.len();
            if removed > 0 {
                tracing::info!(
                    target: "neat_ai_discovery::focus::ranking",
                    focus_ineligible_constant = removed,
                    remaining_focus = neurons.len(),
                    "focus::ranking excluded functionally-constant hidden neurons from focus slots (Issue #1624)",
                );
            }
            removed
        }
    } else {
        0
    };

    // Issue #1635: gate out neurons whose structural impact magnitude is below
    // the configured threshold. This is complementary to the #1624 constant
    // filter above: production snapshot mining (#1631) found ~31.6% of neurons
    // had `|impact| < 1e-6` while still being non-constant, so the constant
    // filter leaves them in the focus pool even though no change feeding them
    // can move the output. Runs *after* removal-candidate identification and
    // does not touch the `selectable` set fed to the constant-removal path
    // below. Opt-in via `NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE`. Never silently
    // dropped — the gated count is logged and surfaced on `RankFocusStats`.
    let focus_ineligible_low_impact = if crate::config::focus_impact_gate_enabled() {
        let gate = crate::config::focus_impact_gate_threshold();
        let removed = apply_focus_impact_gate(&mut neurons, gate);
        if removed > 0 {
            tracing::info!(
                target: "neat_ai_discovery::focus::ranking",
                focus_ineligible_low_impact = removed,
                impact_gate = gate,
                remaining_focus = neurons.len(),
                "focus::ranking gated near-zero-impact neurons from focus slots (Issue #1635)",
            );
        }
        removed
    } else {
        0
    };

    if let Some(limit) = args.max_results
        && neurons.len() > limit
    {
        neurons.truncate(limit);
    }

    // Issue #306: Detect constant-value neurons and create coordinated structural candidates
    // that remove the neuron and adjust downstream biases.
    check_deadline(deadline, "detect_constant_neuron_removals")?;
    let constant_neuron_removals = detect_constant_neuron_removals(
        selectable,
        &records_provider,
        &synapse_counts,
        creature,
        cost_of_growth_threshold,
    );

    // Issue #1808: one breakdown builder, shared with the structure-only path,
    // so the two triage copies cannot report different reason sets.
    let rejection_breakdown = removal_outcome.rejection_breakdown();
    let duration_ms = start.elapsed().as_millis();
    log_focus_ranking_summary(
        meta.mode,
        meta.reason,
        meta.budget_mb,
        meta.projected_mb,
        neurons.len(),
        duration_ms,
    );

    Ok(RankFocusStats {
        neurons,
        removal_candidates: removal_outcome.candidates,
        constant_neuron_removals,
        max_output_error,
        processed_neurons: total_neurons,
        total_neurons,
        duration_ms,
        rejection_breakdown,
        loading_mode: meta.mode,
        lazy_reason: meta.reason,
        budget_mb: meta.budget_mb,
        projected_mb: meta.projected_mb,
        focus_ineligible_constant,
        focus_ineligible_low_impact,
    })
}

/// Rank focus neurons with optional historical discovery success data.
///
/// Issue #227: By tracking which neurons have historically led to successful discoveries
/// (candidates that survived ablation testing), we can prioritise them in future runs,
/// improving the discovery hit rate.
///
/// This function behaves identically to `rank_focus_neurons` when no history is provided.
/// When history is provided, the ranking score is adjusted to favour neurons with
/// higher historical success rates using a Bayesian scoring approach:
///
/// ```text
/// combined_score = base_score × history_factor
/// ```
///
/// where:
/// - `base_score` = error × impact^gamma (same as `rank_focus_neurons`)
/// - `history_factor` = `bayesian_score` from history (0.0 to 1.0)
/// - For neurons not in history, `history_factor` = 0.5 (neutral prior)
///
/// # Arguments
///
/// * `parquet_file` - Path to the parquet file containing discovery records
/// * `creature` - The creature to rank neurons for
/// * `max_results` - Optional maximum number of neurons to return
/// * `cost_of_growth` - Optional cost of growth threshold (default:
///   [`DEFAULT_COST_OF_GROWTH`])
/// * `history` - Optional discovery history for historical success data
///
/// # Returns
///
/// Returns `RankFocusStats` with neurons sorted by combined score (error × impact × history).
///
/// # Example
///
/// ```ignore
/// use neat_ai_discovery::discovery_history::DiscoveryHistory;
/// use neat_ai_discovery::focus::rank_focus_neurons_with_history;
///
/// // Create history from previous runs
/// let mut history = DiscoveryHistory::new();
/// history.record("hidden-1", true, Some(epoch));  // Success
/// history.record("hidden-2", false, None);         // Failure
///
/// // Rank neurons, prioritising those with higher historical success
/// let result = rank_focus_neurons_with_history(
///     "records.parquet",
///     &creature,
///     Some(10),      // max_results
///     None,          // cost_of_growth (use default)
///     Some(&history),
/// )?;
/// ```
pub fn rank_focus_neurons_with_history(
    parquet_file: &str,
    creature: &CreatureJson,
    max_results: Option<usize>,
    cost_of_growth: Option<f32>,
    history: Option<&DiscoveryHistory>,
) -> Result<RankFocusStats> {
    rank_focus_neurons_with_history_and_descriptor(
        parquet_file,
        creature,
        max_results,
        cost_of_growth,
        history,
        None,
    )
}

/// History-aware variant of [`rank_focus_neurons_with_descriptor`] (Issue #1318).
///
/// Combines the historical success multiplier with the margin-aware error
/// reweighting under `OneHot` / `Margin` topologies. Behaviour is identical to
/// [`rank_focus_neurons_with_history`] for other topologies and for `None`.
///
/// # Errors
///
/// Returns an error if the underlying record provider or impact computation
/// fails.
pub fn rank_focus_neurons_with_history_and_descriptor(
    parquet_file: &str,
    creature: &CreatureJson,
    max_results: Option<usize>,
    cost_of_growth: Option<f32>,
    history: Option<&DiscoveryHistory>,
    descriptor: Option<&TaskDescriptor>,
) -> Result<RankFocusStats> {
    rank_focus_core(&RankCoreArgs {
        parquet_file,
        creature,
        max_results,
        cost_of_growth,
        descriptor,
        history,
        shared_deadline_ms: None,
    })
}

/// Test-only seam for the wall-clock budget (Issue #1375).
///
/// Runs the ranking passes over an injected [`RecordProvider`] with an explicit
/// `budget_ms`, bypassing parquet loading so a deliberately slow provider can
/// exercise the deadline guard in isolation. Returns the same
/// [`DiscoveryError::Timeout`] error the production path emits when the budget
/// is exceeded.
#[cfg(test)]
pub(in crate::focus) fn rank_with_provider_for_tests(
    creature: &CreatureJson,
    provider: Arc<dyn RecordProvider>,
    budget_ms: u64,
) -> Result<RankFocusStats> {
    rank_with_provider_and_grace_for_tests(
        creature,
        provider,
        budget_ms,
        FOCUS_RANKING_BUDGET_GRACE_MS,
    )
}

/// Like [`rank_with_provider_for_tests`] but with an explicit deadline
/// `grace_ms` (Issue #1760), so the abort test can use a small grace and assert
/// the observable outcome instead of a contention-sensitive wall-clock bound.
#[cfg(test)]
pub(in crate::focus) fn rank_with_provider_and_grace_for_tests(
    creature: &CreatureJson,
    provider: Arc<dyn RecordProvider>,
    budget_ms: u64,
    grace_ms: u64,
) -> Result<RankFocusStats> {
    let start = Instant::now();
    let deadline = Some(FocusDeadline::with_grace_for_tests(
        start, budget_ms, grace_ms,
    ));
    let selectable: Vec<&NeuronJson> = creature
        .neurons
        .iter()
        .filter(|neuron| is_selectable_type(&neuron.neuron_type))
        .collect();
    let meta = LoadingMeta {
        mode: FocusLoadingMode::Lazy,
        reason: FocusLazyReason::None,
        budget_mb: None,
        projected_mb: 0,
    };
    let args = RankCoreArgs {
        parquet_file: "test.parquet",
        creature,
        max_results: None,
        cost_of_growth: None,
        descriptor: None,
        history: None,
        shared_deadline_ms: None,
    };
    rank_selectable(&args, &selectable, provider, meta, deadline, start)
}

// NOTE: Tests for focus module have been moved to tests/focus.rs
// following the testing philosophy documented in README.md
// (prefer tests/ over inline unit tests for tests that use public APIs).
//
// The tests below stay inline because `FocusDeadline` and its `resolve`
// constructor are private to this module and cannot be reached from `tests/`.

#[cfg(test)]
mod impact_gate_tests {
    //! Issue #1635: the impact-magnitude gate helpers are private to this
    //! module (they operate on the private ranking-pass state), so their unit
    //! tests live inline.
    use super::super::gradient::GradientFlowStats;
    use super::{RankedNeuron, apply_focus_impact_gate, impact_below_gate};

    /// Minimal `RankedNeuron` carrying only the `impact` the gate reads.
    fn neuron(uuid: &str, impact: f32) -> RankedNeuron {
        RankedNeuron {
            neuron_uuid: uuid.to_string(),
            total_error: 0.0,
            raw_error: 0.0,
            impact,
            mean_activation: 0.0,
            activation_weighted_impact: 0.0,
            gradient_flow: GradientFlowStats::default(),
            activation_frequency: 0.0,
            weighted_score: 0.0,
            reconstruction_mismatch: 0.0,
        }
    }

    #[test]
    fn below_gate_predicate_boundary_is_retain_on_equal() {
        let gate = 1e-6f32;
        // Strictly below the gate → gated out.
        assert!(impact_below_gate(1e-9, gate));
        assert!(impact_below_gate(0.0, gate));
        // Exactly at the gate → retained (documented retain-on-equal rule).
        assert!(!impact_below_gate(gate, gate));
        // Above the gate → retained.
        assert!(!impact_below_gate(1e-3, gate));
        // Magnitude is used, so a negative impact below the gate is still gated.
        assert!(impact_below_gate(-1e-9, gate));
        assert!(!impact_below_gate(-1e-3, gate));
        // Non-finite carries no signal → gated out.
        assert!(impact_below_gate(f32::NAN, gate));
    }

    #[test]
    fn gate_drops_low_impact_and_keeps_high_impact() {
        let mut neurons = vec![
            neuron("high", 0.5),
            neuron("low-a", 1e-9),
            neuron("boundary", 1e-6),
            neuron("low-b", 0.0),
        ];
        let removed = apply_focus_impact_gate(&mut neurons, 1e-6);
        assert_eq!(removed, 2, "both sub-gate neurons should be gated out");
        let kept: Vec<&str> = neurons.iter().map(|n| n.neuron_uuid.as_str()).collect();
        assert_eq!(kept, vec!["high", "boundary"]);
    }

    #[test]
    fn gate_preserves_order_and_reports_zero_when_all_above() {
        let mut neurons = vec![neuron("a", 0.9), neuron("b", 0.3), neuron("c", 0.1)];
        let removed = apply_focus_impact_gate(&mut neurons, 1e-6);
        assert_eq!(removed, 0);
        let kept: Vec<&str> = neurons.iter().map(|n| n.neuron_uuid.as_str()).collect();
        assert_eq!(kept, vec!["a", "b", "c"], "survivor order is preserved");
    }
}

#[cfg(test)]
mod focus_deadline_tests {
    //! Issue #1407: `FocusDeadline::resolve` must bound focus selection by the
    //! shared absolute discovery deadline, capped by the focus-ranking budget.
    use super::{FocusDeadline, FocusLoadingMode};
    use std::time::{Duration, Instant, SystemTime};

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("system clock after UNIX epoch")
            .as_millis() as u64
    }

    fn abs_diff(a: Instant, b: Instant) -> Duration {
        if a > b { a - b } else { b - a }
    }

    /// Resolve using the eager default budget (no mode/dataset scaling), the
    /// baseline these #1407 tests were written against.
    fn resolve_eager(start: Instant, shared: Option<u64>) -> Option<FocusDeadline> {
        FocusDeadline::resolve(start, shared, FocusLoadingMode::Preload, 0)
    }

    #[test]
    fn resolve_uses_shared_deadline_when_sooner_than_budget() {
        let start = Instant::now();
        // Absolute deadline 5s out — well inside the default 120s focus budget.
        let shared = Some(now_ms() + 5_000);
        let resolved = resolve_eager(start, shared).expect("deadline present");
        let budget_only = resolve_eager(start, None).expect("budget present");

        // The shared deadline (5s) bounds the run more tightly than the budget,
        // so it must win — this is the unification the issue requires.
        assert!(
            resolved.expires_at < budget_only.expires_at,
            "shared deadline should expire before the focus budget window",
        );
        // Expiry sits roughly 5s from start (tolerant of scheduling jitter).
        assert!(
            abs_diff(resolved.expires_at, start + Duration::from_millis(5_000))
                < Duration::from_millis(1_000),
            "expiry should track the shared deadline",
        );
    }

    #[test]
    fn resolve_uses_budget_when_shared_deadline_is_distant() {
        let start = Instant::now();
        // Absolute deadline far beyond the focus budget (50 minutes out).
        let shared = Some(now_ms() + 3_000_000);
        let resolved = resolve_eager(start, shared).expect("deadline present");
        let budget_only = resolve_eager(start, None).expect("budget present");

        // The nearer focus budget caps the distant deadline, so both expire at
        // effectively the same point.
        assert!(
            abs_diff(resolved.expires_at, budget_only.expires_at) < Duration::from_millis(50),
            "the focus budget should cap a distant shared deadline",
        );
    }

    #[test]
    fn resolve_with_passed_deadline_aborts_immediately() {
        let start = Instant::now();
        // An absolute deadline already in the past saturates to zero remaining.
        let shared = Some(now_ms().saturating_sub(10_000));
        let resolved = resolve_eager(start, shared).expect("deadline present");
        assert!(
            resolved.check("test").is_err(),
            "a passed shared deadline must abort focus ranking immediately",
        );
    }

    // Issue #3172: the budget-only deadline must scale with the resolved loading
    // mode + dataset size — a lazy fallback on a large dataset earns a materially
    // larger window than the fast eager path, which keeps the unscaled default.
    #[test]
    fn resolve_scales_lazy_budget_above_eager_default() {
        let start = Instant::now();
        // No shared deadline: only the (scaled) focus budget bounds the run.
        let eager = FocusDeadline::resolve(start, None, FocusLoadingMode::Preload, 0)
            .expect("eager budget present");
        let lazy = FocusDeadline::resolve(start, None, FocusLoadingMode::Lazy, 14_297)
            .expect("lazy budget present");

        assert!(
            lazy.expires_at > eager.expires_at,
            "a large lazy run must get a later deadline than the eager default \
             so it can finish instead of aborting into recorded-error aggregation",
        );
        // The eager default stays at the fixed 120s net (no regression).
        assert_eq!(
            eager.budget_ms,
            crate::config::DEFAULT_FOCUS_RANKING_BUDGET_MS,
            "eager mode must keep the unscaled default budget",
        );
        assert!(
            lazy.budget_ms > crate::config::DEFAULT_FOCUS_RANKING_BUDGET_MS,
            "lazy mode on a large dataset must exceed the fixed default budget",
        );
    }

    // Issue #3172: a bigger projected dataset earns a bigger lazy budget.
    #[test]
    fn resolve_lazy_budget_grows_with_projected_size() {
        let start = Instant::now();
        let small = FocusDeadline::resolve(start, None, FocusLoadingMode::Lazy, 100)
            .expect("small lazy budget present");
        let large = FocusDeadline::resolve(start, None, FocusLoadingMode::Lazy, 20_000)
            .expect("large lazy budget present");
        assert!(
            large.budget_ms >= small.budget_ms,
            "a larger projected dataset must not shrink the lazy budget",
        );
    }
}

/// Issue #3686: the lazy warm pass must be bounded by the run's wall-clock
/// budget. These stay inline because `build_lazy_provider` and `FocusDeadline`
/// are private to this module and cannot be reached from `tests/`.
#[cfg(test)]
mod warm_pass_deadline_tests {
    use super::{DiscoveryError, FocusDeadline, NeuronJson, build_lazy_provider};
    use crate::parquet_format::write_records_to_parquet;
    use crate::types::DiscoverRecord;
    use std::time::{Duration, Instant, SystemTime};
    use tempfile::TempDir;

    fn neuron(uuid: &str) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }
    }

    fn parquet_with_two_neurons(dir: &TempDir) -> String {
        let path = dir.path().join("records.parquet");
        let path_str = path.to_str().expect("utf-8 temp path").to_string();
        let mut records = Vec::new();
        for uuid in ["neuron-a", "neuron-b"] {
            for obs in 0..8u32 {
                records.push(DiscoverRecord::new(
                    obs,
                    uuid.to_string(),
                    Some(0.5),
                    0.25,
                    vec![0.1, 0.2],
                ));
            }
        }
        write_records_to_parquet(&path_str, &records).expect("write test parquet");
        path_str
    }

    #[test]
    fn warm_pass_seeds_the_cache_within_budget() {
        let dir = TempDir::new().expect("temp dir");
        let path = parquet_with_two_neurons(&dir);
        let a = neuron("neuron-a");
        let b = neuron("neuron-b");
        let selectable = vec![&a, &b];

        let deadline = FocusDeadline::new(Instant::now(), 600_000);
        let provider = build_lazy_provider(&path, &selectable, Some(deadline))
            .expect("an in-budget warm pass must succeed");

        assert_eq!(
            provider.len(),
            2,
            "a warm pass inside its budget must seed every selectable neuron",
        );
    }

    /// Issue #1768 (supersedes #1769's soft-fail): a warm pass that overruns the
    /// wall-clock budget must abort LOUDLY with a distinct `lazy_warm_pass`
    /// [`DiscoveryError::Timeout`], **not** silently degrade to on-demand
    /// loading. The old behaviour masked the timeout as a clean provider, so the
    /// observable abort surfaced hours later under the misleading
    /// `verify_selectable_records` context (the ~2 h field failure where a large
    /// decode on a memory-constrained host ran hours past a ~14-minute budget).
    ///
    /// Business-logic change: this replaces the previous
    /// `expired_budget_abandons_the_warm_pass` test, which asserted the soft
    /// on-demand fallback that #1768 deliberately overturns for the overrun case.
    #[test]
    fn expired_budget_aborts_the_warm_pass_loudly() {
        let dir = TempDir::new().expect("temp dir");
        let path = parquet_with_two_neurons(&dir);
        let a = neuron("neuron-a");
        let selectable = vec![&a];

        // Zero budget + zero grace anchored at "now" is already expired by the
        // time the reader checks it, so the warm pass fails on the deadline.
        let expired = FocusDeadline::with_grace_for_tests(Instant::now(), 0, 0);
        // `.err()` (not `expect_err`) because the Ok variant — an
        // `Arc<dyn RecordProvider>` — does not implement `Debug`.
        let err = build_lazy_provider(&path, &selectable, Some(expired))
            .err()
            .expect("an expired budget must abort the warm pass, not degrade silently");
        let discovery_err = err.downcast_ref::<DiscoveryError>().unwrap_or_else(|| {
            panic!("the abort must be a structured DiscoveryError, got: {err:#}")
        });
        assert!(
            matches!(discovery_err, DiscoveryError::Timeout { .. }),
            "an overrun warm pass must abort with a Timeout, got {discovery_err:?}",
        );
    }

    /// Issue #1768: a genuine read error *within* budget still degrades softly to
    /// on-demand loading — only a deadline overrun aborts loudly. A missing
    /// parquet file is a genuine (non-timeout) read failure, so the provider is
    /// still returned (seeding nothing) and the caller's per-neuron deadline
    /// check decides the run's fate.
    #[test]
    fn genuine_read_error_within_budget_degrades_softly() {
        let dir = TempDir::new().expect("temp dir");
        let missing = dir.path().join("does-not-exist.parquet");
        let missing_str = missing.to_str().expect("utf-8 temp path").to_string();
        let a = neuron("neuron-a");
        let selectable = vec![&a];

        let deadline = FocusDeadline::new(Instant::now(), 600_000);
        let provider = build_lazy_provider(&missing_str, &selectable, Some(deadline))
            .expect("a genuine in-budget read error must degrade softly, not abort");

        assert_eq!(
            provider.len(),
            0,
            "a failed warm pass seeds nothing but still returns a usable provider",
        );
    }

    /// The deadline the reader bills against must be the SAME budget the ranking
    /// passes enforce — an in-budget deadline projects to a future wall clock.
    #[test]
    fn focus_deadline_projects_onto_the_wall_clock() {
        let now = Instant::now();
        let deadline = FocusDeadline::new(now, 60_000);
        let projected = deadline.as_system_time();
        let remaining = projected
            .duration_since(SystemTime::now())
            .expect("an unexpired deadline must project into the future");
        assert!(
            remaining <= Duration::from_millis(61_000),
            "projection must not exceed the configured budget + grace",
        );
        assert!(
            remaining >= Duration::from_millis(55_000),
            "projection must preserve the configured budget",
        );
    }
}
