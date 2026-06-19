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

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
pub(super) mod record_providers;
mod removal_candidates;
mod score_calculation;

// Re-export public API — all items remain accessible via `crate::focus::ranking::*`
pub use record_providers::RecordProvider;
pub use removal_candidates::{RemovalCandidate, SynapseCounts, calculate_removal_savings};
pub use score_calculation::{RankedNeuron, SelectionStats};

// Issue #1172 — `decide_records_loading` budget plumbing.
// Items defined in this module that are part of the public API.

// Re-export internal types needed by focus/tests.rs (unit tests for record providers)
pub(super) use record_providers::LazyRecordProvider;

use record_providers::{EagerRecordProvider, get_records_or_error};
use removal_candidates::{detect_constant_neuron_removals, identify_removal_candidates};
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
    FOCUS_RANKING_BUDGET_GRACE_MS, focus_ranking_budget_ms, focus_ranking_memory_budget_mb,
    focus_ranking_memory_margin_mb, focus_ranking_perf_cliff_ms,
};
use crate::discovery_history::DiscoveryHistory;
use crate::ffi_types::DiscoveryError;
use crate::parquet_format::read_all_records_grouped_by_neuron;
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
    ///
    /// Returns `None` only when both bounds are absent — no shared deadline AND
    /// the focus budget explicitly disabled with `0`.
    fn resolve(start: Instant, shared_deadline_ms: Option<u64>) -> Option<Self> {
        let budget_deadline =
            focus_ranking_budget_ms().map(|budget_ms| Self::new(start, budget_ms));
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

const DEFAULT_COST_OF_GROWTH: f32 = 1e-7;
const IMPACT_EPSILON: f32 = 0.0001;
const IMPACT_GAMMA: f32 = 0.8;

/// Outcome of choosing eager vs lazy loading for a focus-ranking run
/// (Issue #1172).
struct LoadingDecision {
    provider: Arc<dyn RecordProvider>,
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
/// hosts with plenty of free memory (the GRQ-13 regression: ~1.6 GB projection
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

/// Load records provider, taking the optional configurable memory budget into
/// account.
///
/// Behaviour (Issue #1172):
/// - When `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB` is set, the
///   projected in-memory size (file size × 3) is compared against the budget.
///   Lazy mode is selected with a structured `info` log when the projection
///   exceeds the budget.
/// - When the budget is unset, the auto-detect path compares the projection
///   against real OS-available memory minus a configurable safety margin
///   (Issue #1376). Pre-load is chosen whenever the projection fits, so hosts
///   with GBs free stay on the fast path; lazy mode (with a `WARN`) is reserved
///   for genuinely memory-constrained hosts.
fn load_records_provider(
    parquet_file: &str,
    selectable: &[&NeuronJson],
) -> Result<LoadingDecision> {
    let budget_mb = focus_ranking_memory_budget_mb();
    let projected_bytes = estimate_parquet_in_memory_bytes(parquet_file);
    let projected_mb = bytes_to_mb_ceil(projected_bytes);

    if let Some(budget) = budget_mb {
        return decide_with_budget(
            parquet_file,
            selectable,
            budget,
            projected_bytes,
            projected_mb,
        );
    }

    decide_with_auto_detect(parquet_file, selectable, projected_bytes, projected_mb)
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
fn build_lazy_provider(parquet_file: &str, selectable: &[&NeuronJson]) -> Arc<dyn RecordProvider> {
    let provider = LazyRecordProvider::with_capacity(parquet_file, selectable.len());

    match read_all_records_grouped_by_neuron(parquet_file) {
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
            tracing::warn!(
                error = %read_err,
                "Lazy focus-ranking cache warm pass failed; falling back to on-demand per-neuron loading",
            );
        }
    }

    Arc::new(provider)
}

fn decide_with_budget(
    parquet_file: &str,
    selectable: &[&NeuronJson],
    budget_mb: u64,
    projected_bytes: u64,
    projected_mb: u64,
) -> Result<LoadingDecision> {
    let (mode, reason) = decide_loading_mode_for_budget(projected_bytes, budget_mb);
    match mode {
        FocusLoadingMode::Lazy => {
            // Issue #1377: escalate to WARN and record available memory
            // alongside projected/budget so the eager-vs-lazy trade-off is
            // visible at the decision point, not split across log lines.
            let (available_bytes, _total_bytes) = get_memory_info();
            tracing::warn!(
                target: "neat_ai_discovery::focus::ranking",
                mode = FocusLoadingMode::Lazy.as_str(),
                reason = reason.as_str(),
                budget_mb,
                projected_mb,
                available_mb = bytes_to_mb_ceil(available_bytes),
                "focus::ranking selected lazy mode: projected pre-load exceeds configured budget",
            );
            Ok(LoadingDecision {
                provider: build_lazy_provider(parquet_file, selectable),
                mode,
                reason,
                budget_mb: Some(budget_mb),
                projected_mb,
            })
        }
        FocusLoadingMode::Preload => {
            // Issue #1406: decode through the process-shared cache so the
            // following analysis phase can reuse this load instead of scanning
            // the same parquet file a second time.
            let shared = load_grouped_records_shared(parquet_file, None)
                .context("Failed to read discovery records from parquet file")?;
            Ok(LoadingDecision {
                provider: Arc::new(EagerRecordProvider::from_shared(&shared)),
                mode,
                reason,
                budget_mb: Some(budget_mb),
                projected_mb,
            })
        }
    }
}

fn decide_with_auto_detect(
    parquet_file: &str,
    selectable: &[&NeuronJson],
    projected_bytes: u64,
    projected_mb: u64,
) -> Result<LoadingDecision> {
    const BYTES_PER_MB: u64 = 1024 * 1024;

    // Issue #1376: base the decision on real OS-available memory minus a safety
    // margin, rather than the 50%-of-total-RAM cap that dropped mid-sized
    // parquet files onto the slow lazy path while GBs of RAM were free.
    let (available_bytes, _total_bytes) = get_memory_info();
    let margin_mb = focus_ranking_memory_margin_mb();
    let margin_bytes = margin_mb.saturating_mul(BYTES_PER_MB);
    let available_mb = bytes_to_mb_ceil(available_bytes);

    let (mode, reason) =
        decide_loading_mode_for_available_memory(projected_bytes, available_bytes, margin_bytes);

    match mode {
        FocusLoadingMode::Preload => {
            // Issue #1406: decode through the process-shared cache (see the
            // budget-path branch) so the analysis phase reuses this load.
            let shared = load_grouped_records_shared(parquet_file, None)
                .context("Failed to read discovery records from parquet file")?;
            Ok(LoadingDecision {
                provider: Arc::new(EagerRecordProvider::from_shared(&shared)),
                mode,
                reason,
                budget_mb: None,
                projected_mb,
            })
        }
        FocusLoadingMode::Lazy => {
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
                "Insufficient available memory for full pre-load in focus ranking. \
                 Using lazy-loading mode (slower but memory-efficient).",
            );
            Ok(LoadingDecision {
                provider: build_lazy_provider(parquet_file, selectable),
                mode,
                reason,
                budget_mb: None,
                projected_mb,
            })
        }
    }
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

/// Build ranked neurons from selectable neurons with their metrics.
///
/// When `obs_weights` is provided (`OneHot` / `Margin` descriptors), per-neuron
/// errors are aggregated using the per-observation margin weight rather than
/// the unweighted mean. Issue #1318.
fn build_ranked_neurons(
    selectable: &[&NeuronJson],
    records_provider: &dyn RecordProvider,
    impact_map: &std::collections::HashMap<String, f32>,
    squash_map: &std::collections::HashMap<String, String>,
    max_output_error: f32,
    obs_weights: Option<&std::collections::HashMap<u32, f32>>,
    deadline: Option<FocusDeadline>,
) -> Result<Vec<RankedNeuron>> {
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
            let structural_impact = *impact_map.get(&neuron.uuid).unwrap_or(&0.0);
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
                compute_gradient_flow_for_neuron(&neuron.uuid, squash_map, &records);

            // Issue #204: Compute activation frequency for focus neuron ranking
            let activation_frequency = activation_frequency_from_records(&records);

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
#[must_use]
pub(super) fn compute_focus_weighted_score(
    neuron: &RankedNeuron,
    history_factor: Option<f32>,
) -> f32 {
    let base = neuron.total_error * (neuron.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);
    let gradient_factor = compute_gradient_flow_factor(&neuron.gradient_flow);
    let frequency_factor = compute_frequency_factor(neuron.activation_frequency);
    let with_gradient = base * gradient_factor * frequency_factor;
    match history_factor {
        Some(h) => with_gradient * (0.5 + h),
        None => with_gradient,
    }
}

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
    let deadline = FocusDeadline::resolve(start, args.shared_deadline_ms);

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
        });
    }

    let LoadingDecision {
        provider,
        mode,
        reason,
        budget_mb,
        projected_mb,
    } = load_records_provider(args.parquet_file, &selectable)?;
    let meta = LoadingMeta {
        mode,
        reason,
        budget_mb,
        projected_mb,
    };

    rank_selectable(args, &selectable, provider, meta, deadline, start)
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

    let mut neurons = build_ranked_neurons(
        selectable,
        records_provider.as_ref(),
        &impact_map,
        &squash_map,
        max_output_error,
        obs_weights.as_ref(),
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
    for neuron in &mut neurons {
        let history_factor = history.map(|h| h.bayesian_score_for(&neuron.neuron_uuid) as f32);
        neuron.weighted_score = compute_focus_weighted_score(neuron, history_factor);
    }

    neurons.sort_by(|a, b| {
        b.weighted_score
            .total_cmp(&a.weighted_score)
            .then_with(|| b.impact.total_cmp(&a.impact))
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    // Identify removal candidates
    let cost_of_growth_threshold = args.cost_of_growth.unwrap_or(DEFAULT_COST_OF_GROWTH);

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

    let rejection_breakdown = build_rejection_breakdown(&removal_outcome);
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
    })
}

/// Build a stable-keyed rejection breakdown from a [`RemovalCandidateOutcome`]
/// (Issue #1142).
///
/// Reuses the Issue #1129 rejection-reason vocabulary so downstream tooling
/// (FFI consumers, observability dashboards) can merge these counts into the
/// existing `metadata.rejection_breakdown` map without any special-casing.
fn build_rejection_breakdown(
    outcome: &removal_candidates::RemovalCandidateOutcome,
) -> std::collections::HashMap<String, u32> {
    use crate::analysis::diagnostics::rejection_reasons::REJECTION_REMOVAL_BELOW_NOISE_FLOOR;
    let mut map = std::collections::HashMap::new();
    if outcome.noise_floor_rejections > 0 {
        map.insert(
            REJECTION_REMOVAL_BELOW_NOISE_FLOOR.to_string(),
            outcome.noise_floor_rejections,
        );
    }
    map
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
/// * `cost_of_growth` - Optional cost of growth threshold (default: 1e-7)
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
    let start = Instant::now();
    let deadline = Some(FocusDeadline::new(start, budget_ms));
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
mod focus_deadline_tests {
    //! Issue #1407: `FocusDeadline::resolve` must bound focus selection by the
    //! shared absolute discovery deadline, capped by the focus-ranking budget.
    use super::FocusDeadline;
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

    #[test]
    fn resolve_uses_shared_deadline_when_sooner_than_budget() {
        let start = Instant::now();
        // Absolute deadline 5s out — well inside the default 120s focus budget.
        let shared = Some(now_ms() + 5_000);
        let resolved = FocusDeadline::resolve(start, shared).expect("deadline present");
        let budget_only = FocusDeadline::resolve(start, None).expect("budget present");

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
        let resolved = FocusDeadline::resolve(start, shared).expect("deadline present");
        let budget_only = FocusDeadline::resolve(start, None).expect("budget present");

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
        let resolved = FocusDeadline::resolve(start, shared).expect("deadline present");
        assert!(
            resolved.check("test").is_err(),
            "a passed shared deadline must abort focus ranking immediately",
        );
    }
}
