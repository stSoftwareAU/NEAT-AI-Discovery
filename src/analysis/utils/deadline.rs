//! Deadline handling and logging utilities for analysis module (Issue #268).
//!
//! This module contains utility functions for:
//! - Deadline/timeout calculation and handling
//! - Analysis logging (start/timeout notifications)
//! - Randomisation utilities (seed derivation, shuffling)
//! - Environment variable helpers for source ordering
//!
//! These utilities are used throughout the analysis pipeline for:
//! 1. Deadline-constrained analysis (stop when time runs out)
//! 2. Randomised ordering (avoid category starvation)
//! 3. Verbose logging (controlled by `NEAT_AI_DISCOVERY_VERBOSE`)

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::verbose_enabled;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashSet;
use std::time::{Duration, SystemTime};

// ============================================================================
// Deadline Constants
// ============================================================================

/// Default analysis timeout: 10 minutes.
pub const DEFAULT_DURATION_MS: u64 = 600_000; // 10 minutes (10 * 60 * 1000)

/// Threshold for distinguishing absolute timestamps from relative durations.
/// Values >= this are treated as absolute timestamps (milliseconds since UNIX epoch).
pub const YEAR_2000_MS: u64 = 946_684_800_000;

/// Minimum valid timeout duration: 3 seconds.
pub const MIN_DURATION_MS: u64 = 3_000;

/// Maximum valid timeout duration: 1 hour.
pub const MAX_DURATION_MS: u64 = 3_600_000; // 1 hour (60 * 60 * 1000)

/// Minimum timeout (seconds) for the GPU work queue waiting for a response from the GPU thread.
pub const GPU_QUEUE_TIMEOUT_MIN_SECS: u64 = 60;

/// Maximum timeout (seconds) for GPU batch operations.
pub const GPU_QUEUE_TIMEOUT_MAX_SECS: u64 = 300;

// ============================================================================
// Deadline Functions
// ============================================================================

/// Calculate the effective timeout duration in milliseconds.
///
/// This function applies the following logic:
/// 1. Converts absolute timestamps (values >= year 2000 in ms) to relative durations
/// 2. Clamps values outside the 3-second to 1-hour range to the 10-minute default
///
/// Returns `None` if the deadline is in the past (for absolute timestamps), or
/// `Some(effective_duration_ms)` otherwise.
///
/// Used by both `build_deadline` (to create the `SystemTime`) and `log_analysis_start`
/// (to display the effective timeout to users).
pub fn calculate_effective_timeout_ms(deadline_ms: Option<u64>) -> Option<u64> {
    let target_ms = deadline_ms.unwrap_or(DEFAULT_DURATION_MS);

    // Heuristic: if the value is less than year 2000 in milliseconds,
    // treat it as a relative duration. Otherwise, it's likely an absolute timestamp
    // from the calling code, so convert it to a relative duration.
    let relative_ms = if target_ms < YEAR_2000_MS {
        // Small value - treat as relative duration (milliseconds from now)
        target_ms
    } else {
        // Large value - likely an absolute timestamp from calling code.
        // Convert to relative duration by subtracting current time.
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()?
            .as_millis() as u64;

        // If the timestamp is in the past, return None (deadline already passed)
        if target_ms <= now_ms {
            return None;
        }

        // Calculate relative duration
        target_ms - now_ms
    };

    // Validate duration bounds: minimum 3 seconds, maximum 1 hour
    // If invalid, default to 10 minutes (expected typical value)
    // NOTE: If these warnings appear, it's a bug in the calling code (NEAT-AI or GRQ)
    // that should be fixed to pass valid timeout values.
    let validated_ms = if relative_ms < MIN_DURATION_MS {
        let duration_secs = relative_ms as f64 / 1000.0;
        tracing::warn!(
            duration_secs,
            min_secs = 3.0,
            default_secs = DEFAULT_DURATION_MS as f64 / 1000.0,
            "analysis_deadline_ms is below minimum — falling back to default 10 minute timeout. \
             The calling code (NEAT-AI/GRQ) should pass a valid timeout."
        );
        DEFAULT_DURATION_MS
    } else if relative_ms > MAX_DURATION_MS {
        let duration_secs = relative_ms as f64 / 1000.0;
        tracing::warn!(
            duration_secs,
            max_secs = MAX_DURATION_MS as f64 / 1000.0,
            default_secs = DEFAULT_DURATION_MS as f64 / 1000.0,
            "analysis_deadline_ms exceeds maximum — falling back to default 10 minute timeout. \
             The calling code (NEAT-AI/GRQ) should pass a valid timeout."
        );
        DEFAULT_DURATION_MS
    } else {
        relative_ms
    };

    Some(validated_ms)
}

/// Build a deadline from a timeout value in milliseconds.
///
/// The `deadline_ms` parameter can be either:
/// - A relative duration (milliseconds from now)
/// - An absolute timestamp (milliseconds since UNIX epoch)
///
/// If None is passed, applies default 10 minute timeout to prevent runaway analysis.
pub fn build_deadline(deadline_ms: Option<u64>) -> Option<SystemTime> {
    calculate_effective_timeout_ms(deadline_ms)
        .and_then(|validated_ms| SystemTime::now().checked_add(Duration::from_millis(validated_ms)))
}

/// Check if the deadline has passed.
///
/// Returns `true` if the current time is past the deadline.
/// Returns `false` if there is no deadline or the deadline is in the future.
///
/// In test mode, this function first checks for deadline override values
/// to allow deterministic testing of deadline-related behaviour.
pub fn deadline_passed(deadline: &Option<SystemTime>) -> bool {
    #[cfg(test)]
    {
        if let Some(value) = deadline_override::next_override_value() {
            return value;
        }
    }

    matches!(deadline, Some(limit) if SystemTime::now() >= *limit)
}

/// Calculate adaptive GPU batch timeout based on remaining deadline.
///
/// For large datasets (>1GB Parquet files), GPU batch evaluations may take
/// longer than the minimum 60 seconds. This function calculates a reasonable
/// timeout based on:
/// - Minimum: `GPU_QUEUE_TIMEOUT_MIN_SECS` (60s) - catches unresponsive GPU
/// - Maximum: `GPU_QUEUE_TIMEOUT_MAX_SECS` (5 min) - prevents infinite waits
/// - If deadline is available: uses up to half remaining time (capped at max)
pub fn calculate_gpu_batch_timeout(deadline: &Option<SystemTime>) -> Duration {
    let min_timeout = Duration::from_secs(GPU_QUEUE_TIMEOUT_MIN_SECS);
    let max_timeout = Duration::from_secs(GPU_QUEUE_TIMEOUT_MAX_SECS);

    match deadline {
        Some(dl) => {
            if let Ok(remaining) = dl.duration_since(SystemTime::now()) {
                // Use half of remaining time, but capped between min and max
                let half_remaining = remaining / 2;
                if half_remaining < min_timeout {
                    min_timeout
                } else if half_remaining > max_timeout {
                    max_timeout
                } else {
                    half_remaining
                }
            } else {
                // Deadline already passed - use minimum
                min_timeout
            }
        }
        // No deadline - use maximum
        None => max_timeout,
    }
}

// ============================================================================
// Logging Functions
// ============================================================================

/// Log analysis start information including deadline and focus neuron count.
/// This provides visibility into timeout configuration without requiring verbose mode.
pub fn log_analysis_start(
    analysis_type: &str,
    deadline_ms: Option<u64>,
    focus_count: usize,
    shuffled_order: &[String],
) {
    // Calculate the effective deadline duration using the same logic as build_deadline.
    // This ensures the logged timeout matches what's actually used.
    let deadline_duration_ms =
        calculate_effective_timeout_ms(deadline_ms).unwrap_or(DEFAULT_DURATION_MS);
    let deadline_secs = deadline_duration_ms as f64 / 1000.0;

    // Debug: Log the raw deadline_ms value if verbose to help diagnose timeout issues
    if verbose_enabled()
        && let Some(raw_ms) = deadline_ms
    {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if raw_ms >= YEAR_2000_MS {
            // Absolute timestamp
            let elapsed_secs =
                now_ms.saturating_sub(raw_ms.saturating_sub(deadline_duration_ms)) as f64 / 1000.0;
            tracing::debug!(
                analysis_type,
                deadline_ms = raw_ms,
                kind = "absolute_timestamp",
                elapsed_secs,
                remaining_secs = deadline_secs,
                "Deadline is an absolute timestamp"
            );
        } else {
            // Relative duration
            tracing::debug!(
                analysis_type,
                deadline_ms = raw_ms,
                kind = "relative_duration",
                "Deadline is a relative duration"
            );
        }
    }

    // Warn if timeout is very short - likely means focus selection took most of the allotted time
    const MIN_USEFUL_TIMEOUT_SECS: f64 = 60.0; // 1 minute minimum for useful analysis
    if deadline_secs < MIN_USEFUL_TIMEOUT_SECS {
        tracing::warn!(
            remaining_secs = deadline_secs,
            analysis_type,
            "Only {deadline_secs:.1}s remaining for analysis. \
             Focus selection may have consumed most of the timeout. \
             Consider increasing discoveryAnalysisTimeoutMinutes."
        );
    }

    // Format the timeout nicely
    let timeout_str = if deadline_secs >= 60.0 {
        let minutes = deadline_secs / 60.0;
        format!("{minutes:.1} minutes")
    } else {
        format!("{deadline_secs:.1} seconds")
    };

    tracing::info!(
        analysis_type,
        focus_count,
        timeout = %timeout_str,
        "Starting analysis"
    );

    // Log the shuffled order if verbose mode is enabled
    if verbose_enabled() && !shuffled_order.is_empty() {
        let preview: Vec<&str> = shuffled_order
            .iter()
            .take(5)
            .map(std::string::String::as_str)
            .collect();
        let extra = shuffled_order.len().saturating_sub(5);
        tracing::debug!(
            preview = ?preview,
            remaining = extra,
            "Randomised focus order"
        );
    }
}

/// Log when analysis timeout is reached. Always prints (not verbose-only).
pub fn log_analysis_timeout(analysis_type: &str, completed_count: usize, total_count: usize) {
    tracing::info!(
        analysis_type,
        completed_count,
        total_count,
        "Analysis reached timeout. Returning partial results."
    );
}

// ============================================================================
// Randomisation Utilities
// ============================================================================

/// Parse input neuron index from a UUID string.
///
/// Input neurons have UUIDs in the form "input-N" where N is the index.
/// Returns `Some(index)` if the UUID matches this pattern, `None` otherwise.
pub fn parse_input_index(uuid: &str) -> Option<usize> {
    uuid.strip_prefix("input-")?.parse::<usize>().ok()
}

/// Derive a stable per-context seed from a base seed.
///
/// Uses FNV-1a 64-bit hash to combine the base seed with a context string.
/// We deliberately avoid `Hash` here because Rust's hasher is intentionally
/// randomised between processes.
pub fn derive_seed(base_seed: u64, context: &str, salt: u64) -> u64 {
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325 ^ base_seed ^ salt;
    for b in context.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Shuffle a slice, optionally deterministically.
///
/// When `seed` is `None`, uses non-deterministic randomness (best for production
/// timeout runs where we want coverage to drift over time).
pub fn shuffle_slice<T>(items: &mut [T], seed: Option<u64>, context: &str) {
    if items.len() <= 1 {
        return;
    }

    match seed {
        Some(base) => {
            let derived = derive_seed(base, context, 0x9e3779b97f4a7c15);
            let mut rng = StdRng::seed_from_u64(derived);
            items.shuffle(&mut rng);
        }
        None => {
            let mut rng = rand::thread_rng();
            items.shuffle(&mut rng);
        }
    }
}

/// Diversify a best-first candidate list by shuffling within the top-K prefix.
///
/// Rationale (Jan 2026):
/// - In production, discovery runs are deadline-constrained and repeated over time.
/// - The controller (TypeScript) caches failed candidates and will not re-attempt them
///   until the training data changes.
/// - If Rust always returns the same top few candidates, categories can starve because
///   those candidates become permanently skipped for the day.
///
/// Shuffling within the top-K keeps us focused on high-quality candidates while still
/// drifting over time, improving long-run coverage without returning low-quality tails.
pub fn shuffle_within_top_k<T>(items: &mut [T], seed: Option<u64>, context: &str, top_k: usize) {
    if items.len() <= 1 || top_k <= 1 {
        return;
    }
    let k = top_k.min(items.len());
    shuffle_slice(&mut items[..k], seed, context);
}

// ============================================================================
// Environment Variable Helpers
// ============================================================================

/// Optional input-index bias for source ordering.
///
/// This is primarily a production knob for timeout-constrained runs. If you are
/// frequently appending new inputs (e.g., feature engineering), you may want to
/// bias discovery toward higher input indices so newer inputs are evaluated
/// earlier, while still allowing older inputs to be discovered over time.
///
/// Controlled via `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS`:
/// - unset / empty: disabled (pure shuffle)
/// - `> 0`: enabled (weighted random permutation; higher = stronger bias to the end)
pub fn source_input_index_bias_from_env() -> Option<f64> {
    crate::config::source_input_index_bias()
}

/// Check if discovery should focus on unused observations (Issue #182).
///
/// Delegates to [`crate::config::focus_unused_observations()`].
pub fn focus_unused_observations_from_env() -> bool {
    crate::config::focus_unused_observations()
}

/// Neuron with UUID and index for source ordering.
///
/// Used by `order_eligible_sources` to track neuron identity during sorting.
pub struct OrderedNeuron {
    /// The unique identifier for this neuron.
    pub uuid: String,
    /// The topological index of this neuron in the network.
    pub index: usize,
}

/// Order eligible sources for evaluation.
///
/// - Always respects forward-only candidate constraints (caller must filter by index).
/// - **Issue #467**: Input neurons are always evaluated before hidden neurons. Within
///   each group, neurons are randomly shuffled. This ensures that under deadline
///   constraints, input-neuron sources (36.2% success rate) are tried before hidden
///   neurons (2.8–3.3% success rate).
/// - If `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS` is set, input sources are further
///   ordered by a weighted random permutation favouring higher input indices.
/// - If `NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS=1` is set, input neurons with NO
///   existing outgoing synapses are prioritised (moved to the front) before any other
///   ordering is applied (Issue #182).
pub fn order_eligible_sources<S: std::borrow::Borrow<str> + std::hash::Hash + Eq>(
    eligible_sources: &mut Vec<&OrderedNeuron>,
    seed: Option<u64>,
    context: &str,
    creature_input_count: usize,
    used_inputs: Option<&HashSet<S>>,
) {
    if eligible_sources.len() <= 1 {
        return;
    }

    // Issue #182: When NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS=1 is set,
    // prioritise input neurons that have NO existing outgoing synapses.
    // These "unused observations" are moved to the front of the list.
    if focus_unused_observations_from_env()
        && let Some(used) = used_inputs
    {
        // Partition: unused inputs first, then used inputs, then non-inputs
        let (mut unused_inputs, mut others): (Vec<_>, Vec<_>) = eligible_sources
            .drain(..)
            .partition(|n| parse_input_index(&n.uuid).is_some() && !used.contains(n.uuid.as_str()));

        // Issue #467: Within "others", still put used inputs before hidden neurons
        let (mut used_inputs_vec, mut non_inputs): (Vec<_>, Vec<_>) = others
            .drain(..)
            .partition(|n| parse_input_index(&n.uuid).is_some());

        // Log when focusing on unused observations
        if verbose_enabled() && !unused_inputs.is_empty() {
            tracing::debug!(
                unused_input_count = unused_inputs.len(),
                used_input_count = used_inputs_vec.len(),
                other_source_count = non_inputs.len(),
                "Focus unused observations: prioritising unused inputs"
            );
        }

        // Shuffle each partition separately, then concatenate
        shuffle_slice(&mut unused_inputs, seed, &format!("{context}:unused"));
        shuffle_slice(
            &mut used_inputs_vec,
            seed,
            &format!("{context}:used_inputs"),
        );
        shuffle_slice(&mut non_inputs, seed, &format!("{context}:non_inputs"));

        eligible_sources.extend(unused_inputs);
        eligible_sources.extend(used_inputs_vec);
        eligible_sources.extend(non_inputs);
        return;
    }

    // Issue #467: Partition into input neurons and non-input neurons.
    // Input neurons go first so they are evaluated before hidden neurons
    // under deadline constraints.
    let (mut inputs, mut non_inputs): (Vec<_>, Vec<_>) = eligible_sources
        .drain(..)
        .partition(|n| parse_input_index(&n.uuid).is_some());

    let Some(bias) = source_input_index_bias_from_env() else {
        // No index bias: shuffle each partition independently, inputs first
        shuffle_slice(&mut inputs, seed, &format!("{context}:inputs"));
        shuffle_slice(&mut non_inputs, seed, &format!("{context}:non_inputs"));
        eligible_sources.extend(inputs);
        eligible_sources.extend(non_inputs);
        return;
    };

    // Weighted permutation via exponential race (for input neurons only):
    // t = -ln(U) / w, smaller t comes first.
    // Issue #467: Apply weighted permutation to input neurons, then append non-inputs.
    let max_input_index = creature_input_count.saturating_sub(1) as f64;
    let denom = (max_input_index + 1.0).max(1.0);

    let mut rng = match seed {
        Some(base) => {
            let derived = derive_seed(base, context, 0x243f6a8885a308d3);
            StdRng::seed_from_u64(derived)
        }
        None => {
            // Use a random seed then a deterministic RNG instance so we can reuse the same
            // code path without fighting trait object ergonomics.
            let seed: u64 = rand::thread_rng().r#gen();
            StdRng::seed_from_u64(seed)
        }
    };

    let mut keyed: Vec<(f64, &OrderedNeuron)> = Vec::with_capacity(inputs.len());
    for &n in &inputs {
        let weight = if let Some(i) = parse_input_index(&n.uuid) {
            // Normalise to (0, 1] based on input index, then apply power bias.
            // Epsilon keeps weight > 0 even for i=0 with high bias.
            let x = ((i as f64) + 1.0) / denom;
            (1e-6 + x).powf(bias)
        } else {
            1.0
        };

        let u: f64 = rng.r#gen::<f64>().max(f64::MIN_POSITIVE);
        let t = -u.ln() / weight.max(1e-12);
        keyed.push((t, n));
    }

    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    eligible_sources.extend(keyed.into_iter().map(|(_, n)| n));

    // Append non-input neurons after all input neurons
    shuffle_slice(&mut non_inputs, seed, &format!("{context}:non_inputs"));
    eligible_sources.extend(non_inputs);
}

// =============================================================================
// Target-Type Prioritisation (Issue #468)
// =============================================================================

/// Orders focus targets so that existing hidden neurons are evaluated before
/// output neurons during deadline-constrained analysis.
///
/// GRQ-sampler data shows existing hidden neurons as targets have a 31.4%
/// success rate compared to 5.3–5.4% for output neurons. Under deadline
/// pressure, evaluating hidden targets first maximises the chance of finding
/// successful candidates before time runs out.
///
/// Each partition (hidden, non-hidden) is shuffled independently so that
/// repeated runs still explore different neurons within each group.
pub fn order_focus_targets(
    targets: &mut Vec<String>,
    seed: Option<u64>,
    neuron_type_map: &std::collections::HashMap<&str, &str>,
) {
    if targets.len() <= 1 {
        return;
    }

    // Partition into existing hidden neurons and everything else
    let (mut hidden, mut others): (Vec<_>, Vec<_>) = targets.drain(..).partition(|uuid| {
        neuron_type_map
            .get(uuid.as_str())
            .is_some_and(|t| *t == "hidden")
    });

    // Shuffle each partition independently
    shuffle_slice(&mut hidden, seed, "focus_targets:hidden");
    shuffle_slice(&mut others, seed, "focus_targets:others");

    // Hidden neurons first, then everything else
    targets.extend(hidden);
    targets.extend(others);
}

// ============================================================================
// Test Override Mechanism
// ============================================================================

#[cfg(test)]
pub mod deadline_override {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    /// Deadline override state shared across a thread pool.
    ///
    /// We store this behind an `Arc` so a single override sequence can be consumed
    /// by multiple worker threads within a *private* Rayon pool.
    #[derive(Clone)]
    struct OverrideState {
        queue: Arc<Mutex<VecDeque<bool>>>,
    }

    thread_local! {
        /// Thread-local handle to the active override state.
        ///
        /// IMPORTANT (2 Jan 2026):
        /// Tests run in parallel and analysis uses Rayon. A global override is unsafe because
        /// *other tests* running concurrently can observe and consume override values.
        ///
        /// By using thread-local state, and explicitly broadcasting it into a test's private
        /// Rayon pool, we:
        /// - Avoid deadlocks (no cross-thread lock held for the duration of a test)
        /// - Avoid cross-test interference under default parallel test execution
        static OVERRIDE_STATE: RefCell<Option<OverrideState>> = const { RefCell::new(None) };
    }

    fn set_override_state(state: Option<OverrideState>) {
        OVERRIDE_STATE.with(|cell| {
            *cell.borrow_mut() = state;
        });
    }

    pub struct DeadlineOverrideGuard {
        /// When provided, we clear the override from all worker threads on drop.
        pool: Option<Arc<rayon::ThreadPool>>,
    }

    impl DeadlineOverrideGuard {
        /// Install an override sequence for the current thread only.
        pub fn with_sequence(sequence: Vec<bool>) -> Self {
            Self::with_sequence_for_pool(sequence, None)
        }

        /// Install an override sequence for a specific Rayon pool.
        ///
        /// This is the safe way to use deadline overrides in multi-threaded tests: create a private
        /// pool, broadcast the override into that pool, and run analysis inside `pool.install(...)`.
        pub fn with_sequence_for_pool(
            sequence: Vec<bool>,
            pool: Option<Arc<rayon::ThreadPool>>,
        ) -> Self {
            let state = OverrideState {
                queue: Arc::new(Mutex::new(sequence.into_iter().collect::<VecDeque<_>>())),
            };

            // Always install for the current thread (covers non-parallel code paths).
            set_override_state(Some(state.clone()));

            // If a pool is provided, broadcast into each worker thread.
            if let Some(ref p) = pool {
                p.broadcast(|_| set_override_state(Some(state.clone())));
            }

            Self { pool }
        }
    }

    impl Drop for DeadlineOverrideGuard {
        fn drop(&mut self) {
            // Clear on current thread.
            set_override_state(None);
            // Clear on pool worker threads (if any).
            if let Some(ref p) = self.pool {
                p.broadcast(|_| set_override_state(None));
            }
        }
    }

    pub fn next_override_value() -> Option<bool> {
        OVERRIDE_STATE.with(|cell| {
            let state = cell.borrow().clone()?;
            let mut queue = state
                .queue
                .lock()
                .expect("Deadline override queue should not be poisoned");
            queue.pop_front()
        })
    }
}

#[cfg(test)]
#[path = "deadline_tests.rs"]
mod tests;
