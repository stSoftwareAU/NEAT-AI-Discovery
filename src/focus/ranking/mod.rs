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

// Re-export internal types needed by focus/tests.rs (unit tests for record providers)
pub(super) use record_providers::LazyRecordProvider;

use record_providers::{EagerRecordProvider, get_records_or_error};
use removal_candidates::{detect_constant_neuron_removals, identify_removal_candidates};
use score_calculation::{
    activation_frequency_from_records, average_absolute_error_from_records,
    compute_frequency_factor, mean_absolute_activation_from_records,
};

use super::gradient::{
    build_squash_map, compute_gradient_flow_factor, compute_gradient_flow_for_neuron,
};
use super::impact::compute_impacts_with_activations;
use crate::analysis::utils::{check_memory_for_parquet, verbose_enabled};
use crate::discovery_history::DiscoveryHistory;
use crate::parquet_format::read_all_records_grouped_by_neuron;
use crate::{CoordinatedStructuralCandidateJson, CreatureJson, NeuronJson};
use anyhow::{Context, Result};
use rayon::prelude::*;

use std::sync::Arc;
use std::time::Instant;

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
}

pub(super) fn is_selectable_type(neuron_type: &str) -> bool {
    neuron_type != "input" && neuron_type != "constant"
}

const DEFAULT_COST_OF_GROWTH: f32 = 1e-7;
const IMPACT_EPSILON: f32 = 0.0001;
const IMPACT_GAMMA: f32 = 0.8;

/// Load records provider (eager or lazy depending on available memory).
fn load_records_provider(parquet_file: &str) -> Result<(Arc<dyn RecordProvider>, bool)> {
    match check_memory_for_parquet(parquet_file) {
        Ok(()) => {
            let records = read_all_records_grouped_by_neuron(parquet_file)
                .context("Failed to read discovery records from parquet file")?;
            Ok((Arc::new(EagerRecordProvider::new(records)), false))
        }
        Err(memory_error) => {
            tracing::warn!(
                "Insufficient memory for full pre-load in focus ranking. \
                 Using lazy-loading mode (slower but memory-efficient)."
            );
            if verbose_enabled() {
                tracing::debug!(error = %memory_error, "Memory check failed");
            }
            Ok((Arc::new(LazyRecordProvider::new(parquet_file)), true))
        }
    }
}

/// Compute max output error across all output neurons.
fn compute_max_output_error(
    creature: &CreatureJson,
    records_provider: &dyn RecordProvider,
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
            Ok(average_absolute_error_from_records(&records))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(errors.into_iter().fold(0.0, f32::max))
}

/// Build ranked neurons from selectable neurons with their metrics.
fn build_ranked_neurons(
    selectable: &[&NeuronJson],
    records_provider: &dyn RecordProvider,
    impact_map: &std::collections::HashMap<String, f32>,
    squash_map: &std::collections::HashMap<String, String>,
    max_output_error: f32,
) -> Result<Vec<RankedNeuron>> {
    selectable
        .par_iter()
        .map(|neuron| -> Result<RankedNeuron> {
            let records = get_records_or_error(records_provider, &neuron.uuid)?;
            let raw_error = average_absolute_error_from_records(&records);
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
            })
        })
        .collect::<Result<Vec<_>>>()
}

pub fn rank_focus_neurons(
    parquet_file: &str,
    creature: &CreatureJson,
    max_results: Option<usize>,
    cost_of_growth: Option<f32>,
) -> Result<RankFocusStats> {
    let start = Instant::now();
    let selectable: Vec<&NeuronJson> = creature
        .neurons
        .iter()
        .filter(|neuron| is_selectable_type(&neuron.neuron_type))
        .collect();
    let total_neurons = selectable.len();

    if total_neurons == 0 {
        return Ok(RankFocusStats {
            neurons: Vec::new(),
            removal_candidates: Vec::new(),
            constant_neuron_removals: Vec::new(),
            max_output_error: 0.0,
            processed_neurons: 0,
            total_neurons: 0,
            duration_ms: start.elapsed().as_millis(),
            rejection_breakdown: std::collections::HashMap::new(),
        });
    }

    let (records_provider, is_lazy_mode) = load_records_provider(parquet_file)?;

    if is_lazy_mode && verbose_enabled() {
        tracing::debug!(
            cached_neurons = records_provider.len(),
            "Lazy record cache initialised"
        );
    }

    // Verify that all selectable neurons have records (restore old error behaviour)
    for neuron in &selectable {
        get_records_or_error(records_provider.as_ref(), &neuron.uuid)
            .context("Failed to read discovery records for all selectable neurons")?;
    }

    let max_output_error = compute_max_output_error(creature, records_provider.as_ref())?;

    // Use activation-based impact calculation for more accurate MIN/MAX/IF statistics
    let impact_map = compute_impacts_with_activations(creature, records_provider.as_ref())?;

    // Issue #208: Pre-compute synapse counts to eliminate O(n×m) complexity.
    let synapse_counts = SynapseCounts::new(creature);

    // Issue #206: Build squash map for gradient flow analysis
    let squash_map = build_squash_map(creature);

    let mut neurons = build_ranked_neurons(
        &selectable,
        records_provider.as_ref(),
        &impact_map,
        &squash_map,
        max_output_error,
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
    neurons.sort_by(|a, b| {
        // Base weighted score: error × impact^gamma
        let a_base = a.total_error * (a.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);
        let b_base = b.total_error * (b.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);

        // Issue #206: Apply gradient flow factor
        let a_gradient_factor = compute_gradient_flow_factor(&a.gradient_flow);
        let b_gradient_factor = compute_gradient_flow_factor(&b.gradient_flow);

        // Issue #204: Apply activation frequency factor
        let a_frequency_factor = compute_frequency_factor(a.activation_frequency);
        let b_frequency_factor = compute_frequency_factor(b.activation_frequency);

        let a_weighted = a_base * a_gradient_factor * a_frequency_factor;
        let b_weighted = b_base * b_gradient_factor * b_frequency_factor;

        b_weighted
            .total_cmp(&a_weighted)
            .then_with(|| b.impact.total_cmp(&a.impact))
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    // Identify removal candidates
    let cost_of_growth_threshold = cost_of_growth.unwrap_or(DEFAULT_COST_OF_GROWTH);

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

    if let Some(limit) = max_results
        && neurons.len() > limit
    {
        neurons.truncate(limit);
    }

    // Issue #306: Detect constant-value neurons and create coordinated structural candidates
    // that remove the neuron and adjust downstream biases.
    let constant_neuron_removals = detect_constant_neuron_removals(
        &selectable,
        &records_provider,
        &synapse_counts,
        creature,
        cost_of_growth_threshold,
    );

    let rejection_breakdown = build_rejection_breakdown(&removal_outcome);

    Ok(RankFocusStats {
        neurons,
        removal_candidates: removal_outcome.candidates,
        constant_neuron_removals,
        max_output_error,
        processed_neurons: total_neurons,
        total_neurons,
        duration_ms: start.elapsed().as_millis(),
        rejection_breakdown,
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
    let start = Instant::now();
    let selectable: Vec<&NeuronJson> = creature
        .neurons
        .iter()
        .filter(|neuron| is_selectable_type(&neuron.neuron_type))
        .collect();
    let total_neurons = selectable.len();

    if total_neurons == 0 {
        return Ok(RankFocusStats {
            neurons: Vec::new(),
            removal_candidates: Vec::new(),
            constant_neuron_removals: Vec::new(),
            max_output_error: 0.0,
            processed_neurons: 0,
            total_neurons: 0,
            duration_ms: start.elapsed().as_millis(),
            rejection_breakdown: std::collections::HashMap::new(),
        });
    }

    let (records_provider, is_lazy_mode) = load_records_provider(parquet_file)?;

    if is_lazy_mode && verbose_enabled() {
        tracing::debug!(
            cached_neurons = records_provider.len(),
            "Lazy record cache initialised"
        );
    }

    // Verify that all selectable neurons have records
    for neuron in &selectable {
        get_records_or_error(records_provider.as_ref(), &neuron.uuid)
            .context("Failed to read discovery records for all selectable neurons")?;
    }

    let max_output_error = compute_max_output_error(creature, records_provider.as_ref())?;

    // Use activation-based impact calculation for more accurate MIN/MAX/IF statistics
    let impact_map = compute_impacts_with_activations(creature, records_provider.as_ref())?;

    // Pre-compute synapse counts for O(1) lookup
    let synapse_counts = SynapseCounts::new(creature);

    // Issue #206: Build squash map for gradient flow analysis
    let squash_map = build_squash_map(creature);

    // Build neurons with base metrics
    let mut neurons = build_ranked_neurons(
        &selectable,
        records_provider.as_ref(),
        &impact_map,
        &squash_map,
        max_output_error,
    )?;

    // Sort by weighted score with optional history factor, gradient flow factor, and frequency factor
    // Issue #227: Incorporate historical success rate into ranking
    // Issue #206: Incorporate gradient flow analysis into ranking
    neurons.sort_by(|a, b| {
        let a_base = a.total_error * (a.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);
        let b_base = b.total_error * (b.impact + IMPACT_EPSILON).powf(IMPACT_GAMMA);

        // Issue #206: Apply gradient flow factor
        let a_gradient_factor = compute_gradient_flow_factor(&a.gradient_flow);
        let b_gradient_factor = compute_gradient_flow_factor(&b.gradient_flow);

        // Issue #204: Apply activation frequency factor
        let a_frequency_factor = compute_frequency_factor(a.activation_frequency);
        let b_frequency_factor = compute_frequency_factor(b.activation_frequency);

        let a_with_gradient = a_base * a_gradient_factor * a_frequency_factor;
        let b_with_gradient = b_base * b_gradient_factor * b_frequency_factor;

        // Apply history factor if available
        // History factor is in [0, 1], where 0.5 is neutral
        // We scale it so that:
        // - 0.5 (neutral) → multiplier of 1.0 (no change)
        // - 1.0 (perfect success) → multiplier of 1.5 (50% boost)
        // - 0.0 (complete failure) → multiplier of 0.5 (50% penalty)
        // Formula: multiplier = 0.5 + history_score
        let (a_weighted, b_weighted) = if let Some(h) = history {
            let a_history = h.bayesian_score_for(&a.neuron_uuid) as f32;
            let b_history = h.bayesian_score_for(&b.neuron_uuid) as f32;
            let a_multiplier = 0.5 + a_history;
            let b_multiplier = 0.5 + b_history;
            (
                a_with_gradient * a_multiplier,
                b_with_gradient * b_multiplier,
            )
        } else {
            (a_with_gradient, b_with_gradient)
        };

        b_weighted
            .total_cmp(&a_weighted)
            .then_with(|| b.impact.total_cmp(&a.impact))
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    // Identify removal candidates (same logic as rank_focus_neurons)
    let cost_of_growth_threshold = cost_of_growth.unwrap_or(DEFAULT_COST_OF_GROWTH);

    // Issue #414: High-error exploratory ablation DISABLED (see rank_focus_neurons for rationale)

    let removal_outcome =
        identify_removal_candidates(&neurons, &synapse_counts, cost_of_growth_threshold);

    if let Some(limit) = max_results
        && neurons.len() > limit
    {
        neurons.truncate(limit);
    }

    // Constant neuron removals (same as rank_focus_neurons)
    let constant_neuron_removals = detect_constant_neuron_removals(
        &selectable,
        &records_provider,
        &synapse_counts,
        creature,
        cost_of_growth_threshold,
    );

    let rejection_breakdown = build_rejection_breakdown(&removal_outcome);
    let removal_candidates = removal_outcome.candidates;

    Ok(RankFocusStats {
        neurons,
        removal_candidates,
        constant_neuron_removals,
        max_output_error,
        processed_neurons: total_neurons,
        total_neurons,
        duration_ms: start.elapsed().as_millis(),
        rejection_breakdown,
    })
}

// NOTE: Tests for focus module have been moved to tests/focus.rs
// following the testing philosophy documented in README.md
// (prefer tests/ over inline unit tests for tests that use public APIs).
