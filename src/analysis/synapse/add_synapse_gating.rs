//! Add-synapse candidate gating logic (Issue #1057).
//!
//! Determines whether add-synapse candidate generation should be skipped for a
//! creature based on:
//!
//! 1. **Historical success rate**: If the `ModuleOutcomeTracker` shows that
//!    add-synapse candidates have a success rate below a configurable threshold
//!    (default 1%), generation is skipped to save compute.
//!
//! 2. **Synapse density**: If the creature's synapse-to-neuron ratio exceeds a
//!    configurable threshold (default 14), adding one synapse has minimal
//!    structural impact and generation is skipped.
//!
//! GRQ-sampler discovery cache shows add-synapse candidates have a near-zero
//! success rate (~0.1-0.3%) across most creatures — only 3 of 43 creatures have
//! any successes. These gates avoid wasting compute on candidates that will
//! almost certainly fail.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for neural network computation (Issue #873)

use crate::analysis::constants::{
    ADD_SYNAPSE_MAX_DENSITY_RATIO, ADD_SYNAPSE_MIN_SUCCESS_RATE, ADD_SYNAPSE_MODULE_NAME,
    MIN_BOOST_SAMPLES,
};
use crate::analysis::module_weights::ModuleOutcomeTracker;

/// Result of the add-synapse gating check.
#[derive(Debug, Clone)]
pub struct AddSynapseGatingResult {
    /// Whether add-synapse candidate generation should be skipped.
    pub skip: bool,
    /// Human-readable reason for the decision (for logging/diagnostics).
    pub reason: String,
}

/// Compute the synapse-to-neuron density ratio for a creature.
///
/// Returns 0.0 when there are zero neurons (avoids division by zero).
pub fn compute_synapse_density_ratio(synapse_count: usize, neuron_count: usize) -> f64 {
    if neuron_count == 0 {
        return 0.0;
    }
    synapse_count as f64 / neuron_count as f64
}

/// Determine whether add-synapse candidate generation should be skipped.
///
/// Checks two independent gates (either triggers a skip):
///
/// 1. **Success rate gate**: When the tracker has sufficient history
///    (>= `MIN_BOOST_SAMPLES`) and the Bayesian success rate is below
///    `min_success_rate` (default `ADD_SYNAPSE_MIN_SUCCESS_RATE`).
///
/// 2. **Density gate**: When the synapse-to-neuron ratio exceeds
///    `max_density_ratio` (default `ADD_SYNAPSE_MAX_DENSITY_RATIO`).
///
/// # Arguments
///
/// * `tracker` — Optional historical outcome tracker. `None` or no data for
///   the add-synapse module means the gate is not triggered.
/// * `neuron_count` — Total number of neurons in the creature.
/// * `synapse_count` — Total number of synapses in the creature.
/// * `min_success_rate` — Override for the minimum success rate threshold.
///   Pass `None` to use the default (`ADD_SYNAPSE_MIN_SUCCESS_RATE`).
/// * `max_density_ratio` — Override for the maximum density ratio threshold.
///   Pass `None` to use the default (`ADD_SYNAPSE_MAX_DENSITY_RATIO`).
pub fn should_skip_add_synapse_candidates(
    tracker: &Option<ModuleOutcomeTracker>,
    neuron_count: usize,
    synapse_count: usize,
    min_success_rate: Option<f64>,
    max_density_ratio: Option<f64>,
) -> AddSynapseGatingResult {
    let min_rate = min_success_rate.unwrap_or(ADD_SYNAPSE_MIN_SUCCESS_RATE);
    let max_density = max_density_ratio.unwrap_or(ADD_SYNAPSE_MAX_DENSITY_RATIO);

    // Gate 1: Check historical success rate from the ModuleOutcomeTracker.
    if let Some(t) = tracker {
        let stats = t.stats(ADD_SYNAPSE_MODULE_NAME);
        if (stats.attempts as usize) >= MIN_BOOST_SAMPLES {
            let rate = stats.success_rate();
            if rate < min_rate {
                return AddSynapseGatingResult {
                    skip: true,
                    reason: format!(
                        "add-synapse success rate {:.3}% is below threshold {:.1}% \
                         ({} attempts, {} successes) — skipping generation (Issue #1057)",
                        rate * 100.0,
                        min_rate * 100.0,
                        stats.attempts,
                        stats.successes,
                    ),
                };
            }
        }
    }

    // Gate 2: Check synapse density ratio.
    let density = compute_synapse_density_ratio(synapse_count, neuron_count);
    if density > max_density {
        return AddSynapseGatingResult {
            skip: true,
            reason: format!(
                "synapse density ratio {density:.1} ({synapse_count} synapses / \
                 {neuron_count} neurons) exceeds threshold {max_density:.1} — \
                 skipping add-synapse generation (Issue #1057)",
            ),
        };
    }

    AddSynapseGatingResult {
        skip: false,
        reason: String::new(),
    }
}
