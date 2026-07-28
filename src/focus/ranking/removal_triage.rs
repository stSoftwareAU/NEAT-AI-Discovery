//! Structure-only removal triage (Issue #1767).
//!
//! ## Opposite axes
//!
//! Focus and removal answer near-**opposite** questions and must therefore be
//! driven by opposite criteria over the *same* structural impact map:
//!
//! | Concern | Criteria | Records needed |
//! |---------|----------|----------------|
//! | Focus (#1766) | **High** structural impact, weighted-random (outputs seed at 1.0) | Never for choosing the set |
//! | Removal (#1767) | **Low** contribution vs the complexity savings from pruning | Structure only for triage |
//!
//! Removal is *not* "negate the focus score" — high error never means remove
//! (see the Issue #414 disablement). It is **low contribution vs savings**.
//!
//! ## Why this path exists
//!
//! [`super::removal_candidates::identify_removal_candidates`] leans on the same
//! axis, but it consumes [`super::RankedNeuron`]s whose
//! `activation_weighted_impact` is derived from recorded discovery data. That
//! coupled removal triage to a parquet decode at focus-selection time — a
//! production incident burned ~2 h warming a 12.5 GB discovery file before any
//! useful discovery work began.
//!
//! [`triage_removal_candidates`] needs only the creature topology: it derives
//! structural impact from path weights ([`compute_impacts_public`]) and
//! complexity savings from synapse counts. No parquet file is opened, decoded,
//! or even named. Activation-weighted gates that still need records belong in
//! the **analysis** phase, after the focus set is fixed.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use super::DEFAULT_COST_OF_GROWTH;
use super::removal_candidates::{SynapseCounts, calculate_removal_savings};
use crate::CreatureJson;
use crate::analysis::constants::{REMOVAL_CANDIDATE_BOOST, remove_low_impact_noise_floor};
use crate::focus::impact::compute_impacts_public;

/// A hidden neuron whose complexity savings outweigh its **structural**
/// contribution — safe to prune on topology evidence alone (Issue #1767).
///
/// Deliberately carries no record-derived fields (no `mean_activation`, no
/// `activation_weighted_impact`, no `total_error`): every value here is
/// computable from the creature JSON, which is what keeps focus-time triage
/// free of parquet I/O. The activation-weighted view lives on
/// [`super::RemovalCandidate`] and is produced later, in the analysis phase.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralRemovalCandidate {
    pub neuron_uuid: String,
    /// Structural impact magnitude — the weighted path product to the outputs.
    /// Low impact is what makes the neuron a removal candidate.
    pub impact: f32,
    /// Number of synapses pointing TO this neuron.
    pub incoming_synapses: usize,
    /// Number of synapses pointing FROM this neuron.
    pub outgoing_synapses: usize,
    /// Complexity savings from pruning the neuron and its synapses, after the
    /// [`REMOVAL_CANDIDATE_BOOST`] multiplier.
    pub removal_savings: f32,
    /// `removal_savings - impact`: how much better the creature is expected to
    /// score once the neuron is gone.
    pub net_improvement: f32,
    pub reason: String,
}

/// Outcome of [`triage_removal_candidates`] — surviving candidates plus the
/// noise-floor rejection count, surfaced rather than silently dropped.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StructuralRemovalTriage {
    /// Candidates that cleared both the savings-vs-impact test and the
    /// noise floor, best net improvement first.
    pub candidates: Vec<StructuralRemovalCandidate>,
    /// Candidates dropped because `net_improvement` fell below
    /// [`remove_low_impact_noise_floor`] (Issue #1142).
    pub noise_floor_rejections: u32,
}

/// Resolve the effective cost-of-growth, rejecting values that would produce
/// nonsense savings. A non-finite or non-positive input is a caller bug, so it
/// is logged at WARN (never silently accepted) and the default substituted.
fn effective_cost_of_growth(cost_of_growth: Option<f32>) -> f32 {
    match cost_of_growth {
        Some(value) if value.is_finite() && value > 0.0 => value,
        Some(invalid) => {
            tracing::warn!(
                target: "neat_ai_discovery::focus::removal_triage",
                invalid_cost_of_growth = invalid,
                default_cost_of_growth = DEFAULT_COST_OF_GROWTH,
                "removal triage received a non-finite or non-positive costOfGrowth; using the default",
            );
            DEFAULT_COST_OF_GROWTH
        }
        None => DEFAULT_COST_OF_GROWTH,
    }
}

/// Triage removal candidates from the creature **topology alone** (Issue #1767).
///
/// A hidden neuron is a removal candidate when the complexity savings from
/// pruning it exceed its structural contribution:
///
/// ```text
/// savings          = costOfGrowth × (1 + (incoming + outgoing) / 10)
/// boostedSavings   = savings × REMOVAL_CANDIDATE_BOOST
/// candidate  ⟺  boostedSavings > |impact|  and  boostedSavings − |impact| ≥ noiseFloor
/// ```
///
/// This is the near-**opposite** of the focus criterion (high structural
/// impact, weighted-random — Issue #1766) evaluated over the same impact map,
/// so the two never compete for the same neurons.
///
/// Only `hidden` neurons are triaged. Outputs seed at impact `1.0` and are the
/// add-neuron targets, inputs and constants are not prunable — mirroring the
/// hidden-only gate in the constant-neuron removal path (Issue #306).
///
/// # Performance
///
/// Topology-only: no parquet file is opened or decoded, so the seconds bar
/// from Issue #1766 holds whether or not multi-GB discovery data exists.
///
/// # Arguments
/// * `creature` — the creature topology to triage.
/// * `cost_of_growth` — per-hidden-neuron growth cost; `None`, non-finite, or
///   non-positive falls back to the crate default (logged at WARN).
#[must_use]
pub fn triage_removal_candidates(
    creature: &CreatureJson,
    cost_of_growth: Option<f32>,
) -> StructuralRemovalTriage {
    let growth = effective_cost_of_growth(cost_of_growth);
    let noise_floor = remove_low_impact_noise_floor();
    let impacts = compute_impacts_public(creature);
    let synapse_counts = SynapseCounts::new(creature);

    let mut candidates: Vec<StructuralRemovalCandidate> = Vec::new();
    let mut noise_floor_rejections: u32 = 0;

    for neuron in creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
    {
        let (incoming, outgoing) = synapse_counts.get(&neuron.uuid);
        let savings = calculate_removal_savings(incoming, outgoing, growth);
        let boosted_savings = savings * REMOVAL_CANDIDATE_BOOST;

        // A non-finite impact cannot be reasoned about; treat it as maximally
        // contributing so the neuron is never pruned on bad numbers.
        let raw_impact = impacts.get(&neuron.uuid).copied().unwrap_or(0.0);
        let impact = if raw_impact.is_finite() {
            raw_impact.abs()
        } else {
            f32::INFINITY
        };

        if boosted_savings <= impact {
            continue;
        }

        let net_improvement = boosted_savings - impact;
        if net_improvement < noise_floor {
            noise_floor_rejections = noise_floor_rejections.saturating_add(1);
            continue;
        }

        candidates.push(StructuralRemovalCandidate {
            neuron_uuid: neuron.uuid.clone(),
            impact,
            incoming_synapses: incoming,
            outgoing_synapses: outgoing,
            removal_savings: boosted_savings,
            net_improvement,
            reason: format!(
                "Structural removal triage: saves {savings:.2e} (boosted {REMOVAL_CANDIDATE_BOOST:.1}×) \
                 > structural impact {impact:.2e} (net +{net_improvement:.2e}), {} synapses, costOfGrowth={growth:.2e}",
                incoming + outgoing,
            ),
        });
    }

    // Best net improvement first; ties broken by lower contribution, then uuid
    // so the order is deterministic across runs.
    candidates.sort_by(|a, b| {
        b.net_improvement
            .total_cmp(&a.net_improvement)
            .then_with(|| a.impact.total_cmp(&b.impact))
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    if noise_floor_rejections > 0 {
        tracing::debug!(
            target: "neat_ai_discovery::focus::removal_triage",
            noise_floor_rejections,
            noise_floor,
            surviving = candidates.len(),
            "structural removal triage dropped candidates below the noise floor",
        );
    }

    StructuralRemovalTriage {
        candidates,
        noise_floor_rejections,
    }
}
