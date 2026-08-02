//! Resolve the activation-weighted removal gate (Issue #1923).
//!
//! # Why this module exists
//!
//! [`RemovalCandidateJson`](crate::RemovalCandidateJson) documents
//! `activation_weighted_impact = impact × mean_activation` as "the actual
//! contribution the neuron makes during inference", and the record-derived
//! ranking path ranks on exactly that. The structure-only path
//! ([`identify_structural_removal_candidates`](super::identify_structural_removal_candidates))
//! could not: it constructed every candidate with `mean_activation: 0.0` and a
//! reason string admitting the gate was deferred. The deferral never resolved —
//! nothing downstream ever revisited it — so `remove-low-impact`, which
//! produces 53% of every cached candidate and 79% of all realised gain, ranked
//! on a dead field. The #1920 cache study measured the cost:
//! `removalCandidate.impact` correlates with realised gain at **r = −0.036**,
//! and `meanActivation` was `0` in all 67 cached records — zero variance, not
//! even correlatable.
//!
//! # What it does
//!
//! After the structural triage has chosen a (small) candidate set, one
//! projected streaming pass over the discovery parquet
//! ([`read_mean_abs_activation_by_neuron`]) measures each candidate's mean
//! absolute activation. Candidates are then re-gated and re-ranked on
//! `activation_weighted_impact`, exactly as the record-derived path does — the
//! two paths now apply the same criterion to the same field.
//!
//! # Why this does not reintroduce the #1766 focus stall
//!
//! Issue #1766 removed parquet I/O from focus **selection**, because reading
//! multi-GB record sets before a single focus neuron had been picked burned the
//! whole analysis budget. Three properties keep this pass off that path:
//!
//! 1. It runs **after** selection — the focus set is already fixed, and a
//!    failure here cannot delay or alter it.
//! 2. It projects **two** columns and materialises **no** records, so it never
//!    decodes the `errors` `ListArray` that dominates the file.
//! 3. It is bounded by the caller's discovery deadline.
//!
//! # Failing loud
//!
//! An unreadable or absent parquet file leaves the candidates unweighted and
//! their reason strings carrying
//! [`ACTIVATION_GATE_PENDING`](super::removal_candidates::ACTIVATION_GATE_PENDING)
//! plus the error, and logs a WARN. Unmeasured candidates also rank **below**
//! every measured one, so a silent I/O failure can never promote an unranked
//! candidate to the top of the list.

use std::collections::HashSet;
use std::time::SystemTime;

use crate::analysis::constants::{
    REMOVAL_MEAN_ACTIVATION_THRESHOLD, remove_low_impact_noise_floor,
};
use crate::parquet_format::{ActivationSummary, read_mean_abs_activation_by_neuron};

use crate::CreatureJson;

use super::removal_candidates::{
    ACTIVATION_GATE_PENDING, RemovalCandidateOutcome, effective_cost_of_growth,
    identify_structural_removal_candidates_at,
};

/// The shipped focus-path removal triage (Issue #1923).
///
/// Structural triage picks the candidate set from topology alone (Issue #1767,
/// unchanged), then the activation-weighted gate resolves against recorded
/// activations so the candidates carry — and are ranked by — a live signal
/// instead of a hard-coded `0.0`.
///
/// `parquet_file` is the caller's discovery record file; `deadline` bounds the
/// measurement pass. Neither can affect the focus set, which is already fixed
/// by the time this runs.
pub(crate) fn identify_removal_candidates_for_focus(
    creature: &CreatureJson,
    cost_of_growth: Option<f32>,
    parquet_file: &str,
    deadline: Option<SystemTime>,
) -> RemovalCandidateOutcome {
    let growth_cost = effective_cost_of_growth(cost_of_growth);
    let structural = identify_structural_removal_candidates_at(creature, growth_cost);
    resolve_activation_weighted_gate(structural, parquet_file, growth_cost, deadline)
}

/// One candidate's verdict under the resolved gate.
enum GateVerdict {
    Kept,
    ActiveNeuronReject,
    SavingsBelowImpactReject,
    NoiseFloorReject,
}

/// Resolve the activation-weighted gate over an already-triaged outcome.
///
/// `growth_cost` must already be validated by
/// [`effective_cost_of_growth`](super::removal_candidates::effective_cost_of_growth) —
/// it denominates the noise floor, exactly as in the structural triage.
///
/// The returned outcome preserves the conservation invariant: a candidate the
/// gate rejects moves into the matching rejection counter rather than
/// disappearing.
pub(super) fn resolve_activation_weighted_gate(
    outcome: RemovalCandidateOutcome,
    parquet_file: &str,
    growth_cost: f32,
    deadline: Option<SystemTime>,
) -> RemovalCandidateOutcome {
    if outcome.candidates.is_empty() {
        return outcome;
    }

    let wanted: HashSet<&str> = outcome
        .candidates
        .iter()
        .map(|c| c.neuron_uuid.as_str())
        .collect();

    let summaries = match read_mean_abs_activation_by_neuron(parquet_file, &wanted, deadline) {
        Ok(summaries) => summaries,
        Err(read_err) => {
            // Fail loud, not silent: the gate did not run, and every candidate
            // says so on the wire (Issue #1923).
            tracing::warn!(
                target: "neat_ai_discovery::focus::activation_weighting",
                parquet_file,
                error = %read_err,
                candidates = outcome.candidates.len(),
                "activation-weighted removal gate unresolved: discovery records could not be read; \
                 removal candidates keep unmeasured meanActivation and rank last",
            );
            let mut outcome = outcome;
            for candidate in &mut outcome.candidates {
                candidate.reason = candidate.reason.replace(
                    ACTIVATION_GATE_PENDING,
                    &format!("{ACTIVATION_GATE_PENDING} — unresolved: {read_err}"),
                );
            }
            return outcome;
        }
    };

    let noise_floor = remove_low_impact_noise_floor(growth_cost);

    let RemovalCandidateOutcome {
        candidates,
        mut noise_floor_rejections,
        mut savings_below_impact_rejections,
        mut active_neuron_rejections,
        considered,
    } = outcome;

    let mut kept = Vec::with_capacity(candidates.len());
    let mut unmeasured: u32 = 0;

    for mut candidate in candidates {
        let Some(summary) = summaries.get(candidate.neuron_uuid.as_str()) else {
            // No usable samples: leave the fields unmeasured and say so.
            unmeasured = unmeasured.saturating_add(1);
            candidate.reason = candidate.reason.replace(
                ACTIVATION_GATE_PENDING,
                &format!("{ACTIVATION_GATE_PENDING} — no recorded activation samples"),
            );
            kept.push(candidate);
            continue;
        };

        match apply_gate(&mut candidate, *summary, noise_floor) {
            GateVerdict::Kept => kept.push(candidate),
            GateVerdict::ActiveNeuronReject => {
                active_neuron_rejections = active_neuron_rejections.saturating_add(1);
            }
            GateVerdict::SavingsBelowImpactReject => {
                savings_below_impact_rejections = savings_below_impact_rejections.saturating_add(1);
            }
            GateVerdict::NoiseFloorReject => {
                noise_floor_rejections = noise_floor_rejections.saturating_add(1);
            }
        }
    }

    // Rank on the live signal: highest net improvement
    // (`removal_savings − activation_weighted_impact`) first, matching the
    // record-derived path. Measured candidates always outrank unmeasured ones,
    // whose `activation_weighted_impact` of `0.0` would otherwise flatter them
    // to the top of the list.
    kept.sort_by(|a, b| {
        let a_measured = summaries.contains_key(a.neuron_uuid.as_str());
        let b_measured = summaries.contains_key(b.neuron_uuid.as_str());
        b_measured
            .cmp(&a_measured)
            .then_with(|| {
                let a_net = a.removal_savings - a.activation_weighted_impact;
                let b_net = b.removal_savings - b.activation_weighted_impact;
                b_net.total_cmp(&a_net)
            })
            .then_with(|| {
                a.activation_weighted_impact
                    .total_cmp(&b.activation_weighted_impact)
            })
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    if unmeasured > 0 {
        tracing::debug!(
            target: "neat_ai_discovery::focus::activation_weighting",
            parquet_file,
            unmeasured,
            measured = kept.len().saturating_sub(unmeasured as usize),
            "activation-weighted removal gate: some candidates had no recorded activations",
        );
    }

    let resolved = RemovalCandidateOutcome {
        candidates: kept,
        noise_floor_rejections,
        savings_below_impact_rejections,
        active_neuron_rejections,
        considered,
    };
    resolved.assert_conserved();
    resolved
}

/// Fold one candidate's measured activation into its ranking fields and apply
/// the same three gates the record-derived path applies.
fn apply_gate(
    candidate: &mut super::RemovalCandidate,
    summary: ActivationSummary,
    noise_floor: f32,
) -> GateVerdict {
    let mean_activation = summary.mean_abs_activation;
    let activation_weighted_impact = candidate.impact * mean_activation;

    // Issue #1804/#1872: a contribution that cannot be reasoned about is
    // treated as infinitely costly to remove, never as the best candidate.
    if !mean_activation.is_finite() || !activation_weighted_impact.is_finite() {
        return GateVerdict::SavingsBelowImpactReject;
    }

    // Issue #892: a neuron that is still firing is contributing, however small
    // its structural impact. Disconnected neurons (impact ≈ 0) stay removable.
    let has_meaningful_impact = candidate.impact > f32::EPSILON;
    if has_meaningful_impact && mean_activation > REMOVAL_MEAN_ACTIVATION_THRESHOLD {
        return GateVerdict::ActiveNeuronReject;
    }

    // The structural triage compared savings against the *unweighted* impact;
    // re-run that comparison against the weighted one now it exists.
    if candidate.removal_savings <= activation_weighted_impact {
        return GateVerdict::SavingsBelowImpactReject;
    }

    let net_improvement = candidate.removal_savings - activation_weighted_impact;
    if net_improvement < noise_floor {
        return GateVerdict::NoiseFloorReject;
    }

    candidate.mean_activation = mean_activation;
    candidate.activation_weighted_impact = activation_weighted_impact;
    // Issue #117: the expected reduction is the activation-weighted contribution
    // the removal gives up, never the neuron's error.
    candidate.expected_error_reduction = activation_weighted_impact;
    candidate.reason = candidate.reason.replace(
        ACTIVATION_GATE_PENDING,
        &format!(
            "activation-weighted gate resolved (Issue #1923): meanActivation={mean_activation:.3e} over {} samples, activationWeightedImpact={activation_weighted_impact:.2e} (net +{net_improvement:.2e})",
            summary.sample_count,
        ),
    );

    GateVerdict::Kept
}
